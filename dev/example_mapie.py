"""
Example usage of the MAPIE wrapper for confidence interval predictions.

This script demonstrates:
1. Wrapping TensorFlow and LightGBM models for MAPIE
2. Fitting the MapiePredictor
3. Getting predictions at multiple confidence levels
4. Working with the clean DataFrame output
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import build_tx_model, build_lgb_model
from mapie import (
    SklearnTensorFlowWrapper,
    SklearnLightGBMWrapper,
    MapiePredictor,
    create_mapie_predictor_from_model
)


def generate_dummy_data(n_samples=1000, lookback=200, price_features=5, meta_len=8):
    """Generate dummy data for demonstration."""
    np.random.seed(42)
    
    # Generate synthetic price data
    X_price = np.random.randn(n_samples, lookback, price_features).astype(np.float32)
    
    # Generate synthetic meta features
    X_meta = np.random.randn(n_samples, meta_len).astype(np.float32)
    
    # Generate synthetic targets (rewards)
    y = np.random.randn(n_samples).astype(np.float32) * 100 + 500
    
    return X_price, X_meta, y


def example_tensorflow_model():
    """Example: Using MAPIE with TensorFlow transformer model."""
    print("=" * 70)
    print("Example 1: TensorFlow Transformer Model with MAPIE")
    print("=" * 70)
    
    # Generate data
    X_price_train, X_meta_train, y_train = generate_dummy_data(n_samples=800)
    X_price_test, X_meta_test, y_test = generate_dummy_data(n_samples=200)
    
    print(f"Training data: {X_price_train.shape}, {X_meta_train.shape}, {y_train.shape}")
    print(f"Test data: {X_price_test.shape}, {X_meta_test.shape}, {y_test.shape}")
    
    # Build TensorFlow model
    print("\nBuilding transformer model...")
    tf_model = build_tx_model(
        price_shape=(200, 5),
        meta_len=8,
        d_model=64,
        nhead=4,
        tx_blocks=2
    )
    tf_model.compile(optimizer='adam', loss='mse')
    
    # Wrap model for sklearn/MAPIE compatibility
    print("\nWrapping model for MAPIE...")
    wrapped_model = SklearnTensorFlowWrapper(
        tf_model,
        model_type='transformer',
        epochs=5,  # Few epochs for demo
        batch_size=32,
        verbose=0
    )
    
    # Create MapiePredictor
    print("Creating MapiePredictor...")
    mapie_pred = MapiePredictor(
        wrapped_model,
        method='plus',  # Conformal prediction method
        cv=3,           # 3-fold CV for speed
        n_jobs=-1
    )
    
    # Fit the predictor
    print("\nFitting MapiePredictor (this may take a moment)...")
    mapie_pred.fit(X_price_train, X_meta_train, y_train)
    
    # Get predictions at multiple confidence levels
    print("\nGenerating predictions at multiple confidence levels...")
    predictions_df = mapie_pred.predict_intervals(
        X_price_test,
        X_meta_test,
        alphas=[0.05, 0.10, 0.20],  # 95%, 90%, 80% confidence
        include_point_pred=True
    )
    
    print("\nPredictions DataFrame (first 10 rows):")
    print(predictions_df.head(10))
    print(f"\nDataFrame shape: {predictions_df.shape}")
    print(f"Columns: {list(predictions_df.columns)}")
    
    # Compare with actual values
    print("\nComparison with actual values (first 5):")
    comparison = pd.DataFrame({
        'actual': y_test[:5],
        'point_pred': predictions_df['point_pred'][:5],
        'lower_95': predictions_df['lower_95'][:5],
        'upper_95': predictions_df['upper_95'][:5],
        'width_95': predictions_df['width_95'][:5]
    })
    print(comparison)
    
    # Calculate coverage (percentage of actuals within intervals)
    coverage_95 = np.mean(
        (y_test >= predictions_df['lower_95']) & 
        (y_test <= predictions_df['upper_95'])
    )
    coverage_90 = np.mean(
        (y_test >= predictions_df['lower_90']) & 
        (y_test <= predictions_df['upper_90'])
    )
    coverage_80 = np.mean(
        (y_test >= predictions_df['lower_80']) & 
        (y_test <= predictions_df['upper_80'])
    )
    
    print(f"\nCoverage statistics:")
    print(f"  95% CI coverage: {coverage_95:.2%} (target: 95%)")
    print(f"  90% CI coverage: {coverage_90:.2%} (target: 90%)")
    print(f"  80% CI coverage: {coverage_80:.2%} (target: 80%)")
    
    return mapie_pred, predictions_df


def example_lightgbm_model():
    """Example: Using MAPIE with LightGBM model."""
    print("\n" + "=" * 70)
    print("Example 2: LightGBM Model with MAPIE")
    print("=" * 70)
    
    # Generate data
    X_price_train, X_meta_train, y_train = generate_dummy_data(n_samples=800)
    X_price_test, X_meta_test, y_test = generate_dummy_data(n_samples=200)
    
    print(f"Training data: {X_price_train.shape}, {X_meta_train.shape}, {y_train.shape}")
    
    # Build LightGBM model
    print("\nBuilding LightGBM model...")
    lgb_model = build_lgb_model(linear=True)
    
    # Wrap model
    print("Wrapping model for MAPIE...")
    wrapped_model = SklearnLightGBMWrapper(
        lgb_model,
        lookback=200,
        price_features=5
    )
    
    # Create MapiePredictor using convenience function
    print("Creating MapiePredictor...")
    mapie_pred = create_mapie_predictor_from_model(
        lgb_model,
        model_type='lightgbm',
        lookback=200,
        price_features=5,
        method='plus',
        cv=3
    )
    
    # Fit
    print("\nFitting MapiePredictor...")
    mapie_pred.fit(X_price_train, X_meta_train, y_train)
    
    # Predict
    print("\nGenerating predictions...")
    predictions_df = mapie_pred.predict_intervals(
        X_price_test,
        X_meta_test,
        alphas=[0.05, 0.10]  # 95%, 90% confidence
    )
    
    print("\nPredictions DataFrame (first 10 rows):")
    print(predictions_df.head(10))
    
    return mapie_pred, predictions_df


def example_single_interval():
    """Example: Getting a single confidence interval."""
    print("\n" + "=" * 70)
    print("Example 3: Single Confidence Interval")
    print("=" * 70)
    
    # Generate data
    X_price_train, X_meta_train, y_train = generate_dummy_data(n_samples=500)
    X_price_test, X_meta_test, _ = generate_dummy_data(n_samples=100)
    
    # Quick setup
    lgb_model = build_lgb_model(linear=True)
    mapie_pred = create_mapie_predictor_from_model(
        lgb_model,
        model_type='lightgbm',
        method='plus',
        cv=3
    )
    
    print("Fitting...")
    mapie_pred.fit(X_price_train, X_meta_train, y_train)
    
    print("\nGetting single 95% confidence interval...")
    y_pred, y_pis = mapie_pred.predict_single_interval(
        X_price_test,
        X_meta_test,
        alpha=0.05
    )
    
    print(f"\nPoint predictions shape: {y_pred.shape}")
    print(f"Prediction intervals shape: {y_pis.shape}")
    print(f"\nFirst 5 predictions:")
    for i in range(5):
        print(f"  Pred: {y_pred[i]:.2f}, "
              f"95% CI: [{y_pis[i, 0, 0]:.2f}, {y_pis[i, 1, 0]:.2f}], "
              f"Width: {y_pis[i, 1, 0] - y_pis[i, 0, 0]:.2f}")
    
    return mapie_pred


def example_custom_alphas():
    """Example: Custom confidence levels."""
    print("\n" + "=" * 70)
    print("Example 4: Custom Confidence Levels")
    print("=" * 70)
    
    # Generate data
    X_price_train, X_meta_train, y_train = generate_dummy_data(n_samples=500)
    X_price_test, X_meta_test, _ = generate_dummy_data(n_samples=50)
    
    # Setup
    lgb_model = build_lgb_model(linear=True)
    mapie_pred = create_mapie_predictor_from_model(
        lgb_model,
        model_type='lightgbm',
        method='plus',
        cv=3
    )
    
    print("Fitting...")
    mapie_pred.fit(X_price_train, X_meta_train, y_train)
    
    # Custom confidence levels
    custom_alphas = [0.01, 0.05, 0.10, 0.25, 0.50]  # 99%, 95%, 90%, 75%, 50%
    
    print(f"\nGetting predictions at {len(custom_alphas)} confidence levels...")
    predictions_df = mapie_pred.predict_intervals(
        X_price_test,
        X_meta_test,
        alphas=custom_alphas
    )
    
    print("\nAvailable columns:")
    for col in predictions_df.columns:
        print(f"  - {col}")
    
    print("\nInterval widths (mean):")
    width_cols = [col for col in predictions_df.columns if col.startswith('width_')]
    for col in width_cols:
        conf_level = col.split('_')[1]
        print(f"  {conf_level}% CI: {predictions_df[col].mean():.2f} ± {predictions_df[col].std():.2f}")
    
    return predictions_df


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MAPIE Wrapper Examples")
    print("=" * 70)
    
    # Run examples
    try:
        # Example 1: TensorFlow model
        mapie_tf, preds_tf = example_tensorflow_model()
        
        # Example 2: LightGBM model
        mapie_lgb, preds_lgb = example_lightgbm_model()
        
        # Example 3: Single interval
        mapie_single = example_single_interval()
        
        # Example 4: Custom alphas
        preds_custom = example_custom_alphas()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
