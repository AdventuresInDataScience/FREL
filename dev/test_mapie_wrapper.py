"""
Quick test to verify MAPIE wrapper functionality.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import build_tx_model, build_lgb_model
from mapie import (
    SklearnTensorFlowWrapper,
    SklearnLightGBMWrapper,
    MapiePredictor,
    create_mapie_predictor_from_model
)


def test_tensorflow_wrapper():
    """Test TensorFlow model wrapper."""
    print("Testing TensorFlow wrapper...")
    
    # Create dummy data
    X_price = np.random.randn(100, 200, 5).astype(np.float32)
    X_meta = np.random.randn(100, 8).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    
    # Build and wrap model
    tf_model = build_tx_model(price_shape=(200, 5), meta_len=8, d_model=32, tx_blocks=1)
    tf_model.compile(optimizer='adam', loss='mse')
    
    wrapped = SklearnTensorFlowWrapper(tf_model, epochs=2, verbose=0)
    
    # Test fit
    wrapped.fit(wrapped._combine_inputs(X_price, X_meta), y)
    
    # Test predict
    y_pred = wrapped.predict(wrapped._combine_inputs(X_price[:10], X_meta[:10]))
    
    assert y_pred.shape == (10,), f"Expected shape (10,), got {y_pred.shape}"
    print("✓ TensorFlow wrapper works!")
    
    return wrapped


def test_lightgbm_wrapper():
    """Test LightGBM model wrapper."""
    print("\nTesting LightGBM wrapper...")
    
    # Create dummy data
    X_price = np.random.randn(100, 200, 5).astype(np.float32)
    X_meta = np.random.randn(100, 8).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    
    # Build and wrap model
    lgb_model = build_lgb_model(linear=True)
    wrapped = SklearnLightGBMWrapper(lgb_model, lookback=200, price_features=5)
    
    # Combine inputs
    X_combined = wrapped._combine_inputs(X_price, X_meta)
    
    # Test fit
    wrapped.fit(X_combined, y)
    
    # Test predict
    y_pred = wrapped.predict(X_combined[:10])
    
    assert y_pred.shape == (10,), f"Expected shape (10,), got {y_pred.shape}"
    print("✓ LightGBM wrapper works!")
    
    return wrapped


def test_mapie_predictor():
    """Test MapiePredictor with minimal data."""
    print("\nTesting MapiePredictor...")
    
    # Create minimal data
    X_price = np.random.randn(50, 200, 5).astype(np.float32)
    X_meta = np.random.randn(50, 8).astype(np.float32)
    y = np.random.randn(50).astype(np.float32)
    
    # Build lightweight model
    lgb_model = build_lgb_model(linear=True)
    
    # Create MapiePredictor
    mapie_pred = create_mapie_predictor_from_model(
        lgb_model,
        model_type='lightgbm',
        lookback=200,
        price_features=5,
        method='naive',  # Fastest method for testing
        cv=2  # Minimal CV folds
    )
    
    # Fit
    mapie_pred.fit(X_price, X_meta, y)
    
    # Test predict_intervals
    preds = mapie_pred.predict_intervals(
        X_price[:10],
        X_meta[:10],
        alphas=[0.05, 0.10]
    )
    
    # Verify DataFrame structure
    assert preds.shape[0] == 10, "Should have 10 predictions"
    assert 'point_pred' in preds.columns, "Should have point_pred column"
    assert 'lower_95' in preds.columns, "Should have lower_95 column"
    assert 'upper_95' in preds.columns, "Should have upper_95 column"
    assert 'width_95' in preds.columns, "Should have width_95 column"
    
    print("✓ MapiePredictor works!")
    print(f"\nSample output:\n{preds.head()}")
    
    return mapie_pred, preds


if __name__ == "__main__":
    print("=" * 60)
    print("Running MAPIE wrapper tests")
    print("=" * 60)
    
    try:
        # Run tests
        tf_wrapper = test_tensorflow_wrapper()
        lgb_wrapper = test_lightgbm_wrapper()
        mapie_pred, preds = test_mapie_predictor()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
