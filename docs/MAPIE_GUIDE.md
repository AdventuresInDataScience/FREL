# MAPIE Integration Guide

## Overview

The `src/mapie.py` module provides a wrapper for [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) (Model Agnostic Prediction Interval Estimator) to enable confidence interval predictions with your TensorFlow and LightGBM models.

## Key Features

- **Sklearn-compatible wrappers** for TensorFlow/Keras and LightGBM models
- **Clean DataFrame output** with predictions at multiple confidence levels
- **Automatic handling** of the dual-input structure (price + meta features)
- **Conformal prediction** for reliable uncertainty quantification

## Installation

The required dependencies are already added to `pyproject.toml`. Install them with:

```bash
pip install mapie scikit-learn
# or with uv
uv pip install mapie scikit-learn
```

## Quick Start

### 1. Wrap Your Model

```python
from src.mapie import SklearnTensorFlowWrapper, MapiePredictor
from src.model import build_tx_model

# Build and compile your TensorFlow model
tf_model = build_tx_model(price_shape=(200, 5), meta_len=8)
tf_model.compile(optimizer='adam', loss='mse')

# Wrap it for sklearn/MAPIE compatibility
wrapped_model = SklearnTensorFlowWrapper(
    tf_model,
    model_type='transformer',
    epochs=50,
    batch_size=32
)
```

### 2. Create MapiePredictor

```python
# Create the MAPIE predictor
mapie_pred = MapiePredictor(
    wrapped_model,
    method='plus',  # Conformal prediction method
    cv=5,           # Cross-validation folds
    n_jobs=-1       # Use all CPU cores
)
```

### 3. Fit and Predict

```python
# Fit on training data
mapie_pred.fit(X_price_train, X_meta_train, y_train)

# Get predictions at multiple confidence levels
predictions_df = mapie_pred.predict_intervals(
    X_price_test,
    X_meta_test,
    alphas=[0.05, 0.10, 0.20]  # 95%, 90%, 80% confidence
)

# Result: Clean DataFrame
#   point_pred | lower_95 | upper_95 | width_95 | lower_90 | upper_90 | ...
#   -----------------------------------------------------------------------
#   523.45     | 480.12   | 566.78   | 86.66    | 490.23   | 556.67   | ...
```

## Output Format

The `predict_intervals()` method returns a pandas DataFrame with:

- **`point_pred`**: Point prediction (mean/expected value)
- **`lower_X`**: Lower bound of X% confidence interval
- **`upper_X`**: Upper bound of X% confidence interval
- **`width_X`**: Width of X% confidence interval

Example for `alphas=[0.05, 0.10]`:
```
   point_pred  lower_95  upper_95  width_95  lower_90  upper_90  width_90
0      523.45    480.12    566.78     86.66    490.23    556.67     66.44
1      612.33    570.45    654.21     83.76    580.12    644.54     64.42
...
```

## LightGBM Example

For LightGBM models, use the convenience function:

```python
from src.mapie import create_mapie_predictor_from_model
from src.model import build_lgb_model

# Build LightGBM model
lgb_model = build_lgb_model(linear=True)

# Create MapiePredictor (automatically wraps)
mapie_pred = create_mapie_predictor_from_model(
    lgb_model,
    model_type='lightgbm',
    lookback=200,
    price_features=5,
    method='plus',
    cv=5
)

# Fit and predict
mapie_pred.fit(X_price_train, X_meta_train, y_train)
predictions = mapie_pred.predict_intervals(X_price_test, X_meta_test)
```

## Advanced Usage

### Single Confidence Interval

```python
# Get just one confidence interval
y_pred, y_pis = mapie_pred.predict_single_interval(
    X_price_test,
    X_meta_test,
    alpha=0.05  # 95% confidence
)

# y_pred: (N,) point predictions
# y_pis: (N, 2, 1) where [:, 0, 0] is lower, [:, 1, 0] is upper
```

### Point Predictions Only

```python
# No intervals, just predictions
y_pred = mapie_pred.predict_point(X_price_test, X_meta_test)
```

### Custom Confidence Levels

```python
# Any alpha values you want
predictions = mapie_pred.predict_intervals(
    X_price_test,
    X_meta_test,
    alphas=[0.01, 0.05, 0.10, 0.25, 0.50]  # 99%, 95%, 90%, 75%, 50%
)
```

## MAPIE Methods

The `method` parameter controls the conformal prediction strategy:

- **`'plus'`** (recommended): Conformalized quantile regression, provides adaptive intervals
- **`'base'`**: Standard split conformal prediction
- **`'minmax'`**: Min-max calibration
- **`'naive'`**: Simple residual-based intervals

## Integration with Existing Predictor Class

You can integrate this with your existing `Predictor` class:

```python
from src.predictor import Predictor
from src.mapie import create_mapie_predictor_from_model

# Load your trained model
predictor = Predictor.from_checkpoint(
    model_path="path/to/model",
    scaler_path="path/to/scaler.json",
    cfg=config,
    model_type="transformer"
)

# Wrap it with MAPIE
mapie_pred = create_mapie_predictor_from_model(
    predictor.model,
    model_type=predictor.model_type,
    method='plus'
)

# Now you can get confidence intervals
# (Note: you'll need to prepare X_price, X_meta from your data)
```

## Examples

Run the comprehensive example script:

```bash
python dev/example_mapie.py
```

This demonstrates:
- TensorFlow model wrapping
- LightGBM model wrapping
- Single and multiple confidence intervals
- Custom alpha values
- Coverage statistics

## Key Differences from Raw MAPIE

1. **Input handling**: Automatically combines price and meta features
2. **Output format**: Returns clean DataFrame instead of nested arrays
3. **Model compatibility**: Works with your dual-input TensorFlow models
4. **Confidence levels**: Clear column naming (e.g., `lower_95`, `upper_95`)

## Notes

- **Training time**: MAPIE uses cross-validation, so fitting takes ~CV times longer than a single model
- **Memory**: Stores multiple model copies for CV, requires more memory
- **Coverage**: The intervals are theoretically calibrated (e.g., 95% of actuals should fall within 95% CI)
- **TensorFlow models**: The wrapper re-trains the model during MAPIE fitting with the specified epochs

## References

- MAPIE documentation: https://mapie.readthedocs.io/
- Paper: "Conformal Prediction for Reliable Machine Learning" (Angelopoulos & Bates, 2021)
