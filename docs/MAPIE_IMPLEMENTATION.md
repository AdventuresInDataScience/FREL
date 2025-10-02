# MAPIE Wrapper Implementation Summary

## What Was Created

### 1. Core Module: `src/mapie.py`
A comprehensive wrapper around MAPIE (Model Agnostic Prediction Interval Estimator) that provides:

#### Key Classes:

**`SklearnTensorFlowWrapper`**
- Makes TensorFlow/Keras models compatible with sklearn and MAPIE
- Handles the dual-input structure (price + meta features) transparently
- Automatically flattens/unflattens inputs for sklearn compatibility
- Supports configurable training epochs and batch size

**`SklearnLightGBMWrapper`**
- Wraps LightGBM models for consistent interface with TensorFlow wrapper
- Ensures MAPIE compatibility
- Handles the same input format as TensorFlow wrapper

**`MapiePredictor`**
- Main interface for confidence interval predictions
- Returns clean DataFrame output (not nested arrays)
- Supports multiple confidence levels in a single call
- Methods:
  - `fit()`: Train the conformal predictor
  - `predict_intervals()`: Get predictions at multiple confidence levels → **DataFrame**
  - `predict_single_interval()`: Get one confidence interval
  - `predict_point()`: Get point predictions only

**`create_mapie_predictor_from_model()`**
- Convenience factory function
- Automatically detects model type and wraps appropriately
- One-line setup for MAPIE predictions

### 2. Documentation: `MAPIE_GUIDE.md`
Complete usage guide with:
- Installation instructions
- Quick start examples
- TensorFlow and LightGBM examples
- Output format explanation
- Integration with existing Predictor class
- Advanced usage patterns

### 3. Examples: `dev/example_mapie.py`
Comprehensive examples demonstrating:
- TensorFlow transformer model with MAPIE
- LightGBM model with MAPIE
- Single confidence interval usage
- Custom confidence levels
- Coverage statistics calculation

### 4. Tests: `dev/test_mapie_wrapper.py` & `dev/validate_mapie.py`
Validation scripts to verify functionality

### 5. Dependencies Updated: `pyproject.toml`
Added required packages:
- `mapie>=0.9.0`
- `scikit-learn>=1.3.0`

## Key Features

### 1. Clean DataFrame Output
Instead of MAPIE's nested array structure:
```python
# Old MAPIE output: y_pis shape (N, 2, 1)
y_pred, y_pis = mapie.predict(X, alpha=0.05)
lower = y_pis[:, 0, 0]  # Confusing!
upper = y_pis[:, 1, 0]
```

New clean output:
```python
# New wrapper output: DataFrame
df = mapie_pred.predict_intervals(X_price, X_meta, alphas=[0.05, 0.10])
# Columns: point_pred, lower_95, upper_95, width_95, lower_90, upper_90, width_90
```

### 2. Handles TensorFlow's Dual-Input Structure
Your models expect `[price_input, meta_input]`, but sklearn expects a single array. The wrapper handles this automatically:

```python
wrapped = SklearnTensorFlowWrapper(tf_model)
# Internally converts between:
# - Combined: (N, lookback*5 + 8) ← sklearn interface
# - Split: [price(N,200,5), meta(N,8)] ← TensorFlow interface
```

### 3. Multiple Confidence Levels in One Call
```python
df = mapie_pred.predict_intervals(
    X_price, X_meta,
    alphas=[0.05, 0.10, 0.20]  # 95%, 90%, 80% confidence
)
# All confidence intervals in one DataFrame!
```

## Usage Example

```python
from src.mapie import create_mapie_predictor_from_model
from src.model import build_tx_model

# 1. Build your model
tf_model = build_tx_model(price_shape=(200, 5), meta_len=8)
tf_model.compile(optimizer='adam', loss='mse')

# 2. Create MAPIE predictor (one line!)
mapie_pred = create_mapie_predictor_from_model(
    tf_model,
    model_type='transformer',
    method='plus',
    cv=5
)

# 3. Fit
mapie_pred.fit(X_price_train, X_meta_train, y_train)

# 4. Get predictions with confidence intervals
predictions = mapie_pred.predict_intervals(
    X_price_test,
    X_meta_test,
    alphas=[0.05, 0.10, 0.20]
)

# 5. Use the clean DataFrame
print(predictions.head())
#    point_pred  lower_95  upper_95  width_95  lower_90  ...
# 0      523.45    480.12    566.78     86.66    490.23  ...
# 1      612.33    570.45    654.21     83.76    580.12  ...
```

## Integration with Your Existing Code

The wrapper is designed to work seamlessly with your existing `Predictor` class:

```python
from src.predictor import Predictor
from src.mapie import create_mapie_predictor_from_model

# Load your existing trained model
predictor = Predictor.from_checkpoint(
    model_path="checkpoints/model.keras",
    scaler_path="checkpoints/scaler.json",
    cfg=config,
    model_type="transformer"
)

# Wrap it with MAPIE for confidence intervals
mapie_pred = create_mapie_predictor_from_model(
    predictor.model,
    model_type=predictor.model_type
)

# Now you get confidence intervals!
```

## Why This is Useful

1. **Uncertainty Quantification**: Know how confident your model is
2. **Risk Management**: Make decisions based on confidence intervals
3. **Conformal Prediction**: Theoretically calibrated (95% CI contains 95% of actuals)
4. **Model Agnostic**: Works with any model (TensorFlow, LightGBM, etc.)
5. **Clean API**: DataFrame output is much easier to work with than nested arrays

## MAPIE Methods Supported

- `'plus'` (recommended): Adaptive intervals, most accurate
- `'base'`: Standard split conformal
- `'minmax'`: Min-max calibration
- `'naive'`: Simple residual-based

## Files Created

```
src/mapie.py                    # Main wrapper module (559 lines)
MAPIE_GUIDE.md                  # Complete usage guide
dev/example_mapie.py            # Full examples with all features
dev/test_mapie_wrapper.py       # Functional tests
dev/validate_mapie.py           # Quick validation
```

## Next Steps

1. **Install dependencies** (already done):
   ```bash
   uv pip install mapie scikit-learn
   ```

2. **Try the examples**:
   ```bash
   uv run python dev/example_mapie.py
   ```

3. **Read the guide**: See `MAPIE_GUIDE.md` for detailed usage

4. **Integrate into your workflow**: Use in your training/prediction pipeline

## Notes

- MAPIE uses cross-validation, so fitting takes ~CV times longer
- The intervals are theoretically calibrated (conformal prediction)
- Works with both neural networks and tree-based models
- The wrapper handles all the sklearn compatibility issues for you

Enjoy your new confidence intervals! 🎉
