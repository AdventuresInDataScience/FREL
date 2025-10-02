# MAPIE Quick Reference

## One-Line Setup
```python
from src.mapie import create_mapie_predictor_from_model

mapie_pred = create_mapie_predictor_from_model(your_model, model_type='transformer')
```

## Basic Usage
```python
# Fit
mapie_pred.fit(X_price_train, X_meta_train, y_train)

# Predict with confidence intervals
df = mapie_pred.predict_intervals(X_price_test, X_meta_test, alphas=[0.05, 0.10])
```

## Output Format
```
   point_pred  lower_95  upper_95  width_95  lower_90  upper_90  width_90
0      523.45    480.12    566.78     86.66    490.23    556.67     66.44
1      612.33    570.45    654.21     83.76    580.12    644.54     64.42
```

## Methods
- `fit(X_price, X_meta, y)` - Train the predictor
- `predict_intervals(X_price, X_meta, alphas=[...])` - Get DataFrame with multiple CIs
- `predict_single_interval(X_price, X_meta, alpha=0.05)` - Get one CI as tuple
- `predict_point(X_price, X_meta)` - Get point predictions only

## MAPIE Methods
- `'plus'` - Best for most cases (adaptive intervals)
- `'base'` - Standard conformal prediction
- `'minmax'` - Min-max calibration
- `'naive'` - Simple residual-based

## Model Types
- TensorFlow: `'transformer'`, `'informer'`, `'fedformer'`, `'patchtst'`, `'itransformer'`, `'nbeats'`, `'nhits'`
- Tree: `'lightgbm'`

## Common Patterns

### Pattern 1: Multiple Confidence Levels
```python
df = mapie_pred.predict_intervals(X_price, X_meta, alphas=[0.01, 0.05, 0.10])
# Get 99%, 95%, 90% confidence intervals in one DataFrame
```

### Pattern 2: Check Coverage
```python
coverage = np.mean((y_actual >= df['lower_95']) & (y_actual <= df['upper_95']))
print(f"Coverage: {coverage:.1%}")  # Should be ~95%
```

### Pattern 3: Risk-Aware Decisions
```python
# Only trade if prediction is strong enough
strong_signals = df[df['width_95'] < threshold]
```

## See Also
- `MAPIE_GUIDE.md` - Complete documentation
- `dev/example_mapie.py` - Full examples
- `src/mapie.py` - Source code
