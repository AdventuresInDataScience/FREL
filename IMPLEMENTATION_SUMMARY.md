# Predictor Implementation - Summary of Changes

## Overview
Added a comprehensive `Predictor` class for model inference with optimal action search capabilities. The predictor handles both raw (unscaled) and pre-scaled data, making it suitable for both testing and live trading scenarios.

## Files Created

### 1. `src/predictor.py` (NEW)
A complete wrapper class for trained models with the following features:
- **Dual input modes**: `raw=True` for unscaled data, `raw=False` for pre-scaled test batches
- **Model support**: Works with both TensorFlow/Keras and LightGBM models
- **Optimal action search**: Two methods - sampling and optimization
- **Ground truth comparison**: Compare model predictions vs true optimal actions
- **Checkpoint loading**: Easy serialization/deserialization

**Key Methods**:
- `predict()` - Main inference method with flexible input handling
- `predict_all_actions()` - Sample many random actions and predict their rewards
- `find_optimal_action()` - Use differential evolution to find optimal action
- `compute_true_reward()` - Calculate actual reward using reward functions
- `compare_predicted_vs_true()` - Full comparison of model vs ground truth
- `from_checkpoint()` - Class method to load saved models

### 2. `tests/test_predictor.py` (NEW)
Comprehensive test suite covering:
- Initialization and checkpoint loading
- Predictions with raw and pre-scaled data
- Batch and single sample predictions
- Optimal action search (both methods)
- Error handling and edge cases

## Files Updated

### 3. `src/reward.py`
Added three new functions for optimal action computation:

**`compute_all_actions()`**
- Samples random actions within specified bounds
- Computes actual rewards for each action
- Returns DataFrame with all actions and rewards
- Fast, approximate method for finding good actions

**`find_optimal_action()`**
- Uses scipy's differential evolution optimizer
- Finds truly optimal action within bounds
- Slower but more accurate than sampling
- Handles both long and short directions

**`compute_optimal_labels()`**
- Batch processing for multiple samples
- Wrapper that calls either sampling or optimization method
- Returns DataFrame with optimal actions for all samples
- Used during dataset preparation

**Bug fix**: Fixed `compute_many()` to include `sl` and `tp` in action dict

### 4. `src/dataset.py`
Enhanced `build_dataset()` function with optional optimal action computation:

**New Parameters**:
- `compute_optimal` (bool) - Whether to compute optimal actions
- `optimal_method` (str) - 'sample' or 'optimize'

**Behavior**:
- If `compute_optimal=True`, adds columns: `opt_dir`, `opt_dollar`, `opt_sl`, `opt_tp`, `opt_reward`
- Saves to different filename with `_opt` suffix
- Preserves backward compatibility when `compute_optimal=False`

### 5. `tests/test_reward.py`
Completely rewrote with comprehensive tests:
- Tests for all reward metrics (CAR, Sharpe, Sortino, Calmar)
- Tests for `compute_all_actions()` with various scenarios
- Tests for `find_optimal_action()` optimization
- Tests for `compute_optimal_labels()` batch processing
- Edge cases (flat prices, hold actions, etc.)
- Parametrized tests for all reward functions

### 6. `dev/main.py`
Updated example script to demonstrate full workflow:

**New Features**:
1. Train/test split for proper evaluation
2. Model training with separate train/test data
3. Creating `Predictor` instance from checkpoint
4. Testing on pre-scaled test batch (`raw=False`)
5. Testing on raw OHLCV data (`raw=True`)
6. Comparing predicted optimal vs true optimal actions
7. Computing optimality gap metrics
8. Visualization of predictions

**Workflow**:
```
Build Dataset → Split Train/Test → Train Model → Save Checkpoint
→ Load Predictor → Test on Pre-scaled Data → Test on Raw Data
→ Compare vs Ground Truth → Visualize Results
```

### 7. `config/default.yaml`
Added configuration for optimal action computation:
```yaml
compute_optimal: false       # whether to compute optimal actions
optimal_method: "sample"     # sample | optimize
optimal_samples: 1000        # samples per direction for 'sample' method
optimal_maxiter: 100         # max iterations for 'optimize' method
```

### 8. `models/` directory (NEW)
Created directory for saving trained models.

## Key Design Decisions

### 1. Dual Input Mode (`raw` parameter)
**Problem**: Need to handle both pre-scaled test batches and raw live data
**Solution**: Single `predict()` method with `raw` flag
- `raw=False`: Pass pre-scaled `X_price` and `X_meta` (for testing)
- `raw=True`: Pass raw `sample` dict (for live trading)

### 2. Optimal Action Search Methods
**Sample Method** (fast, approximate):
- Randomly samples N actions per direction
- Computes rewards for all samples
- Returns best action found
- Good for: Research, prototyping, large-scale testing

**Optimize Method** (slow, accurate):
- Uses differential evolution optimizer
- Finds true optimal within bounds
- Much slower but more accurate
- Good for: Benchmarking, final evaluation

### 3. Predictor as Wrapper
**Benefits**:
- Encapsulates model + scaler together
- Handles all scaling internally
- Provides high-level interface
- Easy to serialize/deserialize
- Consistent API regardless of model type

### 4. Ground Truth Comparison
**Purpose**: Measure how well model predicts optimal actions
**Metrics**:
- `predicted_action`: What model thinks is best
- `true_optimal_action`: What actually is best
- `predicted_true_reward`: How well predicted action actually performs
- `optimality_gap`: Difference between optimal and predicted

## Usage Examples

### Testing on Pre-scaled Batch
```python
# Load predictor
predictor = Predictor.from_checkpoint(
    model_path="models/transformer_model.h5",
    scaler_path="data/meta_scaler.json",
    cfg=CFG,
    model_type="transformer"
)

# Predict on test batch
y_pred = predictor.predict(
    X_price=X_price_test,
    X_meta=X_meta_test,
    raw=False
)
```

### Live Trading with Raw Data
```python
# Prepare raw market data
ohlcv_window = {
    "open": recent_bars["open"].values[-200:],
    "high": recent_bars["high"].values[-200:],
    "low": recent_bars["low"].values[-200:],
    "close": recent_bars["close"].values[-200:],
    "volume": recent_bars["volume"].values[-200:],
}

state = {
    "equity": current_equity,
    "balance": current_balance,
    "position": current_position,
    "sl_dist": 0.02,
    "tp_dist": 0.04
}

# Find optimal action
optimal = predictor.find_optimal_action(
    ohlcv_window, 
    state,
    maxiter=100
)

print(f"Trade {optimal['dir']} with ${optimal['dollar']:.2f}")
```

### Computing Optimal Labels During Dataset Build
```python
path = build_dataset(
    CFG,
    n_samples=1_000_000,
    compute_optimal=True,
    optimal_method="sample"  # or "optimize"
)

df = pd.read_parquet(path)
# Now df has columns: opt_dir, opt_dollar, opt_sl, opt_tp, opt_reward
```

## Testing
Run tests with:
```bash
pytest tests/test_predictor.py -v
pytest tests/test_reward.py -v
```

## Next Steps

1. **Train models** on optimal labels instead of synthetic actions
2. **Curriculum learning** - train on phases 0→1→2
3. **Hyperparameter tuning** for both sampling and optimization
4. **Ensemble methods** - combine multiple models
5. **Online learning** - update models with new data
6. **Production deployment** - integrate with live trading system

## Notes

- The `Predictor` class is the main interface for all inference
- Use `raw=False` for testing, `raw=True` for production
- Optimal action computation is expensive - use sampling for experimentation
- All functions have type hints and comprehensive docstrings
- Tests cover edge cases and error handling
