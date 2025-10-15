# Predictor Requirements - Production Ready

## Overview
Create production-ready predictor with no placeholders or half-measures.

## Correct Data Structure

### Input Shape
- **OHLCV**: (lookback, 5) - scaled price data
- **Meta**: (14,) - state + action features
  - State (8): equity, balance, long_value, short_value, long_sl, long_tp, short_sl, short_tp
  - Action (6): act_long_value, act_short_value, act_long_sl, act_long_tp, act_short_sl, act_short_tp

### Action Space (6D)
- `act_long_value`: Long position size (0 to action_value_max)
- `act_long_sl`: Long stop-loss distance (action_sl_min to action_sl_max)
- `act_long_tp`: Long take-profit distance (action_tp_min to action_tp_max)
- `act_short_value`: Short position size (0 to action_value_max)
- `act_short_sl`: Short stop-loss distance (action_sl_min to action_sl_max)
- `act_short_tp`: Short take-profit distance (action_tp_min to action_tp_max)

## 4 Core Requirements

### 1. Scale Raw Samples ✅
**Method**: `scale_raw_sample(ohlcv_window, state, action)`

**Input**:
- `ohlcv_window`: Dict with raw OHLCV arrays
- `state`: Dict with 8 state features (raw values)
- `action`: Dict with 6 action features (raw values)

**Output**:
- `X_price`: (lookback, 5) scaled OHLCV
- `X_meta`: (14,) scaled state + action

**Implementation**:
- Use `scale.scale_ohlcv_window()` for price
- Combine state + action into 14-feature dict
- Use `meta_scaler.transform()` for meta features
- Return scaled arrays ready for model

### 2. Make Predictions with Full State + Action ✅
**Method**: `predict(ohlcv_window, state, action, raw=True)` or `predict(X_price, X_meta, raw=False)`

**Input** (raw mode):
- `ohlcv_window`: Raw OHLCV dict
- `state`: Raw state dict (8 features)
- `action`: Raw action dict (6 features)

**Input** (pre-scaled mode):
- `X_price`: Pre-scaled (N, lookback, 5) or (lookback, 5)
- `X_meta`: Pre-scaled (N, 14) or (14,)

**Output**:
- Predicted rewards (N,) or scalar

**Implementation**:
- If raw=True, call `scale_raw_sample()` first
- Handle both single and batch inputs
- PyTorch: Use model(X_price_tensor, X_meta_tensor)
- LightGBM: Flatten price and concatenate with meta
- Return predictions

### 3. Generate Synthetic Actions with Predictions ✅
**Method**: `predict_many_actions(ohlcv_window, state, n_samples, **ranges)`

**Input**:
- `ohlcv_window`: OHLCV window
- `state`: Current state (8 features)
- `n_samples`: Number of actions to generate
- Optional: action ranges (use config defaults if not provided)

**Output**:
- DataFrame with columns: act_long_value, act_short_value, act_long_sl, act_long_tp, act_short_sl, act_short_tp, pred_reward

**Implementation**:
- Sample n_samples random actions across 6D space
- Include hold action (all zeros) if `include_hold=True`
- Batch scale all samples
- Batch predict all samples
- Return DataFrame with actions + predictions

### 4. Optimize to Find Optimal Action + Prediction ✅
**Method**: `find_optimal_action(ohlcv_window, state, **ranges)`

**Input**:
- `ohlcv_window`: OHLCV window
- `state`: Current state (8 features)
- Optional: action ranges and maxiter

**Output**:
- Dict with optimal action + prediction:
  - act_long_value, act_short_value, act_long_sl, act_long_tp, act_short_sl, act_short_tp
  - pred_reward

**Implementation**:
- Define objective function: minimize -pred_reward
- Use `scipy.optimize.differential_evolution` on 6D space
- Bounds: [long_value_range, sl_range, tp_range, short_value_range, sl_range, tp_range]
- Return optimal action dict with predicted reward
- Fallback to hold action if optimization fails

## Additional Methods

### `from_checkpoint(model_path, scaler_path, cfg, model_type)` ✅
Load predictor from saved files.

### `__repr__()` ✅
String representation showing config.

## Error Handling

- **No meta_scaler**: Raise error in raw mode, warn in constructor
- **Invalid inputs**: Raise ValueError with clear message
- **Optimization failure**: Return hold action with warning
- **Prediction failure**: Return penalty value (1e9) in optimization

## Configuration Defaults

```python
cfg = {
    "lookback": 200,
    "forward": 50,
    "action_value_max": 50000.0,
    "action_sl_min": 0.001,
    "action_sl_max": 0.05,
    "action_tp_min": 0.001,
    "action_tp_max": 0.10,
    "action_search_samples": 1000
}
```

## No Placeholders
- ✅ All methods fully implemented
- ✅ No TODO comments
- ✅ No "will implement later" sections
- ✅ Complete error handling
- ✅ Production-ready code quality

## Testing Coverage Needed
1. Scale raw samples (single and batch)
2. Predict with raw inputs
3. Predict with pre-scaled inputs
4. Sample many actions
5. Optimize actions
6. Load from checkpoint
7. All model types (7 neural + LightGBM)
8. Error cases

## Status
REQUIREMENTS DEFINED ✅
Ready for clean implementation.
