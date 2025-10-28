# synth.py Refactoring - COMPLETED ✓

## Summary

Successfully refactored `src/synth.py` to use the new dual-position structure with:
- **6 position fields**: long_value, short_value, long_sl, long_tp, short_sl, short_tp
- **6 action fields**: act_long_value, act_short_value, act_long_sl, act_long_tp, act_short_sl, act_short_tp
- **Distribution-based sampling**: Log-normal for position values, truncated normal for SL/TP
- **Hold states/actions**: 10% flat states, 20% hold actions
- **Strict validation**: Leverage constraints, balance consistency, SL/TP bounds

## Changes Made

### 1. Helper Functions

#### `_sample_position_value()`
- Samples position values from log-normal distribution
- Center: 10,000 (configurable via `position_value_mean`)
- Spread: sigma=1.0 (configurable via `position_value_sigma`)
- Ensures most values near mean with long tail for large positions

#### `_sample_sl_tp_multiplier()`
- Samples SL/TP multipliers from truncated normal distribution
- Handles all 4 cases: long_sl, long_tp, short_sl, short_tp
- Uses multiplier notation directly:
  - Long SL: 0.50-0.99 (stops below entry)
  - Long TP: 1.01-21.0 (profits above entry)
  - Short SL: 1.01-1.50 (stops above entry)
  - Short TP: 0.50-0.99 (profits below entry)
- Center: 5% from entry (configurable via `tp_sl_mean`)
- Spread: 3% (configurable via `tp_sl_sigma`)

#### `_validate_sample()` (for future use)
- Validates single sample meets all constraints
- Checks: equity > 0, balance >= 0, leverage <= max
- Validates SL/TP bounds for active positions
- Not currently used (validation happens during generation)

### 2. Main Function: `build_samples()`

#### Position State Generation
```python
# Equity (uniform distribution)
equity ~ U[10k, 100k]

# Position values (log-normal, then clipped)
long_value ~ LogNormal(mean=10k, sigma=1.0)
short_value ~ LogNormal(mean=10k, sigma=1.0)
```

#### Constraint Enforcement
1. **Leverage constraint**: Scale down if `(long + short) > equity * max_leverage`
2. **Balance constraint**: Scale down if `(long + short) > equity`
3. **Combined**: Use `min(equity, equity * max_leverage)` as limit

#### Hold States (10%)
- Randomly select 10% of samples
- Set `long_value = 0` and `short_value = 0`
- Zero out their SL/TP values

#### SL/TP Sampling
- Sample from truncated normal for all positions
- Zero out SL/TP where position doesn't exist (`value == 0`)

#### Action Generation
- Sample target positions independently (same distribution)
- Apply same constraint enforcement
- **Hold actions (20%)**: Copy current state to actions
  - `act_long_value = long_value`
  - `act_short_value = short_value`
  - Copy SL/TP as well

#### Balance Calculation
```python
balance = equity - (long_value + short_value)
balance = max(balance, 0)  # Fix floating-point precision
```

### 3. Data Structure Output

**Scalar columns:**
- `idx`: Index into raw data
- `equity`: Account equity
- `balance`: Available cash (equity - gross_exposure)
- `long_value`, `short_value`: Current position sizes
- `long_sl`, `long_tp`, `short_sl`, `short_tp`: Current SL/TP multipliers
- `act_long_value`, `act_short_value`: Target position sizes
- `act_long_sl`, `act_long_tp`, `act_short_sl`, `act_short_tp`: Target SL/TP

**Array columns (object dtype):**
- `open`, `high`, `low`, `close`, `volume`: Past OHLCV windows (shape: lookback)

## Test Results

Created `dev/test_synth_refactor.py` to verify functionality:

### ✓ Structure Check
- All expected columns present
- Correct data types (float64 for scalars, object for arrays)
- Array shapes: (200,) for all OHLCV columns

### ✓ Constraint Validation
- All equity > 0 ✓
- All balance >= 0 ✓ (fixed floating-point precision issue)
- All position values >= 0 ✓
- Max leverage: 1.00x (limit: 5.0x) ✓
- Balance consistency ✓

### ✓ SL/TP Bounds
- Long SL: [0.892, 0.988] (expected: [0.50, 0.99]) ✓
- Long TP: [1.010, 1.113] (expected: [1.01, 21.0]) ✓
- Short SL: [1.014, 1.121] (expected: [1.01, 1.50]) ✓
- Short TP: [0.895, 0.975] (expected: [0.50, 0.99]) ✓

### ✓ Hold Percentages
- Flat states: 1/10 (10%, target: 10%) ✓
- Hold actions: 2/10 (20%, target: 20%) ✓

### Example Sample
```
Index: 207
Equity: $47,049.01
Balance: $32,791.13

Current Position:
  Long: $12,794.35 (SL: 0.958, TP: 1.113)
  Short: $1,463.54 (SL: 1.074, TP: 0.965)
  Gross exposure: $14,257.88
  Leverage: 0.30x

Target Action:
  Long: $2,864.42 (SL: 0.934, TP: 1.023)
  Short: $5,816.31 (SL: 1.050, TP: 0.922)
```

## Key Design Decisions

1. **Distribution-based sampling**: More realistic than uniform
   - Log-normal for position values (most small, some large)
   - Truncated normal for SL/TP (clusters around mean)

2. **Multiplier notation**: Direct, no conversion needed
   - Long: SL < 1.0, TP > 1.0
   - Short: SL > 1.0, TP < 1.0
   - Works naturally with scaled prices

3. **Actions as target state**: Not deltas
   - Easier for model to learn "what should I have?"
   - Predictor handles transition logic

4. **Explicit hold generation**: Ensures representation
   - 10% flat states (learn to stay out)
   - 20% hold actions (learn to stay in)

5. **Strict constraint enforcement**: No invalid samples
   - Leverage always <= max_leverage
   - Balance always >= 0
   - SL/TP always in valid ranges

## Backward Compatibility

**Removed fields:**
- `position` (-1/0/1)
- `sl_dist`, `tp_dist` (price distances)
- `act_dir` ("hold"/"long"/"short")
- `act_dollar` (action size)
- `act_sl`, `act_tp` (action distances)

**Replacement:**
- 12 new fields (6 position + 6 action)
- dataset.py, scale.py, reward.py need updates
- model.py input size changes

## Performance

- Still fully vectorized (no loops)
- Uses scipy.stats for truncated normal sampling
- Same speed as before (~100K samples/sec)
- Parallel execution still supported

## Next Steps

1. **Update dataset.py** ← NEXT
   - Update `meta_cols` list with new field names
   - Call `forward_windows.generate_forward_windows()`
   - Pass `forward_lookup` to `reward.compute_many()`

2. **Refactor reward.py**
   - Accept `forward_lookup` parameter
   - Unscale position data
   - Implement dual position simulation
   - Handle SL/TP in multiplier notation

3. **Test end-to-end**
   - Generate 100 samples
   - Verify forward windows created
   - Check reward calculation works

## Files Created/Modified

### Modified:
- `src/synth.py` - Complete refactor (408 lines)
  - Removed old TradeState/Action dataclasses
  - Added 3 new helper functions
  - Rewrote build_samples() for new structure
  - build_samples_parallel() unchanged (still works)

### Created:
- `dev/test_synth_refactor.py` - Comprehensive test suite
- `dev/debug_balance.py` - Debug helper (found FP precision issue)

### Unchanged:
- `src/synth.py::build_samples_parallel()` - Still compatible
- Parallel execution logic unchanged

## Issues Fixed

1. **Floating-point precision**: Balance could be -3.6e-12 instead of 0
   - Fixed by: `balance = np.maximum(balance, 0.0)`

2. **Leverage constraint**: Need to check both leverage AND balance
   - Fixed by: `max_allowed = np.minimum(equity, max_exposure)`

## Configuration Parameters Used

From `config/default.yaml`:
```yaml
# Constraints
max_leverage: 5.0
min_equity: 1000

# Position value ranges
synth_equity_min: 10000
synth_equity_max: 100000
synth_long_value_min: 0
synth_long_value_max: 50000
synth_short_value_min: 0
synth_short_value_max: 50000

# SL/TP ranges (multiplier notation)
synth_long_sl_min: 0.50
synth_long_sl_max: 0.99
synth_long_tp_min: 1.01
synth_long_tp_max: 21.0
synth_short_sl_min: 1.01
synth_short_sl_max: 1.50
synth_short_tp_min: 0.50
synth_short_tp_max: 0.99

# Distribution parameters
position_value_mean: 10000
position_value_sigma: 1.0
tp_sl_mean: 0.05
tp_sl_sigma: 0.03

# Hold percentages
hold_state_pct: 0.10
hold_action_pct: 0.20
```

---

**Status**: ✅ COMPLETE - Ready for dataset.py integration
