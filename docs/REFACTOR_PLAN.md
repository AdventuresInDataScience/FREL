# Reward System Refactor - Implementation Plan

## Overview
Complete refactor of the reward system to support:
- **Dual positions**: Simultaneous long + short (true hedging)
- **Proper SL/TP simulation**: Multiplier notation, checked every bar
- **Position context**: Actions evaluated given current state
- **Efficient storage**: Forward windows stored separately

## Key Design Decisions

### 1. Position Structure
```python
# OLD (single position)
position: -1/0/1
act_dir: "hold"/"long"/"short"
act_dollar: float
act_sl: float (fraction)
act_tp: float (fraction)

# NEW (dual positions)
long_value: float ($)
short_value: float ($)
long_sl: float (multiplier)
long_tp: float (multiplier)
short_sl: float (multiplier)
short_tp: float (multiplier)

# Actions (target state)
act_long_value: float
act_short_value: float
act_long_sl: float
act_long_tp: float
act_short_sl: float
act_short_tp: float
```

### 2. SL/TP Notation (Multiplier Format)
```python
# Long position
entry = 100
long_sl = 0.95  # Stop at 95 (5% loss) → entry * 0.95
long_tp = 1.10  # Profit at 110 (10% gain) → entry * 1.10

# Short position  
entry = 100
short_sl = 1.05  # Stop at 105 (5% loss) → entry * 1.05
short_tp = 0.90  # Profit at 90 (10% gain) → entry * 0.90

# "No SL/TP" representation
long_sl = 0.0   # No stop loss
long_tp = 0.0   # No take profit
```

### 3. Forward Windows - Separate Storage
```python
# Main samples table (many rows, ~10M)
samples = {
    'idx': [...],
    'equity': [...],
    'long_value': [...],
    # ... position/action data
    'close_scaled': [past_window_arrays],  # Only PAST
    # NO forward windows here!
    'y': [rewards]
}

# Separate forward windows table (unique idx, ~10K)
forward_windows = {
    'idx': [unique values],
    'forward_open': [scaled_arrays],
    'forward_high': [scaled_arrays],
    'forward_low': [scaled_arrays],
    'forward_close': [scaled_arrays],
    'forward_volume': [scaled_arrays]
}

# Efficiency: 10M samples with 10K unique indices
# Old: 10M forward windows = 4GB
# New: 10K forward windows = 4MB (1000x reduction!)
```

### 4. Scaling Consistency
```python
# Reference point: close[idx-1]
# (Last value of past window)

# Past window
past = close[idx-lookback:idx]
past_scaled = past / close[idx-1]  # Last value = 1.0

# Forward window
forward = close[idx:idx+forward]
forward_scaled = forward / close[idx-1]  # First value shows gap/continuation

# Example
close[idx-1] = 100  # Last of past
close[idx] = 102    # First of forward (entry price)
forward_scaled[0] = 102/100 = 1.02  ✓ Shows 2% overnight gap
```

## File Structure

```
config/
    default.yaml              # ✅ Updated with new params

src/
    data.py                   # (no changes)
    synth.py                  # ⏭️ TODO: New position structure
    forward_windows.py        # ✅ NEW: Generate/manage forward windows
    scale.py                  # ⏭️ Check if updates needed
    dataset.py                # ⏭️ TODO: Orchestrate pipeline
    reward.py                 # ⏭️ TODO: New simulation engine
    model.py                  # ⏭️ TODO: Handle new inputs
    predictor.py              # ⏭️ TODO: 6-value action space

data/
    raw_^GSPC.parquet         # Raw OHLCV (existing)
    forward_windows.parquet   # NEW: Scaled forward windows
    samples_10M.parquet       # Main dataset (updated structure)
    meta_scaler.json          # Scaler params (updated columns)
```

## Implementation Phases

### Phase 1: Data Infrastructure ✅ DONE
- [x] Update config.yaml
- [x] Create forward_windows.py

### Phase 2: Data Generation ⏭️ NEXT
- [ ] Refactor synth.py
  - New position structure (6 fields + actions)
  - Distribution-based sampling
  - Hold state/action generation
  - Strict validation
- [ ] Update dataset.py
  - Call forward_windows.generate_forward_windows()
  - Update meta_cols list
  - Pass forward_lookup to rewards

### Phase 3: Reward Calculation
- [ ] Refactor reward.py
  - _unscale_position_data() helper
  - _simulate_positions_forward() engine
  - Handle dual positions simultaneously
  - SL/TP in multiplier notation
- [ ] Test simulation
  - SL/TP hits (long/short)
  - Position transitions
  - Cost calculations

### Phase 4: Reward Metrics
- [ ] Implement reward functions
  - CAR, Sharpe, Sortino, Calmar
  - Time horizon considerations
  - Derived features (if needed)

### Phase 5: Model Integration
- [ ] Update model.py (new inputs)
- [ ] Update predictor.py (6-value actions)
- [ ] End-to-end testing

## Configuration Parameters

### Position Constraints
```yaml
max_leverage: 5.0              # Max (long + short) / equity
min_equity: 1000               # Minimum account value

# SL/TP bounds (multiplier notation)
sl_min: 0.50                   # Max 50% loss
sl_max: 1.50                   # Max 50% loss (short)
tp_min: 1.001                  # Min profit (adjusted for costs)
tp_max: 21.0                   # Max 2000% gain
```

### Synthetic Data Generation
```yaml
# Position values (log-normal distribution)
synth_equity_min: 10000
synth_equity_max: 100000
synth_long_value_min: 0
synth_long_value_max: 50000
synth_short_value_min: 0
synth_short_value_max: 50000

# SL/TP ranges (per direction)
synth_long_sl_min: 0.50        # Long stops
synth_long_sl_max: 0.99
synth_long_tp_min: 1.01        # Long profits
synth_long_tp_max: 21.0
synth_short_sl_min: 1.01       # Short stops
synth_short_sl_max: 1.50
synth_short_tp_min: 0.50       # Short profits
synth_short_tp_max: 0.99

# Distribution parameters
position_value_mean: 10000     # Log-normal center
position_value_sigma: 1.0      # Log-normal spread
tp_sl_mean: 0.05               # Truncated normal center
tp_sl_sigma: 0.03              # Truncated normal spread

# Hold sample representation
hold_state_pct: 0.10           # % flat (no positions)
hold_action_pct: 0.20          # % no change
```

## Validation Logic (in synth.py)

```python
def validate_sample(equity, balance, long_value, short_value, 
                   long_sl, long_tp, short_sl, short_tp, cfg):
    """Strict validation - reject invalid samples."""
    
    # Positive constraints
    assert equity > 0, "Equity must be positive"
    assert balance >= 0, "Balance cannot be negative"
    assert long_value >= 0, "Long value cannot be negative"
    assert short_value >= 0, "Short value cannot be negative"
    
    # Leverage constraint
    gross_exposure = long_value + short_value
    max_leverage = cfg['max_leverage']
    assert gross_exposure <= equity * max_leverage, \
        f"Leverage {gross_exposure/equity:.2f}x exceeds {max_leverage}x"
    
    # Balance consistency
    assert balance == equity - gross_exposure, "Balance inconsistent"
    
    # SL/TP bounds (if positions exist)
    if long_value > 0:
        assert cfg['synth_long_sl_min'] <= long_sl <= cfg['synth_long_sl_max']
        assert cfg['synth_long_tp_min'] <= long_tp <= cfg['synth_long_tp_max']
    
    if short_value > 0:
        assert cfg['synth_short_sl_min'] <= short_sl <= cfg['synth_short_sl_max']
        assert cfg['synth_short_tp_min'] <= short_tp <= cfg['synth_short_tp_max']
```

## Reward Simulation Engine

```python
def _simulate_positions_forward(
    forward_close: np.ndarray,     # Scaled prices from forward_lookup
    position_data: dict,            # Unscaled position/action data
    cfg: dict
) -> Tuple[np.ndarray, dict]:
    """
    Simulate dual positions through forward window.
    
    Process:
    1. Calculate position transitions (what changes?)
    2. Close current positions (if changing)
    3. Open target positions
    4. Simulate both long/short with SL/TP checks
    5. Return daily P&L array + metadata
    
    Returns:
        daily_pnl: Array of daily P&L (length = forward-1)
        metadata: Dict with exit info, costs, etc.
    """
    # Implementation in next phase
    pass
```

## Testing Strategy

### Unit Tests
- [ ] synth.py validation logic
- [ ] forward_windows.py generation/scaling
- [ ] reward.py simulation engine
- [ ] SL/TP hit detection (long/short)

### Integration Tests
- [ ] End-to-end data generation
- [ ] Reward calculation with forward lookup
- [ ] Scaling/unscaling consistency
- [ ] Edge cases (insufficient data, extreme values)

### Validation Tests
- [ ] Position constraints enforced
- [ ] SL/TP notation correct
- [ ] Forward window alignment
- [ ] Cost calculations accurate

## Next Steps

**Immediate**: Refactor synth.py with new position structure

1. Update data structure (remove old fields, add new 6+6 fields)
2. Implement distribution-based sampling
3. Add validation logic
4. Generate hold states/actions
5. Test with small dataset (100 samples)

**Ready to proceed?** I can start with synth.py refactoring.
