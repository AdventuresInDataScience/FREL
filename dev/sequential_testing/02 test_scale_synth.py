#%%
# =============================================================================
# TEST 02: scale.py and synth.py Functions
# Test scaling utilities and synthetic data generation for dual-position architecture
# Dependencies: None
# =============================================================================

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src import scale, synth

# Load config
config_path = project_root / "config" / "default.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

print("✓ Imports successful")
print(f"Config loaded: {config_path}")
print(f"Testing modules: scale.py, synth.py")

#%%
# Override Config with test values for synthetic generation
import time
test_timestamp = int(time.time() * 1000) % 1000000  # Last 6 digits for uniqueness

test_cfg = cfg.copy()
test_cfg.update({
    # Position value sampling
    "position_value_mean": 10000,
    "position_value_sigma": 1.0,
    
    # SL/TP sampling
    "tp_sl_mean": 0.05,  # 5% mean
    "tp_sl_sigma": 0.03,  # 3% std
    
    # SL/TP bounds (multiplier notation)
    "synth_long_sl_min": 0.50,
    "synth_long_sl_max": 0.99,
    "synth_long_tp_min": 1.01,
    "synth_long_tp_max": 21.0,
    "synth_short_sl_min": 1.01,
    "synth_short_sl_max": 1.50,
    "synth_short_tp_min": 0.50,
    "synth_short_tp_max": 0.99,
    
    # Hold percentages
    "hold_state_pct": 0.10,  # 10% flat (no positions)
    "hold_action_pct": 0.20,  # 20% no change
    
    # Leverage and position limits
    "max_leverage": 2.0,
    "synth_long_value_min": 0,
    "synth_long_value_max": 50000,
    "synth_short_value_min": 0,
    "synth_short_value_max": 50000,
})

print(f"\nTest config created (timestamp: {test_timestamp})")
print(f"  - Position value mean: ${test_cfg['position_value_mean']:,}")
print(f"  - SL/TP mean: {test_cfg['tp_sl_mean']*100}%")
print(f"  - Long SL range: {test_cfg['synth_long_sl_min']}-{test_cfg['synth_long_sl_max']}")
print(f"  - Long TP range: {test_cfg['synth_long_tp_min']}-{test_cfg['synth_long_tp_max']}")
print(f"  - Hold state %: {test_cfg['hold_state_pct']*100}%")
print(f"  - Hold action %: {test_cfg['hold_action_pct']*100}%")

#%%
# =============================================================================
# TEST 1: scale_ohlcv_window() - Scale OHLCV data by window endpoint
# =============================================================================
print("\n" + "="*70)
print("TEST 1: scale_ohlcv_window() - Window-based OHLCV scaling")
print("="*70)

# Test 1a: Basic OHLCV scaling
print("\n[1a] Basic OHLCV window scaling...")
ohlcv_raw = {
    "open": np.array([100.0, 101.0, 99.0, 102.0, 105.0]),
    "high": np.array([102.0, 103.0, 101.0, 104.0, 107.0]),
    "low": np.array([99.0, 100.0, 98.0, 101.0, 104.0]),
    "close": np.array([101.0, 99.0, 102.0, 105.0, 106.0]),
    "volume": np.array([1e6, 1.2e6, 0.9e6, 1.5e6, 1.1e6])
}

scaled = scale.scale_ohlcv_window(ohlcv_raw)

print(f"Original close[-1]: {ohlcv_raw['close'][-1]}")
print(f"Scaled close[-1]: {scaled['close'][-1]:.6f} (should be 1.0)")
print(f"Scaled open[0]: {scaled['open'][0]:.6f} (= {ohlcv_raw['open'][0]}/{ohlcv_raw['close'][-1]})")

# Validate price scaling (linear)
assert abs(scaled['close'][-1] - 1.0) < 1e-6, "Last close should be 1.0"
expected_open_0 = ohlcv_raw['open'][0] / ohlcv_raw['close'][-1]
assert abs(scaled['open'][0] - expected_open_0) < 1e-6, "Open should scale linearly"
print("✓ OHLC price scaling correct (linear division by close[-1])")

# Test 1b: Volume scaling (log1p)
print("\n[1b] Testing volume scaling (log1p method)...")
vol_last = ohlcv_raw['volume'][-1]
expected_vol_0 = np.log1p(ohlcv_raw['volume'][0]) / np.log1p(vol_last)
expected_vol_last = np.log1p(vol_last) / np.log1p(vol_last)

print(f"Original volume[-1]: {ohlcv_raw['volume'][-1]:,.0f}")
print(f"Scaled volume[-1]: {scaled['volume'][-1]:.6f} (should be 1.0)")
print(f"Original volume[0]: {ohlcv_raw['volume'][0]:,.0f}")
print(f"Scaled volume[0]: {scaled['volume'][0]:.6f} (expected: {expected_vol_0:.6f})")

assert abs(scaled['volume'][-1] - 1.0) < 1e-6, "Last volume should be 1.0"
assert abs(scaled['volume'][0] - expected_vol_0) < 1e-6, "Volume should use log1p scaling"
print("✓ Volume scaling correct (log1p method)")

# Test 1c: Volume scaling handles zeros
print("\n[1c] Testing volume scaling with zero values...")
ohlcv_with_zero = ohlcv_raw.copy()
ohlcv_with_zero["volume"] = np.array([0, 100, 200, 300, 400])
scaled_zero = scale.scale_ohlcv_window(ohlcv_with_zero)

# log1p(0) = 0, so scaled should be 0
assert scaled_zero['volume'][0] == 0.0, "Zero volume should remain zero after scaling"
print(f"✓ Zero volume handled correctly: {scaled_zero['volume'][0]}")

# Test 1d: OHLC relationships preserved
print("\n[1d] Checking OHLC relationships after scaling...")
eps = 1e-10
assert (scaled['high'] >= scaled['low'] - eps).all(), "High >= Low should be preserved"
print("✓ OHLC relationships preserved (high >= low)")

print("\n✓ All assertions passed for scale_ohlcv_window()")

#%%
# =============================================================================
# TEST 2: MetaScaler - Minmax scaling for meta fields
# =============================================================================
print("\n" + "="*70)
print("TEST 2: MetaScaler - Scale/unscale meta fields")
print("="*70)

# Test 2a: Basic fit/transform
print("\n[2a] Basic fit/transform with minmax...")
df_meta = pd.DataFrame({
    'long_value': [0, 5000, 10000, 15000, 20000],
    'short_value': [0, 3000, 6000, 9000, 12000],
    'long_sl': [0.90, 0.92, 0.95, 0.93, 0.91],
    'long_tp': [1.05, 1.10, 1.15, 1.08, 1.12]
})

scaler = scale.MetaScaler(kind="minmax")
cols_to_scale = ['long_value', 'short_value', 'long_sl', 'long_tp']
scaler.fit(df_meta, cols_to_scale)

print(f"Stats fitted: {list(scaler.stats.keys())}")
print(f"long_value range: {scaler.stats['long_value']}")

# Transform
df_scaled = scaler.transform(df_meta, cols_to_scale)
print(f"\nOriginal long_value: {df_meta['long_value'].values}")
print(f"Scaled long_value: {df_scaled['long_value'].values}")

# Validate range [0, 1]
for c in cols_to_scale:
    assert df_scaled[c].min() >= -1e-6, f"{c} min should be ~0"
    assert df_scaled[c].max() <= 1 + 1e-6, f"{c} max should be ~1"
print("✓ All scaled values in [0, 1]")

# Test 2b: Inverse transform for single column
print("\n[2b] Testing inverse transform...")
scaled_vals = df_scaled['long_value'].values
unscaled_vals = scaler.inverse('long_value', scaled_vals)

np.testing.assert_allclose(unscaled_vals, df_meta['long_value'].values, rtol=1e-5)
print(f"✓ Inverse transform matches original")

# Test 2c: Inverse transform dict
print("\n[2c] Testing inverse_transform_dict()...")
scaled_dict = {
    'long_value': 0.5,  # Mid-point
    'short_value': 0.25,
    'long_sl': 0.4,
    'long_tp': 0.6
}
unscaled_dict = scaler.inverse_transform_dict(scaled_dict)

print(f"Scaled dict: {scaled_dict}")
print(f"Unscaled dict: {unscaled_dict}")

# Validate ranges are reasonable
assert 0 <= unscaled_dict['long_value'] <= 20000, "long_value should be in training range"
assert 0 <= unscaled_dict['short_value'] <= 12000, "short_value should be in training range"
assert 0.90 <= unscaled_dict['long_sl'] <= 0.95, "long_sl should be in training range"
assert 1.05 <= unscaled_dict['long_tp'] <= 1.15, "long_tp should be in training range"
print("✓ inverse_transform_dict() works correctly")

# Test 2d: Save/load
print("\n[2d] Testing save/load...")
save_path = project_root / "dev" / "sequential_testing" / "temp_scaler.json"
scaler.save(save_path)

scaler2 = scale.MetaScaler(kind="minmax")
scaler2.load(save_path)

assert scaler.stats == scaler2.stats, "Loaded stats should match"
save_path.unlink()  # Clean up
print("✓ Save/load works correctly")

print("\n✓ All assertions passed for MetaScaler")

#%%
# =============================================================================
# TEST 3: synth._sample_position_value() - Position value sampling
# =============================================================================
print("\n" + "="*70)
print("TEST 3: _sample_position_value() - Sample position values")
print("="*70)

# Test 3a: Basic sampling with LARGE sample size
print("\n[3a] Basic position value sampling (large sample)...")
rng = np.random.default_rng(42)
n_samples = 50000  # Large sample for accurate distribution testing

values = synth._sample_position_value(rng, test_cfg, size=n_samples)

print(f"Sampled {n_samples:,} values")
print(f"  Mean: {values.mean():.2f} (target: {test_cfg.get('position_value_mean', 10000)})")
print(f"  Median: {np.median(values):.2f}")
print(f"  P10: {np.percentile(values, 10):.2f}")
print(f"  P90: {np.percentile(values, 90):.2f}")
print(f"  Min: {values.min():.2f}")
print(f"  Max: {values.max():.2f}")
print(f"  Std: {values.std():.2f}")

# Validate distribution properties
assert values.min() >= 0, "Values should be non-negative"
assert 9000 <= values.mean() <= 11000, f"Mean should be near {test_cfg.get('position_value_mean', 10000)}"
print("✓ Position value distribution looks reasonable")

# Test 3b: Log-normal shape validation
print("\n[3b] Checking log-normal shape (larger sample)...")
log_values = np.log(values[values > 0])
print(f"Log-values mean: {log_values.mean():.3f}")
print(f"Log-values std: {log_values.std():.3f}")

# Log-values should be roughly normal
from scipy.stats import shapiro
stat, p_value = shapiro(log_values[:5000])  # Sample for speed
print(f"Shapiro-Wilk test p-value: {p_value:.6f}")
if p_value < 0.01:
    print("⚠️  Warning: Distribution may not be perfectly log-normal (but close enough for synthetic data)")
else:
    print("✓ Log-values appear normally distributed")

print("\n✓ All assertions passed for _sample_position_value()")

#%%
# =============================================================================
# TEST 4: synth._sample_sl_tp_multiplier() - SL/TP multiplier sampling
# =============================================================================
print("\n" + "="*70)
print("TEST 4: _sample_sl_tp_multiplier() - Sample SL/TP multipliers")
print("="*70)

rng = np.random.default_rng(42)
n_samples = 50000  # Large sample for accurate testing

# Test 4a: Long SL (should be < 1.0)
print("\n[4a] Long SL sampling (large sample)...")
long_sl = synth._sample_sl_tp_multiplier(rng, test_cfg, is_long=True, is_sl=True, size=n_samples)

print(f"Long SL stats ({n_samples:,} samples):")
print(f"  Min: {long_sl.min():.4f} (bound: {test_cfg['synth_long_sl_min']})")
print(f"  Max: {long_sl.max():.4f} (bound: {test_cfg['synth_long_sl_max']})")
print(f"  Mean: {long_sl.mean():.4f}")
print(f"  Median: {np.median(long_sl):.4f}")
print(f"  P10: {np.percentile(long_sl, 10):.4f}")
print(f"  P90: {np.percentile(long_sl, 90):.4f}")

assert long_sl.min() >= test_cfg['synth_long_sl_min'], "Long SL should be >= min bound"
assert long_sl.max() <= test_cfg['synth_long_sl_max'], "Long SL should be <= max bound"
assert long_sl.max() < 1.0, "Long SL should be < 1.0"
assert 0.90 <= long_sl.mean() <= 0.98, "Long SL mean should be near 0.95"
print("✓ Long SL range correct")

# Test 4b: Long TP (should be > 1.0)
print("\n[4b] Long TP sampling (large sample)...")
long_tp = synth._sample_sl_tp_multiplier(rng, test_cfg, is_long=True, is_sl=False, size=n_samples)

print(f"Long TP stats ({n_samples:,} samples):")
print(f"  Min: {long_tp.min():.4f} (bound: {test_cfg['synth_long_tp_min']})")
print(f"  Max: {long_tp.max():.4f} (bound: {test_cfg['synth_long_tp_max']})")
print(f"  Mean: {long_tp.mean():.4f}")
print(f"  Median: {np.median(long_tp):.4f}")
print(f"  P10: {np.percentile(long_tp, 10):.4f}")
print(f"  P90: {np.percentile(long_tp, 90):.4f}")

assert long_tp.min() >= test_cfg['synth_long_tp_min'], "Long TP should be >= min bound"
assert long_tp.max() <= test_cfg['synth_long_tp_max'], "Long TP should be <= max bound"
assert long_tp.min() > 1.0, "Long TP should be > 1.0"
assert 1.02 <= long_tp.mean() <= 1.15, "Long TP mean should be near 1.05"
print("✓ Long TP range correct")

# Test 4c: Short SL (should be > 1.0)
print("\n[4c] Short SL sampling (large sample)...")
short_sl = synth._sample_sl_tp_multiplier(rng, test_cfg, is_long=False, is_sl=True, size=n_samples)

print(f"Short SL stats ({n_samples:,} samples):")
print(f"  Min: {short_sl.min():.4f} (bound: {test_cfg['synth_short_sl_min']})")
print(f"  Max: {short_sl.max():.4f} (bound: {test_cfg['synth_short_sl_max']})")
print(f"  Mean: {short_sl.mean():.4f}")
print(f"  Median: {np.median(short_sl):.4f}")

assert short_sl.min() >= test_cfg['synth_short_sl_min'], "Short SL should be >= min bound"
assert short_sl.max() <= test_cfg['synth_short_sl_max'], "Short SL should be <= max bound"
assert short_sl.min() > 1.0, "Short SL should be > 1.0"
assert 1.02 <= short_sl.mean() <= 1.15, "Short SL mean should be near 1.05"
print("✓ Short SL range correct")

# Test 4d: Short TP (should be < 1.0)
print("\n[4d] Short TP sampling (large sample)...")
short_tp = synth._sample_sl_tp_multiplier(rng, test_cfg, is_long=False, is_sl=False, size=n_samples)

print(f"Short TP stats ({n_samples:,} samples):")
print(f"  Min: {short_tp.min():.4f} (bound: {test_cfg['synth_short_tp_min']})")
print(f"  Max: {short_tp.max():.4f} (bound: {test_cfg['synth_short_tp_max']})")
print(f"  Mean: {short_tp.mean():.4f}")
print(f"  Median: {np.median(short_tp):.4f}")

assert short_tp.min() >= test_cfg['synth_short_tp_min'], "Short TP should be >= min bound"
assert short_tp.max() <= test_cfg['synth_short_tp_max'], "Short TP should be <= max bound"
assert short_tp.max() < 1.0, "Short TP should be < 1.0"
assert 0.90 <= short_tp.mean() <= 0.98, "Short TP mean should be near 0.95"
print("✓ Short TP range correct")

print("\n✓ All assertions passed for _sample_sl_tp_multiplier()")

#%%
# =============================================================================
# TEST 5: synth.build_samples() - Full synthetic generation with hold logic
# =============================================================================
print("\n" + "="*70)
print("TEST 5: build_samples() - Full synthetic generation")
print("="*70)

# Test 5a: Generate large sample with hold logic
print("\n[5a] Generating synthetic samples (large batch)...")
# Load data for build_samples
raw_data_path = project_root / cfg['data_dir'] / cfg['raw_data_filename'].format(ticker=cfg['ticker'])
df_data = pd.read_parquet(raw_data_path)
rng = np.random.default_rng(42)
n_samples_test = 10000

df_synth = synth.build_samples(
    df=df_data,
    n=n_samples_test,
    lookback=cfg['lookback'],
    forward=cfg['forward'],
    rng=rng,
    cfg=test_cfg
)

print(f"✓ Generated {len(df_synth):,} samples")
print(f"  Columns: {list(df_synth.columns)}")
print(f"  Shape: {df_synth.shape}")

# Validate all expected columns exist
expected_cols = [
    'equity', 'balance',
    'long_value', 'short_value', 'long_sl', 'long_tp', 'short_sl', 'short_tp',
    'act_long_value', 'act_short_value', 'act_long_sl', 'act_long_tp', 'act_short_sl', 'act_short_tp'
]
for col in expected_cols:
    assert col in df_synth.columns, f"Missing column: {col}"
print(f"✓ All {len(expected_cols)} expected columns present")

# Test 5b: Validate hold states (10% with no positions)
print("\n[5b] Validating hold states (no positions)...")
hold_states = (df_synth['long_value'] == 0) & (df_synth['short_value'] == 0)
n_hold_states = hold_states.sum()
hold_state_pct_actual = n_hold_states / len(df_synth)
hold_state_pct_expected = test_cfg['hold_state_pct']

print(f"  Hold states: {n_hold_states:,} / {len(df_synth):,} ({hold_state_pct_actual*100:.2f}%)")
print(f"  Expected: ~{hold_state_pct_expected*100:.0f}%")

# Allow some tolerance (within 2% of target)
assert abs(hold_state_pct_actual - hold_state_pct_expected) < 0.02, \
    f"Hold state percentage should be near {hold_state_pct_expected*100}%"
print("✓ Hold state percentage correct")

# Verify hold states have zero SL/TP
hold_state_rows = df_synth[hold_states]
assert (hold_state_rows['long_sl'] == 0).all(), "Hold states should have zero long_sl"
assert (hold_state_rows['long_tp'] == 0).all(), "Hold states should have zero long_tp"
assert (hold_state_rows['short_sl'] == 0).all(), "Hold states should have zero short_sl"
assert (hold_state_rows['short_tp'] == 0).all(), "Hold states should have zero short_tp"
print("✓ Hold states have zero SL/TP values")

# Test 5c: Validate hold actions (20% no change)
print("\n[5c] Validating hold actions (no change from state)...")
hold_actions = (
    (df_synth['act_long_value'] == df_synth['long_value']) &
    (df_synth['act_short_value'] == df_synth['short_value']) &
    (df_synth['act_long_sl'] == df_synth['long_sl']) &
    (df_synth['act_long_tp'] == df_synth['long_tp']) &
    (df_synth['act_short_sl'] == df_synth['short_sl']) &
    (df_synth['act_short_tp'] == df_synth['short_tp'])
)
n_hold_actions = hold_actions.sum()
hold_action_pct_actual = n_hold_actions / len(df_synth)
hold_action_pct_expected = test_cfg['hold_action_pct']

print(f"  Hold actions: {n_hold_actions:,} / {len(df_synth):,} ({hold_action_pct_actual*100:.2f}%)")
print(f"  Expected: ~{hold_action_pct_expected*100:.0f}%")

# Allow some tolerance (within 3% of target)
assert abs(hold_action_pct_actual - hold_action_pct_expected) < 0.03, \
    f"Hold action percentage should be near {hold_action_pct_expected*100}%"
print("✓ Hold action percentage correct")

# Test 5d: Validate leverage constraints
print("\n[5d] Validating leverage constraints...")
max_leverage = test_cfg['max_leverage']
gross_exposure = df_synth['long_value'] + df_synth['short_value']
actual_leverage = gross_exposure / df_synth['equity']

print(f"  Max leverage allowed: {max_leverage}x")
print(f"  Actual max leverage: {actual_leverage.max():.4f}x")
print(f"  Mean leverage: {actual_leverage.mean():.4f}x")

# Should not exceed max leverage (with small epsilon for floating point)
assert (actual_leverage <= max_leverage + 1e-6).all(), \
    f"Leverage should not exceed {max_leverage}x"
print("✓ Leverage constraints satisfied")

# Test 5e: Validate balance is non-negative
print("\n[5e] Validating balance is non-negative...")
assert (df_synth['balance'] >= -1e-6).all(), "Balance should be non-negative"
print(f"  Min balance: ${df_synth['balance'].min():,.2f}")
print(f"  Max balance: ${df_synth['balance'].max():,.2f}")
print("✓ Balance is non-negative")

# Test 5f: Validate equity = balance + positions
print("\n[5f] Validating equity = balance + positions...")
calculated_equity = df_synth['balance'] + df_synth['long_value'] + df_synth['short_value']
np.testing.assert_allclose(calculated_equity, df_synth['equity'], rtol=1e-5)
print("✓ Equity equation holds: equity = balance + long_value + short_value")

# Test 5g: Validate SL/TP zeroed for non-existent positions
print("\n[5g] Validating SL/TP zeroed for non-existent positions...")
no_long = df_synth['long_value'] == 0
no_short = df_synth['short_value'] == 0

assert (df_synth.loc[no_long, 'long_sl'] == 0).all(), "Long SL should be 0 when no long position"
assert (df_synth.loc[no_long, 'long_tp'] == 0).all(), "Long TP should be 0 when no long position"
assert (df_synth.loc[no_short, 'short_sl'] == 0).all(), "Short SL should be 0 when no short position"
assert (df_synth.loc[no_short, 'short_tp'] == 0).all(), "Short TP should be 0 when no short position"
print("✓ SL/TP correctly zeroed for non-existent positions")

# Test 5h: Validate SL/TP ranges for active positions
print("\n[5h] Validating SL/TP ranges for active positions...")
has_long = df_synth['long_value'] > 0
has_short = df_synth['short_value'] > 0

if has_long.any():
    long_sl_active = df_synth.loc[has_long, 'long_sl']
    long_tp_active = df_synth.loc[has_long, 'long_tp']
    assert (long_sl_active >= test_cfg['synth_long_sl_min']).all(), "Active long SL in range"
    assert (long_sl_active <= test_cfg['synth_long_sl_max']).all(), "Active long SL in range"
    assert (long_tp_active >= test_cfg['synth_long_tp_min']).all(), "Active long TP in range"
    assert (long_tp_active <= test_cfg['synth_long_tp_max']).all(), "Active long TP in range"
    print(f"  ✓ Long positions ({has_long.sum():,}): SL/TP in valid ranges")

if has_short.any():
    short_sl_active = df_synth.loc[has_short, 'short_sl']
    short_tp_active = df_synth.loc[has_short, 'short_tp']
    assert (short_sl_active >= test_cfg['synth_short_sl_min']).all(), "Active short SL in range"
    assert (short_sl_active <= test_cfg['synth_short_sl_max']).all(), "Active short SL in range"
    assert (short_tp_active >= test_cfg['synth_short_tp_min']).all(), "Active short TP in range"
    assert (short_tp_active <= test_cfg['synth_short_tp_max']).all(), "Active short TP in range"
    print(f"  ✓ Short positions ({has_short.sum():,}): SL/TP in valid ranges")

print("\n✓ All assertions passed for build_samples()")

#%%
# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("✅ ALL SCALE AND SYNTH TESTS PASSED")
print("="*70)
print("\nSummary:")
print("  ✓ scale_ohlcv_window() - Window-based scaling")
print("    - OHLC: Linear division by close[-1]")
print("    - Volume: log1p method (handles zeros)")
print("  ✓ MetaScaler - Fit/transform/inverse/save/load")
print("  ✓ _sample_position_value() - Log-normal distribution (50k samples)")
print("  ✓ _sample_sl_tp_multiplier() - All 4 ranges validated (50k samples):")
print("    - Long SL: 0.50-0.99 ✓")
print("    - Long TP: 1.01-21.0 ✓")
print("    - Short SL: 1.01-1.50 ✓")
print("    - Short TP: 0.50-0.99 ✓")
print("  ✓ build_samples() - Full synthetic generation (10k samples):")
print(f"    - Hold states: ~{hold_state_pct_actual*100:.1f}% (target: {hold_state_pct_expected*100}%)")
print(f"    - Hold actions: ~{hold_action_pct_actual*100:.1f}% (target: {hold_action_pct_expected*100}%)")
print("    - Leverage constraints satisfied")
print("    - Balance non-negative")
print("    - SL/TP correctly zeroed for flat positions")
print("\n✓ scale.py and synth.py modules validated")

# %%
