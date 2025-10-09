#%%
# imports (including data.py and dataset.py to make test data, and synth.py and scale.py to be tested)
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src import data, synth, scale

# Load config
config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

print("✓ Imports successful")
print(f"Config loaded: {config_path}")
print(f"Testing modules: synth.py and scale.py")

#%%
# Override Config with example values for testing
test_cfg = cfg.copy()
test_cfg.update({
    "ticker": "^GSPC",
    "start": "2020-01-01",
    "lookback": 50,
    "forward": 20,
    "synth_equity_min": 10000,
    "synth_equity_max": 100000,
    "synth_balance_offset_min": -5000,
    "synth_balance_offset_max": 5000,
    "synth_position_values": [0.0, 1.0, -1.0],
    "synth_sl_min": 0.001,
    "synth_sl_max": 0.05,
    "synth_tp_min": 0.001,
    "synth_tp_max": 0.10,
    "synth_dollar_min": 1000,
    "synth_dollar_max": 50000,
    "scale_meta": "minmax",
    "epsilon": 1e-8,
})

print("\n✓ Test config created")
print(f"  - Lookback: {test_cfg['lookback']}, Forward: {test_cfg['forward']}")

#%%
# =============================================================================
# TEST 1: scale.scale_ohlcv_window() - Validate OHLCV scaling logic
# =============================================================================
print("\n" + "="*70)
print("TEST 1: scale.scale_ohlcv_window() - Validate OHLCV scaling logic")
print("="*70)

# Test 1a: Basic scaling with known values
print("\n[1a] Testing OHLCV scaling with mock data (last close = 100)...")

# Create simple mock OHLCV data where last close = 100
mock_ohlcv = {
    "open": np.array([90.0, 95.0, 98.0, 99.0, 100.0]),
    "high": np.array([95.0, 100.0, 102.0, 103.0, 105.0]),
    "low": np.array([88.0, 93.0, 96.0, 97.0, 98.0]),
    "close": np.array([92.0, 97.0, 99.0, 101.0, 100.0]),  # Last close = 100
    "volume": np.array([1000.0, 1500.0, 2000.0, 2500.0, 3000.0])
}

scaled_mock = scale.scale_ohlcv_window(mock_ohlcv)

# Expected: All values divided by last close (100.0)
last_close = mock_ohlcv["close"][-1]
print(f"  - Last close value: {last_close}")

# Validate OHLC scaling (divide by last close)
expected_open = mock_ohlcv["open"] / last_close
expected_high = mock_ohlcv["high"] / last_close
expected_low = mock_ohlcv["low"] / last_close
expected_close = mock_ohlcv["close"] / last_close

np.testing.assert_array_almost_equal(scaled_mock["open"], expected_open, decimal=6, 
                                     err_msg="Open should be divided by last close")
np.testing.assert_array_almost_equal(scaled_mock["high"], expected_high, decimal=6,
                                     err_msg="High should be divided by last close")
np.testing.assert_array_almost_equal(scaled_mock["low"], expected_low, decimal=6,
                                     err_msg="Low should be divided by last close")
np.testing.assert_array_almost_equal(scaled_mock["close"], expected_close, decimal=6,
                                     err_msg="Close should be divided by last close")

print(f"  ✓ OHLC scaled correctly (divided by last close)")
print(f"    Example: close[0]={mock_ohlcv['close'][0]:.1f} -> scaled={scaled_mock['close'][0]:.4f} (expected {expected_close[0]:.4f})")

# Validate Volume scaling (log1p transformation divided by log1p of last volume)
vol_last = mock_ohlcv["volume"][-1]
expected_volume = np.log1p(mock_ohlcv["volume"]) / np.log1p(vol_last)
np.testing.assert_array_almost_equal(scaled_mock["volume"], expected_volume, decimal=6,
                                     err_msg="Volume should be log1p(volume) / log1p(last_volume)")
print(f"  ✓ Volume scaled correctly (log1p transformation)")
print(f"    Example: volume[0]={mock_ohlcv['volume'][0]:.1f} -> scaled={scaled_mock['volume'][0]:.4f} (expected {expected_volume[0]:.4f})")
print(f"    Last volume scaled to: {scaled_mock['volume'][-1]:.6f} (should be 1.0)")

# Test 1b: Validate last close and last volume become 1.0
print("\n[1b] Validating last close and last volume become 1.0 after scaling...")
assert abs(scaled_mock["close"][-1] - 1.0) < 1e-6, "Last close should be exactly 1.0 after scaling"
assert abs(scaled_mock["volume"][-1] - 1.0) < 1e-6, "Last volume should be exactly 1.0 after scaling"
print(f"  ✓ Last close scaled to: {scaled_mock['close'][-1]:.6f} (expected: 1.000000)")
print(f"  ✓ Last volume scaled to: {scaled_mock['volume'][-1]:.6f} (expected: 1.000000)")

# Test 1c: Test with different last close and volume values
print("\n[1c] Testing with different last close and volume values...")
test_last_closes = [50.0, 200.0, 1000.0, 0.5]
for last_val in test_last_closes:
    test_data = {
        "open": np.array([last_val * 0.95, last_val * 0.98, last_val]),
        "high": np.array([last_val * 1.02, last_val * 1.05, last_val * 1.03]),
        "low": np.array([last_val * 0.90, last_val * 0.93, last_val * 0.97]),
        "close": np.array([last_val * 0.96, last_val * 0.99, last_val]),
        "volume": np.array([1000.0, 1500.0, 2000.0])
    }
    scaled_test = scale.scale_ohlcv_window(test_data)
    assert abs(scaled_test["close"][-1] - 1.0) < 1e-6, f"Last close should be 1.0 (was {scaled_test['close'][-1]})"
    assert abs(scaled_test["volume"][-1] - 1.0) < 1e-6, f"Last volume should be 1.0 (was {scaled_test['volume'][-1]})"
    print(f"  ✓ Last close {last_val} -> scaled to {scaled_test['close'][-1]:.6f}, volume -> {scaled_test['volume'][-1]:.6f}")

print("\n✓ All assertions passed for scale_ohlcv_window()")

#%%
# =============================================================================
# TEST 2: scale.MetaScaler - Validate MinMax scaling
# =============================================================================
print("\n" + "="*70)
print("TEST 2: scale.MetaScaler - Validate MinMax scaling")
print("="*70)

# Test 2a: Basic MinMax scaling with known ranges
print("\n[2a] Testing MinMax scaling with mock data...")

# Create mock meta data with known min/max
mock_meta_df = pd.DataFrame({
    "equity": [10000, 15000, 20000, 25000, 30000],
    "balance": [8000, 12000, 16000, 20000, 24000],
    "position": [-1.0, 0.0, 1.0, -1.0, 1.0],
    "sl_dist": [0.01, 0.02, 0.03, 0.04, 0.05],
    "tp_dist": [0.02, 0.04, 0.06, 0.08, 0.10],
    "act_dollar": [1000, 2000, 3000, 4000, 5000],
    "act_sl": [0.01, 0.015, 0.02, 0.025, 0.03],
    "act_tp": [0.02, 0.03, 0.04, 0.05, 0.06],
})

meta_cols = ["equity", "balance", "position", "sl_dist", "tp_dist", "act_dollar", "act_sl", "act_tp"]
scaler = scale.MetaScaler(kind="minmax")
scaler.fit(mock_meta_df, meta_cols)

# Keep original values for comparison
original_values = {col: mock_meta_df[col].values.copy() for col in meta_cols}

# Transform modifies columns in-place
scaled_meta_df = scaler.transform(mock_meta_df, meta_cols, epsilon=test_cfg["epsilon"])

# Validate MinMax scaling: (x - min) / (max - min)
for col in meta_cols:
    original = original_values[col]  # Use saved original values
    scaled = scaled_meta_df[col].values  # Transformed values are in same column
    
    col_min = original.min()
    col_max = original.max()
    col_range = col_max - col_min
    
    if col_range > test_cfg["epsilon"]:
        expected_scaled = (original - col_min) / col_range
    else:
        expected_scaled = np.zeros_like(original)
    
    np.testing.assert_array_almost_equal(scaled, expected_scaled, decimal=6,
                                         err_msg=f"{col} should be MinMax scaled: (x - min) / (max - min)")
    
    print(f"  ✓ {col}: min={col_min:.2f}, max={col_max:.2f}, scaled_range=[{scaled.min():.4f}, {scaled.max():.4f}]")
    assert scaled.min() >= 0.0, f"{col} should have min >= 0"
    assert scaled.max() <= 1.0, f"{col} should have max <= 1"

print(f"\n✓ Meta columns MinMax scaled correctly (all values in [0, 1])")

# Test 2b: Test scaler save/load
print("\n[2b] Testing scaler save and load...")
test_scaler_path = Path("data") / "test_scaler_02.json"
test_scaler_path.parent.mkdir(exist_ok=True, parents=True)

scaler.save(test_scaler_path)
print(f"  ✓ Saved scaler to {test_scaler_path}")

# Load into a new scaler instance
scaler_loaded = scale.MetaScaler(kind="minmax")
scaler_loaded.load(test_scaler_path)

# Create fresh dataframe for reloaded scaler (since transform modifies in-place)
mock_meta_df_reload = pd.DataFrame({
    "equity": [10000, 15000, 20000, 25000, 30000],
    "balance": [8000, 12000, 16000, 20000, 24000],
    "position": [-1.0, 0.0, 1.0, -1.0, 1.0],
    "sl_dist": [0.01, 0.02, 0.03, 0.04, 0.05],
    "tp_dist": [0.02, 0.04, 0.06, 0.08, 0.10],
    "act_dollar": [1000, 2000, 3000, 4000, 5000],
    "act_sl": [0.01, 0.015, 0.02, 0.025, 0.03],
    "act_tp": [0.02, 0.03, 0.04, 0.05, 0.06],
})
scaled_meta_df_reloaded = scaler_loaded.transform(mock_meta_df_reload, meta_cols, epsilon=test_cfg["epsilon"])

# Validate loaded scaler produces same results
for col in meta_cols:
    np.testing.assert_array_almost_equal(
        scaled_meta_df[col].values,
        scaled_meta_df_reloaded[col].values,
        decimal=6,
        err_msg=f"Loaded scaler should produce same results for {col}"
    )
print(f"  ✓ Loaded scaler produces identical results")

test_scaler_path.unlink()
print(f"  ✓ Cleaned up test file")

# Test 2c: Test with edge cases (constant values)
print("\n[2c] Testing with constant/near-constant values...")
edge_case_df = pd.DataFrame({
    "equity": [10000, 10000, 10000, 10000, 10000],  # All same
    "balance": [8000, 8000.01, 8000, 8000, 8000],  # Nearly same
})
edge_cols = ["equity", "balance"]
scaler_edge = scale.MetaScaler(kind="minmax")
scaler_edge.fit(edge_case_df, edge_cols)
scaled_edge = scaler_edge.transform(edge_case_df, edge_cols, epsilon=test_cfg["epsilon"])

# Constant values should scale to 0 (or stay within bounds with epsilon)
assert scaled_edge["equity"].max() <= 1.0, "Constant equity should scale properly"
assert scaled_edge["balance"].max() <= 1.0, "Near-constant balance should scale properly"
print(f"  ✓ Edge cases handled correctly (constant values: equity={scaled_edge['equity'].values}, balance={scaled_edge['balance'].max():.6f})")

print("\n✓ All assertions passed for MetaScaler")

#%%
# =============================================================================
# TEST 3: synth.build_samples() - Generate synthetic samples
# =============================================================================
print("\n" + "="*70)
print("TEST 3: synth.build_samples() - Generate synthetic samples")
print("="*70)

# First, get some real data to sample from
print("\n[3a] Downloading data for synthetic sampling...")
df_data = data.download(ticker=test_cfg["ticker"], start=test_cfg["start"])
print(f"  ✓ Downloaded {len(df_data)} rows")

# Test 3b: Basic sample generation
print("\n[3b] Generating synthetic samples...")
rng = np.random.default_rng(42)
n_samples = 10

samples = synth.build_samples(
    df_data, 
    n=n_samples, 
    lookback=test_cfg["lookback"], 
    forward=test_cfg["forward"], 
    rng=rng, 
    cfg=test_cfg
)

print(f"✓ Generated {len(samples)} samples")
print(f"  - Columns: {list(samples.columns)}")

# Validate structure
assert len(samples) == n_samples, f"Should have {n_samples} samples"
assert "idx" in samples.columns, "Should have idx column"

# Validate OHLCV columns
ohlcv_cols = ["open", "high", "low", "close", "volume"]
for col in ohlcv_cols:
    assert col in samples.columns, f"Missing {col} column"
    # Check that each sample has an array of correct length
    assert len(samples[col].iloc[0]) == test_cfg["lookback"], f"{col} arrays should have length {test_cfg['lookback']}"

# Validate state columns
state_cols = ["equity", "balance", "position", "sl_dist", "tp_dist"]
for col in state_cols:
    assert col in samples.columns, f"Missing state column {col}"

# Validate action columns
action_cols = ["act_dir", "act_dollar", "act_sl", "act_tp"]
for col in action_cols:
    assert col in samples.columns, f"Missing action column {col}"

print(f"  ✓ All required columns present")

# Test 3c: Validate action directions
print("\n[3c] Validating action directions...")
valid_directions = ["hold", "long", "short"]
assert samples["act_dir"].isin(valid_directions).all(), f"Action directions should be in {valid_directions}"
dir_counts = samples["act_dir"].value_counts()
print(f"  - Action distribution: {dir_counts.to_dict()}")

# Test 3d: Validate numeric ranges
print("\n[3d] Validating numeric ranges...")
assert (samples["equity"] >= test_cfg["synth_equity_min"]).all(), "Equity should be >= min"
assert (samples["equity"] <= test_cfg["synth_equity_max"]).all(), "Equity should be <= max"
assert (samples["position"].isin(test_cfg["synth_position_values"])).all(), "Position should match allowed values"
assert (samples["sl_dist"] >= test_cfg["synth_sl_min"]).all(), "SL distance should be >= min"
assert (samples["sl_dist"] <= test_cfg["synth_sl_max"]).all(), "SL distance should be <= max"
print(f"  ✓ All numeric values within expected ranges")

# Test 3e: Validate index boundaries
print("\n[3e] Validating sample indices...")
min_valid_idx = test_cfg["lookback"]
max_valid_idx = len(df_data) - test_cfg["lookback"] - test_cfg["forward"]
assert (samples["idx"] >= min_valid_idx).all(), f"Indices should be >= {min_valid_idx}"
assert (samples["idx"] < max_valid_idx).all(), f"Indices should be < {max_valid_idx}"
print(f"  ✓ All indices within valid bounds [{min_valid_idx}, {max_valid_idx})")

# Test 3f: Test reproducibility with same seed
print("\n[3f] Testing reproducibility...")
rng2 = np.random.default_rng(42)  # Same seed
samples2 = synth.build_samples(df_data, n=n_samples, lookback=test_cfg["lookback"], 
                               forward=test_cfg["forward"], rng=rng2, cfg=test_cfg)
assert (samples["idx"].values == samples2["idx"].values).all(), "Same seed should produce same indices"
print(f"  ✓ Same seed produces reproducible results")

# Test 3g: Different seed produces different results
rng3 = np.random.default_rng(999)  # Different seed
samples3 = synth.build_samples(df_data, n=n_samples, lookback=test_cfg["lookback"], 
                               forward=test_cfg["forward"], rng=rng3, cfg=test_cfg)
assert (samples["idx"].values != samples3["idx"].values).any(), "Different seed should produce different samples"
print(f"  ✓ Different seed produces different results")

print("\n✓ All assertions passed for build_samples()")

#%%
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("ALL TESTS PASSED! ✓")
print("="*70)
print("\nSummary of synth.py and scale.py tests:")
print("  ✓ TEST 1: scale.scale_ohlcv_window() - OHLCV scaling (÷ last close)")
print("  ✓ TEST 2: scale.MetaScaler - MinMax scaling, save/load, edge cases")
print("  ✓ TEST 3: synth.build_samples() - Synthetic data generation, validation")
print("\n" + "="*70)
print("SCALE AND SYNTH FUNCTIONS VERIFIED")
print("These functions can be relied upon for subsequent tests!")
print("="*70)
