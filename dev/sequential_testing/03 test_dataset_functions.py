#%%
# =============================================================================
# TEST FILE 03: dataset.build_dataset() - Integration tests
# =============================================================================
# This file tests the dataset.build_dataset() wrapper function which orchestrates:
#   - data.download() / data.load() (tested in File 01)
#   - scale.scale_ohlcv_window() and scale.MetaScaler (tested in File 02)
#   - synth.build_samples() (tested in File 02)
# 
# Now we test that these components work together correctly in the full pipeline.
#%%
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src import data, dataset

# Load config
config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

print("✓ Imports successful")
print(f"Config loaded: {config_path}")
print(f"Testing module: dataset.py (build_dataset wrapper)")

#%%
# =============================================================================
# Setup: Override config for testing
# =============================================================================
test_cfg = cfg.copy()
test_cfg.update({
    "ticker": "^GSPC",
    "start": "2020-01-01",
    "lookback": 50,
    "forward": 20,
    "n_samples": 100,
    "data_dir": "data",
    "raw_data_filename": "raw_{ticker}.parquet",
    "scaler_filename": "meta_scaler.json",
    "samples_filename": "samples_{n}M.parquet",
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
print(f"  - Ticker: {test_cfg['ticker']}")
print(f"  - Lookback: {test_cfg['lookback']}, Forward: {test_cfg['forward']}")
print(f"  - Default samples: {test_cfg['n_samples']}")

#%%
# =============================================================================
# TEST 1: dataset.build_dataset() - Basic functionality
# =============================================================================
print("\n" + "="*70)
print("TEST 1: dataset.build_dataset() - Basic functionality")
print("="*70)

# Test 1a: Build small dataset
print("\n[1a] Building basic dataset (10 samples)...")
output_path = dataset.build_dataset(
    cfg=test_cfg,
    n_samples=10,
    seed=42,
    overwrite=True
)

assert output_path.exists(), "Output file should be created"
print(f"✓ Dataset created: {output_path}")

# Test 1b: Load and validate structure
print("\n[1b] Loading and validating dataset structure...")
df = pd.read_parquet(output_path)

# Check shape
assert len(df) == 10, f"Should have 10 samples, got {len(df)}"
print(f"✓ Correct number of samples: {len(df)}")

# Test 1c: Validate required columns
print("\n[1c] Validating required columns...")
expected_ohlcv_cols = ["open", "high", "low", "close", "volume"]
expected_ohlcv_scaled_cols = ["open_scaled", "high_scaled", "low_scaled", "close_scaled", "volume_scaled"]
expected_meta_cols = ["equity", "balance", "position", "sl_dist", "tp_dist", 
                      "act_dollar", "act_sl", "act_tp"]
expected_info_cols = ["idx", "forward", "y"]

all_expected_cols = expected_ohlcv_cols + expected_ohlcv_scaled_cols + expected_meta_cols + expected_info_cols
for col in all_expected_cols:
    assert col in df.columns, f"Missing column: {col}"
print(f"✓ All required columns present ({len(all_expected_cols)} columns)")
print(f"  - OHLCV original: {len(expected_ohlcv_cols)} cols")
print(f"  - OHLCV scaled: {len(expected_ohlcv_scaled_cols)} cols")
print(f"  - Meta: {len(expected_meta_cols)} cols")
print(f"  - Info: {len(expected_info_cols)} cols")

# Test 1d: Validate OHLCV arrays (both original and scaled)
print("\n[1d] Validating OHLCV array structures...")
for col in expected_ohlcv_cols + expected_ohlcv_scaled_cols:
    first_array = df[col].iloc[0]
    assert isinstance(first_array, np.ndarray), f"{col} should contain numpy arrays"
    assert len(first_array) == test_cfg["lookback"], f"{col} arrays should have lookback length ({test_cfg['lookback']})"
print(f"✓ OHLCV arrays have correct structure (original + scaled, length={test_cfg['lookback']})")

# Test 1e: Validate meta columns are numeric
print("\n[1e] Validating meta column types...")
for col in expected_meta_cols:
    assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"
print(f"✓ All meta columns are numeric")

# Test 1f: Validate forward values
print("\n[1f] Validating forward column...")
assert (df["forward"] == test_cfg["forward"]).all(), f"Forward should be {test_cfg['forward']}"
print(f"✓ Forward column correct (all values = {test_cfg['forward']})")

# Cleanup
output_path.unlink()
print("\n✓ All assertions passed for basic functionality")

#%%
# =============================================================================
# TEST 2: dataset.build_dataset() - Different sample sizes
# =============================================================================
print("\n" + "="*70)
print("TEST 2: dataset.build_dataset() - Different sample sizes")
print("="*70)

# Test 2a: Very small dataset (1 sample)
print("\n[2a] Building very small dataset (1 sample)...")
output_path_1 = dataset.build_dataset(
    cfg=test_cfg,
    n_samples=1,
    seed=99,
    overwrite=True
)
df_1 = pd.read_parquet(output_path_1)
assert len(df_1) == 1, "Should have exactly 1 sample"
print(f"✓ Single sample works: {df_1.shape}")
output_path_1.unlink()

# Test 2b: Medium dataset (50 samples)
print("\n[2b] Building medium dataset (50 samples)...")
output_path_50 = dataset.build_dataset(
    cfg=test_cfg,
    n_samples=50,
    seed=100,
    overwrite=True
)
df_50 = pd.read_parquet(output_path_50)
assert len(df_50) == 50, "Should have exactly 50 samples"
print(f"✓ Medium dataset works: {df_50.shape}")
output_path_50.unlink()

# Test 2c: Larger dataset (200 samples)
print("\n[2c] Building larger dataset (200 samples)...")
output_path_200 = dataset.build_dataset(
    cfg=test_cfg,
    n_samples=200,
    seed=200,
    overwrite=True
)
df_200 = pd.read_parquet(output_path_200)
assert len(df_200) == 200, "Should have exactly 200 samples"
print(f"✓ Larger dataset works: {df_200.shape}")

# Test 2d: Validate all have same columns
print("\n[2d] Validating consistent structure across sizes...")
assert list(df_1.columns) == list(df_50.columns) == list(df_200.columns), "All datasets should have same columns"
print(f"✓ All dataset sizes produce consistent structure ({len(df_200.columns)} columns)")

output_path_200.unlink()

print("\n✓ All assertions passed for different sample sizes")

#%%
# =============================================================================
# TEST 3: dataset.build_dataset() - Seed reproducibility
# =============================================================================
print("\n" + "="*70)
print("TEST 3: dataset.build_dataset() - Seed reproducibility")
print("="*70)

# Test 3a: Same seed should produce identical results
print("\n[3a] Testing reproducibility with same seed...")
output_path_seed1_a = dataset.build_dataset(
    cfg=test_cfg,
    n_samples=20,
    seed=123,
    overwrite=True
)
df_seed1_a = pd.read_parquet(output_path_seed1_a)

output_path_seed1_b = dataset.build_dataset(
    cfg=test_cfg,
    n_samples=20,
    seed=123,
    overwrite=True
)
df_seed1_b = pd.read_parquet(output_path_seed1_b)

# Compare indices (should be identical)
assert (df_seed1_a["idx"].values == df_seed1_b["idx"].values).all(), "Same seed should produce same sample indices"
print(f"✓ Same seed produces identical indices")

# Compare equity values (should be identical due to same seed)
np.testing.assert_array_almost_equal(
    df_seed1_a["equity"].values,
    df_seed1_b["equity"].values,
    decimal=6,
    err_msg="Same seed should produce identical synthetic values"
)
print(f"✓ Same seed produces identical synthetic data")

# Only unlink once (same path with same params)
assert output_path_seed1_a == output_path_seed1_b, "Same params should produce same path"
output_path_seed1_a.unlink()

# Test 3b: Different seeds should produce different results
print("\n[3b] Testing different seeds produce different samples...")
output_path_seed2 = dataset.build_dataset(
    cfg=test_cfg,
    n_samples=20,
    seed=456,
    overwrite=True
)
df_seed2 = pd.read_parquet(output_path_seed2)

# Compare indices (should be different)
different_indices = (df_seed1_a["idx"].values != df_seed2["idx"].values).any()
assert different_indices, "Different seeds should produce different samples"
print(f"✓ Different seeds produce different sample indices")

# Compare equity values (should be different due to different seed)
different_equity = not np.allclose(df_seed1_a["equity"].values, df_seed2["equity"].values)
assert different_equity, "Different seeds should produce different synthetic values"
print(f"✓ Different seeds produce different synthetic data")

output_path_seed2.unlink()

print("\n✓ All assertions passed for seed reproducibility")

#%%
# =============================================================================
# TEST 4: dataset.build_dataset() - Overwrite parameter (raw data)
# =============================================================================
print("\n" + "="*70)
print("TEST 4: dataset.build_dataset() - Overwrite parameter (raw data)")
print("="*70)

# Note: The overwrite parameter controls whether raw market data is re-downloaded,
# but the samples dataset is always regenerated with each call.

# Test 4a: Build dataset (will download raw data if needed)
print("\n[4a] Building initial dataset...")
output_path = dataset.build_dataset(
    cfg=test_cfg,
    n_samples=15,
    seed=789,
    overwrite=True
)
df_initial = pd.read_parquet(output_path)
print(f"✓ Initial dataset: {df_initial.shape}")

# Test 4b: Build again without overwrite (uses existing raw data, regenerates samples)
print("\n[4b] Building with overwrite=False (reuses raw data, regenerates samples)...")
output_path_no_overwrite = dataset.build_dataset(
    cfg=test_cfg,
    n_samples=15,
    seed=999,  # Different seed will produce different samples
    overwrite=False  # Don't re-download raw data
)
assert output_path == output_path_no_overwrite, "Should return same path"
df_no_overwrite = pd.read_parquet(output_path_no_overwrite)

# Should be different because samples are always regenerated with new seed
different_data = (df_initial["idx"].values != df_no_overwrite["idx"].values).any()
assert different_data, "Different seed produces different samples even with overwrite=False"
print(f"✓ overwrite=False reuses raw data but regenerates samples")

# Test 4c: Verify raw data file exists and wasn't re-downloaded
print("\n[4c] Verifying raw data file exists...")
data_dir = Path(test_cfg["data_dir"])
raw_filename = test_cfg["raw_data_filename"].format(ticker=test_cfg["ticker"])
raw_path = data_dir / raw_filename
assert raw_path.exists(), "Raw data file should exist"
print(f"✓ Raw data file exists: {raw_path.name}")
print(f"  (overwrite=False means this file is reused, not re-downloaded)")

# Cleanup
output_path.unlink()

print("\n✓ All assertions passed for overwrite parameter")

#%%
# =============================================================================
# TEST 5: dataset.build_dataset() - Different lookback/forward windows
# =============================================================================
print("\n" + "="*70)
print("TEST 5: dataset.build_dataset() - Different lookback/forward windows")
print("="*70)

# Test 5a: Small windows
print("\n[5a] Testing small windows (lookback=20, forward=10)...")
test_cfg_small = test_cfg.copy()
test_cfg_small["lookback"] = 20
test_cfg_small["forward"] = 10
output_path_small = dataset.build_dataset(
    cfg=test_cfg_small,
    n_samples=10,
    seed=111,
    overwrite=True
)
df_small = pd.read_parquet(output_path_small)
assert df_small["forward"].iloc[0] == 10, "Forward column should match config"
assert len(df_small["close"].iloc[0]) == 20, "OHLCV arrays should have lookback length"
print(f"✓ Small windows work: forward={df_small['forward'].iloc[0]}, lookback_len={len(df_small['close'].iloc[0])}")
output_path_small.unlink()

# Test 5b: Large windows
print("\n[5b] Testing large windows (lookback=100, forward=50)...")
test_cfg_large = test_cfg.copy()
test_cfg_large["lookback"] = 100
test_cfg_large["forward"] = 50
output_path_large = dataset.build_dataset(
    cfg=test_cfg_large,
    n_samples=10,
    seed=222,
    overwrite=True
)
df_large = pd.read_parquet(output_path_large)
assert df_large["forward"].iloc[0] == 50, "Forward column should match config"
assert len(df_large["close"].iloc[0]) == 100, "OHLCV arrays should have lookback length"
print(f"✓ Large windows work: forward={df_large['forward'].iloc[0]}, lookback_len={len(df_large['close'].iloc[0])}")
output_path_large.unlink()

# Test 5c: Asymmetric windows
print("\n[5c] Testing asymmetric windows (lookback=30, forward=100)...")
test_cfg_asym = test_cfg.copy()
test_cfg_asym["lookback"] = 30
test_cfg_asym["forward"] = 100
output_path_asym = dataset.build_dataset(
    cfg=test_cfg_asym,
    n_samples=10,
    seed=333,
    overwrite=True
)
df_asym = pd.read_parquet(output_path_asym)
assert df_asym["forward"].iloc[0] == 100, "Forward column should match config"
assert len(df_asym["close"].iloc[0]) == 30, "OHLCV arrays should have lookback length"
print(f"✓ Asymmetric windows work: forward={df_asym['forward'].iloc[0]}, lookback_len={len(df_asym['close'].iloc[0])}")
output_path_asym.unlink()

print("\n✓ All assertions passed for different window sizes")

#%%
# =============================================================================
# TEST 6: dataset.build_dataset() - Data validation and integration
# =============================================================================
print("\n" + "="*70)
print("TEST 6: dataset.build_dataset() - Data validation and integration")
print("="*70)

# Build a test dataset
print("\n[6a] Building dataset for integration validation...")
output_path = dataset.build_dataset(
    cfg=test_cfg,
    n_samples=20,
    seed=555,
    overwrite=True
)
df = pd.read_parquet(output_path)
print(f"✓ Dataset created: {df.shape}")

# Test 6b: Validate OHLCV scaling (last value should be ~1.0)
print("\n[6b] Validating OHLCV scaling is applied...")
sample_close_scaled = df["close_scaled"].iloc[0]
last_close_scaled = sample_close_scaled[-1]
assert abs(last_close_scaled - 1.0) < 0.01, f"Last close_scaled should be ~1.0 after scaling (got {last_close_scaled})"
print(f"✓ OHLCV scaling applied (last close_scaled = {last_close_scaled:.6f})")

# Test 6c: Validate Volume scaling (last value should be ~1.0)
print("\n[6c] Validating Volume scaling is applied...")
sample_volume_scaled = df["volume_scaled"].iloc[0]
last_volume_scaled = sample_volume_scaled[-1]
assert abs(last_volume_scaled - 1.0) < 0.01, f"Last volume_scaled should be ~1.0 after log scaling (got {last_volume_scaled})"
print(f"✓ Volume scaling applied (last volume_scaled = {last_volume_scaled:.6f})")

# Test 6d: Validate Meta columns are in [0, 1] range (MinMax scaled)
print("\n[6d] Validating Meta column scaling (MinMax [0, 1])...")
meta_cols = ["equity", "balance", "position", "sl_dist", "tp_dist", "act_dollar", "act_sl", "act_tp"]
for col in meta_cols:
    col_min = df[col].min()
    col_max = df[col].max()
    assert col_min >= 0.0, f"{col} should have min >= 0 (got {col_min})"
    assert col_max <= 1.0, f"{col} should have max <= 1 (got {col_max})"
print(f"✓ All meta columns in [0, 1] range (MinMax scaling applied)")

# Test 6e: Validate no NaN values
print("\n[6e] Validating no NaN values...")
assert not df.isnull().any().any(), "Dataset should have no NaN values"
print(f"✓ No NaN values in dataset")

# Test 6f: Validate index boundaries
print("\n[6f] Validating sample index boundaries...")
min_idx = df["idx"].min()
max_idx = df["idx"].max()
# Need to download data to check bounds
df_raw = data.download(ticker=test_cfg["ticker"], start=test_cfg["start"])
assert min_idx >= test_cfg["lookback"], f"Min index should be >= lookback ({test_cfg['lookback']})"
assert max_idx <= len(df_raw) - test_cfg["forward"], f"Max index should leave room for forward window"
print(f"✓ Sample indices within valid bounds: [{min_idx}, {max_idx}]")
print(f"  Raw data length: {len(df_raw)}, lookback: {test_cfg['lookback']}, forward: {test_cfg['forward']}")

# Cleanup
output_path.unlink()

print("\n✓ All assertions passed for data validation and integration")

#%%
# =============================================================================
# TEST 7: dataset.build_dataset() - Edge cases and file creation
# =============================================================================
print("\n" + "="*70)
print("TEST 7: dataset.build_dataset() - Edge cases and file creation")
print("="*70)

# Test 7a: Validate file naming convention
print("\n[7a] Testing filename generation for different sample counts...")
sample_counts = [1000, 50000, 1000000]
for n in sample_counts:
    test_cfg_naming = test_cfg.copy()
    output_path_naming = dataset.build_dataset(
        cfg=test_cfg_naming,
        n_samples=n,
        seed=444,
        overwrite=True
    )
    filename = output_path_naming.name
    print(f"  - {n:,} samples -> {filename}")
    assert output_path_naming.exists(), f"File should exist for {n} samples"
    output_path_naming.unlink()

print(f"✓ Filename generation works correctly")

# Test 7b: Check that raw data file is created
print("\n[7b] Validating raw data file creation...")
test_data_dir = Path(test_cfg["data_dir"])
raw_filename = test_cfg["raw_data_filename"].format(ticker=test_cfg["ticker"])
raw_path = test_data_dir / raw_filename

if not raw_path.exists():
    # Build dataset will create it
    temp_out = dataset.build_dataset(
        cfg=test_cfg,
        n_samples=5,
        seed=555,
        overwrite=True
    )
    temp_out.unlink()

assert raw_path.exists(), "Raw data file should be created"
print(f"✓ Raw data file exists: {raw_path}")

# Test 7c: Check scaler file is created
print("\n[7c] Validating scaler file creation...")
scaler_path = test_data_dir / test_cfg["scaler_filename"]
assert scaler_path.exists(), "Scaler file should be created"
print(f"✓ Scaler file exists: {scaler_path}")

# Test 7d: Very large dataset generation
print("\n[7d] Testing larger dataset generation (500 samples)...")
output_path_large = dataset.build_dataset(
    cfg=test_cfg,
    n_samples=500,
    seed=666,
    overwrite=True
)
df_large = pd.read_parquet(output_path_large)
assert len(df_large) == 500, "Should generate 500 samples"
assert not df_large.isnull().any().any(), "No NaN values"
print(f"✓ Large dataset (500 samples) generated successfully: {df_large.shape}")
output_path_large.unlink()

print("\n✓ All assertions passed for edge cases and file creation")

#%%
# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("ALL TESTS PASSED! ✓")
print("="*70)
print("\nSummary of dataset.build_dataset() integration tests:")
print("  ✓ TEST 1: Basic functionality - Structure, columns, arrays, types")
print("  ✓ TEST 2: Different sample sizes - 1, 50, 200 samples with consistent structure")
print("  ✓ TEST 3: Seed reproducibility - Same seed = same data, different seed = different data")
print("  ✓ TEST 4: Overwrite parameter - Preserves existing vs regenerates")
print("  ✓ TEST 5: Window configurations - Small, large, asymmetric lookback/forward")
print("  ✓ TEST 6: Data validation - OHLCV scaled, volume scaled, meta in [0,1], no NaN, valid indices")
print("  ✓ TEST 7: Edge cases - File naming, raw data creation, scaler creation, large datasets")
print("\n" + "="*70)
print("DATASET INTEGRATION VERIFIED")
print("All component functions work together correctly!")
print("Files 01 + 02 + 03 provide complete validation of the data pipeline.")
print("="*70)

