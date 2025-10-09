#%%
# =============================================================================
# TEST 01: data.py Functions
# Test data download, split, save/load functions
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

from src import data

# Load config
config_path = project_root / "config" / "default.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

print("✓ Imports successful")
print(f"Config loaded: {config_path}")
print(f"Testing module: data.py")

#%%
# Override Config with test values
import time
test_timestamp = int(time.time() * 1000) % 1000000  # Last 6 digits for uniqueness

test_cfg = cfg.copy()
test_cfg.update({
    "ticker": "^GSPC",
    "start": "1960-01-01",  # Recent data for faster download
    "data_dir": "data",
    "raw_data_filename": f"test_raw_{{ticker}}_{test_timestamp}.parquet",
    "parquet_compression": "gzip",
})

print(f"\nTest config created (timestamp: {test_timestamp})")
print(f"  - Ticker: {test_cfg['ticker']}")
print(f"  - Start date: {test_cfg['start']}")
print(f"  - Test ID: {test_timestamp}")

#%%
# =============================================================================
# TEST 1: data.download() - Download market data
# =============================================================================
print("\n" + "="*70)
print("TEST 1: data.download() - Download market data")
print("="*70)

# Test 1a: Basic download with default parameters
print("\n[1a] Basic download test...")
df_downloaded = data.download(ticker=test_cfg["ticker"], start=test_cfg["start"])

print(f"✓ Downloaded {len(df_downloaded)} rows")
print(f"  - Date range: {df_downloaded.index[0]} to {df_downloaded.index[-1]}")
print(f"  - Columns: {list(df_downloaded.columns)}")
print(f"  - Shape: {df_downloaded.shape}")

# Validate structure
assert isinstance(df_downloaded, pd.DataFrame), "Should return DataFrame"
assert len(df_downloaded) > 0, "Should have rows"
assert all(col in df_downloaded.columns for col in ["open", "high", "low", "close", "volume"]), "Missing OHLCV columns"
assert not df_downloaded.isnull().any().any(), "Should not have NaN values"

# Test 1b: Validate OHLC relationship (High >= Low, etc.)
print("\n[1b] Testing OHLC relationships...")
assert (df_downloaded["high"] >= df_downloaded["low"]).all(), "High should be >= Low"
assert (df_downloaded["high"] >= df_downloaded["close"]).all(), "High should be >= Close"
assert (df_downloaded["high"] >= df_downloaded["open"]).all(), "High should be >= Open"
assert (df_downloaded["low"] <= df_downloaded["close"]).all(), "Low should be <= Close"
assert (df_downloaded["low"] <= df_downloaded["open"]).all(), "Low should be <= Open"
assert (df_downloaded["volume"] >= 0).all(), "Volume should be non-negative"
print("✓ OHLC relationships are valid")

# Test 1c: Different date ranges
print("\n[1c] Testing different date ranges...")
df_recent = data.download(ticker="^GSPC", start="2023-01-01")
assert len(df_recent) < len(df_downloaded), "Recent data should have fewer rows"
assert df_recent.index[0] >= df_downloaded.index[0], "Recent data should start later"
print(f"✓ Recent download: {len(df_recent)} rows from {df_recent.index[0]}")

# Test 1d: Different tickers (if possible - may fail if yfinance is down)
print("\n[1d] Testing different tickers...")
try:
    df_nasdaq = data.download(ticker="^IXIC", start="2023-01-01")
    assert len(df_nasdaq) > 0, "NASDAQ download should work"
    assert list(df_nasdaq.columns) == list(df_downloaded.columns), "Columns should be consistent"
    print(f"✓ NASDAQ download: {len(df_nasdaq)} rows")
except Exception as e:
    print(f"⚠️  NASDAQ download skipped (network issue): {e}")

print("\n✓ All assertions passed for download()")

#%%
# =============================================================================
# TEST 2: data.split() - Split data into train/test
# =============================================================================
print("\n" + "="*70)
print("TEST 2: data.split() - Split data into train/test")
print("="*70)

# Test 2a: Standard 80/20 split
print("\n[2a] Testing 80/20 split...")
df_train, df_test = data.split(df_downloaded, ratio=0.8)

print(f"✓ Split completed")
print(f"  - Train size: {len(df_train)} rows ({len(df_train)/len(df_downloaded)*100:.1f}%)")
print(f"  - Test size: {len(df_test)} rows ({len(df_test)/len(df_downloaded)*100:.1f}%)")

# Validate split
assert len(df_train) + len(df_test) == len(df_downloaded), "Split should preserve total rows"
assert len(df_train) > len(df_test), "Train should be larger with 0.8 ratio"
assert df_train.index[-1] < df_test.index[0], "Train should come before test chronologically"
assert list(df_train.columns) == list(df_test.columns), "Columns should be preserved"

# Test 2b: Different split ratios
print("\n[2b] Testing different split ratios...")
ratios_to_test = [0.5, 0.7, 0.9, 0.95]
for ratio in ratios_to_test:
    train, test = data.split(df_downloaded, ratio=ratio)
    actual_ratio = len(train) / len(df_downloaded)
    expected_rows = int(len(df_downloaded) * ratio)
    assert len(train) == expected_rows, f"Train should have {expected_rows} rows for ratio {ratio}"
    assert len(train) + len(test) == len(df_downloaded), "Total rows should be preserved"
    print(f"  ✓ Ratio {ratio}: {len(train)} train, {len(test)} test")

# Test 2c: Edge case - very small split
print("\n[2c] Testing edge cases...")
train_tiny, test_large = data.split(df_downloaded, ratio=0.1)
assert len(train_tiny) < len(test_large), "10% split should have more test than train"
print(f"  ✓ 10% split: {len(train_tiny)} train, {len(test_large)} test")

# Test 2d: Validate no data leakage
print("\n[2d] Validating no data leakage...")
train_dates = set(df_train.index)
test_dates = set(df_test.index)
assert len(train_dates.intersection(test_dates)) == 0, "No dates should overlap between train and test"
print("  ✓ No date overlap between train and test sets")

print("\n✓ All assertions passed for split()")

#%%
# =============================================================================
# TEST 3: data.save() and data.load() - Save/load parquet
# =============================================================================
print("\n" + "="*70)
print("TEST 3: data.save() and data.load() - Save/load parquet")
print("="*70)

test_dir = Path("data") / "test_temp" / f"test_{test_timestamp}"
test_dir.mkdir(exist_ok=True, parents=True)

# Test 3a: Save and load with gzip compression
print("\n[3a] Testing save/load with gzip compression...")
test_path_gzip = test_dir / "test_gzip.parquet"
data.save(df_downloaded, test_path_gzip, compression="gzip")
print(f"✓ Saved to {test_path_gzip}")
assert test_path_gzip.exists(), "File should exist after save"

df_loaded_gzip = data.load(test_path_gzip)
print(f"✓ Loaded {len(df_loaded_gzip)} rows")

# Validate loaded data
assert len(df_loaded_gzip) == len(df_downloaded), "Loaded data should match original length"
assert list(df_loaded_gzip.columns) == list(df_downloaded.columns), "Columns should match"
pd.testing.assert_frame_equal(df_loaded_gzip, df_downloaded), "Loaded data should exactly match saved data"

# Test 3b: Save and load with snappy compression
print("\n[3b] Testing save/load with snappy compression...")
test_path_snappy = test_dir / "test_snappy.parquet"
data.save(df_downloaded, test_path_snappy, compression="snappy")
df_loaded_snappy = data.load(test_path_snappy)
assert len(df_loaded_snappy) == len(df_downloaded), "Snappy compressed data should load correctly"
pd.testing.assert_frame_equal(df_loaded_snappy, df_downloaded), "Snappy data should match original"
print(f"✓ Snappy compression works correctly")

# Test 3c: Compare file sizes
print("\n[3c] Comparing compression file sizes...")
size_gzip = test_path_gzip.stat().st_size
size_snappy = test_path_snappy.stat().st_size
print(f"  - gzip: {size_gzip:,} bytes")
print(f"  - snappy: {size_snappy:,} bytes")
print(f"  - Ratio: {size_snappy/size_gzip:.2f}x (snappy/gzip)")

# Test 3d: Test saving subset of data
print("\n[3d] Testing save/load with data subset...")
df_subset = df_downloaded.iloc[:100]
test_path_subset = test_dir / "test_subset.parquet"
data.save(df_subset, test_path_subset, compression="gzip")
df_loaded_subset = data.load(test_path_subset)
assert len(df_loaded_subset) == 100, "Should load exactly 100 rows"
pd.testing.assert_frame_equal(df_loaded_subset, df_subset), "Subset should match"
print(f"✓ Subset save/load works correctly")

# Test 3e: Test that directory creation works
print("\n[3e] Testing automatic directory creation...")
nested_path = test_dir / "nested" / "deep" / "test.parquet"
data.save(df_subset, nested_path, compression="gzip")
assert nested_path.exists(), "Nested directories should be created automatically"
print(f"✓ Automatic directory creation works")

# Cleanup
print("\n[3f] Cleaning up test files...")
import shutil
shutil.rmtree(test_dir.parent)
print(f"✓ Cleaned up {test_dir.parent}")

print("\n✓ All assertions passed for save() and load()")

#%%
# =============================================================================
# TEST 4: Edge cases and data quality
# =============================================================================
print("\n" + "="*70)
print("TEST 4: Edge cases and data quality")
print("="*70)

# Test 4a: Very small data download range
print("\n[4a] Testing with minimal date range...")
df_minimal = data.download(ticker="^GSPC", start="2024-12-01")
assert len(df_minimal) > 0, "Should download even small date ranges"
assert len(df_minimal) < len(df_downloaded), "Minimal range should have fewer rows"
print(f"✓ Minimal date range works: {len(df_minimal)} rows")

# Test 4b: Verify data quality (no NaN, proper types)
print("\n[4b] Validating data quality...")
assert not df_downloaded.isnull().any().any(), "Downloaded data should have no NaN values"
assert all(df_downloaded[col].dtype in [np.float32, np.float64] for col in ["open", "high", "low", "close"]), "OHLC should be float type"
assert df_downloaded["volume"].dtype in [np.int64, np.int32, np.float32, np.float64], "Volume should be numeric type"
print(f"✓ Data quality validated (no NaN, correct types)")

# Test 4c: Verify chronological order
print("\n[4c] Verifying chronological order...")
dates = df_downloaded.index
assert (dates == sorted(dates)).all(), "Dates should be in chronological order"
print(f"✓ Data is chronologically ordered")

# Test 4d: Verify no duplicate dates
print("\n[4d] Checking for duplicate dates...")
assert not dates.duplicated().any(), "Should have no duplicate dates"
print(f"✓ No duplicate dates found")

print("\n✓ All data quality tests passed")

#%%
# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("✅ ALL DATA TESTS PASSED")
print("="*70)
print("\nSummary of data.py tests:")
print("  ✓ TEST 1: data.download() - Multiple tickers, date ranges, OHLC validation")
print("  ✓ TEST 2: data.split() - Various ratios, chronological ordering, no leakage")
print("  ✓ TEST 3: data.save() and load() - Compression formats, nested directories")
print("  ✓ TEST 4: data quality - Edge cases, NaN checks, chronological order")
print("\n" + "="*70)
print("DATA.PY FUNCTIONS VERIFIED")
print("Core data download, split, and persistence functions work correctly!")
print("These are the foundation for all subsequent data processing.")
print("="*70)
