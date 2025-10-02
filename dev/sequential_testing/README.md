# Sequential Testing Structure

This directory contains comprehensive tests for the FREL data pipeline, organized into sequential files that test components in dependency order.

## Test Files Overview

### 01 test_data_functions.py
**Tests: `src/data.py` core functions**

Dependencies: None (standalone)

Tests:
- ✓ TEST 1: `data.download()` - Multiple tickers, date ranges, OHLC validation
- ✓ TEST 2: `data.split()` - Various ratios, chronological ordering, no leakage  
- ✓ TEST 3: `data.save()` and `data.load()` - Compression formats (gzip, snappy), nested directories
- ✓ TEST 4: Data quality - Edge cases, minimal date ranges, extreme split ratios, data types

**Purpose:** Validates the foundation - downloading, splitting, and persisting market data.

---

### 02 test_scale_synth_functions.py
**Tests: `src/scale.py` and `src/synth.py`**

Dependencies: Uses `data.download()` for real data (validated in File 01)

Tests:
- ✓ TEST 1: `scale.scale_ohlcv_window()` - OHLCV scaling (÷ last close), Volume scaling (log1p transformation)
  - Uses mock data with known values for precise validation
  - Validates last close = 1.0, last volume = 1.0 after scaling
  - Tests different last close values

- ✓ TEST 2: `scale.MetaScaler` - MinMax scaling for meta columns
  - Validates formula: (x - min) / (max - min)
  - Tests save/load functionality
  - Tests edge cases (constant values, near-zero ranges)

- ✓ TEST 3: `synth.build_samples()` - Synthetic sample generation
  - Validates random sampling with valid indices
  - Tests seed reproducibility
  - Validates synthetic meta column ranges
  - Tests different window configurations

**Purpose:** Validates scaling logic and synthetic data generation work correctly.

---

### 03 test_dataset_functions.py
**Tests: `src/dataset.py` integration wrapper**

Dependencies: Requires all components from Files 01-02 working correctly

Tests:
- ✓ TEST 1: Basic functionality - Structure, columns, arrays, types
- ✓ TEST 2: Different sample sizes - 1, 50, 200 samples with consistent structure
- ✓ TEST 3: Seed reproducibility - Same seed = identical, different seed = different
- ✓ TEST 4: Overwrite parameter - Preserves existing vs regenerates
- ✓ TEST 5: Window configurations - Small, large, asymmetric lookback/forward
- ✓ TEST 6: Data validation - OHLCV scaled, volume scaled, meta in [0,1], no NaN, valid indices
- ✓ TEST 7: Edge cases - File naming, raw data creation, scaler creation, large datasets

**Purpose:** Validates the complete pipeline integrating all components together.

---

## Running the Tests

### Interactive Execution (Recommended)
All test files use `#%%` cell markers for VS Code interactive Python:

1. Open the file in VS Code
2. Click "Run Cell" or use `Shift+Enter` to run individual test blocks
3. Inspect outputs and intermediate results
4. Re-run specific tests without running the entire suite

### Script Execution
Run complete test files as scripts:
```powershell
# Run individual test files
uv run python dev/sequential_testing/01_test_data_functions.py
uv run python dev/sequential_testing/02_test_scale_synth_functions.py
uv run python dev/sequential_testing/03_test_dataset_functions.py

# Or run all in sequence
uv run python dev/sequential_testing/01_test_data_functions.py; uv run python dev/sequential_testing/02_test_scale_synth_functions.py; uv run python dev/sequential_testing/03_test_dataset_functions.py
```

---

## Test Design Philosophy

### 1. Dependency Order
Tests are ordered by dependency:
- **File 01:** Foundation (data download/split/save/load) - no dependencies
- **File 02:** Processing (scaling, synthetic data) - uses data from File 01
- **File 03:** Integration (full pipeline) - uses all components from Files 01-02

### 2. Component Isolation
Each file tests specific modules in isolation:
- File 01: Only `data.py` functions
- File 02: Only `scale.py` and `synth.py` functions
- File 03: Only `dataset.py` wrapper (assumes components work)

### 3. Mock Data Validation
File 02 uses **mock data with known values** to validate scaling formulas:
```python
# Mock OHLCV where last close = 100.0
mock_ohlcv = {
    "close": np.array([92.0, 97.0, 99.0, 101.0, 100.0])
}
# After scaling: should be [0.92, 0.97, 0.99, 1.01, 1.00]
```

This ensures scaling logic is **mathematically correct**, not just "doesn't crash".

### 4. Integration Validation
File 03 validates the **complete pipeline**:
- All functions work together correctly
- Data flows through the pipeline without errors
- Output format is correct and consistent
- Edge cases are handled properly

### 5. Comprehensive Coverage
Each file includes:
- ✅ Happy path tests (normal usage)
- ✅ Edge cases (min/max values, boundary conditions)
- ✅ Parameter variations (different configs)
- ✅ Error conditions (when applicable)
- ✅ Data quality validation (no NaN, correct types, valid ranges)

---

## Key Validation Points

### OHLCV Scaling (File 02 & 03)
```python
# Formula: ohlcv / last_close
# Result: Last close value becomes 1.0
assert abs(scaled_close[-1] - 1.0) < 1e-6
```

### Volume Scaling (File 02 & 03)
```python
# Formula: log1p(volume) / log1p(last_volume)
# Result: Last volume becomes 1.0
assert abs(scaled_volume[-1] - 1.0) < 1e-6
```

### Meta Scaling (File 02 & 03)
```python
# Formula: (x - min) / (max - min)
# Result: All values in [0, 1]
assert meta_col.min() >= 0.0
assert meta_col.max() <= 1.0
```

### Data Quality (All Files)
```python
# No NaN values
assert not df.isnull().any().any()

# Correct data types
assert df["close"].dtype in [np.float32, np.float64]
assert df["volume"].dtype in [np.int64, np.int32, np.float32, np.float64]

# Valid array lengths
assert len(ohlcv_array) == lookback
```

---

## Troubleshooting

### If File 01 fails:
- Check internet connection (yfinance downloads require network)
- Check ticker symbol is valid
- Check date range has data available
- Check data directory exists and is writable

### If File 02 fails:
- Ensure File 01 tests pass first (File 02 uses `data.download()`)
- Check scaling formulas match implementation in `scale.py`
- Verify `synth.py` parameter ranges are valid

### If File 03 fails:
- Ensure Files 01 and 02 pass first
- Check all required files exist (raw data, scaler)
- Verify dataset output directory is writable
- Check memory for large datasets

---

## Future Extensions

Additional test files can be added following the same pattern:

- **04 test_reward_functions.py** - Test reward calculations
- **05 test_curriculum.py** - Test curriculum learning strategies
- **06 test_model_builds.py** - Test model architectures
- **07 test_model_training.py** - Test training loops
- **08 test_mapie.py** - Test conformal prediction
- **09 test_predictor.py** - Test inference pipeline

Each file builds on validated components from previous files.

---

## Summary

✅ **File 01:** Foundation validated (download, split, save, load)  
✅ **File 02:** Processing validated (scaling, synthetic data)  
✅ **File 03:** Integration validated (complete pipeline)  

**Result:** Complete confidence in the data pipeline from raw download to model-ready datasets.
