# Sequential Testing Suite - NEW STRUCTURE

This directory contains comprehensive tests for the refactored FREL data pipeline with dual-position structure.

## Architecture Changes

### What Changed
The pipeline was refactored from single-position to **dual-position structure**:

**OLD:** `position` (-1/0/1), `sl_dist`, `tp_dist`, `act_dir`, `act_dollar`, `act_sl`, `act_tp`  
**NEW:** `long_value`, `short_value`, `long_sl`, `long_tp`, `short_sl`, `short_tp` + 6 action equivalents

**Key Changes:**
1. Dual positions (long + short simultaneously)
2. SL/TP in multiplier notation (e.g., long_sl=0.95 = 5% stop)
3. Forward windows stored separately (~1000x space savings)
4. Complete reward simulation engine with proper SL/TP detection

---

## Test Files (Dependency Order)

### 01_test_data.py ✓ (UNCHANGED)
**Module:** `src/data.py`  
**Dependencies:** None

Already validated in previous tests. No changes to data loading/saving.

---

### 02_test_forward_windows.py 🆕 (NEW MODULE)
**Module:** `src/forward_windows.py`  
**Dependencies:** `data.py`

**Tests:**
1. `generate_forward_windows()` - Extract and scale forward OHLCV
   - Scaling to close[idx-1] reference
   - FP16 precision
   - Edge cases (start/end of data)
   - Vectorization correctness
   
2. `save_forward_windows()` / `load_forward_windows()` - Persistence
   - Compression formats
   - Data integrity
   - Memory efficiency
   
3. `create_forward_lookup()` - Dictionary creation
   - O(1) lookup performance
   - Memory usage
   - Missing key handling
   
4. `validate_forward_windows()` - Coverage validation
   - All indices covered
   - No duplicates
   - Proper error messages

**Benchmarks:**
- Generation time: 10K unique indices
- Lookup creation: 10K entries
- Memory usage: FP16 vs FP32

---

### 03_test_scale.py ✓ (MINOR CHANGES)
**Module:** `src/scale.py`  
**Dependencies:** None

**Tests:**
1. `MetaScaler.fit/transform()` - MinMax and Std scaling
   - Edge cases (zeros, negatives)
   - New columns (14 meta cols vs 8)
   
2. `inverse_transform_dict()` 🆕 - Unscaling
   - Accuracy validation
   - Missing columns handling
   - Round-trip consistency

**Benchmarks:**
- Fit time: 10K samples, 14 columns
- Transform time: 10K samples
- Inverse transform: 1K samples

---

### 04_test_synth.py ✓ (MAJOR CHANGES)
**Module:** `src/synth.py`  
**Dependencies:** None (uses synthetic data)

**Tests:**
1. `build_samples()` - Dual-position generation
   - 14 meta columns present
   - Log-normal distribution for position values
   - Truncated normal for SL/TP multipliers
   - Leverage constraint enforcement
   - Balance constraint (balance >= 0)
   - Hold states (10%)
   - Hold actions (20%)
   
2. Distribution validation
   - Position values: log-normal shape
   - SL/TP: truncated normal within bounds
   - Long SL: [0.50, 0.99]
   - Long TP: [1.01, 21.0]
   - Short SL: [1.01, 1.50]
   - Short TP: [0.50, 0.99]
   
3. Edge cases
   - Max leverage scenarios
   - Zero positions (hold states)
   - Identical current/action (hold actions)
   - Extreme equity values
   
4. Vectorization correctness
   - Single vs batch generation
   - Reproducibility with seed

**Benchmarks:**
- Generation time: 100, 1K, 10K, 100K samples
- Parallel vs single-thread: 100K samples
- Memory usage

---

### 05_test_reward.py ✓ (COMPLETE REWRITE)
**Module:** `src/reward.py`  
**Dependencies:** `scale.py`, `forward_windows.py`

**Tests:**

#### Part A: Helper Functions
1. `_unscale_position_data()`
   - Round-trip accuracy
   - All 14 columns
   
2. `_calculate_costs()`
   - Trading costs formula
   - Zero position handling
   
3. `_calculate_overnight_costs()`
   - Daily accumulation
   - Zero days/position handling

4. `_calculate_unrealized_pnl()` 🆕
   - Long profit/loss
   - Short profit/loss
   - Combined P&L
   
5. `_calculate_current_drawdown()` 🆕
   - Peak detection
   - Drawdown calculation
   - Bars since peak
   
6. `_calculate_bars_in_position()` 🆕
   - Consecutive bars
   - Total bars
   
7. `_calculate_position_metrics()` 🆕
   - Gross/net exposure
   - Leverage calculation

#### Part B: Simulation Engine
8. `_simulate_positions_forward()` - **CRITICAL**
   - Position transitions (bar 0)
     * Close current long
     * Close current short
     * Open target long
     * Open target short
     * Cost accounting
   
   - Long position SL/TP detection
     * SL hit on low
     * TP hit on high
     * Multiple bars active
     * Exit at end of window
   
   - Short position SL/TP detection
     * SL hit on high (inverted)
     * TP hit on low (inverted)
     * Multiple bars active
     * Exit at end of window
   
   - Edge cases
     * Zero positions (hold)
     * SL hit on first bar
     * TP hit on last bar
     * Both long/short active
     * SL=0 (no stop loss)
     * TP=0 (no take profit)
     * Simultaneous long/short exits
   
   - Cost validation
     * Transition costs
     * Overnight costs per bar
     * Exit costs
     * Total cost tracking
   
   - Daily P&L array
     * Correct length
     * Bar 0 includes transitions
     * Bars 1+ include daily changes
     * Last bar includes final exits

#### Part C: Reward Metrics
9. `car()` - Compound Annual Return
   - Positive/negative returns
   - Different time horizons
   - Zero equity handling
   
10. `sharpe()` - Sharpe Ratio
    - Annualization factor
    - Zero std handling
    - Negative returns
    
11. `sortino()` - Sortino Ratio
    - Downside deviation only
    - No downside case
    - Mixed returns
    
12. `calmar()` - Calmar Ratio
    - Drawdown calculation
    - No drawdown case
    - Zero equity handling

#### Part D: Integration
13. `compute_many()` - Full pipeline
    - Unscaling positions
    - Loading forward windows
    - Simulating positions
    - Computing rewards
    - Error handling
    - 100 samples validation

**Benchmarks:**
- Simulation: 1K samples with 200-bar forward window
- Reward computation: CAR, Sharpe, Sortino, Calmar
- compute_many(): 1K, 10K samples
- Memory usage per sample

---

### 06_test_dataset.py ✓ (UPDATED ORCHESTRATION)
**Module:** `src/dataset.py`  
**Dependencies:** All previous modules

**Tests:**
1. `build_dataset()` - Full pipeline integration
   - 100 samples end-to-end
   - All files created
   - Column validation
   - Reward computation
   
2. Forward windows integration
   - Unique indices extracted
   - Forward windows generated
   - Lookup dict created
   - Validation passed
   
3. Scaling integration
   - OHLCV scaled
   - Meta scaled (14 columns)
   - FP16 conversion
   - Scaler saved
   
4. File persistence
   - samples_*.parquet
   - forward_windows.parquet
   - meta_scaler.json
   - Compression working
   - Chunked saving (large datasets)

**Benchmarks:**
- End-to-end: 100, 1K, 10K samples
- Memory usage
- Disk usage
- Parallel scaling (n_jobs)

---

### 07_test_curriculum.py ✓ (UNCHANGED)
**Module:** `src/curriculum.py`  
**Dependencies:** `data.py`

Already validated. No changes needed.

---

### 08_test_model.py ⏭️ (TODO)
**Module:** `src/model.py`  
**Dependencies:** `dataset.py`

**Tests:**
1. Input handling - 14 meta columns (was 8)
2. Model architectures with new structure
3. Training with new data format

---

### 09_test_predictor.py ⏭️ (TODO)
**Module:** `src/predictor.py`  
**Dependencies:** `model.py`, `reward.py`

**Tests:**
1. Action search - 6-value space (was 4)
2. New action structure validation
3. Reward computation with actions

---

## Test Execution Order

```bash
# Phase 1: Foundation (READY)
python dev/sequential_testing_new/01_test_data.py
python dev/sequential_testing_new/02_test_forward_windows.py
python dev/sequential_testing_new/03_test_scale.py
python dev/sequential_testing_new/04_test_synth.py

# Phase 2: Reward System (READY)
python dev/sequential_testing_new/05_test_reward.py

# Phase 3: Integration (READY)
python dev/sequential_testing_new/06_test_dataset.py
python dev/sequential_testing_new/07_test_curriculum.py

# Phase 4: Model/Predictor (TODO)
python dev/sequential_testing_new/08_test_model.py
python dev/sequential_testing_new/09_test_predictor.py
```

---

## Benchmark Targets (40M Samples)

### Time Estimates
- Synth generation: ~10 min (100K/sec with parallel)
- Forward windows: ~5 min (unique indices only)
- Scaling: ~5 min
- Reward computation: ~2 hours (5K/sec)
- **Total: ~2.5 hours**

### Memory Estimates
- Raw samples: ~40 GB (14 meta + 5 OHLCV arrays)
- Forward windows: ~20 MB (unique indices, FP16)
- Peak memory: ~50 GB
- Chunked saving: Works with 16 GB RAM

### Disk Estimates
- samples_40M.parquet: ~15 GB (with snappy compression, FP16)
- forward_windows.parquet: ~20 MB
- meta_scaler.json: <1 MB
- **Total: ~15 GB**

---

## Key Validation Checks

### Data Consistency
- [ ] All 14 meta columns present
- [ ] All 5 OHLCV arrays correct shape
- [ ] Rewards computed for all samples
- [ ] Forward windows cover all indices
- [ ] Scaling reversible (round-trip test)

### Constraint Validation
- [ ] Leverage <= max_leverage (5.0x)
- [ ] Balance >= 0
- [ ] Long SL: [0.50, 0.99]
- [ ] Long TP: [1.01, 21.0]
- [ ] Short SL: [1.01, 1.50]
- [ ] Short TP: [0.50, 0.99]

### Distribution Validation
- [ ] Hold states: ~10%
- [ ] Hold actions: ~20%
- [ ] Position values: log-normal shape
- [ ] SL/TP: truncated normal shape

### Performance Validation
- [ ] Reward computation: <1ms per sample
- [ ] Memory usage: <100 GB for 40M
- [ ] Disk usage: <20 GB
- [ ] Parallel speedup: >4x with 8 cores

---

## Running All Tests

```bash
# Run all tests in sequence
python dev/sequential_testing_new/run_all_tests.py

# Run specific test
python dev/sequential_testing_new/01_test_data.py
python dev/sequential_testing_new/02_test_forward_windows.py
python dev/sequential_testing_new/03_test_scale.py
python dev/sequential_testing_new/04_test_synth.py
python dev/sequential_testing_new/05a_test_reward_helpers.py
```

---

## Current Status

### ✅ Completed Test Files
- [x] `01_test_data.py` - Data loading/saving (UNCHANGED)
- [x] `02_test_forward_windows.py` - Forward window generation (NEW MODULE)
- [x] `03_test_scale.py` - Scaling with 14 columns + inverse_transform_dict()
- [x] `04_test_synth.py` - Dual-position sample generation
- [x] `05a_test_reward_helpers.py` - All helper functions in reward.py
- [x] `run_all_tests.py` - Automated test runner

### 🚧 In Progress
- [ ] `05b_test_reward_simulation.py` - Simulation engine edge cases
- [ ] `05c_test_reward_metrics.py` - CAR, Sharpe, Sortino, Calmar
- [ ] `06_test_dataset.py` - Full pipeline integration
- [ ] `07_test_curriculum.py` - Curriculum (should be unchanged)

### ⏭️ Future
- [ ] `08_test_model.py` - Model with 14 inputs
- [ ] `09_test_predictor.py` - Predictor with 6-value actions
