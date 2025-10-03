# Performance Analysis: dataset.build_dataset() Optimization

## Current Bottlenecks (for 1M samples)

### 1. **synth.build_samples() - Main bottleneck**
**Problem:** Sequential loop with pandas slicing (1M iterations)
```python
for _ in tqdm.trange(n, desc="build samples"):
    idx = rng.integers(lookback, max_idx)
    rows.append(dict(
        open=df["open"].iloc[idx - lookback : idx].values,  # SLOW: 1M pandas slices
        # ... 4 more slices per iteration
    ))
```

**Impact:** O(n) pandas slicing operations = ~5-10 seconds per 1M samples

### 2. **OHLCV Scaling - Row-by-row apply**
**Problem:** `samples.apply(scale_row, axis=1)` processes 1M rows sequentially
```python
scaled_ohlcv = samples.apply(scale_row, axis=1)  # SLOW: Python loop in disguise
```

**Impact:** ~3-5 seconds per 1M samples

### 3. **Reward Computation**
**Problem:** `reward.compute_many()` likely loops over samples
**Impact:** Depends on implementation (need to check)

---

## Optimization Strategies

### Strategy 1: Vectorize `build_samples()` ⚡ (RECOMMENDED)

**Current:** Loop with pandas slicing  
**Better:** Pre-compute indices, use NumPy fancy indexing

```python
def build_samples_vectorized(df: pd.DataFrame, n: int, lookback: int, forward: int, 
                             rng: np.random.Generator, cfg: dict = None) -> pd.DataFrame:
    """Vectorized version using NumPy indexing."""
    if cfg is None:
        cfg = {}
    
    # Pre-generate all random indices at once
    max_idx = len(df) - lookback - forward
    indices = rng.integers(lookback, max_idx, size=n)
    
    # Pre-generate all synthetic states/actions
    equity = rng.uniform(cfg.get("synth_equity_min", 1e4), cfg.get("synth_equity_max", 1e5), size=n)
    balance = equity - rng.uniform(cfg.get("synth_balance_offset_min", -5e3), 
                                   cfg.get("synth_balance_offset_max", 5e3), size=n)
    position = rng.choice(cfg.get("synth_position_values", [0.0, 1.0, -1.0]), size=n)
    sl_dist = rng.uniform(cfg.get("synth_sl_min", 0.001), cfg.get("synth_sl_max", 0.05), size=n)
    tp_dist = rng.uniform(cfg.get("synth_tp_min", 0.001), cfg.get("synth_tp_max", 0.10), size=n)
    
    # Actions
    act_dirs = rng.choice(["hold", "long", "short"], size=n)
    act_dollar = rng.uniform(cfg.get("synth_dollar_min", 1e3), cfg.get("synth_dollar_max", 5e4), size=n)
    act_sl = rng.uniform(cfg.get("synth_sl_min", 0.001), cfg.get("synth_sl_max", 0.05), size=n)
    act_tp = rng.uniform(cfg.get("synth_tp_min", 0.001), cfg.get("synth_tp_max", 0.10), size=n)
    
    # Mask for hold actions
    hold_mask = act_dirs == "hold"
    act_dollar[hold_mask] = 0.0
    act_sl[hold_mask] = 0.0
    act_tp[hold_mask] = 0.0
    
    # Convert df to numpy arrays once
    open_arr = df["open"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    close_arr = df["close"].values
    volume_arr = df["volume"].values
    
    # Pre-allocate arrays for OHLCV windows
    ohlcv_windows = {
        "open": np.zeros((n, lookback), dtype=np.float32),
        "high": np.zeros((n, lookback), dtype=np.float32),
        "low": np.zeros((n, lookback), dtype=np.float32),
        "close": np.zeros((n, lookback), dtype=np.float32),
        "volume": np.zeros((n, lookback), dtype=np.float32),
    }
    
    # Extract windows using fancy indexing (still need loop, but NumPy is faster)
    for i, idx in enumerate(tqdm.tqdm(indices, desc="build samples")):
        start = idx - lookback
        ohlcv_windows["open"][i] = open_arr[start:idx]
        ohlcv_windows["high"][i] = high_arr[start:idx]
        ohlcv_windows["low"][i] = low_arr[start:idx]
        ohlcv_windows["close"][i] = close_arr[start:idx]
        ohlcv_windows["volume"][i] = volume_arr[start:idx]
    
    # Build DataFrame
    return pd.DataFrame({
        "idx": indices,
        "open": list(ohlcv_windows["open"]),
        "high": list(ohlcv_windows["high"]),
        "low": list(ohlcv_windows["low"]),
        "close": list(ohlcv_windows["close"]),
        "volume": list(ohlcv_windows["volume"]),
        "equity": equity,
        "balance": balance,
        "position": position,
        "sl_dist": sl_dist,
        "tp_dist": tp_dist,
        "act_dir": act_dirs,
        "act_dollar": act_dollar,
        "act_sl": act_sl,
        "act_tp": act_tp,
    })
```

**Expected Speedup:** 2-3x faster (5-10s → 2-3s for 1M samples)

---

### Strategy 2: Vectorize OHLCV Scaling ⚡⚡ (HIGH IMPACT)

**Current:** `apply(scale_row, axis=1)` - Python loop  
**Better:** Vectorized NumPy operations

```python
def scale_ohlcv_batch(samples: pd.DataFrame, ohlcv_cols: List[str]) -> pd.DataFrame:
    """Vectorized OHLCV scaling for entire DataFrame."""
    scaled_cols = {}
    
    for col in ohlcv_cols:
        if col == "volume":
            # Volume: log1p(volume) / log1p(last_volume)
            arrays = np.stack(samples[col].values)  # Shape: (n_samples, lookback)
            last_volume = arrays[:, -1]  # Last value per row
            scaled_cols[f"{col}_scaled"] = [
                np.log1p(arr) / np.log1p(last_val) 
                for arr, last_val in zip(arrays, last_volume)
            ]
        else:
            # OHLC: divide by last close
            close_arrays = np.stack(samples["close"].values)
            last_close = close_arrays[:, -1]  # Last close per row
            arrays = np.stack(samples[col].values)
            scaled_cols[f"{col}_scaled"] = [
                arr / last_close_val 
                for arr, last_close_val in zip(arrays, last_close)
            ]
    
    return pd.DataFrame(scaled_cols, index=samples.index)
```

**Even Better:** Use `np.apply_along_axis` or pure broadcasting if possible

**Expected Speedup:** 5-10x faster (3-5s → 0.5-1s for 1M samples)

---

### Strategy 3: Parallel Processing with Joblib/Multiprocessing 🚀

**Use Case:** Split dataset generation across CPU cores

```python
from joblib import Parallel, delayed

def build_samples_parallel(df: pd.DataFrame, n: int, lookback: int, forward: int,
                           rng: np.random.Generator, cfg: dict = None, n_jobs: int = -1) -> pd.DataFrame:
    """Build samples in parallel chunks."""
    # Split into chunks
    chunk_size = n // (n_jobs if n_jobs > 0 else os.cpu_count())
    chunks = []
    
    # Generate separate RNG per worker for reproducibility
    seeds = rng.integers(0, 2**32, size=n_jobs)
    
    def build_chunk(chunk_n, seed):
        chunk_rng = np.random.default_rng(seed)
        return build_samples_vectorized(df, chunk_n, lookback, forward, chunk_rng, cfg)
    
    # Process in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(build_chunk)(chunk_size, seed) 
        for seed in seeds
    )
    
    return pd.concat(results, ignore_index=True)
```

**Expected Speedup:** Near-linear with CPU cores (8 cores = ~8x on CPU-bound parts)

---

### Strategy 4: GPU Acceleration with CuPy/JAX 🚀🚀 (ADVANCED)

**Use Case:** If you have a GPU and dataset generation becomes the bottleneck

```python
import cupy as cp

def build_samples_gpu(df: pd.DataFrame, n: int, lookback: int, forward: int,
                     rng: np.random.Generator, cfg: dict = None) -> pd.DataFrame:
    """GPU-accelerated sample generation."""
    # Move data to GPU
    close_gpu = cp.array(df["close"].values)
    
    # Generate indices on GPU
    max_idx = len(df) - lookback - forward
    indices_gpu = cp.random.randint(lookback, max_idx, size=n)
    
    # ... GPU operations for synthetic data and windowing ...
    
    # Move back to CPU for DataFrame creation
    return pd.DataFrame(...)  # Convert from GPU arrays
```

**Expected Speedup:** 10-50x for large datasets (if properly optimized)  
**Tradeoff:** Increased complexity, GPU memory limits, data transfer overhead

---

## Recommended Implementation Order

### Phase 1: Quick Wins (1-2 hours) ⚡
1. ✅ **Vectorize synthetic data generation** in `build_samples()`
   - Generate all random values at once (equity, balance, etc.)
   - ~2x speedup

2. ✅ **Convert df columns to NumPy arrays once** before loop
   - Avoid repeated `.values` calls
   - ~1.2x speedup

### Phase 2: Major Optimization (2-4 hours) ⚡⚡
3. ✅ **Vectorize OHLCV scaling** 
   - Replace `apply(scale_row, axis=1)` with vectorized operations
   - ~5x speedup on scaling step

4. ✅ **Profile reward.compute_many()**
   - If slow, vectorize that too
   - Potential ~2-5x speedup

**Expected Total:** 5-10x overall speedup (1M samples: 30s → 3-6s)

### Phase 3: Advanced (4-8 hours) 🚀
5. 🔧 **Parallel processing with joblib**
   - Split dataset generation across cores
   - ~4-8x additional speedup (depending on cores)

6. 🔧 **GPU acceleration** (if needed)
   - Only if CPU parallelization isn't enough
   - ~10-50x but high complexity

**Expected Total:** 20-80x overall speedup (1M samples: 30s → 0.4-1.5s)

---

## Profiling Before Optimization

Run this to identify actual bottlenecks:

```python
import cProfile
import pstats

cfg = yaml.safe_load(open("config/default.yaml"))
profiler = cProfile.Profile()
profiler.enable()

# Build dataset
output = dataset.build_dataset(cfg, n_samples=100000, seed=42)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 slowest functions
```

This will tell you exactly where the time is spent.

---

## Implementation Priority

**For your use case (1M samples in tests):**

1. **Start with Phase 1** - Easy vectorization wins
2. **Profile to confirm** - See if more optimization needed
3. **Add Phase 2 if still slow** - Vectorize scaling
4. **Consider Phase 3 only if** - Building 10M+ samples regularly

Most likely, **Phase 1 + Phase 2 will be sufficient** to reduce 1M sample generation from ~30s to ~5s, which is acceptable for tests.

---

## Next Steps

Would you like me to:
1. **Implement Phase 1 optimizations** (vectorized synthetic data generation)?
2. **Implement Phase 2 optimizations** (vectorized OHLCV scaling)?
3. **Create a profiling script** to measure current performance first?
4. **Implement parallel processing** (Phase 3)?

Let me know your preference!
