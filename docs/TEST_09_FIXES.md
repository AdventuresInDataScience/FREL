# Test 09 Predictor - Function Name Corrections

## Errors Fixed:

### 1. **data module**
- ❌ `data.load_raw_data()` 
- ✅ `data.load(path)`

### 2. **MetaScaler loading**
- ❌ `meta_scaler = scale.MetaScaler.load(path)`
- ✅ `meta_scaler = scale.MetaScaler()` then `meta_scaler.load(path)`

### 3. **synth module**
- ❌ `synth.synthesize_ohlcv()`
- ✅ `synth.build_samples(df, n, lookback, forward, rng, cfg)`
  - Returns DataFrame with columns: idx, equity, balance, positions, actions, OHLCV arrays

### 4. **dataset module**
- ❌ `dataset.TradingDataset` class (DOESN'T EXIST)
- ✅ `dataset.build_dataset()` function (returns path to parquet)
- ✅ For testing: use synth.build_samples() directly

### 5. **Data structure**
- synth.build_samples() returns DataFrame with:
  - Scalar columns: equity, balance, long_value, short_value, etc.
  - Array columns: open, high, low, close, volume (np.ndarrays)
- No need for inverse transforms - work with synth output directly

## Correct Usage Pattern:

```python
# Load data
df_raw = data.load(path)

# Load scaler
meta_scaler = scale.MetaScaler()
meta_scaler.load(scaler_path)

# Build synthetic samples (returns DataFrame, not dataset object)
rng = np.random.default_rng(seed)
df_synth = synth.build_samples(
    df=df_raw,
    n=n_samples,
    lookback=lookback,
    forward=forward,
    rng=rng,
    cfg=cfg
)

# Get a row to test with
test_row = df_synth.iloc[0]

# Scale OHLCV
ohlcv_dict = {k: test_row[k] for k in ['open', 'high', 'low', 'close', 'volume']}
ohlcv_scaled = scale.scale_ohlcv_window(ohlcv_dict)

# Create DataFrame for predictor
df_for_pred = pd.DataFrame({
    'open_scaled': [ohlcv_scaled['open']],
    'high_scaled': [ohlcv_scaled['high']],
    # ... other columns from test_row
})
```

## Test File Status:
✅ All syntax errors fixed
✅ All import errors fixed  
✅ All function calls corrected
✅ Removed non-existent TradingDataset class
✅ Using correct data structures
✅ Ready to run
