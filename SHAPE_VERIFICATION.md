"""
DATA SHAPE VERIFICATION ANALYSIS
=================================

This document verifies that data shapes match model expectations throughout the pipeline.

EXPECTED DATA FLOW:
-------------------

1. SYNTH.BUILD_SAMPLES() - Raw data generation
   Creates DataFrame where OHLCV columns contain numpy arrays:
   
   DataFrame structure:
   - Columns: ['idx', 'open', 'high', 'low', 'close', 'volume', 
               'equity', 'balance', 'position', 'sl_dist', 'tp_dist',
               'act_dir', 'act_dollar', 'act_sl', 'act_tp']
   - Shape: (n_samples, 15)
   
   OHLCV columns (object dtype):
   - Each cell contains: np.ndarray of shape (lookback,)
   - Example: df['open'].iloc[0].shape = (200,)
   
   Meta columns (float dtype):
   - Each cell contains: scalar float
   - Example: df['equity'].iloc[0] = 50000.0


2. DATASET.BUILD_DATASET() - Scaling
   Applies scaling and adds scaled columns:
   
   New columns added:
   - 'open_scaled', 'high_scaled', 'low_scaled', 'close_scaled', 'volume_scaled'
   
   Scaled OHLCV columns (object dtype):
   - Each cell contains: np.ndarray of shape (lookback,)
   - Values are min-max scaled within each window
   
   Meta columns are scaled in-place (still float dtype):
   - 'equity', 'balance', 'position', 'sl_dist', 'tp_dist',
     'act_dollar', 'act_sl', 'act_tp'


3. MAIN.PY - Model input preparation
   Code:
   ```python
   price_cols = ["open_scaled", "high_scaled", "low_scaled", "close_scaled", "volume_scaled"]
   X_price_train = np.stack([train_df[col].values for col in price_cols], axis=-1)
   X_meta_train = train_df[meta_cols].values
   ```
   
   CRITICAL QUESTION: What is the shape of X_price_train?
   
   Analysis:
   - train_df[col].values returns: np.ndarray of dtype=object, shape=(n_samples,)
   - Each element is: np.ndarray of shape=(lookback,)
   - np.stack([...], axis=-1) on list of object arrays
   
   EXPECTED behavior:
   - np.stack should handle object arrays containing ndarrays
   - Result should be: (n_samples, lookback, 5)
   
   POTENTIAL ISSUE:
   - If np.stack doesn't properly unpack object arrays, we get: (5, n_samples)
   - This would be WRONG!


4. MODEL EXPECTATIONS
   All models expect:
   - price_in: shape = (batch_size, lookback, 5)
   - meta_in:  shape = (batch_size, 8)
   
   Model definitions (from model.py):
   ```python
   price_in = tf.keras.Input(shape=(200, 5), name="price")
   meta_in = tf.keras.Input(shape=(meta_len,), name="meta")
   ```


VERIFICATION STRATEGY:
----------------------

The key question is: Does this line produce the correct shape?
```python
X_price = np.stack([df[col].values for col in price_cols], axis=-1)
```

When df[col] contains object arrays (each element is an ndarray):
- Option A (CORRECT): np.stack unpacks and creates (n_samples, lookback, 5)
- Option B (INCORRECT): np.stack treats as objects and creates (5, n_samples) or errors


TEST CASES:
-----------

Test 1: Object array stacking
```python
import numpy as np
import pandas as pd

# Simulate synth output
lookback, n_samples = 200, 100
df = pd.DataFrame({
    'open': [np.random.randn(lookback) for _ in range(n_samples)]
})

# What does this produce?
result = df['open'].values
print(result.shape)  # (100,) with dtype=object
print(result[0].shape)  # (200,)

# What about np.stack?
stacked = np.stack(df['open'].values)
print(stacked.shape)  # Should be (100, 200)
```

Test 2: Multi-column stacking (the actual code)
```python
price_cols = ['open', 'high', 'low', 'close', 'volume']
X = np.stack([df[col].values for col in price_cols], axis=-1)
print(X.shape)  # Should be (100, 200, 5)
```


DIAGNOSIS:
----------

CORRECT IMPLEMENTATION:
If np.stack properly handles object arrays, the current code is correct:
```python
X_price = np.stack([df[col].values for col in price_cols], axis=-1)
# Result: (n_samples, lookback, 5) ✓
```

INCORRECT IMPLEMENTATION (if needed):
If np.stack doesn't handle object arrays, we need:
```python
# First stack each column individually
price_arrays = [np.stack(df[col].values) for col in price_cols]
# Then stack across features
X_price = np.stack(price_arrays, axis=-1)
# Result: (n_samples, lookback, 5) ✓
```

OR alternatively:
```python
X_price = np.array([
    np.stack([df[col].iloc[i] for col in price_cols], axis=-1)
    for i in range(len(df))
])
# Result: (n_samples, lookback, 5) ✓
```


LIGHTGBM SPECIAL CASE:
----------------------

LightGBM expects flattened input:
```python
if model_type == "lightgbm":
    X_train = np.concatenate([
        X_price_train.reshape(len(X_price_train), -1),  # (n_samples, lookback*5)
        X_meta_train                                      # (n_samples, 8)
    ], axis=1)
    # Final shape: (n_samples, lookback*5 + 8) = (n_samples, 1008)
```

This requires X_price_train to be (n_samples, lookback, 5) FIRST.


RECOMMENDATION:
---------------

1. Run dev/test_stack.py to verify numpy stacking behavior
2. Run dev/check_shapes.py on actual generated data
3. If shapes are wrong, fix main.py with proper stacking
4. Add shape assertions in main.py for early error detection:

```python
# After stacking
assert X_price_train.shape == (len(train_df), CFG['lookback'], 5), \
    f"X_price_train shape mismatch: {X_price_train.shape}"
assert X_meta_train.shape == (len(train_df), 8), \
    f"X_meta_train shape mismatch: {X_meta_train.shape}"
```


PREDICTOR CONSIDERATIONS:
--------------------------

The Predictor class expects:
```python
def predict(self, X_price, X_meta, ...):
    # X_price: (N, lookback, 5) or (lookback, 5)
    # X_meta: (N, 8) or (8,)
```

It handles single samples by adding batch dimension:
```python
if X_price.ndim == 2:
    X_price = X_price[np.newaxis, ...]  # (1, lookback, 5)
if X_meta.ndim == 1:
    X_meta = X_meta[np.newaxis, ...]    # (1, 8)
```

This is CORRECT for the expected input shapes.


BATCH PROCESSING IN PREDICTOR:
-------------------------------

In predict_all_actions(), batch processing is done correctly:
```python
X_price = np.stack(X_prices)  # (n_samples, lookback, 5)
X_meta = np.stack(X_metas)    # (n_samples, 8)
```

Where X_prices and X_metas are lists of individual samples.
This is CORRECT because each element is already the right shape.


CONCLUSION:
-----------

The data flow SHOULD be correct IF numpy.stack properly handles
object arrays containing ndarrays. The critical line is:

```python
X_price = np.stack([df[col].values for col in price_cols], axis=-1)
```

This should produce (n_samples, lookback, 5) when:
- df[col].values is object array of shape (n_samples,)
- Each element is ndarray of shape (lookback,)

To verify: Run the diagnostic scripts created.
To fix (if needed): Use explicit stacking approach shown above.
"""
