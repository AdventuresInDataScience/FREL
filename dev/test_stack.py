"""
Minimal test to understand how DataFrame columns with arrays stack.
"""
import numpy as np
import pandas as pd

print("Testing DataFrame column stacking behavior...\n")
print("="*70)

# Simulate what synth.build_samples creates
lookback = 200
n_samples = 5

print(f"Creating {n_samples} samples with lookback={lookback}")
print()

# Each OHLCV column contains an ARRAY of length 'lookback'
sample_data = {
    'open': [np.random.randn(lookback) for _ in range(n_samples)],
    'high': [np.random.randn(lookback) for _ in range(n_samples)],
    'low': [np.random.randn(lookback) for _ in range(n_samples)],
    'close': [np.random.randn(lookback) for _ in range(n_samples)],
    'volume': [np.random.randn(lookback) for _ in range(n_samples)],
    'equity': np.random.randn(n_samples),  # scalar per sample
    'balance': np.random.randn(n_samples),  # scalar per sample
}

df = pd.DataFrame(sample_data)

print("DataFrame structure:")
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")
print()

print("Examining 'open' column:")
print(f"  df['open'] type: {type(df['open'])}")
print(f"  df['open'].values type: {type(df['open'].values)}")
print(f"  df['open'].values.shape: {df['open'].values.shape}")
print(f"  df['open'].values.dtype: {df['open'].values.dtype}")
print()

print("First element of 'open' column:")
first_open = df['open'].iloc[0]
print(f"  Type: {type(first_open)}")
print(f"  Shape: {first_open.shape}")
print(f"  First 5 values: {first_open[:5]}")
print()

print("="*70)
print("STACKING TEST (what main.py does)")
print("="*70)

price_cols = ['open', 'high', 'low', 'close', 'volume']

print("\nMethod 1: np.stack([df[col].values for col in price_cols], axis=-1)")
try:
    X_price_v1 = np.stack([df[col].values for col in price_cols], axis=-1)
    print(f"  Result shape: {X_price_v1.shape}")
    print(f"  Result dtype: {X_price_v1.dtype}")
    
    # Check if this is correct
    if X_price_v1.shape == (n_samples, lookback, 5):
        print(f"  ✓ SUCCESS: Shape is (n_samples={n_samples}, lookback={lookback}, n_features=5)")
    else:
        print(f"  ✗ UNEXPECTED: Expected ({n_samples}, {lookback}, 5)")
except Exception as e:
    print(f"  ✗ ERROR: {e}")

print("\nMethod 2: np.stack([df[col] for col in price_cols], axis=-1)")
try:
    X_price_v2 = np.stack([df[col] for col in price_cols], axis=-1)
    print(f"  Result shape: {X_price_v2.shape}")
    print(f"  Result dtype: {X_price_v2.dtype}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")

print("\nMethod 3: Checking what .values does to object arrays")
open_values = df['open'].values
print(f"  df['open'].values.shape: {open_values.shape}")
print(f"  df['open'].values.dtype: {open_values.dtype}")
print(f"  Type of first element: {type(open_values[0])}")
if hasattr(open_values[0], 'shape'):
    print(f"  Shape of first element: {open_values[0].shape}")

# Try to convert object array to 2D
print("\nMethod 4: np.array(df['open'].tolist())")
try:
    open_array = np.array(df['open'].tolist())
    print(f"  Shape: {open_array.shape}")
    print(f"  Dtype: {open_array.dtype}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")

print("\nMethod 5: np.stack(df['open'].values)")
try:
    open_stacked = np.stack(df['open'].values)
    print(f"  Shape: {open_stacked.shape}")
    print(f"  Dtype: {open_stacked.dtype}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")

print("\n" + "="*70)
print("TESTING FULL PIPELINE")
print("="*70)

# This is what should work
print("\nCorrect approach for DataFrame with array-valued columns:")
price_arrays = [np.stack(df[col].values) for col in price_cols]
print(f"  Individual column shapes: {[arr.shape for arr in price_arrays]}")

X_price = np.stack(price_arrays, axis=-1)
print(f"  Final stacked shape: {X_price.shape}")

if X_price.shape == (n_samples, lookback, 5):
    print(f"  ✓ CORRECT: ({n_samples}, {lookback}, 5)")
else:
    print(f"  ✗ WRONG: Expected ({n_samples}, {lookback}, 5)")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
When DataFrame columns contain arrays (object dtype):
1. df[col].values gives an object array of shape (n_samples,)
2. Each element is itself an array of shape (lookback,)
3. To convert to (n_samples, lookback), use: np.stack(df[col].values)
4. To get (n_samples, lookback, n_features), use:
   np.stack([np.stack(df[col].values) for col in cols], axis=-1)

OR simply:
   np.stack([df[col].values for col in cols], axis=-1)
   if numpy is smart enough to handle object arrays...
""")
