"""
Diagnostic script to verify data shapes match model expectations.
"""
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# Load config
CFG = yaml.safe_load(open("config/default.yaml"))

# Check if dataset exists
dataset_path = Path("data/samples_0M.parquet")
if dataset_path.exists():
    print(f"Loading dataset from {dataset_path}")
    df = pd.read_parquet(dataset_path)
    
    print("\n" + "="*70)
    print("DATASET STRUCTURE")
    print("="*70)
    print(f"Number of samples: {len(df)}")
    print(f"\nDataFrame columns: {df.columns.tolist()}")
    print(f"\nDataFrame shape: {df.shape}")
    
    # Check OHLCV columns
    print("\n" + "-"*70)
    print("OHLCV COLUMNS (should contain arrays)")
    print("-"*70)
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    for col in ohlcv_cols:
        if col in df.columns:
            first_val = df[col].iloc[0]
            print(f"{col:8s} - type: {type(first_val).__name__:12s}, shape: {first_val.shape if hasattr(first_val, 'shape') else 'N/A'}")
    
    # Check scaled OHLCV columns
    print("\n" + "-"*70)
    print("SCALED OHLCV COLUMNS (should contain arrays)")
    print("-"*70)
    scaled_cols = ["open_scaled", "high_scaled", "low_scaled", "close_scaled", "volume_scaled"]
    for col in scaled_cols:
        if col in df.columns:
            first_val = df[col].iloc[0]
            print(f"{col:15s} - type: {type(first_val).__name__:12s}, shape: {first_val.shape if hasattr(first_val, 'shape') else 'N/A'}")
    
    # Check meta columns
    print("\n" + "-"*70)
    print("META COLUMNS (should be scalars)")
    print("-"*70)
    meta_cols = ["equity", "balance", "position", "sl_dist", "tp_dist", "act_dollar", "act_sl", "act_tp"]
    for col in meta_cols:
        if col in df.columns:
            first_val = df[col].iloc[0]
            print(f"{col:12s} - type: {type(first_val).__name__:12s}, value: {first_val}")
    
    # Now check what happens when we stack for model input
    print("\n" + "="*70)
    print("MODEL INPUT PREPARATION")
    print("="*70)
    
    print("\nAttempting to stack scaled OHLCV columns for model input...")
    try:
        # This is what main.py does:
        price_cols = ["open_scaled", "high_scaled", "low_scaled", "close_scaled", "volume_scaled"]
        X_price = np.stack([df[col].values for col in price_cols], axis=-1)
        print(f"✓ X_price shape: {X_price.shape}")
        print(f"  Expected: (n_samples, lookback={CFG['lookback']}, 5)")
        
        if X_price.shape[1:] == (CFG['lookback'], 5):
            print(f"  ✓ Shape matches expectation!")
        else:
            print(f"  ✗ SHAPE MISMATCH!")
            print(f"    Got:      {X_price.shape}")
            print(f"    Expected: (n_samples, {CFG['lookback']}, 5)")
    except Exception as e:
        print(f"✗ Error stacking price columns: {e}")
    
    print("\nAttempting to extract meta features...")
    try:
        meta_cols = ["equity", "balance", "position", "sl_dist", "tp_dist", "act_dollar", "act_sl", "act_tp"]
        X_meta = df[meta_cols].values
        print(f"✓ X_meta shape: {X_meta.shape}")
        print(f"  Expected: (n_samples, 8)")
        
        if X_meta.shape[1] == 8:
            print(f"  ✓ Shape matches expectation!")
        else:
            print(f"  ✗ SHAPE MISMATCH!")
    except Exception as e:
        print(f"✗ Error extracting meta: {e}")
    
    # Check model expectations
    print("\n" + "="*70)
    print("MODEL EXPECTATIONS")
    print("="*70)
    print(f"Model type: {CFG['model_type']}")
    print(f"\nExpected inputs:")
    print(f"  price_in: (batch_size, {CFG['lookback']}, 5)  <- OHLCV windows")
    print(f"  meta_in:  (batch_size, 8)                      <- state + action features")
    
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    
    # Check if arrays are properly formed
    try:
        sample_price = df[price_cols[0]].iloc[0]
        if isinstance(sample_price, np.ndarray):
            if sample_price.shape[0] == CFG['lookback']:
                print("✓ OHLCV arrays have correct lookback length")
            else:
                print(f"✗ OHLCV arrays have wrong length: {sample_price.shape[0]} (expected {CFG['lookback']})")
        else:
            print(f"✗ OHLCV columns don't contain arrays! Type: {type(sample_price)}")
    except Exception as e:
        print(f"✗ Error checking OHLCV structure: {e}")
        
else:
    print(f"Dataset not found at {dataset_path}")
    print("\nTo generate a test dataset, run:")
    print("  python -c \"from src.dataset import build_dataset; import yaml; cfg = yaml.safe_load(open('config/default.yaml')); build_dataset(cfg, 1000)\"")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
Expected data flow:
1. synth.build_samples() creates DataFrame where OHLCV columns contain numpy arrays
   - Each row has: open=[array of 200 values], high=[array], etc.
   
2. dataset.build_dataset() scales these arrays:
   - Creates open_scaled, high_scaled, etc. (still arrays)
   
3. main.py stacks them for model input:
   - np.stack([df[col].values for col in price_cols], axis=-1)
   - This should produce: (n_samples, 200, 5)
   
4. Models expect:
   - price_in: (batch_size, 200, 5)
   - meta_in: (batch_size, 8)
""")
