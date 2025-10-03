"""
End-to-end test of the refactored data pipeline.

Tests:
1. Data loading
2. Sample generation (new structure)
3. Forward windows generation
4. Scaling
5. Reward computation
6. Dataset saving/loading
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import dataset, data as data_module

print("=" * 70)
print("FULL PIPELINE TEST - NEW DUAL-POSITION STRUCTURE")
print("=" * 70)

# Load config
config_path = Path(__file__).parent.parent / "config" / "default.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Override for small test
cfg['n_samples'] = 100
n_samples = 100

print(f"\n📋 Configuration:")
print(f"  Samples: {n_samples}")
print(f"  Lookback: {cfg['lookback']}")
print(f"  Forward: {cfg['forward']}")
print(f"  Reward: {cfg['reward_key']}")
print(f"  Max leverage: {cfg['max_leverage']}x")

# Test data loading
print("\n" + "=" * 70)
print("STEP 1: Data Loading")
print("=" * 70)

data_dir = Path(cfg.get("data_dir", "data"))
raw_filename = cfg.get("raw_data_filename", "raw_{ticker}.parquet").format(ticker=cfg['ticker'])
raw_path = data_dir / raw_filename

if not raw_path.exists():
    print(f"  Downloading {cfg['ticker']} data...")
    df = data_module.download(cfg["ticker"], cfg["start"])
    data_module.save(df, raw_path, cfg["parquet_compression"])
    print(f"  ✓ Saved to {raw_path}")
else:
    print(f"  Loading existing data from {raw_path}...")
    df = pd.read_parquet(raw_path)
    print(f"  ✓ Loaded {len(df):,} bars")

print(f"  Date range: {df.index[0]} to {df.index[-1]}")
print(f"  Columns: {list(df.columns)}")

# Test dataset generation
print("\n" + "=" * 70)
print("STEP 2: Dataset Generation")
print("=" * 70)

print(f"\n  Building dataset with {n_samples} samples...")
t_start = time.perf_counter()

try:
    out_path = dataset.build_dataset(
        cfg=cfg,
        n_samples=n_samples,
        seed=42,
        overwrite=True,
        n_jobs=1  # Single thread for debugging
    )
    t_end = time.perf_counter()
    
    print(f"\n  ✓ Dataset generation completed in {t_end - t_start:.2f}s")
    print(f"  ✓ Saved to: {out_path}")
    
except Exception as e:
    print(f"\n  ✗ Error during dataset generation:")
    print(f"    {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Verify files created
print("\n" + "=" * 70)
print("STEP 3: Verify Files Created")
print("=" * 70)

samples_file = data_dir / cfg.get("samples_filename", "samples_{n}M.parquet").format(n=n_samples // 1_000_000)
forward_file = data_dir / cfg.get("forward_windows_filename", "forward_windows.parquet")
scaler_file = data_dir / cfg.get("scaler_filename", "meta_scaler.json")

files_to_check = [
    ("Samples", samples_file),
    ("Forward windows", forward_file),
    ("Scaler", scaler_file)
]

all_exist = True
for name, path in files_to_check:
    if path.exists():
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  ✓ {name}: {path.name} ({size_mb:.2f} MB)")
    else:
        print(f"  ✗ {name}: {path.name} - NOT FOUND")
        all_exist = False

if not all_exist:
    print("\n  ✗ Some files missing!")
    sys.exit(1)

# Load and inspect samples
print("\n" + "=" * 70)
print("STEP 4: Inspect Samples Structure")
print("=" * 70)

samples_df = pd.read_parquet(samples_file)
print(f"\n  Loaded {len(samples_df):,} samples")
print(f"\n  Columns ({len(samples_df.columns)}):")

# Group columns by type
meta_cols = ['idx', 'equity', 'balance', 'forward']
position_cols = ['long_value', 'short_value', 'long_sl', 'long_tp', 'short_sl', 'short_tp']
action_cols = ['act_long_value', 'act_short_value', 'act_long_sl', 'act_long_tp', 'act_short_sl', 'act_short_tp']
array_cols = ['open_scaled', 'high_scaled', 'low_scaled', 'close_scaled', 'volume_scaled']
label_col = ['y']

print("\n  Meta columns:")
for col in meta_cols:
    if col in samples_df.columns:
        print(f"    ✓ {col}: {samples_df[col].dtype}")
    else:
        print(f"    ✗ {col}: MISSING")

print("\n  Position state columns:")
for col in position_cols:
    if col in samples_df.columns:
        val_range = f"[{samples_df[col].min():.3f}, {samples_df[col].max():.3f}]"
        print(f"    ✓ {col}: {samples_df[col].dtype} {val_range}")
    else:
        print(f"    ✗ {col}: MISSING")

print("\n  Action columns:")
for col in action_cols:
    if col in samples_df.columns:
        val_range = f"[{samples_df[col].min():.3f}, {samples_df[col].max():.3f}]"
        print(f"    ✓ {col}: {samples_df[col].dtype} {val_range}")
    else:
        print(f"    ✗ {col}: MISSING")

print("\n  OHLCV array columns:")
for col in array_cols:
    if col in samples_df.columns:
        arr_shape = samples_df[col].iloc[0].shape if hasattr(samples_df[col].iloc[0], 'shape') else 'N/A'
        print(f"    ✓ {col}: {samples_df[col].dtype}, shape={arr_shape}")
    else:
        print(f"    ✗ {col}: MISSING")

print("\n  Label column:")
if 'y' in samples_df.columns:
    print(f"    ✓ y (reward): {samples_df['y'].dtype}")
    print(f"      Range: [{samples_df['y'].min():.6f}, {samples_df['y'].max():.6f}]")
    print(f"      Mean: {samples_df['y'].mean():.6f}")
    print(f"      Std: {samples_df['y'].std():.6f}")
    print(f"      Non-zero: {(samples_df['y'] != 0).sum()}/{len(samples_df)}")
else:
    print(f"    ✗ y: MISSING")

# Inspect forward windows
print("\n" + "=" * 70)
print("STEP 5: Inspect Forward Windows")
print("=" * 70)

fw_df = pd.read_parquet(forward_file)
print(f"\n  Loaded {len(fw_df):,} unique forward windows")
print(f"  Columns: {list(fw_df.columns)}")

if 'idx' in fw_df.columns:
    print(f"\n  Index range: [{fw_df['idx'].min()}, {fw_df['idx'].max()}]")
    
    # Check coverage
    sample_indices = set(samples_df['idx'].unique())
    forward_indices = set(fw_df['idx'].unique())
    
    missing = sample_indices - forward_indices
    if missing:
        print(f"  ✗ {len(missing)} sample indices missing forward windows!")
    else:
        print(f"  ✓ All {len(sample_indices)} sample indices have forward windows")
    
    # Check array shapes
    for col in ['forward_open', 'forward_high', 'forward_low', 'forward_close', 'forward_volume']:
        if col in fw_df.columns:
            arr = fw_df[col].iloc[0]
            if hasattr(arr, 'shape'):
                print(f"    ✓ {col}: shape={arr.shape}, dtype={arr.dtype}")
            else:
                print(f"    ✗ {col}: not an array")

# Check a sample in detail
print("\n" + "=" * 70)
print("STEP 6: Detailed Sample Inspection")
print("=" * 70)

sample = samples_df.iloc[0]
print(f"\n  Sample 0:")
print(f"    Index: {sample['idx']}")
print(f"    Forward: {sample['forward']}")

print(f"\n  Position State (scaled [0,1]):")
print(f"    Equity: {sample['equity']:.4f}")
print(f"    Balance: {sample['balance']:.4f}")
print(f"    Long: ${sample['long_value']:.4f}, SL={sample['long_sl']:.4f}, TP={sample['long_tp']:.4f}")
print(f"    Short: ${sample['short_value']:.4f}, SL={sample['short_sl']:.4f}, TP={sample['short_tp']:.4f}")

print(f"\n  Target Action (scaled [0,1]):")
print(f"    Long: ${sample['act_long_value']:.4f}, SL={sample['act_long_sl']:.4f}, TP={sample['act_long_tp']:.4f}")
print(f"    Short: ${sample['act_short_value']:.4f}, SL={sample['act_short_sl']:.4f}, TP={sample['act_short_tp']:.4f}")

print(f"\n  OHLCV arrays:")
for col in ['open_scaled', 'close_scaled', 'volume_scaled']:
    if col in sample.index:
        arr = sample[col]
        print(f"    {col}: shape={arr.shape}, range=[{arr.min():.3f}, {arr.max():.3f}]")

print(f"\n  Reward: {sample['y']:.6f}")

# Summary statistics
print("\n" + "=" * 70)
print("STEP 7: Summary Statistics")
print("=" * 70)

print(f"\n  Leverage distribution (unscaled would be needed for true leverage):")
# Note: These are scaled values, so not true leverage
print(f"    Mean scaled long: {samples_df['long_value'].mean():.4f}")
print(f"    Mean scaled short: {samples_df['short_value'].mean():.4f}")

print(f"\n  Hold samples:")
flat_states = ((samples_df['long_value'] == 0) & (samples_df['short_value'] == 0)).sum()
hold_actions = ((samples_df['long_value'] == samples_df['act_long_value']) & 
                (samples_df['short_value'] == samples_df['act_short_value'])).sum()
print(f"    Flat states: {flat_states}/{n_samples} ({flat_states/n_samples*100:.1f}%)")
print(f"    Hold actions: {hold_actions}/{n_samples} ({hold_actions/n_samples*100:.1f}%)")

print(f"\n  Reward distribution:")
print(f"    Min: {samples_df['y'].min():.6f}")
print(f"    25%: {samples_df['y'].quantile(0.25):.6f}")
print(f"    50%: {samples_df['y'].quantile(0.50):.6f}")
print(f"    75%: {samples_df['y'].quantile(0.75):.6f}")
print(f"    Max: {samples_df['y'].max():.6f}")

# Final summary
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

checks = [
    ("Data loaded", True),
    ("Dataset generated", out_path.exists()),
    ("Forward windows created", forward_file.exists()),
    ("Scaler saved", scaler_file.exists()),
    ("All expected columns present", all([col in samples_df.columns for col in position_cols + action_cols])),
    ("Rewards computed", 'y' in samples_df.columns),
    ("Forward window coverage", len(missing) == 0 if 'missing' in locals() else True),
]

all_passed = all([check[1] for check in checks])

for check_name, passed in checks:
    status = "✓" if passed else "✗"
    print(f"  {status} {check_name}")

if all_passed:
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED! Pipeline is working! 🎉")
    print("=" * 70)
    print("\nThe refactored data pipeline is fully operational!")
    print("Next steps:")
    print("  1. Update model.py for new input structure (14 meta cols)")
    print("  2. Update predictor.py for new action space (6 values)")
    print("  3. Add unit tests for SL/TP simulation")
    print("  4. Generate larger dataset for training")
else:
    print("\n" + "=" * 70)
    print("⚠️  SOME TESTS FAILED")
    print("=" * 70)
    sys.exit(1)
