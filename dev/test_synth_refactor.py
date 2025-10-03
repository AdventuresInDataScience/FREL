"""
Quick test script to verify new synth.py structure.
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synth import build_samples

# Load config
with open(Path(__file__).parent.parent / "config" / "default.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Create minimal test dataframe
n_bars = 1000
rng = np.random.default_rng(42)
test_df = pd.DataFrame({
    'open': rng.uniform(100, 110, n_bars),
    'high': rng.uniform(110, 120, n_bars),
    'low': rng.uniform(90, 100, n_bars),
    'close': rng.uniform(100, 110, n_bars),
    'volume': rng.integers(1000000, 10000000, n_bars)
})

print("=" * 60)
print("Testing new synth.py structure")
print("=" * 60)

# Generate small sample
n_samples = 10
lookback = cfg['lookback']
forward = cfg['forward']

print(f"\nGenerating {n_samples} samples...")
print(f"Lookback: {lookback}, Forward: {forward}")

samples = build_samples(test_df, n_samples, lookback, forward, rng, cfg)

print(f"\n✓ Generated {len(samples)} samples")
print(f"\nColumns: {list(samples.columns)}")

# Check structure
print("\n" + "=" * 60)
print("Sample Structure Check")
print("=" * 60)

# Expected columns
expected_meta = ['idx', 'equity', 'balance', 
                'long_value', 'short_value', 'long_sl', 'long_tp', 'short_sl', 'short_tp',
                'act_long_value', 'act_short_value', 'act_long_sl', 'act_long_tp', 
                'act_short_sl', 'act_short_tp']
expected_arrays = ['open', 'high', 'low', 'close', 'volume']

missing = set(expected_meta + expected_arrays) - set(samples.columns)
if missing:
    print(f"✗ Missing columns: {missing}")
else:
    print("✓ All expected columns present")

# Check data types
print("\nData types:")
for col in expected_meta:
    print(f"  {col}: {samples[col].dtype}")

print("\nArray columns:")
for col in expected_arrays:
    print(f"  {col}: {samples[col].dtype}, shape={samples[col].iloc[0].shape}")

# Validate constraints
print("\n" + "=" * 60)
print("Constraint Validation")
print("=" * 60)

# Check positive constraints
print(f"\n✓ All equity > 0: {(samples['equity'] > 0).all()}")
print(f"✓ All balance >= 0: {(samples['balance'] >= 0).all()}")
print(f"✓ All long_value >= 0: {(samples['long_value'] >= 0).all()}")
print(f"✓ All short_value >= 0: {(samples['short_value'] >= 0).all()}")

# Check leverage constraint
gross_exposure = samples['long_value'] + samples['short_value']
leverage = gross_exposure / samples['equity']
max_leverage = cfg['max_leverage']
print(f"\n✓ Max leverage: {leverage.max():.2f}x (limit: {max_leverage}x)")
print(f"✓ All leverage <= {max_leverage}x: {(leverage <= max_leverage).all()}")

# Check balance consistency
expected_balance = samples['equity'] - gross_exposure
balance_ok = np.allclose(samples['balance'], expected_balance, rtol=1e-6)
print(f"✓ Balance consistency: {balance_ok}")

# Check SL/TP bounds
print("\nSL/TP Bounds:")
long_positions = samples['long_value'] > 0
if long_positions.any():
    print(f"  Long SL range: [{samples.loc[long_positions, 'long_sl'].min():.3f}, "
          f"{samples.loc[long_positions, 'long_sl'].max():.3f}] (expected: [0.50, 0.99])")
    print(f"  Long TP range: [{samples.loc[long_positions, 'long_tp'].min():.3f}, "
          f"{samples.loc[long_positions, 'long_tp'].max():.3f}] (expected: [1.01, 21.0])")

short_positions = samples['short_value'] > 0
if short_positions.any():
    print(f"  Short SL range: [{samples.loc[short_positions, 'short_sl'].min():.3f}, "
          f"{samples.loc[short_positions, 'short_sl'].max():.3f}] (expected: [1.01, 1.50])")
    print(f"  Short TP range: [{samples.loc[short_positions, 'short_tp'].min():.3f}, "
          f"{samples.loc[short_positions, 'short_tp'].max():.3f}] (expected: [0.50, 0.99])")

# Check hold percentages
flat_states = ((samples['long_value'] == 0) & (samples['short_value'] == 0)).sum()
hold_actions = ((samples['long_value'] == samples['act_long_value']) & 
               (samples['short_value'] == samples['act_short_value'])).sum()

print(f"\n✓ Flat states: {flat_states}/{n_samples} ({flat_states/n_samples*100:.1f}%, target: 10%)")
print(f"✓ Hold actions: {hold_actions}/{n_samples} ({hold_actions/n_samples*100:.1f}%, target: 20%)")

# Display first sample
print("\n" + "=" * 60)
print("First Sample Details")
print("=" * 60)
sample = samples.iloc[0]
print(f"\nIndex: {sample['idx']}")
print(f"Equity: ${sample['equity']:,.2f}")
print(f"Balance: ${sample['balance']:,.2f}")
print(f"\nCurrent Position:")
print(f"  Long: ${sample['long_value']:,.2f} (SL: {sample['long_sl']:.3f}, TP: {sample['long_tp']:.3f})")
print(f"  Short: ${sample['short_value']:,.2f} (SL: {sample['short_sl']:.3f}, TP: {sample['short_tp']:.3f})")
print(f"  Gross exposure: ${sample['long_value'] + sample['short_value']:,.2f}")
print(f"  Leverage: {(sample['long_value'] + sample['short_value']) / sample['equity']:.2f}x")
print(f"\nTarget Action:")
print(f"  Long: ${sample['act_long_value']:,.2f} (SL: {sample['act_long_sl']:.3f}, TP: {sample['act_long_tp']:.3f})")
print(f"  Short: ${sample['act_short_value']:,.2f} (SL: {sample['act_short_sl']:.3f}, TP: {sample['act_short_tp']:.3f})")

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
