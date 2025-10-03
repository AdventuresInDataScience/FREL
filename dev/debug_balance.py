"""
Debug balance issue.
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synth import build_samples

# Load config
with open(Path(__file__).parent.parent / "config" / "default.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Create test data
n_bars = 1000
rng = np.random.default_rng(42)
test_df = pd.DataFrame({
    'open': rng.uniform(100, 110, n_bars),
    'high': rng.uniform(110, 120, n_bars),
    'low': rng.uniform(90, 100, n_bars),
    'close': rng.uniform(100, 110, n_bars),
    'volume': rng.integers(1000000, 10000000, n_bars)
})

samples = build_samples(test_df, 10, cfg['lookback'], cfg['forward'], rng, cfg)

# Check balance
print("Balance values:")
print(samples['balance'])
print(f"\nMin balance: {samples['balance'].min()}")
print(f"Negative balances: {(samples['balance'] < 0).sum()}")

# Show samples with negative balance
negative = samples[samples['balance'] < 0]
if len(negative) > 0:
    print(f"\nSamples with negative balance:")
    for idx, row in negative.iterrows():
        print(f"\nSample {idx}:")
        print(f"  Equity: {row['equity']:.2f}")
        print(f"  Long: {row['long_value']:.2f}")
        print(f"  Short: {row['short_value']:.2f}")
        print(f"  Gross: {row['long_value'] + row['short_value']:.2f}")
        print(f"  Balance: {row['balance']:.2f}")
        print(f"  Leverage: {(row['long_value'] + row['short_value']) / row['equity']:.2f}x")
