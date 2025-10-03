"""
Quick performance test for dataset generation optimizations.
Run this to verify the speedup from vectorization.
"""
import time
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import dataset

# Load config
config_path = Path(__file__).parent.parent / "config" / "default.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# Test config
cfg.update({
    "ticker": "^GSPC",
    "start": "2020-01-01",
    "lookback": 50,
    "forward": 20,
})

print("="*70)
print("PERFORMANCE TEST: dataset.build_dataset()")
print("="*70)

# Test different sample sizes
test_sizes = [1000, 10000, 100000, 1000000]

for n in test_sizes:
    print(f"\n{'='*70}")
    print(f"Testing {n:,} samples...")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    output_path = dataset.build_dataset(
        cfg=cfg,
        n_samples=n,
        seed=42,
        overwrite=(n == test_sizes[0])  # Only redownload on first iteration
    )
    
    elapsed = time.time() - start_time
    samples_per_sec = n / elapsed
    
    print(f"\n✓ Completed in {elapsed:.2f} seconds")
    print(f"  → {samples_per_sec:,.0f} samples/second")
    print(f"  → {elapsed * 1000 / n:.2f} ms per sample")
    
    # Cleanup
    output_path.unlink()
    
    # Estimate time for larger datasets
    if n == 10000:
        est_100k = elapsed * 10
        est_1m = elapsed * 100
        print(f"\n  Estimated time for 100K samples: {est_100k:.1f}s")
        print(f"  Estimated time for 1M samples: {est_1m:.1f}s ({est_1m/60:.1f} minutes)")

print(f"\n{'='*70}")
print("PERFORMANCE TEST COMPLETE")
print("="*70)
print("\nExpected results with optimizations:")
print("  - 1K samples: <1 second")
print("  - 10K samples: ~2-3 seconds")
print("  - 100K samples: ~10-15 seconds")
print("  - 1M samples: ~20-30 seconds (was 5 minutes before)")
print("="*70)
