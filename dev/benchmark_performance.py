# %% Imports
import time
import yaml
import numpy as np
from pathlib import Path
from src import data, dataset

# %% Load configuration
config_path = Path("config/default.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

print("Configuration loaded:")
print(f"  n_jobs: {cfg['performance']['n_jobs']}")

# %% Load market data
print("\nLoading market data...")
df = data.load(cfg, "^GSPC")
print(f"  Loaded {len(df):,} rows of S&P 500 data")
print(f"  Date range: {df.index[0]} to {df.index[-1]}")

# %% Test configurations
test_sizes = [
    100_000,    # 100K - below parallel threshold
    1_000_000,  # 1M - first parallel test
    10_000_000, # 10M - medium scale
]

test_jobs = [1, -1]  # Single-core vs all cores

print("\n" + "="*70)
print("PERFORMANCE BENCHMARK")
print("="*70)

results = []

# %% Run benchmarks
for n in test_sizes:
    print(f"\n{'─'*70}")
    print(f"Testing {n:,} samples:")
    print(f"{'─'*70}")
    
    for jobs in test_jobs:
        # Update config
        cfg["performance"]["n_jobs"] = jobs
        
        # Time the generation
        print(f"  Running with n_jobs={jobs}...", end=" ", flush=True)
        start = time.perf_counter()
        
        samples = dataset.build_dataset(
            df, 
            n, 
            cfg, 
            rng=np.random.default_rng(42),
            overwrite=True
        )
        
        elapsed = time.perf_counter() - start
        samples_per_sec = n / elapsed
        
        results.append({
            "samples": n,
            "n_jobs": "all" if jobs == -1 else jobs,
            "time_sec": elapsed,
            "samples_per_sec": samples_per_sec
        })
        
        print(f"{elapsed:.2f}s ({samples_per_sec:,.0f} samples/sec)")
        
        # Calculate speedup for parallel vs single-core
        if jobs == -1 and len(results) >= 2:
            single_core_time = results[-2]["time_sec"]
            speedup = single_core_time / elapsed
            print(f"    → Speedup: {speedup:.2f}x")

# %% Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n{'Samples':<15} {'Mode':<10} {'Time (s)':<12} {'Throughput':<20}")
print("─"*70)

for r in results:
    mode = "Single-core" if r["n_jobs"] == 1 else "Multi-core"
    print(f"{r['samples']:>13,}  {mode:<10} {r['time_sec']:>10.2f}  {r['samples_per_sec']:>18,.0f}/s")

# %% Extrapolations for target sizes
print("\n" + "="*70)
print("EXTRAPOLATED PERFORMANCE FOR TARGET DATASETS")
print("="*70)

# Use 10M multi-core result for extrapolation
multi_core_results = [r for r in results if r["n_jobs"] == "all"]
if multi_core_results:
    largest_test = max(multi_core_results, key=lambda x: x["samples"])
    throughput = largest_test["samples_per_sec"]
    
    target_sizes = [40_000_000, 320_000_000]  # 40M and 320M
    
    print(f"\nBased on {largest_test['samples']:,} sample test:")
    print(f"  Measured throughput: {throughput:,.0f} samples/sec")
    print(f"\nEstimated times:")
    
    for target in target_sizes:
        estimated_sec = target / throughput
        estimated_min = estimated_sec / 60
        
        if estimated_min < 60:
            time_str = f"{estimated_min:.1f} minutes"
        else:
            estimated_hrs = estimated_min / 60
            time_str = f"{estimated_hrs:.1f} hours ({estimated_min:.0f} min)"
        
        print(f"  {target:>11,} samples: {time_str}")

print("\n" + "="*70)
print("Benchmark complete!")
print("="*70)
