#!/usr/bin/env python3
"""
Quick test of reward computation speed optimizations.

Tests all three implementations:
1. Original (parallel)
2. Vectorized 
3. Ultra-fast (JIT batch)

Expected results:
- Original: ~250s for 1K samples  
- Vectorized: ~50s for 1K samples (5x speedup)
- Ultra-fast: ~5s for 1K samples (50x speedup)
"""

import numpy as np
import pandas as pd
import time
import tempfile
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import dataset, data, synth, scale, forward_windows, reward
import yaml

def create_test_data(n_samples=100):
    """Create minimal test data for reward computation."""
    
    # Load config
    config_path = Path(__file__).parent / "config" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Override with test values
    cfg.update({
        "n_samples_test": n_samples,
        "lookback": 60,
        "forward": 30,
        "ticker": "^GSPC",
        "start": "2020-01-01",
        "end": "2021-01-01",
        "fee_bps": 10,
        "slippage_bps": 5,
        "spread_bps": 2,
        "overnight_bps": 0.5,
        "risk_free_rate": 0.02,
    })
    
    print(f"Creating test data for {n_samples:,} samples...")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="reward_speed_test_"))
    cfg["raw_filename"] = f"test_raw_GSPC_{int(time.time())}.parquet"
    cfg["samples_filename"] = f"test_samples_{int(time.time())}.parquet"
    cfg["forward_windows_filename"] = f"test_fw_{int(time.time())}.parquet"
    cfg["data_dir"] = str(temp_dir)
    
    # Build minimal dataset (fast data generation)
    print("  Building dataset...")
    output_path = dataset.build_dataset(
        cfg=cfg,
        n_samples=n_samples,
        seed=42,
        overwrite=True,
        n_jobs=1  # Sequential for consistency
    )
    
    # Load required components
    samples = pd.read_parquet(output_path)
    df_close_path = temp_dir / cfg["raw_filename"]
    df_close_full = pd.read_parquet(df_close_path)
    df_close = df_close_full['close'].values
    
    # Load forward windows
    fw_path = temp_dir / cfg["forward_windows_filename"]
    forward_lookup = forward_windows.load_forward_lookup(fw_path)
    
    # Load scaler
    scaler_path = temp_dir / "meta_scaler.json"
    scaler = scale.MetaScaler.load(scaler_path)
    
    print(f"  ✓ Test data ready: {len(samples):,} samples")
    return samples, df_close, forward_lookup, scaler, cfg, temp_dir

def benchmark_implementations(samples, df_close, forward_lookup, scaler, cfg):
    """Benchmark all three reward computation implementations."""
    
    print(f"\n{'='*60}")
    print(f"BENCHMARKING REWARD COMPUTATION ({len(samples):,} samples)")
    print(f"{'='*60}")
    
    # Common parameters
    kwargs = {
        'df_close': df_close,
        'samples': samples,
        'forward_lookup': forward_lookup,
        'scaler': scaler,
        'reward_key': 'car',
        'fee_bp': cfg.get('fee_bps', 10),
        'slip_bp': cfg.get('slippage_bps', 5),
        'spread_bp': cfg.get('spread_bps', 2),
        'night_bp': cfg.get('overnight_bps', 0.5),
        'risk_free_rate': cfg.get('risk_free_rate', 0.02),
    }
    
    results = {}
    
    # Test 1: Original implementation
    print(f"\n[1] Testing ORIGINAL implementation...")
    t0 = time.perf_counter()
    try:
        rewards_orig = reward.compute_many_original(**kwargs)
        t1 = time.perf_counter()
        results['original'] = {
            'time': t1 - t0,
            'rewards': rewards_orig,
            'status': 'success'
        }
        print(f"  ✓ Original: {t1-t0:.2f}s")
        print(f"    Sample rewards: {rewards_orig['y'].values[:3]}")
    except Exception as e:
        results['original'] = {'time': 999, 'status': f'error: {e}'}
        print(f"  ✗ Original failed: {e}")
    
    # Test 2: Vectorized implementation  
    print(f"\n[2] Testing VECTORIZED implementation...")
    t0 = time.perf_counter()
    try:
        rewards_vec = reward.compute_many_vectorized(**kwargs)
        t1 = time.perf_counter()
        results['vectorized'] = {
            'time': t1 - t0,
            'rewards': rewards_vec,
            'status': 'success'
        }
        print(f"  ✓ Vectorized: {t1-t0:.2f}s")
        print(f"    Sample rewards: {rewards_vec['y'].values[:3]}")
        
        # Check consistency with original
        if 'original' in results and results['original']['status'] == 'success':
            orig_vals = results['original']['rewards']['y'].values
            vec_vals = rewards_vec['y'].values
            max_diff = np.max(np.abs(orig_vals - vec_vals))
            print(f"    Max difference vs original: {max_diff:.6f}")
            
    except Exception as e:
        results['vectorized'] = {'time': 999, 'status': f'error: {e}'}
        print(f"  ✗ Vectorized failed: {e}")
    
    # Test 3: Ultra-fast implementation (if samples >= 1000)
    if len(samples) >= 1000:
        print(f"\n[3] Testing ULTRA-FAST implementation...")
        t0 = time.perf_counter()
        try:
            rewards_ultra = reward.compute_many_ultra_fast(**kwargs)
            t1 = time.perf_counter()
            results['ultra_fast'] = {
                'time': t1 - t0,
                'rewards': rewards_ultra,
                'status': 'success'
            }
            print(f"  ✓ Ultra-fast: {t1-t0:.2f}s")
            print(f"    Sample rewards: {rewards_ultra['y'].values[:3]}")
            
            # Check consistency with original
            if 'original' in results and results['original']['status'] == 'success':
                orig_vals = results['original']['rewards']['y'].values
                ultra_vals = rewards_ultra['y'].values
                max_diff = np.max(np.abs(orig_vals - ultra_vals))
                print(f"    Max difference vs original: {max_diff:.6f}")
                
        except Exception as e:
            results['ultra_fast'] = {'time': 999, 'status': f'error: {e}'}
            print(f"  ✗ Ultra-fast failed: {e}")
    else:
        print(f"\n[3] Skipping ultra-fast (requires >=1K samples, got {len(samples)})")
    
    # Test 4: Auto-selection (compute_many)
    print(f"\n[4] Testing AUTO-SELECTION (compute_many)...")
    t0 = time.perf_counter()
    try:
        rewards_auto = reward.compute_many(**kwargs)
        t1 = time.perf_counter()
        results['auto'] = {
            'time': t1 - t0,
            'rewards': rewards_auto,
            'status': 'success'
        }
        print(f"  ✓ Auto-selection: {t1-t0:.2f}s")
        print(f"    Sample rewards: {rewards_auto['y'].values[:3]}")
        
    except Exception as e:
        results['auto'] = {'time': 999, 'status': f'error: {e}'}
        print(f"  ✗ Auto-selection failed: {e}")
    
    return results

def print_summary(results, n_samples):
    """Print benchmark summary."""
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY ({n_samples:,} samples)")
    print(f"{'='*60}")
    
    # Extract timings
    times = {}
    for impl, data in results.items():
        if data['status'] == 'success':
            times[impl] = data['time']
    
    if not times:
        print("❌ No implementations succeeded!")
        return
    
    # Find baseline (original or first successful)
    baseline_time = times.get('original', min(times.values()))
    baseline_name = 'original' if 'original' in times else list(times.keys())[0]
    
    print(f"\nTiming Results:")
    print(f"{'Implementation':<15} {'Time (s)':<10} {'Speedup':<10} {'Status'}")
    print(f"{'-'*50}")
    
    for impl, data in results.items():
        if data['status'] == 'success':
            time_val = data['time']
            speedup = baseline_time / time_val if time_val > 0 else float('inf')
            print(f"{impl:<15} {time_val:>8.2f}s {speedup:>8.1f}x {'✓'}")
        else:
            print(f"{impl:<15} {'--':>8} {'--':>8} {'✗ ' + data['status']}")
    
    # Extrapolation to 40M samples
    if times:
        fastest_time = min(times.values())
        fastest_impl = [k for k, v in times.items() if v == fastest_time][0]
        
        # Extrapolate to 40M (assuming linear scaling)
        scale_factor = 40_000_000 / n_samples
        extrapolated_time = fastest_time * scale_factor
        
        print(f"\n📈 Extrapolation to 40M samples:")
        print(f"  Best implementation: {fastest_impl}")
        print(f"  Estimated time: {extrapolated_time:.1f}s ({extrapolated_time/3600:.1f} hours)")
        
        if extrapolated_time < 3600:  # Less than 1 hour
            print(f"  🎯 FEASIBLE for 40M samples!")
        elif extrapolated_time < 24*3600:  # Less than 1 day
            print(f"  ⚠️  Marginal for 40M samples (long but doable)")
        else:
            print(f"  ❌ Still too slow for 40M samples")

def main():
    """Run speed benchmarks."""
    
    print("🚀 REWARD COMPUTATION SPEED BENCHMARK")
    print("=" * 60)
    
    # Test different sample sizes
    test_sizes = [100, 1000]  # Start with smaller tests
    
    for n_samples in test_sizes:
        print(f"\n🔥 Testing with {n_samples:,} samples...")
        
        try:
            # Create test data
            samples, df_close, forward_lookup, scaler, cfg, temp_dir = create_test_data(n_samples)
            
            # Run benchmarks
            results = benchmark_implementations(samples, df_close, forward_lookup, scaler, cfg)
            
            # Print summary
            print_summary(results, n_samples)
            
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            print(f"❌ Test failed for {n_samples:,} samples: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ Benchmark complete!")

if __name__ == "__main__":
    main()