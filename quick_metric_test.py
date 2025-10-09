#!/usr/bin/env python3
"""
Quick test of all reward metrics with ultra-fast implementation.

Tests each metric individually to ensure they all work correctly.
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import yaml
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import dataset
from src.reward import warm_up_jit_functions

def test_individual_metric(metric_name, n_samples=1000):
    """Test a single reward metric."""
    
    print(f"\n🧪 Testing {metric_name.upper()} metric with {n_samples:,} samples...")
    
    # Load config
    config_path = Path('config/default.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Test config
    cfg.update({
        'n_samples_test': n_samples,
        'lookback': 60,
        'forward': 30,
        'ticker': '^GSPC',
        'start': '2020-01-01',
        'end': '2021-01-01',
        'reward_key': metric_name  # Single metric
    })
    
    # Create temp dir
    temp_dir = Path(tempfile.mkdtemp(prefix=f'test_{metric_name}_'))
    cfg['data_dir'] = str(temp_dir)
    cfg['raw_filename'] = f'test_raw_{metric_name}.parquet'
    cfg['samples_filename'] = f'test_samples_{metric_name}.parquet'
    cfg['forward_windows_filename'] = f'test_fw_{metric_name}.parquet'
    
    try:
        # Time the computation
        t0 = time.perf_counter()
        
        output_path = dataset.build_dataset(
            cfg=cfg,
            n_samples=n_samples,
            seed=42,
            overwrite=True,
            n_jobs=1
        )
        
        t1 = time.perf_counter()
        
        # Load and verify
        samples = pd.read_parquet(output_path)
        
        # Check reward column
        if 'y' in samples.columns:
            rewards = samples['y'].values
            n_valid = np.sum(~np.isnan(rewards) & ~np.isinf(rewards))
            reward_range = (np.nanmin(rewards), np.nanmax(rewards))
            
            print(f"  ✅ {metric_name.upper()}: {t1-t0:.2f}s")
            print(f"     Valid rewards: {n_valid:,}/{len(samples):,} ({100*n_valid/len(samples):.1f}%)")
            print(f"     Range: [{reward_range[0]:.4f}, {reward_range[1]:.4f}]")
            print(f"     Speed: {n_samples/(t1-t0):.0f} samples/sec")
            
            return {
                'success': True,
                'time': t1 - t0,
                'valid_ratio': n_valid / len(samples),
                'range': reward_range,
                'samples_per_sec': n_samples / (t1 - t0)
            }
        else:
            print(f"  ❌ {metric_name.upper()}: No reward column found")
            return {'success': False, 'error': 'No reward column'}
            
    except Exception as e:
        print(f"  ❌ {metric_name.upper()}: Error - {e}")
        return {'success': False, 'error': str(e)}
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_multi_metrics(n_samples=1000):
    """Test all metrics together."""
    
    print(f"\n🧪 Testing ALL METRICS together with {n_samples:,} samples...")
    
    # Load config
    config_path = Path('config/default.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Test config
    cfg.update({
        'n_samples_test': n_samples,
        'lookback': 60,
        'forward': 30,
        'ticker': '^GSPC',
        'start': '2020-01-01',
        'end': '2021-01-01',
        'reward_key': ['car', 'sharpe', 'sortino', 'calmar']  # All metrics
    })
    
    # Create temp dir
    temp_dir = Path(tempfile.mkdtemp(prefix='test_multi_'))
    cfg['data_dir'] = str(temp_dir)
    cfg['raw_filename'] = 'test_raw_multi.parquet'
    cfg['samples_filename'] = 'test_samples_multi.parquet'
    cfg['forward_windows_filename'] = 'test_fw_multi.parquet'
    
    try:
        # Time the computation
        t0 = time.perf_counter()
        
        output_path = dataset.build_dataset(
            cfg=cfg,
            n_samples=n_samples,
            seed=42,
            overwrite=True,
            n_jobs=1
        )
        
        t1 = time.perf_counter()
        
        # Load and verify
        samples = pd.read_parquet(output_path)
        
        # Check all reward columns
        expected_cols = ['y_car', 'y_sharpe', 'y_sortino', 'y_calmar']
        found_cols = [col for col in expected_cols if col in samples.columns]
        
        print(f"  ✅ ALL METRICS: {t1-t0:.2f}s")
        print(f"     Found columns: {found_cols}")
        print(f"     Speed: {n_samples/(t1-t0):.0f} samples/sec")
        
        # Verify each metric
        results = {}
        for col in found_cols:
            rewards = samples[col].values
            n_valid = np.sum(~np.isnan(rewards) & ~np.isinf(rewards))
            reward_range = (np.nanmin(rewards), np.nanmax(rewards))
            metric_name = col.replace('y_', '')
            
            print(f"     {metric_name.upper()}: {n_valid:,} valid, range [{reward_range[0]:.4f}, {reward_range[1]:.4f}]")
            results[metric_name] = {
                'valid_ratio': n_valid / len(samples),
                'range': reward_range
            }
        
        return {
            'success': True,
            'time': t1 - t0,
            'metrics': results,
            'samples_per_sec': n_samples / (t1 - t0)
        }
        
    except Exception as e:
        print(f"  ❌ ALL METRICS: Error - {e}")
        return {'success': False, 'error': str(e)}
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    """Run quick metric validation tests."""
    
    print("🚀 QUICK REWARD METRIC VALIDATION")
    print("=" * 50)
    
    # Warm up JIT functions first
    print("\n🔥 Warming up JIT functions...")
    warm_up_jit_functions()
    
    # Test individual metrics
    individual_metrics = ['car', 'sharpe', 'sortino', 'calmar']
    individual_results = {}
    
    for metric in individual_metrics:
        result = test_individual_metric(metric, 1000)
        individual_results[metric] = result
    
    # Test multi-metric
    multi_result = test_multi_metrics(1000)
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("=" * 50)
    
    # Individual results
    print(f"\nIndividual Metrics (1K samples):")
    print(f"{'Metric':<10} {'Status':<10} {'Time':<8} {'Speed':<12}")
    print("-" * 45)
    
    for metric in individual_metrics:
        result = individual_results[metric]
        if result['success']:
            status = "✅ PASS"
            time_str = f"{result['time']:.2f}s"
            speed_str = f"{result['samples_per_sec']:.0f}/s"
        else:
            status = "❌ FAIL"
            time_str = "N/A"
            speed_str = "N/A"
        
        print(f"{metric.upper():<10} {status:<10} {time_str:<8} {speed_str:<12}")
    
    # Multi-metric result
    print(f"\nMulti-Metric (1K samples):")
    if multi_result['success']:
        print(f"✅ ALL 4 METRICS: {multi_result['time']:.2f}s, {multi_result['samples_per_sec']:.0f} samples/s")
    else:
        print(f"❌ ALL 4 METRICS: Failed - {multi_result['error']}")
    
    # Check if all tests passed
    all_individual_passed = all(r['success'] for r in individual_results.values())
    multi_passed = multi_result['success']
    
    if all_individual_passed and multi_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Ultra-fast implementation working for all reward metrics")
        
        # Quick 40M projection
        if multi_result['success']:
            time_40m = 40_000_000 / multi_result['samples_per_sec'] / 3600
            print(f"📈 40M sample projection: {time_40m:.1f} hours (all 4 metrics)")
    else:
        print(f"\n❌ SOME TESTS FAILED")
        failed_metrics = [m for m, r in individual_results.items() if not r['success']]
        if failed_metrics:
            print(f"Failed individual metrics: {failed_metrics}")
        if not multi_passed:
            print(f"Multi-metric test failed")

if __name__ == "__main__":
    main()