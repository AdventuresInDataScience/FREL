#!/usr/bin/env python3
"""
Test just CAR individually with timing debug.
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

def test_car_only():
    """Test just CAR metric with debug timing."""
    
    print(f"\n🔍 Testing CAR metric with 1,000 samples (with timing debug)...")
    
    # Load config
    config_path = Path('config/default.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Test config
    cfg.update({
        'n_samples_test': 1000,
        'lookback': 60,
        'forward': 30,
        'ticker': '^GSPC',
        'start': '2020-01-01',
        'end': '2021-01-01',
        'reward_key': 'car'  # Single metric
    })
    
    # Create temp dir
    temp_dir = Path(tempfile.mkdtemp(prefix='test_car_debug_'))
    cfg['data_dir'] = str(temp_dir)
    cfg['raw_filename'] = 'test_raw_car.parquet'
    cfg['samples_filename'] = 'test_samples_car.parquet'
    cfg['forward_windows_filename'] = 'test_fw_car.parquet'
    
    try:
        # Time the computation
        t0 = time.perf_counter()
        
        output_path = dataset.build_dataset(
            cfg=cfg,
            n_samples=1000,
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
            
            print(f"  ✅ CAR: {t1-t0:.2f}s")
            print(f"     Valid rewards: {n_valid:,}/{len(samples):,} ({100*n_valid/len(samples):.1f}%)")
            print(f"     Range: [{reward_range[0]:.4f}, {reward_range[1]:.4f}]")
            print(f"     Speed: {n_valid/(t1-t0):.0f} samples/sec")
        else:
            print(f"  ❌ CAR: No reward column found")
            
    except Exception as e:
        print(f"  ❌ CAR: Error - {e}")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_car_only()