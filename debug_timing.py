#!/usr/bin/env python3
"""
Debug CAR timing issue.
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

def debug_car_timing():
    """Debug why individual CAR is slow."""
    
    print(f"\n🔍 Debugging CAR timing with 1000 samples...")
    
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
    temp_dir = Path(tempfile.mkdtemp(prefix='debug_car_'))
    cfg['data_dir'] = str(temp_dir)
    cfg['raw_filename'] = 'debug_raw.parquet'
    cfg['samples_filename'] = 'debug_samples.parquet'
    cfg['forward_windows_filename'] = 'debug_fw.parquet'
    
    try:
        # Time just the reward computation part
        print("Running dataset build...")
        t0 = time.perf_counter()
        
        output_path = dataset.build_dataset(
            cfg=cfg,
            n_samples=1000,
            seed=42,
            overwrite=True,
            n_jobs=1
        )
        
        t1 = time.perf_counter()
        print(f"Total time: {t1-t0:.2f}s")
        
        # Load and check results
        samples = pd.read_parquet(output_path)
        print(f"Loaded {len(samples)} samples with columns: {list(samples.columns)}")
        
        if 'y' in samples.columns:
            rewards = samples['y'].values
            n_valid = np.sum(~np.isnan(rewards) & ~np.isinf(rewards))
            reward_range = (np.nanmin(rewards), np.nanmax(rewards))
            print(f"Valid rewards: {n_valid}/{len(samples)} ({100*n_valid/len(samples):.1f}%)")
            print(f"Range: [{reward_range[0]:.4f}, {reward_range[1]:.4f}]")
        
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    debug_car_timing()