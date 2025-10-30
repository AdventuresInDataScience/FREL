#%%
# =============================================================================
# FULL END-TO-END PIPELINE
# Complete workflow from data download to model training and evaluation
# =============================================================================

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
#profiling
script_start = time.time()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src import data, dataset, model, predictor, scale, mapie

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load base config
config_path = project_root / "config" / "default.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

print(f"✓ Config loaded from: {config_path}")

#%%
# =============================================================================
# BLOCK 1: OVERRIDE CONFIG
# User-configurable parameters for the pipeline run
# =============================================================================

# non-cfg variables
pipeline_timestamp = int(time.time() * 1000) % 1000000
seed = 42
train_path = "Data/FREL/df_train.parquet"
test_path = "Data/FREL/df_test.parquet"
raw_df_path = "Data/FREL/df_raw.parquet.gzip"

# Override with pipeline-specific settings
cfg.update({
    # Dataset size - SMALL for testing
    "n_samples": 10000,              # Smallish sample size for profiling
    
    # Data settings
    "ticker": "^GSPC",
    "start": "2010-01-01",         # Recent data for faster download
    
    # Window sizes
    "lookback": 60,                # Past bars
    "forward": 30,                 # Future bars for reward
    
    # Model settings
    "model_type": "transformer",   # Options: transformer, informer, fedformer, patchtst, itransformer, nbeats, nhits, lightgbm
    "d_model": 64,                 # Model dimension (small for speed)
    "nhead": 4,                    # Attention heads
    "tx_blocks": 2,                # Transformer blocks
    "batch_size": 32,              # Training batch size
    "epochs": 10,                  # Training epochs
    "lr": 1e-3,                    # Learning rate
    
    # Training settings
    "train_ratio": 0.80,           # 80% train, 20% test
    
    # Cost parameters (basis points)
    "fee_bps": 10,                 # 0.10% fee
    "slippage_bps": 5,             # 0.05% slippage
    "spread_bps": 2,               # 0.02% spread
    "overnight_bp": 0.5,           # 0.005% overnight
    
    # Reward metric
    "reward_key": "car",           # car | sharpe | sortino | calmar
    
    # Synthetic data generation bounds
    # Account state
    "synth_equity_min": 10000,
    "synth_equity_max": 100000,
    
    # Position values
    "synth_long_value_min": 0,
    "synth_long_value_max": 50000,
    "synth_short_value_min": 0,
    "synth_short_value_max": 50000,
    
    # SL/TP ranges (multiplier notation)
    "synth_long_sl_min": 0.50,     # 50% loss max
    "synth_long_sl_max": 0.99,     # 1% loss min
    "synth_long_tp_min": 1.01,     # 1% profit min
    "synth_long_tp_max": 21.0,     # 2000% profit max
    "synth_short_sl_min": 1.01,    # 1% loss min
    "synth_short_sl_max": 1.50,    # 50% loss max
    "synth_short_tp_min": 0.50,    # 50% profit max
    "synth_short_tp_max": 0.99,    # 1% profit min
    
    # Distribution parameters
    "position_value_mean": 10000,
    "position_value_sigma": 1.0,
    "tp_sl_mean": 0.05,
    "tp_sl_sigma": 0.03,
    
    # Hold percentages
    "hold_state_pct": 0.10,        # 10% samples with no positions
    "hold_action_pct": 0.20,       # 20% samples with no action
    
    # File naming with timestamp
    "raw_data_filename": f"test_raw_{{ticker}}_{pipeline_timestamp}.parquet",
    "samples_filename": f"test_samples_{pipeline_timestamp}.parquet",
    "scaler_filename": f"test_meta_scaler_{pipeline_timestamp}.json",
    "forward_windows_filename": f"test_forward_windows_{pipeline_timestamp}.parquet",
    
    # Data directory
    "data_dir": "Data/FREL",  # Set your desired data directory

    # Performance
    "n_jobs": -1,                   # Use all available CPU cores
})

#%% 
# =============================================================================
# BLOCK 2: DATA PREPARATION
# Load and Prepare Data.
# =============================================================================

data_start = time.time()

#a) Generate synthetic state space data for test set
samples_path = dataset.build_dataset(
        cfg=cfg,
        n_samples=cfg['n_samples'],
        seed=seed,
        overwrite=False,
        n_jobs=1  # Single-threaded for testing
    )
#b) Load the generated samples
df = pd.read_parquet(samples_path)
# eg #df2 = pd.read_parquet('Data/FREL/test_samples_658166.parquet')
data_end = time.time()
print(f"Data preparation time: {data_end - data_start:.2f} seconds")
# Df columns are:
# Index(['idx', 'equity', 'balance', 'long_value', 'short_value', 'long_sl',
#        'long_tp', 'short_sl', 'short_tp', 'act_long_value', 'act_short_value',
#        'act_long_sl', 'act_long_tp', 'act_short_sl', 'act_short_tp', 'phase',
#        'open_scaled', 'high_scaled', 'low_scaled', 'close_scaled',
#        'volume_scaled', 'forward', 'y'],
#       dtype='object')
#%% 
# =============================================================================
# BLOCK 3: MODEL TRAINING
# =============================================================================

training_start = time.time()


frel = model.build_model(cfg, device='cuda')
mapie_model = mapie.MapiePredictor(
                model=frel,
                method="plus",
                cv=3,
                n_jobs=-1,
                random_state=42
            )

mapie_model.fit(df)

training_end = time.time()
print(f"Model training time: {training_end - training_start:.2f} seconds")
# %%
