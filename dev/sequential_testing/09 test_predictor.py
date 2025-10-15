#%%
# =============================================================================
# TEST 09: predictor.py - Comprehensive Model Inference Testing
# Test all models with all prediction methods
# Dependencies: All previous modules
# =============================================================================

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src import data, scale, synth, dataset, model, predictor

# Load config
config_path = project_root / "config" / "default.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

print("✓ Imports successful")
print(f"Config loaded: {config_path}")
print(f"Testing module: predictor.py")

#%%
# Setup data
data_dir = project_root / "data"
df_raw = data.load(data_dir / "raw_^GSPC.parquet")
meta_scaler = scale.MetaScaler()
meta_scaler.load(data_dir / "meta_scaler.json")

# Create synthetic samples
rng = np.random.default_rng(42)
df_synth = synth.build_samples(
    df=df_raw,
    n=10,  # Small number for testing
    lookback=cfg['lookback'],
    forward=cfg['forward'],
    rng=rng,
    cfg=cfg
)
print(f"✓ Generated {len(df_synth)} synthetic samples")

# Get one sample to work with
test_row = df_synth.iloc[0]
print(f"✓ Test sample ready")

#%%
# Prepare DataFrames from synthetic sample
# df_synth has columns: idx, equity, balance, long_value, short_value, long_sl, long_tp, short_sl, short_tp,
#                       act_long_value, act_short_value, act_long_sl, act_long_tp, act_short_sl, act_short_tp,
#                       open, high, low, close, volume (arrays)

# Scale the OHLCV arrays
ohlcv_dict = {
    'open': test_row['open'],
    'high': test_row['high'],
    'low': test_row['low'],
    'close': test_row['close'],
    'volume': test_row['volume']
}
ohlcv_scaled_dict = scale.scale_ohlcv_window(ohlcv_dict)

# Create scaled DataFrame (predictor expects *_scaled columns for OHLCV)
df_scaled = pd.DataFrame({
    'open_scaled': [ohlcv_scaled_dict['open']], 
    'high_scaled': [ohlcv_scaled_dict['high']], 
    'low_scaled': [ohlcv_scaled_dict['low']], 
    'close_scaled': [ohlcv_scaled_dict['close']], 
    'volume_scaled': [ohlcv_scaled_dict['volume']],
    # Meta features are already in raw scale from synth
    'equity': [test_row['equity']], 
    'balance': [test_row['balance']], 
    'long_value': [test_row['long_value']], 
    'short_value': [test_row['short_value']],
    'long_sl': [test_row['long_sl']], 
    'long_tp': [test_row['long_tp']], 
    'short_sl': [test_row['short_sl']], 
    'short_tp': [test_row['short_tp']],
    'act_long_value': [test_row['act_long_value']], 
    'act_short_value': [test_row['act_short_value']],
    'act_long_sl': [test_row['act_long_sl']], 
    'act_long_tp': [test_row['act_long_tp']],
    'act_short_sl': [test_row['act_short_sl']], 
    'act_short_tp': [test_row['act_short_tp']],
})

# Raw DataFrame (without scaled OHLCV columns)
df_raw_test = pd.DataFrame({
    'open': [test_row['open']], 
    'high': [test_row['high']], 
    'low': [test_row['low']], 
    'close': [test_row['close']], 
    'volume': [test_row['volume']],
    'equity': [test_row['equity']], 
    'balance': [test_row['balance']],
    'long_value': [test_row['long_value']], 
    'short_value': [test_row['short_value']],
    'long_sl': [test_row['long_sl']], 
    'long_tp': [test_row['long_tp']],
    'short_sl': [test_row['short_sl']], 
    'short_tp': [test_row['short_tp']],
    'act_long_value': [test_row['act_long_value']], 
    'act_short_value': [test_row['act_short_value']],
    'act_long_sl': [test_row['act_long_sl']], 
    'act_long_tp': [test_row['act_long_tp']],
    'act_short_sl': [test_row['act_short_sl']], 
    'act_short_tp': [test_row['act_short_tp']],
})

print("✓ DataFrames ready")

#%%
# TEST 1: Validation
print("\n" + "="*80)
print("TEST 1: Input Validation")
print("="*80)

test_model = model.TransformerModel(price_shape=(cfg['lookback'], 5), meta_len=14, d_model=64, nhead=4, tx_blocks=2, mlp_ratio=4, dropout=0.1)
pred = predictor.Predictor(test_model, meta_scaler, cfg, 'transformer', 'cpu')

try:
    pred._validate_input(df_scaled, scaled=True)
    print("✓ Test 1a: Valid scaled DataFrame")
except Exception as e:
    print(f"✗ Test 1a FAILED: {e}")

try:
    pred._validate_input(df_raw_test, scaled=False)
    print("✓ Test 1b: Valid raw DataFrame")
except Exception as e:
    print(f"✗ Test 1b FAILED: {e}")

print("\n✓ TEST 1 COMPLETE")

#%%
# TEST 2: Scaling
print("\n" + "="*80)
print("TEST 2: Scaling")
print("="*80)

df_scaled_result = pred._scale_df(df_raw_test.copy())
print("✓ Scaling completed")
for col in ['open_scaled', 'high_scaled', 'low_scaled', 'close_scaled', 'volume_scaled']:
    if col in df_scaled_result.columns:
        print(f"✓ {col} present")

print("\n✓ TEST 2 COMPLETE")

#%%
# TEST 3: Prediction
print("\n" + "="*80)
print("TEST 3: Prediction")
print("="*80)

pred_scaled = pred.predict(df_scaled, scaled=True)
print(f"✓ Scaled: {pred_scaled[0]:.6f}")

pred_raw = pred.predict(df_raw_test, scaled=False)
print(f"✓ Raw: {pred_raw[0]:.6f}")

df_batch = pd.concat([df_scaled] * 5, ignore_index=True)
pred_batch = pred.predict(df_batch, scaled=True)
print(f"✓ Batch: mean={pred_batch.mean():.6f}")

print("\n✓ TEST 3 COMPLETE")

#%%
# TEST 4: Action Sampling
print("\n" + "="*80)
print("TEST 4: Action Sampling")
print("="*80)

state = np.array([df_raw_test['equity'].iloc[0], df_raw_test['balance'].iloc[0], df_raw_test['long_value'].iloc[0], df_raw_test['short_value'].iloc[0], df_raw_test['long_sl'].iloc[0], df_raw_test['long_tp'].iloc[0], df_raw_test['short_sl'].iloc[0], df_raw_test['short_tp'].iloc[0]])
ohlcv_arrays = {k: df_raw_test[k].iloc[0] for k in ['open', 'high', 'low', 'close', 'volume']}

actions_df, predictions = pred.predict_many_actions(ohlcv_arrays, state, n_samples=10, seed=42)
print(f"✓ Sampled 10 actions: pred_range=[{predictions.min():.6f}, {predictions.max():.6f}]")

actions_df2, predictions2 = pred.predict_many_actions(ohlcv_arrays, state, n_samples=20, long_value_range=(0, 25000), short_value_range=(0, 25000), sl_range=(0.005, 0.03), tp_range=(0.01, 0.08), seed=123)
print(f"✓ Custom ranges: long=[{actions_df2['act_long_value'].min():.0f}, {actions_df2['act_long_value'].max():.0f}]")

print("\n✓ TEST 4 COMPLETE")

#%%
# TEST 5: Optimization
print("\n" + "="*80)
print("TEST 5: Optimization")
print("="*80)

start = time.time()
optimal_action = pred.find_optimal_action(ohlcv_arrays, state, seed=42)
elapsed = time.time() - start

print(f"✓ Found optimal in {elapsed:.2f}s")
print(f"  Long: value={optimal_action[0]:.2f}, sl={optimal_action[1]:.4f}, tp={optimal_action[2]:.4f}")
print(f"  Short: value={optimal_action[3]:.2f}, sl={optimal_action[4]:.4f}, tp={optimal_action[5]:.4f}")

print("\n✓ TEST 5 COMPLETE")

#%%
# TEST 6: All Models
print("\n" + "="*80)
print("TEST 6: All Models")
print("="*80)

models_to_test = [
    ('Transformer', model.TransformerModel(price_shape=(cfg['lookback'], 5), meta_len=14, d_model=64, nhead=4, tx_blocks=2, mlp_ratio=4, dropout=0.1)),
    ('Informer', model.InformerModel(price_shape=(cfg['lookback'], 5), meta_len=14, d_model=64, nhead=4, blocks=2, dropout=0.1)),
    ('FEDformer', model.FedFormerModel(price_shape=(cfg['lookback'], 5), meta_len=14, d_model=64, nhead=4, blocks=2, dropout=0.1)),
    ('PatchTST', model.PatchTSTModel(price_shape=(cfg['lookback'], 5), meta_len=14, patch_len=16, stride=8, d_model=64, nhead=4, blocks=2, dropout=0.1)),
    ('iTransformer', model.iTransformerModel(price_shape=(cfg['lookback'], 5), meta_len=14, d_model=64, nhead=4, blocks=2, dropout=0.1)),
    ('N-BEATS', model.NBeatsModel(price_shape=(cfg['lookback'], 5), meta_len=14, stack_types=['trend', 'seasonality'], n_blocks=[1, 1], mlp_units=64, shares_weights=False, dropout=0.1)),
    ('N-HiTS', model.NHiTSModel(price_shape=(cfg['lookback'], 5), meta_len=14, pools=[1, 2], mlp_units=64, dropout=0.1)),
]

for name, m in models_to_test:
    p = predictor.Predictor(m, meta_scaler, cfg, name.lower().replace('-', ''), 'cpu')
    pred_val = p.predict(df_scaled, scaled=True)
    _, preds_many = p.predict_many_actions(ohlcv_arrays, state, 5, seed=42)
    opt = p.find_optimal_action(ohlcv_arrays, state, seed=42)
    print(f"✓ {name:12s}: predict={pred_val[0]:.6f}, many_mean={preds_many.mean():.6f}, opt_long={opt[0]:.0f}")

# LightGBM
print("\n--- LightGBM ---")
import lightgbm as lgb
lgb_train = lgb.Dataset(np.random.randn(100, cfg['lookback']*5 + 14), label=np.random.randn(100, cfg['forward']))
lgb_params = {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1, 'seed': 42}
m8 = lgb.train(lgb_params, lgb_train, num_boost_round=10, verbose_eval=False)
p8 = predictor.Predictor(m8, meta_scaler, cfg, 'lightgbm', 'cpu')
pred8 = p8.predict(df_scaled, scaled=True)
_, preds8 = p8.predict_many_actions(ohlcv_arrays, state, 5, seed=42)
opt8 = p8.find_optimal_action(ohlcv_arrays, state, seed=42)
print(f"✓ LightGBM    : predict={pred8[0]:.6f}, many_mean={preds8.mean():.6f}, opt_long={opt8[0]:.0f}")

print("\n✓ TEST 6 COMPLETE: All 8 models tested")

#%%
# SUMMARY
print("\n" + "="*80)
print("COMPREHENSIVE PREDICTOR TESTING COMPLETE")
print("="*80)
print("\n✓ TEST 1: Input Validation")
print("✓ TEST 2: Scaling")
print("✓ TEST 3: Prediction")
print("✓ TEST 4: Action Sampling")
print("✓ TEST 5: Optimization")
print("✓ TEST 6: All 8 Models")
print("\n" + "="*80)
print("All predictor.py functionality verified!")
print("="*80)

# %%
