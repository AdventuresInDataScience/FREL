# FREL Architecture Overview

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      config/default.yaml                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ model_type: "transformer"  # or any of 8 options          │  │
│  │ lookback: 200                                             │  │
│  │ d_model: 128                                              │  │
│  │ nhead: 4                                                  │  │
│  │ tx_blocks: 4                                              │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         dev/main.py                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ CFG = yaml.safe_load(open("config/default.yaml"))        │  │
│  │ model = build_model(CFG)  # Factory function             │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        src/model.py                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ def build_model(cfg):                                     │  │
│  │     if model_type == 'transformer':                       │  │
│  │         return build_tx_model(...)                        │  │
│  │     elif model_type == 'informer':                        │  │
│  │         return build_informer(...)                        │  │
│  │     elif model_type == 'fedformer':                       │  │
│  │         return build_fedformer(...)                       │  │
│  │     elif model_type == 'patchtst':                        │  │
│  │         return build_patchtst(...)                        │  │
│  │     elif model_type == 'itransformer':                    │  │
│  │         return build_itransformer(...)                    │  │
│  │     elif model_type == 'nbeats':                          │  │
│  │         return build_nbeats(...)                          │  │
│  │     elif model_type == 'nhits':                           │  │
│  │         return build_nhits(...)                           │  │
│  │     elif model_type == 'lightgbm':                        │  │
│  │         return build_lgb_model(...)                       │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Training & Saving                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ if model_type in neural_models:                           │  │
│  │     model.compile(loss='mse', optimizer='adam')           │  │
│  │     model.fit([X_price, X_meta], y)                       │  │
│  │     model.save(f"models/{model_type}_model.h5")           │  │
│  │ elif model_type == 'lightgbm':                            │  │
│  │     model.fit(X_flat, y)                                  │  │
│  │     joblib.dump(model, "models/lgb_model.pkl")            │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    src/predictor.py                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ predictor = Predictor.from_checkpoint(                    │  │
│  │     model_path=f"models/{model_type}_model.h5",           │  │
│  │     scaler_path="data/meta_scaler.json",                  │  │
│  │     cfg=CFG,                                              │  │
│  │     model_type=model_type                                 │  │
│  │ )                                                         │  │
│  │                                                           │  │
│  │ # All models use same interface                          │  │
│  │ if model_type in neural_models:                          │  │
│  │     y_pred = model.predict([X_price, X_meta])            │  │
│  │ elif model_type == 'lightgbm':                           │  │
│  │     y_pred = model.predict(X_flat)                       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Model Architecture Map

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEURAL MODELS (Keras)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  TRANSFORMER-BASED                                       │   │
│  │  ┌────────────────┐  ┌─────────────┐  ┌──────────────┐  │   │
│  │  │  transformer   │  │  informer   │  │  fedformer   │  │   │
│  │  │  Standard      │  │  Sparse     │  │  Frequency   │  │   │
│  │  │  Attention     │  │  Attention  │  │  Decompose   │  │   │
│  │  └────────────────┘  └─────────────┘  └──────────────┘  │   │
│  │  ┌────────────────┐  ┌──────────────────────────────┐   │   │
│  │  │  patchtst      │  │  itransformer                │   │   │
│  │  │  Patch-based   │  │  Inverted (var attention)    │   │   │
│  │  └────────────────┘  └──────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  SPECIALIZED                                             │   │
│  │  ┌────────────────┐  ┌─────────────────────────────┐    │   │
│  │  │  nbeats        │  │  nhits                      │    │   │
│  │  │  Basis Expand  │  │  Hierarchical Interpolate   │    │   │
│  │  └────────────────┘  └─────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Common Interface: model.predict([X_price, X_meta])             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   TREE MODEL (LightGBM)                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  lightgbm                                                  │ │
│  │  Gradient Boosting Trees                                  │ │
│  │  - GPU accelerated                                        │ │
│  │  - Linear trees option                                    │ │
│  │  - Handles missing data                                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Interface: model.predict(X_flat)                                │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Raw Market Data (OHLCV)
         │
         ▼
┌─────────────────────┐
│  src/data.py        │  Fetch historical data
│  yfinance           │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  src/synth.py       │  Generate synthetic trading states
│  Random equity,     │  (equity, balance, position, SL, TP)
│  positions, etc.    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  src/reward.py      │  Compute optimal actions & rewards
│  CAR, Sharpe, etc.  │  Forward-looking simulation
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  src/scale.py       │  Scale features
│  MinMax/Std scaling │  OHLCV + meta features
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  src/dataset.py     │  Build complete dataset
│  Parquet file       │  (price_scaled, meta_scaled, y)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  dev/main.py        │  Train model
│  Build + fit model  │  Save checkpoint
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  src/predictor.py   │  Inference & optimization
│  Load model         │  Find optimal actions
│  Predict rewards    │
└─────────────────────┘
```

## Model Input/Output Schema

### Neural Models
```
Input:
  price_in: (batch, 200, 5)
    - Channels: [open, high, low, close, volume]
    - Scaled using OHLCV window scaling
  
  meta_in: (batch, 8)
    - Features: [equity, balance, position, 
                 sl_dist, tp_dist, 
                 act_dollar, act_sl, act_tp]
    - Scaled using MinMaxScaler

Output:
  prediction: (batch, 1)
    - Predicted reward (CAR, Sharpe, etc.)
```

### LightGBM
```
Input:
  X: (batch, 200*5 + 8) = (batch, 1008)
    - Flattened price window + meta features

Output:
  prediction: (batch,)
    - Predicted reward
```

## Configuration Schema

```yaml
# Data
ticker: ^GSPC
lookback: 200
forward: 5

# Reward
reward_key: car  # car|sharpe|sortino|calmar
fee_bps: 0.2
slippage_bps: 0.1

# Model Selection
model_type: transformer  # 8 options

# Common Hyperparameters (Neural)
d_model: 128
nhead: 4
tx_blocks: 4
dropout: 0.1
lr: 1e-3
batch_size: 1024
epochs: 10

# Model-Specific Hyperparameters
patch_len: 16                # PatchTST
nbeats_stack_types: [...]    # N-BEATS
nhits_pools: [1,2,4]         # NHITS
lgb_linear_tree: true        # LightGBM
```

## Class Hierarchy

```
Predictor
├── __init__(model, meta_scaler, cfg, model_type)
├── from_checkpoint(model_path, scaler_path, cfg, model_type)
├── predict(X_price, X_meta, sample, raw)
├── predict_all_actions(ohlcv_window, state, ...)
├── find_optimal_action(ohlcv_window, state, ...)
└── compare_predicted_vs_true(close, idx, ...)

MetaScaler
├── __init__(meta_cols, method)
├── fit(df, meta_cols)
├── transform(df, meta_cols)
├── save(path)
└── load(path)

Models (all follow same interface)
├── build_tx_model(...)      → keras.Model
├── build_informer(...)      → keras.Model
├── build_fedformer(...)     → keras.Model
├── build_patchtst(...)      → keras.Model
├── build_itransformer(...)  → keras.Model
├── build_nbeats(...)        → keras.Model
├── build_nhits(...)         → keras.Model
└── build_lgb_model(...)     → lgb.LGBMRegressor

Factory
└── build_model(cfg)         → Model (any type)
```

## File Dependencies

```
dev/main.py
├── config/default.yaml
├── src.dataset
│   ├── src.data
│   ├── src.synth
│   ├── src.reward
│   └── src.scale
├── src.model
│   └── (tensorflow, lightgbm)
└── src.predictor
    ├── src.scale
    └── src.reward

src/model.py
├── tensorflow
├── lightgbm
└── numpy

src/predictor.py
├── tensorflow
├── lightgbm
├── numpy
├── pandas
├── scipy
├── src.scale
└── src.reward
```

## Extension Points

### Adding a New Model

1. **Create builder function** in `src/model.py`:
```python
def build_mymodel(price_shape=(200,5), meta_len=10, ...):
    price = Input(shape=price_shape, name='price')
    meta = Input(shape=(meta_len,), name='meta')
    
    # Your architecture here
    
    out = Dense(1)(...)
    return Model([price, meta], out)
```

2. **Add to factory** in `build_model()`:
```python
elif model_type == 'mymodel':
    return build_mymodel(
        price_shape=price_shape,
        meta_len=meta_len,
        ...
    )
```

3. **Add config** in `config/default.yaml`:
```yaml
model_type: "mymodel"
mymodel_param1: 100
mymodel_param2: 200
```

4. **Done!** Run `python dev/main.py`

No changes needed in:
- ✅ Predictor (automatically handles all neural models)
- ✅ main.py (uses factory pattern)
- ✅ Dataset pipeline (model-agnostic)
