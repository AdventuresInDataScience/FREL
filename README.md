# FREL
Financial Reinforcement with Endogenous Learning

A flexible framework for training deep learning models on financial time series data with synthetic reward signals.

## Structure
```
FREL/
├── config/
│   └── default.yaml       # Configuration for all models and training
├── src/
│   ├── __init__.py
│   ├── data.py           # Data loading and preprocessing
│   ├── dataset.py        # Dataset generation
│   ├── synth.py          # Synthetic state generation
│   ├── reward.py         # Reward computation (CAR, Sharpe, etc.)
│   ├── scale.py          # Feature scaling
│   ├── model.py          # All model architectures + factory
│   ├── predictor.py      # Inference and action optimization
│   └── curriculum.py     # Curriculum learning
├── tests/
│   └── test_*.py         # Unit tests
├── dev/
│   └── main.py           # Training script
└── models/               # Saved models
```

## Available Models

The framework supports 8 different model architectures, all following the same API:

### Neural Models (Transformer-based)
- **`transformer`**: Standard transformer with CNN front-end
- **`informer`**: Prob-sparse self-attention for long sequences
- **`fedformer`**: Frequency-enhanced decomposition transformer
- **`patchtst`**: Patch-based time series transformer
- **`itransformer`**: Inverted transformer (attention across variables)

### Neural Models (Specialized)
- **`nbeats`**: Neural basis expansion with interpretable blocks
- **`nhits`**: Hierarchical interpolation for multi-scale forecasting

### Tree-based Model
- **`lightgbm`**: Gradient boosting with GPU support and linear trees

## Quick Start

### 1. Configure Your Model
Edit `config/default.yaml` and set the `model_type`:

```yaml
model_type: "transformer"  # or informer, fedformer, patchtst, itransformer, nbeats, nhits, lightgbm
```

### 2. Run Training
```python
python dev/main.py
```

The script will:
1. Build the dataset with synthetic states and rewards
2. Build the model specified in config
3. Train the model
4. Evaluate predictions vs true rewards
5. Test action optimization

### 3. Using the Factory Function

```python
from src.model import build_model

# Build any model from config
CFG = yaml.safe_load(open("config/default.yaml"))
model = build_model(CFG)

# Neural models: compile and train
model.compile(loss='mse', optimizer='adam', jit_compile=True)
model.fit([X_price, X_meta], y, batch_size=1024, epochs=10)

# Save
model.save(f"models/{CFG['model_type']}_model.h5")
```

## Manual Model Building

You can also build models directly:

```python
from src.model import build_transformer, build_informer, build_patchtst

model = build_transformer(
    price_shape=(200, 5),
    meta_len=8,
    d_model=128,
    nhead=4,
    tx_blocks=4,
    dropout=0.1
)
```

## Streaming Large Datasets

For datasets too large for memory, use streaming:

```python
for chunk in pd.read_parquet("samples_320M.parquet", chunksize=12_000_000):
    ds = tf.data.Dataset.from_tensor_slices(
        ({'price': np.stack(chunk['close_scaled']),
          'meta': chunk[meta_cols].astype('float32')},
         chunk['y'].astype('float32'))
    ).batch(1024)
    model.fit(ds, epochs=1)
```

## Model-Specific Parameters

Each model has specific hyperparameters in `config/default.yaml`:

### Common Neural Parameters
- `d_model`: Hidden dimension (128)
- `nhead`: Number of attention heads (4)
- `tx_blocks`: Number of transformer blocks (4)
- `dropout`: Dropout rate (0.1)

### PatchTST Specific
- `patch_len`: Length of each patch (16)
- `patch_stride`: Stride between patches (8)

### N-BEATS Specific
- `nbeats_stack_types`: Stack types ["trend", "seasonality", "generic"]
- `nbeats_n_blocks`: Blocks per stack [1, 1, 1]
- `nbeats_mlp_units`: MLP units per block (512)

### NHITS Specific
- `nhits_pools`: Multi-rate pooling sizes [1, 2, 4]
- `nhits_mlp_units`: MLP units (512)

### LightGBM Specific
- `lgb_linear_tree`: Enable linear trees (true)
- `lgb_max_depth`: Tree depth (8)
- `lgb_num_leaves`: Number of leaves (256)

## Using the Predictor

The `Predictor` class handles inference and action optimization for all model types:

```python
from src.predictor import Predictor

# Load trained model
predictor = Predictor.from_checkpoint(
    model_path="models/transformer_model.h5",
    scaler_path="data/meta_scaler.json",
    cfg=CFG,
    model_type="transformer"  # or any other model type
)

# Predict rewards
y_pred = predictor.predict(X_price=X_price_test, X_meta=X_meta_test)

# Find optimal action
optimal = predictor.find_optimal_action(ohlcv_window, state)
```

## Supported Reward Metrics
- **CAR**: Compound Annual Return
- **Sharpe**: Sharpe ratio
- **Sortino**: Sortino ratio (downside deviation)
- **Calmar**: Calmar ratio (return/max drawdown)