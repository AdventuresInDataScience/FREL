# Model Selection Quick Reference

## TL;DR
Edit `config/default.yaml` → change `model_type` → run `python dev/main.py`

## Available Models

| Model | Type | Best For | Key Feature |
|-------|------|----------|-------------|
| `transformer` | Neural | General purpose | Standard attention, proven architecture |
| `informer` | Neural | Long sequences | Prob-sparse attention, O(L log L) |
| `fedformer` | Neural | Seasonal patterns | Frequency-domain decomposition |
| `patchtst` | Neural | Multi-scale patterns | Patch-based tokenization |
| `itransformer` | Neural | Multivariate correlation | Attention across variables |
| `nbeats` | Neural | Interpretable forecasts | Trend/seasonality decomposition |
| `nhits` | Neural | Multi-horizon | Hierarchical interpolation |
| `lightgbm` | Tree | Tabular features | Fast, efficient, handles missing data |

## Quick Config Examples

### Standard Transformer (Default)
```yaml
model_type: "transformer"
d_model: 128
nhead: 4
tx_blocks: 4
dropout: 0.1
```

### PatchTST (Recommended for long lookback)
```yaml
model_type: "patchtst"
d_model: 128
nhead: 4
tx_blocks: 4
patch_len: 16
patch_stride: 8
```

### N-BEATS (Interpretable)
```yaml
model_type: "nbeats"
nbeats_stack_types: ["trend", "seasonality", "generic"]
nbeats_n_blocks: [1, 1, 1]
nbeats_mlp_units: 512
```

### LightGBM (Fast baseline)
```yaml
model_type: "lightgbm"
lgb_linear_tree: true
lgb_max_depth: 8
lgb_num_leaves: 256
```

## Model Comparison

### Speed (Training Time)
1. **LightGBM** - Fastest (tree-based, GPU)
2. **PatchTST** - Fast (reduced sequence length)
3. **Transformer** - Medium
4. **N-BEATS/NHITS** - Medium
5. **Informer/FEDformer** - Slower (complex attention)
6. **iTransformer** - Slowest (attention across all variables)

### Memory Usage (200 lookback)
1. **LightGBM** - Lowest (no backprop)
2. **PatchTST** - Low (compressed patches)
3. **N-BEATS** - Medium (stacked blocks)
4. **Transformer** - Medium
5. **FEDformer** - High (FFT overhead)
6. **Informer** - High (prob-sparse selection)

### Accuracy (Typical)
Results vary by dataset. Generally:
- **Transformer/PatchTST**: Excellent general accuracy
- **FEDformer**: Best for seasonal/cyclical patterns
- **N-BEATS**: Excellent + interpretable
- **LightGBM**: Strong baseline, fast iterations

## Common Hyperparameters

### All Neural Models
```yaml
lr: 1e-3              # Learning rate
batch_size: 1024      # Batch size
epochs: 10            # Max epochs (use early stopping)
dropout: 0.1          # Dropout rate
d_model: 128          # Hidden dimension
```

### Attention-Based Models
```yaml
nhead: 4              # Number of attention heads
tx_blocks: 4          # Number of transformer blocks
mlp_ratio: 4          # FFN expansion ratio
```

## Code Snippets

### Build Any Model
```python
from src.model import build_model
import yaml

CFG = yaml.safe_load(open("config/default.yaml"))
model = build_model(CFG)
```

### Train Neural Model
```python
model.compile(
    loss="mse",
    optimizer=tf.keras.optimizers.Adam(CFG["lr"]),
    jit_compile=True  # XLA compilation
)

model.fit(
    [X_price_train, X_meta_train],
    y_train,
    batch_size=CFG["batch_size"],
    epochs=CFG["epochs"],
    validation_split=0.1
)

model.save(f"models/{CFG['model_type']}_model.h5")
```

### Train LightGBM
```python
X_train = np.concatenate([
    X_price_train.reshape(len(X_price_train), -1),
    X_meta_train
], axis=1)

model.fit(X_train, y_train)

import joblib
joblib.dump(model, "models/lgb_model.pkl")
```

### Load and Predict
```python
from src.predictor import Predictor

predictor = Predictor.from_checkpoint(
    model_path=f"models/{CFG['model_type']}_model.h5",
    scaler_path="data/meta_scaler.json",
    cfg=CFG,
    model_type=CFG['model_type']
)

y_pred = predictor.predict(X_price=X_test, X_meta=X_meta_test)
```

## Troubleshooting

### Out of Memory
1. Reduce `batch_size`: 1024 → 512 → 256
2. Reduce `d_model`: 128 → 64
3. Use PatchTST (compressed sequences)
4. Use LightGBM (no GPU memory)

### Training Too Slow
1. Use LightGBM for quick baseline
2. Enable `jit_compile=True` (XLA)
3. Use smaller `epochs` with early stopping
4. Reduce `lookback` window size
5. Use PatchTST (faster attention)

### Poor Accuracy
1. Increase `epochs` and add early stopping
2. Increase `d_model`: 64 → 128 → 256
3. Increase `tx_blocks`: 2 → 4 → 6
4. Try different model types (especially FEDformer for seasonal data)
5. Tune learning rate: 1e-3 → 5e-4 → 1e-4
6. Check data quality and scaling

### Model Won't Load
- Check `model_type` matches the saved model
- For neural models: use `.h5` path
- For LightGBM: use `.pkl` path
- Ensure MetaScaler JSON exists

## When to Use Each Model

### Use **Transformer** when:
- General purpose forecasting
- You want proven, stable architecture
- Balance between speed and accuracy

### Use **Informer** when:
- Very long lookback windows (>200)
- Memory constraints with long sequences
- Need efficient attention

### Use **FEDformer** when:
- Data has strong seasonal patterns
- Cyclical market behavior
- Frequency analysis is important

### Use **PatchTST** when:
- Need faster training/inference
- Long lookback windows
- Local patterns are important

### Use **iTransformer** when:
- Strong cross-variable relationships
- Many meta features (>10)
- Variable interactions matter

### Use **N-BEATS** when:
- Need interpretable forecasts
- Want to understand trend/seasonality split
- Stakeholders require explainability

### Use **NHITS** when:
- Multi-horizon forecasting
- Need different resolutions
- Hierarchical patterns exist

### Use **LightGBM** when:
- Need fast baseline
- Quick experimentation
- Strong tabular baseline
- Missing data in features

## Tips

1. **Start Simple**: Begin with `transformer` or `lightgbm`
2. **Iterate Fast**: Use small `epochs` initially (2-5)
3. **Validate**: Always use validation split (0.1-0.2)
4. **Compare**: Try 2-3 models and compare
5. **Ensemble**: Best results often from combining models
6. **Monitor**: Watch validation loss for overfitting
7. **Save Often**: Save checkpoints during training
8. **Document**: Note which config worked best

## Performance Benchmarks (Approximate)

On 100k samples (200 lookback, batch 1024, 10 epochs):

| Model | Train Time | Memory | Params |
|-------|-----------|--------|--------|
| LightGBM | ~30 sec | ~500 MB | ~50K |
| PatchTST | ~2 min | ~2 GB | ~500K |
| Transformer | ~3 min | ~2.5 GB | ~600K |
| N-BEATS | ~3 min | ~2 GB | ~400K |
| NHITS | ~3 min | ~2 GB | ~400K |
| FEDformer | ~4 min | ~3 GB | ~700K |
| Informer | ~4 min | ~3 GB | ~700K |
| iTransformer | ~5 min | ~3.5 GB | ~800K |

*Times on single NVIDIA RTX 3090, may vary by hardware*
