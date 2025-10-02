# Model Integration Summary

## Overview
This document summarizes the updates made to properly integrate all model architectures into the FREL pipeline.

## Problem
The codebase initially supported only two model types:
- `transformer`: Custom transformer model
- `lightgbm`: Tree-based gradient boosting

However, additional models were added to `src/model.py` but not properly integrated into the pipeline:
- `informer`: Prob-sparse attention transformer
- `fedformer`: Frequency-enhanced decomposition transformer
- `patchtst`: Patch-based time series transformer
- `itransformer`: Inverted transformer (attention across variables)
- `nbeats`: Neural basis expansion
- `nhits`: Hierarchical interpolation

## Changes Made

### 1. Configuration (`config/default.yaml`)
**Added:**
- Documentation of all 8 available model types
- Model-specific hyperparameters for each architecture:
  - **PatchTST**: `patch_len`, `patch_stride`
  - **N-BEATS**: `nbeats_stack_types`, `nbeats_n_blocks`, `nbeats_mlp_units`, `nbeats_shares_weights`
  - **NHITS**: `nhits_pools`, `nhits_mlp_units`
  - **LightGBM**: `lgb_linear_tree`, `lgb_max_depth`, `lgb_num_leaves`, etc.

**Before:**
```yaml
model_type: "transformer"  # transformer | lightgbm
```

**After:**
```yaml
# Available model types: transformer | lightgbm | informer | fedformer | patchtst | itransformer | nbeats | nhits
model_type: "transformer"

# Common neural network hyperparameters
d_model: 128
nhead: 4
tx_blocks: 4
...

# Model-specific parameters
patch_len: 16
nbeats_stack_types: ["trend", "seasonality", "generic"]
nhits_pools: [1, 2, 4]
```

### 2. Model Factory (`src/model.py`)
**Added:**
- `build_model(cfg)` factory function that creates any model based on configuration
- Proper numpy import to fix N-BEATS implementation
- Comprehensive model type validation

**Usage:**
```python
from src.model import build_model

# Build any model from config
model = build_model(CFG)

# Works for all 8 model types
if model_type in neural_models:
    model.compile(loss='mse', optimizer='adam', jit_compile=True)
    model.fit([X_price, X_meta], y, ...)
```

**Benefits:**
- Single entry point for all model creation
- Automatic parameter extraction from config
- Consistent API across all models
- Easy to add new models in the future

### 3. Predictor Class (`src/predictor.py`)
**Updated:**
- `__init__()`: Added `neural_models` list to distinguish neural vs tree models
- `from_checkpoint()`: Updated to handle all 8 model types
- `predict()`: Changed condition from `if self.model_type == "transformer"` to `if self.model_type in self.neural_models`

**Before:**
```python
if self.model_type == "transformer":
    y_pred = self.model.predict([X_price, X_meta], verbose=0)
else:  # lightgbm
    ...
```

**After:**
```python
if self.model_type in self.neural_models:
    # All neural models use the same input format
    y_pred = self.model.predict([X_price, X_meta], verbose=0)
elif self.model_type == "lightgbm":
    ...
else:
    raise ValueError(f"Unknown model_type: {self.model_type}")
```

### 4. Training Script (`dev/main.py`)
**Updated:**
- Import changed from `build_tx_model, build_lgb_model` to `build_model`
- Replaced hardcoded model building with factory function call
- Added dynamic model path generation: `f"models/{model_type}_model.h5"`
- Added model type logging for debugging

**Before:**
```python
if CFG["model_type"] == "transformer":
    model = build_tx_model(...)
    model.compile(...)
    model.save("models/transformer_model.h5")
else:  # lightgbm
    model = build_lgb_model(...)
    joblib.dump(model, "models/lgb_model.pkl")
```

**After:**
```python
# Build any model using factory
model = build_model(CFG)

if model_type in neural_models:
    model.compile(loss="mse", optimizer=..., jit_compile=True)
    model.fit([X_price_train, X_meta_train], y_train, ...)
    model.save(f"models/{model_type}_model.h5")
elif model_type == "lightgbm":
    X_train = np.concatenate([X_price_train.reshape(...), X_meta_train], axis=1)
    model.fit(X_train, y_train)
    joblib.dump(model, "models/lgb_model.pkl")
```

### 5. Documentation (`README.md`)
**Completely Rewritten:**
- Added comprehensive model overview
- Listed all 8 available model types with descriptions
- Added quick start guide
- Documented model-specific hyperparameters
- Added factory function usage examples
- Documented streaming dataset usage
- Added Predictor usage examples

## Model Architecture Overview

### Transformer-Based Models
All use the same input format: `[price_data, meta_features]`
- **transformer**: Standard multi-head attention with CNN front-end
- **informer**: Efficient prob-sparse attention for long sequences
- **fedformer**: Fourier-based frequency decomposition
- **patchtst**: Patch-based tokenization with attention
- **itransformer**: Inverted dimensions (attention across variables not time)

### Specialized Neural Models
- **nbeats**: Doubly residual stacking with trend/seasonality/generic blocks
- **nhits**: Hierarchical multi-rate pooling with interpolation

### Tree-Based Model
- **lightgbm**: GPU-accelerated gradient boosting with optional linear trees

## How to Use Different Models

### 1. Edit Config
```yaml
model_type: "patchtst"  # or any other model
```

### 2. Run Training
```bash
python dev/main.py
```

### 3. Load for Inference
```python
predictor = Predictor.from_checkpoint(
    model_path="models/patchtst_model.h5",
    scaler_path="data/meta_scaler.json",
    cfg=CFG,
    model_type="patchtst"
)
```

## Testing

To test different models, simply change the `model_type` in `config/default.yaml`:

```yaml
# Test transformer
model_type: "transformer"

# Test PatchTST
model_type: "patchtst"

# Test N-BEATS
model_type: "nbeats"

# Test LightGBM
model_type: "lightgbm"
```

All models will work with the existing pipeline without code changes!

## Benefits of This Integration

1. **Consistency**: All models use the same training pipeline
2. **Flexibility**: Easy to switch between models by changing config
3. **Maintainability**: Single factory function to manage model creation
4. **Extensibility**: Easy to add new models by:
   - Adding build function in `model.py`
   - Adding case in `build_model()` factory
   - Adding hyperparameters in `default.yaml`
5. **Type Safety**: Proper error handling for unknown model types
6. **Documentation**: Comprehensive README with examples

## Future Enhancements

Potential improvements:
1. Add automatic hyperparameter tuning per model type
2. Create model comparison script to evaluate all models
3. Add ensemble predictions combining multiple models
4. Implement model-specific learning rate schedules
5. Add early stopping callbacks with model-specific patience
6. Create benchmarking suite for all models

## Migration Guide

If you have existing code using the old API:

**Old:**
```python
from src.model import build_tx_model
model = build_tx_model(...)
```

**New (Option 1 - Factory):**
```python
from src.model import build_model
model = build_model(CFG)
```

**New (Option 2 - Direct):**
```python
from src.model import build_tx_model  # Still works!
model = build_tx_model(...)
```

Both approaches are supported for backward compatibility.
