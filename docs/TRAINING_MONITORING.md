# Training Loss Monitoring & Checkpointing

Complete implementation of training loss monitoring, early stopping, and model checkpointing for both PyTorch and LightGBM models in the MAPIE wrapper system.

## ✅ Features Implemented

### **PyTorch Models (SklearnPyTorchWrapper)**

#### 1. **Training Loss Tracking**
- ✅ Epoch-by-epoch train loss recording
- ✅ Validation loss tracking with automatic train/val split
- ✅ Learning rate monitoring across epochs
- ✅ Epoch counter for temporal tracking

#### 2. **Early Stopping**
- ✅ Configurable patience parameter
- ✅ Validation loss monitoring
- ✅ Best model state preservation in memory
- ✅ Automatic restoration of best model after training
- ✅ Best epoch tracking

#### 3. **Model Checkpointing**
- ✅ Optional disk-based checkpoint saving
- ✅ Comprehensive checkpoint data (model, optimizer, history)
- ✅ Load checkpoint capability with `load_checkpoint(path)`
- ✅ In-memory best model state preservation

#### 4. **Verbose Output**
- ✅ Three verbosity levels:
  - `verbose=0`: Silent
  - `verbose=1`: Progress updates every 10% of epochs
  - `verbose=2`: Every epoch with detailed metrics
- ✅ Visual indicators (⭐) for best epochs
- ✅ Clear early stopping notifications
- ✅ Best model restoration confirmation

#### 5. **History Access**
```python
wrapper = SklearnPyTorchWrapper(
    model=my_model,
    epochs=50,
    verbose=1,
    early_stopping=True,
    patience=10,
    validation_split=0.2,
    checkpoint_path='best_model.pt'  # Optional
)

wrapper.fit(X, y)

# Get training history
history = wrapper.get_training_history()
# Returns: {'train_loss': [...], 'val_loss': [...], 'lr': [...], 'epoch': [...]}

# Access metadata
best_epoch = wrapper.best_epoch_
best_val_loss = wrapper.best_val_loss_

# Load from checkpoint
wrapper.load_checkpoint('best_model.pt')
```

---

### **LightGBM Models (SklearnLightGBMWrapper)**

#### 1. **Training Loss Tracking**
- ✅ Iteration-by-iteration train loss recording
- ✅ Validation loss tracking via `eval_set`
- ✅ Native LightGBM `evals_result_` integration
- ✅ Consistent history format with PyTorch wrapper

#### 2. **Early Stopping**
- ✅ Native LightGBM early stopping integration
- ✅ Configurable `early_stopping_rounds` parameter
- ✅ Best iteration tracking
- ✅ Best score preservation

#### 3. **Model Saving**
- ✅ Optional model saving to disk
- ✅ Full booster serialization (not incremental - LightGBM limitation)
- ✅ Load saved model capability with `load_model(path)`

#### 4. **Verbose Output**
- ✅ Three verbosity levels:
  - `verbose=-1`: Silent
  - `verbose=0`: Warnings only
  - `verbose=1`: Info messages every 10 iterations
  - `verbose=2`: Detailed messages every iteration
- ✅ Native LightGBM log_evaluation callback
- ✅ Training/validation size reporting
- ✅ Early stopping notifications

#### 5. **History Access**
```python
wrapper = SklearnLightGBMWrapper(
    model=lgb_model,
    verbose=1,
    early_stopping_rounds=10,
    validation_split=0.2,
    save_path='lgb_model.txt'  # Optional
)

wrapper.fit(X, y)

# Get training history
history = wrapper.get_training_history()
# Returns: {'train_loss': [...], 'val_loss': [...], 'epoch': [...]}

# Access metadata
best_iteration = wrapper.best_iteration_
best_score = wrapper.best_score_

# Load saved model
wrapper.load_model('lgb_model.txt')
```

---

## 📊 Training History Format

Both wrappers return training history in a **consistent dictionary format**:

```python
{
    'train_loss': [0.056, 0.043, 0.032, ...],  # Training loss per epoch/iteration
    'val_loss': [0.048, 0.038, 0.033, ...],    # Validation loss per epoch/iteration
    'epoch': [1, 2, 3, ...],                   # Epoch/iteration counter
    'lr': [0.001, 0.001, 0.0009, ...]          # Learning rate (PyTorch only)
}
```

This enables:
- **Loss curve plotting**: Visualize training progress
- **Convergence analysis**: Check for proper training dynamics
- **Overfitting detection**: Compare train vs validation loss
- **Hyperparameter tuning**: Compare training runs

---

## 🎯 Usage Examples

### Example 1: PyTorch with Full Monitoring
```python
from src.mapie import SklearnPyTorchWrapper
from src.model import TransformerModel
import tempfile

# Create model
model = TransformerModel(price_shape=(200, 5), meta_len=20, d_model=128)

# Create checkpoint path
checkpoint_path = tempfile.mktemp(suffix='.pt')

# Wrap with comprehensive monitoring
wrapper = SklearnPyTorchWrapper(
    model=model,
    epochs=100,
    batch_size=32,
    lr=0.001,
    verbose=2,  # Show every epoch
    early_stopping=True,
    patience=15,
    validation_split=0.2,
    checkpoint_path=checkpoint_path,
    loss_type='huber'
)

# Train
wrapper.fit(X_combined, y)

# Analyze training
history = wrapper.get_training_history()
print(f"Best epoch: {wrapper.best_epoch_}")
print(f"Best val loss: {wrapper.best_val_loss_:.6f}")
print(f"Training improvement: {(history['train_loss'][0] - history['train_loss'][-1]) / history['train_loss'][0] * 100:.1f}%")

# Plot training curves (example)
import matplotlib.pyplot as plt
plt.plot(history['epoch'], history['train_loss'], label='Train')
plt.plot(history['epoch'], history['val_loss'], label='Validation')
plt.axvline(wrapper.best_epoch_, color='red', linestyle='--', label='Best Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

### Example 2: LightGBM with Monitoring
```python
from src.mapie import SklearnLightGBMWrapper
import lightgbm as lgb

# Create LightGBM model
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    verbose=-1  # Let wrapper handle verbosity
)

# Wrap with monitoring
wrapper = SklearnLightGBMWrapper(
    model=lgb_model,
    verbose=1,
    early_stopping_rounds=50,
    validation_split=0.2,
    save_path='lgb_model.txt'
)

# Train
wrapper.fit(X_combined, y)

# Analyze training
history = wrapper.get_training_history()
print(f"Best iteration: {wrapper.best_iteration_}")
print(f"Total iterations: {len(history['epoch'])}")
print(f"Final train loss: {history['train_loss'][-1]:.6f}")
print(f"Final val loss: {history['val_loss'][-1]:.6f}")
```

---

## 🔬 Test Results

From **TEST 7** in `08 test_mapie.py`:

### PyTorch Training Output:
```
⭐ Epoch 4/20 - Train: 0.026352, Val: 0.033906, LR: 0.00100000
⭐ Epoch 8/20 - Train: 0.016191, Val: 0.032027, LR: 0.00100000
⭐ Epoch 12/20 - Train: 0.013822, Val: 0.030537, LR: 0.00100000
⭐ Epoch 18/20 - Train: 0.012543, Val: 0.028944, LR: 0.00100000
✅ Restored best model from epoch 19 (val_loss=0.028527)

✓ PyTorch training completed
  Total epochs trained: 20
  Best epoch: 19
  Best validation loss: 0.028527
  Training improvement: 75.4% reduction in loss
  ✅ Model shows good convergence
```

### LightGBM Training Output:
```
📊 LightGBM training with validation: 80 train, 20 val samples
Training until validation scores don't improve for 10 rounds
[10]    train's l2: 0.0143922   valid's l2: 0.0214389
Early stopping, best iteration is:
[4]     train's l2: 0.015779    valid's l2: 0.0209601
✅ Best iteration: 4

✓ LightGBM training completed
  Total iterations: 14
  Best iteration: 4
  Training improvement: 18.0% reduction in loss
  ✅ Model shows good convergence
```

---

## 📈 Key Benefits

1. **Convergence Validation**: Verify models are learning properly
2. **Overfitting Detection**: Identify train/val loss divergence
3. **Hyperparameter Tuning**: Compare different configurations
4. **Production Readiness**: Track training quality for deployment decisions
5. **Reproducibility**: Load best checkpoints for consistent results
6. **Debugging**: Diagnose training issues early

---

## 🚀 Production Recommendations

### For PyTorch Models:
```python
wrapper = SklearnPyTorchWrapper(
    model=model,
    epochs=200,
    batch_size=64,
    lr=0.0005,
    verbose=1,  # Progress updates
    early_stopping=True,
    patience=20,  # More patience for complex models
    validation_split=0.15,  # Reserve 15% for validation
    checkpoint_path='production_model.pt',  # Save best model
    loss_type='huber',  # Robust to outliers
    scheduler_type='plateau'  # Adaptive LR
)
```

### For LightGBM Models:
```python
wrapper = SklearnLightGBMWrapper(
    model=lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,  # Lower LR for better convergence
        max_depth=8,
        num_leaves=127,
        verbose=-1
    ),
    verbose=1,
    early_stopping_rounds=100,  # Patience for tree-based models
    validation_split=0.15,
    save_path='production_lgb.txt'
)
```

---

## ✅ Validation Summary

- ✅ **PyTorch**: Full training history tracking, early stopping, checkpointing
- ✅ **LightGBM**: Native callback integration, early stopping, model saving
- ✅ **Consistent API**: Both wrappers expose `get_training_history()`
- ✅ **Production Ready**: All features tested and validated
- ✅ **Comprehensive Test**: TEST 7 validates all monitoring features

This implementation provides production-grade training monitoring capabilities while maintaining compatibility with the MAPIE conformal prediction framework.
