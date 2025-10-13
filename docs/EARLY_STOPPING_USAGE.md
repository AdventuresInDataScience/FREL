# Early Stopping Configuration Guide

Complete guide for configuring early stopping parameters in both PyTorch and LightGBM wrappers.

## ✅ Parameters ARE Already Exposed!

Both wrappers have early stopping parameters available. Here's how to use them:

---

## PyTorch Models (`SklearnPyTorchWrapper`)

### **Parameters:**
- **`early_stopping`** (bool): Enable/disable early stopping (default: `True`)
- **`patience`** (int): Number of epochs with no improvement before stopping (default: `10`)
- **`validation_split`** (float): Fraction of data for validation (default: `0.2`)

### **Usage Examples:**

#### Method 1: Direct Constructor
```python
from src.mapie import SklearnPyTorchWrapper
from src.model import TransformerModel

model = TransformerModel(price_shape=(200, 5), meta_len=20, d_model=128)

wrapper = SklearnPyTorchWrapper(
    model=model,
    epochs=100,
    batch_size=32,
    lr=0.001,
    verbose=1,
    # Early stopping configuration
    early_stopping=True,      # Enable early stopping
    patience=20,              # Wait 20 epochs before stopping
    validation_split=0.15,    # Use 15% of data for validation
    checkpoint_path='model.pt'
)

wrapper.fit(X_combined, y)
```

#### Method 2: Using `set_params()` (sklearn compatibility)
```python
wrapper = SklearnPyTorchWrapper(model=model, epochs=100)

# Configure parameters after creation
wrapper.set_params(
    early_stopping=True,
    patience=15,
    validation_split=0.2,
    verbose=2
)

wrapper.fit(X_combined, y)
```

#### Method 3: Disable Early Stopping
```python
wrapper = SklearnPyTorchWrapper(
    model=model,
    epochs=100,
    early_stopping=False,  # Train for all epochs
    verbose=1
)

wrapper.fit(X_combined, y)
```

---

## LightGBM Models (`SklearnLightGBMWrapper`)

### **Parameters:**
- **`early_stopping_rounds`** (int, optional): Stop if no improvement for N rounds (default: `None` = disabled)
- **`validation_split`** (float): Fraction of data for validation (default: `0.2`)
- **`verbose`** (int): Training verbosity (default: `0`)

### **Usage Examples:**

#### Method 1: Direct Constructor
```python
from src.mapie import SklearnLightGBMWrapper
import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    verbose=-1
)

wrapper = SklearnLightGBMWrapper(
    model=lgb_model,
    # Early stopping configuration
    early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
    validation_split=0.2,      # Use 20% of data for validation
    verbose=1,                 # Show training progress
    save_path='lgb_model.txt'
)

wrapper.fit(X_combined, y)
```

#### Method 2: Using `set_params()` (sklearn compatibility)
```python
wrapper = SklearnLightGBMWrapper(model=lgb_model)

# Configure parameters after creation
wrapper.set_params(
    early_stopping_rounds=100,
    validation_split=0.15,
    verbose=2
)

wrapper.fit(X_combined, y)
```

#### Method 3: Disable Early Stopping
```python
wrapper = SklearnLightGBMWrapper(
    model=lgb_model,
    early_stopping_rounds=None,  # No early stopping
    validation_split=0.0,        # No validation set
    verbose=1
)

wrapper.fit(X_combined, y)
```

---

## MAPIE Integration

Early stopping parameters work seamlessly with MAPIE's cross-validation:

### **PyTorch + MAPIE:**
```python
from src.mapie import SklearnPyTorchWrapper, MapiePredictor

# Create wrapper with early stopping
wrapper = SklearnPyTorchWrapper(
    model=model,
    epochs=100,
    early_stopping=True,
    patience=15,
    validation_split=0.2,
    verbose=1
)

# Wrap with MAPIE
mapie_pred = MapiePredictor(
    model=wrapper,
    method='plus',
    cv=5
)

# Each CV fold will use early stopping independently
mapie_pred.fit(X_price, X_meta, y)
```

### **LightGBM + MAPIE:**
```python
from src.mapie import SklearnLightGBMWrapper, MapiePredictor

# Create wrapper with early stopping
wrapper = SklearnLightGBMWrapper(
    model=lgb_model,
    early_stopping_rounds=50,
    validation_split=0.2,
    verbose=1
)

# Wrap with MAPIE
mapie_pred = MapiePredictor(
    model=wrapper,
    method='plus',
    cv=5
)

# Each CV fold will use early stopping independently
mapie_pred.fit(X_price, X_meta, y)
```

---

## Grid Search / Hyperparameter Tuning

Since the wrappers are sklearn-compatible, you can tune early stopping parameters:

```python
from sklearn.model_selection import GridSearchCV
from src.mapie import SklearnPyTorchWrapper

wrapper = SklearnPyTorchWrapper(model=model, epochs=100)

param_grid = {
    'patience': [5, 10, 15, 20],
    'validation_split': [0.1, 0.15, 0.2],
    'lr': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64]
}

grid_search = GridSearchCV(
    wrapper,
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=2
)

grid_search.fit(X_combined, y)
print(f"Best parameters: {grid_search.best_params_}")
```

---

## Complete Parameter Reference

### **PyTorch (`SklearnPyTorchWrapper`)**

```python
SklearnPyTorchWrapper(
    model: nn.Module,
    model_type: str = 'transformer',
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    device: str = 'cuda',
    verbose: int = 0,
    lookback: int = 200,
    price_features: int = 5,
    meta_len: int = 20,
    
    # Training configuration
    loss_type: str = 'mse',
    optimizer_type: str = 'adam',
    weight_decay: float = 0.0,
    scheduler_type: Optional[str] = None,
    
    # Early stopping (CONFIGURABLE!)
    early_stopping: bool = True,          # ← Enable/disable
    patience: int = 10,                   # ← Epochs to wait
    validation_split: float = 0.2,        # ← Validation fraction
    
    # Loss function parameters
    loss_epsilon: float = 1e-8,
    loss_delta: float = 1.0,
    loss_threshold: float = 0.1,
    
    # Checkpointing
    checkpoint_path: Optional[str] = None
)
```

### **LightGBM (`SklearnLightGBMWrapper`)**

```python
SklearnLightGBMWrapper(
    model: Union[lgb.LGBMRegressor, lgb.Booster],
    lookback: int = 200,
    price_features: int = 5,
    
    # Training monitoring (CONFIGURABLE!)
    verbose: int = 0,                          # ← Verbosity level
    early_stopping_rounds: Optional[int] = None,  # ← Rounds to wait
    validation_split: float = 0.2,             # ← Validation fraction
    save_path: Optional[str] = None
)
```

---

## Practical Recommendations

### **For Development/Debugging:**
```python
# PyTorch
wrapper = SklearnPyTorchWrapper(
    model=model,
    epochs=50,
    verbose=2,              # Show every epoch
    early_stopping=True,
    patience=5,             # Short patience for quick iteration
    validation_split=0.2
)

# LightGBM
wrapper = SklearnLightGBMWrapper(
    model=lgb_model,
    verbose=2,              # Detailed output
    early_stopping_rounds=10,  # Quick stopping for testing
    validation_split=0.2
)
```

### **For Production Training:**
```python
# PyTorch
wrapper = SklearnPyTorchWrapper(
    model=model,
    epochs=200,
    verbose=1,              # Progress updates only
    early_stopping=True,
    patience=25,            # More patience for convergence
    validation_split=0.15,  # Smaller validation set
    checkpoint_path='production_model.pt'
)

# LightGBM
wrapper = SklearnLightGBMWrapper(
    model=lgb_model,
    verbose=1,
    early_stopping_rounds=100,  # Patient for tree-based models
    validation_split=0.15,
    save_path='production_lgb.txt'
)
```

### **For Quick Experiments:**
```python
# PyTorch - No early stopping, fixed epochs
wrapper = SklearnPyTorchWrapper(
    model=model,
    epochs=20,
    verbose=1,
    early_stopping=False  # Train all epochs
)

# LightGBM - No early stopping
wrapper = SklearnLightGBMWrapper(
    model=lgb_model,
    verbose=1,
    early_stopping_rounds=None  # No early stopping
)
```

---

## Monitoring Early Stopping

Check if early stopping occurred:

### **PyTorch:**
```python
wrapper.fit(X_combined, y)

history = wrapper.get_training_history()
total_epochs = len(history['epoch'])
max_epochs = wrapper.epochs

if total_epochs < max_epochs:
    print(f"✅ Early stopping activated at epoch {wrapper.best_epoch_}")
    print(f"   Trained {total_epochs}/{max_epochs} epochs")
else:
    print(f"⚠️  No early stopping - trained all {max_epochs} epochs")

print(f"Best validation loss: {wrapper.best_val_loss_:.6f}")
```

### **LightGBM:**
```python
wrapper.fit(X_combined, y)

history = wrapper.get_training_history()
total_iterations = len(history['epoch'])

if wrapper.best_iteration_ is not None:
    print(f"✅ Early stopping activated at iteration {wrapper.best_iteration_}")
    print(f"   Trained {total_iterations} iterations")
else:
    print(f"⚠️  No early stopping")

if wrapper.best_score_ is not None:
    print(f"Best validation score: {wrapper.best_score_}")
```

---

## Summary

✅ **Both wrappers have early stopping parameters fully exposed**
✅ **Parameters are sklearn-compatible** (work with `set_params()`, GridSearchCV)
✅ **Default values are reasonable** but easily overridden
✅ **Works seamlessly with MAPIE** conformal prediction
✅ **Production-ready** with comprehensive monitoring

No changes needed - the functionality is already there! 🎉
