#%%
"""
Tinker script: build dataset, train model, evaluate with Predictor.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(r"c:\Users\malha\Documents\Projects\FREL")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.dataset import build_dataset
from src.model import build_model
from src.predictor import Predictor
import tensorflow as tf

# Change to project root for relative paths to work
os.chdir(project_root)

#%%
CFG = yaml.safe_load(open("config/default.yaml"))
n = CFG['n_samples']  # Read from config

print(f"Using model type: {CFG['model_type']}")

print("Building dataset...")
path = build_dataset(CFG, n, overwrite=False)
df = pd.read_parquet(path)

#%%
# Split train/test
split_idx = int(CFG['train_ratio'] * len(df))
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

# ---------- Prepare X, y ----------
price_cols = ["open_scaled", "high_scaled", "low_scaled", "close_scaled", "volume_scaled"]
meta_cols = ["equity", "balance", "position", "sl_dist", "tp_dist", 
             "act_dollar", "act_sl", "act_tp"]

# Training data (pre-scaled)
# Fix: Explicitly stack object arrays to ensure correct shape (n_samples, lookback, 5)
X_price_train = np.stack([np.stack(train_df[col].values) for col in price_cols], axis=-1)
X_meta_train = train_df[meta_cols].values
y_train = train_df["y"].values

# Verify shapes
assert X_price_train.shape == (len(train_df), CFG['lookback'], 5), \
    f"X_price_train shape mismatch: got {X_price_train.shape}, expected ({len(train_df)}, {CFG['lookback']}, 5)"
assert X_meta_train.shape == (len(train_df), 8), \
    f"X_meta_train shape mismatch: got {X_meta_train.shape}, expected ({len(train_df)}, 8)"

# Test data (pre-scaled)
X_price_test = np.stack([np.stack(test_df[col].values) for col in price_cols], axis=-1)
X_meta_test = test_df[meta_cols].values
y_test = test_df["y"].values

# Verify shapes
assert X_price_test.shape == (len(test_df), CFG['lookback'], 5), \
    f"X_price_test shape mismatch: got {X_price_test.shape}, expected ({len(test_df)}, {CFG['lookback']}, 5)"
assert X_meta_test.shape == (len(test_df), 8), \
    f"X_meta_test shape mismatch: got {X_meta_test.shape}, expected ({len(test_df)}, 8)"

print(f"Train: {len(train_df)}, Test: {len(test_df)}")
print(f"X_price_train shape: {X_price_train.shape}")
print(f"X_meta_train shape: {X_meta_train.shape}")

# ---------- Train model ----------
model_type = CFG["model_type"]
print(f"Building {model_type} model...")

# Build model using factory function
model = build_model(CFG)

# Neural models (all except lightgbm)
neural_models = ['transformer', 'informer', 'fedformer', 'patchtst', 
                 'itransformer', 'nbeats', 'nhits']

if model_type in neural_models:
    # Compile and train neural model
    model.compile(
        loss="mse", 
        optimizer=tf.keras.optimizers.Adam(CFG["lr"]),
        jit_compile=True  # Enable XLA compilation for faster training
    )
    
    print(f"Training {model_type} model...")
    model.fit(
        [X_price_train, X_meta_train],
        y_train,
        batch_size=CFG["batch_size"],
        epochs=CFG["epochs"],
        validation_split=0.1,
        verbose=1
    )
    
    # Save model
    model_path = f"models/{model_type}_model.h5"
    model.save(model_path)
    print(f"Saved model to {model_path}")

elif model_type == "lightgbm":
    # Train LightGBM model
    print("Training LightGBM model...")
    X_train = np.concatenate([
        X_price_train.reshape(len(X_price_train), -1), 
        X_meta_train
    ], axis=1)
    model.fit(X_train, y_train)
    
    # Save model
    import joblib
    model_path = "models/lgb_model.pkl"
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

else:
    raise ValueError(f"Unknown model_type: {model_type}")

print("Model trained!")

# ---------- Create Predictor ----------
predictor = Predictor.from_checkpoint(
    model_path=model_path,
    scaler_path="data/meta_scaler.json",
    cfg=CFG,
    model_type=CFG["model_type"]
)

# ---------- Test on pre-scaled data ----------
print("\n=== Testing on pre-scaled test data ===")
y_pred = predictor.predict(X_price=X_price_test, X_meta=X_meta_test, raw=False)
mse = np.mean((y_test - y_pred) ** 2)
mae = np.mean(np.abs(y_test - y_pred))
print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")

# Plot predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test[:1000], y_pred[:1000], alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("True Reward")
plt.ylabel("Predicted Reward")
plt.title("Model Predictions vs True Rewards")
plt.savefig("predictions.png")
print("Saved predictions.png")

# ---------- Test on raw data ----------
print("\n=== Testing optimal action search on raw data ===")

# Load raw price data
raw_df = pd.read_parquet(f"data/raw_{CFG['ticker']}.parquet")
close_prices = raw_df["close"].values

# Take a test sample
test_idx = 1000
ohlcv_window = {
    "open": raw_df["open"].values[test_idx:test_idx + CFG["lookback"]],
    "high": raw_df["high"].values[test_idx:test_idx + CFG["lookback"]],
    "low": raw_df["low"].values[test_idx:test_idx + CFG["lookback"]],
    "close": raw_df["close"].values[test_idx:test_idx + CFG["lookback"]],
    "volume": raw_df["volume"].values[test_idx:test_idx + CFG["lookback"]],
}

state = {
    "equity": 5e4,
    "balance": 5e4,
    "position": 0.0,
    "sl_dist": 0.02,
    "tp_dist": 0.04
}

# Compare model's predicted optimal vs true optimal
comparison = predictor.compare_predicted_vs_true(
    close_prices,
    idx=test_idx + CFG["lookback"],
    ohlcv_window=ohlcv_window,
    state=state,
    method="sample",
    n_samples=500,
    seed=42
)

print("\nModel's predicted optimal action:")
print(comparison["predicted_action"])
print("\nTrue optimal action:")
print(comparison["true_optimal_action"])
print(f"\nOptimality gap: {comparison['optimality_gap']:.6f}")
print(f"Model achieved {(1 - comparison['optimality_gap'] / comparison['true_optimal_action']['reward']) * 100:.2f}% of optimal reward")

print("\nDone!")