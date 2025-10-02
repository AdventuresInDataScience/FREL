#%%
"""
PyTorch Training Script: build dataset, train model, evaluate with Predictor.
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.dataset import build_dataset
from src.model import build_model
from src.predictor import Predictor

# Change to project root for relative paths to work
os.chdir(project_root)

#%%
CFG = yaml.safe_load(open("config/default.yaml"))
n = CFG['n_samples']  # Read from config

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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
model = build_model(CFG, device=device)

# Neural models (all except lightgbm)
neural_models = ['transformer', 'informer', 'fedformer', 'patchtst', 
                 'itransformer', 'nbeats', 'nhits']

if model_type in neural_models:
    print(f"Training {model_type} model on {device}...")
    
    # Convert to PyTorch tensors
    X_price_train_t = torch.FloatTensor(X_price_train).to(device)
    X_meta_train_t = torch.FloatTensor(X_meta_train).to(device)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    
    X_price_test_t = torch.FloatTensor(X_price_test).to(device)
    X_meta_test_t = torch.FloatTensor(X_meta_test).to(device)
    y_test_t = torch.FloatTensor(y_test).reshape(-1, 1).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_price_train_t, X_meta_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=CFG["batch_size"], shuffle=True)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"])
    criterion = nn.MSELoss()
    
    # Training loop with progress tracking
    model.train()
    best_loss = float('inf')
    
    for epoch in range(CFG["epochs"]):
        epoch_loss = 0
        n_batches = 0
        
        for batch_price, batch_meta, batch_y in train_loader:
            # Forward pass
            optimizer.zero_grad()
            y_pred = model(batch_price, batch_meta)
            loss = criterion(y_pred, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        
        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                y_pred_val = model(X_price_test_t, X_meta_test_t)
                val_loss = criterion(y_pred_val, y_test_t).item()
            model.train()
            
            print(f"Epoch {epoch+1}/{CFG['epochs']}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                model_path = f"models/{model_type}_model.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': CFG,
                    'model_type': model_type
                }, model_path)
    
    print(f"Saved best model to {model_path}")
    
    # Load best model for evaluation
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred = model(X_price_test_t, X_meta_test_t).cpu().numpy().flatten()
    
    y_test_np = y_test_t.cpu().numpy().flatten()
    mse = np.mean((y_test_np - y_pred) ** 2)
    mae = np.mean(np.abs(y_test_np - y_pred))
    print(f"\nFinal Test MSE: {mse:.6f}, MAE: {mae:.6f}")

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
    
    # Evaluate
    X_test = np.concatenate([
        X_price_test.reshape(len(X_price_test), -1),
        X_meta_test
    ], axis=1)
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    print(f"\nTest MSE: {mse:.6f}, MAE: {mae:.6f}")

else:
    raise ValueError(f"Unknown model_type: {model_type}")

print("Model trained!")

# ---------- Plot Results ----------
plt.figure(figsize=(12, 5))

# Plot 1: Predictions vs True
plt.subplot(1, 2, 1)
plt.scatter(y_test[:1000], y_pred[:1000], alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("True Reward")
plt.ylabel("Predicted Reward")
plt.title(f"{model_type} Predictions vs True Rewards")
plt.grid(True, alpha=0.3)

# Plot 2: Residuals
plt.subplot(1, 2, 2)
residuals = y_test[:1000] - y_pred[:1000]
plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title(f"{model_type} Prediction Errors")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("predictions_pytorch.png", dpi=150)
print("Saved predictions_pytorch.png")

# ---------- GPU Utilization Check ----------
if device == 'cuda':
    print("\n" + "=" * 70)
    print("GPU Memory Usage")
    print("=" * 70)
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print("\nRun 'nvidia-smi' in another terminal to see GPU utilization!")
    print("=" * 70)

print("\nDone!")
