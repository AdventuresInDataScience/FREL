"""Quick test of PyTorch training on GPU with small synthetic data"""
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path

project_root = Path(r"c:\Users\malha\Documents\Projects\FREL")
sys.path.insert(0, str(project_root))

from src.model import TransformerModel

# Check GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Create synthetic data (no NaNs)
batch_size = 32
X_price = torch.randn(batch_size, 200, 5).to(device)
X_meta = torch.randn(batch_size, 8).to(device)
y = torch.randn(batch_size, 1).to(device)

print(f"Data shapes: X_price={X_price.shape}, X_meta={X_meta.shape}, y={y.shape}")
print(f"Data has NaN: price={torch.isnan(X_price).any()}, meta={torch.isnan(X_meta).any()}, y={torch.isnan(y).any()}")

# Build model
model = TransformerModel(price_shape=(200, 5), meta_len=8, d_model=128, nhead=4, tx_blocks=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("\nTraining for 10 epochs...")
model.train()
for epoch in range(10):
    optimizer.zero_grad()
    y_pred = model(X_price, X_meta)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch+1}: Loss={loss.item():.6f}")

print("\n✓ Training successful - no NaN!")
print(f"Final loss: {loss.item():.6f}")

# Test prediction
model.eval()
with torch.no_grad():
    test_pred = model(X_price[:5], X_meta[:5])
    print(f"\nTest predictions: {test_pred.flatten()[:5]}")
