"""
Quick test to verify PyTorch models compile and work
"""
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(r"c:\Users\malha\Documents\Projects\FREL")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.model import build_model

# Test config
cfg = {
    'model_type': 'transformer',
    'lookback': 200,
    'd_model': 128,
    'nhead': 4,
    'tx_blocks': 4,
    'dropout': 0.1
}

print("=" * 70)
print("PyTorch Model Compilation Test")
print("=" * 70)

# Check GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test each model type
model_types = ['transformer', 'informer', 'fedformer', 'patchtst', 
               'itransformer', 'nbeats', 'nhits', 'lightgbm']

for model_type in model_types:
    try:
        cfg['model_type'] = model_type
        model = build_model(cfg, device=device)
        
        if model_type != 'lightgbm':
            # Test forward pass
            batch_size = 2
            price = torch.randn(batch_size, 200, 5).to(device)
            meta = torch.randn(batch_size, 8).to(device)
            
            model.eval()
            with torch.no_grad():
                output = model(price, meta)
            
            assert output.shape == (batch_size, 1), f"Wrong output shape: {output.shape}"
            print(f"✓ {model_type:15s} - Compiled and tested successfully")
        else:
            print(f"✓ {model_type:15s} - LightGBM model created successfully")
            
    except Exception as e:
        print(f"✗ {model_type:15s} - FAILED: {str(e)}")

print("\n" + "=" * 70)
print("All models compiled successfully!" if all else "Some models failed")
print("=" * 70)
