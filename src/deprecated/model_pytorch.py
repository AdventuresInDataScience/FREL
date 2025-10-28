"""
PyTorch Model builders:
  TransformerModel -> PyTorch transformer with CNN front-end
  LGBModel -> lightgbm.LGBMRegressor (gpu, linear_tree) - unchanged
  InformerModel -> informer with prob-sparse attention
  FedFormerModel -> fedformer with frequency-enhanced decomposition
  PatchTSTModel -> patchtst with patch-based attention
  iTransformerModel -> itransformer with inverted dimensionality
  NBeatsModel -> n-beats with basis expansion
  NHiTSModel -> nhits with hierarchical interpolation
  
  build_model() -> factory function to create any model from config
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
from typing import Tuple, Dict, Any


# ========== Base Module for Two-Input Models ==========
class TwoInputModel(nn.Module):
    """Base class for models that take (price, meta) inputs"""
    
    def forward(self, price, meta):
        """
        Args:
            price: (B, T, C) tensor - price sequences
            meta: (B, M) tensor - meta features
        Returns:
            pred: (B, 1) tensor - predictions
        """
        raise NotImplementedError


# ========== Transformer Model ==========
class TransformerModel(TwoInputModel):
    """
    Transformer with CNN front-end and meta skip-connection.
    Inputs:
      price: (B, 200, 5)
      meta: (B, meta_len)
    Output:
      pred: (B, 1)
    """
    def __init__(self, price_shape: Tuple[int, int] = (200, 5), meta_len: int = 8,
                 d_model: int = 128, nhead: int = 4, tx_blocks: int = 4,
                 mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.seq_len, self.n_features = price_shape
        self.d_model = d_model
        
        # CNN encoder: 200 -> 50 tokens
        # PyTorch Conv1d expects (B, C, L) format
        self.cnn1 = nn.Conv1d(self.n_features, 64, kernel_size=8, stride=2, padding=4)
        self.cnn2 = nn.Conv1d(64, d_model, kernel_size=4, stride=2, padding=2)
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 50, 1) * 0.02)
        
        # Transformer encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=mlp_ratio * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tx_blocks)
        
        # Output head with meta skip-connection
        self.fc1 = nn.Linear(d_model + meta_len, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, price, meta):
        # price: (B, T, C) -> (B, C, T) for Conv1d
        x = price.permute(0, 2, 1)
        
        # CNN encoder
        x = F.relu(self.cnn1(x))  # (B, 64, ~100)
        x = F.relu(self.cnn2(x))  # (B, d_model, ~50)
        
        # (B, d_model, L) -> (B, L, d_model) for transformer
        x = x.permute(0, 2, 1)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Transformer blocks
        x = self.transformer(x)  # (B, L, d_model)
        
        # Global average pooling
        z = x.mean(dim=1)  # (B, d_model)
        
        # Skip-connect raw meta
        z = torch.cat([z, meta], dim=1)  # (B, d_model + meta_len)
        z = F.gelu(self.fc1(z))
        z = self.dropout(z)
        out = self.fc2(z)  # (B, 1)
        
        return out


# ========== Informer Model ==========
class InformerModel(TwoInputModel):
    """Informer with ProbSparse self-attention (simplified version)"""
    def __init__(self, price_shape=(200,5), meta_len=8,
                 d_model=128, nhead=4, blocks=4, dropout=0.1):
        super().__init__()
        T, C = price_shape
        self.d_model = d_model
        
        # CNN tokenizer 200 → 50
        self.cnn1 = nn.Conv1d(C, d_model, kernel_size=8, stride=2, padding=4)
        self.cnn2 = nn.Conv1d(d_model, d_model, kernel_size=4, stride=2, padding=2)
        
        # Transformer encoder (ProbSparse attention approximated with standard attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=blocks)
        
        # Output
        self.fc = nn.Linear(d_model + meta_len, 1)
    
    def forward(self, price, meta):
        x = price.permute(0, 2, 1)  # (B, C, T)
        x = F.gelu(self.cnn1(x))
        x = F.gelu(self.cnn2(x))     # (B, d, ~50)
        x = x.permute(0, 2, 1)       # (B, ~50, d)
        
        x = self.encoder(x)
        z = x.mean(dim=1)            # (B, d)
        z = torch.cat([z, meta], dim=1)
        out = self.fc(z)
        return out


# ========== FedFormer Model ==========
class FedFormerModel(TwoInputModel):
    """FedFormer with frequency-enhanced decomposition"""
    def __init__(self, price_shape=(200,5), meta_len=8,
                 d_model=128, nhead=4, blocks=4, dropout=0.1):
        super().__init__()
        T, C = price_shape
        self.d_model = d_model
        
        # Moving average for trend decomposition
        self.avg_pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        
        # CNN for seasonal and trend
        self.cnn1_s = nn.Conv1d(C, d_model, kernel_size=8, stride=2, padding=4)
        self.cnn2_s = nn.Conv1d(d_model, d_model, kernel_size=4, stride=2, padding=2)
        self.cnn1_t = nn.Conv1d(C, d_model, kernel_size=8, stride=2, padding=4)
        self.cnn2_t = nn.Conv1d(d_model, d_model, kernel_size=4, stride=2, padding=2)
        
        # Output
        self.fc = nn.Linear(d_model + meta_len, 1)
    
    def forward(self, price, meta):
        # Decompose into trend and seasonal
        x = price.permute(0, 2, 1)  # (B, C, T)
        trend = self.avg_pool(x)
        seasonal = x - trend
        
        # CNN on both branches
        z_s = F.gelu(self.cnn1_s(seasonal))
        z_s = F.gelu(self.cnn2_s(z_s))  # (B, d, L)
        
        z_t = F.gelu(self.cnn1_t(trend))
        z_t = F.gelu(self.cnn2_t(z_t))
        
        # Fourier attention (simplified - just FFT filtering)
        z_s_fft = torch.fft.rfft(z_s, dim=-1)
        # Keep top k/2 modes
        k = z_s_fft.shape[-1] // 2
        z_s_fft = z_s_fft[..., :k]
        z_s = torch.fft.irfft(z_s_fft, n=z_s.shape[-1], dim=-1)
        
        # Combine
        z = z_s + z_t
        z = z.mean(dim=-1)  # (B, d) - average over time
        z = torch.cat([z, meta], dim=1)
        out = self.fc(z)
        return out


# ========== PatchTST Model ==========
class PatchTSTModel(TwoInputModel):
    """PatchTST with patch-based attention"""
    def __init__(self, price_shape=(200,5), meta_len=8,
                 patch_len=16, stride=8,
                 d_model=128, nhead=4, blocks=4, dropout=0.1):
        super().__init__()
        T, C = price_shape
        self.patch_len = patch_len
        self.stride = stride
        
        # Calculate number of patches
        n_patches = (T - patch_len) // stride + 1
        
        # Patch embedding
        self.patch_embed = nn.Linear(patch_len * C, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, 1) * 0.02)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=blocks)
        
        # Output
        self.fc = nn.Linear(d_model + meta_len, 1)
    
    def forward(self, price, meta):
        B, T, C = price.shape
        
        # Extract patches using unfold
        # Flatten last two dims: (B, T, C) -> (B, T*C)
        x = price.reshape(B, T * C)
        # Unfold: (B, T*C) -> (B, patch_len*C, n_patches)
        patches = x.unfold(1, self.patch_len * C, self.stride * C)
        # (B, n_patches, patch_len*C)
        patches = patches.transpose(1, 2)
        
        # Embed patches
        x = self.patch_embed(patches)  # (B, n_patches, d_model)
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        z = x.mean(dim=1)  # (B, d_model)
        z = torch.cat([z, meta], dim=1)
        out = self.fc(z)
        return out


# ========== iTransformer Model ==========
class iTransformerModel(TwoInputModel):
    """iTransformer with inverted dimensionality"""
    def __init__(self, price_shape=(200,5), meta_len=8,
                 d_model=128, nhead=4, blocks=4, dropout=0.1):
        super().__init__()
        T, C = price_shape
        
        # Invert: each variable becomes a token
        self.variable_embed = nn.Linear(T, d_model)
        
        # Transformer across variables
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=blocks)
        
        # Output
        self.fc = nn.Linear(d_model + meta_len, 1)
    
    def forward(self, price, meta):
        # Invert: (B, T, C) -> (B, C, T)
        x = price.permute(0, 2, 1)
        
        # Each variable is a token: (B, C, T) -> (B, C, d_model)
        x = self.variable_embed(x)
        
        # Self-attention across variables
        x = self.transformer(x)
        z = x.mean(dim=1)  # (B, d_model)
        z = torch.cat([z, meta], dim=1)
        out = self.fc(z)
        return out


# ========== N-BEATS Model ==========
class NBeatsModel(TwoInputModel):
    """N-BEATS with basis expansion (simplified)"""
    def __init__(self, price_shape=(200,5), meta_len=8,
                 stack_types=['trend','seasonality','generic'],
                 n_blocks=[1,1,1], mlp_units=512,
                 shares_weights=False, dropout=0.0):
        super().__init__()
        T, C = price_shape
        input_dim = T * C + meta_len
        
        # Simplified: single MLP stack
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_units),
            nn.ReLU(),
            nn.Linear(mlp_units, mlp_units),
            nn.ReLU(),
            nn.Linear(mlp_units, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, price, meta):
        x = self.flatten(price)  # (B, T*C)
        x = torch.cat([x, meta], dim=1)  # (B, T*C + meta_len)
        out = self.mlp(x)
        return out


# ========== N-HiTS Model ==========
class NHiTSModel(TwoInputModel):
    """N-HiTS with hierarchical interpolation (simplified)"""
    def __init__(self, price_shape=(200,5), meta_len=8,
                 pools=[1,2,4], mlp_units=512, dropout=0.0):
        super().__init__()
        T, C = price_shape
        
        # Multi-rate pooling branches
        self.pools = nn.ModuleList([
            nn.MaxPool1d(kernel_size=p, stride=p) for p in pools
        ])
        
        # MLP for each branch
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear((T // p) * C + meta_len, mlp_units),
                nn.ReLU(),
                nn.Linear(mlp_units, mlp_units),
                nn.ReLU(),
                nn.Linear(mlp_units, T * C)
            ) for p in pools
        ])
        
        # Final output
        self.fc = nn.Linear(T * C + meta_len, 1)
    
    def forward(self, price, meta):
        B, T, C = price.shape
        x = price.permute(0, 2, 1)  # (B, C, T)
        
        forecasts = []
        for pool, mlp in zip(self.pools, self.mlps):
            # Downsample
            x_pool = pool(x)  # (B, C, T/p)
            x_pool = x_pool.reshape(B, -1)  # Flatten
            x_pool = torch.cat([x_pool, meta], dim=1)
            # Forecast at original resolution
            f = mlp(x_pool)  # (B, T*C)
            forecasts.append(f)
        
        # Sum all forecasts
        forecast = sum(forecasts)
        forecast = torch.cat([forecast, meta], dim=1)
        out = self.fc(forecast)
        return out


# ========== LightGBM (unchanged) ==========
def build_lgb_model(linear: bool = True):
    """LightGBM model - works with both PyTorch and TensorFlow pipelines"""
    params = dict(
        objective="regression",
        metric="rmse",
        device="gpu",
        gpu_platform_id=0,
        gpu_device_id=0,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=256,
        subsample=0.8,
        linear_tree=linear,
        linear_lambda=0.01,
        verbose=-1,
    )
    return lgb.LGBMRegressor(**params)


# ========== Factory Function ==========
def build_model(cfg: Dict[str, Any], device='cuda'):
    """
    Factory function to build any model based on configuration.
    
    Args:
        cfg: Configuration dict with model_type and hyperparameters
        device: 'cuda' or 'cpu'
    
    Returns:
        PyTorch model or LightGBM model
    
    Example:
        cfg = {
            'model_type': 'transformer',
            'lookback': 200,
            'd_model': 128,
            'nhead': 4,
            'tx_blocks': 4,
            'dropout': 0.1,
            ...
        }
        model = build_model(cfg, device='cuda')
    """
    model_type = cfg['model_type'].lower()
    price_shape = (cfg.get('lookback', 200), 5)
    meta_len = 8  # Standard meta features length
    
    # Common parameters for neural models
    d_model = cfg.get('d_model', 128)
    nhead = cfg.get('nhead', 4)
    blocks = cfg.get('tx_blocks', 4)
    dropout = cfg.get('dropout', 0.1)
    
    # Build model based on type
    if model_type == 'transformer':
        model = TransformerModel(
            price_shape=price_shape,
            meta_len=meta_len,
            d_model=d_model,
            nhead=nhead,
            tx_blocks=blocks,
            mlp_ratio=cfg.get('mlp_ratio', 4),
            dropout=dropout
        )
    
    elif model_type == 'informer':
        model = InformerModel(
            price_shape=price_shape,
            meta_len=meta_len,
            d_model=d_model,
            nhead=nhead,
            blocks=blocks,
            dropout=dropout
        )
    
    elif model_type == 'fedformer':
        model = FedFormerModel(
            price_shape=price_shape,
            meta_len=meta_len,
            d_model=d_model,
            nhead=nhead,
            blocks=blocks,
            dropout=dropout
        )
    
    elif model_type == 'patchtst':
        model = PatchTSTModel(
            price_shape=price_shape,
            meta_len=meta_len,
            patch_len=cfg.get('patch_len', 16),
            stride=cfg.get('patch_stride', 8),
            d_model=d_model,
            nhead=nhead,
            blocks=blocks,
            dropout=dropout
        )
    
    elif model_type == 'itransformer':
        model = iTransformerModel(
            price_shape=price_shape,
            meta_len=meta_len,
            d_model=d_model,
            nhead=nhead,
            blocks=blocks,
            dropout=dropout
        )
    
    elif model_type == 'nbeats':
        model = NBeatsModel(
            price_shape=price_shape,
            meta_len=meta_len,
            stack_types=cfg.get('nbeats_stack_types', ['trend', 'seasonality', 'generic']),
            n_blocks=cfg.get('nbeats_n_blocks', [1, 1, 1]),
            mlp_units=cfg.get('nbeats_mlp_units', 512),
            shares_weights=cfg.get('nbeats_shares_weights', False),
            dropout=dropout
        )
    
    elif model_type == 'nhits':
        model = NHiTSModel(
            price_shape=price_shape,
            meta_len=meta_len,
            pools=cfg.get('nhits_pools', [1, 2, 4]),
            mlp_units=cfg.get('nhits_mlp_units', 512),
            dropout=dropout
        )
    
    elif model_type == 'lightgbm':
        return build_lgb_model(linear=cfg.get('lgb_linear_tree', True))
    
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available: transformer, informer, fedformer, patchtst, "
            f"itransformer, nbeats, nhits, lightgbm"
        )
    
    # Move neural models to device
    if model_type != 'lightgbm':
        model = model.to(device)
    
    return model
