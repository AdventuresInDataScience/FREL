"""
Model builders:
  build_tx_model() -> functional TF model (transformer)
  build_lgb_model() -> lightgbm.LGBMRegressor (gpu, linear_tree)
  build_informer() -> informer with prob-sparse attention
  build_fedformer() -> fedformer with frequency-enhanced decomposition
  build_patchtst() -> patchtst with patch-based attention
  build_itransformer() -> itransformer with inverted dimensionality
  build_nbeats() -> n-beats with basis expansion
  build_nhits() -> nhits with hierarchical interpolation
  
  build_model() -> factory function to create any model from config
"""
import numpy as np
import tensorflow as tf
import lightgbm as lgb
from typing import Tuple, Dict, Any


# ---------- transformer ----------
def build_tx_model(price_shape: Tuple[int, int] = (200, 5), meta_len: int = 10,
                   d_model: int = 128, nhead: int = 4, tx_blocks: int = 4,
                   mlp_ratio: int = 4, dropout: float = 0.1):
    """
    Functional API transformer with CNN front-end and meta skip-connection.
    Inputs:
      price_in : (B, 200, 5)
      meta_in  : (B, meta_len)
    Output:
      pred     : (B,)
    """
    price_in = tf.keras.Input(shape=price_shape, name="price")
    meta_in = tf.keras.Input(shape=(meta_len,), name="meta")

    # CNN encoder: 200 -> 50 tokens
    x = tf.keras.layers.Conv1D(64, 8, strides=2, padding="same", activation="relu")(price_in)
    x = tf.keras.layers.Conv1D(d_model, 4, strides=2, padding="same", activation="relu")(x)  # (B, 50, d_model)

    # Positional encoding
    seq_len = tf.shape(x)[1]
    pos = tf.range(seq_len, dtype=tf.float32)[None, :, None]  # (1,50,1)
    x = x + pos

    # Transformer encoder blocks
    for _ in range(tx_blocks):
        attn = tf.keras.layers.MultiHeadAttention(nhead, d_model // nhead, dropout=dropout)(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attn)
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_ratio * d_model, activation="gelu"),
            tf.keras.layers.Dense(d_model)
        ])
        x = tf.keras.layers.LayerNormalization()(x + ffn(x))

    z = tf.keras.layers.GlobalAveragePooling1D()(x)  # (B, d_model)

    # skip-connect raw meta
    z = tf.keras.layers.Concatenate()([z, meta_in])
    z = tf.keras.layers.Dense(128, activation="gelu")(z)
    z = tf.keras.layers.Dropout(dropout)(z)
    out = tf.keras.layers.Dense(1)(z)

    return tf.keras.Model([price_in, meta_in], out)


# ---------- lightgbm ----------
def build_lgb_model(linear: bool = True):
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

def build_informer(price_shape=(200,5), meta_len=10,
                   d_model=128, nhead=4, blocks=4, dropout=0.1):
    from tensorflow_addons.layers import MultiHeadAttention as SparseMHA

    price = tf.keras.Input(shape=price_shape, name='price')
    meta  = tf.keras.Input(shape=(meta_len,), name='meta')

    # CNN tokenizer 200 → 50
    x = tf.keras.layers.Conv1D(d_model, 8, strides=2, padding='same', activation='gelu')(price)
    x = tf.keras.layers.Conv1D(d_model, 4, strides=2, padding='same', activation='gelu')(x)  # (B,50,d)

    # Prob-Sparse encoder
    for _ in range(blocks):
        attn = SparseMHA(nhead, d_model//nhead, dropout=dropout)(x, x)   # log-linear
        x = tf.keras.layers.LayerNormalization()(x + attn)
        ffn = tf.keras.Sequential([tf.keras.layers.Dense(d_model*4, activation='gelu'),
                                   tf.keras.layers.Dense(d_model)])
        x = tf.keras.layers.LayerNormalization()(x + ffn(x))

    z = tf.keras.layers.GlobalAveragePooling1D()(x)
    z = tf.keras.layers.Concatenate()([z, meta])
    out = tf.keras.layers.Dense(1)(z)
    return tf.keras.Model([price, meta], out)

def build_fedformer(price_shape=(200,5), meta_len=10,
                    d_model=128, nhead=4, blocks=4, dropout=0.1):
    price = tf.keras.Input(shape=price_shape, name='price')
    meta  = tf.keras.Input(shape=(meta_len,), name='meta')

    # lightweight decompose: moving-avg trend + residual seasonal
    trend = tf.keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(price)
    seasonal = price - trend

    # CNN on both forks
    def _cnn(x):
        x = tf.keras.layers.Conv1D(d_model, 8, strides=2, padding='same', activation='gelu')(x)
        return tf.keras.layers.Conv1D(d_model, 4, strides=2, padding='same', activation='gelu')(x)
    z_s = _cnn(seasonal)
    z_t = _cnn(trend)

    # 1-D Fourier attention (only top k modes)
    z_s = tf.signal.rfft(z_s, fft_length=[z_s.shape[1]])[:, :, :d_model//2]
    z_s = tf.signal.irfft(z_s, fft_length=[z_s.shape[1]])

    z = z_s + z_t
    z = tf.keras.layers.GlobalAveragePooling1D()(z)
    z = tf.keras.layers.Concatenate()([z, meta])
    out = tf.keras.layers.Dense(1)(z)
    return tf.keras.Model([price, meta], out)

def build_patchtst(price_shape=(200,5), meta_len=10,
                   patch_len=16, stride=8,
                   d_model=128, nhead=4, blocks=4, dropout=0.1):
    price = tf.keras.Input(shape=price_shape, name='price')   # (B,T,C)
    meta  = tf.keras.Input(shape=(meta_len,), name='meta')

    # patch: (B, N_patches, patch_len*C)
    B, T, C = tf.shape(price)[0], price_shape[0], price_shape[1]
    patches = tf.image.extract_patches(
        images=tf.expand_dims(price, axis=3),
        sizes=[1, patch_len, 1, 1],
        strides=[1, stride, 1, 1],
        rates=[1, 1, 1, 1],
        padding='VALID')
    n_patches = patches.shape[1]
    patches = tf.reshape(patches, [B, n_patches, patch_len*C])

    # linear projection + positional
    x = tf.keras.layers.Dense(d_model)(patches)
    pos = tf.range(n_patches, dtype=tf.float32)[None, :, None]
    x = x + pos

    # transformer blocks
    for _ in range(blocks):
        attn = tf.keras.layers.MultiHeadAttention(nhead, d_model//nhead, dropout=dropout)(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attn)
        ffn = tf.keras.Sequential([tf.keras.layers.Dense(d_model*4, activation='gelu'),
                                   tf.keras.layers.Dense(d_model)])
        x = tf.keras.layers.LayerNormalization()(x + ffn(x))

    z = tf.keras.layers.GlobalAveragePooling1D()(x)
    z = tf.keras.layers.Concatenate()([z, meta])
    out = tf.keras.layers.Dense(1)(z)
    return tf.keras.Model([price, meta], out)

def build_itransformer(price_shape=(200,5), meta_len=10,
                       d_model=128, nhead=4, blocks=4, dropout=0.1):
    price = tf.keras.Input(shape=price_shape, name='price')  # (B,T,C)
    meta  = tf.keras.Input(shape=(meta_len,), name='meta')

    # invert: (B,T,C) -> (B,C,T)
    x = tf.transpose(price, [0, 2, 1])  # now each *variable* is a token
    x = tf.keras.layers.Dense(d_model)(x)  # (B,C,d_model)

    # self-attn across variables
    for _ in range(blocks):
        attn = tf.keras.layers.MultiHeadAttention(nhead, d_model//nhead, dropout=dropout)(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attn)
        ffn = tf.keras.Sequential([tf.keras.layers.Dense(d_model*4, activation='gelu'),
                                   tf.keras.layers.Dense(d_model)])
        x = tf.keras.layers.LayerNormalization()(x + ffn(x))

    z = tf.keras.layers.GlobalAveragePooling1D()(x)  # (B,d_model)
    z = tf.keras.layers.Concatenate()([z, meta])
    out = tf.keras.layers.Dense(1)(z)
    return tf.keras.Model([price, meta], out)

# --- N-BEATS ---
def build_nbeats(price_shape=(200,5), meta_len=10,
                 stack_types=['trend','seasonality','generic'],
                 n_blocks=[1,1,1], mlp_units=512,
                 shares_weights=False, dropout=0.0):
    from tensorflow.keras import layers, Input, Model
    T, C = price_shape
    price = Input(shape=price_shape, name='price')
    meta  = Input(shape=(meta_len,), name='meta')

    # flatten price + concat meta
    x = layers.Flatten()(price)          # (B, T*C)
    x = layers.Concatenate()([x, meta])  # (B, T*C + meta_len)

    theta_size = 4*max(T, 4)   # basis dim (trend+season)
    forecast = 0
    backcast = 0
    for st, nb in zip(stack_types, n_blocks):
        for b in range(nb):
            # Block MLP
            blk = layers.Dense(mlp_units, activation='relu')(x)
            blk = layers.Dense(mlp_units, activation='relu')(blk)
            theta = layers.Dense(theta_size, activation='linear')(blk)

            if st == 'trend':
                # polynomial basis 1,2,3...
                t = np.arange(T*C) / (T*C)
                basis = np.stack([t**i for i in range(1, theta_size//2+1)], axis=-1)  # (T*C, basis)
            elif st == 'seasonality':
                # harmonic basis
                t = np.arange(T*C) / (T*C) * 2*np.pi
                basis = np.concatenate([np.sin(t*(i+1))[:,None] for i in range(theta_size//2)], axis=-1)
            else:  # generic
                basis = layers.Dense(T*C, activation='linear')(theta)  # identity basis

            f = layers.Dense(T*C, activation='linear')(theta)  # forecast
            forecast += f
            backcast += basis @ theta[..., None]   # residual

            x = x - layers.Flatten()(backcast)     # doubly residual

    out = layers.Dense(1)(layers.Concatenate()([layers.Flatten()(forecast), meta]))
    return Model([price, meta], out)

# --- NHITS (hierarchical multi-rate) ---
def build_nhits(price_shape=(200,5), meta_len=10,
                pools=[1,2,4], mlp_units=512, dropout=0.0):
    from tensorflow.keras import layers, Input, Model
    import tensorflow as tf
    T, C = price_shape
    price = Input(shape=price_shape, name='price')
    meta  = Input(shape=(meta_len,), name='meta')

    forecast = 0
    for p in pools:  # multi-rate pooling
        x = tf.keras.layers.MaxPool1D(pool_size=p, strides=p, padding='same')(price)  # downsample
        x = layers.Flatten()(x)
        x = layers.Concatenate()([x, meta])
        # MLP block
        x = layers.Dense(mlp_units, activation='relu')(x)
        x = layers.Dense(mlp_units, activation='relu')(x)
        f = layers.Dense(T*C)(x)  # forecast at *original* resolution
        forecast += f

    out = layers.Dense(1)(layers.Concatenate()([forecast, meta]))
    return Model([price, meta], out)


# ---------- Factory Function ----------
def build_model(cfg: Dict[str, Any]):
    """
    Factory function to build any model based on configuration.
    
    Args:
        cfg: Configuration dict with model_type and hyperparameters
    
    Returns:
        Keras Model or LightGBM model
    
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
        model = build_model(cfg)
    """
    model_type = cfg['model_type'].lower()
    price_shape = (cfg.get('lookback', 200), 5)
    meta_len = 8  # Standard meta features length
    
    # Common parameters for neural models
    d_model = cfg.get('d_model', 128)
    nhead = cfg.get('nhead', 4)
    blocks = cfg.get('tx_blocks', 4)
    dropout = cfg.get('dropout', 0.1)
    
    if model_type == 'transformer':
        return build_tx_model(
            price_shape=price_shape,
            meta_len=meta_len,
            d_model=d_model,
            nhead=nhead,
            tx_blocks=blocks,
            mlp_ratio=cfg.get('mlp_ratio', 4),
            dropout=dropout
        )
    
    elif model_type == 'informer':
        return build_informer(
            price_shape=price_shape,
            meta_len=meta_len,
            d_model=d_model,
            nhead=nhead,
            blocks=blocks,
            dropout=dropout
        )
    
    elif model_type == 'fedformer':
        return build_fedformer(
            price_shape=price_shape,
            meta_len=meta_len,
            d_model=d_model,
            nhead=nhead,
            blocks=blocks,
            dropout=dropout
        )
    
    elif model_type == 'patchtst':
        return build_patchtst(
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
        return build_itransformer(
            price_shape=price_shape,
            meta_len=meta_len,
            d_model=d_model,
            nhead=nhead,
            blocks=blocks,
            dropout=dropout
        )
    
    elif model_type == 'nbeats':
        return build_nbeats(
            price_shape=price_shape,
            meta_len=meta_len,
            stack_types=cfg.get('nbeats_stack_types', ['trend', 'seasonality', 'generic']),
            n_blocks=cfg.get('nbeats_n_blocks', [1, 1, 1]),
            mlp_units=cfg.get('nbeats_mlp_units', 512),
            shares_weights=cfg.get('nbeats_shares_weights', False),
            dropout=dropout
        )
    
    elif model_type == 'nhits':
        return build_nhits(
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

