"""
Tinker script: build 1 M samples, train Tx or LGB, predict, plot.
"""
import yaml, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from src.dataset import build_dataset
from src.model import build_tx_model, build_lgb_model
from src.scale import MetaScaler
import tensorflow as tf
import joblib, lightgbm as lgb

CFG = yaml.safe_load(open("config/default.yaml"))
n = 1_000_000
print("Building dataset ...")
path = build_dataset(CFG, n, overwrite=False)
df = pd.read_parquet(path)

# ---------- prepare X ----------
price_cols = ["open_scaled", "high_scaled", "low_scaled", "close_scaled", "volume_scaled"]
meta_cols = ["equity", "balance", "position", "sl_dist", "tp_dist", "act_dollar", "act_sl", "act_tp"]

X_price = np.stack([df[col].values for col in price_cols], axis=-1)  # (N, 200, 5)
X_meta = df[meta_cols].values
y = df["y"].values

# ---------- model ----------
if CFG["model_type"] == "transformer":
    model = build_tx_model(price_shape=(200, 5), meta_len=len(meta_cols),
                           d_model=CFG["d_model"], nhead=CFG["nhead"], tx_blocks=CFG["tx_blocks"])
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(CFG["lr"]))
    model.fit([X_price, X_meta], y, batch_size=CFG["batch_size"], epochs=CFG["epochs"], validation_split=0.1, verbose=1)
else:
    X = np.concatenate([X_price.reshape(len(X_price), -1), X_meta], axis=1)
    lgbm = build_lgb_model(linear=True)
    lgbm.fit(X, y)
    model = lgbm
    joblib.dump(model, "lgb_model.pkl")

print("Done")