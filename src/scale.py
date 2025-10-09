"""
Scaling utilities:
  1) per-window endpoint scaling for OHLCV
  2) global min-max for meta fields
  3) reversible via JSON store
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any


# ---------- window scalers ----------
def scale_ohlcv_window(ohlcv: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """OHLCV dict -> scaled clone; close[-1] used as divisor."""
    close_last = ohlcv["close"][-1]
    vol_last = ohlcv["volume"][-1]
    out = {}
    for name in ["open", "high", "low", "close"]:
        out[name] = ohlcv[name] / close_last
    out["volume"] = np.log1p(ohlcv["volume"]) / np.log1p(vol_last)
    return out


# ---------- meta scaler ----------
class MetaScaler:
    def __init__(self, kind: str = "minmax"):
        self.kind = kind
        self.stats: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame, cols: list):
        for c in cols:
            x = df[c].values
            if self.kind == "minmax":
                self.stats[c] = dict(min=float(x.min()), max=float(x.max()))
            else:  # std
                self.stats[c] = dict(mean=float(x.mean()), std=float(x.std()))

    def transform(self, df: pd.DataFrame, cols: list, epsilon: float = 1e-8) -> pd.DataFrame:
        df = df.copy()
        for c in cols:
            s = self.stats[c]
            if self.kind == "minmax":
                df[c] = (df[c] - s["min"]) / (s["max"] - s["min"] + epsilon)
            else:
                df[c] = (df[c] - s["mean"]) / (s["std"] + epsilon)
        return df

    def inverse(self, col: str, vals: np.ndarray) -> np.ndarray:
        s = self.stats[col]
        if self.kind == "minmax":
            return vals * (s["max"] - s["min"]) + s["min"]
        else:
            return vals * s["std"] + s["mean"]
    
    def inverse_transform_dict(self, scaled_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Inverse transform a dictionary of scaled values.
        
        Args:
            scaled_dict: Dict mapping column names to scaled values
            
        Returns:
            Dict with unscaled values
        """
        unscaled = {}
        for col, val in scaled_dict.items():
            if col in self.stats:
                s = self.stats[col]
                if self.kind == "minmax":
                    unscaled[col] = val * (s["max"] - s["min"]) + s["min"]
                else:
                    unscaled[col] = val * s["std"] + s["mean"]
            else:
                # Column not in stats - pass through unchanged
                unscaled[col] = val
        return unscaled

    def save(self, path: Path):
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.stats))

    def load(self, path: Path):
        self.stats = json.loads(path.read_text())