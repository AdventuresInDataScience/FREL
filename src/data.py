"""
Download daily OHLCV via pandas-datareader.
Split and save/load parquet (compressed).
"""
import pandas_datareader as pdr
import pandas as pd
from pathlib import Path
from typing import Tuple


def download(ticker: str = "^GSPC", start: str = "2000-01-01") -> pd.DataFrame:
    """Return DataFrame with columns open,high,low,close,vol."""
    df = pdr.get_data_yahoo(ticker, start=start)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]]


def split(df: pd.DataFrame, ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = int(len(df) * ratio)
    return df.iloc[:n], df.iloc[n:]


def save(df: pd.DataFrame, path: Path, compression: str = "gzip"):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression=compression, index=True)


def load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)