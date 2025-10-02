"""
Download daily OHLCV via yfinance.
Split and save/load parquet (compressed).
"""
import pandas as pd
from pathlib import Path
from typing import Tuple


def download(ticker: str = "^GSPC", start: str = "2000-01-01") -> pd.DataFrame:
    """Return DataFrame with columns open,high,low,close,vol."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for data download. Install with: pip install yfinance"
        )
    
    # Download data using yfinance (much more reliable than pandas-datareader)
    df = yf.download(ticker, start=start, progress=False)
    
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker} from {start}")
    
    # Handle MultiIndex columns (yfinance returns MultiIndex for some cases)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Standardize column names to lowercase with underscores
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    
    return df[["open", "high", "low", "close", "volume"]]


def split(df: pd.DataFrame, ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = int(len(df) * ratio)
    return df.iloc[:n], df.iloc[n:]


def save(df: pd.DataFrame, path: Path, compression: str = "gzip"):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression=compression, index=True)


def load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)