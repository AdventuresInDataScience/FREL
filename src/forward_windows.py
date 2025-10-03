"""
Generate and manage forward windows for reward calculation.

Forward windows are price/volume arrays extending FORWARD from each unique index.
They are:
- Scaled to close[idx-1] (last value of past window) for consistency
- Stored separately from main samples (efficiency - one per unique idx)
- Used during reward calculation to simulate trades

Structure:
    idx: int - Starting index
    forward_open: np.array[forward] - Scaled open prices
    forward_high: np.array[forward] - Scaled high prices
    forward_low: np.array[forward] - Scaled low prices
    forward_close: np.array[forward] - Scaled close prices
    forward_volume: np.array[forward] - Scaled volume
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
import pyarrow as pa
import pyarrow.parquet as pq


def generate_forward_windows(
    df: pd.DataFrame,
    unique_indices: np.ndarray,
    forward: int,
    use_fp16: bool = True
) -> pd.DataFrame:
    """
    Generate scaled forward windows for unique indices.
    
    Args:
        df: Raw OHLCV DataFrame with columns: open, high, low, close, volume
        unique_indices: Array of unique index values to generate windows for
        forward: Number of bars in forward window
        use_fp16: Use float16 for storage (default: True)
    
    Returns:
        DataFrame with columns: idx, forward_open, forward_high, forward_low, 
                               forward_close, forward_volume
    """
    dtype = np.float16 if use_fp16 else np.float32
    
    # Pre-extract arrays for speed
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    close_arr = df['close'].values
    volume_arr = df['volume'].values
    
    # Pre-allocate arrays
    n_unique = len(unique_indices)
    forward_open = np.zeros((n_unique, forward), dtype=dtype)
    forward_high = np.zeros((n_unique, forward), dtype=dtype)
    forward_low = np.zeros((n_unique, forward), dtype=dtype)
    forward_close = np.zeros((n_unique, forward), dtype=dtype)
    forward_volume = np.zeros((n_unique, forward), dtype=dtype)
    
    # Vectorized extraction and scaling
    for i, idx in enumerate(unique_indices):
        # Check bounds
        if idx + forward > len(df):
            # Insufficient data - pad with last known value
            available = len(df) - idx
            
            # Extract available data
            fwd_open = open_arr[idx:idx+available]
            fwd_high = high_arr[idx:idx+available]
            fwd_low = low_arr[idx:idx+available]
            fwd_close = close_arr[idx:idx+available]
            fwd_volume = volume_arr[idx:idx+available]
            
            # Pad with last value
            if available > 0:
                forward_open[i, :available] = fwd_open
                forward_open[i, available:] = fwd_open[-1]
                forward_high[i, :available] = fwd_high
                forward_high[i, available:] = fwd_high[-1]
                forward_low[i, :available] = fwd_low
                forward_low[i, available:] = fwd_low[-1]
                forward_close[i, :available] = fwd_close
                forward_close[i, available:] = fwd_close[-1]
                forward_volume[i, :available] = fwd_volume
                forward_volume[i, available:] = fwd_volume[-1]
        else:
            # Normal case - extract full forward window
            forward_open[i] = open_arr[idx:idx+forward]
            forward_high[i] = high_arr[idx:idx+forward]
            forward_low[i] = low_arr[idx:idx+forward]
            forward_close[i] = close_arr[idx:idx+forward]
            forward_volume[i] = volume_arr[idx:idx+forward]
        
        # Scale to close[idx-1] (reference point - last value of past window)
        if idx > 0:
            reference_close = close_arr[idx-1]
            reference_volume = volume_arr[idx-1]
        else:
            # Edge case: first bar, use itself as reference
            reference_close = close_arr[idx]
            reference_volume = volume_arr[idx]
        
        # Scale OHLC to reference close
        forward_open[i] = forward_open[i] / reference_close
        forward_high[i] = forward_high[i] / reference_close
        forward_low[i] = forward_low[i] / reference_close
        forward_close[i] = forward_close[i] / reference_close
        
        # Scale volume to reference volume (log-space)
        log_volume = np.log1p(forward_volume[i])
        log_ref_volume = np.log1p(reference_volume)
        with np.errstate(divide='ignore', invalid='ignore'):
            forward_volume[i] = np.where(
                log_ref_volume > 0,
                log_volume / log_ref_volume,
                1.0
            )
    
    # Build DataFrame
    df_result = pd.DataFrame({
        'idx': unique_indices
    })
    
    # Add array columns as lists (object dtype)
    df_result['forward_open'] = [row for row in forward_open]
    df_result['forward_high'] = [row for row in forward_high]
    df_result['forward_low'] = [row for row in forward_low]
    df_result['forward_close'] = [row for row in forward_close]
    df_result['forward_volume'] = [row for row in forward_volume]
    
    return df_result


def save_forward_windows(
    forward_windows: pd.DataFrame,
    path: Path,
    compression: str = "snappy",
    compression_level: int = None
) -> None:
    """
    Save forward windows to parquet file.
    
    Args:
        forward_windows: DataFrame from generate_forward_windows()
        path: Output file path
        compression: Compression type (snappy, zstd, gzip)
        compression_level: Compression level (if applicable)
    """
    # Convert to PyArrow table
    table = pa.Table.from_pandas(forward_windows, preserve_index=False)
    
    # Build writer kwargs
    writer_kwargs = {"compression": compression}
    if compression_level is not None:
        writer_kwargs["compression_level"] = int(compression_level)
    
    # Write to file
    pq.write_table(table, path, **writer_kwargs)
    print(f"Saved {len(forward_windows):,} forward windows to {path}")


def load_forward_windows(path: Path) -> pd.DataFrame:
    """
    Load forward windows from parquet file.
    
    Args:
        path: Path to forward windows file
    
    Returns:
        DataFrame with forward windows
    """
    return pd.read_parquet(path)


def create_forward_lookup(forward_windows: pd.DataFrame) -> dict:
    """
    Create fast lookup dictionary: idx -> forward window data.
    
    Args:
        forward_windows: DataFrame from load_forward_windows()
    
    Returns:
        Dict mapping idx to dict of arrays:
        {
            idx: {
                'open': np.array,
                'high': np.array,
                'low': np.array,
                'close': np.array,
                'volume': np.array
            },
            ...
        }
    """
    lookup = {}
    for _, row in forward_windows.iterrows():
        lookup[row['idx']] = {
            'open': row['forward_open'],
            'high': row['forward_high'],
            'low': row['forward_low'],
            'close': row['forward_close'],
            'volume': row['forward_volume']
        }
    return lookup


def validate_forward_windows(
    forward_windows: pd.DataFrame,
    samples: pd.DataFrame
) -> None:
    """
    Validate that forward windows cover all indices in samples.
    
    Args:
        forward_windows: Forward windows DataFrame
        samples: Main samples DataFrame
    
    Raises:
        ValueError: If missing indices or invalid data
    """
    sample_indices = set(samples['idx'].unique())
    forward_indices = set(forward_windows['idx'].values)
    
    missing = sample_indices - forward_indices
    if missing:
        raise ValueError(f"Forward windows missing {len(missing)} indices: {sorted(missing)[:10]}...")
    
    print(f"✓ Forward windows validation passed: {len(forward_indices):,} unique indices")
