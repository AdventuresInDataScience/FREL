"""
Synthetic trade-state + action generation.
build_samples() produces a DataFrame ready for reward/y-label computation.

NEW STRUCTURE:
- Dual positions: long_value, short_value (simultaneous hedging)
- SL/TP in multiplier notation: long_sl=0.95 (5% stop), long_tp=1.10 (10% profit)
- Actions specify target state (not deltas)
- Hold states (10% flat) and hold actions (20% no change)
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import tqdm
from joblib import Parallel, delayed
import multiprocessing
import time
from scipy import stats


def _sample_position_value(rng: np.random.Generator, cfg: dict, size: int = None) -> np.ndarray:
    """
    Sample position values from log-normal distribution.
    
    Ensures most values near mean, with long tail for large positions.
    Returns 0 with appropriate probability (no position).
    """
    mean = cfg.get("position_value_mean", 10000)
    sigma = cfg.get("position_value_sigma", 1.0)
    
    # Log-normal distribution
    # ln(X) ~ Normal(mu, sigma) where mu = ln(mean) - sigma^2/2
    mu = np.log(mean) - sigma**2 / 2
    
    if size is None:
        value = rng.lognormal(mu, sigma)
    else:
        value = rng.lognormal(mu, sigma, size=size)
    
    return value


def _sample_sl_tp_multiplier(rng: np.random.Generator, cfg: dict, 
                             is_long: bool, is_sl: bool, size: int = None) -> np.ndarray:
    """
    Sample SL/TP multipliers from truncated normal distribution.
    
    Args:
        rng: Random number generator
        cfg: Configuration dict
        is_long: True for long positions, False for short
        is_sl: True for stop loss, False for take profit
        size: Number of samples (None for single value)
        
    Returns:
        Multipliers in correct range:
        - Long SL: 0.50-0.99 (stops below entry)
        - Long TP: 1.01-21.0 (profits above entry)
        - Short SL: 1.01-1.50 (stops above entry)
        - Short TP: 0.50-0.99 (profits below entry)
    """
    # Get distribution params
    mean_frac = cfg.get("tp_sl_mean", 0.05)  # Mean as fraction (e.g., 5%)
    sigma_frac = cfg.get("tp_sl_sigma", 0.03)  # Sigma as fraction
    
    # Determine bounds based on position type and SL/TP
    if is_long:
        if is_sl:
            lower = cfg.get("synth_long_sl_min", 0.50)
            upper = cfg.get("synth_long_sl_max", 0.99)
            # SL below 1.0: sample negative fractions, convert to multiplier
            center = 1.0 - mean_frac  # e.g., 0.95 for 5% stop
        else:  # TP
            lower = cfg.get("synth_long_tp_min", 1.01)
            upper = cfg.get("synth_long_tp_max", 21.0)
            # TP above 1.0: sample positive fractions, convert to multiplier
            center = 1.0 + mean_frac  # e.g., 1.05 for 5% profit
    else:  # Short
        if is_sl:
            lower = cfg.get("synth_short_sl_min", 1.01)
            upper = cfg.get("synth_short_sl_max", 1.50)
            # SL above 1.0: sample positive fractions, convert to multiplier
            center = 1.0 + mean_frac  # e.g., 1.05 for 5% stop
        else:  # TP
            lower = cfg.get("synth_short_tp_min", 0.50)
            upper = cfg.get("synth_short_tp_max", 0.99)
            # TP below 1.0: sample negative fractions, convert to multiplier
            center = 1.0 - mean_frac  # e.g., 0.95 for 5% profit
    
    # Use truncated normal in multiplier space directly
    a = (lower - center) / sigma_frac
    b = (upper - center) / sigma_frac
    
    if size is None:
        sample = stats.truncnorm.rvs(a, b, loc=center, scale=sigma_frac, random_state=rng)
    else:
        sample = stats.truncnorm.rvs(a, b, loc=center, scale=sigma_frac, size=size, random_state=rng)
    
    return sample


def _validate_sample(equity: float, balance: float, 
                    long_value: float, short_value: float,
                    long_sl: float, long_tp: float,
                    short_sl: float, short_tp: float,
                    cfg: dict) -> bool:
    """
    Validate a single sample meets all constraints.
    
    Returns:
        True if valid, False otherwise
    """
    try:
        # Positive constraints
        if equity <= 0:
            return False
        if balance < 0:
            return False
        if long_value < 0 or short_value < 0:
            return False
        
        # Leverage constraint
        gross_exposure = long_value + short_value
        max_leverage = cfg.get("max_leverage", 5.0)
        if gross_exposure > equity * max_leverage:
            return False
        
        # Balance consistency
        expected_balance = equity - gross_exposure
        if not np.isclose(balance, expected_balance, rtol=1e-6):
            return False
        
        # SL/TP bounds for long position
        if long_value > 0:
            if not (cfg.get("synth_long_sl_min", 0.50) <= long_sl <= cfg.get("synth_long_sl_max", 0.99)):
                return False
            if not (cfg.get("synth_long_tp_min", 1.01) <= long_tp <= cfg.get("synth_long_tp_max", 21.0)):
                return False
        
        # SL/TP bounds for short position
        if short_value > 0:
            if not (cfg.get("synth_short_sl_min", 1.01) <= short_sl <= cfg.get("synth_short_sl_max", 1.50)):
                return False
            if not (cfg.get("synth_short_tp_min", 0.50) <= short_tp <= cfg.get("synth_short_tp_max", 0.99)):
                return False
        
        return True
        
    except Exception:
        return False


def build_samples(df: pd.DataFrame, n: int, lookback: int, forward: int, rng: np.random.Generator, cfg: dict = None) -> pd.DataFrame:
    """
    Generate synthetic samples with new dual-position structure.
    
    Returns DataFrame with columns:
    - idx: index into df
    - equity, balance: account state
    - long_value, short_value: position sizes ($)
    - long_sl, long_tp, short_sl, short_tp: SL/TP multipliers
    - act_long_value, act_short_value: target position sizes
    - act_long_sl, act_long_tp, act_short_sl, act_short_tp: target SL/TP
    - open, high, low, close, volume: past OHLCV windows (arrays)
    - phase: curriculum phase (0,1,2) if available in df
    
    Includes:
    - Hold states: 10% with no positions (flat)
    - Hold actions: 20% with no change from current state
    - Validation: All samples meet leverage/SL/TP constraints
    """
    if cfg is None:
        cfg = {}
    
    max_idx = len(df) - lookback - forward
    
    # Hold percentages
    hold_state_pct = cfg.get("hold_state_pct", 0.10)
    hold_action_pct = cfg.get("hold_action_pct", 0.20)
    
    # ========== VECTORIZED: Generate all random data at once ==========
    # Sample indices
    indices = rng.integers(lookback, max_idx, size=n)
    
    # ========== POSITION STATES ==========
    # Equity (uniform for simplicity, could be log-normal too)
    equity = rng.uniform(cfg.get("synth_equity_min", 1e4), 
                        cfg.get("synth_equity_max", 1e5), size=n)
    
    # Position values from log-normal distribution
    long_value_raw = _sample_position_value(rng, cfg, size=n)
    short_value_raw = _sample_position_value(rng, cfg, size=n)
    
    # Clip to configured ranges
    long_value_raw = np.clip(long_value_raw, 
                            cfg.get("synth_long_value_min", 0),
                            cfg.get("synth_long_value_max", 50000))
    short_value_raw = np.clip(short_value_raw,
                             cfg.get("synth_short_value_min", 0),
                             cfg.get("synth_short_value_max", 50000))
    
    # Enforce two constraints: leverage and positive balance
    gross_exposure = long_value_raw + short_value_raw
    max_leverage = cfg.get("max_leverage", 5.0)
    max_exposure = equity * max_leverage
    
    # Can't exceed equity (balance must be >= 0)
    # Can't exceed max_leverage * equity
    max_allowed = np.minimum(equity, max_exposure)
    over_limit = gross_exposure > max_allowed
    
    scale_factor = np.ones(n)
    scale_factor[over_limit] = max_allowed[over_limit] / gross_exposure[over_limit]
    
    long_value = long_value_raw * scale_factor
    short_value = short_value_raw * scale_factor
    
    # Generate hold states (10% with no positions)
    n_hold_states = int(n * hold_state_pct)
    hold_state_indices = rng.choice(n, size=n_hold_states, replace=False)
    long_value[hold_state_indices] = 0.0
    short_value[hold_state_indices] = 0.0
    
    # Calculate balance (should always be >= 0 now)
    balance = equity - (long_value + short_value)
    # Handle floating point precision issues
    balance = np.maximum(balance, 0.0)
    
    # Sample SL/TP multipliers
    long_sl = _sample_sl_tp_multiplier(rng, cfg, is_long=True, is_sl=True, size=n)
    long_tp = _sample_sl_tp_multiplier(rng, cfg, is_long=True, is_sl=False, size=n)
    short_sl = _sample_sl_tp_multiplier(rng, cfg, is_long=False, is_sl=True, size=n)
    short_tp = _sample_sl_tp_multiplier(rng, cfg, is_long=False, is_sl=False, size=n)
    
    # Zero out SL/TP for positions that don't exist
    long_sl[long_value == 0] = 0.0
    long_tp[long_value == 0] = 0.0
    short_sl[short_value == 0] = 0.0
    short_tp[short_value == 0] = 0.0
    
    # ========== ACTIONS (TARGET STATE) ==========
    # Sample new target positions
    act_long_value_raw = _sample_position_value(rng, cfg, size=n)
    act_short_value_raw = _sample_position_value(rng, cfg, size=n)
    
    # Clip to ranges
    act_long_value_raw = np.clip(act_long_value_raw,
                                cfg.get("synth_long_value_min", 0),
                                cfg.get("synth_long_value_max", 50000))
    act_short_value_raw = np.clip(act_short_value_raw,
                                 cfg.get("synth_short_value_min", 0),
                                 cfg.get("synth_short_value_max", 50000))
    
    # Enforce leverage and balance constraints for actions
    act_gross_exposure = act_long_value_raw + act_short_value_raw
    act_max_allowed = np.minimum(equity, max_exposure)
    act_over_limit = act_gross_exposure > act_max_allowed
    
    act_scale_factor = np.ones(n)
    act_scale_factor[act_over_limit] = act_max_allowed[act_over_limit] / act_gross_exposure[act_over_limit]
    
    act_long_value = act_long_value_raw * act_scale_factor
    act_short_value = act_short_value_raw * act_scale_factor
    
    # Generate hold actions (20% no change - copy current state)
    n_hold_actions = int(n * hold_action_pct)
    hold_action_indices = rng.choice(n, size=n_hold_actions, replace=False)
    act_long_value[hold_action_indices] = long_value[hold_action_indices]
    act_short_value[hold_action_indices] = short_value[hold_action_indices]
    
    # Sample action SL/TP
    act_long_sl = _sample_sl_tp_multiplier(rng, cfg, is_long=True, is_sl=True, size=n)
    act_long_tp = _sample_sl_tp_multiplier(rng, cfg, is_long=True, is_sl=False, size=n)
    act_short_sl = _sample_sl_tp_multiplier(rng, cfg, is_long=False, is_sl=True, size=n)
    act_short_tp = _sample_sl_tp_multiplier(rng, cfg, is_long=False, is_sl=False, size=n)
    
    # For hold actions, also copy SL/TP
    act_long_sl[hold_action_indices] = long_sl[hold_action_indices]
    act_long_tp[hold_action_indices] = long_tp[hold_action_indices]
    act_short_sl[hold_action_indices] = short_sl[hold_action_indices]
    act_short_tp[hold_action_indices] = short_tp[hold_action_indices]
    
    # Zero out action SL/TP where no target position
    act_long_sl[act_long_value == 0] = 0.0
    act_long_tp[act_long_value == 0] = 0.0
    act_short_sl[act_short_value == 0] = 0.0
    act_short_tp[act_short_value == 0] = 0.0
    
    # ========== OPTIMIZED: Extract arrays once, pre-allocate ==========
    open_arr = df["open"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    close_arr = df["close"].values
    volume_arr = df["volume"].values
    
    # Pre-allocate arrays for OHLCV windows
    open_windows = np.zeros((n, lookback), dtype=np.float32)
    high_windows = np.zeros((n, lookback), dtype=np.float32)
    low_windows = np.zeros((n, lookback), dtype=np.float32)
    close_windows = np.zeros((n, lookback), dtype=np.float32)
    volume_windows = np.zeros((n, lookback), dtype=np.float32)
    
    # Extract windows - FULLY VECTORIZED using advanced indexing
    # Create index arrays for all windows at once
    window_indices = np.arange(lookback).reshape(1, -1)  # Shape: (1, lookback)
    start_indices = (indices - lookback).reshape(-1, 1)  # Shape: (n, 1)
    all_indices = start_indices + window_indices  # Broadcasting: Shape (n, lookback)
    
    # Extract all windows at once using fancy indexing (no loop!)
    open_windows = open_arr[all_indices].astype(np.float32)
    high_windows = high_arr[all_indices].astype(np.float32)
    low_windows = low_arr[all_indices].astype(np.float32)
    close_windows = close_arr[all_indices].astype(np.float32)
    volume_windows = volume_arr[all_indices].astype(np.float32)
    
    # Extract phase information if available
    phase_values = None
    if 'phase' in df.columns:
        phase_values = df['phase'].iloc[indices].values
    
    # Build DataFrame efficiently
    # Create dict with scalar columns first (fast)
    data = {
        "idx": indices,
        "equity": equity,
        "balance": balance,
        # Position state
        "long_value": long_value,
        "short_value": short_value,
        "long_sl": long_sl,
        "long_tp": long_tp,
        "short_sl": short_sl,
        "short_tp": short_tp,
        # Actions (target state)
        "act_long_value": act_long_value,
        "act_short_value": act_short_value,
        "act_long_sl": act_long_sl,
        "act_long_tp": act_long_tp,
        "act_short_sl": act_short_sl,
        "act_short_tp": act_short_tp,
    }
    
    # Add phase column if available
    if phase_values is not None:
        data["phase"] = phase_values
    
    # Add array columns using pd.Series (more efficient for object dtype)
    df_result = pd.DataFrame(data)
    df_result["open"] = pd.Series([row for row in open_windows], dtype=object)
    df_result["high"] = pd.Series([row for row in high_windows], dtype=object)
    df_result["low"] = pd.Series([row for row in low_windows], dtype=object)
    df_result["close"] = pd.Series([row for row in close_windows], dtype=object)
    df_result["volume"] = pd.Series([row for row in volume_windows], dtype=object)
    
    return df_result


def build_samples_parallel(df: pd.DataFrame, n: int, lookback: int, forward: int, 
                           rng: np.random.Generator, cfg: dict = None, n_jobs: int = -1) -> pd.DataFrame:
    """
    Build samples in parallel using multiple CPU cores.
    
    For large datasets (10M+ samples), this provides significant speedup.
    
    Args:
        df: OHLCV DataFrame
        n: Number of samples to generate
        lookback: Lookback window size
        forward: Forward window size
        rng: Random number generator
        cfg: Configuration dict
        n_jobs: Number of parallel jobs (-1 = all cores, -2 = all but one)
        
    Returns:
        DataFrame with samples
    """
    if cfg is None:
        cfg = {}
    
    # Determine number of jobs
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    elif n_jobs == -2:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    elif n_jobs <= 0:
        n_jobs = 1
    
    # For small datasets, parallel overhead not worth it
    if n < 10000 or n_jobs == 1:
        return build_samples(df, n, lookback, forward, rng, cfg)
    
    # Split into chunks
    chunk_size = n // n_jobs
    remainder = n % n_jobs
    chunk_sizes = [chunk_size + (1 if i < remainder else 0) for i in range(n_jobs)]
    
    # Generate separate seeds for each worker (for reproducibility)
    worker_seeds = rng.integers(0, 2**31 - 1, size=n_jobs)
    
    print(f"Building {n:,} samples using {n_jobs} parallel workers...")
    
    # Build chunks in parallel
    t0 = time.perf_counter()
    chunks = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(build_samples)(
            df, 
            chunk_n, 
            lookback, 
            forward, 
            np.random.default_rng(seed),
            cfg
        ) 
        for chunk_n, seed in zip(chunk_sizes, worker_seeds)
    )
    t1 = time.perf_counter()
    print(f"  Parallel execution: {t1-t0:.2f}s")
    
    # Concatenate results efficiently (copy=False to avoid unnecessary copying)
    print("  Combining results...")
    t2 = time.perf_counter()
    result = pd.concat(chunks, ignore_index=True, copy=False)
    t3 = time.perf_counter()
    print(f"  Concat time: {t3-t2:.2f}s")
    return result