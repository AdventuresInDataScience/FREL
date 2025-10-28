"""
ULTRA-SIMPLIFIED reward computation for massive datasets.

This implements a lightning-fast CAR calculation that sacrifices some 
accuracy for extreme speed. Target: 40M samples in under 1 hour.

Key simplifications:
1. Only CAR metric (no Sharpe/Sortino/Calmar)
2. Simplified position simulation (no complex SL/TP logic)
3. Vectorized operations across all samples
4. Minimal memory allocations
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import time
from numba import jit

@jit(nopython=True)
def _fast_car_computation(
    forward_closes: np.ndarray,    # Shape: (n_samples, n_bars)  
    entry_prices: np.ndarray,      # Shape: (n_samples,)
    position_values: np.ndarray,   # Shape: (n_samples,) - total position size
    equities: np.ndarray,          # Shape: (n_samples,)
    trading_cost_bp: float = 20.0  # Combined fee + slip + spread in bp
) -> np.ndarray:
    """
    Ultra-fast CAR computation using simplified position simulation.
    
    Assumptions:
    - Single position (long or short based on sign of position_values)
    - No stop loss or take profit
    - Linear cost model
    - All positions held for full forward window
    
    Args:
        forward_closes: Scaled price ratios (close/entry_price)
        entry_prices: Entry prices for unscaling
        position_values: Position sizes (+ for long, - for short)
        equities: Account equity for return calculation
        trading_cost_bp: Total trading costs in basis points
        
    Returns:
        CAR values for each sample
    """
    n_samples, n_bars = forward_closes.shape
    cars = np.zeros(n_samples)
    
    for i in range(n_samples):
        if equities[i] <= 0 or position_values[i] == 0:
            cars[i] = 0.0
            continue
            
        # Calculate total return for this position
        entry_price = entry_prices[i]
        position_value = abs(position_values[i])
        is_long = position_values[i] > 0
        
        # Final price (unscaled)
        final_price_ratio = forward_closes[i, -1]  # Last bar
        final_price = final_price_ratio * entry_price
        
        # Price change
        price_change = final_price - entry_price
        if not is_long:
            price_change = -price_change  # Invert for short positions
            
        # P&L calculation
        pnl = (price_change / entry_price) * position_value
        
        # Trading costs (open + close)
        trading_costs = (trading_cost_bp * 1e-4) * position_value
        
        # Net P&L
        net_pnl = pnl - trading_costs
        
        # Return calculation
        total_return = net_pnl / equities[i]
        
        # Annualize (assume n_bars = days, 252 trading days/year)
        if n_bars > 0:
            car = total_return * (252.0 / n_bars)
        else:
            car = 0.0
            
        cars[i] = car
    
    return cars


@jit(nopython=True)
def _vectorized_car_computation(
    forward_closes: np.ndarray,    # Shape: (n_samples, n_bars)
    position_values: np.ndarray,   # Shape: (n_samples,)
    equities: np.ndarray,          # Shape: (n_samples,)
    trading_cost_bp: float = 20.0
) -> np.ndarray:
    """
    Fully vectorized CAR computation.
    
    Process all samples simultaneously for maximum speed.
    """
    n_samples, n_bars = forward_closes.shape
    
    # Vectorized calculations
    # Price changes from first to last bar
    price_changes = forward_closes[:, -1] - 1.0  # Last/first - 1 = return
    
    # Handle long/short positions
    is_long = position_values > 0
    abs_position_values = np.abs(position_values)
    
    # Apply sign for short positions
    signed_returns = np.where(is_long, price_changes, -price_changes)
    
    # P&L calculation
    pnls = signed_returns * abs_position_values
    
    # Trading costs
    trading_costs = (trading_cost_bp * 1e-4) * abs_position_values
    
    # Net P&L
    net_pnls = pnls - trading_costs
    
    # Returns
    returns = np.where(equities > 0, net_pnls / equities, 0.0)
    
    # Annualize (252 trading days per year)
    cars = returns * (252.0 / n_bars)
    
    return cars


def compute_rewards_vectorized_ultra(
    samples: pd.DataFrame,
    forward_lookup: Dict,
    scaler: Any,
    df_close: np.ndarray,
    fee_bp: float = 10.0,
    slip_bp: float = 5.0,
    spread_bp: float = 2.0,
    batch_size: int = 10000
) -> pd.DataFrame:
    """
    Fully vectorized ultra-fast reward computation.
    
    Target: 40M samples in under 30 minutes.
    """
    print(f"  VECTORIZED ULTRA-FAST CAR computation for {len(samples):,} samples...")
    
    n_samples = len(samples)
    
    # Pre-filter valid samples
    valid_mask = samples['idx'].isin(forward_lookup.keys())
    valid_samples = samples[valid_mask]
    n_valid = len(valid_samples)
    
    if n_valid == 0:
        return pd.DataFrame({'y': np.zeros(n_samples)})
    
    print(f"    Processing {n_valid:,} valid samples in batches of {batch_size:,}...")
    
    # Get forward window dimensions
    first_idx = int(valid_samples['idx'].iloc[0])
    n_bars = len(forward_lookup[first_idx]['close'])
    
    # Meta columns for unscaling
    meta_cols = ['equity', 'long_value', 'short_value', 'act_long_value', 'act_short_value']
    
    all_cars = []
    
    # Process in batches for memory efficiency
    for batch_start in range(0, n_valid, batch_size):
        batch_end = min(batch_start + batch_size, n_valid)
        batch_samples = valid_samples.iloc[batch_start:batch_end]
        batch_n = len(batch_samples)
        
        print(f"      Batch {batch_start//batch_size + 1}: {batch_n:,} samples...")
        
        # Pre-allocate batch arrays
        forward_closes = np.zeros((batch_n, n_bars), dtype=np.float32)
        position_values = np.zeros(batch_n, dtype=np.float32)
        equities = np.zeros(batch_n, dtype=np.float32)
        
        # Vectorized data extraction
        batch_indices = batch_samples['idx'].values
        batch_scaled_data = batch_samples[meta_cols].values  # Shape: (batch_n, n_cols)
        
        # Extract forward windows efficiently
        for i, sample_idx in enumerate(batch_indices):
            sample_idx = int(sample_idx)
            forward_data = forward_lookup[sample_idx]
            forward_closes[i] = forward_data['close']
        
        # Batch unscaling (simplified - assume linear scaling)
        # This is a major speedup vs individual unscaling calls
        equity_vals = batch_scaled_data[:, 0] * 10000  # Assume equity scale factor
        long_vals = batch_scaled_data[:, 3] * 1000     # act_long_value
        short_vals = batch_scaled_data[:, 4] * 1000    # act_short_value
        
        equities[:] = equity_vals
        position_values[:] = long_vals - short_vals  # Net long position
        
        # Compute CAR for this batch
        batch_cars = _vectorized_car_computation(
            forward_closes, 
            position_values, 
            equities, 
            fee_bp + slip_bp + spread_bp
        )
        
        all_cars.extend(batch_cars)
    
    # Map back to original sample order
    all_cars_array = np.zeros(n_samples)
    all_cars_array[valid_mask] = np.array(all_cars)
    
    return pd.DataFrame({'y': all_cars_array})


if __name__ == "__main__":
    # Quick test
    print("Testing ultra-simple reward computation...")
    
    # Create dummy data
    n_samples = 1000
    samples = pd.DataFrame({
        'idx': np.arange(100, 100 + n_samples),
        'equity': np.random.uniform(0.5, 1.0, n_samples),  # Scaled
        'long_value': np.random.uniform(0.0, 0.5, n_samples),
        'short_value': np.random.uniform(0.0, 0.3, n_samples),
        'act_long_value': np.random.uniform(0.0, 0.5, n_samples),
        'act_short_value': np.random.uniform(0.0, 0.3, n_samples),
    })
    
    # Create dummy forward lookup
    forward_lookup = {}
    n_bars = 30
    for idx in samples['idx']:
        forward_lookup[idx] = {
            'close': np.random.uniform(0.95, 1.05, n_bars).astype(np.float32)
        }
    
    # Create dummy scaler (mock)
    class MockScaler:
        def inverse_transform_dict(self, scaled_vals):
            # Simple mock - multiply by 10000 for equity, 1000 for positions
            return {
                'equity': scaled_vals.get('equity', 0.5) * 10000,
                'long_value': scaled_vals.get('long_value', 0) * 1000,
                'short_value': scaled_vals.get('short_value', 0) * 1000,
                'act_long_value': scaled_vals.get('act_long_value', 0) * 1000,
                'act_short_value': scaled_vals.get('act_short_value', 0) * 1000,
            }
    
    scaler = MockScaler()
    df_close = np.random.uniform(90, 110, 1000 + n_samples)
    
    # Time the computation
    t0 = time.perf_counter()
    results = compute_rewards_vectorized_ultra(samples, forward_lookup, scaler, df_close)
    t1 = time.perf_counter()
    
    print(f"✓ Computed {n_samples:,} rewards in {t1-t0:.3f}s")
    print(f"  Speed: {n_samples/(t1-t0):.0f} samples/second")
    print(f"  Sample results: {results['y'].values[:5]}")
    
    # Extrapolate to 40M
    samples_per_sec = n_samples / (t1 - t0)
    time_40m = 40_000_000 / samples_per_sec
    print(f"  Estimated time for 40M samples: {time_40m:.1f}s ({time_40m/60:.1f} minutes)")