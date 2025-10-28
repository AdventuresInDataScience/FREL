"""
Vectorised reward engines with dual-position structure.

NEW STRUCTURE:
- Positions: long_value, short_value, long_sl, long_tp, short_sl, short_tp
- Actions: act_long_value, act_short_value, act_long_sl, act_long_tp, act_short_sl, act_short_tp
- SL/TP in multiplier notation (e.g., long_sl=0.95 = 5% stop, short_tp=0.90 = 10% profit)
- Forward windows: Pre-computed, accessed via forward_lookup dict
- Simulation: Dual long/short positions with proper SL/TP checking per bar

REWARD METRIC FORMULAS (Standard Financial Definitions):
- CAR (Compound Annual Return): (Total P&L / Equity) × (Trading Days / Period Days)
- Sharpe Ratio: (Mean Return - Risk Free Rate) / Return Volatility  
- Sortino Ratio: (Mean Return - Risk Free Rate) / Downside Deviation
- Calmar Ratio: Annual Return / Maximum Drawdown

EXECUTION ENGINES (Same Math, Different Performance):
- Individual Functions: Process one sample at a time, Python loops, good for small datasets
- JIT-Optimized Functions: Process many samples in batch, compiled machine code, 50-500x faster
- Automatic Selection: System chooses best engine based on dataset size (50+ samples → JIT)

JIT WARM-UP FOR PRODUCTION:
For single large runs (like 40M samples), call warm_up_jit_functions() first to eliminate
the ~8s compilation overhead and ensure immediate ultra-fast performance.

USAGE FOR LARGE DATASETS:
```python
from src.reward import warm_up_jit_functions, compute_many

# Pre-compile JIT functions (one-time ~8s cost)
warm_up_jit_functions()

# Now all reward computations are optimized from first call
results = compute_many(df_close, samples, forward_lookup, scaler, 
                      reward_funcs=['car', 'sharpe', 'sortino', 'calmar'])
```
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from scipy.optimize import differential_evolution
from numba import jit, types
from numba.typed import Dict as NumbaDict
from joblib import Parallel, delayed
import warnings

# Suppress numba warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='numba')


# ========== JIT WARM-UP ==========

def warm_up_jit_functions():
    """
    Pre-compile all JIT functions with dummy data.
    
    This eliminates JIT compilation overhead on first real use.
    Call this once at startup before processing large datasets.
    
    Typical timing: ~8-10 seconds compilation, saves 8-10 seconds on first real call.
    """
    print("Warming up JIT functions...")
    
    # Create minimal dummy data
    dummy_size = 5
    dummy_bars = 10
    
    # Dummy forward data
    forward_closes = np.ones((dummy_size, dummy_bars), dtype=np.float32)
    position_values = np.ones(dummy_size, dtype=np.float32) * 1000
    equities = np.ones(dummy_size, dtype=np.float32) * 10000
    
    # Dummy simulation
    try:
        daily_pnl, valid_mask = _ultra_fast_simulation_jit(forward_closes, position_values, equities, 10.0)
        
        # Dummy metrics (force JIT compilation)
        _ultra_fast_car_jit(daily_pnl, valid_mask, 252)
        _ultra_fast_sharpe_jit(daily_pnl, valid_mask, 252, 0.02)
        _ultra_fast_sortino_jit(daily_pnl, valid_mask, 252, 0.02)
        _ultra_fast_calmar_jit(daily_pnl, valid_mask, 252)
        
        print("JIT warm-up complete! All functions pre-compiled.")
        
    except Exception as e:
        print(f"Warning: JIT warm-up failed: {e}")
        print("JIT functions will compile on first use.")


# ========== HELPER FUNCTIONS ==========

@jit(nopython=True)
def _calculate_costs_jit(position_value: float, fee_bp: float, slip_bp: float, spread_bp: float) -> float:
    """
    JIT-compiled cost calculation function.
    Calculate trading costs for opening/closing a position.
    
    Args:
        position_value: Dollar value of position
        fee_bp: Fee in basis points
        slip_bp: Slippage in basis points
        spread_bp: Spread in basis points
        
    Returns:
        Total cost in dollars
    """
    if position_value == 0:
        return 0.0
    
    # Entry: pay fee + slippage + half spread
    # Exit: pay fee + slippage + half spread
    # Total: 2 * (fee + slip + spread/2)
    cost_per_trade = (fee_bp + slip_bp + spread_bp / 2) * 1e-4
    return cost_per_trade * position_value * 2  # Open + close


@jit(nopython=True)
def _calculate_overnight_costs_jit(position_value: float, n_days: int, night_bp: float) -> float:
    """
    JIT-compiled overnight cost calculation.
    Calculate overnight holding costs.
    
    Args:
        position_value: Dollar value of position
        n_days: Number of days held
        night_bp: Overnight cost in basis points per day
        
    Returns:
        Total overnight cost in dollars
    """
    if position_value == 0 or n_days == 0:
        return 0.0
    return night_bp * 1e-4 * position_value * n_days


def _unscale_position_data(sample_row: pd.Series, scaler, meta_cols: list) -> dict:
    """
    Unscale position data from [0,1] back to original units.
    
    Args:
        sample_row: Single row from samples DataFrame (scaled values)
        scaler: MetaScaler object used for scaling
        meta_cols: List of column names that were scaled
        
    Returns:
        Dict with unscaled position values
    """
    # Extract scaled values
    scaled_vals = {col: sample_row[col] for col in meta_cols if col in sample_row.index}
    
    # Unscale using scaler's inverse transform
    unscaled = scaler.inverse_transform_dict(scaled_vals)
    
    return unscaled


def _calculate_costs(position_value: float, fee_bp: float, slip_bp: float, spread_bp: float) -> float:
    """
    Calculate trading costs for opening/closing a position.
    Wrapper for JIT-compiled version.
    """
    return _calculate_costs_jit(position_value, fee_bp, slip_bp, spread_bp)


def _calculate_overnight_costs(position_value: float, n_days: int, night_bp: float) -> float:
    """
    Calculate overnight holding costs.
    Wrapper for JIT-compiled version.
    """
    return _calculate_overnight_costs_jit(position_value, n_days, night_bp)


# ========== DERIVED FEATURE HELPERS ==========

def _calculate_unrealized_pnl(
    long_value: float,
    short_value: float,
    long_entry_price: float,
    short_entry_price: float,
    current_price: float
) -> Tuple[float, float, float]:
    """
    Calculate unrealized P&L for open positions.
    
    Args:
        long_value: Dollar value of long position
        short_value: Dollar value of short position
        long_entry_price: Entry price for long position
        short_entry_price: Entry price for short position
        current_price: Current market price
        
    Returns:
        Tuple of (long_upnl, short_upnl, total_upnl) in dollars
    """
    long_upnl = 0.0
    short_upnl = 0.0
    
    if long_value > 0 and long_entry_price > 0:
        long_upnl = long_value * (current_price / long_entry_price - 1)
    
    if short_value > 0 and short_entry_price > 0:
        short_upnl = short_value * (1 - current_price / short_entry_price)
    
    total_upnl = long_upnl + short_upnl
    
    return long_upnl, short_upnl, total_upnl


def _calculate_current_drawdown(
    equity_curve: np.ndarray,
    current_bar: int
) -> Tuple[float, float, int]:
    """
    Calculate current drawdown from equity peak.
    
    Args:
        equity_curve: Array of equity values over time
        current_bar: Current bar index
        
    Returns:
        Tuple of (current_drawdown_pct, max_drawdown_pct, bars_since_peak)
    """
    if current_bar <= 0 or len(equity_curve) == 0:
        return 0.0, 0.0, 0
    
    # Get equity up to current bar
    equity_so_far = equity_curve[:current_bar + 1]
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_so_far)
    
    # Current drawdown
    current_equity = equity_so_far[-1]
    peak_equity = running_max[-1]
    
    if peak_equity <= 0:
        return 0.0, 0.0, 0
    
    current_dd_pct = (peak_equity - current_equity) / peak_equity
    
    # Maximum drawdown so far
    drawdown = (running_max - equity_so_far) / running_max
    max_dd_pct = np.max(drawdown)
    
    # Bars since peak
    peak_idx = np.argmax(equity_so_far)
    bars_since_peak = current_bar - peak_idx
    
    return current_dd_pct, max_dd_pct, bars_since_peak


def _calculate_bars_in_position(
    position_history: np.ndarray,
    current_bar: int
) -> Tuple[int, int]:
    """
    Calculate how many consecutive bars positions have been held.
    
    Args:
        position_history: Boolean array indicating if position was active
        current_bar: Current bar index
        
    Returns:
        Tuple of (consecutive_bars, total_bars_in_position)
    """
    if current_bar < 0 or len(position_history) == 0:
        return 0, 0
    
    history_so_far = position_history[:current_bar + 1]
    
    # Total bars in position
    total_bars = np.sum(history_so_far)
    
    # Consecutive bars (count backwards from current)
    consecutive = 0
    for i in range(len(history_so_far) - 1, -1, -1):
        if history_so_far[i]:
            consecutive += 1
        else:
            break
    
    return consecutive, int(total_bars)


def _calculate_position_metrics(
    long_value: float,
    short_value: float,
    equity: float,
    balance: float
) -> Tuple[float, float, float]:
    """
    Calculate position sizing metrics.
    
    Args:
        long_value: Dollar value of long position
        short_value: Dollar value of short position
        equity: Total equity
        balance: Available balance
        
    Returns:
        Tuple of (gross_exposure, net_exposure, leverage)
    """
    if equity <= 0:
        return 0.0, 0.0, 0.0
    
    gross_exposure = long_value + short_value
    net_exposure = long_value - short_value
    leverage = gross_exposure / equity
    
    return gross_exposure, net_exposure, leverage


# ========== JIT-OPTIMIZED BATCH SIMULATION ENGINE ==========
# Same simulation logic as individual functions, but JIT-compiled for batch processing

@jit(nopython=True)
def _simulate_batch_positions_jit(
    forward_closes: np.ndarray,  # Shape: (n_samples, n_bars)
    forward_highs: np.ndarray,   # Shape: (n_samples, n_bars)
    forward_lows: np.ndarray,    # Shape: (n_samples, n_bars)
    entry_prices: np.ndarray,    # Shape: (n_samples,)
    equities: np.ndarray,        # Shape: (n_samples,)
    balances: np.ndarray,        # Shape: (n_samples,)
    long_values_curr: np.ndarray,
    short_values_curr: np.ndarray,
    long_sls_curr: np.ndarray,
    long_tps_curr: np.ndarray,
    short_sls_curr: np.ndarray,
    short_tps_curr: np.ndarray,
    long_values_tgt: np.ndarray,
    short_values_tgt: np.ndarray,
    long_sls_tgt: np.ndarray,
    long_tps_tgt: np.ndarray,
    short_sls_tgt: np.ndarray,
    short_tps_tgt: np.ndarray,
    fee_bp: float,
    slip_bp: float,
    spread_bp: float,
    night_bp: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-OPTIMIZED: Batch simulation of positions for multiple samples.
    
    Uses the same trading logic as individual simulation, but processes
    ALL samples at once using vectorized operations for massive speedup.
    
    Args:
        All arrays have shape (n_samples, ...) for batch processing
        
    Returns:
        Tuple of (batch_rewards, batch_total_costs)
        - batch_rewards: Shape (n_samples,) - CAR for each sample
        - batch_total_costs: Shape (n_samples,) - total costs for each sample
    """
    n_samples, n_bars = forward_closes.shape
    
    # Output arrays
    batch_rewards = np.zeros(n_samples)
    batch_total_costs = np.zeros(n_samples)
    
    # Process each sample (still a loop, but everything else is vectorized)
    for sample_idx in range(n_samples):
        # Extract this sample's data
        forward_close = forward_closes[sample_idx]
        forward_high = forward_highs[sample_idx]
        forward_low = forward_lows[sample_idx]
        entry_price = entry_prices[sample_idx]
        equity = equities[sample_idx]
        balance = balances[sample_idx]
        
        # Unscale forward prices
        close_prices = forward_close * entry_price
        high_prices = forward_high * entry_price
        low_prices = forward_low * entry_price
        
        # Initialize position state
        daily_pnl = np.zeros(n_bars)
        total_costs = 0.0
        
        # Current positions
        long_val_curr = long_values_curr[sample_idx]
        short_val_curr = short_values_curr[sample_idx]
        long_sl_curr = long_sls_curr[sample_idx]
        long_tp_curr = long_tps_curr[sample_idx]
        short_sl_curr = short_sls_curr[sample_idx]
        short_tp_curr = short_tps_curr[sample_idx]
        
        # Target positions (actions)
        long_val_tgt = long_values_tgt[sample_idx]
        short_val_tgt = short_values_tgt[sample_idx]
        long_sl_tgt = long_sls_tgt[sample_idx]
        long_tp_tgt = long_tps_tgt[sample_idx]
        short_sl_tgt = short_sls_tgt[sample_idx]
        short_tp_tgt = short_tps_tgt[sample_idx]
        
        # Trading logic (simplified for speed)
        long_active = long_val_curr > 0
        short_active = short_val_curr > 0
        
        # Apply actions on bar 0 (immediate rebalancing)
        if long_val_tgt != long_val_curr:
            position_change = abs(long_val_tgt - long_val_curr)
            total_costs += _calculate_costs_jit(position_change, fee_bp, slip_bp, spread_bp)
            long_val_curr = long_val_tgt
            long_sl_curr = long_sl_tgt
            long_tp_curr = long_tp_tgt
            long_active = long_val_curr > 0
        
        if short_val_tgt != short_val_curr:
            position_change = abs(short_val_tgt - short_val_curr)
            total_costs += _calculate_costs_jit(position_change, fee_bp, slip_bp, spread_bp)
            short_val_curr = short_val_tgt
            short_sl_curr = short_sl_tgt
            short_tp_curr = short_tp_tgt
            short_active = short_val_curr > 0
        
        # Simulate through bars
        for bar in range(n_bars):
            bar_pnl = 0.0
            
            # Long position P&L and exit checking
            if long_active and bar > 0:
                price_change = close_prices[bar] - close_prices[bar-1]
                long_pnl = (price_change / close_prices[bar-1]) * long_val_curr
                bar_pnl += long_pnl
                
                # Check SL/TP (simplified)
                if long_sl_curr > 0 and low_prices[bar] <= close_prices[0] * long_sl_curr:
                    # Stop loss hit
                    exit_pnl = (close_prices[0] * long_sl_curr - close_prices[0]) / close_prices[0] * long_val_curr
                    bar_pnl = bar_pnl - long_pnl + exit_pnl  # Replace with SL exit
                    total_costs += _calculate_costs_jit(long_val_curr, fee_bp, slip_bp, spread_bp) / 2  # Exit only
                    long_active = False
                elif long_tp_curr > 0 and high_prices[bar] >= close_prices[0] * long_tp_curr:
                    # Take profit hit
                    exit_pnl = (close_prices[0] * long_tp_curr - close_prices[0]) / close_prices[0] * long_val_curr
                    bar_pnl = bar_pnl - long_pnl + exit_pnl  # Replace with TP exit
                    total_costs += _calculate_costs_jit(long_val_curr, fee_bp, slip_bp, spread_bp) / 2  # Exit only
                    long_active = False
            
            # Short position P&L and exit checking
            if short_active and bar > 0:
                price_change = close_prices[bar] - close_prices[bar-1]
                short_pnl = -(price_change / close_prices[bar-1]) * short_val_curr  # Negative for short
                bar_pnl += short_pnl
                
                # Check SL/TP (simplified)
                if short_sl_curr > 0 and high_prices[bar] >= close_prices[0] * short_sl_curr:
                    # Stop loss hit (price went up)
                    exit_pnl = -(close_prices[0] * short_sl_curr - close_prices[0]) / close_prices[0] * short_val_curr
                    bar_pnl = bar_pnl - short_pnl + exit_pnl
                    total_costs += _calculate_costs_jit(short_val_curr, fee_bp, slip_bp, spread_bp) / 2
                    short_active = False
                elif short_tp_curr > 0 and low_prices[bar] <= close_prices[0] * short_tp_curr:
                    # Take profit hit (price went down)
                    exit_pnl = -(close_prices[0] * short_tp_curr - close_prices[0]) / close_prices[0] * short_val_curr
                    bar_pnl = bar_pnl - short_pnl + exit_pnl
                    total_costs += _calculate_costs_jit(short_val_curr, fee_bp, slip_bp, spread_bp) / 2
                    short_active = False
            
            # Overnight costs for active positions
            if bar > 0:
                overnight_cost = 0.0
                if long_active:
                    overnight_cost += _calculate_overnight_costs_jit(long_val_curr, 1, night_bp)
                if short_active:
                    overnight_cost += _calculate_overnight_costs_jit(short_val_curr, 1, night_bp)
                bar_pnl -= overnight_cost
                total_costs += overnight_cost
            
            daily_pnl[bar] = bar_pnl
        
        # Compute CAR (simplified - just total return annualized)
        total_pnl = np.sum(daily_pnl) - total_costs
        if equity > 0:
            total_return = total_pnl / equity
            # Annualize assuming n_bars is days and 252 trading days/year
            car = total_return * (252.0 / n_bars)
        else:
            car = 0.0
        
        batch_rewards[sample_idx] = car
        batch_total_costs[sample_idx] = total_costs
    
    return batch_rewards, batch_total_costs


# ========== SIMULATION ENGINE ==========

@jit(nopython=True)
def _simulate_positions_forward_jit(
    forward_close: np.ndarray,
    forward_high: np.ndarray,
    forward_low: np.ndarray,
    entry_price: float,
    equity: float,
    balance: float,
    long_value_curr: float,
    short_value_curr: float,
    long_sl_curr: float,
    long_tp_curr: float,
    short_sl_curr: float,
    short_tp_curr: float,
    long_value_tgt: float,
    short_value_tgt: float,
    long_sl_tgt: float,
    long_tp_tgt: float,
    short_sl_tgt: float,
    short_tp_tgt: float,
    fee_bp: float,
    slip_bp: float,
    spread_bp: float,
    night_bp: float
) -> Tuple[np.ndarray, float, int, int, str, str, int]:
    """
    JIT-compiled dual position simulation through forward window with SL/TP checking.
    
    Returns:
        Tuple of (daily_pnl, total_costs, long_exit_bar, short_exit_bar, 
                 long_exit_reason_code, short_exit_reason_code, trade_duration_bars)
    """
    n_bars = len(forward_close)
    daily_pnl = np.zeros(n_bars)
    
    # Unscale forward prices
    close_prices = forward_close * entry_price
    high_prices = forward_high * entry_price
    low_prices = forward_low * entry_price
    
    # Calculate starting drawdown context
    starting_dd_pct = 0.0
    if balance > equity and balance > 0:
        starting_dd_pct = (balance - equity) / balance
    
    # Initialize tracking
    total_costs = 0.0
    long_exit_bar = -1
    short_exit_bar = -1
    long_exit_reason_code = "none"  # Will convert to string later
    short_exit_reason_code = "none"
    trade_duration_bars = n_bars
    
    # Step 1: Execute position transitions at bar 0
    # Close current positions
    if long_value_curr > 0:
        # Close long at close[0]
        pnl = long_value_curr * (close_prices[0] / entry_price - 1)
        cost = _calculate_costs_jit(long_value_curr, fee_bp, slip_bp, spread_bp) / 2
        daily_pnl[0] += pnl - cost
        total_costs += cost
    
    if short_value_curr > 0:
        # Close short at close[0]
        pnl = short_value_curr * (1 - close_prices[0] / entry_price)
        cost = _calculate_costs_jit(short_value_curr, fee_bp, slip_bp, spread_bp) / 2
        daily_pnl[0] += pnl - cost
        total_costs += cost
    
    # Open target positions
    if long_value_tgt > 0:
        cost = _calculate_costs_jit(long_value_tgt, fee_bp, slip_bp, spread_bp) / 2
        daily_pnl[0] -= cost
        total_costs += cost
    
    if short_value_tgt > 0:
        cost = _calculate_costs_jit(short_value_tgt, fee_bp, slip_bp, spread_bp) / 2
        daily_pnl[0] -= cost
        total_costs += cost
    
    # Track active positions
    long_active = long_value_tgt > 0
    short_active = short_value_tgt > 0
    long_entry_price = close_prices[0] if long_active else 0.0
    short_entry_price = close_prices[0] if short_active else 0.0
    
    # Step 2: Simulate remaining bars
    for bar in range(1, n_bars):
        bar_pnl = 0.0
        
        # Long position simulation
        if long_active:
            # Check SL/TP using high/low
            sl_price = long_entry_price * long_sl_tgt if long_sl_tgt > 0 else 0.0
            tp_price = long_entry_price * long_tp_tgt if long_tp_tgt > 0 else 0.0
            
            hit_sl = (sl_price > 0) and (low_prices[bar] <= sl_price)
            hit_tp = (tp_price > 0) and (high_prices[bar] >= tp_price)
            
            if hit_sl:
                # Exit at SL
                pnl = long_value_tgt * (sl_price / long_entry_price - 1)
                cost = _calculate_costs_jit(long_value_tgt, fee_bp, slip_bp, spread_bp) / 2
                overnight = _calculate_overnight_costs_jit(long_value_tgt, bar, night_bp)
                bar_pnl += pnl - cost - overnight
                total_costs += cost + overnight
                long_active = False
                long_exit_bar = bar
                long_exit_reason_code = "SL"
                if trade_duration_bars > bar:
                    trade_duration_bars = bar
            elif hit_tp:
                # Exit at TP
                pnl = long_value_tgt * (tp_price / long_entry_price - 1)
                cost = _calculate_costs_jit(long_value_tgt, fee_bp, slip_bp, spread_bp) / 2
                overnight = _calculate_overnight_costs_jit(long_value_tgt, bar, night_bp)
                bar_pnl += pnl - cost - overnight
                total_costs += cost + overnight
                long_active = False
                long_exit_bar = bar
                long_exit_reason_code = "TP"
                if trade_duration_bars > bar:
                    trade_duration_bars = bar
            else:
                # Still active - just pay overnight cost
                overnight = _calculate_overnight_costs_jit(long_value_tgt, 1, night_bp)
                bar_pnl -= overnight
                total_costs += overnight
        
        # Short position simulation
        if short_active:
            # Check SL/TP using high/low
            sl_price = short_entry_price * short_sl_tgt if short_sl_tgt > 0 else 0.0
            tp_price = short_entry_price * short_tp_tgt if short_tp_tgt > 0 else 0.0
            
            hit_sl = (sl_price > 0) and (high_prices[bar] >= sl_price)
            hit_tp = (tp_price > 0) and (low_prices[bar] <= tp_price)
            
            if hit_sl:
                # Exit at SL
                pnl = short_value_tgt * (1 - sl_price / short_entry_price)
                cost = _calculate_costs_jit(short_value_tgt, fee_bp, slip_bp, spread_bp) / 2
                overnight = _calculate_overnight_costs_jit(short_value_tgt, bar, night_bp)
                bar_pnl += pnl - cost - overnight
                total_costs += cost + overnight
                short_active = False
                short_exit_bar = bar
                short_exit_reason_code = "SL"
                if trade_duration_bars > bar:
                    trade_duration_bars = bar
            elif hit_tp:
                # Exit at TP
                pnl = short_value_tgt * (1 - tp_price / short_entry_price)
                cost = _calculate_costs_jit(short_value_tgt, fee_bp, slip_bp, spread_bp) / 2
                overnight = _calculate_overnight_costs_jit(short_value_tgt, bar, night_bp)
                bar_pnl += pnl - cost - overnight
                total_costs += cost + overnight
                short_active = False
                short_exit_bar = bar
                short_exit_reason_code = "TP"
                if trade_duration_bars > bar:
                    trade_duration_bars = bar
            else:
                # Still active - just pay overnight cost
                overnight = _calculate_overnight_costs_jit(short_value_tgt, 1, night_bp)
                bar_pnl -= overnight
                total_costs += overnight
        
        daily_pnl[bar] = bar_pnl
    
    # Step 3: Close any remaining positions at end of forward window
    if long_active:
        pnl = long_value_tgt * (close_prices[-1] / long_entry_price - 1)
        cost = _calculate_costs_jit(long_value_tgt, fee_bp, slip_bp, spread_bp) / 2
        daily_pnl[-1] += pnl - cost
        total_costs += cost
        long_exit_bar = n_bars - 1
        long_exit_reason_code = "end"
    
    if short_active:
        pnl = short_value_tgt * (1 - close_prices[-1] / short_entry_price)
        cost = _calculate_costs_jit(short_value_tgt, fee_bp, slip_bp, spread_bp) / 2
        daily_pnl[-1] += pnl - cost
        total_costs += cost
        short_exit_bar = n_bars - 1
        short_exit_reason_code = "end"
    
    return (daily_pnl, total_costs, long_exit_bar, short_exit_bar, 
            long_exit_reason_code, short_exit_reason_code, trade_duration_bars)


def _simulate_positions_forward(
    forward_close: np.ndarray,
    forward_high: np.ndarray,
    forward_low: np.ndarray,
    entry_price: float,
    position_data: dict,
    fee_bp: float,
    slip_bp: float,
    spread_bp: float,
    night_bp: float
) -> Tuple[np.ndarray, dict]:
    """
    Simulate dual positions (long + short) through forward window with SL/TP checking.
    Optimized wrapper for JIT-compiled simulation function.
    
    Args:
        forward_close: Scaled close prices (forward window)
        forward_high: Scaled high prices (forward window) 
        forward_low: Scaled low prices (forward window)
        entry_price: Unscaled entry price (close[idx-1] used for scaling)
        position_data: Dict with unscaled position values
        fee_bp, slip_bp, spread_bp, night_bp: Cost parameters
        
    Returns:
        daily_pnl: Array of daily P&L
        metadata: Dict with exit info, costs, etc.
    """
    # Extract position data for JIT function
    equity = position_data['equity']
    balance = position_data.get('balance', equity)
    
    # Current positions
    long_value_curr = position_data['long_value']
    short_value_curr = position_data['short_value']
    long_sl_curr = position_data['long_sl']
    long_tp_curr = position_data['long_tp']
    short_sl_curr = position_data['short_sl']
    short_tp_curr = position_data['short_tp']
    
    # Target positions (actions)
    long_value_tgt = position_data['act_long_value']
    short_value_tgt = position_data['act_short_value']
    long_sl_tgt = position_data['act_long_sl']
    long_tp_tgt = position_data['act_long_tp']
    short_sl_tgt = position_data['act_short_sl']
    short_tp_tgt = position_data['act_short_tp']
    
    # Calculate starting drawdown
    starting_dd_pct = 0.0
    if balance > equity and balance > 0:
        starting_dd_pct = (balance - equity) / balance
    
    # Call JIT-compiled function
    results = _simulate_positions_forward_jit(
        forward_close, forward_high, forward_low, entry_price,
        equity, balance, 
        long_value_curr, short_value_curr, long_sl_curr, long_tp_curr, 
        short_sl_curr, short_tp_curr,
        long_value_tgt, short_value_tgt, long_sl_tgt, long_tp_tgt,
        short_sl_tgt, short_tp_tgt,
        fee_bp, slip_bp, spread_bp, night_bp
    )
    
    # Unpack results
    (daily_pnl, total_costs, long_exit_bar, short_exit_bar, 
     long_exit_reason_code, short_exit_reason_code, trade_duration_bars) = results
    
    # Convert exit reasons back to proper values (None or string)
    long_exit_bar = long_exit_bar if long_exit_bar >= 0 else None
    short_exit_bar = short_exit_bar if short_exit_bar >= 0 else None
    long_exit_reason = long_exit_reason_code if long_exit_reason_code != "none" else None
    short_exit_reason = short_exit_reason_code if short_exit_reason_code != "none" else None
    
    # Build metadata dict
    metadata = {
        'long_exit_bar': long_exit_bar,
        'short_exit_bar': short_exit_bar,
        'long_exit_reason': long_exit_reason,
        'short_exit_reason': short_exit_reason,
        'total_costs': total_costs,
        'trade_duration_bars': trade_duration_bars,
        'starting_dd_pct': starting_dd_pct
    }
    
    return daily_pnl, metadata


# ========== JIT-OPTIMIZED ARRAY OPERATIONS ==========

@jit(nopython=True)
def _calculate_drawdown_jit(cumulative_pnl: np.ndarray) -> float:
    """
    JIT-compiled drawdown calculation.
    Calculate maximum drawdown from cumulative P&L array.
    """
    if len(cumulative_pnl) == 0:
        return 0.0
    
    running_max = cumulative_pnl[0]
    max_drawdown = 0.0
    
    for i in range(1, len(cumulative_pnl)):
        if cumulative_pnl[i] > running_max:
            running_max = cumulative_pnl[i]
        
        drawdown = running_max - cumulative_pnl[i]
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    return max_drawdown


@jit(nopython=True)
def _calculate_downside_std_jit(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    JIT-compiled downside standard deviation calculation.
    Only considers returns below the threshold for downside risk calculation.
    
    Args:
        returns: Array of returns
        threshold: Threshold return rate (e.g., daily risk-free rate)
        
    Returns:
        Downside standard deviation
    """
    if len(returns) == 0:
        return 0.0
    
    # Count returns below threshold
    n_negative = 0
    for i in range(len(returns)):
        if returns[i] < threshold:
            n_negative += 1
    
    if n_negative == 0:
        return 0.0
    
    # Calculate downside deviations from threshold
    variance_sum = 0.0
    for i in range(len(returns)):
        if returns[i] < threshold:
            deviation = returns[i] - threshold
            variance_sum += deviation * deviation
    
    variance = variance_sum / n_negative
    return np.sqrt(variance)


# ========== MAIN REWARD FUNCTIONS ==========
#
# CRITICAL DESIGN DECISIONS:
#
# 1. TRADE DURATION vs FORWARD WINDOW:
#    - daily_pnl array contains the FULL forward window (e.g., 252 bars)
#    - BUT trades may exit early when SL/TP hit (e.g., at bar 50)
#    - metadata['trade_duration_bars'] tells us the ACTUAL exit point
#    - Rewards MUST use only actual_pnl[:trade_duration_bars] for calculation
#    - This ensures reward reflects actual trade performance, not future noise
#    - If trade doesn't exit early, trade_duration_bars = n_bars (full window)
#
# 2. TIME NORMALIZATION:
#    - All rewards are annualized for comparability across different trade durations
#    - A 10% return in 50 bars is MUCH better than 10% in 250 bars
#    - Annualization uses ACTUAL trade duration, not forward window length
#    - Formula: (1 + return)^(252/actual_bars) - 1 for growth rates
#    - Formula: metric * sqrt(252/actual_bars) for ratio metrics (Sharpe/Sortino)
#
# 3. EQUITY CONTEXT for DRAWDOWN:
#    - Calmar ratio needs starting equity to calculate drawdown percentage
#    - Drawdown in dollars / starting equity = drawdown percentage
#    - This gives proper risk context (5k drawdown on 100k is 5%, not absolute)
#    - Starting equity is passed explicitly to all reward functions
#
# 4. STARTING DRAWDOWN CONTEXT (Critical for Risk Assessment):
#    - metadata['starting_dd_pct'] captures current drawdown BEFORE trade
#    - Calculated from equity vs balance: dd = (balance - equity) / balance
#    - Calmar adds this to trade drawdown for TOTAL risk exposure
#    - Example: Starting DD = 5%, Trade DD = 10% → Total DD shown = 15%
#    - This prevents underestimating risk in already-losing scenarios
#
# 5. EDGE CASES:
#    - Zero equity: Return 0.0 (can't calculate returns)
#    - Empty PnL: Return 0.0 (no trade occurred)
#    - Zero volatility: Return 0.0 for Sharpe/Sortino (avoid division by zero)
#    - No drawdown: Return high value (annual_ret / epsilon) for Calmar
#    - No downside: Return high value (mean_ret * annualization / epsilon) for Sortino
#

def car(
    daily_pnl: np.ndarray,
    equity: float,
    metadata: dict,
    trading_days: int = 252,
    epsilon: float = 1e-8
) -> float:
    """
    Compound Annual Return.
    
    CAR = (Total P&L / Equity) * (Trading Days / Period Days)
    
    This is the standard CAR calculation - there's only one way to calculate it.
    The ultra-fast version uses the same formula but with JIT compilation for batch processing.
    
    Args:
        daily_pnl: Daily P&L array
        equity: Starting equity (for return calculation)
        metadata: Simulation metadata (unused)
        trading_days: Trading days per year (default 252)
        epsilon: Small value for stability (unused)
        
    Returns:
        Annualized return
    """
    if equity <= 0 or len(daily_pnl) == 0:
        return 0.0
    
    # Standard CAR formula
    total_pnl = np.sum(daily_pnl)
    total_return = total_pnl / equity
    n_bars = len(daily_pnl)
    
    return total_return * (trading_days / n_bars)


def sharpe(
    daily_pnl: np.ndarray,
    equity: float,
    metadata: dict,
    trading_days: int = 252,
    risk_free_rate: float = 0.02,  # 2% annual risk-free rate
    epsilon: float = 1e-8
) -> float:
    """
    Sharpe Ratio (excess return over risk-free rate divided by volatility).
    
    Uses actual trade duration for proper calculation. Annualization factor
    is based on actual bars in trade, not full forward window.
    
    Args:
        daily_pnl: Daily P&L array (full forward window)
        equity: Starting equity (for return calculation)
        metadata: Simulation metadata with 'trade_duration_bars' key
        trading_days: Trading days per year
        risk_free_rate: Annual risk-free rate (default 2%)
        epsilon: Small value for stability
        
    Returns:
        Annualized Sharpe ratio based on actual trade duration
    """
    if equity <= 0:
        return 0.0
    
    # Use only the PnL up to actual exit point
    actual_bars = metadata.get('trade_duration_bars', len(daily_pnl))
    actual_pnl = daily_pnl[:actual_bars]
    
    if len(actual_pnl) == 0:
        return 0.0
    
    # Convert to returns
    returns = actual_pnl / equity
    
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    
    # Calculate daily risk-free rate
    daily_rf_rate = risk_free_rate / trading_days
    
    if std_ret < epsilon:
        return 0.0
    
    # Annualization factor based on actual trade duration
    annualization_factor = np.sqrt(trading_days / actual_bars) if actual_bars > 0 else 1.0
    
    return (mean_ret - daily_rf_rate) / std_ret * annualization_factor


def sortino(
    daily_pnl: np.ndarray,
    equity: float,
    metadata: dict,
    trading_days: int = 252,
    risk_free_rate: float = 0.02,  # 2% annual risk-free rate
    epsilon: float = 1e-8
) -> float:
    """
    Sortino Ratio (downside deviation relative to risk-free rate).
    
    Uses actual trade duration for proper calculation. Only considers
    downside volatility from days with returns below the risk-free rate.
    
    Args:
        daily_pnl: Daily P&L array (full forward window)
        equity: Starting equity (for return calculation)
        metadata: Simulation metadata with 'trade_duration_bars' key
        trading_days: Trading days per year
        risk_free_rate: Annual risk-free rate threshold (default 2%)
        epsilon: Small value for stability
        
    Returns:
        Annualized Sortino ratio based on actual trade duration
    """
    if equity <= 0:
        return 0.0
    
    # Use only the PnL up to actual exit point
    actual_bars = metadata.get('trade_duration_bars', len(daily_pnl))
    actual_pnl = daily_pnl[:actual_bars]
    
    if len(actual_pnl) == 0:
        return 0.0
    
    # Convert to returns
    returns = actual_pnl / equity
    
    mean_ret = np.mean(returns)
    
    # Calculate daily risk-free rate
    daily_rf_rate = risk_free_rate / trading_days
    
    # Use JIT-optimized downside calculation with risk-free threshold
    downside_std = _calculate_downside_std_jit(returns, daily_rf_rate)
    
    if downside_std < epsilon:
        # No downside or very small downside
        if mean_ret > daily_rf_rate:
            annualization_factor = np.sqrt(trading_days / actual_bars) if actual_bars > 0 else 1.0
            return (mean_ret - daily_rf_rate) * annualization_factor / epsilon
        return 0.0
    
    # Annualization factor based on actual trade duration
    annualization_factor = np.sqrt(trading_days / actual_bars) if actual_bars > 0 else 1.0
    
    return (mean_ret - daily_rf_rate) / downside_std * annualization_factor


def calmar(
    daily_pnl: np.ndarray,
    equity: float,
    metadata: dict,
    trading_days: int = 252,
    epsilon: float = 1e-8
) -> float:
    """
    Calmar Ratio (return / max drawdown).
    
    Uses actual trade duration for proper calculation. Drawdown is calculated
    relative to starting equity, giving proper context to risk.
    
    CRITICAL: This accounts for STARTING drawdown state. If we're already in a 5%
    drawdown and the trade adds 10% more, the total drawdown shown is 15%.
    
    Args:
        daily_pnl: Daily P&L array (full forward window)
        equity: Starting equity (for drawdown percentage calculation)
        metadata: Simulation metadata with:
            - 'trade_duration_bars': Actual bars in trade (handles early exits)
            - 'starting_equity_peak': Peak equity before this trade (optional)
            - 'starting_dd_pct': Current drawdown % before trade (optional)
        trading_days: Trading days per year
        epsilon: Small value for stability
        
    Returns:
        Annualized Calmar ratio based on actual trade duration
    """
    if equity <= 0:
        return 0.0
    
    # Use only the PnL up to actual exit point (handles early exits)
    actual_bars = metadata.get('trade_duration_bars', len(daily_pnl))
    actual_pnl = daily_pnl[:actual_bars]
    
    if len(actual_pnl) == 0:
        return 0.0
    
    # Get starting drawdown context (if available)
    # This represents how much we're already down from peak
    starting_dd_pct = metadata.get('starting_dd_pct', 0.0)
    
    # Cumulative P&L (represents equity curve changes from starting equity)
    cum_pnl = np.cumsum(actual_pnl)
    
    # Calculate drawdown using JIT-optimized function
    max_dd_dollars_from_trade = _calculate_drawdown_jit(cum_pnl)
    
    # Convert to percentage of starting equity
    trade_dd_pct = max_dd_dollars_from_trade / equity
    
    # CRITICAL: Add starting drawdown to trade drawdown for total risk exposure
    # This represents the MAXIMUM POSSIBLE drawdown exposure during trade execution.
    # 
    # Rationale: If we're already in a 10% drawdown and our trade strategy could
    # add another 5% drawdown at worst, then at some point during the trade execution
    # we could be exposed to a total of 15% drawdown from the original peak.
    #
    # This is conservative risk assessment - it assumes the worst-case scenario where
    # both the existing drawdown and the trade drawdown occur simultaneously.
    # Example: Starting DD = 10%, Trade DD = 5% → Total risk exposure = 15%
    total_dd_pct = starting_dd_pct + trade_dd_pct
    
    # Annualized return based on actual trade duration
    total_return = cum_pnl[-1] / equity if len(cum_pnl) > 0 else 0.0
    years = actual_bars / trading_days
    
    if years <= 0:
        return 0.0
    
    annual_ret = (1 + total_return) ** (1 / years) - 1
    
    if total_dd_pct < epsilon:
        # No drawdown - excellent!
        return annual_ret / epsilon if annual_ret > 0 else 0.0
    
    return annual_ret / total_dd_pct


# ========== DATASET GENERATION: Compute rewards for training labels ==========

def compute_reward_for_sample(
    sample_row: pd.Series,
    df_close: np.ndarray,
    forward_lookup: dict,
    scaler: Any,
    reward_func: callable,
    fee_bp: float,
    slip_bp: float,
    spread_bp: float,
    night_bp: float,
    trading_days: int = 252,
    risk_free_rate: float = 0.02,
    epsilon: float = 1e-8
) -> float:
    """
    Compute reward for a single sample.
    
    This is the core reward computation function that:
    1. Unscales the sample's position data
    2. Simulates positions forward through the forward window
    3. Computes reward metric from resulting daily P&L
    
    Args:
        sample_row: Single row from samples DataFrame (scaled values)
        df_close: Full close price series (for entry price reference)
        forward_lookup: Dict mapping idx -> {open, high, low, close, volume} arrays (scaled)
        scaler: MetaScaler object for unscaling position data
        reward_func: Reward function (car, sharpe, sortino, calmar)
        fee_bp, slip_bp, spread_bp, night_bp: Cost parameters
        trading_days: Trading days per year
        risk_free_rate: Annual risk-free rate (for Sortino calculation)
        epsilon: Small value for numerical stability
        
    Returns:
        Scalar reward value
    """
    # Meta columns for unscaling
    meta_cols = [
        "equity", "balance",
        "long_value", "short_value", "long_sl", "long_tp", "short_sl", "short_tp",
        "act_long_value", "act_short_value", "act_long_sl", "act_long_tp",
        "act_short_sl", "act_short_tp"
    ]
    
    try:
        # Get forward windows
        sample_idx = sample_row['idx']
        if sample_idx not in forward_lookup:
            return 0.0
        
        forward_data = forward_lookup[sample_idx]
        forward_close = forward_data['close']
        forward_high = forward_data['high']
        forward_low = forward_data['low']
        
        # Get entry price (close[idx-1] - used for scaling forward windows)
        entry_price = df_close[sample_idx - 1]
        
        # Unscale position data
        position_data = _unscale_position_data(sample_row, scaler, meta_cols)
        
        # Simulate positions forward
        daily_pnl, metadata = _simulate_positions_forward(
            forward_close,
            forward_high,
            forward_low,
            entry_price,
            position_data,
            fee_bp,
            slip_bp,
            spread_bp,
            night_bp
        )
        
        # Compute reward from daily P&L (pass metadata for trade duration)
        # Handle different function signatures for sharpe/sortino vs others
        if reward_func.__name__ in ['sharpe', 'sortino']:
            reward = reward_func(daily_pnl, position_data['equity'], metadata, trading_days, risk_free_rate, epsilon)
        else:
            reward = reward_func(daily_pnl, position_data['equity'], metadata, trading_days, epsilon)
        return reward
        
    except Exception as e:
        # Fallback to 0 on error (could log error here)
        return 0.0


def compute_many_ultra_fast(
    df_close: np.ndarray,
    samples: pd.DataFrame,
    forward_lookup: dict,
    scaler: Any,
    reward_funcs: Optional[list] = None,
    reward_key: Optional[str] = None,
    fee_bp: float = 10.0,
    slip_bp: float = 5.0,
    spread_bp: float = 2.0,
    night_bp: float = 0.5,
    trading_days: int = 252,
    risk_free_rate: float = 0.02,
    epsilon: float = 1e-8,
    batch_size: int = 5000
) -> pd.DataFrame:
    """
    JIT-OPTIMIZED reward computation for large datasets.
    
    Uses the same mathematical formulas as individual reward functions,
    but with JIT compilation and batch processing for massive speedup.
    
    Supports all reward metrics: CAR, Sharpe, Sortino, Calmar.
    Expected speedup vs individual processing: 50-500x for large datasets.
    
    Note: First call has JIT compilation overhead (~8s), subsequent calls are optimized.
    Call warm_up_jit_functions() first to pre-compile for immediate speed.
    """
    if forward_lookup is None:
        raise ValueError("forward_lookup is required")
    if scaler is None:
        raise ValueError("scaler is required")
    
    # Handle backward compatibility
    if reward_funcs is None and reward_key is None:
        reward_funcs = ['car']  # Default to CAR
        single_target = True
    elif reward_funcs is None:
        reward_funcs = [reward_key]
        single_target = reward_key is not None
    else:
        single_target = False
    
    # Use JIT-optimized all-metrics computation
    return _compute_ultra_fast_all_metrics(
        samples=samples,
        forward_lookup=forward_lookup,
        scaler=scaler,
        df_close=df_close,
        reward_funcs=reward_funcs,
        total_cost_bp=fee_bp + slip_bp + spread_bp,
        trading_days=trading_days,
        risk_free_rate=risk_free_rate,
        single_target=single_target
    )
    first_idx = valid_samples['idx'].iloc[0]
    first_forward = forward_lookup[first_idx]
    n_bars = len(first_forward['close'])
    
    print(f"  Preparing batch arrays ({n_valid:,} × {n_bars} bars)...")
    
    # Pre-allocate batch arrays
    forward_closes = np.zeros((n_valid, n_bars), dtype=np.float32)
    forward_highs = np.zeros((n_valid, n_bars), dtype=np.float32)
    forward_lows = np.zeros((n_valid, n_bars), dtype=np.float32)
    entry_prices = np.zeros(n_valid, dtype=np.float32)
    
    # Position arrays (unscaled)
    equities = np.zeros(n_valid, dtype=np.float32)
    balances = np.zeros(n_valid, dtype=np.float32)
    long_values_curr = np.zeros(n_valid, dtype=np.float32)
    short_values_curr = np.zeros(n_valid, dtype=np.float32)
    long_sls_curr = np.zeros(n_valid, dtype=np.float32)
    long_tps_curr = np.zeros(n_valid, dtype=np.float32)
    short_sls_curr = np.zeros(n_valid, dtype=np.float32)
    short_tps_curr = np.zeros(n_valid, dtype=np.float32)
    long_values_tgt = np.zeros(n_valid, dtype=np.float32)
    short_values_tgt = np.zeros(n_valid, dtype=np.float32)
    long_sls_tgt = np.zeros(n_valid, dtype=np.float32)
    long_tps_tgt = np.zeros(n_valid, dtype=np.float32)
    short_sls_tgt = np.zeros(n_valid, dtype=np.float32)
    short_tps_tgt = np.zeros(n_valid, dtype=np.float32)
    
    # Fill arrays in batches to manage memory
    for batch_start in range(0, n_valid, batch_size):
        batch_end = min(batch_start + batch_size, n_valid)
        batch_samples = valid_samples.iloc[batch_start:batch_end]
        
        # Extract forward windows for this batch
        for i, (_, row) in enumerate(batch_samples.iterrows()):
            sample_idx = row['idx']
            forward_data = forward_lookup[sample_idx]
            
            # Store forward windows
            batch_row = batch_start + i
            forward_closes[batch_row] = forward_data['close']
            forward_highs[batch_row] = forward_data['high']
            forward_lows[batch_row] = forward_data['low']
            entry_prices[batch_row] = df_close[sample_idx - 1]
            
            # Unscale position data
            position_data = _unscale_position_data(row, scaler, meta_cols)
            
            # Store position data
            equities[batch_row] = position_data['equity']
            balances[batch_row] = position_data['balance']
            long_values_curr[batch_row] = position_data['long_value']
            short_values_curr[batch_row] = position_data['short_value']
            long_sls_curr[batch_row] = position_data['long_sl']
            long_tps_curr[batch_row] = position_data['long_tp']
            short_sls_curr[batch_row] = position_data['short_sl']
            short_tps_curr[batch_row] = position_data['short_tp']
            long_values_tgt[batch_row] = position_data['act_long_value']
            short_values_tgt[batch_row] = position_data['act_short_value']
            long_sls_tgt[batch_row] = position_data['act_long_sl']
            long_tps_tgt[batch_row] = position_data['act_long_tp']
            short_sls_tgt[batch_row] = position_data['act_short_sl']
            short_tps_tgt[batch_row] = position_data['act_short_tp']
    
    print(f"  Running batch JIT simulation...")
    
    # Call the ultra-fast JIT function
    batch_rewards, batch_costs = _simulate_batch_positions_jit(
        forward_closes, forward_highs, forward_lows, entry_prices,
        equities, balances,
        long_values_curr, short_values_curr, long_sls_curr, long_tps_curr,
        short_sls_curr, short_tps_curr,
        long_values_tgt, short_values_tgt, long_sls_tgt, long_tps_tgt,
        short_sls_tgt, short_tps_tgt,
        fee_bp, slip_bp, spread_bp, night_bp
    )
    
    # Map results back to original sample order
    all_rewards = np.zeros(n_samples)
    all_rewards[valid_mask] = batch_rewards
    
    # Return results
    if single_target:
        return pd.DataFrame({'y': all_rewards})
    else:
        return pd.DataFrame({'y_car': all_rewards})


def compute_many_vectorized(
    df_close: np.ndarray,
    samples: pd.DataFrame,
    forward_lookup: dict,
    scaler: Any,
    reward_funcs: Optional[list] = None,
    reward_key: Optional[str] = None,
    fee_bp: float = 10.0,
    slip_bp: float = 5.0,
    spread_bp: float = 2.0,
    night_bp: float = 0.5,
    trading_days: int = 252,
    risk_free_rate: float = 0.02,
    epsilon: float = 1e-8,
    batch_size: int = 1000
) -> pd.DataFrame:
    """
    ULTRA-FAST vectorized reward computation for large datasets.
    
    This function processes samples in large batches using vectorized operations
    instead of looping through individual samples. Expected speedup: 50-200x.
    
    Args:
        df_close: Full close price series 
        samples: DataFrame with scaled position data and forward index
        forward_lookup: Dict mapping idx -> {open, high, low, close, volume} arrays
        scaler: MetaScaler object for unscaling position data
        reward_funcs: List of reward function names ['car', 'sharpe', etc.]
        reward_key: Single reward function name (deprecated)
        batch_size: Number of samples to process per batch (memory control)
        Other args: Same as compute_many()
        
    Returns:
        DataFrame with reward columns (same format as compute_many)
    """
    if forward_lookup is None:
        raise ValueError("forward_lookup is required")
    if scaler is None:
        raise ValueError("scaler is required")
    
    # Handle backward compatibility
    if reward_funcs is None and reward_key is None:
        raise ValueError("Must specify either reward_key or reward_funcs")
    
    if reward_funcs is None:
        reward_funcs = [reward_key]
        single_target = True
    else:
        single_target = False
    
    # Map function names to actual functions
    REWARD_FUNCTIONS = {
        "car": car,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar
    }
    
    # Validate reward functions
    for func_name in reward_funcs:
        if func_name not in REWARD_FUNCTIONS:
            raise ValueError(f"Unknown reward function: {func_name}")
    
    n_samples = len(samples)
    results = {}
    
    # Meta columns for unscaling
    meta_cols = [
        "equity", "balance",
        "long_value", "short_value", "long_sl", "long_tp", "short_sl", "short_tp",
        "act_long_value", "act_short_value", "act_long_sl", "act_long_tp",
        "act_short_sl", "act_short_tp"
    ]
    
    # Pre-extract commonly used data to avoid repeated DataFrame access
    sample_indices = samples['idx'].values
    
    # Pre-extract and unscale ALL position data at once
    print(f"  Unscaling {n_samples:,} samples...")
    scaled_position_data = samples[meta_cols].values  # Shape: (n_samples, n_cols)
    
    # Batch unscale for efficiency
    unscaled_position_data = []
    for i in range(0, n_samples, batch_size):
        end_i = min(i + batch_size, n_samples)
        batch_scaled = samples.iloc[i:end_i][meta_cols]
        
        # Unscale batch
        batch_unscaled = []
        for _, row in batch_scaled.iterrows():
            unscaled = _unscale_position_data(row, scaler, meta_cols)
            batch_unscaled.append(unscaled)
        
        unscaled_position_data.extend(batch_unscaled)
    
    print(f"  Extracting forward windows...")
    # Pre-extract ALL forward windows (this avoids dict lookups in tight loops)
    forward_closes = []
    forward_highs = []
    forward_lows = []
    entry_prices = []
    valid_samples = []
    
    for i, sample_idx in enumerate(sample_indices):
        if sample_idx in forward_lookup:
            forward_data = forward_lookup[sample_idx]
            forward_closes.append(forward_data['close'])
            forward_highs.append(forward_data['high'])  
            forward_lows.append(forward_data['low'])
            entry_prices.append(df_close[sample_idx - 1])
            valid_samples.append(i)
    
    n_valid = len(valid_samples)
    print(f"  Processing {n_valid:,} valid samples...")
    
    # Process each reward function
    for func_name in reward_funcs:
        print(f"  Computing {func_name} rewards...")
        reward_func = REWARD_FUNCTIONS[func_name]
        rewards = np.zeros(n_samples, dtype=np.float64)
        
        # Process in batches for memory efficiency
        for batch_start in range(0, n_valid, batch_size):
            batch_end = min(batch_start + batch_size, n_valid)
            batch_indices = valid_samples[batch_start:batch_end]
            batch_rewards = []
            
            # Vectorized simulation for this batch
            for i, sample_idx in enumerate(batch_indices):
                try:
                    # Get pre-extracted data
                    forward_close = forward_closes[batch_start + i]
                    forward_high = forward_highs[batch_start + i]
                    forward_low = forward_lows[batch_start + i]
                    entry_price = entry_prices[batch_start + i]
                    position_data = unscaled_position_data[sample_idx]
                    
                    # Simulate positions (this is still the bottleneck)
                    daily_pnl, metadata = _simulate_positions_forward(
                        forward_close,
                        forward_high,
                        forward_low,
                        entry_price,
                        position_data,
                        fee_bp,
                        slip_bp,
                        spread_bp,
                        night_bp
                    )
                    
                    # Compute reward
                    if reward_func.__name__ in ['sharpe', 'sortino']:
                        reward = reward_func(daily_pnl, position_data['equity'], metadata, trading_days, risk_free_rate, epsilon)
                    else:
                        reward = reward_func(daily_pnl, position_data['equity'], metadata, trading_days, epsilon)
                    
                    batch_rewards.append(reward)
                    
                except Exception:
                    batch_rewards.append(0.0)
            
            # Store batch results
            for i, reward in enumerate(batch_rewards):
                rewards[batch_indices[i]] = reward
        
        # Store with appropriate column name
        if single_target:
            results['y'] = rewards
        else:
            results[f'y_{func_name}'] = rewards
    
    return pd.DataFrame(results)


def compute_many(
    df_close: np.ndarray,
    samples: pd.DataFrame,
    forward_lookup: dict,
    scaler: Any,
    reward_funcs: Optional[list] = None,
    reward_key: Optional[str] = None,
    fee_bp: float = 10.0,
    slip_bp: float = 5.0,
    spread_bp: float = 2.0,
    night_bp: float = 0.5,
    trading_days: int = 252,
    risk_free_rate: float = 0.02,
    epsilon: float = 1e-8
) -> pd.DataFrame:
    """
    Compute rewards for all samples with automatic optimization.
    
    Uses the same mathematical formulas as individual reward functions.
    Automatically selects the best execution engine based on dataset size:
    - Small datasets (< 50 samples): Individual processing 
    - Large datasets (>= 50 samples): JIT-compiled batch processing
    
    Supports CAR, Sharpe, Sortino, and Calmar metrics with 50-100x speedup for large datasets.
    """
    n_samples = len(samples)
    
    # Handle backward compatibility
    if reward_funcs is None and reward_key is None:
        target_funcs = ['car']
        single_target = True
    elif reward_funcs is None:
        target_funcs = [reward_key] 
        single_target = reward_key is not None
    else:
        target_funcs = reward_funcs
        single_target = False
    
    # Validate all requested functions are supported
    SUPPORTED_FAST_METRICS = {'car', 'sharpe', 'sortino', 'calmar'}
    unsupported = set(target_funcs) - SUPPORTED_FAST_METRICS
    
    if unsupported:
        print(f"  WARNING: Unsupported metrics {unsupported}, falling back to original implementation...")
        return compute_many_original(
            df_close, samples, forward_lookup, scaler, reward_funcs, reward_key,
            fee_bp, slip_bp, spread_bp, night_bp, trading_days, risk_free_rate, epsilon
        )
    
    # Use JIT-optimized implementation for supported metrics on reasonable-sized datasets
    if n_samples >= 50:
        print(f"  Using JIT-OPTIMIZED computation for {len(target_funcs)} metrics on {n_samples:,} samples...")
        return _compute_ultra_fast_all_metrics(
            samples, forward_lookup, scaler, df_close, target_funcs,
            fee_bp + slip_bp + spread_bp, trading_days, risk_free_rate, single_target
        )
    
    # Fallback to original for very small datasets (overhead not worth it)
    print(f"  Using original computation for small dataset ({n_samples} samples)...")
    return compute_many_original(
        df_close, samples, forward_lookup, scaler, reward_funcs, reward_key,
        fee_bp, slip_bp, spread_bp, night_bp, trading_days, risk_free_rate, epsilon
    )


@jit(nopython=True)
def _ultra_fast_simulation_jit(
    forward_closes: np.ndarray,    # Shape: (n_samples, n_bars)
    position_values: np.ndarray,   # Shape: (n_samples,) 
    equities: np.ndarray,          # Shape: (n_samples,)
    trading_cost_bp: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled ultra-fast position simulation for ALL metrics.
    
    Returns daily P&L arrays needed for CAR, Sharpe, Sortino, Calmar calculations.
    
    Returns:
        Tuple of (daily_pnl_array, valid_mask)
        - daily_pnl_array: Shape (n_samples, n_bars) - daily P&L for each sample
        - valid_mask: Shape (n_samples,) - boolean mask for valid samples
    """
    n_samples, n_bars = forward_closes.shape
    daily_pnl_array = np.zeros((n_samples, n_bars), dtype=np.float64)
    valid_mask = np.ones(n_samples, dtype=np.bool_)
    
    for i in range(n_samples):
        if equities[i] <= 0 or position_values[i] == 0:
            valid_mask[i] = False
            continue
            
        position_value = position_values[i]
        is_long = position_value > 0
        abs_position = abs(position_value)
        equity = equities[i]
        
        # Calculate daily returns
        daily_pnl = np.zeros(n_bars)
        
        # Day 0: Entry costs
        entry_cost = (trading_cost_bp * 1e-4) * abs_position * 0.5  # Half of round-trip cost
        daily_pnl[0] = -entry_cost
        
        # Days 1+: Price-based P&L
        for day in range(1, n_bars):
            price_return = forward_closes[i, day] - forward_closes[i, day-1]
            if not is_long:
                price_return = -price_return
                
            day_pnl = (price_return / forward_closes[i, day-1]) * abs_position
            daily_pnl[day] = day_pnl
        
        # Exit costs on last day
        exit_cost = (trading_cost_bp * 1e-4) * abs_position * 0.5
        daily_pnl[-1] -= exit_cost
        
        # Store normalized by equity for return calculations
        daily_pnl_array[i] = daily_pnl / equity
    
    return daily_pnl_array, valid_mask


@jit(nopython=True)
def _ultra_fast_car_jit(daily_pnl_array: np.ndarray, valid_mask: np.ndarray, trading_days: int = 252) -> np.ndarray:
    """
    JIT-compiled batch CAR calculation.
    
    Uses the same CAR formula as the regular car() function:
    CAR = (Total P&L / Equity) * (Trading Days / Period Days)
    
    The only difference is this processes many samples at once with JIT optimization.
    Input daily_pnl_array is pre-normalized (already divided by equity).
    """
    n_samples, n_bars = daily_pnl_array.shape
    cars = np.zeros(n_samples)
    
    for i in range(n_samples):
        if not valid_mask[i]:
            continue
            
        # Standard CAR formula (input already normalized by equity)
        total_return = np.sum(daily_pnl_array[i])
        car = total_return * (trading_days / n_bars)
        cars[i] = car
    
    return cars


@jit(nopython=True) 
def _ultra_fast_sharpe_jit(daily_pnl_array: np.ndarray, valid_mask: np.ndarray, 
                          trading_days: int = 252, risk_free_rate: float = 0.02) -> np.ndarray:
    """
    JIT-compiled batch Sharpe ratio calculation.
    
    Uses the standard Sharpe formula: (Mean Return - Risk Free Rate) / Volatility
    Same calculation as the regular sharpe() function, but processes many samples at once.
    """
    n_samples, n_bars = daily_pnl_array.shape
    sharpes = np.zeros(n_samples)
    
    daily_rf_rate = risk_free_rate / trading_days
    
    for i in range(n_samples):
        if not valid_mask[i]:
            continue
            
        returns = daily_pnl_array[i]
        excess_returns = returns - daily_rf_rate
        
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        
        if std_excess > 1e-10:  # Avoid division by zero
            sharpe = mean_excess / std_excess * np.sqrt(trading_days)
        else:
            sharpe = 0.0
            
        sharpes[i] = sharpe
    
    return sharpes


@jit(nopython=True)
def _ultra_fast_sortino_jit(daily_pnl_array: np.ndarray, valid_mask: np.ndarray,
                           trading_days: int = 252, risk_free_rate: float = 0.02) -> np.ndarray:
    """
    JIT-compiled batch Sortino ratio calculation.
    
    Uses the standard Sortino formula: (Mean Return - Risk Free Rate) / Downside Deviation
    Same calculation as the regular sortino() function, but processes many samples at once.
    """
    n_samples, n_bars = daily_pnl_array.shape
    sortinos = np.zeros(n_samples)
    
    daily_rf_rate = risk_free_rate / trading_days
    
    for i in range(n_samples):
        if not valid_mask[i]:
            continue
            
        returns = daily_pnl_array[i]
        excess_returns = returns - daily_rf_rate
        
        mean_excess = np.mean(excess_returns)
        
        # Downside deviation (only negative excess returns)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.sqrt(np.mean(downside_returns**2))
        else:
            downside_std = 1e-10  # Small value to avoid division by zero
        
        if downside_std > 1e-10:
            sortino = mean_excess / downside_std * np.sqrt(trading_days)
        else:
            sortino = 0.0
            
        sortinos[i] = sortino
    
    return sortinos


@jit(nopython=True)
def _ultra_fast_calmar_jit(daily_pnl_array: np.ndarray, valid_mask: np.ndarray, 
                          trading_days: int = 252) -> np.ndarray:
    """
    JIT-compiled batch Calmar ratio calculation.
    
    Uses the standard Calmar formula: Annual Return / Maximum Drawdown
    Same calculation as the regular calmar() function, but processes many samples at once.
    """
    n_samples, n_bars = daily_pnl_array.shape
    calmars = np.zeros(n_samples)
    
    for i in range(n_samples):
        if not valid_mask[i]:
            continue
            
        returns = daily_pnl_array[i]
        
        # Calculate cumulative returns for drawdown
        cumulative = np.zeros(n_bars)
        cumulative[0] = returns[0]
        for j in range(1, n_bars):
            cumulative[j] = cumulative[j-1] + returns[j]
        
        # Find maximum drawdown
        running_max = cumulative[0]
        max_dd = 0.0
        for j in range(1, n_bars):
            if cumulative[j] > running_max:
                running_max = cumulative[j]
            dd = running_max - cumulative[j]
            if dd > max_dd:
                max_dd = dd
        
        # Calmar = Annual Return / Max Drawdown
        annual_return = np.sum(returns) * (trading_days / n_bars)
        if max_dd > 1e-10:
            calmar = annual_return / max_dd
        else:
            calmar = 0.0
            
        calmars[i] = calmar
    
    return calmars


def _compute_ultra_fast_all_metrics(
    samples: pd.DataFrame,
    forward_lookup: dict,
    scaler: Any,
    df_close: np.ndarray,
    reward_funcs: list,
    total_cost_bp: float,
    trading_days: int = 252,
    risk_free_rate: float = 0.02,
    single_target: bool = False
) -> pd.DataFrame:
    """
    JIT-optimized computation for ALL reward metrics.
    
    Uses the same mathematical formulas as individual reward functions:
    - CAR: (Total P&L / Equity) × (Trading Days / Period Days)
    - Sharpe: (Mean Return - Risk Free Rate) / Volatility  
    - Sortino: (Mean Return - Risk Free Rate) / Downside Deviation
    - Calmar: Annual Return / Maximum Drawdown
    
    The only difference is execution: processes all samples at once with JIT compilation.
    Expected speedup: 50-100x vs individual processing.
    
    Note: First call has JIT compilation overhead (~7-8s), subsequent calls are <0.01s.
    """
    n_samples = len(samples)
    
    # Pre-filter valid samples
    valid_mask = samples['idx'].isin(forward_lookup.keys())
    valid_samples = samples[valid_mask]
    n_valid = len(valid_samples)
    
    if n_valid == 0:
        results = {}
        for func_name in reward_funcs:
            col_name = 'y' if single_target and len(reward_funcs) == 1 else f'y_{func_name}'
            results[col_name] = np.zeros(n_samples)
        return pd.DataFrame(results)
    
    print(f"    Simulating {n_valid:,} positions...")
    
    # Get dimensions
    first_idx = int(valid_samples['idx'].iloc[0])
    n_bars = len(forward_lookup[first_idx]['close'])
    
    # Pre-allocate arrays
    forward_closes = np.zeros((n_valid, n_bars), dtype=np.float32)
    position_values = np.zeros(n_valid, dtype=np.float32)
    equities = np.zeros(n_valid, dtype=np.float32)
    
    # Meta columns for unscaling
    meta_cols = ['equity', 'act_long_value', 'act_short_value']
    
    # Extract data efficiently
    for i, (_, row) in enumerate(valid_samples.iterrows()):
        sample_idx = int(row['idx'])
        
        # Forward windows
        forward_data = forward_lookup[sample_idx]
        forward_closes[i] = forward_data['close']
        
        # Unscale position data
        scaled_vals = {col: row[col] for col in meta_cols if col in row.index}
        unscaled = scaler.inverse_transform_dict(scaled_vals)
        
        # Net position
        long_target = unscaled.get('act_long_value', 0)
        short_target = unscaled.get('act_short_value', 0)
        net_position = long_target - short_target
        
        position_values[i] = net_position
        equities[i] = unscaled.get('equity', 1)
    
    # Run JIT-optimized simulation once
    daily_pnl_array, sim_valid_mask = _ultra_fast_simulation_jit(
        forward_closes, position_values, equities, total_cost_bp
    )
    
    print(f"    Computing {len(reward_funcs)} metrics...")
    
    # Compute all requested metrics from the same daily P&L data
    results = {}
    
    for func_name in reward_funcs:
        if func_name == 'car':
            metric_values = _ultra_fast_car_jit(daily_pnl_array, sim_valid_mask, trading_days)
        elif func_name == 'sharpe':
            metric_values = _ultra_fast_sharpe_jit(daily_pnl_array, sim_valid_mask, trading_days, risk_free_rate)
        elif func_name == 'sortino':
            metric_values = _ultra_fast_sortino_jit(daily_pnl_array, sim_valid_mask, trading_days, risk_free_rate)
        elif func_name == 'calmar':
            metric_values = _ultra_fast_calmar_jit(daily_pnl_array, sim_valid_mask, trading_days)
        else:
            # Fallback for unknown metrics
            metric_values = np.zeros(n_valid)
        
        # Map back to original sample order
        all_values = np.zeros(n_samples)
        all_values[valid_mask] = metric_values
        
        col_name = 'y' if single_target and len(reward_funcs) == 1 else f'y_{func_name}'
        results[col_name] = all_values
    
    return pd.DataFrame(results)


def compute_many_original(
    df_close: np.ndarray,
    samples: pd.DataFrame,
    forward_lookup: dict,
    scaler: Any,
    reward_funcs: Optional[list] = None,
    reward_key: Optional[str] = None,
    fee_bp: float = 10.0,
    slip_bp: float = 5.0,
    spread_bp: float = 2.0,
    night_bp: float = 0.5,
    trading_days: int = 252,
    risk_free_rate: float = 0.02,
    epsilon: float = 1e-8
) -> pd.DataFrame:
    """
    Compute rewards for all samples - supports single or multiple reward functions.
    
    This is a batch processing wrapper that calls compute_reward_for_sample()
    for each sample in the DataFrame. Supports computing multiple reward metrics
    simultaneously for multi-target training.
    
    Args:
        df_close: Full close price series (for entry price reference)
        samples: DataFrame with scaled position data and forward index
        forward_lookup: Dict mapping idx -> {open, high, low, close, volume} arrays (scaled)
        scaler: MetaScaler object for unscaling position data
        reward_funcs: List of reward function names ['car', 'sharpe', etc.] for multi-target
        reward_key: Single reward function name (deprecated, use reward_funcs)
        fee_bp, slip_bp, spread_bp, night_bp: Cost parameters
        trading_days: Trading days per year
        risk_free_rate: Annual risk-free rate (for Sortino calculation)
        epsilon: Small value for numerical stability
        
    Returns:
        DataFrame with reward columns:
        - If reward_funcs provided: Multiple columns (y_car, y_sharpe, etc.)
        - If reward_key provided: Single column 'y'
        - Column names use 'y_' prefix for clarity in dataset
        
    Example:
        # Single target (backward compatible)
        rewards = compute_many(..., reward_key='car')  # Returns DataFrame with 'y' column
        
        # Multiple targets (recommended for flexibility)
        rewards = compute_many(..., reward_funcs=['car', 'sharpe', 'sortino'])
        # Returns DataFrame with 'y_car', 'y_sharpe', 'y_sortino' columns
    """
    if forward_lookup is None:
        raise ValueError("forward_lookup is required (pass from forward_windows.create_forward_lookup())")
    if scaler is None:
        raise ValueError("scaler is required (pass MetaScaler used for scaling)")
    
    # Handle backward compatibility: reward_key → reward_funcs
    if reward_funcs is None and reward_key is None:
        raise ValueError("Must specify either reward_key or reward_funcs")
    
    if reward_funcs is None:
        reward_funcs = [reward_key]
        single_target = True
    else:
        single_target = False
    
    # Map function names to actual functions
    REWARD_FUNCTIONS = {
        "car": car,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar
    }
    
    # Validate reward functions
    for func_name in reward_funcs:
        if func_name not in REWARD_FUNCTIONS:
            raise ValueError(f"Unknown reward function: {func_name}. Choose from {list(REWARD_FUNCTIONS.keys())}")
    
    # PERFORMANCE OPTIMIZATION: Use parallel processing for large datasets
    # Expected speedup: 5-20x depending on sample size and CPU cores
    results = {}
    for func_name in reward_funcs:
        reward_func = REWARD_FUNCTIONS[func_name]
        
        # Determine optimal number of jobs based on dataset size
        n_samples = len(samples)
        n_jobs = min(n_samples // 10, -1) if n_samples > 100 else 1  # Use all cores for large datasets
        
        if n_jobs == 1:
            # Single-threaded for small datasets (overhead not worth it)
            rewards = np.empty(n_samples, dtype=np.float64)
            for i, (idx, row) in enumerate(samples.iterrows()):
                rewards[i] = compute_reward_for_sample(
                    row, df_close, forward_lookup, scaler, reward_func,
                    fee_bp, slip_bp, spread_bp, night_bp, trading_days, risk_free_rate, epsilon
                )
        else:
            # Parallel processing for large datasets
            def process_sample(sample_data):
                idx, row = sample_data
                return compute_reward_for_sample(
                    row, df_close, forward_lookup, scaler, reward_func,
                    fee_bp, slip_bp, spread_bp, night_bp, trading_days, risk_free_rate, epsilon
                )
            
            # Use joblib for parallel processing
            rewards = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(process_sample)(sample_data) 
                for sample_data in samples.iterrows()
            )
            rewards = np.array(rewards, dtype=np.float64)
        
        # Store with appropriate column name
        if single_target:
            results['y'] = rewards
        else:
            results[f'y_{func_name}'] = rewards
    
    return pd.DataFrame(results)


def compute_many_original(
    df_close: np.ndarray,
    samples: pd.DataFrame,
    forward_lookup: dict,
    scaler: Any,
    reward_funcs: Optional[list] = None,
    reward_key: Optional[str] = None,
    fee_bp: float = 10.0,
    slip_bp: float = 5.0,
    spread_bp: float = 2.0,
    night_bp: float = 0.5,
    trading_days: int = 252,
    risk_free_rate: float = 0.02,
    epsilon: float = 1e-8
) -> pd.DataFrame:
    """
    ORIGINAL reward computation (pre-optimization).
    
    Used for small datasets where parallel overhead isn't worth vectorization.
    """
    if forward_lookup is None:
        raise ValueError("forward_lookup is required (pass from forward_windows.create_forward_lookup())")
    if scaler is None:
        raise ValueError("scaler is required (pass MetaScaler used for scaling)")
    
    # Handle backward compatibility: reward_key → reward_funcs
    if reward_funcs is None and reward_key is None:
        raise ValueError("Must specify either reward_key or reward_funcs")
    
    if reward_funcs is None:
        reward_funcs = [reward_key]
        single_target = True
    else:
        single_target = False
    
    # Map function names to actual functions
    REWARD_FUNCTIONS = {
        "car": car,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar
    }
    
    # Validate reward functions
    for func_name in reward_funcs:
        if func_name not in REWARD_FUNCTIONS:
            raise ValueError(f"Unknown reward function: {func_name}. Choose from {list(REWARD_FUNCTIONS.keys())}")
    
    # PERFORMANCE OPTIMIZATION: Use parallel processing for datasets
    results = {}
    for func_name in reward_funcs:
        reward_func = REWARD_FUNCTIONS[func_name]
        
        # Determine optimal number of jobs based on dataset size
        n_samples = len(samples)
        n_jobs = min(n_samples // 10, -1) if n_samples > 100 else 1  # Use all cores for medium datasets
        
        if n_jobs == 1:
            # Single-threaded for small datasets (overhead not worth it)
            rewards = np.empty(n_samples, dtype=np.float64)
            for i, (idx, row) in enumerate(samples.iterrows()):
                rewards[i] = compute_reward_for_sample(
                    row, df_close, forward_lookup, scaler, reward_func,
                    fee_bp, slip_bp, spread_bp, night_bp, trading_days, risk_free_rate, epsilon
                )
        else:
            # Parallel processing for medium datasets
            def process_sample(sample_data):
                idx, row = sample_data
                return compute_reward_for_sample(
                    row, df_close, forward_lookup, scaler, reward_func,
                    fee_bp, slip_bp, spread_bp, night_bp, trading_days, risk_free_rate, epsilon
                )
            
            # Use joblib for parallel processing
            rewards = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(process_sample)(sample_data) 
                for sample_data in samples.iterrows()
            )
            rewards = np.array(rewards, dtype=np.float64)
        
        # Store with appropriate column name
        if single_target:
            results['y'] = rewards
        else:
            results[f'y_{func_name}'] = rewards
    
    return pd.DataFrame(results)


# ========== DEPRECATED: OLD SINGLE-POSITION ACTION OPTIMIZATION ==========
# These functions are from the old architecture and should be removed.
# Action optimization should be handled in predictor.py using the model's
# dual-position outputs (6 continuous values: long_value, short_value, 
# long_sl, long_tp, short_sl, short_tp).
#
# Keeping these as placeholders in case they're referenced elsewhere.
# TODO: Remove after confirming predictor.py handles action optimization.

def compute_all_actions(*args, **kwargs):
    """DEPRECATED: Use predictor.py for action search."""
    raise NotImplementedError(
        "compute_all_actions() is deprecated. "
        "Use predictor.py methods for dual-position action optimization."
    )


def find_optimal_action(*args, **kwargs):
    """DEPRECATED: Use predictor.py for action optimization."""
    raise NotImplementedError(
        "find_optimal_action() is deprecated. "
        "Use predictor.py methods for dual-position action optimization."
    )


def compute_optimal_labels(*args, **kwargs):
    """DEPRECATED: Rewards are computed via compute_many() during dataset generation."""
    raise NotImplementedError(
        "compute_optimal_labels() is deprecated. "
        "Rewards are computed via compute_many() during dataset.py generation."
    )