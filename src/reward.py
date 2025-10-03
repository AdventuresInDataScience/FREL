"""
Vectorised reward engines with dual-position structure.

NEW STRUCTURE:
- Positions: long_value, short_value, long_sl, long_tp, short_sl, short_tp
- Actions: act_long_value, act_short_value, act_long_sl, act_long_tp, act_short_sl, act_short_tp
- SL/TP in multiplier notation (e.g., long_sl=0.95 = 5% stop, short_tp=0.90 = 10% profit)
- Forward windows: Pre-computed, accessed via forward_lookup dict
- Simulation: Dual long/short positions with proper SL/TP checking per bar
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from scipy.optimize import differential_evolution


# ========== HELPER FUNCTIONS ==========

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


def _calculate_overnight_costs(position_value: float, n_days: int, night_bp: float) -> float:
    """
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


# ========== SIMULATION ENGINE ==========

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
    
    Args:
        forward_close: Scaled close prices (forward window)
        forward_high: Scaled high prices (forward window) 
        forward_low: Scaled low prices (forward window)
        entry_price: Unscaled entry price (close[idx-1] used for scaling)
        position_data: Dict with unscaled position values:
            - equity, balance
            - long_value, short_value (current positions)
            - long_sl, long_tp, short_sl, short_tp (current SL/TP multipliers)
            - act_long_value, act_short_value (target positions)
            - act_long_sl, act_long_tp, act_short_sl, act_short_tp (target SL/TP)
        fee_bp, slip_bp, spread_bp, night_bp: Cost parameters
        
    Returns:
        daily_pnl: Array of daily P&L (length = forward-1)
        metadata: Dict with exit info, costs, etc.
    """
    n_bars = len(forward_close)
    daily_pnl = np.zeros(n_bars)
    
    # Unscale forward prices
    close_prices = forward_close * entry_price
    high_prices = forward_high * entry_price
    low_prices = forward_low * entry_price
    
    # Extract position data
    equity = position_data['equity']
    
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
    
    # Calculate starting drawdown context
    # If equity < balance, we're in drawdown (lost money from peak)
    balance = position_data.get('balance', equity)
    starting_dd_pct = 0.0
    if balance > equity and balance > 0:
        # We're down from balance (peak cash), calculate drawdown %
        starting_dd_pct = (balance - equity) / balance
    
    # Initialize tracking
    total_costs = 0.0
    metadata = {
        'long_exit_bar': None,
        'short_exit_bar': None,
        'long_exit_reason': None,
        'short_exit_reason': None,
        'total_costs': 0.0,
        'trade_duration_bars': n_bars,  # Will be updated to earliest exit
        'starting_dd_pct': starting_dd_pct  # Current drawdown before trade
    }
    
    # Step 1: Execute position transitions at bar 0 (before first overnight)
    # Close current positions
    if long_value_curr > 0:
        # Close long at close[0]
        pnl = long_value_curr * (close_prices[0] / entry_price - 1)
        cost = _calculate_costs(long_value_curr, fee_bp, slip_bp, spread_bp) / 2  # Only closing cost
        daily_pnl[0] += pnl - cost
        total_costs += cost
    
    if short_value_curr > 0:
        # Close short at close[0]
        pnl = short_value_curr * (1 - close_prices[0] / entry_price)
        cost = _calculate_costs(short_value_curr, fee_bp, slip_bp, spread_bp) / 2  # Only closing cost
        daily_pnl[0] += pnl - cost
        total_costs += cost
    
    # Open target positions
    if long_value_tgt > 0:
        cost = _calculate_costs(long_value_tgt, fee_bp, slip_bp, spread_bp) / 2  # Only opening cost
        daily_pnl[0] -= cost
        total_costs += cost
    
    if short_value_tgt > 0:
        cost = _calculate_costs(short_value_tgt, fee_bp, slip_bp, spread_bp) / 2  # Only opening cost
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
                cost = _calculate_costs(long_value_tgt, fee_bp, slip_bp, spread_bp) / 2
                overnight = _calculate_overnight_costs(long_value_tgt, bar, night_bp)
                bar_pnl += pnl - cost - overnight
                total_costs += cost + overnight
                long_active = False
                metadata['long_exit_bar'] = bar
                metadata['long_exit_reason'] = 'SL'
                # Update trade duration to earliest exit
                if metadata['trade_duration_bars'] > bar:
                    metadata['trade_duration_bars'] = bar
            elif hit_tp:
                # Exit at TP
                pnl = long_value_tgt * (tp_price / long_entry_price - 1)
                cost = _calculate_costs(long_value_tgt, fee_bp, slip_bp, spread_bp) / 2
                overnight = _calculate_overnight_costs(long_value_tgt, bar, night_bp)
                bar_pnl += pnl - cost - overnight
                total_costs += cost + overnight
                long_active = False
                metadata['long_exit_bar'] = bar
                metadata['long_exit_reason'] = 'TP'
                # Update trade duration to earliest exit
                if metadata['trade_duration_bars'] > bar:
                    metadata['trade_duration_bars'] = bar
            else:
                # Still active - just pay overnight cost
                overnight = _calculate_overnight_costs(long_value_tgt, 1, night_bp)
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
                cost = _calculate_costs(short_value_tgt, fee_bp, slip_bp, spread_bp) / 2
                overnight = _calculate_overnight_costs(short_value_tgt, bar, night_bp)
                bar_pnl += pnl - cost - overnight
                total_costs += cost + overnight
                short_active = False
                metadata['short_exit_bar'] = bar
                metadata['short_exit_reason'] = 'SL'
                # Update trade duration to earliest exit
                if metadata['trade_duration_bars'] > bar:
                    metadata['trade_duration_bars'] = bar
            elif hit_tp:
                # Exit at TP
                pnl = short_value_tgt * (1 - tp_price / short_entry_price)
                cost = _calculate_costs(short_value_tgt, fee_bp, slip_bp, spread_bp) / 2
                overnight = _calculate_overnight_costs(short_value_tgt, bar, night_bp)
                bar_pnl += pnl - cost - overnight
                total_costs += cost + overnight
                short_active = False
                metadata['short_exit_bar'] = bar
                metadata['short_exit_reason'] = 'TP'
                # Update trade duration to earliest exit
                if metadata['trade_duration_bars'] > bar:
                    metadata['trade_duration_bars'] = bar
            else:
                # Still active - just pay overnight cost
                overnight = _calculate_overnight_costs(short_value_tgt, 1, night_bp)
                bar_pnl -= overnight
                total_costs += overnight
        
        daily_pnl[bar] = bar_pnl
    
    # Step 3: Close any remaining positions at end of forward window
    if long_active:
        pnl = long_value_tgt * (close_prices[-1] / long_entry_price - 1)
        cost = _calculate_costs(long_value_tgt, fee_bp, slip_bp, spread_bp) / 2
        daily_pnl[-1] += pnl - cost
        total_costs += cost
        metadata['long_exit_bar'] = n_bars - 1
        metadata['long_exit_reason'] = 'end'
    
    if short_active:
        pnl = short_value_tgt * (1 - close_prices[-1] / short_entry_price)
        cost = _calculate_costs(short_value_tgt, fee_bp, slip_bp, spread_bp) / 2
        daily_pnl[-1] += pnl - cost
        total_costs += cost
        metadata['short_exit_bar'] = n_bars - 1
        metadata['short_exit_reason'] = 'end'
    
    metadata['total_costs'] = total_costs
    
    return daily_pnl, metadata


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
    
    Uses actual trade duration (when SL/TP hit) for proper annualization.
    If trade exits early at bar 50, only first 50 bars are used for calculation.
    
    Args:
        daily_pnl: Daily P&L array (full forward window)
        equity: Starting equity (for return calculation)
        metadata: Simulation metadata with 'trade_duration_bars' key
        trading_days: Trading days per year
        epsilon: Small value for stability
        
    Returns:
        Annualized return based on actual trade duration
    """
    if equity <= 0:
        return 0.0
    
    # Use only the PnL up to actual exit point
    actual_bars = metadata.get('trade_duration_bars', len(daily_pnl))
    actual_pnl = daily_pnl[:actual_bars]
    
    if len(actual_pnl) == 0:
        return 0.0
    
    total_pnl = np.sum(actual_pnl)
    years = actual_bars / trading_days
    
    if years <= 0:
        return 0.0
    
    # Annualize based on actual trade duration
    return (1 + total_pnl / equity) ** (1 / years) - 1


def sharpe(
    daily_pnl: np.ndarray,
    equity: float,
    metadata: dict,
    trading_days: int = 252,
    epsilon: float = 1e-8
) -> float:
    """
    Sharpe Ratio.
    
    Uses actual trade duration for proper calculation. Annualization factor
    is based on actual bars in trade, not full forward window.
    
    Args:
        daily_pnl: Daily P&L array (full forward window)
        equity: Starting equity (for return calculation)
        metadata: Simulation metadata with 'trade_duration_bars' key
        trading_days: Trading days per year
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
    
    if std_ret < epsilon:
        return 0.0
    
    # Annualization factor based on actual trade duration
    annualization_factor = np.sqrt(trading_days / actual_bars) if actual_bars > 0 else 1.0
    
    return mean_ret / std_ret * annualization_factor * np.sqrt(actual_bars)


def sortino(
    daily_pnl: np.ndarray,
    equity: float,
    metadata: dict,
    trading_days: int = 252,
    epsilon: float = 1e-8
) -> float:
    """
    Sortino Ratio (downside deviation).
    
    Uses actual trade duration for proper calculation. Only considers
    downside volatility from days with losses.
    
    Args:
        daily_pnl: Daily P&L array (full forward window)
        equity: Starting equity (for return calculation)
        metadata: Simulation metadata with 'trade_duration_bars' key
        trading_days: Trading days per year
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
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        # No downside - excellent! Return high value scaled by actual performance
        if mean_ret > 0:
            annualization_factor = np.sqrt(trading_days / actual_bars) if actual_bars > 0 else 1.0
            return mean_ret * annualization_factor * np.sqrt(actual_bars) / epsilon
        return 0.0
    
    downside_std = np.std(downside_returns)
    
    if downside_std < epsilon:
        return 0.0
    
    # Annualization factor based on actual trade duration
    annualization_factor = np.sqrt(trading_days / actual_bars) if actual_bars > 0 else 1.0
    
    return mean_ret / downside_std * annualization_factor * np.sqrt(actual_bars)


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
    
    # Calculate drawdown in dollar terms FROM STARTING EQUITY
    # We start at 0 (current equity level), but factor in existing drawdown
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown_dollars = running_max - cum_pnl
    max_dd_dollars_from_trade = np.max(drawdown_dollars) if len(drawdown_dollars) > 0 else 0.0
    
    # Convert to percentage of starting equity
    trade_dd_pct = max_dd_dollars_from_trade / equity
    
    # CRITICAL: Add starting drawdown to trade drawdown for total risk exposure
    # Example: Starting DD = 5%, Trade DD = 10% → Total DD = 15%
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
        reward = reward_func(daily_pnl, position_data['equity'], metadata, trading_days, epsilon)
        return reward
        
    except Exception as e:
        # Fallback to 0 on error (could log error here)
        return 0.0


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
    
    # Compute rewards for each function
    # TODO: PERFORMANCE OPTIMIZATION - This loop is slow for large datasets
    #       Candidate optimizations (apply after architecture is validated):
    #       1. Numba JIT: Add @numba.jit(nopython=True) to _simulate_positions_forward()
    #       2. Multiprocessing: Use joblib.Parallel to process samples in parallel
    #       3. Expected speedup: 40-240x (similar to synth.py optimizations)
    #       See docs/PERFORMANCE.md for benchmarking methodology
    results = {}
    for func_name in reward_funcs:
        reward_func = REWARD_FUNCTIONS[func_name]
        rewards = np.empty(len(samples), dtype=np.float64)
        
        # Process each sample
        for i, (idx, row) in enumerate(samples.iterrows()):
            rewards[i] = compute_reward_for_sample(
                row,
                df_close,
                forward_lookup,
                scaler,
                reward_func,
                fee_bp,
                slip_bp,
                spread_bp,
                night_bp,
                trading_days,
                epsilon
            )
        
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