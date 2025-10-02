"""
Vectorised reward engines.
All functions expect:
  close  : ndarray[float] shape (T,)
  idx    : int               start bar
  forward: int
  action : dict with keys dir,dollar,sl,tp
  fee_*  : scalars in basis-points
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from scipy.optimize import differential_evolution


# ---------- helpers ----------
def _pnl_scalar(close: np.ndarray, idx: int, forward: int, action: dict, fee_bp: float, slip_bp: float, spread_bp: float, night_bp: float) -> float:
    """Single-sample P&L."""
    entry = close[idx]
    exit_ = close[idx + forward - 1]
    if action["dir"] == "hold":
        return 0.0
    if action["dir"] == "long":
        raw = (exit_ - entry) / entry * action["dollar"]
    else:  # short
        raw = (entry - exit_) / entry * action["dollar"]
    cost = (fee_bp + slip_bp + spread_bp / 2) * 1e-4 * action["dollar"]
    cost += night_bp * 1e-4 * action["dollar"] * (forward - 1)
    return raw - cost


def _daily_returns(close: np.ndarray, idx: int, forward: int, action: dict, fee_bp: float, slip_bp: float, spread_bp: float, night_bp: float) -> np.ndarray:
    """Return daily P&L vector for Sharpe/Sortino."""
    if action["dir"] == "hold":
        return np.zeros(forward - 1)
    prices = close[idx : idx + forward]
    if action["dir"] == "long":
        rets = np.diff(prices) / prices[:-1]
    else:
        rets = -np.diff(prices) / prices[:-1]
    dollar = action["dollar"]
    costs = (fee_bp + slip_bp + spread_bp / 2) * 1e-4 * dollar + night_bp * 1e-4 * dollar
    return rets * dollar - costs


# ---------- public registry ----------
def car(close: np.ndarray, idx: int, forward: int, action: dict, fee_bp: float, slip_bp: float, spread_bp: float, night_bp: float, trading_days: int = 252, default_dollar: float = 1e4) -> float:
    """Compound annualised return (simple)."""
    pnl = _pnl_scalar(close, idx, forward, action, fee_bp, slip_bp, spread_bp, night_bp)
    years = forward / trading_days
    dollar = action.get("dollar", default_dollar)
    # Handle zero dollar (hold action) - return 0 reward
    if dollar <= 0:
        return 0.0
    return (1 + pnl / dollar) ** (1 / years) - 1


def sharpe(close: np.ndarray, idx: int, forward: int, action: dict, fee_bp: float, slip_bp: float, spread_bp: float, night_bp: float, trading_days: int = 252, epsilon: float = 1e-8) -> float:
    rets = _daily_returns(close, idx, forward, action, fee_bp, slip_bp, spread_bp, night_bp)
    return np.mean(rets) / (np.std(rets) + epsilon) * np.sqrt(trading_days)


def sortino(close: np.ndarray, idx: int, forward: int, action: dict, fee_bp: float, slip_bp: float, spread_bp: float, night_bp: float, trading_days: int = 252, epsilon: float = 1e-8) -> float:
    rets = _daily_returns(close, idx, forward, action, fee_bp, slip_bp, spread_bp, night_bp)
    downside = rets[rets < 0]
    return np.mean(rets) / (np.std(downside) + epsilon) * np.sqrt(trading_days)


def calmar(close: np.ndarray, idx: int, forward: int, action: dict, fee_bp: float, slip_bp: float, spread_bp: float, night_bp: float, trading_days: int = 252, epsilon: float = 1e-8) -> float:
    rets = _daily_returns(close, idx, forward, action, fee_bp, slip_bp, spread_bp, night_bp)
    cum = np.cumsum(rets)
    draw = np.maximum.accumulate(cum) - cum
    max_dd = np.max(draw) if draw.size else epsilon
    annual_ret = np.sum(rets) * (trading_days / (forward - 1))
    return annual_ret / (max_dd + epsilon)


# vectorised dispatcher
def compute_many(df_close: np.ndarray, samples: pd.DataFrame, reward_key: str, fee_bp: float, slip_bp: float, spread_bp: float, night_bp: float, trading_days: int = 252, epsilon: float = 1e-8, default_dollar: float = 1e4) -> np.ndarray:
    """Return ndarray[float] len == len(samples)."""
    func = {"car": car, "sharpe": sharpe, "sortino": sortino, "calmar": calmar}[reward_key]
    return np.array(
        [
            func(df_close, row.idx, row.forward, 
                 dict(dir=row.act_dir, dollar=row.act_dollar, sl=row.act_sl, tp=row.act_tp), 
                 fee_bp, slip_bp, spread_bp, night_bp, trading_days, epsilon if reward_key != 'car' else default_dollar)
            for _, row in samples.iterrows()
        ]
    )


def compute_all_actions(
    close: np.ndarray,
    idx: int,
    forward: int,
    reward_key: str,
    fee_bp: float,
    slip_bp: float,
    spread_bp: float,
    night_bp: float,
    dollar_range: Optional[Tuple[float, float]] = None,
    sl_range: Optional[Tuple[float, float]] = None,
    tp_range: Optional[Tuple[float, float]] = None,
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    trading_days: int = 252,
    epsilon: float = 1e-8,
    cfg: Optional[dict] = None
) -> pd.DataFrame:
    """
    Compute rewards for randomly sampled actions within bounds.
    
    Args:
        close: Price series
        idx: Starting index
        forward: Look-ahead bars
        reward_key: Reward metric ('car', 'sharpe', 'sortino', 'calmar')
        fee_bp, slip_bp, spread_bp, night_bp: Cost parameters
        dollar_range: (min, max) position size in dollars
        sl_range: (min, max) stop-loss distance as fraction
        tp_range: (min, max) take-profit distance as fraction
        n_samples: Number of random actions to sample per direction
        seed: Random seed for reproducibility
        trading_days: Number of trading days per year
        epsilon: Small value for numerical stability
        cfg: Configuration dict (overrides other parameters if provided)
        
    Returns:
        DataFrame with columns: dir, dollar, sl, tp, reward
    """
    # Use config defaults if not specified
    if cfg is None:
        cfg = {}
    if dollar_range is None:
        dollar_range = (cfg.get("action_dollar_min", 1e3), cfg.get("action_dollar_max", 5e4))
    if sl_range is None:
        sl_range = (cfg.get("action_sl_min", 0.001), cfg.get("action_sl_max", 0.05))
    if tp_range is None:
        tp_range = (cfg.get("action_tp_min", 0.001), cfg.get("action_tp_max", 0.10))
    if n_samples is None:
        n_samples = cfg.get("action_search_samples", 1000)
    
    rng = np.random.default_rng(seed)
    func = {"car": car, "sharpe": sharpe, "sortino": sortino, "calmar": calmar}[reward_key]
    
    results = []
    
    # Hold action
    results.append({
        "dir": "hold",
        "dollar": 0.0,
        "sl": 0.0,
        "tp": 0.0,
        "reward": 0.0
    })
    
    # Sample random actions for long/short
    for direction in ["long", "short"]:
        dollars = rng.uniform(dollar_range[0], dollar_range[1], n_samples)
        sls = rng.uniform(sl_range[0], sl_range[1], n_samples)
        tps = rng.uniform(tp_range[0], tp_range[1], n_samples)
        
        for d, sl, tp in zip(dollars, sls, tps):
            action = {"dir": direction, "dollar": d, "sl": sl, "tp": tp}
            try:
                if reward_key == 'car':
                    r = func(close, idx, forward, action, fee_bp, slip_bp, spread_bp, night_bp, trading_days, d)
                else:
                    r = func(close, idx, forward, action, fee_bp, slip_bp, spread_bp, night_bp, trading_days, epsilon)
                results.append({
                    "dir": direction,
                    "dollar": d,
                    "sl": sl,
                    "tp": tp,
                    "reward": r
                })
            except Exception:
                # Skip invalid actions
                continue
    
    return pd.DataFrame(results)


def find_optimal_action(
    close: np.ndarray,
    idx: int,
    forward: int,
    reward_key: str,
    fee_bp: float,
    slip_bp: float,
    spread_bp: float,
    night_bp: float,
    dollar_range: Optional[Tuple[float, float]] = None,
    sl_range: Optional[Tuple[float, float]] = None,
    tp_range: Optional[Tuple[float, float]] = None,
    maxiter: int = 100,
    seed: Optional[int] = None,
    trading_days: int = 252,
    epsilon: float = 1e-8,
    cfg: Optional[dict] = None
) -> Dict[str, Any]:
    """
    Find optimal action using differential evolution optimizer.
    
    Args:
        close: Price series
        idx: Starting index
        forward: Look-ahead bars
        reward_key: Reward metric ('car', 'sharpe', 'sortino', 'calmar')
        fee_bp, slip_bp, spread_bp, night_bp: Cost parameters
        dollar_range: (min, max) position size in dollars
        sl_range: (min, max) stop-loss distance as fraction
        tp_range: (min, max) take-profit distance as fraction
        maxiter: Maximum optimization iterations
        seed: Random seed for reproducibility
        trading_days: Number of trading days per year
        epsilon: Small value for numerical stability
        cfg: Configuration dict (overrides other parameters if provided)
        
    Returns:
        Dict with keys: dir, dollar, sl, tp, reward
    """
    # Use config defaults if not specified
    if cfg is None:
        cfg = {}
    if dollar_range is None:
        dollar_range = (cfg.get("action_dollar_min", 1e3), cfg.get("action_dollar_max", 5e4))
    if sl_range is None:
        sl_range = (cfg.get("action_sl_min", 0.001), cfg.get("action_sl_max", 0.05))
    if tp_range is None:
        tp_range = (cfg.get("action_tp_min", 0.001), cfg.get("action_tp_max", 0.10))
    
    func = {"car": car, "sharpe": sharpe, "sortino": sortino, "calmar": calmar}[reward_key]
    
    best_result = {"dir": "hold", "dollar": 0.0, "sl": 0.0, "tp": 0.0, "reward": 0.0}
    
    # Try long and short directions
    for direction in ["long", "short"]:
        def objective(x):
            """Negative reward (for minimization)."""
            action = {"dir": direction, "dollar": x[0], "sl": x[1], "tp": x[2]}
            try:
                if reward_key == 'car':
                    r = func(close, idx, forward, action, fee_bp, slip_bp, spread_bp, night_bp, trading_days, x[0])
                else:
                    r = func(close, idx, forward, action, fee_bp, slip_bp, spread_bp, night_bp, trading_days, epsilon)
                return -r  # Minimize negative = maximize positive
            except Exception:
                return 1e9  # Penalty for invalid actions
        
        bounds = [dollar_range, sl_range, tp_range]
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=maxiter,
                seed=seed,
                workers=1,
                updating='deferred',
                atol=1e-6,
                tol=1e-6
            )
            
            reward_val = -result.fun
            if reward_val > best_result["reward"]:
                best_result = {
                    "dir": direction,
                    "dollar": result.x[0],
                    "sl": result.x[1],
                    "tp": result.x[2],
                    "reward": reward_val
                }
        except Exception:
            # Skip if optimization fails
            continue
    
    return best_result


def compute_optimal_labels(
    df_close: np.ndarray,
    samples: pd.DataFrame,
    reward_key: str,
    fee_bp: float,
    slip_bp: float,
    spread_bp: float,
    night_bp: float,
    method: str = "optimize",
    **kwargs
) -> pd.DataFrame:
    """
    Compute optimal actions and rewards for all samples.
    
    Args:
        df_close: Full price series
        samples: DataFrame with 'idx' and 'forward' columns
        reward_key: Reward metric
        fee_bp, slip_bp, spread_bp, night_bp: Cost parameters
        method: 'optimize' (slower, better) or 'sample' (faster, approximate)
        **kwargs: Additional arguments for compute_all_actions or find_optimal_action
        
    Returns:
        DataFrame with columns: opt_dir, opt_dollar, opt_sl, opt_tp, opt_reward
    """
    results = []
    
    if method == "optimize":
        func = find_optimal_action
    else:
        def sample_best(close, idx, forward, reward_key, fee_bp, slip_bp, spread_bp, night_bp, **kw):
            df_actions = compute_all_actions(close, idx, forward, reward_key, fee_bp, slip_bp, spread_bp, night_bp, **kw)
            return df_actions.nlargest(1, 'reward').iloc[0].to_dict()
        func = sample_best
    
    for i, row in samples.iterrows():
        try:
            opt = func(
                df_close, 
                row.idx, 
                row.forward, 
                reward_key,
                fee_bp, 
                slip_bp, 
                spread_bp, 
                night_bp,
                **kwargs
            )
            results.append({
                "opt_dir": opt["dir"],
                "opt_dollar": opt["dollar"],
                "opt_sl": opt["sl"],
                "opt_tp": opt["tp"],
                "opt_reward": opt["reward"]
            })
        except Exception:
            # Fallback to hold
            results.append({
                "opt_dir": "hold",
                "opt_dollar": 0.0,
                "opt_sl": 0.0,
                "opt_tp": 0.0,
                "opt_reward": 0.0
            })
    
    return pd.DataFrame(results)