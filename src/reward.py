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
def car(close: np.ndarray, idx: int, forward: int, action: dict, fee_bp: float, slip_bp: float, spread_bp: float, night_bp: float) -> float:
    """Compound annualised return (simple)."""
    pnl = _pnl_scalar(close, idx, forward, action, fee_bp, slip_bp, spread_bp, night_bp)
    years = forward / 252
    return (1 + pnl / action.get("dollar", 1e4)) ** (1 / years) - 1


def sharpe(close: np.ndarray, idx: int, forward: int, action: dict, fee_bp: float, slip_bp: float, spread_bp: float, night_bp: float) -> float:
    rets = _daily_returns(close, idx, forward, action, fee_bp, slip_bp, spread_bp, night_bp)
    return np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252)


def sortino(close: np.ndarray, idx: int, forward: int, action: dict, fee_bp: float, slip_bp: float, spread_bp: float, night_bp: float) -> float:
    rets = _daily_returns(close, idx, forward, action, fee_bp, slip_bp, spread_bp, night_bp)
    downside = rets[rets < 0]
    return np.mean(rets) / (np.std(downside) + 1e-8) * np.sqrt(252)


def calmar(close: np.ndarray, idx: int, forward: int, action: dict, fee_bp: float, slip_bp: float, spread_bp: float, night_bp: float) -> float:
    rets = _daily_returns(close, idx, forward, action, fee_bp, slip_bp, spread_bp, night_bp)
    cum = np.cumsum(rets)
    draw = np.maximum.accumulate(cum) - cum
    max_dd = np.max(draw) if draw.size else 1e-8
    annual_ret = np.sum(rets) * (252 / (forward - 1))
    return annual_ret / (max_dd + 1e-8)


# vectorised dispatcher
def compute_many(df_close: np.ndarray, samples: pd.DataFrame, reward_key: str, fee_bp: float, slip_bp: float, spread_bp: float, night_bp: float) -> np.ndarray:
    """Return ndarray[float] len == len(samples)."""
    func = {"car": car, "sharpe": sharpe, "sortino": sortino, "calmar": calmar}[reward_key]
    return np.array(
        [
            func(df_close, row.idx, row.forward, dict(dir=row.act_dir, dollar=row.act_dollar), fee_bp, slip_bp, spread_bp, night_bp)
            for _, row in samples.iterrows()
        ]
    )