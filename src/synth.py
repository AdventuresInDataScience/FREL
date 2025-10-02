"""
Synthetic trade-state + action generation.
build_samples() produces a DataFrame ready for reward/y-label computation.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any
import tqdm


@dataclass
class TradeState:
    equity: float
    balance: float
    position: float  # +1 long / -1 short / 0 flat
    sl_dist: float  # stop-distance in price units
    tp_dist: float


@dataclass
class Action:
    dir: str  # hold / long / short
    dollar: float
    sl: float  # distance
    tp: float


def _random_trade_state(rng: np.random.Generator, cfg: dict) -> TradeState:
    equity = rng.uniform(cfg.get("synth_equity_min", 1e4), cfg.get("synth_equity_max", 1e5))
    balance = equity - rng.uniform(cfg.get("synth_balance_offset_min", -5e3), cfg.get("synth_balance_offset_max", 5e3))
    position = rng.choice(cfg.get("synth_position_values", [0.0, 1.0, -1.0]))
    sl_dist = rng.uniform(cfg.get("synth_sl_min", 0.001), cfg.get("synth_sl_max", 0.05))
    tp_dist = rng.uniform(cfg.get("synth_tp_min", 0.001), cfg.get("synth_tp_max", 0.10))
    return TradeState(equity, balance, position, sl_dist, tp_dist)


def _random_action(rng: np.random.Generator, cfg: dict) -> Action:
    dir_ = rng.choice(["hold", "long", "short"])
    if dir_ == "hold":
        return Action("hold", 0.0, 0.0, 0.0)
    dollar = rng.uniform(cfg.get("synth_dollar_min", 1e3), cfg.get("synth_dollar_max", 5e4))
    sl = rng.uniform(cfg.get("synth_sl_min", 0.001), cfg.get("synth_sl_max", 0.05))
    tp = rng.uniform(cfg.get("synth_tp_min", 0.001), cfg.get("synth_tp_max", 0.10))
    return Action(dir_, dollar, sl, tp)


def build_samples(df: pd.DataFrame, n: int, lookback: int, forward: int, rng: np.random.Generator, cfg: dict = None) -> pd.DataFrame:
    """Return DataFrame with one row per sample."""
    if cfg is None:
        cfg = {}
    rows: List[Dict[str, Any]] = []
    max_idx = len(df) - lookback - forward
    for _ in tqdm.trange(n, desc="build samples"):
        idx = rng.integers(lookback, max_idx)
        ts = _random_trade_state(rng, cfg)
        act = _random_action(rng, cfg)
        rows.append(
            dict(
                idx=idx,
                open=df["open"].iloc[idx - lookback : idx].values,
                high=df["high"].iloc[idx - lookback : idx].values,
                low=df["low"].iloc[idx - lookback : idx].values,
                close=df["close"].iloc[idx - lookback : idx].values,
                volume=df["volume"].iloc[idx - lookback : idx].values,
                equity=ts.equity,
                balance=ts.balance,
                position=ts.position,
                sl_dist=ts.sl_dist,
                tp_dist=ts.tp_dist,
                act_dir=act.dir,
                act_dollar=act.dollar,
                act_sl=act.sl,
                act_tp=act.tp,
            )
        )
    return pd.DataFrame(rows)