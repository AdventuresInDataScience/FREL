"""
High-level helper: build n samples, scale, attach y, save parquet, return paths.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from . import synth, reward, scale, curriculum


def build_dataset(cfg: dict, n_samples: int, seed: int = 42, overwrite: bool = False) -> Path:
    """
    Main convenience wrapper. Returns Path to gzipped parquet file.
    """
    rng = np.random.default_rng(seed)
    raw_path = Path(f"data/raw_{cfg['ticker']}.parquet")
    if raw_path.exists() and not overwrite:
        df_close = pd.read_parquet(raw_path)
    else:
        from . import data
        df_close = data.download(cfg["ticker"], cfg["start"])
        data.save(df_close, raw_path, cfg["parquet_compression"])

    # curriculum column
    df_close["phase"] = curriculum.assign_phase(df_close, vol_window=20,
                                                phase0_vol_pct=cfg["phase0_vol_pct"],
                                                phase0_skew_max=cfg["phase0_skew_max"])

    samples = synth.build_samples(df_close, n_samples, cfg["lookback"], cfg["forward"], rng)
    # attach forward column for reward
    samples["forward"] = cfg["forward"]

    # ---------- scaling ----------
    # 1) per-window OHLCV
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    for col in ohlcv_cols:
        samples[f"{col}_scaled"] = samples[col].apply(lambda x: scale.scale_ohlcv_window(dict(zip(ohlcv_cols, x)))[col])
    # 2) meta global min-max
    meta_cols = ["equity", "balance", "position", "sl_dist", "tp_dist", "act_dollar", "act_sl", "act_tp"]
    scaler = scale.MetaScaler(kind=cfg.get("scale_meta", "minmax"))
    scaler.fit(samples, meta_cols)
    samples = scaler.transform(samples, meta_cols)
    scaler.save(Path("data/meta_scaler.json"))

    # ---------- reward ----------
    samples["y"] = reward.compute_many(df_close["close"].values, samples, cfg["reward_key"],
                                       cfg["fee_bps"], cfg["slippage_bps"], cfg["spread_bps"], cfg["overnight_bp"])

    # ---------- save ----------
    out_path = Path(f"data/samples_{n_samples // 1_000_000}M.parquet")
    samples.to_parquet(out_path, compression=cfg["parquet_compression"], index=False)
    return out_path