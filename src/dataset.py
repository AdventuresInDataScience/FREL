"""
High-level helper: build n samples, scale, attach y, save parquet.

Note: Optimal action computation is handled by the Predictor class during inference,
not during dataset generation.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from . import synth, reward, scale, curriculum


def build_dataset(
    cfg: dict, 
    n_samples: int, 
    seed: int = 42, 
    overwrite: bool = False
) -> Path:
    """
    Main convenience wrapper. Returns Path to gzipped parquet file.
    
    Args:
        cfg: Configuration dict
        n_samples: Number of samples to generate
        seed: Random seed
        overwrite: Force rebuild raw data if exists
        
    Returns:
        Path to saved parquet file
    """
    rng = np.random.default_rng(seed)
    data_dir = Path(cfg.get("data_dir", "data"))
    raw_filename = cfg.get("raw_data_filename", "raw_{ticker}.parquet").format(ticker=cfg['ticker'])
    raw_path = data_dir / raw_filename
    if raw_path.exists() and not overwrite:
        df_close = pd.read_parquet(raw_path)
    else:
        from . import data
        df_close = data.download(cfg["ticker"], cfg["start"])
        data.save(df_close, raw_path, cfg["parquet_compression"])

    # curriculum column
    df_close["phase"] = curriculum.assign_phase(df_close, 
                                                vol_window=cfg.get("curriculum_vol_window", 20),
                                                phase0_vol_pct=cfg["phase0_vol_pct"],
                                                phase0_skew_max=cfg["phase0_skew_max"])

    samples = synth.build_samples(df_close, n_samples, cfg["lookback"], cfg["forward"], rng, cfg)
    # attach forward column for reward
    samples["forward"] = cfg["forward"]

    # ---------- scaling ----------
    # 1) per-window OHLCV
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    # Scale all OHLCV columns together for each row
    def scale_row(row):
        ohlcv_dict = {col: row[col] for col in ohlcv_cols}
        scaled = scale.scale_ohlcv_window(ohlcv_dict)
        return pd.Series({f"{col}_scaled": scaled[col] for col in ohlcv_cols})
    
    scaled_ohlcv = samples.apply(scale_row, axis=1)
    samples = pd.concat([samples, scaled_ohlcv], axis=1)
    
    # 2) meta global min-max
    meta_cols = ["equity", "balance", "position", "sl_dist", "tp_dist", "act_dollar", "act_sl", "act_tp"]
    scaler = scale.MetaScaler(kind=cfg.get("scale_meta", "minmax"))
    scaler.fit(samples, meta_cols)
    samples = scaler.transform(samples, meta_cols, epsilon=cfg.get("epsilon", 1e-8))
    scaler_filename = cfg.get("scaler_filename", "meta_scaler.json")
    scaler.save(data_dir / scaler_filename)

    # ---------- reward ----------
    samples["y"] = reward.compute_many(df_close["close"].values, samples, cfg["reward_key"],
                                       cfg["fee_bps"], cfg["slippage_bps"], cfg["spread_bps"], cfg["overnight_bp"],
                                       trading_days=cfg.get("trading_days_per_year", 252),
                                       epsilon=cfg.get("epsilon", 1e-8))

    # ---------- save ----------
    samples_filename = cfg.get("samples_filename", "samples_{n}M.parquet")
    samples_filename = samples_filename.format(n=n_samples // 1_000_000)
    out_path = data_dir / samples_filename
    samples.to_parquet(out_path, compression=cfg["parquet_compression"], index=False)
    return out_path