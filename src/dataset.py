"""
High-level helper: build n samples, scale, attach y, save parquet.

Note: Optimal action computation is handled by the Predictor class during inference,
not during dataset generation.

NEW: Forward windows are generated separately and saved to forward_windows.parquet
for efficient storage (~1000x space savings vs duplicating per sample).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from . import synth, reward, scale, curriculum, forward_windows


def build_dataset(
    cfg: dict, 
    n_samples: int, 
    seed: int = 42, 
    overwrite: bool = False,
    n_jobs: int = None
) -> Path:
    """
    Main convenience wrapper. Returns Path to gzipped parquet file.
    
    Args:
        cfg: Configuration dict
        n_samples: Number of samples to generate
        seed: Random seed
        overwrite: Force rebuild raw data if exists
        n_jobs: Number of parallel workers for large datasets (-1 = all cores, -2 = all but one, 1 = no parallelization)
                If None, uses value from cfg['performance']['n_jobs'] (default: -1)
        
    Returns:
        Path to saved parquet file
    """
    rng = np.random.default_rng(seed)
    
    # Get n_jobs from config if not explicitly provided
    if n_jobs is None:
        n_jobs = cfg.get("performance", {}).get("n_jobs", -1)
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

    # Use parallel processing for large datasets
    if n_samples >= 100000 and n_jobs != 1:
        print(f"Using parallel processing with n_jobs={n_jobs} for {n_samples:,} samples...")
        samples = synth.build_samples_parallel(df_close, n_samples, cfg["lookback"], cfg["forward"], rng, cfg=cfg, n_jobs=n_jobs)
    else:
        if n_samples >= 100000:
            print(f"Parallel processing disabled (n_jobs={n_jobs})")
        samples = synth.build_samples(df_close, n_samples, cfg["lookback"], cfg["forward"], rng, cfg)
    
    # attach forward column for reward
    samples["forward"] = cfg["forward"]

    # ---------- scaling ----------
    # 1) per-window OHLCV (rename to *_scaled, then scale in-place)
    import time
    t0 = time.perf_counter()
    
    n_samples = len(samples)
    use_fp16 = cfg.get("use_fp16", True)
    dtype = np.float16 if use_fp16 else np.float32
    
    # Rename columns to *_scaled (model expects these names)
    samples = samples.rename(columns={
        "open": "open_scaled",
        "high": "high_scaled", 
        "low": "low_scaled",
        "close": "close_scaled",
        "volume": "volume_scaled"
    })
    
    # Extract ALL arrays at once (vstack is expensive, do it once per column)
    ohlcv_cols = ["open_scaled", "high_scaled", "low_scaled", "close_scaled", "volume_scaled"]
    arrays_dict = {}
    for col in ohlcv_cols:
        arrays_dict[col] = np.vstack(samples[col].values)
    
    # Get last values for normalization
    last_close = arrays_dict["close_scaled"][:, -1:].reshape(-1, 1)
    last_volume = arrays_dict["volume_scaled"][:, -1:].reshape(-1, 1)
    
    t1 = time.perf_counter()
    print(f"  Extract arrays: {t1-t0:.2f}s")
    
    # Scale OHLC in-place (use pre-extracted arrays, convert to FP16)
    for col in ["open_scaled", "high_scaled", "low_scaled", "close_scaled"]:
        scaled_arrays = (arrays_dict[col] / last_close).astype(dtype)
        samples[col] = list(scaled_arrays)
    
    # Scale volume in-place
    log_volume = np.log1p(arrays_dict["volume_scaled"])
    log_last_volume = np.log1p(last_volume)
    with np.errstate(divide='ignore', invalid='ignore'):
        volume_scaled = log_volume / log_last_volume
        volume_scaled = np.where(log_last_volume == 0, 1.0, volume_scaled)
    samples["volume_scaled"] = list(volume_scaled.astype(dtype))
    
    t2 = time.perf_counter()
    print(f"  OHLCV scaling (in-place, {dtype.__name__}): {t2-t1:.2f}s")
    
    # 2) meta global min-max (convert to FP16 after scaling)
    # NEW: Updated column names for dual-position structure
    meta_cols = [
        "equity", "balance",
        "long_value", "short_value", "long_sl", "long_tp", "short_sl", "short_tp",
        "act_long_value", "act_short_value", "act_long_sl", "act_long_tp", "act_short_sl", "act_short_tp"
    ]
    scaler = scale.MetaScaler(kind=cfg.get("scale_meta", "minmax"))
    scaler.fit(samples, meta_cols)
    samples = scaler.transform(samples, meta_cols, epsilon=cfg.get("epsilon", 1e-8))
    
    # Convert meta to FP16 if enabled (they're in [0,1] range, sufficient precision)
    if use_fp16:
        for col in meta_cols:
            samples[col] = samples[col].astype(dtype)
    
    scaler_filename = cfg.get("scaler_filename", "meta_scaler.json")
    scaler.save(data_dir / scaler_filename)
    
    t3 = time.perf_counter()
    print(f"  Meta scaling: {t3-t2:.2f}s")

    # ---------- forward windows ----------
    # Generate forward OHLCV windows for unique indices (stored separately for efficiency)
    print("Generating forward windows...")
    unique_indices = samples["idx"].unique()
    print(f"  {len(unique_indices):,} unique indices out of {len(samples):,} samples")
    
    fw_df = forward_windows.generate_forward_windows(
        df_close, 
        unique_indices, 
        cfg["forward"],
        use_fp16=cfg.get("use_fp16", True)
    )
    
    # Save forward windows
    fw_filename = cfg.get("forward_windows_filename", "forward_windows.parquet")
    fw_path = data_dir / fw_filename
    forward_windows.save_forward_windows(fw_df, fw_path, cfg.get("parquet_compression", "snappy"))
    print(f"  Saved forward windows to {fw_path}")
    
    # Create fast lookup dict for reward computation
    forward_lookup = forward_windows.create_forward_lookup(fw_df)
    print(f"  Created forward lookup with {len(forward_lookup):,} entries")
    
    # Validate all samples have forward windows
    forward_windows.validate_forward_windows(fw_df, samples)
    print(f"  ✓ All samples have forward windows")
    
    t4 = time.perf_counter()
    print(f"  Forward windows: {t4-t3:.2f}s")

    # ---------- reward ----------
    # Compute reward(s) for training labels
    # Single target mode: reward_key='car' → DataFrame with 'y' column
    # Multi-target mode: reward_key=['car', 'sharpe'] → DataFrame with 'y_car', 'y_sharpe' columns
    
    reward_config = cfg["reward_key"]
    if isinstance(reward_config, list):
        print(f"Computing rewards using {len(reward_config)} metrics: {reward_config}...")
        rewards_df = reward.compute_many(
            df_close=df_close["close"].values,
            samples=samples,
            forward_lookup=forward_lookup,
            scaler=scaler,
            reward_funcs=reward_config,  # Multi-target mode
            fee_bp=cfg["fee_bps"],
            slip_bp=cfg["slippage_bps"],
            spread_bp=cfg["spread_bps"],
            night_bp=cfg["overnight_bp"],
            trading_days=cfg.get("trading_days_per_year", 252),
            risk_free_rate=cfg.get("risk_free_rate", 0.02),
            epsilon=cfg.get("epsilon", 1e-8)
        )
    else:
        print(f"Computing rewards using '{reward_config}' metric...")
        rewards_df = reward.compute_many(
            df_close=df_close["close"].values,
            samples=samples,
            forward_lookup=forward_lookup,
            scaler=scaler,
            reward_key=reward_config,  # Single target mode (backward compatible)
            fee_bp=cfg["fee_bps"],
            slip_bp=cfg["slippage_bps"],
            spread_bp=cfg["spread_bps"],
            night_bp=cfg["overnight_bp"],
            trading_days=cfg.get("trading_days_per_year", 252),
            risk_free_rate=cfg.get("risk_free_rate", 0.02),  # Add risk-free rate parameter
            epsilon=cfg.get("epsilon", 1e-8)
        )
    
    # Attach reward column(s) to samples DataFrame
    # Single target: rewards_df has 'y' column
    # Multi-target: rewards_df has 'y_car', 'y_sharpe', etc. columns
    for col in rewards_df.columns:
        samples[col] = rewards_df[col].values
    
    # Keep reward as FP32 for precision (it's the target variable)
    
    t5 = time.perf_counter()
    print(f"  Reward computation: {t5-t4:.2f}s")

    # ---------- save ----------
    if use_fp16:
        print(f"  Using FP16 for scaled arrays (~50% size reduction)")
    
    samples_filename = cfg.get("samples_filename", "samples_{n}M.parquet")
    # Fix: use len(samples) instead of n_samples variable (which might not exist)
    samples_filename = samples_filename.format(n=len(samples) // 1_000_000)
    out_path = data_dir / samples_filename
    
    # Always use PyArrow writer (more efficient for nested arrays than pandas.to_parquet)
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    # Setup compression
    compression = cfg.get("parquet_compression", "snappy")
    compression_level = cfg.get("compression_level", None)
    
    # Handle YAML's None becoming string "None" or null
    if compression_level in (None, "None", "null"):
        compression_level = None
    
    # Build writer kwargs (only include compression_level if specified)
    writer_kwargs = {"compression": compression}
    if compression_level is not None:
        writer_kwargs["compression_level"] = int(compression_level)
    
    chunk_size = cfg.get("chunk_rows", 5_000_000)
    n_samples_total = len(samples)  # Use actual length
    
    if n_samples_total > chunk_size:
        # Large dataset - use chunked writing to avoid OOM
        # NOTE: This writes ONE file with multiple internal "row groups", not separate files
        # Example: 10M samples → ONE file "samples_10M.parquet" with 2 row groups of 5M rows each
        print(f"  Saving in chunks of {chunk_size:,} rows...")
        
        # Write first chunk to create file with schema
        first_chunk = samples.iloc[:chunk_size]
        table = pa.Table.from_pandas(first_chunk, preserve_index=False)
        
        # Open writer for the SINGLE output file
        writer = pq.ParquetWriter(
            out_path,
            table.schema,
            **writer_kwargs
        )
        writer.write_table(table)  # Write first row group
        
        # Write remaining chunks as additional row groups to the SAME file
        for start_idx in range(chunk_size, n_samples_total, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples_total)
            chunk = samples.iloc[start_idx:end_idx]
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            writer.write_table(table)  # Append next row group
            print(f"    Written {end_idx:,} / {n_samples_total:,} rows")
        
        writer.close()  # Close the single file
    else:
        # Small dataset - write directly with PyArrow (still faster than pandas.to_parquet)
        table = pa.Table.from_pandas(samples, preserve_index=False)
        pq.write_table(table, out_path, **writer_kwargs)
    
    t6 = time.perf_counter()
    print(f"  Parquet save: {t6-t5:.2f}s")
    print(f"  TOTAL: {t6-t0:.2f}s")
    return out_path