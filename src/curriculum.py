"""
Assign curriculum phase (0,1,2) to each sample row.
Phase-0 = low-vol + low-skew centre.
"""
import numpy as np
import pandas as pd


def assign_phase(df: pd.DataFrame, vol_window: int = 20, phase0_vol_pct: float = 40, phase0_skew_max: float = 0.5) -> pd.Series:
    """Return Series[int] same index as df."""
    ret = np.log(df["close"] / df["close"].shift(1))
    vol = ret.rolling(vol_window).std()
    skew = ret.rolling(vol_window).skew()

    # volatility ranking
    vol_med = vol.median()
    vol_band = np.percentile(vol.dropna(), [50 - phase0_vol_pct / 2, 50 + phase0_vol_pct / 2])

    # masks
    low_vol = (vol >= vol_band[0]) & (vol <= vol_band[1])
    low_skew = skew.abs() <= phase0_skew_max

    phase = pd.Series(2, index=df.index)  # default
    phase.loc[low_vol & low_skew] = 0
    # phase 1 = either vol or skew moderate
    phase.loc[(low_vol | low_skew) & (phase == 2)] = 1
    return phase.astype(int)