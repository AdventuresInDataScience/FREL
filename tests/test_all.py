"""
Quick smoke tests for the new modules.
Run:  pytest tests/test_all.py
"""
import tempfile, json
from pathlib import Path
import pandas as pd, numpy as np, yaml
from src import data, synth, reward, scale, curriculum, dataset


def test_download():
    df = data.download("^GSPC", "2020-01-01")
    assert not df.empty
    assert set(df.columns) == {"open", "high", "low", "close", "volume"}


def test_synth():
    df = data.download("^GSPC", "2020-01-01")
    samples = synth.build_samples(df, 50, 200, 5, np.random.default_rng(0), {})
    assert len(samples) == 50
    assert "close" in samples.columns


def test_reward():
    close = np.arange(100, 110, dtype=float)
    samples = pd.DataFrame([dict(idx=0, forward=5, act_dir="long", act_dollar=1000)])
    y = reward.compute_many(close, samples, "car", 0.2, 0.1, 0.05, 2.0)
    assert y.shape == (1,)


def test_scaler():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    sc = scale.MetaScaler("minmax")
    sc.fit(df, ["a", "b"])
    df2 = sc.transform(df, ["a", "b"])
    assert df2["a"].min() == 0.0 and df2["a"].max() == 1.0


def test_curriculum():
    df = data.download("^GSPC", "2020-01-01")
    phase = curriculum.assign_phase(df)
    assert phase.nunique() == 3


def test_dataset():
    cfg = yaml.safe_load(open("config/default.yaml"))
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "data").mkdir()
        cfg["parquet_compression"] = "gzip"
        path = dataset.build_dataset(cfg, 5000, overwrite=True)
        assert path.exists()
        df = pd.read_parquet(path)
        assert "y" in df.columns