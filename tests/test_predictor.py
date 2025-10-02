"""
Tests for Predictor class.
"""
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from pathlib import Path

from src.predictor import Predictor
from src.scale import MetaScaler
from src.model import build_tx_model


@pytest.fixture
def simple_price():
    """Rising price series."""
    return np.linspace(100, 110, 250)


@pytest.fixture
def mock_model():
    """Simple transformer model."""
    return build_tx_model(price_shape=(200, 5), meta_len=8, d_model=32, nhead=2, tx_blocks=1)


@pytest.fixture
def mock_scaler():
    """Fitted MetaScaler."""
    meta_cols = ["equity", "balance", "position", "sl_dist", "tp_dist", 
                 "act_dollar", "act_sl", "act_tp"]
    df = pd.DataFrame({
        "equity": np.random.uniform(1e4, 1e5, 100),
        "balance": np.random.uniform(1e4, 1e5, 100),
        "position": np.random.uniform(-1e4, 1e4, 100),
        "sl_dist": np.random.uniform(0, 0.05, 100),
        "tp_dist": np.random.uniform(0, 0.1, 100),
        "act_dollar": np.random.uniform(0, 5e4, 100),
        "act_sl": np.random.uniform(0, 0.05, 100),
        "act_tp": np.random.uniform(0, 0.1, 100),
    })
    scaler = MetaScaler(kind="minmax")
    scaler.fit(df, meta_cols)
    return scaler


@pytest.fixture
def cfg():
    """Mock config."""
    return {
        "reward_key": "car",
        "fee_bps": 0.2,
        "slippage_bps": 0.1,
        "spread_bps": 0.05,
        "overnight_bp": 2.0,
        "lookback": 200,
        "forward": 50
    }


@pytest.fixture
def predictor(mock_model, mock_scaler, cfg):
    """Predictor instance."""
    return Predictor(mock_model, mock_scaler, cfg, model_type="transformer")


@pytest.fixture
def sample_ohlcv(simple_price):
    """Sample OHLCV window."""
    return {
        "open": simple_price[:200],
        "high": simple_price[:200] * 1.01,
        "low": simple_price[:200] * 0.99,
        "close": simple_price[:200],
        "volume": np.random.uniform(1e6, 1e7, 200)
    }


@pytest.fixture
def sample_state():
    """Sample trading state."""
    return {
        "equity": 5e4,
        "balance": 5e4,
        "position": 0.0,
        "sl_dist": 0.02,
        "tp_dist": 0.04
    }


def test_predictor_init(predictor):
    """Test predictor initialization."""
    assert predictor.model is not None
    assert predictor.meta_scaler is not None
    assert predictor.reward_key == "car"


def test_predict_raw(predictor, sample_ohlcv, sample_state):
    """Test prediction with raw sample."""
    sample = {
        "ohlcv_window": sample_ohlcv,
        **sample_state,
        "act_dollar": 1000.0,
        "act_sl": 0.02,
        "act_tp": 0.04
    }
    
    pred = predictor.predict(sample=sample, raw=True)
    assert pred.shape == (1,)
    assert isinstance(pred[0], (float, np.floating))


def test_predict_batch_scaled(predictor):
    """Test prediction with pre-scaled batch."""
    X_price = np.random.randn(10, 200, 5)
    X_meta = np.random.randn(10, 8)
    
    preds = predictor.predict(X_price=X_price, X_meta=X_meta, raw=False)
    assert preds.shape == (10,)


def test_predict_single_scaled(predictor):
    """Test prediction with single pre-scaled sample."""
    X_price = np.random.randn(200, 5)
    X_meta = np.random.randn(8)
    
    pred = predictor.predict(X_price=X_price, X_meta=X_meta, raw=False)
    assert pred.shape == (1,)


def test_predict_all_actions(predictor, sample_ohlcv, sample_state):
    """Test sampling all actions."""
    df = predictor.predict_all_actions(
        sample_ohlcv,
        sample_state,
        n_samples=50,
        seed=42
    )
    
    assert len(df) == 101  # 1 hold + 50 long + 50 short
    assert "pred_reward" in df.columns
    assert set(df["dir"].unique()) == {"hold", "long", "short"}


def test_find_optimal_action(predictor, sample_ohlcv, sample_state):
    """Test optimization."""
    result = predictor.find_optimal_action(
        sample_ohlcv,
        sample_state,
        maxiter=20,
        seed=42
    )
    
    assert "dir" in result
    assert "pred_reward" in result
    assert result["dir"] in ["hold", "long", "short"]


def test_compute_true_reward(predictor, simple_price):
    """Test true reward computation."""
    action = {"dir": "long", "dollar": 1000, "sl": 0.02, "tp": 0.04}
    r = predictor.compute_true_reward(simple_price, idx=0, action=action)
    assert isinstance(r, (float, np.floating))


def test_compare_predicted_vs_true(predictor, simple_price, sample_ohlcv, sample_state):
    """Test comparison of predicted vs true optimal."""
    result = predictor.compare_predicted_vs_true(
        simple_price,
        idx=0,
        ohlcv_window=sample_ohlcv,
        state=sample_state,
        method="sample",
        n_samples=50,
        seed=42
    )
    
    assert "predicted_action" in result
    assert "true_optimal_action" in result
    assert "optimality_gap" in result


def test_predictor_from_checkpoint(mock_model, mock_scaler, cfg, tmp_path):
    """Test loading from checkpoint."""
    # Save model and scaler
    model_path = tmp_path / "model.h5"
    scaler_path = tmp_path / "scaler.json"
    
    mock_model.save(model_path)
    mock_scaler.save(scaler_path)
    
    # Load
    predictor = Predictor.from_checkpoint(
        model_path,
        scaler_path,
        cfg,
        model_type="transformer"
    )
    
    assert predictor.model is not None
    assert predictor.meta_scaler is not None


def test_predict_raises_without_inputs(predictor):
    """Test that predict raises when inputs are missing."""
    with pytest.raises(ValueError):
        predictor.predict(raw=True)  # No sample provided
    
    with pytest.raises(ValueError):
        predictor.predict(raw=False)  # No X_price/X_meta provided
