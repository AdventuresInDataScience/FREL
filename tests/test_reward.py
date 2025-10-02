"""
Tests for reward module including optimal action computation.
"""
import numpy as np
import pandas as pd
import pytest
from src import reward


@pytest.fixture
def simple_price():
    """Linearly increasing price series."""
    return np.linspace(100, 110, 20)


@pytest.fixture
def cost_params():
    return dict(fee_bp=0.2, slip_bp=0.1, spread_bp=0.05, night_bp=2.0)


def test_car_long_profit(simple_price, cost_params):
    """Long position should profit on rising prices."""
    action = {"dir": "long", "dollar": 1000, "sl": 0.05, "tp": 0.10}
    r = reward.car(simple_price, 0, 10, action, **cost_params)
    assert r > 0, "Long should profit on rising prices"


def test_car_short_loss(simple_price, cost_params):
    """Short position should lose on rising prices."""
    action = {"dir": "short", "dollar": 1000, "sl": 0.05, "tp": 0.10}
    r = reward.car(simple_price, 0, 10, action, **cost_params)
    assert r < 0, "Short should lose on rising prices"


def test_car_hold_zero(simple_price, cost_params):
    """Hold action should have zero reward."""
    action = {"dir": "hold", "dollar": 0, "sl": 0, "tp": 0}
    r = reward.car(simple_price, 0, 10, action, **cost_params)
    assert r == 0, "Hold should return zero"


def test_compute_many(simple_price, cost_params):
    """Test vectorized computation."""
    samples = pd.DataFrame([
        {"idx": 0, "forward": 10, "act_dir": "long", "act_dollar": 1000, "act_sl": 0.02, "act_tp": 0.04},
        {"idx": 5, "forward": 10, "act_dir": "short", "act_dollar": 1000, "act_sl": 0.02, "act_tp": 0.04},
        {"idx": 0, "forward": 5, "act_dir": "hold", "act_dollar": 0, "act_sl": 0, "act_tp": 0},
    ])
    y = reward.compute_many(simple_price, samples, "car", **cost_params)
    assert y.shape == (3,)
    assert y[0] > 0  # long profits
    assert y[1] < 0  # short loses
    assert y[2] == 0  # hold zero


def test_compute_all_actions(simple_price, cost_params):
    """Test sampling all actions."""
    df = reward.compute_all_actions(
        simple_price, 
        idx=0, 
        forward=10,
        reward_key="car",
        n_samples=100,
        seed=42,
        **cost_params
    )
    
    assert len(df) == 201  # 1 hold + 100 long + 100 short
    assert set(df["dir"].unique()) == {"hold", "long", "short"}
    assert df["reward"].dtype == np.float64
    
    # Long should have higher average reward on rising prices
    assert df[df["dir"] == "long"]["reward"].mean() > df[df["dir"] == "short"]["reward"].mean()


def test_find_optimal_action(simple_price, cost_params):
    """Test optimization finds reasonable action."""
    result = reward.find_optimal_action(
        simple_price,
        idx=0,
        forward=10,
        reward_key="car",
        maxiter=50,
        seed=42,
        **cost_params
    )
    
    assert "dir" in result
    assert "reward" in result
    assert result["dir"] in ["hold", "long", "short"]
    
    # On rising prices, optimal should be long
    assert result["dir"] == "long", "Optimal direction should be long on rising prices"
    assert result["reward"] > 0, "Optimal reward should be positive"


def test_find_optimal_action_flat_prices(cost_params):
    """On flat prices, hold should be optimal."""
    flat_prices = np.full(20, 100.0)
    result = reward.find_optimal_action(
        flat_prices,
        idx=0,
        forward=10,
        reward_key="car",
        maxiter=50,
        seed=42,
        **cost_params
    )
    
    # With costs, hold should dominate on flat prices
    assert result["reward"] <= 0, "Reward should be zero or negative due to costs"


def test_compute_optimal_labels(simple_price, cost_params):
    """Test batch optimal computation."""
    samples = pd.DataFrame([
        {"idx": 0, "forward": 10},
        {"idx": 5, "forward": 5},
    ])
    
    opt_df = reward.compute_optimal_labels(
        simple_price,
        samples,
        "car",
        method="sample",
        n_samples=50,
        seed=42,
        **cost_params
    )
    
    assert len(opt_df) == 2
    assert "opt_dir" in opt_df.columns
    assert "opt_reward" in opt_df.columns
    assert all(opt_df["opt_dir"].isin(["hold", "long", "short"]))


def test_sharpe_ratio(simple_price, cost_params):
    """Test Sharpe ratio computation."""
    action = {"dir": "long", "dollar": 1000, "sl": 0.05, "tp": 0.10}
    s = reward.sharpe(simple_price, 0, 10, action, **cost_params)
    assert isinstance(s, float)


def test_sortino_ratio(simple_price, cost_params):
    """Test Sortino ratio computation."""
    action = {"dir": "long", "dollar": 1000, "sl": 0.05, "tp": 0.10}
    s = reward.sortino(simple_price, 0, 10, action, **cost_params)
    assert isinstance(s, float)


def test_calmar_ratio(simple_price, cost_params):
    """Test Calmar ratio computation."""
    action = {"dir": "long", "dollar": 1000, "sl": 0.05, "tp": 0.10}
    c = reward.calmar(simple_price, 0, 10, action, **cost_params)
    assert isinstance(c, float)


@pytest.mark.parametrize("reward_key", ["car", "sharpe", "sortino", "calmar"])
def test_all_reward_metrics(simple_price, cost_params, reward_key):
    """Test all reward metrics work."""
    action = {"dir": "long", "dollar": 1000, "sl": 0.05, "tp": 0.10}
    func = {"car": reward.car, "sharpe": reward.sharpe, "sortino": reward.sortino, "calmar": reward.calmar}[reward_key]
    r = func(simple_price, 0, 10, action, **cost_params)
    assert isinstance(r, float)
    assert not np.isnan(r)


def test_compute_optimal_labels_optimize_method(simple_price, cost_params):
    """Test optimal labels with optimize method."""
    samples = pd.DataFrame([
        {"idx": 0, "forward": 10},
    ])
    
    opt_df = reward.compute_optimal_labels(
        simple_price,
        samples,
        "car",
        method="optimize",
        maxiter=20,
        seed=42,
        **cost_params
    )
    
    assert len(opt_df) == 1
    assert "opt_dir" in opt_df.columns
    assert "opt_reward" in opt_df.columns