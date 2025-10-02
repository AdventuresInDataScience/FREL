# Quick Reference Guide - Predictor Class

## Installation
Ensure you have these dependencies (already in pyproject.toml):
```bash
pip install scipy  # For differential_evolution optimizer
```

## Basic Usage

### 1. Training and Saving

```python
import yaml
from src.dataset import build_dataset
from src.model import build_tx_model

# Build dataset
CFG = yaml.safe_load(open("config/default.yaml"))
path = build_dataset(CFG, n_samples=100_000)
df = pd.read_parquet(path)

# Train model
model = build_tx_model(price_shape=(200, 5), meta_len=8)
model.compile(loss="mse", optimizer="adam")
model.fit([X_price, X_meta], y, epochs=10)
model.save("models/my_model.h5")
```

### 2. Loading Predictor

```python
from src.predictor import Predictor

predictor = Predictor.from_checkpoint(
    model_path="models/my_model.h5",
    scaler_path="data/meta_scaler.json",
    cfg=CFG,
    model_type="transformer"
)
```

### 3. Inference on Pre-scaled Test Data

```python
# Perfect for testing - data is already scaled
y_pred = predictor.predict(
    X_price=X_price_test,  # (N, 200, 5)
    X_meta=X_meta_test,    # (N, 8)
    raw=False
)
```

### 4. Inference on Raw Live Data

```python
# Perfect for live trading - predictor handles scaling
sample = {
    "ohlcv_window": {
        "open": live_data["open"].values[-200:],
        "high": live_data["high"].values[-200:],
        "low": live_data["low"].values[-200:],
        "close": live_data["close"].values[-200:],
        "volume": live_data["volume"].values[-200:],
    },
    "equity": 50000.0,
    "balance": 50000.0,
    "position": 0.0,
    "sl_dist": 0.02,
    "tp_dist": 0.04,
    "act_dollar": 1000.0,
    "act_sl": 0.02,
    "act_tp": 0.04
}

reward_pred = predictor.predict(sample=sample, raw=True)
```

### 5. Find Optimal Action (Sampling Method - Fast)

```python
# Sample 1000 random actions and pick best
optimal = predictor.find_optimal_action(
    ohlcv_window=ohlcv_window,
    state=state,
    dollar_range=(1000, 50000),
    sl_range=(0.001, 0.05),
    tp_range=(0.001, 0.10),
    method="sample",
    n_samples=1000,
    seed=42
)

print(f"Trade: {optimal['dir']}")
print(f"Position size: ${optimal['dollar']:.2f}")
print(f"Stop loss: {optimal['sl']*100:.2f}%")
print(f"Take profit: {optimal['tp']*100:.2f}%")
print(f"Expected reward: {optimal['pred_reward']:.4f}")
```

### 6. Find Optimal Action (Optimization - Accurate)

```python
# Use differential evolution optimizer
optimal = predictor.find_optimal_action(
    ohlcv_window=ohlcv_window,
    state=state,
    dollar_range=(1000, 50000),
    sl_range=(0.001, 0.05),
    tp_range=(0.001, 0.10),
    maxiter=100,
    seed=42
)
```

### 7. Predict All Actions (Exploration)

```python
# Get predictions for many random actions
all_actions = predictor.predict_all_actions(
    ohlcv_window=ohlcv_window,
    state=state,
    n_samples=500,
    seed=42
)

# Analyze the landscape
print(all_actions.nlargest(10, 'pred_reward'))
print(f"\nBest long: {all_actions[all_actions['dir']=='long']['pred_reward'].max():.4f}")
print(f"Best short: {all_actions[all_actions['dir']=='short']['pred_reward'].max():.4f}")
```

### 8. Compare Model vs Ground Truth

```python
# See how well your model predicts optimal actions
comparison = predictor.compare_predicted_vs_true(
    close=price_series,
    idx=start_idx,
    ohlcv_window=ohlcv_window,
    state=state,
    method="sample",
    n_samples=500,
    seed=42
)

print("Model predicted:", comparison["predicted_action"])
print("True optimal:", comparison["true_optimal_action"])
print(f"Gap: {comparison['optimality_gap']:.4f}")
print(f"Efficiency: {(1 - comparison['optimality_gap'] / comparison['true_optimal_action']['reward']) * 100:.1f}%")
```

### 9. Compute Optimal Labels for Dataset

```python
# Generate dataset with optimal labels
path = build_dataset(
    CFG,
    n_samples=1_000_000,
    compute_optimal=True,
    optimal_method="sample"  # Fast
)

df = pd.read_parquet(path)
# df now has: opt_dir, opt_dollar, opt_sl, opt_tp, opt_reward

# Train model to predict optimal actions
model.fit([X_price, X_meta], df["opt_reward"].values)
```

## Common Patterns

### Pattern 1: Live Trading Bot

```python
# Initialize once
predictor = Predictor.from_checkpoint(...)

while trading:
    # Get current market data
    ohlcv_window = fetch_last_n_bars(200)
    state = get_account_state()
    
    # Find best action
    action = predictor.find_optimal_action(
        ohlcv_window, 
        state,
        maxiter=50
    )
    
    # Execute if profitable
    if action["pred_reward"] > 0.01:  # 1% threshold
        execute_trade(action)
```

### Pattern 2: Model Evaluation

```python
# Test set evaluation
y_pred = predictor.predict(X_price_test, X_meta_test, raw=False)
mse = np.mean((y_test - y_pred) ** 2)

# Optimality gap analysis
gaps = []
for i in range(100):
    comp = predictor.compare_predicted_vs_true(...)
    gaps.append(comp["optimality_gap"])

print(f"Average optimality gap: {np.mean(gaps):.4f}")
```

### Pattern 3: Hyperparameter Search

```python
for sl_range in [(0.01, 0.03), (0.01, 0.05), (0.01, 0.10)]:
    optimal = predictor.find_optimal_action(
        ohlcv_window,
        state,
        sl_range=sl_range
    )
    print(f"SL range {sl_range}: reward={optimal['pred_reward']:.4f}")
```

## Tips

1. **Use `raw=False` for testing**: Pre-scaled data is faster and more accurate
2. **Use `raw=True` for production**: Predictor handles scaling automatically
3. **Start with sampling**: Optimization is 10-100x slower
4. **Batch predictions**: Pass arrays instead of loops
5. **Cache predictor**: Loading from checkpoint is slow, do it once
6. **Monitor optimality gap**: Track how well your model predicts optimal actions

## Error Handling

```python
try:
    optimal = predictor.find_optimal_action(...)
except Exception as e:
    print(f"Optimization failed: {e}")
    # Fallback to hold
    optimal = {"dir": "hold", "dollar": 0, "sl": 0, "tp": 0}
```

## Configuration

Key settings in `config/default.yaml`:
```yaml
reward_key: "car"        # car | sharpe | sortino | calmar
fee_bps: 0.2            # Transaction fees
slippage_bps: 0.1       # Slippage
spread_bps: 0.05        # Bid-ask spread
overnight_bp: 2.0       # Overnight holding cost
lookback: 200           # OHLCV window size
forward: 50             # Reward calculation horizon
```
