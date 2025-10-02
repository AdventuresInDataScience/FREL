"""
Predictor class for model inference with optimal action search.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
from scipy.optimize import differential_evolution
import torch
import torch.nn as nn
import lightgbm as lgb

from . import reward, scale


class Predictor:
    """
    Wrapper for trained models with action optimization capabilities.
    
    Handles both raw (unscaled) and pre-scaled data.
    """
    
    def __init__(
        self,
        model: Union[nn.Module, lgb.Booster],
        meta_scaler: Optional[scale.MetaScaler] = None,
        cfg: Optional[Dict] = None,
        model_type: str = "transformer",
        device: str = 'cuda'
    ):
        """
        Initialize predictor.
        
        Args:
            model: Trained model (PyTorch or LightGBM)
            meta_scaler: MetaScaler for raw data (optional if only using pre-scaled)
            cfg: Configuration dict with reward parameters
            model_type: Model architecture type. Options:
                - Neural models: 'transformer', 'informer', 'fedformer', 'patchtst', 
                                 'itransformer', 'nbeats', 'nhits'
                - Tree model: 'lightgbm'
            device: 'cuda' or 'cpu' for PyTorch models
        """
        self.model = model
        self.meta_scaler = meta_scaler
        self.cfg = cfg or {}
        self.model_type = model_type.lower()
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Move PyTorch model to device
        if isinstance(model, nn.Module):
            self.model = self.model.to(self.device)
            self.model.eval()
        
        # Define which models are neural (vs tree-based)
        self.neural_models = ['transformer', 'informer', 'fedformer', 'patchtst', 
                              'itransformer', 'nbeats', 'nhits']
        
        # Extract reward config
        self.reward_key = self.cfg.get("reward_key", "car")
        self.fee_bp = self.cfg.get("fee_bps", 0.2)
        self.slip_bp = self.cfg.get("slippage_bps", 0.1)
        self.spread_bp = self.cfg.get("spread_bps", 0.05)
        self.night_bp = self.cfg.get("overnight_bp", 2.0)
        self.lookback = self.cfg.get("lookback", 200)
        self.forward = self.cfg.get("forward", 50)
    
    @classmethod
    def from_checkpoint(
        cls,
        model_path: Union[str, Path],
        scaler_path: Union[str, Path],
        cfg: Dict,
        model_type: str = "transformer"
    ) -> "Predictor":
        """
        Load predictor from saved model and scaler.
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved MetaScaler JSON
            cfg: Configuration dict
            model_type: Model architecture type (transformer, lightgbm, informer, etc.)
        
        Returns:
            Predictor instance
        """
        model_type_lower = model_type.lower()
        neural_models = ['transformer', 'informer', 'fedformer', 'patchtst', 
                        'itransformer', 'nbeats', 'nhits']
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if model_type_lower in neural_models:
            # Load PyTorch model
            from . import model as model_module
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            saved_cfg = checkpoint.get('config', cfg)
            
            # Build model architecture
            model = model_module.build_model(saved_cfg, device=device)
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
        elif model_type_lower == 'lightgbm':
            import joblib
            model = joblib.load(model_path)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        meta_scaler = scale.MetaScaler.load(Path(scaler_path))
        
        return cls(model, meta_scaler, cfg, model_type, device=device)
    
    def _scale_raw_sample(self, sample: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale raw sample data (OHLCV window + meta features).
        
        Args:
            sample: Dict with keys:
                - ohlcv_window: Dict with keys ['open', 'high', 'low', 'close', 'volume']
                                Each value is array of shape (lookback,)
                - equity, balance, position, sl_dist, tp_dist, act_dollar, act_sl, act_tp: scalars
        
        Returns:
            (X_price, X_meta) as numpy arrays
        """
        # Scale OHLCV window
        ohlcv_scaled = scale.scale_ohlcv_window(sample["ohlcv_window"])
        X_price = np.stack([
            ohlcv_scaled["open"],
            ohlcv_scaled["high"],
            ohlcv_scaled["low"],
            ohlcv_scaled["close"],
            ohlcv_scaled["volume"]
        ], axis=-1)  # (lookback, 5)
        
        # Scale meta features
        meta_cols = ["equity", "balance", "position", "sl_dist", "tp_dist", 
                     "act_dollar", "act_sl", "act_tp"]
        meta_df = pd.DataFrame([{k: sample[k] for k in meta_cols}])
        meta_scaled = self.meta_scaler.transform(meta_df, meta_cols)
        X_meta = meta_scaled[meta_cols].values[0]  # (8,)
        
        return X_price, X_meta
    
    def predict(
        self,
        X_price: Optional[np.ndarray] = None,
        X_meta: Optional[np.ndarray] = None,
        sample: Optional[Dict[str, Any]] = None,
        raw: bool = False
    ) -> np.ndarray:
        """
        Predict reward for given input(s).
        
        Args:
            X_price: Pre-scaled price data (N, lookback, 5) or (lookback, 5)
            X_meta: Pre-scaled meta features (N, 8) or (8,)
            sample: Raw sample dict (if raw=True)
            raw: If True, scale sample before prediction
        
        Returns:
            Predicted rewards (N,) or scalar
        """
        if raw:
            if sample is None:
                raise ValueError("Must provide 'sample' when raw=True")
            X_price, X_meta = self._scale_raw_sample(sample)
            X_price = X_price[np.newaxis, ...]  # (1, lookback, 5)
            X_meta = X_meta[np.newaxis, ...]    # (1, 8)
        else:
            if X_price is None or X_meta is None:
                raise ValueError("Must provide X_price and X_meta when raw=False")
            
            # Handle single sample
            if X_price.ndim == 2:
                X_price = X_price[np.newaxis, ...]
            if X_meta.ndim == 1:
                X_meta = X_meta[np.newaxis, ...]
        
        # Predict
        if self.model_type in self.neural_models:
            # PyTorch neural models
            X_price_t = torch.FloatTensor(X_price).to(self.device)
            X_meta_t = torch.FloatTensor(X_meta).to(self.device)
            
            with torch.no_grad():
                y_pred = self.model(X_price_t, X_meta_t).cpu().numpy()
                
        elif self.model_type == "lightgbm":
            # LightGBM: flatten price window
            X = np.concatenate([
                X_price.reshape(len(X_price), -1),
                X_meta
            ], axis=1)
            y_pred = self.model.predict(X)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        return y_pred.flatten()
    
    def predict_all_actions(
        self,
        ohlcv_window: Dict[str, np.ndarray],
        state: Dict[str, float],
        dollar_range: Optional[Tuple[float, float]] = None,
        sl_range: Optional[Tuple[float, float]] = None,
        tp_range: Optional[Tuple[float, float]] = None,
        n_samples: Optional[int] = None,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Predict rewards for many random actions using the model.
        
        Args:
            ohlcv_window: Dict with OHLCV arrays of shape (lookback,)
            state: Dict with keys: equity, balance, position, sl_dist, tp_dist
            dollar_range: (min, max) position size
            sl_range: (min, max) stop-loss distance
            tp_range: (min, max) take-profit distance
            n_samples: Number of actions per direction
            seed: Random seed
        
        Returns:
            DataFrame with columns: dir, dollar, sl, tp, pred_reward
        """
        # Use config defaults if not specified
        if dollar_range is None:
            dollar_range = (self.cfg.get("action_dollar_min", 1e3), self.cfg.get("action_dollar_max", 5e4))
        if sl_range is None:
            sl_range = (self.cfg.get("action_sl_min", 0.001), self.cfg.get("action_sl_max", 0.05))
        if tp_range is None:
            tp_range = (self.cfg.get("action_tp_min", 0.001), self.cfg.get("action_tp_max", 0.10))
        if n_samples is None:
            n_samples = self.cfg.get("action_search_samples", 1000)
        
        rng = np.random.default_rng(seed)
        results = []
        
        # Hold action
        sample = {
            "ohlcv_window": ohlcv_window,
            **state,
            "act_dollar": 0.0,
            "act_sl": 0.0,
            "act_tp": 0.0
        }
        pred = self.predict(sample=sample, raw=True)[0]
        results.append({
            "dir": "hold",
            "dollar": 0.0,
            "sl": 0.0,
            "tp": 0.0,
            "pred_reward": pred
        })
        
        # Sample actions for long/short
        for direction in ["long", "short"]:
            dollars = rng.uniform(dollar_range[0], dollar_range[1], n_samples)
            sls = rng.uniform(sl_range[0], sl_range[1], n_samples)
            tps = rng.uniform(tp_range[0], tp_range[1], n_samples)
            
            # Batch predict for efficiency
            batch_samples = []
            for d, sl, tp in zip(dollars, sls, tps):
                batch_samples.append({
                    "ohlcv_window": ohlcv_window,
                    **state,
                    "act_dollar": d,
                    "act_sl": sl,
                    "act_tp": tp
                })
            
            # Scale all samples
            X_prices, X_metas = [], []
            for s in batch_samples:
                xp, xm = self._scale_raw_sample(s)
                X_prices.append(xp)
                X_metas.append(xm)
            
            X_price = np.stack(X_prices)  # (n_samples, lookback, 5)
            X_meta = np.stack(X_metas)    # (n_samples, 8)
            
            preds = self.predict(X_price=X_price, X_meta=X_meta, raw=False)
            
            for (d, sl, tp), pred in zip(zip(dollars, sls, tps), preds):
                results.append({
                    "dir": direction,
                    "dollar": d,
                    "sl": sl,
                    "tp": tp,
                    "pred_reward": pred
                })
        
        return pd.DataFrame(results)
    
    def find_optimal_action(
        self,
        ohlcv_window: Dict[str, np.ndarray],
        state: Dict[str, float],
        dollar_range: Optional[Tuple[float, float]] = None,
        sl_range: Optional[Tuple[float, float]] = None,
        tp_range: Optional[Tuple[float, float]] = None,
        maxiter: int = 100,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Find optimal action using optimizer (model-based).
        
        Args:
            ohlcv_window: Dict with OHLCV arrays
            state: Dict with trading state
            dollar_range, sl_range, tp_range: Action bounds
            maxiter: Max optimization iterations
            seed: Random seed
        
        Returns:
            Dict with keys: dir, dollar, sl, tp, pred_reward
        """
        # Use config defaults if not specified
        if dollar_range is None:
            dollar_range = (self.cfg.get("action_dollar_min", 1e3), self.cfg.get("action_dollar_max", 5e4))
        if sl_range is None:
            sl_range = (self.cfg.get("action_sl_min", 0.001), self.cfg.get("action_sl_max", 0.05))
        if tp_range is None:
            tp_range = (self.cfg.get("action_tp_min", 0.001), self.cfg.get("action_tp_max", 0.10))
        
        best_result = {
            "dir": "hold",
            "dollar": 0.0,
            "sl": 0.0,
            "tp": 0.0,
            "pred_reward": self.predict(sample={
                "ohlcv_window": ohlcv_window,
                **state,
                "act_dollar": 0.0,
                "act_sl": 0.0,
                "act_tp": 0.0
            }, raw=True)[0]
        }
        
        for direction in ["long", "short"]:
            def objective(x):
                """Negative predicted reward."""
                sample = {
                    "ohlcv_window": ohlcv_window,
                    **state,
                    "act_dollar": x[0],
                    "act_sl": x[1],
                    "act_tp": x[2]
                }
                try:
                    pred = self.predict(sample=sample, raw=True)[0]
                    return -pred
                except Exception:
                    return 1e9
            
            bounds = [dollar_range, sl_range, tp_range]
            
            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    maxiter=maxiter,
                    seed=seed,
                    workers=1,
                    updating='deferred',
                    atol=1e-6,
                    tol=1e-6
                )
                
                pred_reward = -result.fun
                if pred_reward > best_result["pred_reward"]:
                    best_result = {
                        "dir": direction,
                        "dollar": result.x[0],
                        "sl": result.x[1],
                        "tp": result.x[2],
                        "pred_reward": pred_reward
                    }
            except Exception:
                continue
        
        return best_result
    
    def compute_true_reward(
        self,
        close: np.ndarray,
        idx: int,
        action: Dict[str, Any]
    ) -> float:
        """
        Compute actual reward for an action (ground truth).
        
        Args:
            close: Full price series
            idx: Starting index
            action: Dict with keys: dir, dollar, sl, tp
        
        Returns:
            True reward value
        """
        func = {
            "car": reward.car,
            "sharpe": reward.sharpe,
            "sortino": reward.sortino,
            "calmar": reward.calmar
        }[self.reward_key]
        
        return func(
            close, idx, self.forward, action,
            self.fee_bp, self.slip_bp, self.spread_bp, self.night_bp
        )
    
    def compare_predicted_vs_true(
        self,
        close: np.ndarray,
        idx: int,
        ohlcv_window: Dict[str, np.ndarray],
        state: Dict[str, float],
        method: str = "optimize",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare model's predicted optimal action vs true optimal action.
        
        Args:
            close: Full price series
            idx: Starting index
            ohlcv_window: OHLCV window
            state: Trading state
            method: 'optimize' or 'sample'
            **kwargs: Additional arguments for optimization
        
        Returns:
            Dict with predicted and true optimal actions and rewards
        """
        # Model's predicted optimal
        if method == "optimize":
            pred_opt = self.find_optimal_action(ohlcv_window, state, **kwargs)
        else:
            df = self.predict_all_actions(ohlcv_window, state, **kwargs)
            pred_opt = df.nlargest(1, "pred_reward").iloc[0].to_dict()
            pred_opt["pred_reward"] = pred_opt.pop("pred_reward")
        
        # True optimal (using reward functions)
        if method == "optimize":
            true_opt = reward.find_optimal_action(
                close, idx, self.forward, self.reward_key,
                self.fee_bp, self.slip_bp, self.spread_bp, self.night_bp,
                **kwargs
            )
        else:
            df_true = reward.compute_all_actions(
                close, idx, self.forward, self.reward_key,
                self.fee_bp, self.slip_bp, self.spread_bp, self.night_bp,
                **kwargs
            )
            true_opt = df_true.nlargest(1, "reward").iloc[0].to_dict()
        
        # Compute true reward for predicted action
        pred_action = {
            "dir": pred_opt["dir"],
            "dollar": pred_opt["dollar"],
            "sl": pred_opt["sl"],
            "tp": pred_opt["tp"]
        }
        pred_true_reward = self.compute_true_reward(close, idx, pred_action)
        
        return {
            "predicted_action": pred_opt,
            "true_optimal_action": true_opt,
            "predicted_true_reward": pred_true_reward,
            "optimality_gap": true_opt["reward"] - pred_true_reward
        }
