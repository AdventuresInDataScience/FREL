"""
Predictor for trained trading models with action optimization.

Handles inference, action sampling, and action optimization for dual-position trading.

Data Structure:
- DataFrame with columns:
  - OHLCV: open, high, low, close, volume (arrays of length lookback)
  - State (8): equity, balance, long_value, short_value, long_sl, long_tp, short_sl, short_tp
  - Action (6): act_long_value, act_short_value, act_long_sl, act_long_tp, act_short_sl, act_short_tp

Total: 5 OHLCV arrays + 14 meta scalars = model input
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from pathlib import Path
from scipy.optimize import differential_evolution
import torch
import torch.nn as nn
import lightgbm as lgb
import joblib

from . import scale


class Predictor:
    """
    Production-ready predictor for trained models.
    
    Core Methods:
    1. predict() - Make predictions on DataFrame (scaled or raw)
    2. predict_many_actions() - Sample actions and predict
    3. find_optimal_action() - Optimize action to maximize reward
    """
    
    # Expected column names
    OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']
    STATE_COLS = ['equity', 'balance', 'long_value', 'short_value', 
                  'long_sl', 'long_tp', 'short_sl', 'short_tp']
    ACTION_COLS = ['act_long_value', 'act_short_value', 'act_long_sl', 
                   'act_long_tp', 'act_short_sl', 'act_short_tp']
    META_COLS = STATE_COLS + ACTION_COLS  # 14 features total
    
    def __init__(
        self,
        model: Union[nn.Module, lgb.Booster],
        meta_scaler: Optional[scale.MetaScaler],
        cfg: Dict[str, Any],
        model_type: str = "transformer",
        device: str = 'cuda'
    ):
        """
        Initialize predictor.
        
        Args:
            model: Trained model
            meta_scaler: MetaScaler for raw→scaled conversion (None if only using pre-scaled)
            cfg: Config dict with lookback, forward, action constraints
            model_type: 'transformer', 'informer', 'fedformer', 'patchtst',
                       'itransformer', 'nbeats', 'nhits', 'lightgbm'
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.meta_scaler = meta_scaler
        self.cfg = cfg
        self.model_type = model_type.lower()
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Set eval mode for PyTorch
        if isinstance(model, nn.Module):
            self.model = self.model.to(self.device)
            self.model.eval()
        
        self.neural_models = ['transformer', 'informer', 'fedformer', 'patchtst',
                              'itransformer', 'nbeats', 'nhits']
        
        # Config params
        self.lookback = cfg.get('lookback', 200)
        self.forward = cfg.get('forward', 50)
        
        # Action space constraints (with defaults)
        self.action_value_max = cfg.get('action_value_max', 50000.0)
        self.action_sl_min = cfg.get('action_sl_min', 0.001)
        self.action_sl_max = cfg.get('action_sl_max', 0.05)
        self.action_tp_min = cfg.get('action_tp_min', 0.001)
        self.action_tp_max = cfg.get('action_tp_max', 0.10)
        self.action_search_samples = cfg.get('action_search_samples', 1000)
    
    @classmethod
    def from_checkpoint(
        cls,
        model_path: Union[str, Path],
        scaler_path: Union[str, Path],
        cfg: Dict[str, Any],
        model_type: str = 'transformer',
        device: Optional[str] = None
    ) -> "Predictor":
        """Load from checkpoint files."""
        model_path = Path(model_path)
        scaler_path = Path(scaler_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model_type_lower = model_type.lower()
        
        neural_models = ['transformer', 'informer', 'fedformer', 'patchtst',
                        'itransformer', 'nbeats', 'nhits']
        
        if model_type_lower in neural_models:
            from . import model as model_module
            
            checkpoint = torch.load(model_path, map_location=device)
            model_cfg = checkpoint.get('config', cfg)
            
            model_class = {
                'transformer': model_module.TransformerModel,
                'informer': model_module.InformerModel,
                'fedformer': model_module.FedFormerModel,
                'patchtst': model_module.PatchTSTModel,
                'itransformer': model_module.iTransformerModel,
                'nbeats': model_module.NBeatsModel,
                'nhits': model_module.NHiTSModel
            }[model_type_lower]
            
            lookback = model_cfg.get('lookback', cfg.get('lookback', 200))
            meta_len = model_cfg.get('meta_len', 14)
            
            model = model_class(
                price_shape=(lookback, 5),
                meta_len=meta_len,
                **{k: v for k, v in model_cfg.items()
                   if k not in ['lookback', 'meta_len', 'model_type', 'price_shape']}
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
        elif model_type_lower == 'lightgbm':
            model = joblib.load(model_path)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        meta_scaler = scale.MetaScaler.load(scaler_path)
        return cls(model, meta_scaler, cfg, model_type, device)
    
    def _validate_input(self, df: pd.DataFrame, scaled: bool = False) -> None:
        """
        Validate input DataFrame has correct columns and shapes.
        
        Args:
            df: Input DataFrame
            scaled: If True, expect scaled OHLCV columns (open_scaled, etc.)
        
        Raises:
            ValueError: If validation fails
        """
        # Check OHLCV columns
        if scaled:
            ohlcv_cols = [f"{col}_scaled" for col in self.OHLCV_COLS]
        else:
            ohlcv_cols = self.OHLCV_COLS
        
        missing_ohlcv = [col for col in ohlcv_cols if col not in df.columns]
        if missing_ohlcv:
            raise ValueError(f"Missing OHLCV columns: {missing_ohlcv}")
        
        # Check meta columns
        missing_meta = [col for col in self.META_COLS if col not in df.columns]
        if missing_meta:
            raise ValueError(f"Missing meta columns: {missing_meta}")
        
        # Validate OHLCV array shapes
        for col in ohlcv_cols:
            sample_array = df[col].iloc[0]
            if not isinstance(sample_array, np.ndarray):
                raise ValueError(f"Column '{col}' must contain numpy arrays")
            if sample_array.shape[0] != self.lookback:
                raise ValueError(
                    f"Column '{col}' has shape {sample_array.shape}, "
                    f"expected ({self.lookback},)"
                )
        
        # Validate meta are scalars
        for col in self.META_COLS:
            if df[col].dtype == object:
                raise ValueError(f"Column '{col}' must be numeric, not object")
    
    def _scale_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale raw DataFrame to model-ready format.
        
        Args:
            df: Raw DataFrame with OHLCV arrays + meta columns
        
        Returns:
            Scaled DataFrame with *_scaled columns
        """
        if self.meta_scaler is None:
            raise ValueError(
                "Cannot scale data without meta_scaler. "
                "Provide scaler in constructor or use scaled=True in predict()."
            )
        
        df = df.copy()
        
        # Initialize scaled columns
        for col in self.OHLCV_COLS:
            df[f"{col}_scaled"] = None
        
        # Scale OHLCV arrays
        for i in range(len(df)):
            ohlcv_dict = {col: df[col].iloc[i] for col in self.OHLCV_COLS}
            ohlcv_scaled = scale.scale_ohlcv_window(ohlcv_dict)
            
            for col in self.OHLCV_COLS:
                df.at[df.index[i], f"{col}_scaled"] = ohlcv_scaled[col]
        
        # Scale meta features
        df = self.meta_scaler.transform(df, self.META_COLS)
        
        return df
    
    def _prepare_model_inputs(
        self,
        df: pd.DataFrame,
        scaled: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert DataFrame to model input arrays.
        
        Args:
            df: DataFrame (scaled or raw)
            scaled: If True, use *_scaled columns; if False, scale first
        
        Returns:
            X_price: (N, lookback, 5)
            X_meta: (N, 14)
        """
        if not scaled:
            df = self._scale_df(df)
        
        # Extract OHLCV arrays
        ohlcv_cols = [f"{col}_scaled" for col in self.OHLCV_COLS]
        X_price_list = []
        for i in range(len(df)):
            arrays = [df[col].iloc[i] for col in ohlcv_cols]
            X_price_list.append(np.stack(arrays, axis=-1))  # (lookback, 5)
        
        X_price = np.stack(X_price_list)  # (N, lookback, 5)
        
        # Extract meta features
        X_meta = df[self.META_COLS].values  # (N, 14)
        
        return X_price, X_meta
    
    # =========================================================================
    # REQUIREMENT 2: Make predictions given full state + action
    # =========================================================================
    
    def predict(
        self,
        df: pd.DataFrame,
        scaled: bool = False
    ) -> np.ndarray:
        """
        Predict rewards for DataFrame rows.
        
        REQUIREMENT 2: Full state + action space prediction
        
        Args:
            df: DataFrame with OHLCV + state + action columns
                Columns:
                - OHLCV: open, high, low, close, volume (arrays)
                - State: equity, balance, long_value, short_value, long_sl, long_tp, short_sl, short_tp
                - Action: act_long_value, act_short_value, act_long_sl, act_long_tp, act_short_sl, act_short_tp
            scaled: If True, df already has *_scaled columns; if False, will scale first
        
        Returns:
            Predictions (N,)
        """
        # Validate input
        self._validate_input(df, scaled=scaled)
        
        # Prepare model inputs
        X_price, X_meta = self._prepare_model_inputs(df, scaled=scaled)
        
        # Predict
        if self.model_type in self.neural_models:
            X_price_t = torch.FloatTensor(X_price).to(self.device)
            X_meta_t = torch.FloatTensor(X_meta).to(self.device)
            
            with torch.no_grad():
                y_pred = self.model(X_price_t, X_meta_t).cpu().numpy()
        
        elif self.model_type == 'lightgbm':
            # Flatten price + concatenate meta
            X_flat = np.concatenate([
                X_price.reshape(len(X_price), -1),  # (N, lookback*5)
                X_meta                               # (N, 14)
            ], axis=1)
            y_pred = self.model.predict(X_flat)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        return y_pred.flatten()
    
    # =========================================================================
    # REQUIREMENT 3: Generate synthetic action samples with predictions
    # =========================================================================
    
    def predict_many_actions(
        self,
        ohlcv_arrays: Dict[str, np.ndarray],
        state: Dict[str, float],
        n_samples: Optional[int] = None,
        long_value_range: Optional[tuple[float, float]] = None,
        short_value_range: Optional[tuple[float, float]] = None,
        sl_range: Optional[tuple[float, float]] = None,
        tp_range: Optional[tuple[float, float]] = None,
        include_hold: bool = True,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate random actions and predict their rewards.
        
        REQUIREMENT 3: Synthetic action generation with predictions
        
        Args:
            ohlcv_arrays: Dict with 'open', 'high', 'low', 'close', 'volume' arrays (lookback,)
            state: Dict with 8 state features (equity, balance, positions)
            n_samples: Number of random actions to generate
            long_value_range, short_value_range, sl_range, tp_range: Action bounds
            include_hold: Include hold action (all zeros)
            seed: Random seed
        
        Returns:
            DataFrame with action columns + pred_reward
        """
        # Use config defaults
        n_samples = n_samples or self.action_search_samples
        long_value_range = long_value_range or (0.0, self.action_value_max)
        short_value_range = short_value_range or (0.0, self.action_value_max)
        sl_range = sl_range or (self.action_sl_min, self.action_sl_max)
        tp_range = tp_range or (self.action_tp_min, self.action_tp_max)
        
        rng = np.random.default_rng(seed)
        
        # Build DataFrame rows
        rows = []
        
        # Hold action
        if include_hold:
            row = {**ohlcv_arrays, **state}
            for col in self.ACTION_COLS:
                row[col] = 0.0
            rows.append(row)
        
        # Random actions
        for _ in range(n_samples):
            row = {**ohlcv_arrays, **state}
            row['act_long_value'] = rng.uniform(long_value_range[0], long_value_range[1])
            row['act_short_value'] = rng.uniform(short_value_range[0], short_value_range[1])
            row['act_long_sl'] = rng.uniform(sl_range[0], sl_range[1])
            row['act_long_tp'] = rng.uniform(tp_range[0], tp_range[1])
            row['act_short_sl'] = rng.uniform(sl_range[0], sl_range[1])
            row['act_short_tp'] = rng.uniform(tp_range[0], tp_range[1])
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Predict
        predictions = self.predict(df, scaled=False)
        df['pred_reward'] = predictions
        
        # Return only action + prediction columns
        result_cols = self.ACTION_COLS + ['pred_reward']
        return df[result_cols]
    
    # =========================================================================
    # REQUIREMENT 4: Optimize to find optimal action + prediction
    # =========================================================================
    
    def find_optimal_action(
        self,
        ohlcv_arrays: Dict[str, np.ndarray],
        state: Dict[str, float],
        long_value_range: Optional[tuple[float, float]] = None,
        short_value_range: Optional[tuple[float, float]] = None,
        sl_range: Optional[tuple[float, float]] = None,
        tp_range: Optional[tuple[float, float]] = None,
        maxiter: int = 100,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Find optimal action that maximizes predicted reward.
        
        REQUIREMENT 4: Action optimization with prediction
        
        Args:
            ohlcv_arrays: Dict with OHLCV arrays
            state: Dict with 8 state features
            long_value_range, short_value_range, sl_range, tp_range: Action bounds
            maxiter: Max optimization iterations
            seed: Random seed
        
        Returns:
            Dict with optimal action parameters + pred_reward
        """
        # Use config defaults
        long_value_range = long_value_range or (0.0, self.action_value_max)
        short_value_range = short_value_range or (0.0, self.action_value_max)
        sl_range = sl_range or (self.action_sl_min, self.action_sl_max)
        tp_range = tp_range or (self.action_tp_min, self.action_tp_max)
        
        # Objective function
        def objective(x: np.ndarray) -> float:
            """Minimize -reward (maximize reward)."""
            row = {
                **ohlcv_arrays,
                **state,
                'act_long_value': x[0],
                'act_long_sl': x[1],
                'act_long_tp': x[2],
                'act_short_value': x[3],
                'act_short_sl': x[4],
                'act_short_tp': x[5]
            }
            
            try:
                df = pd.DataFrame([row])
                pred = self.predict(df, scaled=False)[0]
                return -pred
            except Exception:
                return 1e9  # Penalty
        
        # Bounds
        bounds = [
            long_value_range,   # x[0]
            sl_range,           # x[1]
            tp_range,           # x[2]
            short_value_range,  # x[3]
            sl_range,           # x[4]
            tp_range            # x[5]
        ]
        
        # Optimize
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
            
            return {
                'act_long_value': result.x[0],
                'act_long_sl': result.x[1],
                'act_long_tp': result.x[2],
                'act_short_value': result.x[3],
                'act_short_sl': result.x[4],
                'act_short_tp': result.x[5],
                'pred_reward': -result.fun
            }
        
        except Exception as e:
            # Fallback to hold
            import warnings
            warnings.warn(f"Optimization failed: {e}. Returning hold action.")
            
            row = {**ohlcv_arrays, **state}
            for col in self.ACTION_COLS:
                row[col] = 0.0
            
            df = pd.DataFrame([row])
            pred = self.predict(df, scaled=False)[0]
            
            return {
                'act_long_value': 0.0,
                'act_long_sl': 0.0,
                'act_long_tp': 0.0,
                'act_short_value': 0.0,
                'act_short_sl': 0.0,
                'act_short_tp': 0.0,
                'pred_reward': pred
            }
    
    def find_optimal_action_long(
        self,
        ohlcv_arrays: Dict[str, np.ndarray],
        state: Dict[str, float],
        long_value_range: Optional[tuple[float, float]] = None,
        sl_range: Optional[tuple[float, float]] = None,
        tp_range: Optional[tuple[float, float]] = None,
        maxiter: int = 100,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Find optimal LONG-ONLY action that maximizes predicted reward.
        
        Convenience wrapper that forces short position to zero and only
        optimizes long position parameters.
        
        Args:
            ohlcv_arrays: Dict with OHLCV arrays
            state: Dict with 8 state features
            long_value_range, sl_range, tp_range: Long position action bounds
            maxiter: Max optimization iterations
            seed: Random seed
        
        Returns:
            Dict with optimal long action parameters (short set to 0) + pred_reward
        """
        # Force short position to zero by setting range to (0, 0)
        return self.find_optimal_action(
            ohlcv_arrays=ohlcv_arrays,
            state=state,
            long_value_range=long_value_range,
            short_value_range=(0.0, 0.0),  # Force short to zero
            sl_range=sl_range,
            tp_range=tp_range,
            maxiter=maxiter,
            seed=seed
        )
    
    def find_optimal_action_short(
        self,
        ohlcv_arrays: Dict[str, np.ndarray],
        state: Dict[str, float],
        short_value_range: Optional[tuple[float, float]] = None,
        sl_range: Optional[tuple[float, float]] = None,
        tp_range: Optional[tuple[float, float]] = None,
        maxiter: int = 100,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Find optimal SHORT-ONLY action that maximizes predicted reward.
        
        Convenience wrapper that forces long position to zero and only
        optimizes short position parameters.
        
        Args:
            ohlcv_arrays: Dict with OHLCV arrays
            state: Dict with 8 state features
            short_value_range, sl_range, tp_range: Short position action bounds
            maxiter: Max optimization iterations
            seed: Random seed
        
        Returns:
            Dict with optimal short action parameters (long set to 0) + pred_reward
        """
        # Force long position to zero by setting range to (0, 0)
        return self.find_optimal_action(
            ohlcv_arrays=ohlcv_arrays,
            state=state,
            long_value_range=(0.0, 0.0),  # Force long to zero
            short_value_range=short_value_range,
            sl_range=sl_range,
            tp_range=tp_range,
            maxiter=maxiter,
            seed=seed
        )
    
    def __repr__(self) -> str:
        return (
            f"Predictor(\n"
            f"  model_type={self.model_type},\n"
            f"  has_scaler={self.meta_scaler is not None},\n"
            f"  lookback={self.lookback},\n"
            f"  forward={self.forward},\n"
            f"  device={self.device}\n"
            f")"
        )
