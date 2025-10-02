"""
MAPIE wrapper for confidence interval predictions.

Provides sklearn-compatible wrappers for PyTorch and LightGBM models,
and a MapiePredictor class that returns predictions at multiple confidence
levels as a clean DataFrame.

Usage:
    # Wrap your model
    wrapped_model = SklearnPyTorchWrapper(pytorch_model, model_type='transformer')
    
    # Create MapiePredictor
    mapie_pred = MapiePredictor(
        wrapped_model,
        method='plus',  # or 'base', 'minmax', 'naive'
        cv=5
    )
    
    # Fit on training data
    mapie_pred.fit(X_price_train, X_meta_train, y_train)
    
    # Get predictions at multiple confidence levels
    predictions_df = mapie_pred.predict_intervals(
        X_price_test, 
        X_meta_test,
        alphas=[0.05, 0.10, 0.20]  # 95%, 90%, 80% confidence
    )
"""
import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import lightgbm as lgb
from mapie.regression import MapieRegressor


class SklearnPyTorchWrapper(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible wrapper for PyTorch models.
    
    This wrapper allows MAPIE to use PyTorch models that expect
    two inputs (price and meta) by combining them for the sklearn interface.
    
    Args:
        model: PyTorch model expecting (price, meta) inputs
        model_type: Architecture type (transformer, informer, etc.)
        epochs: Training epochs (default: 50)
        batch_size: Training batch size (default: 32)
        lr: Learning rate (default: 0.001)
        device: Device to train on ('cuda' or 'cpu')
        verbose: Training verbosity (default: 0)
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_type: str = 'transformer',
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001,
        device: str = 'cuda',
        verbose: int = 0,
        lookback: int = 200,
        price_features: int = 5,
        meta_len: int = 8
    ):
        self.model = model
        self.model_type = model_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.verbose = verbose
        
        # Store shapes
        self.lookback = lookback
        self.price_features = price_features
        self.meta_len = meta_len
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SklearnPyTorchWrapper':
        """
        Fit the PyTorch model.
        
        Args:
            X: Combined features (N, lookback*price_features + meta_len)
            y: Target values (N,) or (N, 1)
        
        Returns:
            self
        """
        # Split X back into price and meta
        X_price, X_meta = self._split_X(X)
        
        # Convert to torch tensors
        X_price_t = torch.FloatTensor(X_price).to(self.device)
        X_meta_t = torch.FloatTensor(X_meta).to(self.device)
        y_t = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        
        # Train model
        self.model.train()
        for epoch in range(self.epochs):
            # Mini-batch training
            n_samples = len(X_price)
            indices = np.random.permutation(n_samples)
            
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                
                # Forward pass
                self.optimizer.zero_grad()
                y_pred = self.model(X_price_t[batch_idx], X_meta_t[batch_idx])
                loss = self.criterion(y_pred, y_t[batch_idx])
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if self.verbose > 0 and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/n_batches:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the PyTorch model.
        
        Args:
            X: Combined features (N, lookback*price_features + meta_len)
        
        Returns:
            Predictions (N,)
        """
        # Split X back into price and meta
        X_price, X_meta = self._split_X(X)
        
        # Convert to torch tensors
        X_price_t = torch.FloatTensor(X_price).to(self.device)
        X_meta_t = torch.FloatTensor(X_meta).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_price_t, X_meta_t)
        
        return y_pred.cpu().numpy().flatten()
    
    def _split_X(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split combined feature array into price and meta components.
        
        Args:
            X: Combined features (N, lookback*price_features + meta_len)
        
        Returns:
            (X_price, X_meta) tuple
        """
        price_size = self.lookback * self.price_features
        
        X_price = X[:, :price_size].reshape(-1, self.lookback, self.price_features)
        X_meta = X[:, price_size:]
        
        return X_price, X_meta


class SklearnLightGBMWrapper(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible wrapper for LightGBM models.
    
    This wrapper ensures LightGBM models work consistently with MAPIE
    and handle the same input format as the TensorFlow wrapper.
    
    Args:
        model: LightGBM model (LGBMRegressor or Booster)
        lookback: Price window length
        price_features: Number of price features (usually 5 for OHLCV)
    """
    
    def __init__(
        self,
        model: Union[lgb.LGBMRegressor, lgb.Booster],
        lookback: int = 200,
        price_features: int = 5
    ):
        self.model = model
        self.lookback = lookback
        self.price_features = price_features
        self.is_booster = isinstance(model, lgb.Booster)
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SklearnLightGBMWrapper':
        """
        Fit the LightGBM model.
        
        Args:
            X: Combined features (N, lookback*price_features + meta_len)
            y: Target values (N,)
        
        Returns:
            self
        """
        if not self.is_booster:
            # LGBMRegressor already has fit method
            self.model.fit(X, y, **kwargs)
        else:
            # For Booster, we need to create a dataset and train
            train_data = lgb.Dataset(X, label=y)
            # Note: Booster training would need additional params
            # This is a simplified version - you may need to adjust
            raise NotImplementedError(
                "Booster fitting not implemented. Use LGBMRegressor instead."
            )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the LightGBM model.
        
        Args:
            X: Combined features (N, lookback*price_features + meta_len)
        
        Returns:
            Predictions (N,)
        """
        y_pred = self.model.predict(X)
        return y_pred.flatten() if y_pred.ndim > 1 else y_pred


class MapiePredictor:
    """
    MAPIE-based predictor that returns confidence intervals as a clean DataFrame.
    
    This class wraps MAPIE's conformal prediction to provide predictions at
    multiple confidence levels in an easy-to-use format.
    
    Args:
        model: Wrapped model (SklearnTensorFlowWrapper or SklearnLightGBMWrapper)
        method: MAPIE method ('plus', 'base', 'minmax', 'naive')
        cv: Cross-validation strategy (int or sklearn splitter)
        n_jobs: Number of parallel jobs (-1 for all cores)
        random_state: Random seed
    
    Example:
        >>> wrapped = SklearnTensorFlowWrapper(tf_model)
        >>> mapie_pred = MapiePredictor(wrapped, method='plus', cv=5)
        >>> mapie_pred.fit(X_price, X_meta, y)
        >>> df = mapie_pred.predict_intervals(X_price_test, X_meta_test, alphas=[0.05, 0.10])
    """
    
    def __init__(
        self,
        model: Union[SklearnPyTorchWrapper, SklearnLightGBMWrapper],
        method: str = 'plus',
        cv: Union[int, object] = 5,
        n_jobs: int = -1,
        random_state: Optional[int] = None
    ):
        self.wrapped_model = model
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Create MAPIE regressor
        self.mapie = MapieRegressor(
            estimator=model,
            method=method,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state
        )
        
        self.is_fitted_ = False
    
    def fit(
        self,
        X_price: np.ndarray,
        X_meta: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> 'MapiePredictor':
        """
        Fit the MAPIE predictor on training data.
        
        Args:
            X_price: Price data (N, lookback, price_features)
            X_meta: Meta features (N, meta_len)
            y: Target values (N,)
            **kwargs: Additional arguments passed to fit
        
        Returns:
            self
        """
        # Combine X_price and X_meta
        X_combined = self._combine_inputs(X_price, X_meta)
        
        # Fit MAPIE
        self.mapie.fit(X_combined, y, **kwargs)
        self.is_fitted_ = True
        
        return self
    
    def predict_intervals(
        self,
        X_price: np.ndarray,
        X_meta: np.ndarray,
        alphas: List[float] = [0.05, 0.10, 0.20],
        include_point_pred: bool = True
    ) -> pd.DataFrame:
        """
        Predict with confidence intervals at multiple alpha levels.
        
        Args:
            X_price: Price data (N, lookback, price_features)
            X_meta: Meta features (N, meta_len)
            alphas: List of alpha values (e.g., 0.05 for 95% confidence)
            include_point_pred: Include point prediction in output
        
        Returns:
            DataFrame with columns:
                - 'point_pred' (optional): point prediction
                - 'lower_X': lower bound at (1-X)% confidence
                - 'upper_X': upper bound at (1-X)% confidence
                - 'width_X': interval width at (1-X)% confidence
            
            Where X is the confidence level (e.g., 95, 90, 80)
        
        Example output columns:
            ['point_pred', 'lower_95', 'upper_95', 'width_95', 
             'lower_90', 'upper_90', 'width_90', ...]
        """
        if not self.is_fitted_:
            raise RuntimeError("MapiePredictor must be fitted before prediction")
        
        # Combine inputs
        X_combined = self._combine_inputs(X_price, X_meta)
        
        # Initialize results dictionary
        n_samples = len(X_price)
        results = {}
        
        # Get predictions for each alpha
        for alpha in alphas:
            y_pred, y_pis = self.mapie.predict(X_combined, alpha=alpha)
            
            # Calculate confidence level
            conf_level = int((1 - alpha) * 100)
            
            # Store results
            if include_point_pred and alpha == alphas[0]:
                # Only add point prediction once
                results['point_pred'] = y_pred.flatten()
            
            results[f'lower_{conf_level}'] = y_pis[:, 0, 0]
            results[f'upper_{conf_level}'] = y_pis[:, 1, 0]
            results[f'width_{conf_level}'] = y_pis[:, 1, 0] - y_pis[:, 0, 0]
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        return df
    
    def predict_single_interval(
        self,
        X_price: np.ndarray,
        X_meta: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with a single confidence interval.
        
        Args:
            X_price: Price data (N, lookback, price_features)
            X_meta: Meta features (N, meta_len)
            alpha: Alpha value (e.g., 0.05 for 95% confidence)
        
        Returns:
            (y_pred, y_pis) tuple where:
                - y_pred: Point predictions (N,)
                - y_pis: Prediction intervals (N, 2, 1)
                         [:, 0, 0] = lower bound
                         [:, 1, 0] = upper bound
        """
        if not self.is_fitted_:
            raise RuntimeError("MapiePredictor must be fitted before prediction")
        
        # Combine inputs
        X_combined = self._combine_inputs(X_price, X_meta)
        
        # Predict
        y_pred, y_pis = self.mapie.predict(X_combined, alpha=alpha)
        
        return y_pred.flatten(), y_pis
    
    def predict_point(
        self,
        X_price: np.ndarray,
        X_meta: np.ndarray
    ) -> np.ndarray:
        """
        Get point predictions only (no intervals).
        
        Args:
            X_price: Price data (N, lookback, price_features)
            X_meta: Meta features (N, meta_len)
        
        Returns:
            Point predictions (N,)
        """
        if not self.is_fitted_:
            raise RuntimeError("MapiePredictor must be fitted before prediction")
        
        # Combine inputs
        X_combined = self._combine_inputs(X_price, X_meta)
        
        # Predict (MAPIE predict without alpha returns only point predictions)
        y_pred = self.mapie.predict(X_combined)
        
        return y_pred.flatten()
    
    def _combine_inputs(
        self,
        X_price: np.ndarray,
        X_meta: np.ndarray
    ) -> np.ndarray:
        """
        Combine price and meta features into single array.
        
        Args:
            X_price: Price data (N, lookback, price_features)
            X_meta: Meta features (N, meta_len)
        
        Returns:
            Combined features (N, lookback*price_features + meta_len)
        """
        # Flatten price window
        X_price_flat = X_price.reshape(len(X_price), -1)
        
        # Concatenate with meta
        X_combined = np.concatenate([X_price_flat, X_meta], axis=1)
        
        return X_combined


def create_mapie_predictor_from_model(
    model: Union[nn.Module, lgb.LGBMRegressor, lgb.Booster],
    model_type: str = 'transformer',
    lookback: int = 200,
    price_features: int = 5,
    meta_len: int = 8,
    method: str = 'plus',
    cv: Union[int, object] = 5,
    device: str = 'cuda',
    **mapie_kwargs
) -> MapiePredictor:
    """
    Convenience function to create a MapiePredictor from a trained model.
    
    Args:
        model: Trained model (PyTorch nn.Module or LightGBM)
        model_type: Model architecture type
        lookback: Price window length
        price_features: Number of price features
        meta_len: Number of meta features
        method: MAPIE method
        cv: Cross-validation strategy
        device: 'cuda' or 'cpu'
        **mapie_kwargs: Additional arguments for MapiePredictor
    
    Returns:
        MapiePredictor instance ready for fitting
    
    Example:
        >>> mapie_pred = create_mapie_predictor_from_model(
        ...     my_pytorch_model,
        ...     model_type='transformer',
        ...     method='plus',
        ...     cv=5
        ... )
        >>> mapie_pred.fit(X_price, X_meta, y)
    """
    # Determine model type and wrap appropriately
    if isinstance(model, nn.Module):
        wrapped = SklearnPyTorchWrapper(
            model,
            model_type=model_type,
            lookback=lookback,
            price_features=price_features,
            meta_len=meta_len,
            device=device
        )
    elif isinstance(model, (lgb.LGBMRegressor, lgb.Booster)):
        wrapped = SklearnLightGBMWrapper(
            model,
            lookback=lookback,
            price_features=price_features
        )
    else:
        raise ValueError(
            f"Unsupported model type: {type(model)}. "
            "Expected torch.nn.Module, lgb.LGBMRegressor, or lgb.Booster"
        )
    
    # Create MapiePredictor
    mapie_pred = MapiePredictor(
        wrapped,
        method=method,
        cv=cv,
        **mapie_kwargs
    )
    
    return mapie_pred
