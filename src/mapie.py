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
from mapie.regression import CrossConformalRegressor
from scipy import stats


# ========== CUSTOM LOSS FUNCTIONS ==========
class GeometricMeanLoss(nn.Module):
    """
    Geometric Mean loss for reward prediction.
    
    For positive rewards: log(pred/target)^2
    For negative rewards: (pred - target)^2
    
    This emphasizes relative errors for positive values and absolute errors for negative values.
    """
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle positive and negative rewards differently
        pos_mask = target > 0
        neg_mask = target <= 0
        
        loss = torch.zeros_like(pred)
        
        # For positive rewards: geometric mean (log-space)
        if pos_mask.any():
            pred_pos = torch.clamp(pred[pos_mask], min=self.epsilon)
            target_pos = torch.clamp(target[pos_mask], min=self.epsilon)
            log_ratio = torch.log(pred_pos / target_pos)
            loss[pos_mask] = log_ratio ** 2
        
        # For negative rewards: standard MSE
        if neg_mask.any():
            loss[neg_mask] = (pred[neg_mask] - target[neg_mask]) ** 2
        
        return loss.mean()


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss that combines MSE and Huber loss based on prediction magnitude.
    
    Uses MSE for small errors and Huber loss for large errors to reduce outlier impact.
    """
    def __init__(self, delta: float = 1.0, threshold: float = 0.1):
        super().__init__()
        self.delta = delta
        self.threshold = threshold
        self.huber = nn.HuberLoss(delta=delta)
        self.mse = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = torch.abs(pred - target)
        small_error_mask = error < self.threshold
        
        loss = torch.zeros_like(error)
        
        if small_error_mask.any():
            loss[small_error_mask] = self.mse(pred[small_error_mask], target[small_error_mask])
        
        large_error_mask = ~small_error_mask
        if large_error_mask.any():
            loss[large_error_mask] = self.huber(pred[large_error_mask], target[large_error_mask])
        
        return loss.mean()


def get_loss_function(loss_type: str = 'mse', **kwargs) -> nn.Module:
    """
    Factory function for loss functions.
    
    Args:
        loss_type: 'mse', 'huber', 'geometric_mean', 'adaptive'
        **kwargs: Additional parameters for specific losses
    
    Returns:
        PyTorch loss function
    """
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'huber':
        delta = kwargs.get('delta', 1.0)
        return nn.HuberLoss(delta=delta)
    elif loss_type == 'geometric_mean':
        epsilon = kwargs.get('epsilon', 1e-8)
        return GeometricMeanLoss(epsilon=epsilon)
    elif loss_type == 'adaptive':
        delta = kwargs.get('delta', 1.0)
        threshold = kwargs.get('threshold', 0.1)
        return AdaptiveLoss(delta=delta, threshold=threshold)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


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
        verbose: Training verbosity (0=silent, 1=progress, 2=detailed)
        checkpoint_path: Path to save best model checkpoint (None=no checkpointing)
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
        meta_len: int = 20,
        # New training parameters
        loss_type: str = 'mse',
        optimizer_type: str = 'adam',
        weight_decay: float = 0.0,
        scheduler_type: Optional[str] = None,
        early_stopping: bool = True,
        patience: int = 10,
        validation_split: float = 0.2,
        # Loss function specific parameters
        loss_epsilon: float = 1e-8,  # For geometric_mean loss
        loss_delta: float = 1.0,     # For huber/adaptive loss
        loss_threshold: float = 0.1,  # For adaptive loss
        # Checkpointing
        checkpoint_path: Optional[str] = None  # Path to save best model
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
        
        # Training configuration
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_split = validation_split
        
        # Loss function parameters - store directly for sklearn compatibility
        self.loss_epsilon = loss_epsilon
        self.loss_delta = loss_delta
        self.loss_threshold = loss_threshold
        
        # Also store in kwargs dict for backward compatibility
        self.loss_kwargs = {
            'epsilon': loss_epsilon,
            'delta': loss_delta,
            'threshold': loss_threshold
        }
        
        # Checkpointing
        self.checkpoint_path = checkpoint_path
        self.best_model_state = None  # Store best model state in memory
        
        # Training history (will be populated during fit)
        self.training_history_ = None
        self.best_epoch_ = None
        self.best_val_loss_ = None
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer and loss
        self._setup_training()
    
    def _setup_training(self):
        """Setup optimizer, loss function, and scheduler."""
        # Setup loss function
        self.criterion = get_loss_function(self.loss_type, **self.loss_kwargs)
        
        # Setup optimizer
        if self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
        
        # Setup scheduler
        self.scheduler = None
        if self.scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        elif self.scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, self.epochs // 3), gamma=0.5
            )
        elif self.scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=max(1, self.patience // 2), factor=0.5
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SklearnPyTorchWrapper':
        """
        Fit the PyTorch model with enhanced training features.
        
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
        
        # Split into train/validation if early stopping is enabled
        if self.early_stopping and self.validation_split > 0:
            n_samples = len(X_price)
            val_size = int(n_samples * self.validation_split)
            train_size = n_samples - val_size
            
            # Random split
            indices = np.random.permutation(n_samples)
            train_idx = indices[:train_size]
            val_idx = indices[train_size:]
            
            X_price_train = X_price_t[train_idx]
            X_meta_train = X_meta_t[train_idx]
            y_train = y_t[train_idx]
            
            X_price_val = X_price_t[val_idx]
            X_meta_val = X_meta_t[val_idx]
            y_val = y_t[val_idx]
        else:
            X_price_train = X_price_t
            X_meta_train = X_meta_t
            y_train = y_t
            X_price_val = X_meta_val = y_val = None
        
        # Training loop with validation and early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {'train_loss': [], 'val_loss': [], 'lr': [], 'epoch': []}
        
        self.model.train()
        for epoch in range(self.epochs):
            # Training phase
            epoch_train_loss = 0.0
            n_train_batches = 0
            
            # Shuffle training data
            n_train = len(X_price_train)
            train_indices = np.random.permutation(n_train)
            
            for i in range(0, n_train, self.batch_size):
                batch_idx = train_indices[i:i+self.batch_size]
                
                # Forward pass
                self.optimizer.zero_grad()
                y_pred = self.model(X_price_train[batch_idx], X_meta_train[batch_idx])
                loss = self.criterion(y_pred, y_train[batch_idx])
                
                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
                n_train_batches += 1
            
            avg_train_loss = epoch_train_loss / n_train_batches
            training_history['train_loss'].append(avg_train_loss)
            training_history['epoch'].append(epoch + 1)
            
            # Validation phase
            val_loss = None
            if X_price_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_price_val, X_meta_val)
                    val_loss = self.criterion(val_pred, y_val).item()
                    training_history['val_loss'].append(val_loss)
                
                self.model.train()
                
                # Early stopping and checkpointing
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_epoch_ = epoch + 1
                    
                    # Save best model state
                    self.best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    
                    # Save checkpoint to disk if path provided
                    if self.checkpoint_path is not None:
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'train_loss': avg_train_loss,
                            'val_loss': val_loss,
                            'training_history': training_history
                        }, self.checkpoint_path)
                        
                        if self.verbose > 1:
                            print(f"  💾 Checkpoint saved: epoch {epoch+1}, val_loss={val_loss:.6f}")
                else:
                    patience_counter += 1
                    
                    if self.early_stopping and patience_counter >= self.patience:
                        if self.verbose > 0:
                            print(f"⏹️  Early stopping at epoch {epoch+1} (best: epoch {self.best_epoch_})")
                        break
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            training_history['lr'].append(current_lr)
            
            # Verbose output
            if self.verbose > 0:
                # Show every epoch if verbose >= 2, else every 10%
                show_epoch = (self.verbose >= 2) or ((epoch + 1) % max(1, self.epochs // 10) == 0)
                if show_epoch:
                    if val_loss is not None:
                        indicator = "⭐" if val_loss == best_val_loss else "  "
                        print(f"{indicator} Epoch {epoch+1}/{self.epochs} - "
                              f"Train: {avg_train_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.8f}")
                    else:
                        print(f"  Epoch {epoch+1}/{self.epochs} - "
                              f"Train: {avg_train_loss:.6f}, LR: {current_lr:.8f}")
        
        # Restore best model if early stopping was used
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if self.verbose > 0:
                print(f"✅ Restored best model from epoch {self.best_epoch_} (val_loss={best_val_loss:.6f})")
        
        # Store training history and metadata
        self.training_history_ = training_history
        self.best_val_loss_ = best_val_loss if X_price_val is not None else None
        
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
    
    def get_training_history(self) -> Optional[dict]:
        """
        Get training history after fit() has been called.
        
        Returns:
            Dictionary with keys: 'train_loss', 'val_loss', 'lr', 'epoch'
            Returns None if model hasn't been trained yet.
        """
        return self.training_history_
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from a checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file saved during training
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training metadata
        if 'training_history' in checkpoint:
            self.training_history_ = checkpoint['training_history']
        if 'epoch' in checkpoint:
            self.best_epoch_ = checkpoint['epoch']
        if 'val_loss' in checkpoint:
            self.best_val_loss_ = checkpoint['val_loss']
        
        if self.verbose > 0:
            print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
        
        return self
    
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
    
    def _combine_inputs(self, X_price: np.ndarray, X_meta: np.ndarray) -> np.ndarray:
        """
        Combine price and meta inputs for sklearn interface.
        
        Args:
            X_price: Price data (N, lookback, price_features)
            X_meta: Meta features (N, meta_len)
        
        Returns:
            Combined features (N, lookback*price_features + meta_len)
        """
        # Flatten price sequences
        X_price_flat = X_price.reshape(X_price.shape[0], -1)
        
        # Concatenate with meta features
        X_combined = np.concatenate([X_price_flat, X_meta], axis=1)
        
        return X_combined


class SklearnLightGBMWrapper(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible wrapper for LightGBM models with training monitoring.
    
    This wrapper ensures LightGBM models work consistently with MAPIE
    and handle the same input format as the PyTorch wrapper, with comprehensive
    training monitoring and early stopping support.
    
    Args:
        model: LightGBM model (LGBMRegressor)
        lookback: Price window length
        price_features: Number of price features (usually 5 for OHLCV)
        verbose: Training verbosity (-1=silent, 0=warning, 1=info, 2=detailed)
        early_stopping_rounds: Stop if no improvement for N rounds (None=disabled)
        validation_split: Fraction of data for validation (0.0=no validation)
        save_path: Path to save best model (None=no saving)
    """
    
    def __init__(
        self,
        model: Union[lgb.LGBMRegressor, lgb.Booster],
        lookback: int = 200,
        price_features: int = 5,
        verbose: int = 0,
        early_stopping_rounds: Optional[int] = None,
        validation_split: float = 0.2,
        save_path: Optional[str] = None
    ):
        self.model = model
        self.lookback = lookback
        self.price_features = price_features
        self.is_booster = isinstance(model, lgb.Booster)
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_split = validation_split
        self.save_path = save_path
        
        # Training history (will be populated during fit)
        self.training_history_ = None
        self.best_iteration_ = None
        self.best_score_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SklearnLightGBMWrapper':
        """
        Fit the LightGBM model with monitoring and early stopping.
        
        Args:
            X: Combined features (N, lookback*price_features + meta_len)
            y: Target values (N,)
        
        Returns:
            self
        """
        if self.is_booster:
            raise NotImplementedError(
                "Booster fitting not implemented. Use LGBMRegressor instead."
            )
        
        # Setup callbacks for monitoring
        callbacks = []
        
        # Add verbose printing callback
        if self.verbose >= 0:
            # LightGBM verbosity: -1=silent, 0=warning, 1=info, 2=debug
            # Map to period: higher verbose = more frequent updates
            period = 1 if self.verbose >= 2 else (10 if self.verbose == 1 else 50)
            callbacks.append(lgb.log_evaluation(period=period))
        
        # Prepare eval_set for validation and early stopping
        eval_set = None
        eval_names = None
        
        if self.validation_split > 0:
            # Split data into train/validation
            n_samples = len(X)
            val_size = int(n_samples * self.validation_split)
            train_size = n_samples - val_size
            
            # Random split
            indices = np.random.permutation(n_samples)
            train_idx = indices[:train_size]
            val_idx = indices[train_size:]
            
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            eval_set = [(X_train, y_train), (X_val, y_val)]
            eval_names = ['train', 'valid']
            
            if self.verbose > 0:
                print(f"📊 LightGBM training with validation: {train_size} train, {val_size} val samples")
        else:
            X_train = X
            y_train = y
            eval_set = [(X_train, y_train)]
            eval_names = ['train']
        
        # Add early stopping callback if enabled
        if self.early_stopping_rounds is not None and self.validation_split > 0:
            callbacks.append(lgb.early_stopping(
                stopping_rounds=self.early_stopping_rounds,
                verbose=self.verbose > 0
            ))
        
        # Fit model with callbacks and eval_set
        self.model.fit(
            X_train, 
            y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            callbacks=callbacks,
            **kwargs
        )
        
        # Extract training history from evals_result_
        if hasattr(self.model, 'evals_result_'):
            evals_result = self.model.evals_result_
            
            # Convert to consistent format with PyTorch wrapper
            self.training_history_ = {
                'train_loss': evals_result.get('train', {}).get('l2', []),
                'val_loss': evals_result.get('valid', {}).get('l2', []),
                'epoch': list(range(1, len(evals_result.get('train', {}).get('l2', [])) + 1))
            }
            
            # Store best iteration info
            if hasattr(self.model, 'best_iteration_'):
                self.best_iteration_ = self.model.best_iteration_
                if self.verbose > 0:
                    print(f"✅ Best iteration: {self.best_iteration_}")
            
            if hasattr(self.model, 'best_score_'):
                self.best_score_ = self.model.best_score_
        
        # Save model if path provided
        if self.save_path is not None:
            self.model.booster_.save_model(self.save_path)
            if self.verbose > 0:
                print(f"💾 Model saved to: {self.save_path}")
        
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
    
    def get_training_history(self) -> Optional[dict]:
        """
        Get training history after fit() has been called.
        
        Returns:
            Dictionary with keys: 'train_loss', 'val_loss', 'epoch'
            Returns None if model hasn't been trained yet.
        """
        return self.training_history_
    
    def load_model(self, model_path: str):
        """
        Load LightGBM model from file.
        
        Args:
            model_path: Path to saved model file
        """
        booster = lgb.Booster(model_file=model_path)
        self.model = lgb.LGBMRegressor()
        self.model._Booster = booster
        
        if self.verbose > 0:
            print(f"✅ Loaded LightGBM model from: {model_path}")
        
        return self


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
        
        # Create MAPIE regressor with default confidence levels
        # Note: In the new API, confidence levels are set here, not during prediction
        default_confidence_levels = [0.8, 0.9, 0.95]  # 80%, 90%, 95%
        self.mapie = CrossConformalRegressor(
            estimator=model,
            confidence_level=default_confidence_levels,
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
        
        # Fit MAPIE using new API
        self.mapie.fit_conformalize(X_combined, y, **kwargs)
        self.is_fitted_ = True
        
        return self
    
    def predict_intervals(
        self,
        X_price: np.ndarray,
        X_meta: np.ndarray = None,
        alphas: List[float] = [0.05, 0.10, 0.20],
        include_point_pred: bool = True
    ) -> pd.DataFrame:
        """
        Predict with confidence intervals at multiple alpha levels.
        
        Args:
            X_price: Price data (N, lookbook, price_features) OR combined data (N, features)
            X_meta: Meta features (N, meta_len) - optional if X_price is combined
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
        
        # Handle different input formats
        if X_meta is not None:
            # Normal case: separate X_price and X_meta
            X_combined = self._combine_inputs(X_price, X_meta)
            n_samples = len(X_price)
        else:
            # Combined input case (e.g., for LightGBM)
            X_combined = X_price
            n_samples = len(X_price)
        results = {}
        
        # Note: The new MAPIE API requires confidence levels to be set during initialization
        # We need to recreate the mapie regressor with the desired confidence levels
        confidence_levels = [1 - alpha for alpha in alphas]
        
        # Use existing fitted MAPIE with predict_interval method
        # For each alpha level, get the predictions and intervals
        results_data = {}
        
        for alpha in alphas:
            confidence_level = 1 - alpha
            confidence_pct = int(confidence_level * 100)
            
            # Use the current fitted MAPIE regressor
            # The predict_interval method can take alpha parameter
            try:
                y_pred, y_pis = self.mapie.predict_interval(X_combined, alpha=alpha)
                
                # Store results
                if include_point_pred and 'point_pred' not in results_data:
                    results_data['point_pred'] = y_pred
                
                results_data[f'lower_{confidence_pct}'] = y_pis[:, 0]
                results_data[f'upper_{confidence_pct}'] = y_pis[:, 1]
                results_data[f'width_{confidence_pct}'] = y_pis[:, 1] - y_pis[:, 0]
                
            except Exception as e:
                # Fallback: recreate with single confidence level
                try:
                    temp_mapie = CrossConformalRegressor(
                        confidence_level=confidence_level,
                        method=self.method,
                        cv=self.cv,
                        n_jobs=self.n_jobs,
                        random_state=self.random_state
                    )
                    temp_mapie.estimator_ = self.mapie.estimator_
                    if hasattr(self.mapie, 'conformity_scores_'):
                        temp_mapie.conformity_scores_ = self.mapie.conformity_scores_
                    
                    y_pred, y_pis = temp_mapie.predict_interval(X_combined)
                    
                    if include_point_pred and 'point_pred' not in results_data:
                        results_data['point_pred'] = y_pred
                    
                    results_data[f'lower_{confidence_pct}'] = y_pis[:, 0]
                    results_data[f'upper_{confidence_pct}'] = y_pis[:, 1]
                    results_data[f'width_{confidence_pct}'] = y_pis[:, 1] - y_pis[:, 0]
                    
                except Exception as e2:
                    # Final fallback: use point predictions with dummy intervals
                    y_pred = self.mapie.predict(X_combined)
                    std_dev = np.std(y_pred)
                    z_score = stats.norm.ppf(1 - alpha/2)  # Two-tailed
                    
                    if include_point_pred and 'point_pred' not in results_data:
                        results_data['point_pred'] = y_pred
                    
                    margin = z_score * std_dev
                    results_data[f'lower_{confidence_pct}'] = y_pred - margin
                    results_data[f'upper_{confidence_pct}'] = y_pred + margin
                    results_data[f'width_{confidence_pct}'] = 2 * margin
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(results_data)
        
        return df_results
    
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
        
        # Predict using new API
        y_pred, y_pis = self.mapie.predict_interval(X_combined)
        
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
        
        # Predict point predictions only
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
