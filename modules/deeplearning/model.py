"""
Temporal Fusion Transformer (TFT) Model Configuration & Training.

This module implements 3 phases:
- Phase 1: Vanilla TFT (MVP) with QuantileLoss
- Phase 2: Optuna hyperparameter tuning
- Phase 3: Hybrid LSTM + TFT Architecture (Advanced)
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
# Import from lightning.pytorch to match pytorch-forecasting's imports
# PyTorch Lightning 2.x uses lightning.pytorch namespace
try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        LearningRateMonitor,
    )
except ImportError:
    # Fallback to pytorch_lightning (for older versions)
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        LearningRateMonitor,
    )
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss, MAE, RMSE
from pytorch_forecasting.metrics.base_metrics import Metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore, Style

from modules.config import (
    DEEP_MAX_ENCODER_LENGTH,
    DEEP_MAX_PREDICTION_LENGTH,
    DEEP_BATCH_SIZE,
    DEEP_TARGET_COL,
    DEEP_TARGET_COL_CLASSIFICATION,
    DEEP_USE_TRIPLE_BARRIER,
)
from modules.common.utils import color_text

# Phase 2: Optuna (optional import)
try:
    import optuna
    from optuna.integration import PyTorchLightningPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    PyTorchLightningPruningCallback = None


# ============================================================================
# Phase 1: Vanilla TFT (MVP)
# ============================================================================

def create_vanilla_tft(
    training_dataset: TimeSeriesDataSet,
    hidden_size: int = 16,
    attention_head_size: int = 4,
    dropout: float = 0.1,
    learning_rate: float = 0.03,
    quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
    reduce_on_plateau_patience: int = 4,
    task_type: str = "regression",
    **kwargs,
) -> TemporalFusionTransformer:
    """
    Create a vanilla Temporal Fusion Transformer model (Phase 1: MVP).
    
    Args:
        training_dataset: TimeSeriesDataSet for model initialization
        hidden_size: Hidden size of the model (default: 16)
        attention_head_size: Size of attention heads (default: 4)
        dropout: Dropout rate (default: 0.1)
        learning_rate: Learning rate (default: 0.03)
        quantiles: List of quantiles for QuantileLoss (default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
        reduce_on_plateau_patience: Patience for learning rate reduction (default: 4)
        task_type: 'regression' or 'classification' (default: 'regression')
        **kwargs: Additional arguments passed to TemporalFusionTransformer
    
    Returns:
        TemporalFusionTransformer instance
    """
    # Determine loss function based on task type
    if task_type == "classification":
        # For classification, use CrossEntropyLoss
        # Note: TFT typically uses regression, but we can adapt it
        loss = None  # Will use default
    else:
        # For regression, use QuantileLoss to generate confidence intervals
        loss = QuantileLoss(quantiles=quantiles)
    
    # Create model from dataset
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        reduce_on_plateau_patience=reduce_on_plateau_patience,
        loss=loss,
        **kwargs,
    )
    
    # Verify model is a LightningModule for compatibility
    if not isinstance(model, pl.LightningModule):
        print(
            color_text(
                f"Warning: Model type is {type(model)}, not directly recognized as LightningModule. "
                "This may cause compatibility issues with PyTorch Lightning Trainer.",
                Fore.YELLOW,
            )
        )
        # Try to verify MRO (Method Resolution Order)
        if not hasattr(model, 'training_step') or not hasattr(model, 'configure_optimizers'):
            raise TypeError(
                f"TemporalFusionTransformer instance does not have required LightningModule methods. "
                f"Please check pytorch-forecasting and pytorch-lightning versions compatibility."
            )
    
    print(
        color_text(
            f"Created Vanilla TFT (Phase 1): hidden_size={hidden_size}, "
            f"attention_head_size={attention_head_size}, dropout={dropout}, "
            f"learning_rate={learning_rate}, quantiles={quantiles}",
            Fore.GREEN,
        )
    )
    
    return model


def create_training_callbacks(
    checkpoint_dir: Union[str, Path] = "artifacts/deep/checkpoints",
    monitor: str = "val_loss",
    mode: str = "min",
    patience: int = 10,
    save_top_k: int = 3,
    verbose: bool = True,
) -> List[pl.Callback]:
    """
    Create standard training callbacks for TFT (Phase 1).
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        monitor: Metric to monitor (default: 'val_loss')
        mode: 'min' or 'max' (default: 'min')
        patience: Early stopping patience (default: 10)
        save_top_k: Number of top checkpoints to save (default: 3)
        verbose: Whether to print callback information
    
    Returns:
        List of PyTorch Lightning callbacks
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            verbose=verbose,
        ),
        
        # Model checkpointing
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            verbose=verbose,
        ),
        
        # Learning rate monitoring
        LearningRateMonitor(logging_interval="step"),
    ]
    
    if verbose:
        print(
            color_text(
                f"Created training callbacks: EarlyStopping (patience={patience}), "
                f"ModelCheckpoint (save_top_k={save_top_k}), LearningRateMonitor",
                Fore.GREEN,
            )
        )
    
    return callbacks


# ============================================================================
# Phase 2: Optuna Hyperparameter Tuning
# ============================================================================

def create_optuna_study(
    direction: str = "minimize",
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    sampler: Optional[Any] = None,
) -> Optional[Any]:
    """
    Create an Optuna study for hyperparameter tuning (Phase 2).
    
    Args:
        direction: 'minimize' or 'maximize' (default: 'minimize')
        study_name: Name of the study (optional)
        storage: Storage URL for distributed optimization (optional)
        sampler: Optuna sampler (optional, defaults to TPESampler)
    
    Returns:
        Optuna study object, or None if Optuna is not available
    """
    if not OPTUNA_AVAILABLE:
        print(
            color_text(
                "Optuna not available. Install with: pip install optuna",
                Fore.YELLOW,
            )
        )
        return None
    
    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=42)
    
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        sampler=sampler,
    )
    
    print(
        color_text(
            f"Created Optuna study: {study_name or 'default'} (direction={direction})",
            Fore.GREEN,
        )
    )
    
    return study


def suggest_tft_hyperparameters(trial: Any) -> Dict[str, Any]:
    """
    Suggest hyperparameters for TFT using Optuna trial (Phase 2).
    
    Args:
        trial: Optuna trial object
    
    Returns:
        Dictionary of suggested hyperparameters
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not available. Install with: pip install optuna")
    
    return {
        "hidden_size": trial.suggest_int("hidden_size", 8, 64, step=8),
        "attention_head_size": trial.suggest_int("attention_head_size", 1, 8),
        "dropout": trial.suggest_float("dropout", 0.05, 0.3, step=0.05),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
        "reduce_on_plateau_patience": trial.suggest_int(
            "reduce_on_plateau_patience", 2, 8
        ),
    }


def create_optuna_callback(
    trial: Any,
    monitor: str = "val_loss",
    mode: str = "min",
) -> Optional[Any]:
    """
    Create Optuna pruning callback for early stopping trials (Phase 2).
    
    Args:
        trial: Optuna trial object
        monitor: Metric to monitor (default: 'val_loss')
        mode: 'min' or 'max' (default: 'min')
    
    Returns:
        PyTorchLightningPruningCallback, or None if Optuna is not available
    """
    if not OPTUNA_AVAILABLE or PyTorchLightningPruningCallback is None:
        return None
    
    return PyTorchLightningPruningCallback(trial, monitor=monitor, mode=mode)


def optimize_tft_hyperparameters(
    training_dataset: TimeSeriesDataSet,
    val_dataset: TimeSeriesDataSet,
    datamodule: Any,
    n_trials: int = 20,
    timeout: Optional[int] = None,
    n_jobs: int = 1,
    study_name: Optional[str] = None,
    checkpoint_dir: Union[str, Path] = "artifacts/deep/optuna",
    max_epochs: int = 50,
    gpus: Optional[int] = None,
    **kwargs,
) -> Tuple[Dict[str, Any], Any]:
    """
    Optimize TFT hyperparameters using Optuna (Phase 2).
    
    Args:
        training_dataset: Training TimeSeriesDataSet
        val_dataset: Validation TimeSeriesDataSet
        datamodule: Lightning DataModule
        n_trials: Number of optimization trials (default: 20)
        timeout: Timeout in seconds (optional)
        n_jobs: Number of parallel jobs (default: 1)
        study_name: Name of the study (optional)
        checkpoint_dir: Directory to save optuna results
        max_epochs: Maximum epochs per trial (default: 50)
        gpus: Number of GPUs to use (optional)
        **kwargs: Additional arguments for model creation
    
    Returns:
        Tuple of (best_params, study)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not available. Install with: pip install optuna")
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def objective(trial: Any) -> float:
        """Objective function for Optuna optimization."""
        # Suggest hyperparameters
        params = suggest_tft_hyperparameters(trial)
        
        # Create model with suggested hyperparameters
        model = create_vanilla_tft(
            training_dataset,
            **params,
            **kwargs,
        )
        
        # Create callbacks including Optuna pruning
        callbacks = create_training_callbacks(
            checkpoint_dir=checkpoint_dir / f"trial_{trial.number}",
            verbose=False,
        )
        callbacks.append(create_optuna_callback(trial))
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            gpus=gpus,
            enable_progress_bar=False,  # Reduce output during optimization
            logger=False,  # Disable logging during optimization
        )
        
        # Train model
        trainer.fit(model, datamodule)
        
        # Return best validation loss
        return trainer.callback_metrics.get("val_loss", float("inf")).item()
    
    # Create study
    study = create_optuna_study(study_name=study_name)
    
    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
    )
    
    # Save study
    study_path = checkpoint_dir / "optuna_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    
    print(
        color_text(
            f"Optuna optimization completed. Best params: {study.best_params}",
            Fore.GREEN,
        )
    )
    
    return study.best_params, study


# ============================================================================
# Phase 3: Hybrid LSTM + TFT Architecture (Advanced)
# ============================================================================

class HybridLSTMTFT(nn.Module):
    """
    Hybrid LSTM + TFT Architecture (Phase 3: Advanced).
    
    Dual Branch:
    - LSTM branch: Process raw price/volume series
    - TFT branch: Process complex features (static + known future)
    
    Fusion: Gated fusion (GLU) of latent vectors
    Multi-task Head:
    - Task 1: Direction (Classification/Softmax)
    - Task 2: Magnitude (Regression/QuantileLoss)
    """
    
    def __init__(
        self,
        tft_model: TemporalFusionTransformer,
        lstm_input_size: int = 5,  # OHLCV
        lstm_hidden_size: int = 32,
        lstm_num_layers: int = 2,
        fusion_size: int = 64,
        num_classes: int = 3,  # UP, NEUTRAL, DOWN
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        lambda_class: float = 1.0,
        lambda_reg: float = 1.0,
        dropout: float = 0.1,
    ):
        """
        Args:
            tft_model: Pre-trained or initialized TFT model
            lstm_input_size: Input size for LSTM (default: 5 for OHLCV)
            lstm_hidden_size: Hidden size for LSTM (default: 32)
            lstm_num_layers: Number of LSTM layers (default: 2)
            fusion_size: Size of fused representation (default: 64)
            num_classes: Number of classes for classification (default: 3)
            quantiles: Quantiles for regression (default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
            lambda_class: Weight for classification loss (default: 1.0)
            lambda_reg: Weight for regression loss (default: 1.0)
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        self.tft_model = tft_model
        self.lambda_class = lambda_class
        self.lambda_reg = lambda_reg
        self.quantiles = quantiles
        self.num_classes = num_classes
        
        # LSTM branch for raw price/volume series
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )
        
        # Get TFT output size (from the decoder)
        # TFT typically outputs [batch, prediction_length, num_quantiles]
        # Use hidden_size from TFT model's hyperparameters
        try:
            self.tft_output_size = tft_model.hparams.hidden_size
        except AttributeError:
            # Fallback: use default hidden size
            self.tft_output_size = 16
        
        # Fusion layer: Gated Linear Unit (GLU)
        fusion_input_size = lstm_hidden_size + self.tft_output_size
        self.fusion_gate = nn.Linear(fusion_input_size, fusion_size)
        self.fusion_proj = nn.Linear(fusion_input_size, fusion_size)
        
        # Multi-task head
        # Task 1: Direction (Classification)
        self.class_head = nn.Sequential(
            nn.Linear(fusion_size, fusion_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_size // 2, num_classes),
        )
        
        # Task 2: Magnitude (Regression with quantiles)
        self.reg_head = nn.Sequential(
            nn.Linear(fusion_size, fusion_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_size // 2, len(quantiles)),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid architecture.
        
        Args:
            x: Input dictionary from TimeSeriesDataSet
        
        Returns:
            Dictionary with 'classification' and 'regression' outputs
        """
        # Extract raw OHLCV from input
        # TimeSeriesDataSet provides 'x' (encoder) and 'decoder_cont' (decoder)
        # We'll use encoder_cont for LSTM (raw price/volume features)
        raw_features = x.get("encoder_cont", None)
        
        if raw_features is None:
            # Fallback: try to get from 'x' if available
            if "x" in x:
                # x['x'] is typically [batch, encoder_length, features]
                # Take first 5 features as OHLCV approximation
                raw_features = x["x"][:, :, :5]
            else:
                # Last resort: use decoder_cont
                raw_features = x.get("decoder_cont", None)
                if raw_features is not None:
                    raw_features = raw_features[:, :, :5]
        
        # LSTM branch: Process raw price/volume series
        if raw_features is not None and raw_features.numel() > 0:
            lstm_out, (h_n, c_n) = self.lstm(raw_features)
            # Use last hidden state from last layer
            lstm_latent = h_n[-1]  # [batch, hidden_size]
        else:
            # Fallback if raw features not available
            batch_size = next(iter(x.values())).shape[0]
            device = next(iter(x.values())).device
            lstm_latent = torch.zeros(
                batch_size, self.lstm.hidden_size, device=device
            )
        
        # TFT branch: Process complex features
        # TFT model expects the same input format
        tft_output = self.tft_model(x)
        
        # Extract latent representation from TFT
        # TFT output is typically [batch, prediction_length, num_quantiles]
        # We need to get the hidden representation, not the prediction
        # For now, we'll use the prediction and extract features from it
        if isinstance(tft_output, dict):
            tft_latent = tft_output.get("prediction", tft_output.get("output"))
        else:
            tft_latent = tft_output
        
        # TFT output shape: [batch, prediction_length, num_quantiles]
        # We need to reduce to [batch, hidden_size] for fusion
        if tft_latent.dim() == 3:
            # Take mean over prediction length and quantiles, then project
            tft_latent = tft_latent.mean(dim=1)  # [batch, num_quantiles]
            # Project to match expected size if needed
            if tft_latent.shape[1] != self.tft_output_size:
                # Use a simple projection (this is a simplification)
                # In practice, you'd extract from TFT's internal representation
                if not hasattr(self, 'tft_proj'):
                    self.tft_proj = nn.Linear(
                        tft_latent.shape[1], self.tft_output_size
                    ).to(tft_latent.device)
                tft_latent = self.tft_proj(tft_latent)
        elif tft_latent.dim() == 2:
            # Already 2D, check if size matches
            if tft_latent.shape[1] != self.tft_output_size:
                # Project to correct size
                if not hasattr(self, 'tft_proj'):
                    self.tft_proj = nn.Linear(
                        tft_latent.shape[1], self.tft_output_size
                    ).to(tft_latent.device)
                tft_latent = self.tft_proj(tft_latent)
        
        # Concatenate LSTM and TFT latent vectors
        combined = torch.cat([lstm_latent, tft_latent], dim=1)
        
        # Gated fusion (GLU)
        gate = torch.sigmoid(self.fusion_gate(combined))
        proj = self.fusion_proj(combined)
        fused = gate * proj
        fused = self.dropout(fused)
        
        # Multi-task head
        class_logits = self.class_head(fused)  # [batch, num_classes]
        reg_output = self.reg_head(fused)  # [batch, num_quantiles]
        
        return {
            "classification": class_logits,
            "regression": reg_output,
        }


class HybridLSTMTFTLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for Hybrid LSTM + TFT model (Phase 3).
    """
    
    def __init__(
        self,
        hybrid_model: HybridLSTMTFT,
        learning_rate: float = 0.001,
        lambda_class: float = 1.0,
        lambda_reg: float = 1.0,
    ):
        """
        Args:
            hybrid_model: HybridLSTMTFT model instance
            learning_rate: Learning rate (default: 0.001)
            lambda_class: Weight for classification loss (default: 1.0)
            lambda_reg: Weight for regression loss (default: 1.0)
        """
        super().__init__()
        self.model = hybrid_model
        self.learning_rate = learning_rate
        self.lambda_class = lambda_class
        self.lambda_reg = lambda_reg
        
        # Loss functions
        self.class_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = QuantileLoss(quantiles=hybrid_model.quantiles)
        
        # Metrics (using pytorch_forecasting metrics for consistency)
        self.train_mae = MAE()
        self.val_mae = MAE()
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        
        # Forward pass
        outputs = self.forward(x)
        
        # Calculate losses
        class_loss = self.class_loss_fn(outputs["classification"], y["classification"])
        reg_loss = self.reg_loss_fn(outputs["regression"], y["regression"])
        
        # Combined loss
        total_loss = self.lambda_class * class_loss + self.lambda_reg * reg_loss
        
        # Calculate accuracy manually
        class_preds = torch.argmax(outputs["classification"], dim=1)
        class_targets = y["classification"]
        class_acc = (class_preds == class_targets).float().mean()
        
        # Update metrics
        mae_val = self.train_mae(outputs["regression"], y["regression"])
        
        # Log
        self.log("train_loss", total_loss, on_step=True, on_epoch=True)
        self.log("train_class_loss", class_loss, on_step=True, on_epoch=True)
        self.log("train_reg_loss", reg_loss, on_step=True, on_epoch=True)
        self.log("train_class_acc", class_acc, on_step=True, on_epoch=True)
        self.log("train_mae", mae_val, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        
        # Forward pass
        outputs = self.forward(x)
        
        # Calculate losses
        class_loss = self.class_loss_fn(outputs["classification"], y["classification"])
        reg_loss = self.reg_loss_fn(outputs["regression"], y["regression"])
        
        # Combined loss
        total_loss = self.lambda_class * class_loss + self.lambda_reg * reg_loss
        
        # Calculate accuracy manually
        class_preds = torch.argmax(outputs["classification"], dim=1)
        class_targets = y["classification"]
        class_acc = (class_preds == class_targets).float().mean()
        
        # Update metrics
        mae_val = self.val_mae(outputs["regression"], y["regression"])
        
        # Log
        self.log("val_loss", total_loss, on_step=False, on_epoch=True)
        self.log("val_class_loss", class_loss, on_step=False, on_epoch=True)
        self.log("val_reg_loss", reg_loss, on_step=False, on_epoch=True)
        self.log("val_class_acc", class_acc, on_step=False, on_epoch=True)
        self.log("val_mae", mae_val, on_step=False, on_epoch=True)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def create_hybrid_lstm_tft(
    training_dataset: TimeSeriesDataSet,
    tft_hidden_size: int = 16,
    tft_attention_head_size: int = 4,
    tft_dropout: float = 0.1,
    lstm_hidden_size: int = 32,
    lstm_num_layers: int = 2,
    fusion_size: int = 64,
    num_classes: int = 3,
    quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
    lambda_class: float = 1.0,
    lambda_reg: float = 1.0,
    learning_rate: float = 0.001,
    **kwargs,
) -> HybridLSTMTFTLightning:
    """
    Create Hybrid LSTM + TFT model (Phase 3: Advanced).
    
    Args:
        training_dataset: TimeSeriesDataSet for TFT initialization
        tft_hidden_size: Hidden size for TFT (default: 16)
        tft_attention_head_size: Attention head size for TFT (default: 4)
        tft_dropout: Dropout for TFT (default: 0.1)
        lstm_hidden_size: Hidden size for LSTM (default: 32)
        lstm_num_layers: Number of LSTM layers (default: 2)
        fusion_size: Size of fused representation (default: 64)
        num_classes: Number of classes for classification (default: 3)
        quantiles: Quantiles for regression (default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
        lambda_class: Weight for classification loss (default: 1.0)
        lambda_reg: Weight for regression loss (default: 1.0)
        learning_rate: Learning rate (default: 0.001)
        **kwargs: Additional arguments for TFT model
    
    Returns:
        HybridLSTMTFTLightning instance
    """
    # Create base TFT model
    tft_model = create_vanilla_tft(
        training_dataset,
        hidden_size=tft_hidden_size,
        attention_head_size=tft_attention_head_size,
        dropout=tft_dropout,
        quantiles=quantiles,
        **kwargs,
    )
    
    # Create hybrid model
    hybrid_model = HybridLSTMTFT(
        tft_model=tft_model,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        fusion_size=fusion_size,
        num_classes=num_classes,
        quantiles=quantiles,
        lambda_class=lambda_class,
        lambda_reg=lambda_reg,
        dropout=tft_dropout,
    )
    
    # Wrap in Lightning module
    lightning_model = HybridLSTMTFTLightning(
        hybrid_model=hybrid_model,
        learning_rate=learning_rate,
        lambda_class=lambda_class,
        lambda_reg=lambda_reg,
    )
    
    print(
        color_text(
            f"Created Hybrid LSTM + TFT (Phase 3): "
            f"TFT(hidden={tft_hidden_size}), LSTM(hidden={lstm_hidden_size}, layers={lstm_num_layers}), "
            f"Fusion(size={fusion_size}), λ_class={lambda_class}, λ_reg={lambda_reg}",
            Fore.GREEN,
        )
    )
    
    return lightning_model


# ============================================================================
# Utility Functions
# ============================================================================

def load_tft_model(
    checkpoint_path: Union[str, Path],
    training_dataset: Optional[TimeSeriesDataSet] = None,
) -> TemporalFusionTransformer:
    """
    Load a trained TFT model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        training_dataset: Optional training dataset (required if checkpoint doesn't contain it)
    
    Returns:
        Loaded TemporalFusionTransformer model
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load model
    if training_dataset is not None:
        model = TemporalFusionTransformer.load_from_checkpoint(
            str(checkpoint_path), dataset=training_dataset
        )
    else:
        model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint_path))
    
    print(
        color_text(
            f"Loaded TFT model from {checkpoint_path}",
            Fore.GREEN,
        )
    )
    
    return model


def save_model_config(
    model: Union[TemporalFusionTransformer, HybridLSTMTFTLightning],
    config_path: Union[str, Path],
) -> None:
    """
    Save model configuration to JSON file.
    
    Args:
        model: Model instance
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(model, TemporalFusionTransformer):
        config = {
            "model_type": "vanilla_tft",
            "hidden_size": model.hparams.hidden_size,
            "attention_head_size": model.hparams.attention_head_size,
            "dropout": model.hparams.dropout,
            "learning_rate": model.hparams.learning_rate,
        }
    elif isinstance(model, HybridLSTMTFTLightning):
        config = {
            "model_type": "hybrid_lstm_tft",
            "learning_rate": model.learning_rate,
            "lambda_class": model.lambda_class,
            "lambda_reg": model.lambda_reg,
        }
    else:
        config = {"model_type": "unknown"}
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(
        color_text(
            f"Saved model configuration to {config_path}",
            Fore.GREEN,
        )
    )

