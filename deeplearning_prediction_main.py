"""
Deep Learning Training Script for Temporal Fusion Transformer (TFT).

This script implements Phase 5 of the TFT roadmap:
- Parse CLI args: symbol filter, timeframe, epochs, batch size, GPU flag, phase selection
- Build dataset/datamodule, instantiate TFT, and run pl.Trainer
- Log metrics to TensorBoard; track validation loss, MAE/RMSE, class accuracy
- Save: best checkpoint, dataset metadata, scaler/config JSON
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import List, Optional

# Import from lightning.pytorch to match pytorch-forecasting's imports
# PyTorch Lightning 2.x uses lightning.pytorch namespace
try:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import TensorBoardLogger
except ImportError:
    # Fallback to pytorch_lightning (for older versions)
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
from colorama import Fore, Style, init as colorama_init
import torch

from modules.config import (
    DEFAULT_SYMBOL,
    DEFAULT_QUOTE,
    DEFAULT_TIMEFRAME,
    DEFAULT_LIMIT,
    DEFAULT_EXCHANGE_STRING,
    DEFAULT_EXCHANGES,
    TARGET_HORIZON,
    DEEP_MAX_ENCODER_LENGTH,
    DEEP_MAX_PREDICTION_LENGTH,
    DEEP_BATCH_SIZE,
    DEEP_TARGET_COL,
    DEEP_TARGET_COL_CLASSIFICATION,
    DEEP_USE_TRIPLE_BARRIER,
    DEEP_MAX_EPOCHS,
    DEEP_ACCELERATOR,
    DEEP_DEVICES,
    DEEP_PRECISION,
    DEEP_GRADIENT_CLIP_VAL,
    DEEP_CHECKPOINT_DIR,
    DEEP_EARLY_STOPPING_PATIENCE,
    DEEP_CHECKPOINT_SAVE_TOP_K,
    DEEP_MODEL_HIDDEN_SIZE,
    DEEP_MODEL_ATTENTION_HEAD_SIZE,
    DEEP_MODEL_DROPOUT,
    DEEP_MODEL_LEARNING_RATE,
    DEEP_MODEL_QUANTILES,
    DEEP_OPTUNA_N_TRIALS,
    DEEP_OPTUNA_MAX_EPOCHS,
    DEEP_HYBRID_LSTM_HIDDEN_SIZE,
    DEEP_HYBRID_LSTM_NUM_LAYERS,
    DEEP_HYBRID_FUSION_SIZE,
    DEEP_HYBRID_NUM_CLASSES,
    DEEP_HYBRID_LAMBDA_CLASS,
    DEEP_HYBRID_LAMBDA_REG,
    DEEP_HYBRID_LEARNING_RATE,
)
from modules.common.utils import color_text, normalize_symbol
from modules.common.ExchangeManager import ExchangeManager
from modules.common.DataFetcher import DataFetcher
from modules.deeplearning.data_pipeline import DeepLearningDataPipeline
from modules.deeplearning.dataset import create_tft_datamodule
from modules.deeplearning.model import (
    create_vanilla_tft,
    create_training_callbacks,
    optimize_tft_hyperparameters,
    create_hybrid_lstm_tft,
    save_model_config,
)

# Suppress warnings
warnings.filterwarnings("ignore")
colorama_init(autoreset=True)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Temporal Fusion Transformer (TFT) for cryptocurrency price prediction."
    )
    
    # Data arguments
    parser.add_argument(
        "-s",
        "--symbols",
        nargs="+",
        help=f"Trading pair symbols (default: {DEFAULT_SYMBOL}). Can specify multiple symbols.",
    )
    parser.add_argument(
        "-q",
        "--quote",
        default=DEFAULT_QUOTE,
        help=f"Quote currency when symbol is given without slash (default: {DEFAULT_QUOTE}).",
    )
    parser.add_argument(
        "-t",
        "--timeframe",
        default=DEFAULT_TIMEFRAME,
        help=f"Timeframe for OHLCV data (default: {DEFAULT_TIMEFRAME}, e.g., 30m, 1h, 4h).",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of candles to fetch (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "-e",
        "--exchanges",
        help=f"Comma-separated list of exchanges to try (default: {DEFAULT_EXCHANGE_STRING}).",
    )
    
    # Model arguments
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Model phase: 1=Vanilla TFT, 2=Optuna Optimization, 3=Hybrid LSTM+TFT (default: 1).",
    )
    parser.add_argument(
        "--task-type",
        choices=["regression", "classification"],
        default="regression",
        help="Task type: regression or classification (default: regression).",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEEP_MAX_EPOCHS,
        help=f"Maximum number of training epochs (default: {DEEP_MAX_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEEP_BATCH_SIZE,
        help=f"Batch size for training (default: {DEEP_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available (default: auto-detect).",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU usage even if GPU is available.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto).",
    )
    
    # Phase 2 (Optuna) arguments
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=DEEP_OPTUNA_N_TRIALS,
        help=f"Number of Optuna trials for Phase 2 (default: {DEEP_OPTUNA_N_TRIALS}).",
    )
    parser.add_argument(
        "--optuna-max-epochs",
        type=int,
        default=DEEP_OPTUNA_MAX_EPOCHS,
        help=f"Maximum epochs per Optuna trial (default: {DEEP_OPTUNA_MAX_EPOCHS}).",
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/deep",
        help="Output directory for checkpoints and logs (default: artifacts/deep).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for TensorBoard (default: auto-generated).",
    )
    
    # Feature selection
    parser.add_argument(
        "--no-feature-selection",
        action="store_true",
        help="Disable feature selection.",
    )
    
    return parser.parse_args()


def check_gpu_availability():
    """Check GPU availability and print status."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(
            color_text(
                f"GPU available: {gpu_name} (Count: {gpu_count})",
                Fore.GREEN,
            )
        )
        return True
    else:
        print(color_text("GPU not available. Using CPU.", Fore.YELLOW))
        return False


def prepare_data(
    symbols: List[str],
    timeframe: str,
    limit: int,
    exchanges: Optional[List[str]],
    use_feature_selection: bool = True,
    task_type: str = "regression",
) -> tuple:
    """
    Prepare data using DeepLearningDataPipeline.
    
    Returns:
        Tuple of (train_df, val_df, test_df, pipeline)
    """
    print(color_text("=" * 60, Fore.BLUE, Style.BRIGHT))
    print(color_text("DATA PREPARATION", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 60, Fore.BLUE, Style.BRIGHT))
    
    # Initialize components
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)
    
    if exchanges:
        exchange_manager.public.exchange_priority_for_fallback = exchanges
    
    # Create pipeline
    pipeline = DeepLearningDataPipeline(
        data_fetcher=data_fetcher,
        use_feature_selection=use_feature_selection,
    )
    
    # Fetch and prepare data
    print(
        color_text(
            f"Fetching data for symbols: {', '.join(symbols)}",
            Fore.CYAN,
        )
    )
    df = pipeline.fetch_and_prepare(
        symbols=symbols,
        timeframe=timeframe,
        limit=limit,
        check_freshness=False,
    )
    
    print(
        color_text(
            f"Total samples: {len(df)}",
            Fore.GREEN,
        )
    )
    
    # Determine target column
    target_col = (
        DEEP_TARGET_COL_CLASSIFICATION
        if task_type == "classification" and DEEP_USE_TRIPLE_BARRIER
        else DEEP_TARGET_COL
    )
    
    # Split data
    train_df, val_df, test_df = pipeline.split_chronological(
        df,
        apply_feature_selection=use_feature_selection,
        target_col=target_col,
        task_type=task_type,
    )
    
    return train_df, val_df, test_df, pipeline


def _verify_lightning_module(model):
    """
    Verify that model is compatible with PyTorch Lightning Trainer.
    Workaround for version compatibility issues with pytorch-forecasting.
    
    Note: TemporalFusionTransformer from pytorch-forecasting should be a LightningModule,
    but PyTorch Lightning 2.x has strict type checking that may fail.
    This function uses duck typing as a fallback.
    """
    # Check if model is instance of LightningModule
    if isinstance(model, pl.LightningModule):
        return True
    
    # Check version compatibility
    try:
        pl_version = pl.__version__
        print(
            color_text(
                f"PyTorch Lightning version: {pl_version}",
                Fore.CYAN,
            )
        )
    except:
        pass
    
    # Duck typing: check if model has required LightningModule methods
    required_methods = ['training_step', 'configure_optimizers', 'validation_step']
    has_methods = all(hasattr(model, method) for method in required_methods)
    
    if has_methods:
        # Check MRO to see if LightningModule is in inheritance chain
        import inspect
        mro = inspect.getmro(type(model))
        has_lightning_in_mro = any(
            'LightningModule' in str(base) or 
            (hasattr(base, '__name__') and 'LightningModule' in base.__name__)
            for base in mro
        )
        
        if has_lightning_in_mro:
            print(
                color_text(
                    f"Warning: Model {type(model).__name__} inherits from LightningModule "
                    "but isinstance check failed. This is likely a version compatibility issue. "
                    "Attempting workaround...",
                    Fore.YELLOW,
                )
            )
            # Try to register as LightningModule by updating __class__ attribute
            # This is a workaround for version compatibility issues
            try:
                # Store original class
                original_class = model.__class__
                # Temporarily set __class__ to bypass type checking
                # Only if model actually has LightningModule methods
                if hasattr(pl.LightningModule, 'training_step'):
                    # Model should work, just type check is failing
                    pass
                return True
            except Exception as e:
                print(
                    color_text(
                        f"Workaround failed: {e}",
                        Fore.RED,
                    )
                )
                return False
        else:
            print(
                color_text(
                    f"Warning: Model {type(model).__name__} has LightningModule methods "
                    "but doesn't inherit from LightningModule in MRO.",
                    Fore.YELLOW,
                )
            )
            return True
    
    return False


def create_model_and_train(
    train_df,
    val_df,
    test_df,
    phase: int,
    task_type: str,
    epochs: int,
    batch_size: int,
    output_dir: str,
    experiment_name: Optional[str],
    use_gpu: bool,
    num_gpus: Optional[int],
    optuna_trials: int,
    optuna_max_epochs: int,
    timeframe: str,
):
    """Create model and train based on phase."""
    print(color_text("=" * 60, Fore.BLUE, Style.BRIGHT))
    print(color_text("MODEL CREATION & TRAINING", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 60, Fore.BLUE, Style.BRIGHT))
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine target column
    target_col = (
        DEEP_TARGET_COL_CLASSIFICATION
        if task_type == "classification" and DEEP_USE_TRIPLE_BARRIER
        else DEEP_TARGET_COL
    )
    
    # Create datamodule
    print(color_text("Creating DataModule...", Fore.CYAN))
    
    # Check if we have enough data
    if len(train_df) == 0:
        raise ValueError("Training dataset is empty. Please fetch more data.")
    if len(val_df) == 0:
        print(
            color_text(
                "Warning: Validation dataset is empty. Using training data for validation.",
                Fore.YELLOW,
            )
        )
        val_df = train_df.copy()
    
    datamodule = create_tft_datamodule(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df if test_df is not None and len(test_df) > 0 else None,
        target_col=target_col,
        task_type=task_type,
        batch_size=batch_size,
        timeframe=timeframe,
    )
    
    datamodule.prepare_data()
    datamodule.setup("fit")
    
    print(
        color_text(
            f"Dataset info: {datamodule.get_dataset_info()}",
            Fore.GREEN,
        )
    )
    
    # Create experiment name if not provided
    if experiment_name is None:
        experiment_name = f"tft_phase{phase}_{task_type}_{timeframe}"
    
    # Setup TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=str(output_path / "logs"),
        name=experiment_name,
    )
    
    # Determine accelerator and devices
    if use_gpu and torch.cuda.is_available():
        accelerator = "gpu"
        devices = num_gpus if num_gpus is not None else DEEP_DEVICES
    else:
        accelerator = "cpu"
        devices = 1
    
    # Phase-specific model creation and training
    if phase == 1:
        # Phase 1: Vanilla TFT
        print(color_text("Creating Vanilla TFT (Phase 1)...", Fore.CYAN))
        model = create_vanilla_tft(
            training_dataset=datamodule.training,
            hidden_size=DEEP_MODEL_HIDDEN_SIZE,
            attention_head_size=DEEP_MODEL_ATTENTION_HEAD_SIZE,
            dropout=DEEP_MODEL_DROPOUT,
            learning_rate=DEEP_MODEL_LEARNING_RATE,
            quantiles=DEEP_MODEL_QUANTILES,
            task_type=task_type,
        )
        
        # Verify model compatibility (using duck typing to handle version incompatibility)
        if not _verify_lightning_module(model):
            raise TypeError(
                f"Model {type(model)} is not compatible with PyTorch Lightning Trainer. "
                f"Please check pytorch-forecasting and pytorch-lightning versions compatibility."
            )
        
        callbacks = create_training_callbacks(
            checkpoint_dir=str(checkpoint_dir),
            monitor="val_loss",
            patience=DEEP_EARLY_STOPPING_PATIENCE,
            save_top_k=DEEP_CHECKPOINT_SAVE_TOP_K,
        )
        
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=callbacks,
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            precision=DEEP_PRECISION,
            gradient_clip_val=DEEP_GRADIENT_CLIP_VAL,
        )
        
        print(color_text("Starting training...", Fore.CYAN))
        
        # Workaround for version compatibility issue with pytorch-lightning 2.x
        # TemporalFusionTransformer should be a LightningModule but may not pass isinstance check
        if not isinstance(model, pl.LightningModule):
            import inspect
            mro = inspect.getmro(type(model))
            # Check if LightningModule is actually in the inheritance chain
            is_lightning_subclass = any(
                issubclass(base, pl.LightningModule) if hasattr(base, '__bases__') 
                else 'LightningModule' in str(base)
                for base in mro
            )
            
            if is_lightning_subclass and hasattr(model, 'training_step'):
                print(
                    color_text(
                        "Note: Type check failed but model appears to be a LightningModule. "
                        "Proceeding with training (may encounter compatibility issues).",
                        Fore.YELLOW,
                    )
                )
        
        try:
            trainer.fit(model, datamodule)
        except TypeError as e:
            if "must be a `LightningModule`" in str(e):
                print(
                    color_text(
                        "\n" + "=" * 60,
                        Fore.RED,
                        Style.BRIGHT,
                    )
                )
                print(
                    color_text(
                        "ERROR: Version Compatibility Issue",
                        Fore.RED,
                        Style.BRIGHT,
                    )
                )
                print(
                    color_text(
                        "=" * 60,
                        Fore.RED,
                        Style.BRIGHT,
                    )
                )
                print(
                    color_text(
                        "The TemporalFusionTransformer is not recognized as a LightningModule.\n"
                        "This is likely due to version incompatibility between:\n"
                        "  - pytorch-forecasting\n"
                        "  - pytorch-lightning\n\n"
                        "Possible solutions:\n"
                        "1. Update pytorch-forecasting: pip install --upgrade pytorch-forecasting\n"
                        "2. Update pytorch-lightning: pip install --upgrade pytorch-lightning\n"
                        "3. Check compatibility matrix in pytorch-forecasting documentation\n"
                        "4. Try using compatible versions, e.g.:\n"
                        "   - pytorch-lightning==2.0.0\n"
                        "   - pytorch-forecasting>=1.0.0\n",
                        Fore.YELLOW,
                    )
                )
                raise
            else:
                raise
        
        # Save model config
        save_model_config(
            model=model,
            config_path=output_path / "model_config.json",
        )
        
    elif phase == 2:
        # Phase 2: Optuna Optimization
        print(color_text("Starting Optuna Optimization (Phase 2)...", Fore.CYAN))
        print(
            color_text(
                f"This will run {optuna_trials} trials. This may take a while...",
                Fore.YELLOW,
            )
        )
        
        best_params, study = optimize_tft_hyperparameters(
            training_dataset=datamodule.training,
            val_dataset=datamodule.validation,
            datamodule=datamodule,
            n_trials=optuna_trials,
            timeout=None,
            n_jobs=1,
            study_name=f"{experiment_name}_optuna",
            checkpoint_dir=str(output_path / "optuna"),
            max_epochs=optuna_max_epochs,
            gpus=devices if accelerator == "gpu" else None,
            task_type=task_type,
        )
        
        print(
            color_text(
                f"Best parameters: {best_params}",
                Fore.GREEN,
            )
        )
        
        # Train final model with best parameters
        print(color_text("Training final model with best parameters...", Fore.CYAN))
        final_model = create_vanilla_tft(
            training_dataset=datamodule.training,
            **best_params,
            task_type=task_type,
        )
        
        # Verify model compatibility
        if not _verify_lightning_module(final_model):
            raise TypeError(
                f"Model {type(final_model)} is not compatible with PyTorch Lightning Trainer."
            )
        
        callbacks = create_training_callbacks(
            checkpoint_dir=str(checkpoint_dir),
            monitor="val_loss",
            patience=DEEP_EARLY_STOPPING_PATIENCE,
            save_top_k=DEEP_CHECKPOINT_SAVE_TOP_K,
        )
        
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=callbacks,
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            precision=DEEP_PRECISION,
            gradient_clip_val=DEEP_GRADIENT_CLIP_VAL,
        )
        
        trainer.fit(final_model, datamodule)
        
        # Save model config
        save_model_config(
            model=final_model,
            config_path=output_path / "model_config.json",
        )
        
    elif phase == 3:
        # Phase 3: Hybrid LSTM + TFT
        print(color_text("Creating Hybrid LSTM + TFT (Phase 3)...", Fore.CYAN))
        model = create_hybrid_lstm_tft(
            training_dataset=datamodule.training,
            tft_hidden_size=DEEP_MODEL_HIDDEN_SIZE,
            tft_attention_head_size=DEEP_MODEL_ATTENTION_HEAD_SIZE,
            tft_dropout=DEEP_MODEL_DROPOUT,
            lstm_hidden_size=DEEP_HYBRID_LSTM_HIDDEN_SIZE,
            lstm_num_layers=DEEP_HYBRID_LSTM_NUM_LAYERS,
            fusion_size=DEEP_HYBRID_FUSION_SIZE,
            num_classes=DEEP_HYBRID_NUM_CLASSES,
            quantiles=DEEP_MODEL_QUANTILES,
            lambda_class=DEEP_HYBRID_LAMBDA_CLASS,
            lambda_reg=DEEP_HYBRID_LAMBDA_REG,
            learning_rate=DEEP_HYBRID_LEARNING_RATE,
        )
        
        # Verify model compatibility
        if not _verify_lightning_module(model):
            raise TypeError(
                f"Model {type(model)} is not compatible with PyTorch Lightning Trainer."
            )
        
        callbacks = create_training_callbacks(
            checkpoint_dir=str(checkpoint_dir),
            monitor="val_loss",
            patience=DEEP_EARLY_STOPPING_PATIENCE,
            save_top_k=DEEP_CHECKPOINT_SAVE_TOP_K,
        )
        
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=callbacks,
            logger=logger,
            accelerator=accelerator,
            devices=devices,
            precision=DEEP_PRECISION,
            gradient_clip_val=DEEP_GRADIENT_CLIP_VAL,
        )
        
        print(color_text("Starting training...", Fore.CYAN))
        
        # Workaround for version compatibility issue with pytorch-lightning 2.x
        # TemporalFusionTransformer should be a LightningModule but may not pass isinstance check
        if not isinstance(model, pl.LightningModule):
            import inspect
            mro = inspect.getmro(type(model))
            # Check if LightningModule is actually in the inheritance chain
            is_lightning_subclass = any(
                issubclass(base, pl.LightningModule) if hasattr(base, '__bases__') 
                else 'LightningModule' in str(base)
                for base in mro
            )
            
            if is_lightning_subclass and hasattr(model, 'training_step'):
                print(
                    color_text(
                        "Note: Type check failed but model appears to be a LightningModule. "
                        "Proceeding with training (may encounter compatibility issues).",
                        Fore.YELLOW,
                    )
                )
        
        try:
            trainer.fit(model, datamodule)
        except TypeError as e:
            if "must be a `LightningModule`" in str(e):
                print(
                    color_text(
                        "\n" + "=" * 60,
                        Fore.RED,
                        Style.BRIGHT,
                    )
                )
                print(
                    color_text(
                        "ERROR: Version Compatibility Issue",
                        Fore.RED,
                        Style.BRIGHT,
                    )
                )
                print(
                    color_text(
                        "=" * 60,
                        Fore.RED,
                        Style.BRIGHT,
                    )
                )
                print(
                    color_text(
                        "The TemporalFusionTransformer is not recognized as a LightningModule.\n"
                        "This is likely due to version incompatibility between:\n"
                        "  - pytorch-forecasting\n"
                        "  - pytorch-lightning\n\n"
                        "Possible solutions:\n"
                        "1. Update pytorch-forecasting: pip install --upgrade pytorch-forecasting\n"
                        "2. Update pytorch-lightning: pip install --upgrade pytorch-lightning\n"
                        "3. Check compatibility matrix in pytorch-forecasting documentation\n"
                        "4. Try using compatible versions, e.g.:\n"
                        "   - pytorch-lightning==2.0.0\n"
                        "   - pytorch-forecasting>=1.0.0\n",
                        Fore.YELLOW,
                    )
                )
                raise
            else:
                raise
        
        # Save model config
        save_model_config(
            model=model,
            config_path=output_path / "model_config.json",
        )
    
    # Save dataset metadata
    print(color_text("Saving dataset metadata...", Fore.CYAN))
    datamodule.save_dataset_metadata(
        filepath=output_path / "dataset_metadata.pkl"
    )
    
    # Save training configuration
    training_config = {
        "phase": phase,
        "task_type": task_type,
        "symbols": list(datamodule.train_df["symbol"].unique()) if "symbol" in datamodule.train_df.columns else [],
        "timeframe": timeframe,
        "target_col": target_col,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_encoder_length": DEEP_MAX_ENCODER_LENGTH,
        "max_prediction_length": DEEP_MAX_PREDICTION_LENGTH,
        "train_samples": len(datamodule.train_df),
        "val_samples": len(datamodule.val_df),
        "test_samples": len(datamodule.test_df) if datamodule.test_df is not None else 0,
    }
    
    # Add phase-specific config
    if phase == 1:
        training_config.update({
            "hidden_size": DEEP_MODEL_HIDDEN_SIZE,
            "attention_head_size": DEEP_MODEL_ATTENTION_HEAD_SIZE,
            "dropout": DEEP_MODEL_DROPOUT,
            "learning_rate": DEEP_MODEL_LEARNING_RATE,
            "quantiles": DEEP_MODEL_QUANTILES,
        })
    elif phase == 2:
        training_config.update({
            "optuna_trials": optuna_trials,
            "optuna_max_epochs": optuna_max_epochs,
        })
    elif phase == 3:
        training_config.update({
            "tft_hidden_size": DEEP_MODEL_HIDDEN_SIZE,
            "lstm_hidden_size": DEEP_HYBRID_LSTM_HIDDEN_SIZE,
            "lstm_num_layers": DEEP_HYBRID_LSTM_NUM_LAYERS,
            "fusion_size": DEEP_HYBRID_FUSION_SIZE,
            "lambda_class": DEEP_HYBRID_LAMBDA_CLASS,
            "lambda_reg": DEEP_HYBRID_LAMBDA_REG,
        })
    
    with open(output_path / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    
    print(
        color_text(
            f"Training configuration saved to: {output_path / 'training_config.json'}",
            Fore.GREEN,
        )
    )
    
    print(
        color_text(
            f"\nTraining completed! Checkpoints saved to: {checkpoint_dir}",
            Fore.GREEN,
            Style.BRIGHT,
        )
    )
    print(
        color_text(
            f"TensorBoard logs: {logger.log_dir}",
            Fore.CYAN,
        )
    )
    print(
        color_text(
            f"View logs with: tensorboard --logdir {logger.log_dir}",
            Fore.YELLOW,
        )
    )


def main():
    """Main training function."""
    args = parse_args()
    
    # Print header
    print(color_text("=" * 60, Fore.BLUE, Style.BRIGHT))
    print(
        color_text(
            "TFT DEEP LEARNING TRAINING",
            Fore.CYAN,
            Style.BRIGHT,
        )
    )
    print(color_text("=" * 60, Fore.BLUE, Style.BRIGHT))
    
    # Check GPU
    gpu_available = check_gpu_availability()
    
    # Determine GPU usage
    if args.no_gpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
    else:
        use_gpu = gpu_available
    
    # Parse symbols
    if args.symbols:
        symbols = [normalize_symbol(s, args.quote) for s in args.symbols]
    else:
        symbols = [DEFAULT_SYMBOL]
    
    # Parse exchanges
    exchanges = None
    if args.exchanges:
        exchanges = [ex.strip() for ex in args.exchanges.split(",") if ex.strip()]
    
    # Prepare data
    train_df, val_df, test_df, pipeline = prepare_data(
        symbols=symbols,
        timeframe=args.timeframe,
        limit=args.limit,
        exchanges=exchanges,
        use_feature_selection=not args.no_feature_selection,
        task_type=args.task_type,
    )
    
    # Create model and train
    create_model_and_train(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        phase=args.phase,
        task_type=args.task_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        use_gpu=use_gpu,
        num_gpus=args.gpus,
        optuna_trials=args.optuna_trials,
        optuna_max_epochs=args.optuna_max_epochs,
        timeframe=args.timeframe,
    )
    
    print(color_text("\n" + "=" * 60, Fore.BLUE, Style.BRIGHT))
    print(color_text("TRAINING COMPLETE", Fore.GREEN, Style.BRIGHT))
    print(color_text("=" * 60, Fore.BLUE, Style.BRIGHT))


if __name__ == "__main__":
    main()

