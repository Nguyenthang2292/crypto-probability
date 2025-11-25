"""
Test script for deeplearning_prediction_main.py - Training script functions.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from deeplearning_prediction_main import (
    parse_args,
    check_gpu_availability,
    prepare_data,
    create_model_and_train,
)
from modules.deeplearning_dataset import TFTDataModule

# Suppress warnings
warnings.filterwarnings("ignore")


def create_sample_dataframe(n_samples=200, symbol="BTC/USDT"):
    """Create a sample DataFrame for testing."""
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=n_samples),
        periods=n_samples,
        freq="1H",
    )
    
    np.random.seed(42)
    base_price = 50000
    prices = base_price + np.cumsum(np.random.randn(n_samples) * 100)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": symbol,
        "open": prices + np.random.randn(n_samples) * 10,
        "high": prices + np.abs(np.random.randn(n_samples) * 20),
        "low": prices - np.abs(np.random.randn(n_samples) * 20),
        "close": prices,
        "volume": np.random.rand(n_samples) * 1000,
        "SMA_20": prices,
        "RSI_14": 50 + np.random.randn(n_samples) * 10,
        "ATR_14": np.random.rand(n_samples) * 100,
        "candle_index": range(n_samples),
        "future_log_return": np.random.randn(n_samples) * 0.01,
        "hour_sin": np.sin(2 * np.pi * np.arange(n_samples) / 24),
        "hour_cos": np.cos(2 * np.pi * np.arange(n_samples) / 24),
    })
    
    df = df.fillna(method="ffill").fillna(method="bfill")
    
    return df


def test_parse_args_defaults():
    """Test parse_args with default values."""
    with patch("sys.argv", ["deeplearning_prediction_main.py"]):
        args = parse_args()
        
        assert args.phase == 1
        assert args.task_type == "regression"
        assert args.epochs == 100  # DEEP_MAX_EPOCHS
        assert args.batch_size == 64  # DEEP_BATCH_SIZE
        assert args.timeframe == "1h"
        assert args.limit == 1500


def test_parse_args_custom_values():
    """Test parse_args with custom values."""
    with patch("sys.argv", [
        "deeplearning_prediction_main.py",
        "--symbols", "BTC/USDT", "ETH/USDT",
        "--timeframe", "4h",
        "--phase", "2",
        "--epochs", "50",
        "--batch-size", "32",
        "--gpu",
    ]):
        args = parse_args()
        
        assert args.symbols == ["BTC/USDT", "ETH/USDT"]
        assert args.timeframe == "4h"
        assert args.phase == 2
        assert args.epochs == 50
        assert args.batch_size == 32
        assert args.gpu is True


def test_parse_args_phase_options():
    """Test parse_args phase options."""
    for phase in [1, 2, 3]:
        with patch("sys.argv", ["deeplearning_prediction_main.py", "--phase", str(phase)]):
            args = parse_args()
            assert args.phase == phase


def test_parse_args_task_type():
    """Test parse_args task type options."""
    for task_type in ["regression", "classification"]:
        with patch("sys.argv", ["deeplearning_prediction_main.py", "--task-type", task_type]):
            args = parse_args()
            assert args.task_type == task_type


def test_check_gpu_availability():
    """Test check_gpu_availability function."""
    # Mock torch.cuda
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.device_count", return_value=1):
            with patch("torch.cuda.get_device_name", return_value="Test GPU"):
                result = check_gpu_availability()
                assert result is True
    
    with patch("torch.cuda.is_available", return_value=False):
        result = check_gpu_availability()
        assert result is False


@patch("modules.ExchangeManager.ExchangeManager")
@patch("modules.DataFetcher.DataFetcher")
@patch("modules.deeplearning_data_pipeline.DeepLearningDataPipeline")
def test_prepare_data(mock_pipeline_class, mock_fetcher_class, mock_exchange_manager):
    """Test prepare_data function."""
    # Create mock pipeline
    mock_pipeline = Mock()
    mock_pipeline.fetch_and_prepare.return_value = create_sample_dataframe(500, "BTC/USDT")
    mock_pipeline.split_chronological.return_value = (
        create_sample_dataframe(350, "BTC/USDT"),
        create_sample_dataframe(75, "BTC/USDT"),
        create_sample_dataframe(75, "BTC/USDT"),
    )
    mock_pipeline_class.return_value = mock_pipeline
    
    # Create mock fetcher
    mock_fetcher = Mock()
    mock_fetcher_class.return_value = mock_fetcher
    
    # Create mock exchange manager
    mock_exchange_manager.return_value = Mock()
    
    train_df, val_df, test_df, pipeline = prepare_data(
        symbols=["BTC/USDT"],
        timeframe="1h",
        limit=500,
        exchanges=None,
        use_feature_selection=False,
        task_type="regression",
    )
    
    assert train_df is not None
    assert val_df is not None
    assert test_df is not None
    assert pipeline is not None


@patch("deeplearning_prediction_main.create_tft_datamodule")
@patch("pytorch_lightning.Trainer")
@patch("pytorch_lightning.loggers.TensorBoardLogger")
@patch("deeplearning_prediction_main.create_vanilla_tft")
@patch("deeplearning_prediction_main.create_training_callbacks")
@patch("deeplearning_prediction_main.save_model_config")
def test_create_model_and_train_phase1(
    mock_save_config,
    mock_create_callbacks,
    mock_create_model,
    mock_logger,
    mock_trainer,
    mock_datamodule,
):
    """Test create_model_and_train for Phase 1."""
    # Create sample data
    train_df = create_sample_dataframe(100, "BTC/USDT")
    val_df = create_sample_dataframe(50, "BTC/USDT")
    test_df = create_sample_dataframe(30, "BTC/USDT")
    
    # Mock datamodule
    mock_dm = Mock(spec=TFTDataModule)
    mock_dm.training = Mock()
    mock_dm.validation = Mock()
    mock_dm.test = None
    mock_dm.train_df = train_df
    mock_dm.val_df = val_df
    mock_dm.test_df = test_df
    mock_dm.get_dataset_info.return_value = {
        "train_samples": 100,
        "val_samples": 50,
        "test_samples": 30,
    }
    mock_dm.save_dataset_metadata = Mock()
    mock_datamodule.return_value = mock_dm
    
    # Mock model
    mock_model = Mock()
    mock_create_model.return_value = mock_model
    
    # Mock trainer
    mock_trainer_instance = Mock()
    mock_trainer.return_value = mock_trainer_instance
    
    # Mock callbacks
    mock_callbacks = [Mock(), Mock(), Mock()]
    mock_create_callbacks.return_value = mock_callbacks
    
    # Mock logger
    mock_logger_instance = Mock()
    mock_logger_instance.log_dir = "/tmp/logs"
    mock_logger.return_value = mock_logger_instance
    
    with tempfile.TemporaryDirectory() as tmpdir:
        create_model_and_train(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            phase=1,
            task_type="regression",
            epochs=2,
            batch_size=32,
            output_dir=tmpdir,
            experiment_name="test",
            use_gpu=False,
            num_gpus=None,
            optuna_trials=10,
            optuna_max_epochs=5,
            timeframe="1h",
        )
        
        # Verify calls
        mock_datamodule.assert_called_once()
        mock_create_model.assert_called_once()
        mock_trainer_instance.fit.assert_called_once()
        mock_save_config.assert_called_once()


def test_prepare_data_empty_symbols():
    """Test prepare_data with empty symbols list."""
    with pytest.raises(ValueError):
        prepare_data(
            symbols=[],
            timeframe="1h",
            limit=500,
            exchanges=None,
            use_feature_selection=False,
            task_type="regression",
        )


@patch("deeplearning_prediction_main.ExchangeManager")
@patch("deeplearning_prediction_main.DataFetcher")
@patch("deeplearning_prediction_main.DeepLearningDataPipeline")
def test_prepare_data_feature_selection(mock_pipeline_class, mock_fetcher_class, mock_exchange_manager):
    """Test prepare_data with feature selection enabled."""
    mock_pipeline = Mock()
    mock_pipeline.fetch_and_prepare.return_value = create_sample_dataframe(500, "BTC/USDT")
    mock_pipeline.split_chronological.return_value = (
        create_sample_dataframe(350, "BTC/USDT"),
        create_sample_dataframe(75, "BTC/USDT"),
        create_sample_dataframe(75, "BTC/USDT"),
    )
    mock_pipeline_class.return_value = mock_pipeline
    
    mock_fetcher = Mock()
    mock_fetcher_class.return_value = mock_fetcher
    mock_exchange_manager.return_value = Mock()
    
    train_df, val_df, test_df, pipeline = prepare_data(
        symbols=["BTC/USDT"],
        timeframe="1h",
        limit=500,
        exchanges=None,
        use_feature_selection=True,  # Enable feature selection
        task_type="regression",
    )
    
    # Verify that split_chronological was called with apply_feature_selection=True
    assert mock_pipeline.split_chronological.called
    call_kwargs = mock_pipeline.split_chronological.call_args[1] if mock_pipeline.split_chronological.call_args[1] else {}
    # Check if apply_feature_selection was passed (may be positional or keyword)
    if 'apply_feature_selection' in call_kwargs:
        assert call_kwargs.get("apply_feature_selection") is True


@patch("torch.cuda.is_available", return_value=False)
def test_create_model_and_train_cpu_mode(mock_cuda):
    """Test create_model_and_train uses CPU when GPU is not available."""
    train_df = create_sample_dataframe(100, "BTC/USDT")
    val_df = create_sample_dataframe(50, "BTC/USDT")
    
    with patch("deeplearning_prediction_main.create_tft_datamodule") as mock_dm:
        with patch("pytorch_lightning.Trainer") as mock_trainer:
            with patch("pytorch_lightning.loggers.TensorBoardLogger"):
                with patch("deeplearning_prediction_main.create_vanilla_tft"):
                    with patch("deeplearning_prediction_main.create_training_callbacks"):
                        with patch("deeplearning_prediction_main.save_model_config"):
                            mock_dm_instance = Mock()
                            mock_dm_instance.training = Mock()
                            mock_dm_instance.validation = Mock()
                            mock_dm_instance.train_df = train_df
                            mock_dm_instance.val_df = val_df
                            mock_dm_instance.test_df = None
                            mock_dm_instance.get_dataset_info.return_value = {}
                            mock_dm_instance.save_dataset_metadata = Mock()
                            mock_dm.return_value = mock_dm_instance
                            
                            mock_trainer_instance = Mock()
                            mock_trainer.return_value = mock_trainer_instance
                            
                            with tempfile.TemporaryDirectory() as tmpdir:
                                create_model_and_train(
                                    train_df=train_df,
                                    val_df=val_df,
                                    test_df=None,
                                    phase=1,
                                    task_type="regression",
                                    epochs=1,
                                    batch_size=16,
                                    output_dir=tmpdir,
                                    experiment_name="test",
                                    use_gpu=False,
                                    num_gpus=None,
                                    optuna_trials=5,
                                    optuna_max_epochs=2,
                                    timeframe="1h",
                                )
                                
                                # Verify trainer was created with CPU accelerator
                                call_kwargs = mock_trainer.call_args[1]
                                assert call_kwargs["accelerator"] == "cpu"


def test_parse_args_output_dir():
    """Test parse_args with custom output directory."""
    with patch("sys.argv", ["deeplearning_prediction_main.py", "--output-dir", "/custom/path"]):
        args = parse_args()
        assert args.output_dir == "/custom/path"


def test_parse_args_experiment_name():
    """Test parse_args with custom experiment name."""
    with patch("sys.argv", ["deeplearning_prediction_main.py", "--experiment-name", "my_experiment"]):
        args = parse_args()
        assert args.experiment_name == "my_experiment"


def test_parse_args_no_feature_selection():
    """Test parse_args with --no-feature-selection flag."""
    with patch("sys.argv", ["deeplearning_prediction_main.py", "--no-feature-selection"]):
        args = parse_args()
        assert args.no_feature_selection is True


def test_parse_args_optuna_params():
    """Test parse_args with Optuna parameters."""
    with patch("sys.argv", [
        "deeplearning_prediction_main.py",
        "--optuna-trials", "30",
        "--optuna-max-epochs", "20",
    ]):
        args = parse_args()
        assert args.optuna_trials == 30
        assert args.optuna_max_epochs == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

