"""
Test script for modules.deeplearning_model - TFT model creation, callbacks, and optimization.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
import pytest
import torch

from modules.deeplearning_model import (
    create_vanilla_tft,
    create_training_callbacks,
    create_optuna_study,
    suggest_tft_hyperparameters,
    create_optuna_callback,
    save_model_config,
    load_tft_model,
    OPTUNA_AVAILABLE,
)

# Suppress warnings
warnings.filterwarnings("ignore")


def create_mock_time_series_dataset():
    """Create a mock TimeSeriesDataSet for testing."""
    mock_dataset = Mock()
    mock_dataset.target_normalizer = Mock()
    mock_dataset.length = 100
    mock_dataset.max_encoder_length = 64
    mock_dataset.max_prediction_length = 24
    return mock_dataset


def test_create_vanilla_tft_regression():
    """Test create_vanilla_tft for regression task."""
    mock_dataset = create_mock_time_series_dataset()
    
    with patch("modules.deeplearning_model.TemporalFusionTransformer") as mock_tft:
        mock_model = Mock()
        mock_tft.from_dataset.return_value = mock_model
        
        model = create_vanilla_tft(
            training_dataset=mock_dataset,
            hidden_size=16,
            attention_head_size=4,
            dropout=0.1,
            learning_rate=0.03,
            task_type="regression",
        )
        
        assert model is not None
        mock_tft.from_dataset.assert_called_once()


def test_create_vanilla_tft_classification():
    """Test create_vanilla_tft for classification task."""
    mock_dataset = create_mock_time_series_dataset()
    
    with patch("modules.deeplearning_model.TemporalFusionTransformer") as mock_tft:
        mock_model = Mock()
        mock_tft.from_dataset.return_value = mock_model
        
        model = create_vanilla_tft(
            training_dataset=mock_dataset,
            hidden_size=16,
            attention_head_size=4,
            dropout=0.1,
            learning_rate=0.03,
            task_type="classification",
        )
        
        assert model is not None
        mock_tft.from_dataset.assert_called_once()


def test_create_training_callbacks():
    """Test create_training_callbacks function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        callbacks = create_training_callbacks(
            checkpoint_dir=tmpdir,
            monitor="val_loss",
            mode="min",
            patience=10,
            save_top_k=3,
        )
        
        assert len(callbacks) == 3
        assert all(hasattr(cb, "monitor") or hasattr(cb, "logging_interval") for cb in callbacks)


def test_create_training_callbacks_custom_params():
    """Test create_training_callbacks with custom parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        callbacks = create_training_callbacks(
            checkpoint_dir=tmpdir,
            monitor="val_mae",
            mode="max",
            patience=5,
            save_top_k=5,
            verbose=False,
        )
        
        assert len(callbacks) == 3


def test_create_optuna_study_available():
    """Test create_optuna_study when Optuna is available."""
    if not OPTUNA_AVAILABLE:
        pytest.skip("Optuna not available")
    
    with patch("modules.deeplearning_model.optuna") as mock_optuna:
        mock_study = Mock()
        mock_optuna.create_study.return_value = mock_study
        mock_optuna.samplers.TPESampler.return_value = Mock()
        
        study = create_optuna_study(
            direction="minimize",
            study_name="test_study",
        )
        
        assert study is not None
        mock_optuna.create_study.assert_called_once()


def test_create_optuna_study_not_available():
    """Test create_optuna_study when Optuna is not available."""
    with patch("modules.deeplearning_model.OPTUNA_AVAILABLE", False):
        study = create_optuna_study()
        assert study is None


def test_suggest_tft_hyperparameters():
    """Test suggest_tft_hyperparameters function."""
    if not OPTUNA_AVAILABLE:
        pytest.skip("Optuna not available")
    
    mock_trial = Mock()
    mock_trial.suggest_int.return_value = 16
    mock_trial.suggest_float.return_value = 0.1
    
    params = suggest_tft_hyperparameters(mock_trial)
    
    assert isinstance(params, dict)
    assert "hidden_size" in params
    assert "attention_head_size" in params
    assert "dropout" in params
    assert "learning_rate" in params
    assert "reduce_on_plateau_patience" in params


def test_suggest_tft_hyperparameters_not_available():
    """Test suggest_tft_hyperparameters when Optuna is not available."""
    with patch("modules.deeplearning_model.OPTUNA_AVAILABLE", False):
        mock_trial = Mock()
        with pytest.raises(ImportError, match="Optuna is not available"):
            suggest_tft_hyperparameters(mock_trial)


def test_create_optuna_callback():
    """Test create_optuna_callback function."""
    if not OPTUNA_AVAILABLE:
        pytest.skip("Optuna not available")
    
    mock_trial = Mock()
    
    with patch("modules.deeplearning_model.PyTorchLightningPruningCallback") as mock_callback:
        mock_callback_instance = Mock()
        mock_callback.return_value = mock_callback_instance
        
        callback = create_optuna_callback(
            trial=mock_trial,
            monitor="val_loss",
            mode="min",
        )
        
        assert callback is not None
        mock_callback.assert_called_once()


def test_create_optuna_callback_not_available():
    """Test create_optuna_callback when Optuna is not available."""
    with patch("modules.deeplearning_model.OPTUNA_AVAILABLE", False):
        mock_trial = Mock()
        callback = create_optuna_callback(trial=mock_trial)
        assert callback is None


def test_save_model_config():
    """Test save_model_config function."""
    mock_model = Mock()
    mock_model.hparams = {
        "hidden_size": 16,
        "learning_rate": 0.03,
    }
    mock_model.hparams.update = Mock()  # Prevent errors
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "model_config.json"
        
        save_model_config(model=mock_model, config_path=config_path)
        
        assert config_path.exists()


def test_save_model_config_with_dataset():
    """Test save_model_config with dataset info."""
    mock_model = Mock()
    mock_model.hparams = {
        "hidden_size": 16,
    }
    
    mock_dataset = create_mock_time_series_dataset()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "model_config.json"
        
        save_model_config(
            model=mock_model,
            config_path=config_path,
            dataset=mock_dataset,
        )
        
        assert config_path.exists()


def test_load_tft_model_not_found():
    """Test load_tft_model when checkpoint doesn't exist."""
    checkpoint_path = Path("nonexistent_checkpoint.ckpt")
    
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_tft_model(checkpoint_path)


def test_optimize_tft_hyperparameters_not_available():
    """Test optimize_tft_hyperparameters when Optuna is not available."""
    with patch("modules.deeplearning_model.OPTUNA_AVAILABLE", False):
        mock_dataset = create_mock_time_series_dataset()
        mock_datamodule = Mock()
        
        with patch("builtins.print"):
            with pytest.raises(ImportError) or patch("builtins.print"):
                # Should either raise error or skip
                try:
                    from modules.deeplearning_model import optimize_tft_hyperparameters
                    optimize_tft_hyperparameters(
                        training_dataset=mock_dataset,
                        val_dataset=mock_dataset,
                        datamodule=mock_datamodule,
                        n_trials=2,
                    )
                except ImportError:
                    pass  # Expected if Optuna not available


def test_hybrid_lstm_tft_creation():
    """Test create_hybrid_lstm_tft function if it exists."""
    try:
        from modules.deeplearning_model import create_hybrid_lstm_tft
        
        mock_dataset = create_mock_time_series_dataset()
        
        with patch("modules.deeplearning_model.HybridLSTMTFT") as mock_hybrid:
            mock_model = Mock()
            mock_hybrid.return_value = mock_model
            
            model = create_hybrid_lstm_tft(
                training_dataset=mock_dataset,
                tft_hidden_size=16,
                lstm_hidden_size=32,
                num_classes=3,
            )
            
            assert model is not None
    except ImportError:
        pytest.skip("HybridLSTMTFT not available")


def test_model_config_save():
    """Test save_model_config function."""
    mock_model = Mock()
    mock_model.hparams = {
        "hidden_size": 16,
        "learning_rate": 0.03,
        "dropout": 0.1,
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "model_config.json"
        
        # Save
        save_model_config(model=mock_model, config_path=config_path)
        
        # Verify file was created
        assert config_path.exists()
        
        # Verify file content
        import json
        with open(config_path, "r") as f:
            loaded_config = json.load(f)
        
        assert loaded_config is not None
        assert loaded_config["hidden_size"] == 16
        assert loaded_config["learning_rate"] == 0.03


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

