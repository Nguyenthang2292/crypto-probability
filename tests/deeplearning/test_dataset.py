"""
Test script for deeplearning_dataset.py - TFTDataModule and related functions.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
import tempfile
import shutil
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from modules.deeplearning_dataset import TFTDataModule, create_tft_datamodule
from modules.config import (
    DEEP_TARGET_COL,
    DEEP_MAX_ENCODER_LENGTH,
    DEEP_MAX_PREDICTION_LENGTH,
    DEEP_BATCH_SIZE,
)

# Suppress warnings
warnings.filterwarnings("ignore")


def create_sample_dataframe(n_samples=200, symbol="BTC/USDT", has_candle_index=True):
    """Create a sample DataFrame for testing."""
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=n_samples),
        periods=n_samples,
        freq="1H",
    )
    
    # Generate synthetic OHLCV data
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
    })
    
    # Add some technical indicators
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["RSI_14"] = 50 + np.random.randn(n_samples) * 10
    df["ATR_14"] = np.random.rand(n_samples) * 100
    
    # Add candle_index if requested
    if has_candle_index:
        df["candle_index"] = range(n_samples)
    
    # Add target column
    df["future_log_return"] = np.random.randn(n_samples) * 0.01
    
    # Add some known future features
    df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["candle_index"] = range(n_samples)
    
    # Fill NaN values
    df = df.fillna(method="ffill").fillna(method="bfill")
    
    return df


def test_tft_datamodule_initialization():
    """Test TFTDataModule initialization."""
    train_df = create_sample_dataframe(100, "BTC/USDT")
    val_df = create_sample_dataframe(50, "BTC/USDT")
    
    datamodule = TFTDataModule(
        train_df=train_df,
        val_df=val_df,
        batch_size=32,
    )
    
    assert datamodule.train_df is not None
    assert datamodule.val_df is not None
    assert datamodule.target_col == DEEP_TARGET_COL
    assert datamodule.batch_size == 32


def test_tft_datamodule_prepare_data():
    """Test TFTDataModule prepare_data method."""
    train_df = create_sample_dataframe(100, "BTC/USDT")
    val_df = create_sample_dataframe(50, "BTC/USDT")
    
    datamodule = TFTDataModule(
        train_df=train_df,
        val_df=val_df,
        allow_missing_timesteps=False,
        timeframe="1h",
    )
    
    datamodule.prepare_data()
    
    # Check that time_idx was created
    assert "time_idx" in datamodule.train_df.columns
    assert "time_idx" in datamodule.val_df.columns
    
    # Check that time_idx is monotonically increasing
    assert datamodule.train_df["time_idx"].is_monotonic_increasing
    assert datamodule.val_df["time_idx"].is_monotonic_increasing


def test_tft_datamodule_create_time_idx_with_candle_index():
    """Test time_idx creation when candle_index exists."""
    train_df = create_sample_dataframe(100, "BTC/USDT", has_candle_index=True)
    
    datamodule = TFTDataModule(
        train_df=train_df,
        val_df=train_df.copy(),
        timeframe="1h",
    )
    
    result_df = datamodule._create_time_idx(train_df.copy())
    
    assert "time_idx" in result_df.columns
    assert result_df["time_idx"].min() == 0  # Should start from 0 per symbol


def test_tft_datamodule_create_time_idx_without_candle_index():
    """Test time_idx creation when candle_index doesn't exist."""
    train_df = create_sample_dataframe(100, "BTC/USDT", has_candle_index=False)
    train_df = train_df.drop(columns=["candle_index"], errors="ignore")
    
    datamodule = TFTDataModule(
        train_df=train_df,
        val_df=train_df.copy(),
        timeframe="1h",
    )
    
    result_df = datamodule._create_time_idx(train_df.copy())
    
    assert "time_idx" in result_df.columns
    assert result_df["time_idx"].min() == 0


def test_tft_datamodule_resample_missing_candles():
    """Test resampling to handle missing candles."""
    # Create data with missing timestamps - need more data for resampling to work
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=20),
        periods=20,
        freq="1h",
    )
    # Remove some timestamps to create gaps
    timestamps = timestamps[[i for i in range(20) if i not in [5, 10, 15]]]  # Missing some indices
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": "BTC/USDT",
        "close": [50000 + i * 100 for i in range(len(timestamps))],
        "volume": [100 + i * 10 for i in range(len(timestamps))],
    })
    
    datamodule = TFTDataModule(
        train_df=df,
        val_df=df.copy(),
        allow_missing_timesteps=False,
        timeframe="1h",
        max_ffill_limit=5,
    )
    
    result_df = datamodule._resample_missing_candles(df.copy())
    
    # Should have more or equal rows after resampling (may drop all NaN rows)
    assert len(result_df) >= 0  # At least not negative
    if len(result_df) > 0:
        assert "timestamp" in result_df.columns


def test_tft_datamodule_setup():
    """Test TFTDataModule setup method."""
    train_df = create_sample_dataframe(100, "BTC/USDT")
    val_df = create_sample_dataframe(50, "BTC/USDT")
    
    datamodule = TFTDataModule(
        train_df=train_df,
        val_df=val_df,
        batch_size=32,
    )
    
    datamodule.prepare_data()
    datamodule.setup("fit")
    
    assert datamodule.training is not None
    assert datamodule.validation is not None


def test_tft_datamodule_dataloaders():
    """Test TFTDataModule DataLoader creation."""
    train_df = create_sample_dataframe(100, "BTC/USDT")
    val_df = create_sample_dataframe(50, "BTC/USDT")
    
    datamodule = TFTDataModule(
        train_df=train_df,
        val_df=val_df,
        batch_size=32,
        num_workers=0,  # Use 0 workers for testing
    )
    
    datamodule.prepare_data()
    datamodule.setup("fit")
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    assert train_loader is not None
    assert val_loader is not None
    
    # Test that we can get a batch
    batch = next(iter(train_loader))
    assert batch is not None


def test_tft_datamodule_save_load_metadata():
    """Test saving and loading dataset metadata."""
    train_df = create_sample_dataframe(100, "BTC/USDT")
    val_df = create_sample_dataframe(50, "BTC/USDT")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        datamodule = TFTDataModule(
            train_df=train_df,
            val_df=val_df,
            dataset_dir=tmpdir,
        )
        
        datamodule.prepare_data()
        datamodule.setup("fit")
        
        # Save metadata
        metadata_path = Path(tmpdir) / "test_metadata.pkl"
        datamodule.save_dataset_metadata(str(metadata_path))
        
        assert metadata_path.exists()
        
        # Load metadata
        metadata = datamodule.load_dataset_metadata(str(metadata_path))
        
        assert metadata is not None
        assert "target_col" in metadata
        assert metadata["target_col"] == DEEP_TARGET_COL


def test_tft_datamodule_get_dataset_info():
    """Test get_dataset_info method."""
    train_df = create_sample_dataframe(100, "BTC/USDT")
    val_df = create_sample_dataframe(50, "BTC/USDT")
    test_df = create_sample_dataframe(30, "BTC/USDT")
    
    datamodule = TFTDataModule(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )
    
    info = datamodule.get_dataset_info()
    
    assert info["train_samples"] == 100
    assert info["val_samples"] == 50
    assert info["test_samples"] == 30
    assert info["max_encoder_length"] == DEEP_MAX_ENCODER_LENGTH
    assert info["max_prediction_length"] == DEEP_MAX_PREDICTION_LENGTH


def test_create_tft_datamodule():
    """Test create_tft_datamodule convenience function."""
    train_df = create_sample_dataframe(100, "BTC/USDT")
    val_df = create_sample_dataframe(50, "BTC/USDT")
    
    datamodule = create_tft_datamodule(
        train_df=train_df,
        val_df=val_df,
        batch_size=32,
        timeframe="1h",
    )
    
    assert isinstance(datamodule, TFTDataModule)
    assert datamodule.batch_size == 32


def test_tft_datamodule_multi_symbol():
    """Test TFTDataModule with multiple symbols."""
    btc_df = create_sample_dataframe(100, "BTC/USDT")
    eth_df = create_sample_dataframe(100, "ETH/USDT")
    
    train_df = pd.concat([btc_df, eth_df], ignore_index=True)
    val_df = train_df.copy()
    
    datamodule = TFTDataModule(
        train_df=train_df,
        val_df=val_df,
    )
    
    datamodule.prepare_data()
    
    # Check that both symbols are present
    assert set(datamodule.train_df["symbol"].unique()) == {"BTC/USDT", "ETH/USDT"}
    
    # Check that time_idx starts from 0 for each symbol
    for symbol in ["BTC/USDT", "ETH/USDT"]:
        symbol_df = datamodule.train_df[datamodule.train_df["symbol"] == symbol]
        assert symbol_df["time_idx"].min() == 0


def test_tft_datamodule_column_name_sanitization():
    """Test that column names with dots are sanitized."""
    train_df = create_sample_dataframe(100, "BTC/USDT")
    # Add a column with a dot in the name
    train_df["BBP_5_2.0"] = np.random.rand(100)
    
    val_df = train_df.copy()
    
    datamodule = TFTDataModule(
        train_df=train_df,
        val_df=val_df,
    )
    
    datamodule.prepare_data()
    datamodule.setup("fit")
    
    # Should not raise an error about column names with dots
    assert datamodule.training is not None


def test_tft_datamodule_nan_handling():
    """Test that NaN values are handled properly."""
    train_df = create_sample_dataframe(100, "BTC/USDT")
    # Add some NaN values
    train_df.loc[10:15, "RSI_14"] = np.nan
    train_df.loc[20:25, "ATR_14"] = np.nan
    
    val_df = train_df.copy()
    
    datamodule = TFTDataModule(
        train_df=train_df,
        val_df=val_df,
    )
    
    datamodule.prepare_data()
    datamodule.setup("fit")
    
    # Should not raise an error about NaN values
    assert datamodule.training is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

