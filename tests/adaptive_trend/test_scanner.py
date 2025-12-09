"""
Tests for scanner module.
"""
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import warnings
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, Mock
from types import SimpleNamespace

from modules.adaptive_trend.core.scanner import scan_all_symbols
from modules.adaptive_trend.utils.config import ATCConfig

# Suppress warnings
warnings.filterwarnings("ignore")


def _create_mock_ohlcv_data(
    start_price: float = 100.0,
    num_candles: int = 200,
) -> pd.DataFrame:
    """Create mock OHLCV DataFrame for testing."""
    timestamps = pd.date_range(
        start="2024-01-01", periods=num_candles, freq="1h", tz="UTC"
    )

    np.random.seed(42)
    returns = np.random.normal(0, 0.01, num_candles)
    prices = start_price * (1 + returns).cumprod()

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices * (1 + np.random.normal(0, 0.001, num_candles)),
            "high": prices * (1 + abs(np.random.normal(0, 0.002, num_candles))),
            "low": prices * (1 - abs(np.random.normal(0, 0.002, num_candles))),
            "close": prices,
            "volume": np.random.uniform(10000, 100000, num_candles),
        }
    )

    return df


def _create_mock_atc_results(signal_value: float = 0.05) -> dict:
    """Create mock ATC results."""
    signal_series = pd.Series([0.0, 0.01, signal_value, signal_value])
    return {
        "Average_Signal": signal_series,
        "EMA_Signal": signal_series,
        "HMA_Signal": signal_series,
    }


# ============================================================================
# Tests for ATCConfig validation
# ============================================================================

def test_scan_all_symbols_none_data_fetcher():
    """Test that scan_all_symbols raises error for None data_fetcher."""
    config = ATCConfig()
    
    with pytest.raises(ValueError, match="data_fetcher cannot be None"):
        scan_all_symbols(None, config)


def test_scan_all_symbols_invalid_config_type():
    """Test that scan_all_symbols raises error for invalid config type."""
    mock_fetcher = MagicMock()
    
    with pytest.raises(ValueError, match="atc_config must be an ATCConfig instance"):
        scan_all_symbols(mock_fetcher, "not_a_config")


def test_scan_all_symbols_missing_methods():
    """Test that scan_all_symbols raises error for missing methods."""
    config = ATCConfig()
    mock_fetcher = MagicMock()
    del mock_fetcher.list_binance_futures_symbols
    
    with pytest.raises(AttributeError, match="must have method"):
        scan_all_symbols(mock_fetcher, config)


def test_scan_all_symbols_invalid_timeframe():
    """Test that scan_all_symbols validates timeframe."""
    config = ATCConfig(timeframe="")
    mock_fetcher = MagicMock()
    mock_fetcher.list_binance_futures_symbols = MagicMock(return_value=[])
    
    with pytest.raises(ValueError, match="timeframe must be a non-empty string"):
        scan_all_symbols(mock_fetcher, config)


def test_scan_all_symbols_invalid_limit():
    """Test that scan_all_symbols validates limit."""
    config = ATCConfig(limit=0)
    mock_fetcher = MagicMock()
    mock_fetcher.list_binance_futures_symbols = MagicMock(return_value=[])
    
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        scan_all_symbols(mock_fetcher, config)


def test_scan_all_symbols_invalid_ma_lengths():
    """Test that scan_all_symbols validates MA lengths."""
    config = ATCConfig(ema_len=0)
    mock_fetcher = MagicMock()
    mock_fetcher.list_binance_futures_symbols = MagicMock(return_value=[])
    
    with pytest.raises(ValueError, match="ema_len must be a positive integer"):
        scan_all_symbols(mock_fetcher, config)


def test_scan_all_symbols_invalid_robustness():
    """Test that scan_all_symbols validates robustness."""
    config = ATCConfig(robustness="Invalid")
    mock_fetcher = MagicMock()
    mock_fetcher.list_binance_futures_symbols = MagicMock(return_value=[])
    
    with pytest.raises(ValueError, match="robustness must be one of"):
        scan_all_symbols(mock_fetcher, config)


def test_scan_all_symbols_invalid_execution_mode():
    """Test that scan_all_symbols validates execution_mode."""
    config = ATCConfig()
    mock_fetcher = MagicMock()
    mock_fetcher.list_binance_futures_symbols = MagicMock(return_value=[])
    
    with pytest.raises(ValueError, match="execution_mode must be one of"):
        scan_all_symbols(mock_fetcher, config, execution_mode="invalid")


def test_scan_all_symbols_invalid_max_workers():
    """Test that scan_all_symbols validates max_workers."""
    config = ATCConfig()
    mock_fetcher = MagicMock()
    mock_fetcher.list_binance_futures_symbols = MagicMock(return_value=[])
    
    with pytest.raises(ValueError, match="max_workers must be a positive integer"):
        scan_all_symbols(mock_fetcher, config, max_workers=0)


# ============================================================================
# Tests for scan_all_symbols - No symbols
# ============================================================================

def test_scan_all_symbols_no_symbols():
    """Test that scan_all_symbols returns empty DataFrames when no symbols."""
    config = ATCConfig()
    mock_fetcher = MagicMock()
    mock_fetcher.list_binance_futures_symbols = MagicMock(return_value=[])
    
    long_signals, short_signals = scan_all_symbols(mock_fetcher, config)
    
    assert long_signals.empty
    assert short_signals.empty
    assert isinstance(long_signals, pd.DataFrame)
    assert isinstance(short_signals, pd.DataFrame)


# ============================================================================
# Tests for scan_all_symbols - Sequential mode
# ============================================================================

@patch('modules.adaptive_trend.scanner.compute_atc_signals')
@patch('modules.adaptive_trend.scanner.trend_sign')
def test_scan_all_symbols_sequential_mode(mock_trend_sign, mock_compute_atc):
    """Test scan_all_symbols in sequential mode."""
    config = ATCConfig(limit=200, timeframe="1h")
    mock_fetcher = MagicMock()
    
    # Setup mock data fetcher
    symbols = ["BTC/USDT", "ETH/USDT"]
    mock_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    
    mock_df = _create_mock_ohlcv_data(num_candles=200)
    mock_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(
        return_value=(mock_df, "binance")
    )
    
    # Setup mock ATC results
    mock_compute_atc.return_value = _create_mock_atc_results(signal_value=0.05)
    mock_trend_sign.return_value = pd.Series([0, 0, 1, 1])  # Bullish trend
    
    long_signals, short_signals = scan_all_symbols(
        mock_fetcher, config, execution_mode="sequential", max_symbols=2
    )
    
    assert isinstance(long_signals, pd.DataFrame)
    assert isinstance(short_signals, pd.DataFrame)
    assert len(long_signals) > 0  # Should find signals
    mock_fetcher.fetch_ohlcv_with_fallback_exchange.assert_called()


# ============================================================================
# Tests for scan_all_symbols - ThreadPool mode
# ============================================================================

@patch('modules.adaptive_trend.scanner.compute_atc_signals')
@patch('modules.adaptive_trend.scanner.trend_sign')
def test_scan_all_symbols_threadpool_mode(mock_trend_sign, mock_compute_atc):
    """Test scan_all_symbols in threadpool mode."""
    config = ATCConfig(limit=200, timeframe="1h")
    mock_fetcher = MagicMock()
    
    symbols = ["BTC/USDT", "ETH/USDT"]
    mock_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    
    mock_df = _create_mock_ohlcv_data(num_candles=200)
    mock_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(
        return_value=(mock_df, "binance")
    )
    
    mock_compute_atc.return_value = _create_mock_atc_results(signal_value=0.05)
    mock_trend_sign.return_value = pd.Series([0, 0, 1, 1])
    
    long_signals, short_signals = scan_all_symbols(
        mock_fetcher, config, execution_mode="threadpool", max_symbols=2, max_workers=2
    )
    
    assert isinstance(long_signals, pd.DataFrame)
    assert isinstance(short_signals, pd.DataFrame)


# ============================================================================
# Tests for scan_all_symbols - Asyncio mode
# ============================================================================

@patch('modules.adaptive_trend.scanner.compute_atc_signals')
@patch('modules.adaptive_trend.scanner.trend_sign')
def test_scan_all_symbols_asyncio_mode(mock_trend_sign, mock_compute_atc):
    """Test scan_all_symbols in asyncio mode."""
    config = ATCConfig(limit=200, timeframe="1h")
    mock_fetcher = MagicMock()
    
    symbols = ["BTC/USDT", "ETH/USDT"]
    mock_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    
    mock_df = _create_mock_ohlcv_data(num_candles=200)
    mock_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(
        return_value=(mock_df, "binance")
    )
    
    mock_compute_atc.return_value = _create_mock_atc_results(signal_value=0.05)
    mock_trend_sign.return_value = pd.Series([0, 0, 1, 1])
    
    long_signals, short_signals = scan_all_symbols(
        mock_fetcher, config, execution_mode="asyncio", max_symbols=2, max_workers=2
    )
    
    assert isinstance(long_signals, pd.DataFrame)
    assert isinstance(short_signals, pd.DataFrame)


# ============================================================================
# Tests for scan_all_symbols - Edge cases
# ============================================================================

@patch('modules.adaptive_trend.scanner.compute_atc_signals')
@patch('modules.adaptive_trend.scanner.trend_sign')
def test_scan_all_symbols_empty_dataframe(mock_trend_sign, mock_compute_atc):
    """Test scan_all_symbols handles empty DataFrame."""
    config = ATCConfig(limit=200, timeframe="1h")
    mock_fetcher = MagicMock()
    
    symbols = ["BTC/USDT"]
    mock_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    mock_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(
        return_value=(pd.DataFrame(), "binance")
    )
    
    long_signals, short_signals = scan_all_symbols(
        mock_fetcher, config, execution_mode="sequential", max_symbols=1
    )
    
    assert long_signals.empty
    assert short_signals.empty


@patch('modules.adaptive_trend.scanner.compute_atc_signals')
@patch('modules.adaptive_trend.scanner.trend_sign')
def test_scan_all_symbols_missing_close_column(mock_trend_sign, mock_compute_atc):
    """Test scan_all_symbols handles missing close column."""
    config = ATCConfig(limit=200, timeframe="1h")
    mock_fetcher = MagicMock()
    
    symbols = ["BTC/USDT"]
    mock_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    
    # DataFrame without 'close' column
    mock_df = pd.DataFrame({"open": [100.0], "high": [101.0]})
    mock_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(
        return_value=(mock_df, "binance")
    )
    
    long_signals, short_signals = scan_all_symbols(
        mock_fetcher, config, execution_mode="sequential", max_symbols=1
    )
    
    assert long_signals.empty
    assert short_signals.empty


@patch('modules.adaptive_trend.scanner.compute_atc_signals')
@patch('modules.adaptive_trend.scanner.trend_sign')
def test_scan_all_symbols_insufficient_data(mock_trend_sign, mock_compute_atc):
    """Test scan_all_symbols handles insufficient data."""
    config = ATCConfig(limit=200, timeframe="1h")
    mock_fetcher = MagicMock()
    
    symbols = ["BTC/USDT"]
    mock_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    
    # DataFrame with less data than required
    mock_df = _create_mock_ohlcv_data(num_candles=50)  # Less than limit=200
    mock_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(
        return_value=(mock_df, "binance")
    )
    
    long_signals, short_signals = scan_all_symbols(
        mock_fetcher, config, execution_mode="sequential", max_symbols=1
    )
    
    assert long_signals.empty
    assert short_signals.empty


@patch('modules.adaptive_trend.scanner.compute_atc_signals')
@patch('modules.adaptive_trend.scanner.trend_sign')
def test_scan_all_symbols_signal_below_threshold(mock_trend_sign, mock_compute_atc):
    """Test scan_all_symbols filters signals below threshold."""
    config = ATCConfig(limit=200, timeframe="1h")
    mock_fetcher = MagicMock()
    
    symbols = ["BTC/USDT"]
    mock_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    
    mock_df = _create_mock_ohlcv_data(num_candles=200)
    mock_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(
        return_value=(mock_df, "binance")
    )
    
    # Signal below threshold (0.005 < 0.01)
    mock_compute_atc.return_value = _create_mock_atc_results(signal_value=0.005)
    mock_trend_sign.return_value = pd.Series([0, 0, 1, 1])
    
    long_signals, short_signals = scan_all_symbols(
        mock_fetcher, config, execution_mode="sequential", max_symbols=1, min_signal=0.01
    )
    
    assert long_signals.empty
    assert short_signals.empty


@patch('modules.adaptive_trend.scanner.compute_atc_signals')
@patch('modules.adaptive_trend.scanner.trend_sign')
def test_scan_all_symbols_long_and_short_signals(mock_trend_sign, mock_compute_atc):
    """Test scan_all_symbols separates LONG and SHORT signals."""
    config = ATCConfig(limit=200, timeframe="1h")
    mock_fetcher = MagicMock()
    
    symbols = ["BTC/USDT", "ETH/USDT"]
    mock_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    
    mock_df = _create_mock_ohlcv_data(num_candles=200)
    mock_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(
        return_value=(mock_df, "binance")
    )
    
    # First symbol: bullish
    # Second symbol: bearish
    def side_effect_compute(*args, **kwargs):
        # Return different signals based on call count
        if not hasattr(side_effect_compute, 'call_count'):
            side_effect_compute.call_count = 0
        side_effect_compute.call_count += 1
        
        if side_effect_compute.call_count == 1:
            return _create_mock_atc_results(signal_value=0.05)  # Bullish
        else:
            return _create_mock_atc_results(signal_value=-0.05)  # Bearish
    
    def side_effect_trend(*args, **kwargs):
        signal = args[0]
        if not hasattr(side_effect_trend, 'call_count'):
            side_effect_trend.call_count = 0
        side_effect_trend.call_count += 1
        
        if side_effect_trend.call_count == 1:
            return pd.Series([0, 0, 1, 1])  # Bullish
        else:
            return pd.Series([0, 0, -1, -1])  # Bearish
    
    mock_compute_atc.side_effect = side_effect_compute
    mock_trend_sign.side_effect = side_effect_trend
    
    long_signals, short_signals = scan_all_symbols(
        mock_fetcher, config, execution_mode="sequential", max_symbols=2
    )
    
    assert len(long_signals) > 0
    assert len(short_signals) > 0
    assert all(long_signals["trend"] > 0)
    assert all(short_signals["trend"] < 0)


@patch('modules.adaptive_trend.scanner.compute_atc_signals')
@patch('modules.adaptive_trend.scanner.trend_sign')
def test_scan_all_symbols_max_symbols_limit(mock_trend_sign, mock_compute_atc):
    """Test scan_all_symbols respects max_symbols limit."""
    config = ATCConfig(limit=200, timeframe="1h")
    mock_fetcher = MagicMock()
    
    all_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
    mock_fetcher.list_binance_futures_symbols = MagicMock(return_value=all_symbols)
    
    mock_df = _create_mock_ohlcv_data(num_candles=200)
    mock_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(
        return_value=(mock_df, "binance")
    )
    
    mock_compute_atc.return_value = _create_mock_atc_results(signal_value=0.05)
    mock_trend_sign.return_value = pd.Series([0, 0, 1, 1])
    
    long_signals, short_signals = scan_all_symbols(
        mock_fetcher, config, execution_mode="sequential", max_symbols=3
    )
    
    # Should only process 3 symbols
    assert mock_fetcher.fetch_ohlcv_with_fallback_exchange.call_count == 3


# ============================================================================
# Tests for error handling
# ============================================================================

@patch('modules.adaptive_trend.scanner.compute_atc_signals')
def test_scan_all_symbols_handles_exceptions(mock_compute_atc):
    """Test scan_all_symbols handles exceptions gracefully."""
    config = ATCConfig(limit=200, timeframe="1h")
    mock_fetcher = MagicMock()
    
    symbols = ["BTC/USDT"]
    mock_fetcher.list_binance_futures_symbols = MagicMock(return_value=symbols)
    
    mock_df = _create_mock_ohlcv_data(num_candles=200)
    mock_fetcher.fetch_ohlcv_with_fallback_exchange = MagicMock(
        return_value=(mock_df, "binance")
    )
    
    # Make compute_atc_signals raise an exception
    mock_compute_atc.side_effect = Exception("Test error")
    
    long_signals, short_signals = scan_all_symbols(
        mock_fetcher, config, execution_mode="sequential", max_symbols=1
    )
    
    # Should return empty DataFrames and continue
    assert isinstance(long_signals, pd.DataFrame)
    assert isinstance(short_signals, pd.DataFrame)

