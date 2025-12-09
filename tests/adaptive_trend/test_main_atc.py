"""
Tests for main_atc.py

Tests all functionality including:
- Mode determination
- Auto mode execution
- Manual mode execution
- Configuration display
- Symbol input handling
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import warnings
import pandas as pd
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, Mock
from colorama import Fore
from io import StringIO
import contextlib

from modules.common.ExchangeManager import ExchangeManager
from modules.common.DataFetcher import DataFetcher

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


# ============================================================================
# Tests for ATCAnalyzer class
# ============================================================================

def test_atc_analyzer_init():
    """Test ATCAnalyzer initialization."""
    from main_atc import ATCAnalyzer
    
    args = SimpleNamespace(
        timeframe="1h",
        limit=1500,
    )
    mock_data_fetcher = MagicMock()
    
    analyzer = ATCAnalyzer(args, mock_data_fetcher)
    
    assert analyzer.args == args
    assert analyzer.data_fetcher == mock_data_fetcher
    assert analyzer.selected_timeframe == "1h"
    assert analyzer.mode == "manual"
    assert analyzer._atc_params is None


def test_get_atc_params():
    """Test get_atc_params extracts and caches parameters correctly."""
    from main_atc import ATCAnalyzer
    
    args = SimpleNamespace(
        timeframe="1h",
        limit=1500,
        ema_len=28,
        hma_len=28,
        wma_len=28,
        dema_len=28,
        lsma_len=28,
        kama_len=28,
        robustness="Medium",
        lambda_param=0.02,
        decay=0.03,
        cutout=0,
    )
    mock_data_fetcher = MagicMock()
    
    analyzer = ATCAnalyzer(args, mock_data_fetcher)
    params = analyzer.get_atc_params()
    
    assert params["limit"] == 1500
    assert params["ema_len"] == 28
    assert params["robustness"] == "Medium"
    assert params["lambda_param"] == 0.02
    assert params["decay"] == 0.03
    
    # Test caching
    params2 = analyzer.get_atc_params()
    assert params is params2  # Same object (cached)


def test_determine_mode_and_timeframe_auto():
    """Test determine_mode_and_timeframe with auto flag."""
    from main_atc import ATCAnalyzer
    
    args = SimpleNamespace(
        auto=True,
        no_menu=False,
        timeframe="1h",
    )
    mock_data_fetcher = MagicMock()
    
    analyzer = ATCAnalyzer(args, mock_data_fetcher)
    mode, timeframe = analyzer.determine_mode_and_timeframe()
    
    assert mode == "auto"
    assert timeframe == "1h"
    assert analyzer.mode == "auto"
    assert analyzer.selected_timeframe == "1h"


def test_determine_mode_and_timeframe_manual():
    """Test determine_mode_and_timeframe with manual mode."""
    from main_atc import ATCAnalyzer
    
    args = SimpleNamespace(
        auto=False,
        no_menu=True,
        timeframe="4h",
    )
    mock_data_fetcher = MagicMock()
    
    analyzer = ATCAnalyzer(args, mock_data_fetcher)
    mode, timeframe = analyzer.determine_mode_and_timeframe()
    
    assert mode == "manual"
    assert timeframe == "4h"


@patch('main_atc.prompt_interactive_mode')
def test_determine_mode_and_timeframe_interactive(mock_prompt):
    """Test determine_mode_and_timeframe with interactive menu."""
    from main_atc import ATCAnalyzer
    
    args = SimpleNamespace(
        auto=False,
        no_menu=False,
        timeframe="1h",
    )
    mock_data_fetcher = MagicMock()
    
    mock_prompt.return_value = {
        "mode": "auto",
        "timeframe": "2h"
    }
    
    analyzer = ATCAnalyzer(args, mock_data_fetcher)
    mode, timeframe = analyzer.determine_mode_and_timeframe()
    
    assert mode == "auto"
    assert timeframe == "2h"
    mock_prompt.assert_called_once()


def test_get_symbol_input_from_args():
    """Test get_symbol_input with symbol in args."""
    from main_atc import ATCAnalyzer
    
    args = SimpleNamespace(
        timeframe="1h",
        symbol="ETH/USDT",
        quote="USDT",
        no_prompt=True,
    )
    mock_data_fetcher = MagicMock()
    
    analyzer = ATCAnalyzer(args, mock_data_fetcher)
    symbol = analyzer.get_symbol_input()
    
    assert symbol == "ETH/USDT"


@patch('builtins.input', return_value='BTC/USDT')
def test_get_symbol_input_from_prompt(mock_input):
    """Test get_symbol_input with user prompt."""
    from main_atc import ATCAnalyzer
    
    args = SimpleNamespace(
        timeframe="1h",
        symbol=None,
        quote="USDT",
        no_prompt=False,
    )
    mock_data_fetcher = MagicMock()
    
    analyzer = ATCAnalyzer(args, mock_data_fetcher)
    symbol = analyzer.get_symbol_input()
    
    assert symbol == "BTC/USDT"
    mock_input.assert_called_once()


def test_display_auto_mode_config():
    """Test display_auto_mode_config displays configuration."""
    from main_atc import ATCAnalyzer
    
    args = SimpleNamespace(
        limit=1500,
        robustness="Medium",
        ema_len=28,
        hma_len=28,
        wma_len=28,
        dema_len=28,
        lsma_len=28,
        kama_len=28,
        lambda_param=0.02,
        decay=0.03,
        cutout=0,
        min_signal=0.01,
        max_symbols=None,
        timeframe="1h",
    )
    mock_data_fetcher = MagicMock()
    
    analyzer = ATCAnalyzer(args, mock_data_fetcher)
    analyzer.selected_timeframe = "1h"
    
    # Should not raise exception
    try:
        analyzer.display_auto_mode_config()
        assert True
    except Exception as e:
        assert False, f"display_auto_mode_config raised exception: {e}"


def test_display_manual_mode_config():
    """Test display_manual_mode_config displays configuration."""
    from main_atc import ATCAnalyzer
    
    args = SimpleNamespace(
        limit=1500,
        robustness="Medium",
        ema_len=28,
        hma_len=28,
        wma_len=28,
        dema_len=28,
        lsma_len=28,
        kama_len=28,
        lambda_param=0.02,
        decay=0.03,
        cutout=0,
        timeframe="1h",
    )
    mock_data_fetcher = MagicMock()
    
    analyzer = ATCAnalyzer(args, mock_data_fetcher)
    analyzer.selected_timeframe = "1h"
    
    # Should not raise exception
    try:
        analyzer.display_manual_mode_config("BTC/USDT")
        assert True
    except Exception as e:
        assert False, f"display_manual_mode_config raised exception: {e}"


# ============================================================================
# Tests for main function
# ============================================================================

@patch('main_atc.list_futures_symbols')
@patch('main_atc.ExchangeManager')
@patch('main_atc.DataFetcher')
@patch('main_atc.parse_args')
def test_main_list_symbols(mock_parse, mock_data_fetcher, mock_exchange, mock_list):
    """Test main function with --list-symbols flag."""
    from main_atc import main
    
    args = SimpleNamespace(list_symbols=True)
    mock_parse.return_value = args
    
    mock_exchange_instance = MagicMock()
    mock_exchange.return_value = mock_exchange_instance
    
    mock_fetcher_instance = MagicMock()
    mock_data_fetcher.return_value = mock_fetcher_instance
    
    main()
    
    mock_list.assert_called_once_with(mock_fetcher_instance)


@patch('main_atc.ATCAnalyzer.run_auto_mode')
@patch('main_atc.ATCAnalyzer.determine_mode_and_timeframe')
@patch('main_atc.ATCAnalyzer')
@patch('main_atc.ExchangeManager')
@patch('main_atc.DataFetcher')
@patch('main_atc.parse_args')
def test_main_auto_mode(mock_parse, mock_data_fetcher, mock_exchange, mock_analyzer_class, mock_determine, mock_run_auto):
    """Test main function with auto mode."""
    from main_atc import main
    
    args = SimpleNamespace(
        list_symbols=False,
        auto=True,
        timeframe="1h",
    )
    mock_parse.return_value = args
    
    mock_exchange_instance = MagicMock()
    mock_exchange.return_value = mock_exchange_instance
    
    mock_fetcher_instance = MagicMock()
    mock_data_fetcher.return_value = mock_fetcher_instance
    
    mock_analyzer = MagicMock()
    mock_analyzer_class.return_value = mock_analyzer
    mock_analyzer.determine_mode_and_timeframe.return_value = ("auto", "1h")
    
    main()
    
    mock_analyzer_class.assert_called_once_with(args, mock_fetcher_instance)
    mock_analyzer.run_auto_mode.assert_called_once()


@patch('main_atc.ATCAnalyzer.run_manual_mode')
@patch('main_atc.ATCAnalyzer.determine_mode_and_timeframe')
@patch('main_atc.ATCAnalyzer')
@patch('main_atc.ExchangeManager')
@patch('main_atc.DataFetcher')
@patch('main_atc.parse_args')
def test_main_manual_mode(mock_parse, mock_data_fetcher, mock_exchange, mock_analyzer_class, mock_determine, mock_run_manual):
    """Test main function with manual mode."""
    from main_atc import main
    
    args = SimpleNamespace(
        list_symbols=False,
        auto=False,
        timeframe="1h",
    )
    mock_parse.return_value = args
    
    mock_exchange_instance = MagicMock()
    mock_exchange.return_value = mock_exchange_instance
    
    mock_fetcher_instance = MagicMock()
    mock_data_fetcher.return_value = mock_fetcher_instance
    
    mock_analyzer = MagicMock()
    mock_analyzer_class.return_value = mock_analyzer
    mock_analyzer.determine_mode_and_timeframe.return_value = ("manual", "1h")
    
    main()
    
    mock_analyzer_class.assert_called_once_with(args, mock_fetcher_instance)
    mock_analyzer.run_manual_mode.assert_called_once()


# ============================================================================
# Tests for run methods
# ============================================================================

@patch('main_atc.display_scan_results')
@patch('main_atc.scan_all_symbols')
@patch('main_atc.ATCAnalyzer.display_auto_mode_config')
def test_run_auto_mode(mock_display_config, mock_scan, mock_display_results):
    """Test run_auto_mode executes correctly."""
    from main_atc import ATCAnalyzer
    
    args = SimpleNamespace(
        limit=1500,
        ema_len=28,
        hma_len=28,
        wma_len=28,
        dema_len=28,
        lsma_len=28,
        kama_len=28,
        robustness="Medium",
        lambda_param=0.02,
        decay=0.03,
        cutout=0,
        max_symbols=None,
        min_signal=0.01,
        timeframe="1h",
    )
    
    mock_data_fetcher = MagicMock()
    long_signals = pd.DataFrame({"symbol": ["BTC/USDT"], "signal": [0.05]})
    short_signals = pd.DataFrame({"symbol": ["ETH/USDT"], "signal": [-0.03]})
    
    mock_scan.return_value = (long_signals, short_signals)
    
    analyzer = ATCAnalyzer(args, mock_data_fetcher)
    analyzer.selected_timeframe = "1h"
    analyzer.run_auto_mode()
    
    mock_display_config.assert_called_once()
    mock_scan.assert_called_once()
    mock_display_results.assert_called_once_with(long_signals, short_signals, 0.01)


@patch('main_atc.ATCAnalyzer.run_interactive_loop')
@patch('main_atc.analyze_symbol')
@patch('main_atc.ATCAnalyzer.display_manual_mode_config')
@patch('main_atc.ATCAnalyzer.get_symbol_input')
def test_run_manual_mode_success(mock_get_symbol, mock_display_config, mock_analyze, mock_interactive):
    """Test run_manual_mode with successful analysis."""
    from main_atc import ATCAnalyzer
    
    args = SimpleNamespace(
        no_prompt=False,
        quote="USDT",
        limit=1500,
        ema_len=28,
        hma_len=28,
        wma_len=28,
        dema_len=28,
        lsma_len=28,
        kama_len=28,
        robustness="Medium",
        lambda_param=0.02,
        decay=0.03,
        cutout=0,
        timeframe="1h",
    )
    
    mock_data_fetcher = MagicMock()
    analyzer = ATCAnalyzer(args, mock_data_fetcher)
    analyzer.selected_timeframe = "1h"
    
    mock_get_symbol.return_value = "BTC/USDT"
    mock_analyze.return_value = {
        "symbol": "BTC/USDT",
        "df": pd.DataFrame(),
        "atc_results": {},
        "current_price": 50000.0,
        "exchange_label": "binance",
    }
    
    analyzer.run_manual_mode()
    
    mock_get_symbol.assert_called_once()
    mock_display_config.assert_called_once()
    mock_analyze.assert_called_once()
    mock_interactive.assert_called_once()


@patch('main_atc.log_error')
@patch('main_atc.analyze_symbol')
@patch('main_atc.ATCAnalyzer.display_manual_mode_config')
@patch('main_atc.ATCAnalyzer.get_symbol_input')
def test_run_manual_mode_failure(mock_get_symbol, mock_display_config, mock_analyze, mock_log_error):
    """Test run_manual_mode with failed analysis."""
    from main_atc import ATCAnalyzer
    
    args = SimpleNamespace(
        no_prompt=False,
        quote="USDT",
        limit=1500,
        ema_len=28,
        hma_len=28,
        wma_len=28,
        dema_len=28,
        lsma_len=28,
        kama_len=28,
        robustness="Medium",
        lambda_param=0.02,
        decay=0.03,
        cutout=0,
        timeframe="1h",
    )
    
    mock_data_fetcher = MagicMock()
    analyzer = ATCAnalyzer(args, mock_data_fetcher)
    analyzer.selected_timeframe = "1h"
    
    mock_get_symbol.return_value = "BTC/USDT"
    mock_analyze.return_value = None  # Analysis failed
    
    analyzer.run_manual_mode()
    
    mock_log_error.assert_called_once_with("Analysis failed")


@patch('main_atc.analyze_symbol')
@patch('builtins.input', side_effect=['ETH/USDT', 'BTC/USDT', KeyboardInterrupt])
def test_run_interactive_loop(mock_input, mock_analyze):
    """Test run_interactive_loop handles multiple symbols."""
    from main_atc import ATCAnalyzer
    
    args = SimpleNamespace(timeframe="1h")
    mock_data_fetcher = MagicMock()
    analyzer = ATCAnalyzer(args, mock_data_fetcher)
    analyzer.selected_timeframe = "1h"
    
    atc_params = {
        "limit": 1500,
        "ema_len": 28,
        "hma_len": 28,
        "wma_len": 28,
        "dema_len": 28,
        "lsma_len": 28,
        "kama_len": 28,
        "robustness": "Medium",
        "lambda_param": 0.02,
        "decay": 0.03,
        "cutout": 0,
    }
    
    try:
        analyzer.run_interactive_loop(
            symbol="BTC/USDT",
            quote="USDT",
            atc_params=atc_params,
        )
    except KeyboardInterrupt:
        pass  # Expected
    
    # Should have called analyze_symbol for each input
    assert mock_analyze.call_count >= 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

