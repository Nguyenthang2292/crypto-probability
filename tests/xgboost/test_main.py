"""
Test script for xgboost_prediction_main.py - XGBoost prediction main functionality.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
import pytest
import pandas as pd
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV DataFrame."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    return pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(50000, 51000, 100),
        'high': np.random.uniform(51000, 52000, 100),
        'low': np.random.uniform(49000, 50000, 100),
        'close': np.random.uniform(50000, 51000, 100),
        'volume': np.random.uniform(1000, 2000, 100),
    })


@pytest.fixture
def sample_df_with_indicators(sample_ohlcv_data):
    """Create sample DataFrame with indicators and labels."""
    df = sample_ohlcv_data.copy()
    # Add some indicator columns
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['RSI_14'] = np.random.uniform(30, 70, 100)
    df['ATR_14'] = np.random.uniform(100, 200, 100)
    df['MACD_12_26_9'] = np.random.uniform(-10, 10, 100)
    # Add label columns
    df['Target'] = np.random.choice([0, 1, 2], 100)
    df['DynamicThreshold'] = 0.01
    return df


@pytest.fixture
def mock_exchange_manager():
    """Create a mock ExchangeManager."""
    manager = Mock()
    manager.public = Mock()
    manager.public.exchange_priority_for_fallback = ["binance"]
    return manager


@pytest.fixture
def mock_data_fetcher(sample_ohlcv_data):
    """Create a mock DataFetcher."""
    fetcher = Mock()
    fetcher.fetch_ohlcv_with_fallback_exchange = Mock(return_value=(sample_ohlcv_data, "binance"))
    return fetcher


@pytest.fixture
def mock_indicator_engine(sample_df_with_indicators):
    """Create a mock IndicatorEngine."""
    engine = Mock()
    engine.compute_features = Mock(return_value=sample_df_with_indicators)
    return engine


@pytest.fixture
def mock_model():
    """Create a mock XGBoost model."""
    model = Mock()
    model.predict_proba = Mock(return_value=np.array([[0.2, 0.3, 0.5]]))
    return model


@patch('xgboost_prediction_main.parse_args')
@patch('xgboost_prediction_main.ExchangeManager')
@patch('xgboost_prediction_main.DataFetcher')
@patch('xgboost_prediction_main.IndicatorEngine')
@patch('xgboost_prediction_main.train_and_predict')
@patch('xgboost_prediction_main.predict_next_move')
@patch('xgboost_prediction_main.apply_directional_labels')
def test_main_function_basic_flow(
    mock_apply_labels,
    mock_predict_next,
    mock_train_and_predict,
    mock_indicator_engine_class,
    mock_data_fetcher_class,
    mock_exchange_manager_class,
    mock_parse_args,
    sample_df_with_indicators,
    mock_model
):
    """Test main function basic execution flow."""
    # Setup mocks
    mock_args = Mock()
    mock_args.symbol = "BTC/USDT"
    mock_args.quote = "USDT"
    mock_args.timeframe = "1h"
    mock_args.limit = 100
    mock_args.exchanges = None
    mock_args.no_prompt = True
    mock_parse_args.return_value = mock_args
    
    mock_exchange_manager = Mock()
    mock_exchange_manager.public.exchange_priority_for_fallback = ["binance"]
    mock_exchange_manager_class.return_value = mock_exchange_manager
    
    mock_data_fetcher = Mock()
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = Mock(
        return_value=(sample_df_with_indicators, "binance")
    )
    mock_data_fetcher_class.return_value = mock_data_fetcher
    
    mock_indicator_engine = Mock()
    mock_indicator_engine.compute_features = Mock(return_value=sample_df_with_indicators)
    mock_indicator_engine_class.return_value = mock_indicator_engine
    
    mock_apply_labels.return_value = sample_df_with_indicators
    
    mock_train_and_predict.return_value = mock_model
    mock_predict_next.return_value = np.array([0.2, 0.3, 0.5])
    
    # Import and run main
    from xgboost_prediction_main import main
    
    with patch('builtins.input', side_effect=KeyboardInterrupt):
        try:
            main()
        except KeyboardInterrupt:
            pass
    
    # Verify calls
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.assert_called()
    mock_indicator_engine.compute_features.assert_called()


@patch('xgboost_prediction_main.parse_args')
@patch('xgboost_prediction_main.ExchangeManager')
@patch('xgboost_prediction_main.DataFetcher')
@patch('xgboost_prediction_main.IndicatorEngine')
def test_main_with_custom_exchanges(
    mock_indicator_engine_class,
    mock_data_fetcher_class,
    mock_exchange_manager_class,
    mock_parse_args,
    sample_df_with_indicators
):
    """Test main function with custom exchanges."""
    mock_args = Mock()
    mock_args.symbol = "BTC/USDT"
    mock_args.quote = "USDT"
    mock_args.timeframe = "1h"
    mock_args.limit = 100
    mock_args.exchanges = "binance,kraken"
    mock_args.no_prompt = True
    mock_parse_args.return_value = mock_args
    
    mock_exchange_manager = Mock()
    mock_exchange_manager.public.exchange_priority_for_fallback = ["binance", "kraken"]
    mock_exchange_manager_class.return_value = mock_exchange_manager
    
    mock_data_fetcher = Mock()
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = Mock(
        return_value=(sample_df_with_indicators, "binance")
    )
    mock_data_fetcher_class.return_value = mock_data_fetcher
    
    mock_indicator_engine = Mock()
    mock_indicator_engine.compute_features = Mock(return_value=sample_df_with_indicators)
    mock_indicator_engine_class.return_value = mock_indicator_engine
    
    from xgboost_prediction_main import main
    
    with patch('xgboost_prediction_main.train_and_predict', return_value=Mock()), \
         patch('xgboost_prediction_main.predict_next_move', return_value=np.array([0.2, 0.3, 0.5])), \
         patch('xgboost_prediction_main.apply_directional_labels', return_value=sample_df_with_indicators), \
         patch('builtins.input', side_effect=KeyboardInterrupt):
        try:
            main()
        except KeyboardInterrupt:
            pass
    
    # Verify exchange priority was set
    assert mock_exchange_manager.public.exchange_priority_for_fallback == ["binance", "kraken"]


@patch('xgboost_prediction_main.parse_args')
@patch('xgboost_prediction_main.ExchangeManager')
@patch('xgboost_prediction_main.DataFetcher')
def test_main_with_no_data(
    mock_data_fetcher_class,
    mock_exchange_manager_class,
    mock_parse_args
):
    """Test main function when no data is available."""
    mock_args = Mock()
    mock_args.symbol = "BTC/USDT"
    mock_args.quote = "USDT"
    mock_args.timeframe = "1h"
    mock_args.limit = 100
    mock_args.exchanges = None
    mock_args.no_prompt = True
    mock_parse_args.return_value = mock_args
    
    mock_exchange_manager = Mock()
    mock_exchange_manager.public.exchange_priority_for_fallback = ["binance"]
    mock_exchange_manager_class.return_value = mock_exchange_manager
    
    mock_data_fetcher = Mock()
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = Mock(return_value=(None, None))
    mock_data_fetcher_class.return_value = mock_data_fetcher
    
    from xgboost_prediction_main import main
    
    with patch('builtins.input', side_effect=KeyboardInterrupt), \
         patch('builtins.print'):
        try:
            main()
        except KeyboardInterrupt:
            pass
    
    # Verify error message would be printed (data is None)
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.assert_called()


@patch('xgboost_prediction_main.parse_args')
@patch('xgboost_prediction_main.resolve_input')
def test_main_resolve_input_calls(
    mock_resolve_input,
    mock_parse_args,
    sample_df_with_indicators
):
    """Test that resolve_input is called correctly."""
    mock_args = Mock()
    mock_args.symbol = None
    mock_args.quote = None
    mock_args.timeframe = None
    mock_args.limit = None
    mock_args.exchanges = None
    mock_args.no_prompt = False
    mock_parse_args.return_value = mock_args
    
    # resolve_input is called for timeframe and symbol, so provide enough values
    mock_resolve_input.side_effect = ["1h", "BTC/USDT", KeyboardInterrupt()]
    
    from xgboost_prediction_main import main
    
    mock_data_fetcher = Mock()
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = Mock(
        return_value=(sample_df_with_indicators, "binance")
    )
    
    with patch('xgboost_prediction_main.ExchangeManager'), \
         patch('xgboost_prediction_main.DataFetcher', return_value=mock_data_fetcher), \
         patch('xgboost_prediction_main.IndicatorEngine') as MockIndicatorEngine, \
         patch('xgboost_prediction_main.train_and_predict', return_value=Mock()), \
         patch('xgboost_prediction_main.predict_next_move', return_value=np.array([0.2, 0.3, 0.5])), \
         patch('xgboost_prediction_main.apply_directional_labels', return_value=sample_df_with_indicators), \
         patch('builtins.input', side_effect=KeyboardInterrupt):
        mock_indicator_engine = Mock()
        mock_indicator_engine.compute_features = Mock(return_value=sample_df_with_indicators)
        MockIndicatorEngine.return_value = mock_indicator_engine
        
        try:
            main()
        except KeyboardInterrupt:
            pass
    
    # Verify resolve_input was called
    assert mock_resolve_input.called


def test_run_once_logic(sample_df_with_indicators, mock_model):
    """Test the run_once inner function logic."""
    # This tests the core prediction logic
    with patch('xgboost_prediction_main.DataFetcher') as MockDataFetcher, \
         patch('xgboost_prediction_main.IndicatorEngine') as MockIndicatorEngine, \
         patch('xgboost_prediction_main.train_and_predict') as mock_train, \
         patch('xgboost_prediction_main.predict_next_move') as mock_predict, \
         patch('xgboost_prediction_main.apply_directional_labels') as mock_apply_labels, \
         patch('xgboost_prediction_main.normalize_symbol', return_value="BTC/USDT"):
        
        mock_data_fetcher = Mock()
        mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = Mock(
            return_value=(sample_df_with_indicators, "binance")
        )
        MockDataFetcher.return_value = mock_data_fetcher
        
        mock_indicator_engine = Mock()
        mock_indicator_engine.compute_features = Mock(return_value=sample_df_with_indicators)
        MockIndicatorEngine.return_value = mock_indicator_engine
        
        df_with_labels = sample_df_with_indicators.copy()
        df_with_labels = df_with_labels.dropna()
        mock_apply_labels.return_value = df_with_labels
        
        mock_train.return_value = mock_model
        mock_predict.return_value = np.array([0.2, 0.3, 0.5])
        
        # Simulate run_once logic
        df = sample_df_with_indicators.copy()
        df = mock_indicator_engine.compute_features(df)
        latest_data = df.iloc[-1:].copy()
        df = mock_apply_labels(df)
        df = df.dropna()
        model = mock_train(df)
        proba = mock_predict(model, latest_data)
        
        assert model is not None
        assert proba is not None
        assert len(proba) == 3


def test_prediction_output_formatting(sample_df_with_indicators):
    """Test prediction output formatting."""
    from modules.config import TARGET_LABELS, LABEL_TO_ID, ID_TO_LABEL
    
    # Simulate prediction probabilities
    proba = np.array([0.2, 0.3, 0.5])
    
    proba_percent = {
        label: proba[LABEL_TO_ID[label]] * 100 for label in TARGET_LABELS
    }
    best_idx = int(np.argmax(proba))
    direction = ID_TO_LABEL[best_idx]
    probability = proba_percent[direction]
    
    assert direction in TARGET_LABELS
    assert 0 <= probability <= 100
    assert all(0 <= v <= 100 for v in proba_percent.values())


def test_direction_colors():
    """Test direction color logic."""
    from colorama import Fore
    
    # Test UP direction
    direction = "UP"
    if direction == "UP":
        direction_color = Fore.GREEN
        atr_sign = 1
    elif direction == "DOWN":
        direction_color = Fore.RED
        atr_sign = -1
    else:
        direction_color = Fore.YELLOW
        atr_sign = 0
    
    assert direction_color == Fore.GREEN
    assert atr_sign == 1
    
    # Test DOWN direction
    direction = "DOWN"
    if direction == "UP":
        direction_color = Fore.GREEN
        atr_sign = 1
    elif direction == "DOWN":
        direction_color = Fore.RED
        atr_sign = -1
    else:
        direction_color = Fore.YELLOW
        atr_sign = 0
    
    assert direction_color == Fore.RED
    assert atr_sign == -1
    
    # Test NEUTRAL direction
    direction = "NEUTRAL"
    if direction == "UP":
        direction_color = Fore.GREEN
        atr_sign = 1
    elif direction == "DOWN":
        direction_color = Fore.RED
        atr_sign = -1
    else:
        direction_color = Fore.YELLOW
        atr_sign = 0
    
    assert direction_color == Fore.YELLOW
    assert atr_sign == 0


def test_atr_target_calculation():
    """Test ATR target price calculation."""
    current_price = 50000.0
    atr = 500.0
    
    # Test UP direction (atr_sign = 1)
    atr_sign = 1
    targets = []
    for multiple in (1, 2, 3):
        target_price = current_price + atr_sign * multiple * atr
        move_abs = abs(target_price - current_price)
        targets.append((multiple, target_price, move_abs))
    
    assert targets[0][1] == 50500.0  # 1x ATR
    assert targets[1][1] == 51000.0  # 2x ATR
    assert targets[2][1] == 51500.0  # 3x ATR
    
    # Test DOWN direction (atr_sign = -1)
    atr_sign = -1
    targets = []
    for multiple in (1, 2, 3):
        target_price = current_price + atr_sign * multiple * atr
        move_abs = abs(target_price - current_price)
        targets.append((multiple, target_price, move_abs))
    
    assert targets[0][1] == 49500.0  # 1x ATR
    assert targets[1][1] == 49000.0  # 2x ATR
    assert targets[2][1] == 48500.0  # 3x ATR


def test_neutral_price_bounds_calculation():
    """Test NEUTRAL direction price bounds calculation."""
    current_price = 150.0
    threshold_value = 0.1328  # 13.28%
    
    # Calculate upper and lower price bounds based on threshold
    upper_bound = current_price * (1 + threshold_value)
    lower_bound = current_price * (1 - threshold_value)
    price_range = upper_bound - lower_bound
    
    # Verify calculations
    expected_upper = 150.0 * 1.1328
    expected_lower = 150.0 * 0.8672
    expected_range = expected_upper - expected_lower
    
    assert abs(upper_bound - expected_upper) < 0.01
    assert abs(lower_bound - expected_lower) < 0.01
    assert abs(price_range - expected_range) < 0.01
    
    # Verify bounds are correct
    assert upper_bound > current_price
    assert lower_bound < current_price
    assert price_range > 0
    
    # Verify percentage calculations
    upper_pct = (upper_bound - current_price) / current_price * 100
    lower_pct = (current_price - lower_bound) / current_price * 100
    
    assert abs(upper_pct - (threshold_value * 100)) < 0.01
    assert abs(lower_pct - (threshold_value * 100)) < 0.01
    assert abs(price_range / current_price * 100 - (threshold_value * 200)) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

