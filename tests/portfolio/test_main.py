"""
Test script for portfolio_manager_main.py - Portfolio Manager main functionality.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import threading
import signal

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
import pytest

from portfolio_manager_main import (
    PortfolioManager,
    display_portfolio_analysis,
    display_portfolio_with_hedge_analysis,
)
from modules.Position import Position
from modules.config import DEFAULT_VAR_CONFIDENCE, DEFAULT_VAR_LOOKBACK_DAYS, BENCHMARK_SYMBOL

# Suppress warnings
warnings.filterwarnings("ignore")


@pytest.fixture
def mock_exchange_manager():
    """Create a mock ExchangeManager."""
    manager = Mock()
    manager.public = Mock()
    manager.api_key = None
    manager.api_secret = None
    manager.testnet = False
    return manager


@pytest.fixture
def mock_data_fetcher():
    """Create a mock DataFetcher."""
    fetcher = Mock()
    fetcher.market_prices = {}
    fetcher.fetch_current_prices_from_binance = Mock()
    fetcher.fetch_ohlcv_with_fallback_exchange = Mock()
    return fetcher


@pytest.fixture
def mock_risk_calculator():
    """Create a mock PortfolioRiskCalculator."""
    calculator = Mock()
    calculator.calculate_stats = Mock(return_value=(None, 0.0, 0.0, 0.0))
    calculator.calculate_beta = Mock(return_value=1.0)
    calculator.calculate_portfolio_var = Mock(return_value=100.0)
    calculator.last_var_value = 100.0
    calculator.last_var_confidence = DEFAULT_VAR_CONFIDENCE
    return calculator


@pytest.fixture
def portfolio_manager(mock_exchange_manager, mock_data_fetcher, mock_risk_calculator):
    """Create a PortfolioManager instance with mocked dependencies."""
    with patch('portfolio_manager_main.ExchangeManager', return_value=mock_exchange_manager), \
         patch('portfolio_manager_main.DataFetcher', return_value=mock_data_fetcher), \
         patch('portfolio_manager_main.PortfolioRiskCalculator', return_value=mock_risk_calculator):
        pm = PortfolioManager()
        pm.data_fetcher = mock_data_fetcher
        pm.risk_calculator = mock_risk_calculator
        return pm


def test_portfolio_manager_initialization(portfolio_manager):
    """Test PortfolioManager initialization."""
    assert portfolio_manager is not None
    assert isinstance(portfolio_manager.positions, list)
    assert portfolio_manager.benchmark_symbol == BENCHMARK_SYMBOL
    assert portfolio_manager.shutdown_event is not None


def test_portfolio_manager_add_position(portfolio_manager):
    """Test adding positions to portfolio."""
    portfolio_manager.add_position("BTC/USDT", "LONG", 50000.0, 1000.0)
    
    assert len(portfolio_manager.positions) == 1
    assert portfolio_manager.positions[0].symbol == "BTC/USDT"
    assert portfolio_manager.positions[0].direction == "LONG"
    assert portfolio_manager.positions[0].entry_price == 50000.0
    assert portfolio_manager.positions[0].size_usdt == 1000.0


def test_portfolio_manager_add_multiple_positions(portfolio_manager):
    """Test adding multiple positions."""
    portfolio_manager.add_position("BTC/USDT", "LONG", 50000.0, 1000.0)
    portfolio_manager.add_position("ETH/USDT", "SHORT", 3000.0, 500.0)
    
    assert len(portfolio_manager.positions) == 2
    assert portfolio_manager.positions[0].symbol == "BTC/USDT"
    assert portfolio_manager.positions[1].symbol == "ETH/USDT"


def test_portfolio_manager_load_from_binance_success(portfolio_manager, mock_data_fetcher):
    """Test loading positions from Binance successfully."""
    mock_positions = [
        {
            "symbol": "BTC/USDT",
            "direction": "LONG",
            "entry_price": 50000.0,
            "size_usdt": 1000.0,
        }
    ]
    mock_data_fetcher.fetch_binance_futures_positions = Mock(return_value=mock_positions)
    
    portfolio_manager.load_from_binance()
    
    assert len(portfolio_manager.positions) == 1
    assert portfolio_manager.positions[0].symbol == "BTC/USDT"
    mock_data_fetcher.fetch_binance_futures_positions.assert_called_once()


def test_portfolio_manager_load_from_binance_empty(portfolio_manager, mock_data_fetcher):
    """Test loading positions from Binance when no positions exist."""
    mock_data_fetcher.fetch_binance_futures_positions = Mock(return_value=[])
    
    portfolio_manager.load_from_binance()
    
    assert len(portfolio_manager.positions) == 0


def test_portfolio_manager_load_from_binance_error(portfolio_manager, mock_data_fetcher):
    """Test loading positions from Binance when error occurs."""
    mock_data_fetcher.fetch_binance_futures_positions = Mock(side_effect=Exception("API Error"))
    
    with pytest.raises(ValueError, match="Error loading positions from Binance"):
        portfolio_manager.load_from_binance()


def test_portfolio_manager_fetch_prices(portfolio_manager, mock_data_fetcher):
    """Test fetching prices for all symbols."""
    portfolio_manager.add_position("BTC/USDT", "LONG", 50000.0, 1000.0)
    portfolio_manager.add_position("ETH/USDT", "SHORT", 3000.0, 500.0)
    
    portfolio_manager.fetch_prices()
    
    # Verify call was made (order may vary due to set conversion)
    mock_data_fetcher.fetch_current_prices_from_binance.assert_called_once()
    call_args = mock_data_fetcher.fetch_current_prices_from_binance.call_args[0][0]
    assert set(call_args) == {"BTC/USDT", "ETH/USDT"}


def test_portfolio_manager_fetch_prices_empty(portfolio_manager, mock_data_fetcher):
    """Test fetching prices when no positions exist."""
    portfolio_manager.fetch_prices()
    
    # Should not call fetch_current_prices_from_binance when no positions
    mock_data_fetcher.fetch_current_prices_from_binance.assert_not_called()


def test_portfolio_manager_market_prices(portfolio_manager, mock_data_fetcher):
    """Test accessing market prices property."""
    mock_data_fetcher.market_prices = {"BTC/USDT": 51000.0}
    
    assert portfolio_manager.market_prices == {"BTC/USDT": 51000.0}


def test_portfolio_manager_calculate_stats(portfolio_manager, mock_risk_calculator, mock_data_fetcher):
    """Test calculating portfolio statistics."""
    portfolio_manager.add_position("BTC/USDT", "LONG", 50000.0, 1000.0)
    mock_data_fetcher.market_prices = {"BTC/USDT": 51000.0}
    
    mock_risk_calculator.calculate_stats.return_value = (
        Mock(), 100.0, 50.0, 75.0  # df, pnl, delta, beta_delta
    )
    
    result = portfolio_manager.calculate_stats()
    
    assert result is not None
    mock_risk_calculator.calculate_stats.assert_called_once()


def test_portfolio_manager_calculate_beta(portfolio_manager, mock_risk_calculator):
    """Test calculating beta for a symbol."""
    mock_risk_calculator.calculate_beta.return_value = 1.5
    
    beta = portfolio_manager.calculate_beta("ETH/USDT", "BTC/USDT")
    
    assert beta == 1.5
    mock_risk_calculator.calculate_beta.assert_called_once()


def test_portfolio_manager_calculate_portfolio_var(portfolio_manager, mock_risk_calculator):
    """Test calculating portfolio VaR."""
    portfolio_manager.add_position("BTC/USDT", "LONG", 50000.0, 1000.0)
    mock_risk_calculator.calculate_portfolio_var.return_value = 200.0
    
    var = portfolio_manager.calculate_portfolio_var(
        confidence=0.95, lookback_days=60
    )
    
    assert var == 200.0
    mock_risk_calculator.calculate_portfolio_var.assert_called_once()


def test_portfolio_manager_var_properties(portfolio_manager, mock_risk_calculator):
    """Test VaR-related properties."""
    mock_risk_calculator.last_var_value = 150.0
    mock_risk_calculator.last_var_confidence = 0.95
    
    assert portfolio_manager.last_var_value == 150.0
    assert portfolio_manager.last_var_confidence == 0.95


def test_portfolio_manager_fetch_ohlcv(portfolio_manager, mock_data_fetcher):
    """Test fetching OHLCV data."""
    mock_df = Mock()
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange = Mock(return_value=(mock_df, "binance"))
    
    result = portfolio_manager.fetch_ohlcv("BTC/USDT", limit=100, timeframe="1h")
    
    assert result == mock_df
    mock_data_fetcher.fetch_ohlcv_with_fallback_exchange.assert_called_once()


def test_portfolio_manager_calculate_weighted_correlation(portfolio_manager, mock_data_fetcher):
    """Test calculating weighted correlation."""
    portfolio_manager.add_position("BTC/USDT", "LONG", 50000.0, 1000.0)
    
    with patch('portfolio_manager_main.PortfolioCorrelationAnalyzer') as MockAnalyzer:
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.calculate_weighted_correlation_with_new_symbol.return_value = (
            0.75, {"BTC/USDT": 0.8}
        )
        MockAnalyzer.return_value = mock_analyzer_instance
        
        result = portfolio_manager.calculate_weighted_correlation("ETH/USDT", verbose=True)
        
        assert result is not None
        MockAnalyzer.assert_called_once()


def test_portfolio_manager_find_best_hedge_candidate(portfolio_manager):
    """Test finding best hedge candidate."""
    portfolio_manager.add_position("BTC/USDT", "LONG", 50000.0, 1000.0)
    
    with patch('portfolio_manager_main.PortfolioCorrelationAnalyzer'), \
         patch('portfolio_manager_main.HedgeFinder') as MockHedgeFinder:
        mock_finder_instance = Mock()
        mock_finder_instance.find_best_hedge_candidate.return_value = {
            "symbol": "ETH/USDT",
            "correlation": -0.9
        }
        MockHedgeFinder.return_value = mock_finder_instance
        
        result = portfolio_manager.find_best_hedge_candidate(
            total_delta=100.0, total_beta_delta=150.0
        )
        
        assert result is not None
        MockHedgeFinder.assert_called_once()


def test_portfolio_manager_analyze_new_trade(portfolio_manager):
    """Test analyzing a new trade."""
    portfolio_manager.add_position("BTC/USDT", "LONG", 50000.0, 1000.0)
    
    with patch('portfolio_manager_main.PortfolioCorrelationAnalyzer'), \
         patch('portfolio_manager_main.HedgeFinder') as MockHedgeFinder:
        mock_finder_instance = Mock()
        mock_finder_instance.analyze_new_trade.return_value = (
            "SHORT", 500.0, -0.85
        )
        MockHedgeFinder.return_value = mock_finder_instance
        
        direction, size, correlation = portfolio_manager.analyze_new_trade(
            "ETH/USDT", total_delta=100.0, total_beta_delta=150.0
        )
        
        assert direction == "SHORT"
        assert size == 500.0
        assert correlation == -0.85
        MockHedgeFinder.assert_called_once()


def test_portfolio_manager_shutdown_signal(portfolio_manager):
    """Test shutdown signal handling."""
    assert not portfolio_manager.shutdown_event.is_set()
    
    # Simulate shutdown
    portfolio_manager.shutdown_event.set()
    
    assert portfolio_manager._should_stop()


@patch('builtins.print')
def test_display_portfolio_analysis(mock_print, portfolio_manager, mock_risk_calculator, mock_data_fetcher):
    """Test display_portfolio_analysis function."""
    portfolio_manager.add_position("BTC/USDT", "LONG", 50000.0, 1000.0)
    mock_data_fetcher.market_prices = {"BTC/USDT": 51000.0}
    
    # Mock calculate_stats return
    mock_df = Mock()
    mock_df.to_string.return_value = "Mock DataFrame"
    mock_risk_calculator.calculate_stats.return_value = (mock_df, 100.0, 50.0, 75.0)
    mock_risk_calculator.calculate_portfolio_var.return_value = 200.0
    mock_risk_calculator.last_var_value = 200.0
    mock_risk_calculator.last_var_confidence = 0.95
    
    with patch('portfolio_manager_main.PortfolioCorrelationAnalyzer') as MockAnalyzer:
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.calculate_weighted_correlation.return_value = (
            0.5, {"BTC/USDT": 0.6}
        )
        MockAnalyzer.return_value = mock_analyzer_instance
        
        display_portfolio_analysis(portfolio_manager)
        
        # Verify some print calls were made
        assert mock_print.called


@patch('builtins.print')
def test_display_portfolio_with_hedge_analysis(mock_print, portfolio_manager, mock_risk_calculator, mock_data_fetcher):
    """Test display_portfolio_with_hedge_analysis function."""
    portfolio_manager.add_position("BTC/USDT", "LONG", 50000.0, 1000.0)
    mock_data_fetcher.market_prices = {"BTC/USDT": 51000.0}
    
    # Mock calculate_stats return
    mock_df = Mock()
    mock_risk_calculator.calculate_stats.return_value = (mock_df, 100.0, 50.0, 75.0)
    mock_risk_calculator.calculate_portfolio_var.return_value = 200.0
    mock_risk_calculator.last_var_value = 200.0
    mock_risk_calculator.last_var_confidence = 0.95
    
    with patch('portfolio_manager_main.PortfolioCorrelationAnalyzer'), \
         patch('portfolio_manager_main.HedgeFinder') as MockHedgeFinder:
        mock_finder_instance = Mock()
        mock_finder_instance.find_best_hedge_candidate.return_value = {
            "symbol": "ETH/USDT",
            "correlation": -0.9
        }
        mock_finder_instance.analyze_new_trade.return_value = (
            "SHORT", 500.0, -0.85
        )
        MockHedgeFinder.return_value = mock_finder_instance
        
        display_portfolio_with_hedge_analysis(portfolio_manager)
        
        # Verify some print calls were made
        assert mock_print.called


def test_portfolio_manager_empty_positions(portfolio_manager):
    """Test portfolio manager with empty positions."""
    assert len(portfolio_manager.positions) == 0
    
    # Should handle empty positions gracefully
    result = portfolio_manager.calculate_stats()
    assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

