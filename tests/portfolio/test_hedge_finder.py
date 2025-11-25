"""
Test script for modules.HedgeFinder - Hedge finding and scoring logic.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
import pytest

from modules.HedgeFinder import HedgeFinder
from modules.Position import Position

# Suppress warnings
warnings.filterwarnings("ignore")


def test_hedge_finder_initialization():
    """Test HedgeFinder initialization."""
    exchange_manager = Mock()
    correlation_analyzer = Mock()
    risk_calculator = Mock()
    positions = [Position("BTC/USDT", "LONG", 50000.0, 1000.0)]
    
    finder = HedgeFinder(
        exchange_manager=exchange_manager,
        correlation_analyzer=correlation_analyzer,
        risk_calculator=risk_calculator,
        positions=positions,
    )
    
    assert finder.exchange_manager == exchange_manager
    assert finder.correlation_analyzer == correlation_analyzer
    assert finder.risk_calculator == risk_calculator
    assert finder.positions == positions


def test_hedge_finder_with_data_fetcher():
    """Test HedgeFinder initialization with data_fetcher."""
    exchange_manager = Mock()
    correlation_analyzer = Mock()
    risk_calculator = Mock()
    data_fetcher = Mock()
    positions = [Position("BTC/USDT", "LONG", 50000.0, 1000.0)]
    
    finder = HedgeFinder(
        exchange_manager=exchange_manager,
        correlation_analyzer=correlation_analyzer,
        risk_calculator=risk_calculator,
        positions=positions,
        data_fetcher=data_fetcher,
    )
    
    assert finder.data_fetcher == data_fetcher


def test_should_stop_with_shutdown_event():
    """Test should_stop method with shutdown event."""
    exchange_manager = Mock()
    correlation_analyzer = Mock()
    risk_calculator = Mock()
    positions = []
    shutdown_event = Mock()
    shutdown_event.is_set.return_value = True
    
    finder = HedgeFinder(
        exchange_manager=exchange_manager,
        correlation_analyzer=correlation_analyzer,
        risk_calculator=risk_calculator,
        positions=positions,
        shutdown_event=shutdown_event,
    )
    
    assert finder.should_stop() is True


def test_should_stop_without_shutdown_event():
    """Test should_stop method without shutdown event."""
    exchange_manager = Mock()
    correlation_analyzer = Mock()
    risk_calculator = Mock()
    positions = []
    
    finder = HedgeFinder(
        exchange_manager=exchange_manager,
        correlation_analyzer=correlation_analyzer,
        risk_calculator=risk_calculator,
        positions=positions,
    )
    
    assert finder.should_stop() is False


def test_score_candidate():
    """Test _score_candidate method."""
    exchange_manager = Mock()
    correlation_analyzer = Mock()
    correlation_analyzer.calculate_weighted_correlation_with_new_symbol.return_value = (0.8, {})
    correlation_analyzer.calculate_portfolio_return_correlation.return_value = (0.75, {})
    risk_calculator = Mock()
    positions = []
    data_fetcher = Mock()
    
    finder = HedgeFinder(
        exchange_manager=exchange_manager,
        correlation_analyzer=correlation_analyzer,
        risk_calculator=risk_calculator,
        positions=positions,
        data_fetcher=data_fetcher,
    )
    
    result = finder._score_candidate("ETH/USDT")
    
    assert result is not None
    assert result["symbol"] == "ETH/USDT"
    assert result["weighted_corr"] == 0.8
    assert result["return_corr"] == 0.75
    assert "score" in result


def test_score_candidate_none_correlations():
    """Test _score_candidate when correlations are None."""
    exchange_manager = Mock()
    correlation_analyzer = Mock()
    correlation_analyzer.calculate_weighted_correlation_with_new_symbol.return_value = (None, {})
    correlation_analyzer.calculate_portfolio_return_correlation.return_value = (None, {})
    risk_calculator = Mock()
    positions = []
    data_fetcher = Mock()
    
    finder = HedgeFinder(
        exchange_manager=exchange_manager,
        correlation_analyzer=correlation_analyzer,
        risk_calculator=risk_calculator,
        positions=positions,
        data_fetcher=data_fetcher,
    )
    
    result = finder._score_candidate("ETH/USDT")
    
    assert result is None


def test_score_candidate_partial_correlations():
    """Test _score_candidate when only one correlation is available."""
    exchange_manager = Mock()
    correlation_analyzer = Mock()
    correlation_analyzer.calculate_weighted_correlation_with_new_symbol.return_value = (0.8, {})
    correlation_analyzer.calculate_portfolio_return_correlation.return_value = (None, {})
    risk_calculator = Mock()
    positions = []
    data_fetcher = Mock()
    
    finder = HedgeFinder(
        exchange_manager=exchange_manager,
        correlation_analyzer=correlation_analyzer,
        risk_calculator=risk_calculator,
        positions=positions,
        data_fetcher=data_fetcher,
    )
    
    result = finder._score_candidate("ETH/USDT")
    
    assert result is not None
    assert result["weighted_corr"] == 0.8
    assert result["return_corr"] is None


def test_find_best_hedge_candidate_no_positions():
    """Test find_best_hedge_candidate with no positions."""
    exchange_manager = Mock()
    correlation_analyzer = Mock()
    risk_calculator = Mock()
    positions = []
    data_fetcher = Mock()
    
    finder = HedgeFinder(
        exchange_manager=exchange_manager,
        correlation_analyzer=correlation_analyzer,
        risk_calculator=risk_calculator,
        positions=positions,
        data_fetcher=data_fetcher,
    )
    
    with patch("modules.HedgeFinder.color_text", return_value="test"):
        with patch("builtins.print"):
            result = finder.find_best_hedge_candidate(total_delta=1000.0, total_beta_delta=500.0)
    
    assert result is None


def test_find_best_hedge_candidate_no_candidates():
    """Test find_best_hedge_candidate when no candidates found."""
    exchange_manager = Mock()
    correlation_analyzer = Mock()
    risk_calculator = Mock()
    positions = [Position("BTC/USDT", "LONG", 50000.0, 1000.0)]
    data_fetcher = Mock()
    data_fetcher.list_binance_futures_symbols.return_value = []
    
    finder = HedgeFinder(
        exchange_manager=exchange_manager,
        correlation_analyzer=correlation_analyzer,
        risk_calculator=risk_calculator,
        positions=positions,
        data_fetcher=data_fetcher,
    )
    
    with patch("modules.HedgeFinder.normalize_symbol", side_effect=lambda x, quote="USDT": x.upper()):
        with patch("modules.HedgeFinder.color_text", return_value="test"):
            with patch("builtins.print"):
                result = finder.find_best_hedge_candidate(total_delta=1000.0, total_beta_delta=500.0)
    
    assert result is None


def test_find_best_hedge_candidate_with_candidates():
    """Test find_best_hedge_candidate with valid candidates."""
    exchange_manager = Mock()
    correlation_analyzer = Mock()
    risk_calculator = Mock()
    positions = [Position("BTC/USDT", "LONG", 50000.0, 1000.0)]
    data_fetcher = Mock()
    data_fetcher.list_binance_futures_symbols.return_value = ["ETH/USDT", "BNB/USDT"]
    
    # Mock scoring to return different scores
    def score_candidate(symbol):
        scores = {"ETH/USDT": 0.8, "BNB/USDT": 0.9}
        return {
            "symbol": symbol,
            "weighted_corr": scores[symbol],
            "return_corr": scores[symbol] * 0.9,
            "score": scores[symbol],
        }
    
    finder = HedgeFinder(
        exchange_manager=exchange_manager,
        correlation_analyzer=correlation_analyzer,
        risk_calculator=risk_calculator,
        positions=positions,
        data_fetcher=data_fetcher,
    )
    finder._score_candidate = Mock(side_effect=score_candidate)
    
    with patch("modules.HedgeFinder.normalize_symbol", side_effect=lambda x, quote="USDT": x.upper()):
        with patch("modules.HedgeFinder.color_text", return_value="test"):
            with patch("builtins.print"):
                with patch("modules.HedgeFinder.ProgressBar"):
                    result = finder.find_best_hedge_candidate(
                        total_delta=1000.0,
                        total_beta_delta=500.0,
                        max_candidates=2,
                    )
    
    assert result is not None
    assert result["symbol"] == "BNB/USDT"  # Should be the one with highest score


def test_find_best_hedge_candidate_shutdown():
    """Test find_best_hedge_candidate with shutdown event."""
    exchange_manager = Mock()
    correlation_analyzer = Mock()
    risk_calculator = Mock()
    positions = [Position("BTC/USDT", "LONG", 50000.0, 1000.0)]
    data_fetcher = Mock()
    shutdown_event = Mock()
    shutdown_event.is_set.return_value = True
    
    finder = HedgeFinder(
        exchange_manager=exchange_manager,
        correlation_analyzer=correlation_analyzer,
        risk_calculator=risk_calculator,
        positions=positions,
        data_fetcher=data_fetcher,
        shutdown_event=shutdown_event,
    )
    
    with patch("modules.HedgeFinder.normalize_symbol", side_effect=lambda x, quote="USDT": x.upper()):
        with patch("modules.HedgeFinder.color_text", return_value="test"):
            with patch("builtins.print"):
                result = finder.find_best_hedge_candidate(total_delta=1000.0, total_beta_delta=500.0)
    
    assert result is None


def test_analyze_new_trade():
    """Test analyze_new_trade method."""
    exchange_manager = Mock()
    correlation_analyzer = Mock()
    correlation_analyzer.calculate_weighted_correlation_with_new_symbol.return_value = (0.8, {})
    correlation_analyzer.calculate_portfolio_return_correlation.return_value = (0.75, {})
    risk_calculator = Mock()
    risk_calculator.calculate_beta.return_value = 1.2
    positions = [Position("BTC/USDT", "LONG", 50000.0, 1000.0)]
    data_fetcher = Mock()
    
    finder = HedgeFinder(
        exchange_manager=exchange_manager,
        correlation_analyzer=correlation_analyzer,
        risk_calculator=risk_calculator,
        positions=positions,
        data_fetcher=data_fetcher,
    )
    
    with patch("modules.HedgeFinder.normalize_symbol", side_effect=lambda x, quote="USDT": x.upper()):
        with patch("modules.HedgeFinder.color_text", return_value="test"):
            with patch("modules.HedgeFinder.Fore"):
                with patch("modules.HedgeFinder.Style"):
                    with patch("builtins.print"):
                        # Should not raise error
                        finder.analyze_new_trade(
                            new_symbol="ETH/USDT",
                            total_delta=1000.0,
                            total_beta_delta=500.0,
                        )


def test_analyze_new_trade_no_beta():
    """Test analyze_new_trade when beta cannot be calculated."""
    exchange_manager = Mock()
    correlation_analyzer = Mock()
    correlation_analyzer.calculate_weighted_correlation_with_new_symbol.return_value = (0.8, {})
    correlation_analyzer.calculate_portfolio_return_correlation.return_value = (0.75, {})
    risk_calculator = Mock()
    risk_calculator.calculate_beta.return_value = None
    positions = [Position("BTC/USDT", "LONG", 50000.0, 1000.0)]
    data_fetcher = Mock()
    
    finder = HedgeFinder(
        exchange_manager=exchange_manager,
        correlation_analyzer=correlation_analyzer,
        risk_calculator=risk_calculator,
        positions=positions,
        data_fetcher=data_fetcher,
    )
    
    with patch("modules.HedgeFinder.normalize_symbol", side_effect=lambda x, quote="USDT": x.upper()):
        with patch("modules.HedgeFinder.color_text", return_value="test"):
            with patch("modules.HedgeFinder.Fore"):
                with patch("modules.HedgeFinder.Style"):
                    with patch("builtins.print"):
                        # Should not raise error, falls back to simple delta hedging
                        finder.analyze_new_trade(
                            new_symbol="ETH/USDT",
                            total_delta=1000.0,
                            total_beta_delta=500.0,
                        )


def test_list_candidate_symbols():
    """Test _list_candidate_symbols method."""
    exchange_manager = Mock()
    correlation_analyzer = Mock()
    risk_calculator = Mock()
    positions = []
    data_fetcher = Mock()
    data_fetcher.list_binance_futures_symbols.return_value = ["ETH/USDT", "BNB/USDT"]
    
    finder = HedgeFinder(
        exchange_manager=exchange_manager,
        correlation_analyzer=correlation_analyzer,
        risk_calculator=risk_calculator,
        positions=positions,
        data_fetcher=data_fetcher,
    )
    
    result = finder._list_candidate_symbols(exclude_symbols={"BTC/USDT"}, max_candidates=10)
    
    assert result == ["ETH/USDT", "BNB/USDT"]
    data_fetcher.list_binance_futures_symbols.assert_called_once()


def test_list_candidate_symbols_no_data_fetcher():
    """Test _list_candidate_symbols raises error without data_fetcher."""
    exchange_manager = Mock()
    correlation_analyzer = Mock()
    risk_calculator = Mock()
    positions = []
    
    finder = HedgeFinder(
        exchange_manager=exchange_manager,
        correlation_analyzer=correlation_analyzer,
        risk_calculator=risk_calculator,
        positions=positions,
    )
    finder.data_fetcher = None
    
    with pytest.raises(ImportError, match="DataFetcher is required"):
        finder._list_candidate_symbols()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

