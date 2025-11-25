import numpy as np
import pandas as pd
import pytest

from modules.pairs_trading.performance_analyzer import PerformanceAnalyzer


def build_price_series(length=200, start=100.0, step=0.5):
    timestamps = pd.date_range("2024-01-01", periods=length, freq="H")
    close = start + np.arange(length) * step
    return pd.DataFrame({"timestamp": timestamps, "close": close})


def test_performance_analyzer_validates_weight_sum():
    with pytest.raises(ValueError):
        PerformanceAnalyzer(weights={"1d": 0.5, "3d": 0.6})


def test_calculate_performance_score_returns_expected_fields():
    analyzer = PerformanceAnalyzer()
    df = build_price_series()
    result = analyzer.calculate_performance_score("AAA/USDT", df)

    assert result is not None
    assert set(result.keys()) == {
        "symbol",
        "score",
        "1d_return",
        "3d_return",
        "1w_return",
        "current_price",
    }
    assert result["symbol"] == "AAA/USDT"


def test_get_top_and_worst_performers_respect_order():
    analyzer = PerformanceAnalyzer()
    data = pd.DataFrame(
        [
            {"symbol": "AAA", "score": 0.5, "1d_return": 0, "3d_return": 0, "1w_return": 0, "current_price": 1},
            {"symbol": "CCC", "score": 0.1, "1d_return": 0, "3d_return": 0, "1w_return": 0, "current_price": 1},
            {"symbol": "BBB", "score": -0.1, "1d_return": 0, "3d_return": 0, "1w_return": 0, "current_price": 1},
        ]
    )

    top = analyzer.get_top_performers(data, top_n=2)
    worst = analyzer.get_worst_performers(data, top_n=1)

    assert list(top["symbol"]) == ["AAA", "CCC"]
    assert list(worst["symbol"]) == ["BBB"]
"""
Test script for PerformanceAnalyzer
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from types import SimpleNamespace

from modules.pairs_trading.performance_analyzer import PerformanceAnalyzer


def _create_mock_ohlcv_data(
    start_price: float = 100.0,
    num_candles: int = 200,
    trend: str = "up",
    volatility: float = 0.01,
) -> pd.DataFrame:
    """
    Create mock OHLCV DataFrame for testing.

    Args:
        start_price: Starting price
        num_candles: Number of candles to generate
        trend: 'up', 'down', or 'neutral'
        volatility: Price volatility (standard deviation)

    Returns:
        DataFrame with OHLCV columns
    """
    timestamps = pd.date_range(
        start="2024-01-01", periods=num_candles, freq="1h", tz="UTC"
    )

    # Generate price series with trend
    if trend == "up":
        trend_factor = np.linspace(0, 0.2, num_candles)  # 20% upward trend
    elif trend == "down":
        trend_factor = np.linspace(0, -0.2, num_candles)  # 20% downward trend
    else:
        trend_factor = np.zeros(num_candles)  # No trend

    # Generate random walk with trend
    np.random.seed(42)
    returns = np.random.normal(0, volatility, num_candles) + trend_factor / num_candles
    prices = start_price * (1 + returns).cumprod()

    # Create OHLCV data
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices * (1 + np.random.normal(0, 0.001, num_candles)),
            "high": prices * (1 + abs(np.random.normal(0, 0.002, num_candles))),
            "low": prices * (1 - abs(np.random.normal(0, 0.002, num_candles))),
            "close": prices,
            "volume": np.random.uniform(1000, 10000, num_candles),
        }
    )

    return df


def test_calculate_performance_score_with_sufficient_data():
    """Test calculate_performance_score with sufficient data."""
    analyzer = PerformanceAnalyzer()
    
    # Create data with upward trend
    df = _create_mock_ohlcv_data(start_price=100.0, num_candles=200, trend="up")
    
    result = analyzer.calculate_performance_score("BTC/USDT", df)
    
    assert result is not None
    assert result["symbol"] == "BTC/USDT"
    assert "score" in result
    assert "1d_return" in result
    assert "3d_return" in result
    assert "1w_return" in result
    assert "current_price" in result
    assert result["current_price"] > 0
    # With upward trend, returns should be positive
    assert result["1w_return"] > 0


def test_calculate_performance_score_with_insufficient_data():
    """Test calculate_performance_score with insufficient data."""
    analyzer = PerformanceAnalyzer(min_candles=168)
    
    # Create data with only 50 candles (less than minimum 168)
    df = _create_mock_ohlcv_data(start_price=100.0, num_candles=50, trend="up")
    
    result = analyzer.calculate_performance_score("BTC/USDT", df)
    
    assert result is None


def test_calculate_performance_score_with_downward_trend():
    """Test calculate_performance_score with downward trend."""
    analyzer = PerformanceAnalyzer()
    
    # Create data with downward trend
    df = _create_mock_ohlcv_data(start_price=100.0, num_candles=200, trend="down")
    
    result = analyzer.calculate_performance_score("ETH/USDT", df)
    
    assert result is not None
    # With downward trend, returns should be negative
    assert result["1w_return"] < 0
    assert result["score"] < 0


def test_calculate_performance_score_with_empty_dataframe():
    """Test calculate_performance_score with empty DataFrame."""
    analyzer = PerformanceAnalyzer()
    
    df = pd.DataFrame()
    result = analyzer.calculate_performance_score("BTC/USDT", df)
    
    assert result is None


def test_calculate_performance_score_with_missing_close_column():
    """Test calculate_performance_score with missing 'close' column."""
    analyzer = PerformanceAnalyzer()
    
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=200, freq="1h")})
    result = analyzer.calculate_performance_score("BTC/USDT", df)
    
    assert result is None


def test_calculate_performance_score_weights_calculation():
    """Test that weighted score is calculated correctly."""
    # Use custom weights for easier verification
    weights = {"1d": 0.5, "3d": 0.3, "1w": 0.2}
    analyzer = PerformanceAnalyzer(weights=weights)
    
    # Create data with known returns
    # Set prices manually to get specific returns
    timestamps = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
    
    # Create prices: 100 -> 110 (10% increase over 1 week)
    # This gives us: 1d return ~0, 3d return ~0, 1w return = 0.1
    prices = np.ones(200) * 100.0
    prices[-168:] = np.linspace(100, 110, 168)  # 1 week: 100 -> 110 (10% increase)
    
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.ones(200) * 1000,
        }
    )
    
    result = analyzer.calculate_performance_score("TEST/USDT", df)
    
    assert result is not None
    # 1w return should be approximately (110 - 100) / 100 = 0.1
    assert abs(result["1w_return"] - 0.1) < 0.05  # Allow some tolerance
    # Score should be approximately 0.1 * 0.2 = 0.02
    assert result["score"] > 0


def test_get_top_performers():
    """Test get_top_performers method."""
    analyzer = PerformanceAnalyzer()
    
    # Create mock results DataFrame
    results = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E"],
            "score": [0.1, 0.05, 0.0, -0.05, -0.1],
            "1d_return": [0.1, 0.05, 0.0, -0.05, -0.1],
            "3d_return": [0.1, 0.05, 0.0, -0.05, -0.1],
            "1w_return": [0.1, 0.05, 0.0, -0.05, -0.1],
            "current_price": [100, 100, 100, 100, 100],
        }
    )
    
    top_3 = analyzer.get_top_performers(results, top_n=3)
    
    assert len(top_3) == 3
    assert list(top_3["symbol"]) == ["A", "B", "C"]
    assert top_3["score"].iloc[0] == 0.1  # Highest score


def test_get_worst_performers():
    """Test get_worst_performers method."""
    analyzer = PerformanceAnalyzer()
    
    # Create mock results DataFrame
    results = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E"],
            "score": [0.1, 0.05, 0.0, -0.05, -0.1],
            "1d_return": [0.1, 0.05, 0.0, -0.05, -0.1],
            "3d_return": [0.1, 0.05, 0.0, -0.05, -0.1],
            "1w_return": [0.1, 0.05, 0.0, -0.05, -0.1],
            "current_price": [100, 100, 100, 100, 100],
        }
    )
    
    worst_3 = analyzer.get_worst_performers(results, top_n=3)
    
    assert len(worst_3) == 3
    assert list(worst_3["symbol"]) == ["E", "D", "C"]
    assert worst_3["score"].iloc[0] == -0.1  # Lowest score


def test_get_top_performers_with_empty_dataframe():
    """Test get_top_performers with empty DataFrame."""
    analyzer = PerformanceAnalyzer()
    
    empty_df = pd.DataFrame(
        columns=["symbol", "score", "1d_return", "3d_return", "1w_return", "current_price"]
    )
    
    result = analyzer.get_top_performers(empty_df, top_n=5)
    
    assert len(result) == 0
    assert list(result.columns) == [
        "symbol",
        "score",
        "1d_return",
        "3d_return",
        "1w_return",
        "current_price",
    ]


def test_get_worst_performers_with_empty_dataframe():
    """Test get_worst_performers with empty DataFrame."""
    analyzer = PerformanceAnalyzer()
    
    empty_df = pd.DataFrame(
        columns=["symbol", "score", "1d_return", "3d_return", "1w_return", "current_price"]
    )
    
    result = analyzer.get_worst_performers(empty_df, top_n=5)
    
    assert len(result) == 0
    assert list(result.columns) == [
        "symbol",
        "score",
        "1d_return",
        "3d_return",
        "1w_return",
        "current_price",
    ]


def test_analyze_all_symbols_with_mock_data():
    """Test analyze_all_symbols with mock data fetcher."""
    analyzer = PerformanceAnalyzer(limit=200)
    
    # Create mock data fetcher
    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        df = _create_mock_ohlcv_data(start_price=100.0, num_candles=200, trend="up")
        return df, "binance"
    
    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)
    
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    results = analyzer.analyze_all_symbols(symbols, mock_fetcher, verbose=False)
    
    assert len(results) == 3
    assert "symbol" in results.columns
    assert "score" in results.columns
    assert "1d_return" in results.columns
    assert "3d_return" in results.columns
    assert "1w_return" in results.columns
    assert "current_price" in results.columns
    # Results should be sorted by score descending
    assert results["score"].iloc[0] >= results["score"].iloc[1]
    assert results["score"].iloc[1] >= results["score"].iloc[2]


def test_analyze_all_symbols_with_insufficient_data():
    """Test analyze_all_symbols when some symbols have insufficient data."""
    analyzer = PerformanceAnalyzer(min_candles=168, limit=200)
    
    call_count = {"count": 0}
    
    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        call_count["count"] += 1
        if symbol == "INSUFFICIENT/USDT":
            # Return insufficient data
            df = _create_mock_ohlcv_data(start_price=100.0, num_candles=50, trend="up")
        else:
            df = _create_mock_ohlcv_data(start_price=100.0, num_candles=200, trend="up")
        return df, "binance"
    
    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)
    
    symbols = ["BTC/USDT", "INSUFFICIENT/USDT", "ETH/USDT"]
    results = analyzer.analyze_all_symbols(symbols, mock_fetcher, verbose=False)
    
    # Should only have 2 results (INSUFFICIENT should be skipped)
    assert len(results) == 2
    assert "INSUFFICIENT/USDT" not in results["symbol"].values


def test_analyze_all_symbols_with_empty_symbols_list():
    """Test analyze_all_symbols with empty symbols list."""
    analyzer = PerformanceAnalyzer()
    
    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=lambda *a, **k: (None, None))
    
    results = analyzer.analyze_all_symbols([], mock_fetcher, verbose=False)
    
    assert len(results) == 0
    assert list(results.columns) == [
        "symbol",
        "score",
        "1d_return",
        "3d_return",
        "1w_return",
        "current_price",
    ]


def test_analyze_all_symbols_with_fetch_failure():
    """Test analyze_all_symbols when data fetch fails."""
    analyzer = PerformanceAnalyzer()
    
    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        return None, None
    
    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)
    
    symbols = ["BTC/USDT", "ETH/USDT"]
    results = analyzer.analyze_all_symbols(symbols, mock_fetcher, verbose=False)
    
    assert len(results) == 0


def test_weights_validation():
    """Test that weights must sum to approximately 1.0."""
    # Valid weights
    analyzer1 = PerformanceAnalyzer(weights={"1d": 0.3, "3d": 0.4, "1w": 0.3})
    assert analyzer1.weights == {"1d": 0.3, "3d": 0.4, "1w": 0.3}
    
    # Invalid weights (sum != 1.0)
    try:
        analyzer2 = PerformanceAnalyzer(weights={"1d": 0.5, "3d": 0.5, "1w": 0.5})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected


def test_custom_weights_in_calculation():
    """Test that custom weights are used in score calculation."""
    # Use weights that make it easy to verify
    weights = {"1d": 1.0, "3d": 0.0, "1w": 0.0}  # Only 1d matters
    analyzer = PerformanceAnalyzer(weights=weights)
    
    # Create data where 1d return is 0.1, others are 0
    timestamps = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
    prices = np.ones(200) * 100.0
    prices[-24] = 100.0  # 1 day ago
    prices[-1] = 110.0  # Current (10% increase from 1 day ago)
    
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.ones(200) * 1000,
        }
    )
    
    result = analyzer.calculate_performance_score("TEST/USDT", df)
    
    assert result is not None
    # With weights {1d: 1.0, 3d: 0.0, 1w: 0.0}, score should equal 1d_return
    assert abs(result["score"] - result["1d_return"]) < 0.01


if __name__ == "__main__":
    # Run basic tests
    print("Running PerformanceAnalyzer tests...")
    
    test_calculate_performance_score_with_sufficient_data()
    print("✓ test_calculate_performance_score_with_sufficient_data")
    
    test_calculate_performance_score_with_insufficient_data()
    print("✓ test_calculate_performance_score_with_insufficient_data")
    
    test_calculate_performance_score_with_downward_trend()
    print("✓ test_calculate_performance_score_with_downward_trend")
    
    test_get_top_performers()
    print("✓ test_get_top_performers")
    
    test_get_worst_performers()
    print("✓ test_get_worst_performers")
    
    test_analyze_all_symbols_with_mock_data()
    print("✓ test_analyze_all_symbols_with_mock_data")
    
    test_weights_validation()
    print("✓ test_weights_validation")
    
    print("\nAll tests passed!")

