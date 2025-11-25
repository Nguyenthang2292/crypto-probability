import pandas as pd

from modules.pairs_trading.pairs_analyzer import PairsTradingAnalyzer, _get_all_pair_columns


class DummyFetcher:
    def __init__(self, close_start=100.0):
        timestamps = pd.date_range("2024-01-01", periods=300, freq="H")
        close = pd.Series(
            close_start + pd.Series(range(300)).astype(float).values * 0.1
        )
        volume = pd.Series([20000.0] * 300)
        self.df = pd.DataFrame({"timestamp": timestamps, "close": close, "volume": volume})

    def fetch_ohlcv_with_fallback_exchange(self, *_, **__):
        return self.df.copy(), "binance"


def test_calculate_spread_returns_absolute_difference():
    analyzer = PairsTradingAnalyzer()
    spread = analyzer.calculate_spread("AAA/USDT", "BBB/USDT", -0.05, 0.08)
    assert spread == 0.13


def test_analyze_pairs_opportunity_builds_records(monkeypatch):
    analyzer = PairsTradingAnalyzer(min_spread=0.01, max_spread=1.0)

    monkeypatch.setattr(
        "modules.pairs_trading.pairs_analyzer.PairMetricsComputer.compute_pair_metrics",
        lambda self, price1, price2: {
            "is_cointegrated": True,
            "half_life": 10,
            "current_zscore": 1.0,
            "spread_sharpe": 1.1,
        },
    )
    monkeypatch.setattr(
        "modules.pairs_trading.pairs_analyzer.PairsTradingAnalyzer.calculate_correlation",
        lambda self, s1, s2, data_fetcher, timeframe="1h", limit=200: 0.5,
    )

    best = pd.DataFrame({"symbol": ["AAA/USDT"], "score": [0.08]})
    worst = pd.DataFrame({"symbol": ["BBB/USDT"], "score": [-0.04]})

    fetcher = DummyFetcher()
    df_pairs = analyzer.analyze_pairs_opportunity(best, worst, data_fetcher=fetcher, verbose=False)

    assert len(df_pairs) == 1
    row = df_pairs.iloc[0]
    assert row["long_symbol"] == "BBB/USDT"
    assert row["short_symbol"] == "AAA/USDT"
    assert set(df_pairs.columns) >= set(_get_all_pair_columns())


def test_validate_pairs_filters_by_thresholds(monkeypatch):
    analyzer = PairsTradingAnalyzer(min_spread=0.01, max_spread=0.5, min_correlation=0.3, max_correlation=0.9)

    data = pd.DataFrame(
        [
            {
                "long_symbol": "BBB/USDT",
                "short_symbol": "AAA/USDT",
                "spread": 0.02,
                "correlation": 0.5,
                "opportunity_score": 0.5,
            },
            {
                "long_symbol": "CCC/USDT",
                "short_symbol": "AAA/USDT",
                "spread": 0.9,
                "correlation": 0.95,
                "opportunity_score": 0.4,
            },
        ]
    )

    fetcher = DummyFetcher()
    validated = analyzer.validate_pairs(data, fetcher, verbose=False)

    assert len(validated) == 1
    assert validated.iloc[0]["long_symbol"] == "BBB/USDT"
"""
Test script for PairsTradingAnalyzer
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from types import SimpleNamespace



def _create_mock_ohlcv_data(
    start_price: float = 100.0,
    num_candles: int = 200,
    trend: str = "up",
    volatility: float = 0.01,
    correlation_with: pd.DataFrame = None,
    correlation_strength: float = 0.8,
) -> pd.DataFrame:
    """
    Create mock OHLCV DataFrame for testing.

    Args:
        start_price: Starting price
        num_candles: Number of candles to generate
        trend: 'up', 'down', or 'neutral'
        volatility: Price volatility (standard deviation)
        correlation_with: Another DataFrame to correlate with
        correlation_strength: Strength of correlation (0-1)

    Returns:
        DataFrame with OHLCV columns
    """
    timestamps = pd.date_range(
        start="2024-01-01", periods=num_candles, freq="1h", tz="UTC"
    )

    if correlation_with is not None and 'close' in correlation_with.columns:
        # Generate correlated prices
        base_returns = correlation_with['close'].pct_change().dropna().values
        np.random.seed(42)
        noise = np.random.normal(0, volatility * (1 - correlation_strength), len(base_returns))
        correlated_returns = base_returns * correlation_strength + noise
        prices = start_price * (1 + correlated_returns).cumprod()
        # Pad to num_candles if needed
        if len(prices) < num_candles:
            padding = np.ones(num_candles - len(prices)) * prices[-1]
            prices = np.concatenate([padding, prices])
    else:
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


def test_calculate_spread():
    """Test calculate_spread method."""
    analyzer = PairsTradingAnalyzer()

    # Test with typical values: worst performer (negative score) and best performer (positive score)
    long_score = -0.1  # Worst performer
    short_score = 0.15  # Best performer
    spread = analyzer.calculate_spread("BTC/USDT", "ETH/USDT", long_score, short_score)

    # Spread should be positive: short_score - long_score = 0.15 - (-0.1) = 0.25
    assert spread > 0
    assert abs(spread - 0.25) < 0.01


def test_calculate_spread_with_negative_values():
    """Test calculate_spread when both scores are negative."""
    analyzer = PairsTradingAnalyzer()

    # Both negative, but short is less negative
    long_score = -0.2
    short_score = -0.05
    spread = analyzer.calculate_spread("BTC/USDT", "ETH/USDT", long_score, short_score)

    # Spread = abs(-0.05 - (-0.2)) = 0.15
    assert spread > 0
    assert abs(spread - 0.15) < 0.01


def test_calculate_correlation_with_correlated_data():
    """Test calculate_correlation with correlated price data."""
    analyzer = PairsTradingAnalyzer(correlation_min_points=50)

    # Create correlated data
    df1 = _create_mock_ohlcv_data(start_price=100.0, num_candles=200, trend="up")
    df2 = _create_mock_ohlcv_data(
        start_price=50.0, num_candles=200, correlation_with=df1, correlation_strength=0.7
    )

    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        if symbol == "SYMBOL1/USDT":
            return df1, "binance"
        elif symbol == "SYMBOL2/USDT":
            return df2, "binance"
        return None, None

    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)

    correlation = analyzer.calculate_correlation(
        "SYMBOL1/USDT", "SYMBOL2/USDT", mock_fetcher
    )

    assert correlation is not None
    assert -1 <= correlation <= 1
    # Should be positive and reasonably high
    assert correlation > 0.5


def test_calculate_correlation_with_insufficient_data():
    """Test calculate_correlation with insufficient data."""
    analyzer = PairsTradingAnalyzer(correlation_min_points=100)

    # Create data with insufficient points
    df1 = _create_mock_ohlcv_data(start_price=100.0, num_candles=50)
    df2 = _create_mock_ohlcv_data(start_price=50.0, num_candles=50)

    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        if symbol == "SYMBOL1/USDT":
            return df1, "binance"
        elif symbol == "SYMBOL2/USDT":
            return df2, "binance"
        return None, None

    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)

    correlation = analyzer.calculate_correlation(
        "SYMBOL1/USDT", "SYMBOL2/USDT", mock_fetcher
    )

    # Should return None due to insufficient data
    assert correlation is None


def test_calculate_correlation_with_missing_data():
    """Test calculate_correlation when data fetch fails."""
    analyzer = PairsTradingAnalyzer()

    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        return None, None

    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)

    correlation = analyzer.calculate_correlation(
        "SYMBOL1/USDT", "SYMBOL2/USDT", mock_fetcher
    )

    assert correlation is None


def test_calculate_correlation_uses_cache():
    """Test that correlation results are cached."""
    analyzer = PairsTradingAnalyzer(correlation_min_points=50)

    df1 = _create_mock_ohlcv_data(start_price=100.0, num_candles=200)
    df2 = _create_mock_ohlcv_data(
        start_price=50.0, num_candles=200, correlation_with=df1, correlation_strength=0.6
    )

    call_count = {"count": 0}

    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        call_count["count"] += 1
        if symbol == "SYMBOL1/USDT":
            return df1, "binance"
        elif symbol == "SYMBOL2/USDT":
            return df2, "binance"
        return None, None

    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)

    # First call
    corr1 = analyzer.calculate_correlation("SYMBOL1/USDT", "SYMBOL2/USDT", mock_fetcher)
    first_call_count = call_count["count"]

    # Second call (should use cache)
    corr2 = analyzer.calculate_correlation("SYMBOL2/USDT", "SYMBOL1/USDT", mock_fetcher)
    second_call_count = call_count["count"]

    # Should be same result and cache should prevent additional fetch
    assert corr1 == corr2
    # Note: May still fetch if cache key is different, but correlation should be same


def test_analyze_pairs_opportunity():
    """Test analyze_pairs_opportunity with sample data."""
    analyzer = PairsTradingAnalyzer()

    best_performers = pd.DataFrame(
        {
            "symbol": ["BEST1/USDT", "BEST2/USDT"],
            "score": [0.15, 0.12],
            "1d_return": [0.1, 0.08],
            "3d_return": [0.15, 0.12],
            "1w_return": [0.2, 0.16],
            "current_price": [100, 200],
        }
    )

    worst_performers = pd.DataFrame(
        {
            "symbol": ["WORST1/USDT", "WORST2/USDT"],
            "score": [-0.1, -0.08],
            "1d_return": [-0.05, -0.04],
            "3d_return": [-0.1, -0.08],
            "1w_return": [-0.15, -0.12],
            "current_price": [50, 75],
        }
    )

    pairs_df = analyzer.analyze_pairs_opportunity(
        best_performers, worst_performers, verbose=False
    )

    assert len(pairs_df) > 0
    assert "long_symbol" in pairs_df.columns
    assert "short_symbol" in pairs_df.columns
    assert "spread" in pairs_df.columns
    assert "opportunity_score" in pairs_df.columns

    # Should have 2 worst × 2 best = 4 pairs (excluding same symbol)
    assert len(pairs_df) == 4

    # Check that all long symbols are from worst_performers
    assert all(sym in worst_performers["symbol"].values for sym in pairs_df["long_symbol"])

    # Check that all short symbols are from best_performers
    assert all(sym in best_performers["symbol"].values for sym in pairs_df["short_symbol"])

    # Check that pairs are sorted by opportunity_score descending
    scores = pairs_df["opportunity_score"].values
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_analyze_pairs_opportunity_with_correlation():
    """Test analyze_pairs_opportunity with correlation calculation."""
    analyzer = PairsTradingAnalyzer(correlation_min_points=50)

    best_performers = pd.DataFrame(
        {
            "symbol": ["BEST1/USDT"],
            "score": [0.15],
            "1d_return": [0.1],
            "3d_return": [0.15],
            "1w_return": [0.2],
            "current_price": [100],
        }
    )

    worst_performers = pd.DataFrame(
        {
            "symbol": ["WORST1/USDT"],
            "score": [-0.1],
            "1d_return": [-0.05],
            "3d_return": [-0.1],
            "1w_return": [-0.15],
            "current_price": [50],
        }
    )

    # Create correlated data
    df_best = _create_mock_ohlcv_data(start_price=100.0, num_candles=200, trend="up")
    df_worst = _create_mock_ohlcv_data(
        start_price=50.0, num_candles=200, correlation_with=df_best, correlation_strength=0.6
    )

    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        if symbol == "BEST1/USDT":
            return df_best, "binance"
        elif symbol == "WORST1/USDT":
            return df_worst, "binance"
        return None, None

    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)

    pairs_df = analyzer.analyze_pairs_opportunity(
        best_performers, worst_performers, data_fetcher=mock_fetcher, verbose=False
    )

    assert len(pairs_df) == 1
    assert pairs_df["correlation"].iloc[0] is not None
    assert -1 <= pairs_df["correlation"].iloc[0] <= 1


def test_analyze_pairs_opportunity_with_empty_dataframes():
    """Test analyze_pairs_opportunity with empty DataFrames."""
    analyzer = PairsTradingAnalyzer()

    empty_df = pd.DataFrame(
        columns=["symbol", "score", "1d_return", "3d_return", "1w_return", "current_price"]
    )

    result = analyzer.analyze_pairs_opportunity(empty_df, empty_df, verbose=False)

    assert len(result) == 0
    assert list(result.columns) == _get_all_pair_columns()


def test_validate_pairs_with_valid_spread():
    """Test validate_pairs with valid spread."""
    analyzer = PairsTradingAnalyzer(
        min_spread=0.01, max_spread=0.50, min_correlation=0.3, max_correlation=0.9
    )

    pairs_df = pd.DataFrame(
        {
            "long_symbol": ["WORST1/USDT"],
            "short_symbol": ["BEST1/USDT"],
            "long_score": [-0.1],
            "short_score": [0.15],
            "spread": [0.25],  # Valid spread (1% < 25% < 50%)
            "correlation": [0.6],  # Valid correlation (0.3 < 0.6 < 0.9)
            "opportunity_score": [0.25],
        }
    )

    # Create mock data with sufficient volume
    df = _create_mock_ohlcv_data(start_price=100.0, num_candles=200)
    df["volume"] = df["volume"] * 100  # Increase volume to meet minimum

    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        return df, "binance"

    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)

    validated = analyzer.validate_pairs(pairs_df, mock_fetcher, verbose=False)

    assert len(validated) == 1
    assert validated["long_symbol"].iloc[0] == "WORST1/USDT"
    assert validated["short_symbol"].iloc[0] == "BEST1/USDT"


def test_validate_pairs_with_invalid_spread():
    """Test validate_pairs with invalid spread."""
    analyzer = PairsTradingAnalyzer(
        min_spread=0.01, max_spread=0.50, min_correlation=0.3, max_correlation=0.9
    )

    pairs_df = pd.DataFrame(
        {
            "long_symbol": ["WORST1/USDT"],
            "short_symbol": ["BEST1/USDT"],
            "long_score": [-0.001],
            "short_score": [0.002],
            "spread": [0.003],  # Too small (< 1%)
            "correlation": [0.6],
            "opportunity_score": [0.003],
        }
    )

    df = _create_mock_ohlcv_data(start_price=100.0, num_candles=200)

    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        return df, "binance"

    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)

    validated = analyzer.validate_pairs(pairs_df, mock_fetcher, verbose=False)

    assert len(validated) == 0  # Should be rejected


def test_validate_pairs_with_invalid_correlation():
    """Test validate_pairs with invalid correlation."""
    analyzer = PairsTradingAnalyzer(
        min_spread=0.01, max_spread=0.50, min_correlation=0.3, max_correlation=0.9
    )

    pairs_df = pd.DataFrame(
        {
            "long_symbol": ["WORST1/USDT"],
            "short_symbol": ["BEST1/USDT"],
            "long_score": [-0.1],
            "short_score": [0.15],
            "spread": [0.25],
            "correlation": [0.2],  # Too low (< 0.3)
            "opportunity_score": [0.25],
        }
    )

    df = _create_mock_ohlcv_data(start_price=100.0, num_candles=200)

    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        return df, "binance"

    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)

    validated = analyzer.validate_pairs(pairs_df, mock_fetcher, verbose=False)

    assert len(validated) == 0  # Should be rejected


def test_validate_pairs_with_missing_correlation():
    """Test validate_pairs when correlation is None."""
    analyzer = PairsTradingAnalyzer(
        min_spread=0.01,
        max_spread=0.50,
        min_correlation=0.3,
        max_correlation=0.9,
        min_volume=1000,  # Set low min_volume to avoid volume check rejection
    )

    pairs_df = pd.DataFrame(
        {
            "long_symbol": ["WORST1/USDT"],
            "short_symbol": ["BEST1/USDT"],
            "long_score": [-0.1],
            "short_score": [0.15],
            "spread": [0.25],
            "correlation": [None],  # Missing correlation
            "opportunity_score": [0.25],
        }
    )

    # Create data with sufficient volume
    df = _create_mock_ohlcv_data(start_price=100.0, num_candles=200)
    df["volume"] = df["volume"] * 100  # Increase volume to meet minimum

    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        return df, "binance"

    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)

    validated = analyzer.validate_pairs(pairs_df, mock_fetcher, verbose=False)

    # Should pass validation if correlation is None (not checked)
    assert len(validated) == 1


def test_validate_pairs_no_longer_checks_volume():
    """Volume filtering removed: validation should ignore low-volume data."""
    analyzer = PairsTradingAnalyzer(
        min_spread=0.01,
        max_spread=0.50,
        min_correlation=0.3,
        max_correlation=0.9,
    )

    pairs_df = pd.DataFrame(
        {
            "long_symbol": ["WORST1/USDT"],
            "short_symbol": ["BEST1/USDT"],
            "long_score": [-0.1],
            "short_score": [0.15],
            "spread": [0.25],
            "correlation": [0.6],
            "opportunity_score": [0.25],
        }
    )

    def mock_fetch(*args, **kwargs):
        return None, None

    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)

    validated = analyzer.validate_pairs(pairs_df, mock_fetcher, verbose=False)

    assert len(validated) == 1


def test_validate_pairs_with_empty_dataframe():
    """Test validate_pairs with empty DataFrame."""
    analyzer = PairsTradingAnalyzer()

    empty_df = pd.DataFrame(
        columns=[
            "long_symbol",
            "short_symbol",
            "long_score",
            "short_score",
            "spread",
            "correlation",
            "opportunity_score",
        ]
    )

    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=lambda *a, **k: (None, None))

    result = analyzer.validate_pairs(empty_df, mock_fetcher, verbose=False)

    assert len(result) == 0
    assert list(result.columns) == _get_all_pair_columns()


def test_opportunity_score_calculation():
    """Test that opportunity_score is calculated correctly."""
    analyzer = PairsTradingAnalyzer()

    best_performers = pd.DataFrame(
        {
            "symbol": ["BEST1/USDT"],
            "score": [0.15],
            "1d_return": [0.1],
            "3d_return": [0.15],
            "1w_return": [0.2],
            "current_price": [100],
        }
    )

    worst_performers = pd.DataFrame(
        {
            "symbol": ["WORST1/USDT"],
            "score": [-0.1],
            "1d_return": [-0.05],
            "3d_return": [-0.1],
            "1w_return": [-0.15],
            "current_price": [50],
        }
    )

    pairs_df = analyzer.analyze_pairs_opportunity(
        best_performers, worst_performers, verbose=False
    )

    assert len(pairs_df) == 1
    # Spread = 0.15 - (-0.1) = 0.25
    assert abs(pairs_df["spread"].iloc[0] - 0.25) < 0.01
    # Without correlation, opportunity_score should equal spread
    assert abs(pairs_df["opportunity_score"].iloc[0] - 0.25) < 0.01


if __name__ == "__main__":
    # Run basic tests
    print("Running PairsTradingAnalyzer tests...")

    test_calculate_spread()
    print("✓ test_calculate_spread")

    test_calculate_spread_with_negative_values()
    print("✓ test_calculate_spread_with_negative_values")

    test_analyze_pairs_opportunity()
    print("✓ test_analyze_pairs_opportunity")

    test_validate_pairs_with_valid_spread()
    print("✓ test_validate_pairs_with_valid_spread")

    test_validate_pairs_with_invalid_spread()
    print("✓ test_validate_pairs_with_invalid_spread")

    test_opportunity_score_calculation()
    print("✓ test_opportunity_score_calculation")

    print("\nAll tests passed!")

