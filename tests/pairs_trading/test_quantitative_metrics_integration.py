"""
Test cases for quantitative metrics integration in pairs trading.

Tests the new features:
- Quantitative score display
- Cointegration status display
- Sorting by quantitative_score
- Validation filters for quantitative metrics
- Verbose mode display
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from types import SimpleNamespace

from modules.pairs_trading.pairs_analyzer import PairsTradingAnalyzer, _get_all_pair_columns


def _create_mock_ohlcv_data(
    start_price: float = 100.0,
    num_candles: int = 200,
    trend: str = "up",
    volatility: float = 0.01,
) -> pd.DataFrame:
    """Create mock OHLCV DataFrame for testing."""
    timestamps = pd.date_range(
        start="2024-01-01", periods=num_candles, freq="1h", tz="UTC"
    )

    if trend == "up":
        trend_factor = np.linspace(0, 0.2, num_candles)
    elif trend == "down":
        trend_factor = np.linspace(0, -0.2, num_candles)
    else:
        trend_factor = np.zeros(num_candles)

    np.random.seed(42)
    returns = np.random.normal(0, volatility, num_candles) + trend_factor / num_candles
    prices = start_price * (1 + returns).cumprod()

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


def test_quantitative_score_in_dataframe():
    """Test that quantitative_score is included in pairs DataFrame."""
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

    df1 = _create_mock_ohlcv_data(start_price=100.0, num_candles=200)
    df2 = _create_mock_ohlcv_data(start_price=50.0, num_candles=200)

    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        if symbol == "BEST1/USDT":
            return df1, "binance"
        elif symbol == "WORST1/USDT":
            return df2, "binance"
        return None, None

    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)

    pairs_df = analyzer.analyze_pairs_opportunity(
        best_performers, worst_performers, data_fetcher=mock_fetcher, verbose=False
    )

    assert "quantitative_score" in pairs_df.columns
    assert len(pairs_df) > 0

    # Check that quantitative_score is numeric or None
    quant_scores = pairs_df["quantitative_score"].dropna()
    if len(quant_scores) > 0:
        assert all(isinstance(s, (int, float)) for s in quant_scores)
        assert all(0 <= s <= 100 for s in quant_scores)


def test_quantitative_metrics_columns_present():
    """Test that all quantitative metrics columns are present in DataFrame."""
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

    df1 = _create_mock_ohlcv_data(start_price=100.0, num_candles=200)
    df2 = _create_mock_ohlcv_data(start_price=50.0, num_candles=200)

    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        if symbol == "BEST1/USDT":
            return df1, "binance"
        elif symbol == "WORST1/USDT":
            return df2, "binance"
        return None, None

    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)

    pairs_df = analyzer.analyze_pairs_opportunity(
        best_performers, worst_performers, data_fetcher=mock_fetcher, verbose=False
    )

    # Check that all expected columns are present
    expected_columns = [
        "quantitative_score",
        "adf_pvalue",
        "is_cointegrated",
        "half_life",
        "hurst_exponent",
        "spread_sharpe",
        "max_drawdown",
        "hedge_ratio",
    ]

    for col in expected_columns:
        assert col in pairs_df.columns, f"Column {col} missing from DataFrame"


def test_validate_pairs_with_cointegration_requirement():
    """Test validate_pairs filters out non-cointegrated pairs when require_cointegration=True."""
    analyzer = PairsTradingAnalyzer(
        min_spread=0.01,
        max_spread=0.50,
        require_cointegration=True,
        max_half_life=100,
    )

    pairs_df = pd.DataFrame(
        [
            {
                "long_symbol": "WORST1/USDT",
                "short_symbol": "BEST1/USDT",
                "long_score": -0.1,
                "short_score": 0.15,
                "spread": 0.25,
                "correlation": 0.6,
                "opportunity_score": 0.25,
                "is_cointegrated": True,  # Cointegrated
                "half_life": 20,
                "quantitative_score": 75,
            },
            {
                "long_symbol": "WORST2/USDT",
                "short_symbol": "BEST2/USDT",
                "long_score": -0.1,
                "short_score": 0.15,
                "spread": 0.25,
                "correlation": 0.6,
                "opportunity_score": 0.25,
                "is_cointegrated": False,  # Not cointegrated
                "half_life": 20,
                "quantitative_score": 40,
            },
        ]
    )

    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        df = _create_mock_ohlcv_data(start_price=100.0, num_candles=200)
        return df, "binance"

    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)

    validated = analyzer.validate_pairs(pairs_df, mock_fetcher, verbose=False)

    # Should only keep cointegrated pair
    assert len(validated) == 1
    assert validated.iloc[0]["long_symbol"] == "WORST1/USDT"
    assert validated.iloc[0]["is_cointegrated"] == True


def test_validate_pairs_with_half_life_threshold():
    """Test validate_pairs filters pairs based on half_life threshold."""
    analyzer = PairsTradingAnalyzer(
        min_spread=0.01,
        max_spread=0.50,
        max_half_life=30,  # Only accept half-life <= 30
    )

    pairs_df = pd.DataFrame(
        [
            {
                "long_symbol": "WORST1/USDT",
                "short_symbol": "BEST1/USDT",
                "spread": 0.25,
                "correlation": 0.6,
                "opportunity_score": 0.25,
                "half_life": 20,  # Valid (< 30)
                "quantitative_score": 75,
            },
            {
                "long_symbol": "WORST2/USDT",
                "short_symbol": "BEST2/USDT",
                "spread": 0.25,
                "correlation": 0.6,
                "opportunity_score": 0.25,
                "half_life": 50,  # Invalid (> 30)
                "quantitative_score": 40,
            },
        ]
    )

    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        df = _create_mock_ohlcv_data(start_price=100.0, num_candles=200)
        return df, "binance"

    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)

    validated = analyzer.validate_pairs(pairs_df, mock_fetcher, verbose=False)

    # Should only keep pair with half_life <= 30
    assert len(validated) == 1
    assert validated.iloc[0]["half_life"] <= 30


def test_validate_pairs_with_min_quantitative_score():
    """Test validate_pairs filters pairs based on minimum quantitative_score."""
    analyzer = PairsTradingAnalyzer(
        min_spread=0.01,
        max_spread=0.50,
        min_quantitative_score=60,  # Only accept quantitative_score >= 60
    )

    pairs_df = pd.DataFrame(
        [
            {
                "long_symbol": "WORST1/USDT",
                "short_symbol": "BEST1/USDT",
                "spread": 0.25,
                "correlation": 0.6,
                "opportunity_score": 0.25,
                "quantitative_score": 75,  # Valid (>= 60)
            },
            {
                "long_symbol": "WORST2/USDT",
                "short_symbol": "BEST2/USDT",
                "spread": 0.25,
                "correlation": 0.6,
                "opportunity_score": 0.25,
                "quantitative_score": 45,  # Invalid (< 60)
            },
        ]
    )

    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        df = _create_mock_ohlcv_data(start_price=100.0, num_candles=200)
        return df, "binance"

    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)

    validated = analyzer.validate_pairs(pairs_df, mock_fetcher, verbose=False)

    # Should only keep pair with quantitative_score >= 60
    assert len(validated) == 1
    assert validated.iloc[0]["quantitative_score"] >= 60


def test_validate_pairs_with_hurst_threshold():
    """Test validate_pairs filters pairs based on Hurst exponent threshold."""
    analyzer = PairsTradingAnalyzer(
        min_spread=0.01,
        max_spread=0.50,
        hurst_threshold=0.5,  # Only accept Hurst < 0.5 (mean-reverting)
    )

    pairs_df = pd.DataFrame(
        [
            {
                "long_symbol": "WORST1/USDT",
                "short_symbol": "BEST1/USDT",
                "spread": 0.25,
                "correlation": 0.6,
                "opportunity_score": 0.25,
                "hurst_exponent": 0.3,  # Valid (< 0.5, mean-reverting)
                "quantitative_score": 75,
            },
            {
                "long_symbol": "WORST2/USDT",
                "short_symbol": "BEST2/USDT",
                "spread": 0.25,
                "correlation": 0.6,
                "opportunity_score": 0.25,
                "hurst_exponent": 0.6,  # Invalid (>= 0.5, trending)
                "quantitative_score": 40,
            },
        ]
    )

    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        df = _create_mock_ohlcv_data(start_price=100.0, num_candles=200)
        return df, "binance"

    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)

    validated = analyzer.validate_pairs(pairs_df, mock_fetcher, verbose=False)

    # Should only keep pair with Hurst < 0.5
    assert len(validated) == 1
    assert validated.iloc[0]["hurst_exponent"] < 0.5


def test_sorting_by_quantitative_score():
    """Test that pairs can be sorted by quantitative_score."""
    pairs_df = pd.DataFrame(
        [
            {
                "long_symbol": "WORST1/USDT",
                "short_symbol": "BEST1/USDT",
                "spread": 0.25,
                "correlation": 0.6,
                "opportunity_score": 0.25,
                "quantitative_score": 50,  # Lower score
            },
            {
                "long_symbol": "WORST2/USDT",
                "short_symbol": "BEST2/USDT",
                "spread": 0.20,
                "correlation": 0.5,
                "opportunity_score": 0.20,
                "quantitative_score": 80,  # Higher score
            },
        ]
    )

    # Sort by quantitative_score descending
    sorted_df = pairs_df.sort_values("quantitative_score", ascending=False).reset_index(drop=True)

    # First row should have higher quantitative_score
    assert sorted_df.iloc[0]["quantitative_score"] == 80
    assert sorted_df.iloc[1]["quantitative_score"] == 50


def test_sorting_by_opportunity_score():
    """Test that pairs can be sorted by opportunity_score (default behavior)."""
    pairs_df = pd.DataFrame(
        [
            {
                "long_symbol": "WORST1/USDT",
                "short_symbol": "BEST1/USDT",
                "spread": 0.25,
                "correlation": 0.6,
                "opportunity_score": 0.20,  # Lower score
                "quantitative_score": 80,
            },
            {
                "long_symbol": "WORST2/USDT",
                "short_symbol": "BEST2/USDT",
                "spread": 0.30,
                "correlation": 0.7,
                "opportunity_score": 0.30,  # Higher score
                "quantitative_score": 50,
            },
        ]
    )

    # Sort by opportunity_score descending (default)
    sorted_df = pairs_df.sort_values("opportunity_score", ascending=False).reset_index(drop=True)

    # First row should have higher opportunity_score
    assert sorted_df.iloc[0]["opportunity_score"] == 0.30
    assert sorted_df.iloc[1]["opportunity_score"] == 0.20


def test_cointegration_status_display():
    """Test that is_cointegrated column is present and contains boolean values."""
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

    df1 = _create_mock_ohlcv_data(start_price=100.0, num_candles=200)
    df2 = _create_mock_ohlcv_data(start_price=50.0, num_candles=200)

    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        if symbol == "BEST1/USDT":
            return df1, "binance"
        elif symbol == "WORST1/USDT":
            return df2, "binance"
        return None, None

    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)

    pairs_df = analyzer.analyze_pairs_opportunity(
        best_performers, worst_performers, data_fetcher=mock_fetcher, verbose=False
    )

    assert "is_cointegrated" in pairs_df.columns

    # Check that is_cointegrated contains boolean or None values
    coint_values = pairs_df["is_cointegrated"].dropna()
    if len(coint_values) > 0:
        assert all(isinstance(v, bool) for v in coint_values)


def test_validate_pairs_with_multiple_filters():
    """Test validate_pairs with multiple quantitative metric filters combined."""
    analyzer = PairsTradingAnalyzer(
        min_spread=0.01,
        max_spread=0.50,
        require_cointegration=True,
        max_half_life=30,
        hurst_threshold=0.5,
        min_quantitative_score=60,
    )

    pairs_df = pd.DataFrame(
        [
            {
                "long_symbol": "WORST1/USDT",
                "short_symbol": "BEST1/USDT",
                "spread": 0.25,
                "correlation": 0.6,
                "opportunity_score": 0.25,
                "is_cointegrated": True,
                "half_life": 20,
                "hurst_exponent": 0.3,
                "quantitative_score": 75,  # Valid: passes all filters
            },
            {
                "long_symbol": "WORST2/USDT",
                "short_symbol": "BEST2/USDT",
                "spread": 0.25,
                "correlation": 0.6,
                "opportunity_score": 0.25,
                "is_cointegrated": False,  # Fails: not cointegrated
                "half_life": 20,
                "hurst_exponent": 0.3,
                "quantitative_score": 75,
            },
            {
                "long_symbol": "WORST3/USDT",
                "short_symbol": "BEST3/USDT",
                "spread": 0.25,
                "correlation": 0.6,
                "opportunity_score": 0.25,
                "is_cointegrated": True,
                "half_life": 50,  # Fails: half_life > 30
                "hurst_exponent": 0.3,
                "quantitative_score": 75,
            },
            {
                "long_symbol": "WORST4/USDT",
                "short_symbol": "BEST4/USDT",
                "spread": 0.25,
                "correlation": 0.6,
                "opportunity_score": 0.25,
                "is_cointegrated": True,
                "half_life": 20,
                "hurst_exponent": 0.6,  # Fails: Hurst >= 0.5
                "quantitative_score": 75,
            },
            {
                "long_symbol": "WORST5/USDT",
                "short_symbol": "BEST5/USDT",
                "spread": 0.25,
                "correlation": 0.6,
                "opportunity_score": 0.25,
                "is_cointegrated": True,
                "half_life": 20,
                "hurst_exponent": 0.3,
                "quantitative_score": 45,  # Fails: quantitative_score < 60
            },
        ]
    )

    def mock_fetch(symbol, limit=None, timeframe=None, check_freshness=None):
        df = _create_mock_ohlcv_data(start_price=100.0, num_candles=200)
        return df, "binance"

    mock_fetcher = SimpleNamespace(fetch_ohlcv_with_fallback_exchange=mock_fetch)

    validated = analyzer.validate_pairs(pairs_df, mock_fetcher, verbose=False)

    # Should only keep the first pair (passes all filters)
    assert len(validated) == 1
    assert validated.iloc[0]["long_symbol"] == "WORST1/USDT"
    assert validated.iloc[0]["is_cointegrated"] == True
    assert validated.iloc[0]["half_life"] <= 30
    assert validated.iloc[0]["hurst_exponent"] < 0.5
    assert validated.iloc[0]["quantitative_score"] >= 60


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])

