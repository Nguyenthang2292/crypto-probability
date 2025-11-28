"""
Tests for zscore_stats module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.pairs_trading.metrics import calculate_zscore_stats


def test_calculate_zscore_stats_matches_manual_computation():
    """Test that calculate_zscore_stats matches manual computation."""
    spread = pd.Series(np.linspace(1.0, 3.0, 200))
    lookback = 50

    result = calculate_zscore_stats(spread, lookback)

    # Manual computation for comparison
    spread_clean = spread.dropna()
    rolling_mean = spread_clean.rolling(lookback, min_periods=lookback).mean()
    rolling_std = spread_clean.rolling(lookback, min_periods=lookback).std().replace(0, np.nan)
    zscore = ((spread_clean - rolling_mean) / rolling_std).dropna()

    assert result["mean_zscore"] == float(zscore.mean())
    assert result["std_zscore"] == float(zscore.std())
    assert result["current_zscore"] == float(zscore.iloc[-1])


def test_calculate_zscore_stats_with_nan_values():
    """Test that calculate_zscore_stats handles NaN values correctly."""
    spread = pd.Series([1.0, 2.0, np.nan, 3.0, 4.0, 5.0, np.nan, 6.0])
    lookback = 3

    result = calculate_zscore_stats(spread, lookback)

    # Should still compute valid statistics after dropping NaN
    assert result["mean_zscore"] is not None
    assert result["std_zscore"] is not None
    assert result["current_zscore"] is not None


def test_calculate_zscore_stats_insufficient_data():
    """Test that calculate_zscore_stats returns None for insufficient data."""
    spread = pd.Series([1.0, 2.0, 3.0])
    lookback = 50  # More than available data points

    result = calculate_zscore_stats(spread, lookback)

    assert result["mean_zscore"] is None
    assert result["std_zscore"] is None
    assert result["skewness"] is None
    assert result["kurtosis"] is None
    assert result["current_zscore"] is None


def test_calculate_zscore_stats_none_input():
    """Test that calculate_zscore_stats handles None input."""
    result = calculate_zscore_stats(None, 50)

    assert result["mean_zscore"] is None
    assert result["std_zscore"] is None
    assert result["skewness"] is None
    assert result["kurtosis"] is None
    assert result["current_zscore"] is None


def test_calculate_zscore_stats_non_series_input():
    """Test that calculate_zscore_stats handles non-Series input."""
    result = calculate_zscore_stats([1.0, 2.0, 3.0], 50)

    assert result["mean_zscore"] is None
    assert result["std_zscore"] is None
    assert result["skewness"] is None
    assert result["kurtosis"] is None
    assert result["current_zscore"] is None


def test_calculate_zscore_stats_skewness_kurtosis():
    """Test that skewness and kurtosis are calculated when sufficient data exists."""
    # Create spread with clear skewness
    rng = np.random.default_rng(42)
    spread = pd.Series(rng.normal(0, 1, 200))
    lookback = 50

    result = calculate_zscore_stats(spread, lookback)

    # Should have all metrics including skewness and kurtosis
    assert result["mean_zscore"] is not None
    assert result["std_zscore"] is not None
    assert result["skewness"] is not None
    assert result["kurtosis"] is not None
    assert result["current_zscore"] is not None


def test_calculate_zscore_stats_constant_spread():
    """Test that calculate_zscore_stats handles constant spread (zero std)."""
    # Constant spread will have std = 0, should be handled gracefully
    spread = pd.Series([1.0] * 200)
    lookback = 50

    result = calculate_zscore_stats(spread, lookback)

    # With constant spread, z-score cannot be calculated (division by zero)
    # Should return None for all metrics
    assert result["mean_zscore"] is None
    assert result["std_zscore"] is None
    assert result["skewness"] is None
    assert result["kurtosis"] is None
    assert result["current_zscore"] is None


def test_calculate_zscore_stats_default_lookback():
    """Test that calculate_zscore_stats uses default lookback when not specified."""
    spread = pd.Series(np.linspace(1.0, 3.0, 200))

    result = calculate_zscore_stats(spread)

    # Should work with default lookback (60)
    assert result["mean_zscore"] is not None
    assert result["std_zscore"] is not None
    assert result["current_zscore"] is not None

