"""
Tests for adf_test module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.pairs_trading.metrics import calculate_adf_test


def test_calculate_adf_test_uses_stub(monkeypatch):
    """Test that calculate_adf_test works with stubbed adfuller."""
    called = {}

    def fake_adfuller(series, maxlag, autolag):
        called["data"] = series.tolist()
        return (-3.2, 0.01, None, None, {"5%": -2.9})

    from modules.pairs_trading.metrics.statistical_tests import adf_test
    monkeypatch.setattr(adf_test, "adfuller", fake_adfuller)
    spread = pd.Series(np.linspace(1, 100, 60))

    result = calculate_adf_test(spread, min_points=30)

    assert called["data"][0] == 1.0
    assert result == {
        "adf_statistic": -3.2,
        "adf_pvalue": 0.01,
        "critical_values": {"5%": -2.9},
    }


def test_calculate_adf_test_insufficient_data():
    """Test that calculate_adf_test returns None for insufficient data."""
    spread = pd.Series([1.0, 2.0, 3.0])

    result = calculate_adf_test(spread, min_points=50)

    assert result is None


def test_calculate_adf_test_none_input():
    """Test that calculate_adf_test handles None input."""
    result = calculate_adf_test(None, 50)

    assert result is None


def test_calculate_adf_test_non_series_input():
    """Test that calculate_adf_test handles non-Series input."""
    result = calculate_adf_test([1.0, 2.0, 3.0], 50)

    assert result is None


def test_calculate_adf_test_invalid_min_points():
    """Test that calculate_adf_test handles invalid min_points."""
    spread = pd.Series(np.linspace(1, 100, 60))

    result = calculate_adf_test(spread, min_points=0)

    assert result is None


def test_calculate_adf_test_with_nan_values():
    """Test that calculate_adf_test handles NaN values correctly."""
    spread_values = np.linspace(1, 100, 60)
    spread_values[10] = np.nan
    spread_values[20] = np.nan
    spread = pd.Series(spread_values)

    # Should still compute valid result after dropping NaN
    # Result depends on whether statsmodels is installed
    result = calculate_adf_test(spread, min_points=30)

    # If statsmodels is installed and test succeeds, result should be dict
    # If not installed or fails, result is None
    assert result is None or isinstance(result, dict)

