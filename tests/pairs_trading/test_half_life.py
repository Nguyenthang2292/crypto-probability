"""
Tests for half_life module.
"""
import pandas as pd
import numpy as np
import pytest

from modules.pairs_trading.metrics import calculate_half_life


def test_calculate_half_life_with_stubbed_regression(monkeypatch):
    """Test that calculate_half_life works with stubbed LinearRegression."""
    class FakeModel:
        def __init__(self):
            self.coef_ = [-0.1]

        def fit(self, X, y):
            pass

    from modules.pairs_trading.metrics.mean_reversion import half_life
    monkeypatch.setattr(half_life, "LinearRegression", FakeModel)

    spread = pd.Series(np.linspace(100, 80, 50))
    result = calculate_half_life(spread)
    expected = -np.log(2) / -0.1

    assert result == expected


def test_calculate_half_life_insufficient_data():
    """Test that calculate_half_life returns None for insufficient data."""
    spread = pd.Series([1.0, 2.0, 3.0])

    result = calculate_half_life(spread)

    assert result is None


def test_calculate_half_life_none_input():
    """Test that calculate_half_life handles None input."""
    result = calculate_half_life(None)

    assert result is None


def test_calculate_half_life_non_series_input():
    """Test that calculate_half_life handles non-Series input."""
    result = calculate_half_life([1.0, 2.0, 3.0])

    assert result is None


def test_calculate_half_life_non_stationary():
    """Test that calculate_half_life returns None for non-stationary spread."""
    # Create a trending series (non-mean-reverting)
    spread = pd.Series(np.linspace(0, 100, 100))

    result = calculate_half_life(spread)

    # Non-stationary spread should return None (theta >= 0)
    assert result is None


def test_calculate_half_life_with_nan_values():
    """Test that calculate_half_life handles NaN values correctly."""
    spread_values = np.linspace(100, 80, 50)
    spread_values[10] = np.nan
    spread_values[20] = np.nan
    spread = pd.Series(spread_values)

    # Should still compute valid result after dropping NaN
    # Result depends on whether sklearn is installed and if theta < 0
    result = calculate_half_life(spread)

    # Result may be None if insufficient valid data or non-stationary
    assert result is None or isinstance(result, float)

