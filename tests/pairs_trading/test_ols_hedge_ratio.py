"""
Tests for ols_hedge_ratio module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.pairs_trading.metrics import calculate_ols_hedge_ratio


def test_calculate_ols_hedge_ratio_recovers_linear_beta():
    """Test that calculate_ols_hedge_ratio recovers the correct beta from linear relationship."""
    price2 = pd.Series(np.linspace(1, 20, 50))
    price1 = 2.5 * price2 + 5

    beta = calculate_ols_hedge_ratio(price1, price2)
    assert beta is not None
    assert np.isclose(beta, 2.5, atol=1e-2)


def test_calculate_ols_hedge_ratio_with_stubbed_regression(monkeypatch):
    """Test that calculate_ols_hedge_ratio uses LinearRegression correctly."""
    class FakeModel:
        def __init__(self, fit_intercept=True):
            self.fit_intercept = fit_intercept
            self.coef_ = np.array([1.5])

        def fit(self, X, y):
            pass

    from modules.pairs_trading.metrics.hedge_ratios import ols_hedge_ratio
    monkeypatch.setattr(ols_hedge_ratio, "LinearRegression", FakeModel)

    # Need at least 10 data points after alignment
    price1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    price2 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    beta = calculate_ols_hedge_ratio(price1, price2)
    assert beta == 1.5


def test_calculate_ols_hedge_ratio_insufficient_data():
    """Test that calculate_ols_hedge_ratio returns None for insufficient data."""
    price1 = pd.Series([1.0])
    price2 = pd.Series([1.0])

    beta = calculate_ols_hedge_ratio(price1, price2)
    assert beta is None


def test_calculate_ols_hedge_ratio_none_input():
    """Test that calculate_ols_hedge_ratio handles None input."""
    beta = calculate_ols_hedge_ratio(None, None)
    assert beta is None


def test_calculate_ols_hedge_ratio_with_nan_values():
    """Test that calculate_ols_hedge_ratio handles NaN values correctly."""
    # Need at least 10 data points after alignment and dropping NaN
    price1 = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    price2 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])

    # Should align and drop NaN, still have >= 10 valid points
    beta = calculate_ols_hedge_ratio(price1, price2)
    assert beta is not None
