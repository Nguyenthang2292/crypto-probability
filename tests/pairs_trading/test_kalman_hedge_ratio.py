"""
Tests for kalman_hedge_ratio module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.pairs_trading.metrics import calculate_kalman_hedge_ratio


def test_calculate_kalman_hedge_ratio_uses_stubbed_filter(monkeypatch):
    """Test that calculate_kalman_hedge_ratio uses KalmanFilter correctly."""
    class DummyKalman:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def filter(self, observations):
            steps = len(observations)
            means = np.column_stack([np.linspace(0.5, 1.5, steps), np.zeros(steps)])
            cov = np.zeros((steps, 2))
            return means, cov

    from modules.pairs_trading.metrics.hedge_ratios import kalman_hedge_ratio
    monkeypatch.setattr(kalman_hedge_ratio, "KalmanFilter", DummyKalman)

    price2 = pd.Series(np.linspace(1, 10, 40))
    price1 = pd.Series(np.linspace(2, 12, 40))

    beta = calculate_kalman_hedge_ratio(price1, price2)

    assert beta is not None
    assert np.isclose(beta, 1.5)


def test_calculate_kalman_hedge_ratio_insufficient_data():
    """Test that calculate_kalman_hedge_ratio returns None for insufficient data."""
    price1 = pd.Series([1, 2, 3, 4, 5])  # Less than 10 points
    price2 = pd.Series([2, 4, 6, 8, 10])

    result = calculate_kalman_hedge_ratio(price1, price2)

    assert result is None


def test_calculate_kalman_hedge_ratio_none_input():
    """Test that calculate_kalman_hedge_ratio handles None input."""
    price1 = pd.Series([1, 2, 3])
    result1 = calculate_kalman_hedge_ratio(None, price1)
    result2 = calculate_kalman_hedge_ratio(price1, None)
    
    assert result1 is None
    assert result2 is None


def test_calculate_kalman_hedge_ratio_non_series_input():
    """Test that calculate_kalman_hedge_ratio handles non-Series input."""
    price1 = pd.Series([1, 2, 3] * 10)
    price2 = [2, 4, 6] * 10
    
    result = calculate_kalman_hedge_ratio(price1, price2)
    
    assert result is None


def test_calculate_kalman_hedge_ratio_mismatched_lengths():
    """Test that calculate_kalman_hedge_ratio handles mismatched lengths."""
    price1 = pd.Series([1, 2, 3] * 10)
    price2 = pd.Series([2, 4] * 10)
    
    result = calculate_kalman_hedge_ratio(price1, price2)
    
    assert result is None


def test_calculate_kalman_hedge_ratio_invalid_delta():
    """Test that calculate_kalman_hedge_ratio handles invalid delta parameter."""
    price1 = pd.Series([1, 2, 3] * 10)
    price2 = pd.Series([2, 4, 6] * 10)
    
    result1 = calculate_kalman_hedge_ratio(price1, price2, delta=0)
    result2 = calculate_kalman_hedge_ratio(price1, price2, delta=1)
    result3 = calculate_kalman_hedge_ratio(price1, price2, delta=-1)
    
    assert result1 is None
    assert result2 is None
    assert result3 is None


def test_calculate_kalman_hedge_ratio_invalid_observation_covariance():
    """Test that calculate_kalman_hedge_ratio handles invalid observation_covariance."""
    price1 = pd.Series([1, 2, 3] * 10)
    price2 = pd.Series([2, 4, 6] * 10)
    
    result = calculate_kalman_hedge_ratio(price1, price2, observation_covariance=0)
    
    assert result is None

