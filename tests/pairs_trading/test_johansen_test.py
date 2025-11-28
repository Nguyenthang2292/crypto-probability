"""
Tests for johansen_test module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.pairs_trading.metrics import calculate_johansen_test


def test_calculate_johansen_test_with_stub(monkeypatch):
    """Test that calculate_johansen_test works with stubbed coint_johansen."""
    class DummyResult:
        def __init__(self):
            self.lr1 = np.array([20.0])
            self.cvt = np.array([[15.0, 18.0, 25.0]])

    def fake_coint_johansen(data, det_order, k_ar_diff):
        return DummyResult()

    from modules.pairs_trading.metrics.statistical_tests import johansen_test
    monkeypatch.setattr(johansen_test, "coint_johansen", fake_coint_johansen)

    price1 = pd.Series(np.arange(60, dtype=float))
    price2 = pd.Series(np.arange(60, dtype=float) * 0.5)

    result = calculate_johansen_test(price1, price2, min_points=30, confidence=0.95)

    assert result == {
        "johansen_trace_stat": 20.0,
        "johansen_critical_value": 18.0,
        "is_johansen_cointegrated": True,
    }


def test_calculate_johansen_test_insufficient_data():
    """Test that calculate_johansen_test returns None for insufficient data."""
    price1 = pd.Series([1.0, 2.0, 3.0])
    price2 = pd.Series([2.0, 4.0, 6.0])

    result = calculate_johansen_test(price1, price2, min_points=50)

    assert result is None


def test_calculate_johansen_test_none_input():
    """Test that calculate_johansen_test handles None input."""
    price1 = pd.Series([1.0, 2.0, 3.0])

    result = calculate_johansen_test(None, price1)
    assert result is None

    result = calculate_johansen_test(price1, None)
    assert result is None


def test_calculate_johansen_test_non_series_input():
    """Test that calculate_johansen_test handles non-Series input."""
    price1 = pd.Series([1.0, 2.0, 3.0] * 20)

    result = calculate_johansen_test([1.0, 2.0, 3.0], price1)
    assert result is None

    result = calculate_johansen_test(price1, [1.0, 2.0, 3.0])
    assert result is None


def test_calculate_johansen_test_invalid_parameters():
    """Test that calculate_johansen_test handles invalid parameters."""
    price1 = pd.Series(np.arange(60, dtype=float))
    price2 = pd.Series(np.arange(60, dtype=float) * 0.5)

    # Invalid min_points
    result = calculate_johansen_test(price1, price2, min_points=0)
    assert result is None

    # Invalid confidence
    result = calculate_johansen_test(price1, price2, confidence=0.85)
    assert result is None

    # Invalid k_ar_diff
    result = calculate_johansen_test(price1, price2, k_ar_diff=-1)
    assert result is None

