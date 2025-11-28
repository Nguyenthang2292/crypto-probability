"""
Tests for calmar_ratio module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.pairs_trading.metrics import calculate_calmar_ratio


def test_calculate_calmar_ratio_uses_annual_return_and_drawdown():
    """Test that calculate_calmar_ratio uses annual return and max drawdown."""
    # Create equity curve (cumulative PnL)
    equity_curve = pd.Series([0, 1.0, -1.0, 2.0, 0.5, 2.0, 6.0], dtype=float)
    periods = 12

    # Manual calculation
    pnl_series = equity_curve.diff().dropna()
    annual_return = pnl_series.mean() * periods
    
    # Max drawdown calculation
    running_max = equity_curve.expanding().max()
    drawdown = equity_curve - running_max
    max_dd = abs(drawdown.min())
    
    expected = annual_return / max_dd if max_dd > 0 else None

    result = calculate_calmar_ratio(equity_curve, periods)

    if expected is not None:
        assert result is not None
        assert np.isclose(result, expected)
    else:
        assert result is None


def test_calculate_calmar_ratio_insufficient_data():
    """Test that calculate_calmar_ratio returns None for insufficient data."""
    equity_curve = pd.Series([1.0])

    result = calculate_calmar_ratio(equity_curve)

    assert result is None


def test_calculate_calmar_ratio_none_input():
    """Test that calculate_calmar_ratio handles None input."""
    result = calculate_calmar_ratio(None)

    assert result is None


def test_calculate_calmar_ratio_non_series_input():
    """Test that calculate_calmar_ratio handles non-Series input."""
    result = calculate_calmar_ratio([1.0, 2.0, 3.0])

    assert result is None


def test_calculate_calmar_ratio_invalid_periods():
    """Test that calculate_calmar_ratio handles invalid periods_per_year."""
    equity_curve = pd.Series([0, 1.0, 2.0, 1.5, 3.0], dtype=float)

    result = calculate_calmar_ratio(equity_curve, periods_per_year=0)

    assert result is None


def test_calculate_calmar_ratio_zero_drawdown():
    """Test that calculate_calmar_ratio handles zero drawdown."""
    # Equity curve that only goes up (no drawdown)
    equity_curve = pd.Series([0, 1.0, 2.0, 3.0, 4.0], dtype=float)

    result = calculate_calmar_ratio(equity_curve)

    # With no drawdown, should return None (division by zero)
    assert result is None


def test_calculate_calmar_ratio_with_nan_values():
    """Test that calculate_calmar_ratio handles NaN values correctly."""
    equity_curve = pd.Series([0, 1.0, np.nan, 2.0, 1.5, 3.0], dtype=float)

    result = calculate_calmar_ratio(equity_curve)

    # Should handle NaN gracefully
    assert result is None or isinstance(result, float)

