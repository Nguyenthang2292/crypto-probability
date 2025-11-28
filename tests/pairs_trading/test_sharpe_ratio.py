"""
Tests for sharpe_ratio module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.pairs_trading.metrics import calculate_spread_sharpe


def test_calculate_spread_sharpe_matches_manual_computation():
    """Test that calculate_spread_sharpe matches manual computation."""
    # Create PnL series (difference series, not spread directly)
    pnl_series = pd.Series([2.0, -1.0, 4.0, 2.0, 3.0], dtype=float)
    periods = 4

    # Manual calculation: Sharpe = (mean / std) * sqrt(periods)
    expected_sharpe = (pnl_series.mean() / pnl_series.std()) * np.sqrt(periods)

    result = calculate_spread_sharpe(pnl_series, periods)

    assert result is not None
    assert np.isclose(result, expected_sharpe)


def test_calculate_spread_sharpe_handles_zero_std():
    """Test that calculate_spread_sharpe handles zero std correctly."""
    spread = pd.Series([100] * 10, dtype=float)
    assert calculate_spread_sharpe(spread, 4) is None


def test_calculate_spread_sharpe_insufficient_data():
    """Test that calculate_spread_sharpe returns None for insufficient data."""
    spread = pd.Series([100])
    
    result = calculate_spread_sharpe(spread, 4)
    
    assert result is None


def test_calculate_spread_sharpe_none_input():
    """Test that calculate_spread_sharpe handles None input."""
    result = calculate_spread_sharpe(None, 4)
    
    assert result is None


def test_calculate_spread_sharpe_non_series_input():
    """Test that calculate_spread_sharpe handles non-Series input."""
    result = calculate_spread_sharpe([100, 102, 103], 4)
    
    assert result is None


def test_calculate_spread_sharpe_with_nan_values():
    """Test that calculate_spread_sharpe handles NaN values correctly."""
    spread = pd.Series([100, 102, np.nan, 105, 107, 110], dtype=float)
    
    result = calculate_spread_sharpe(spread, 4)
    
    # Should still compute valid result after dropping NaN
    assert result is not None or spread.dropna().std() == 0


def test_calculate_spread_sharpe_invalid_periods():
    """Test that calculate_spread_sharpe handles invalid periods_per_year."""
    spread = pd.Series([100, 102, 101, 105, 107, 110], dtype=float)
    
    result = calculate_spread_sharpe(spread, periods_per_year=0)
    
    assert result is None


def test_calculate_spread_sharpe_with_risk_free_rate():
    """Test that calculate_spread_sharpe works with risk-free rate."""
    spread = pd.Series([100, 102, 101, 105, 107, 110], dtype=float)
    
    sharpe_no_rf = calculate_spread_sharpe(spread, periods_per_year=4, risk_free_rate=0.0)
    sharpe_with_rf = calculate_spread_sharpe(spread, periods_per_year=4, risk_free_rate=0.05)
    
    # Sharpe with risk-free rate should be lower than without (for positive returns)
    assert sharpe_no_rf is not None
    assert sharpe_with_rf is not None
    assert sharpe_with_rf < sharpe_no_rf

