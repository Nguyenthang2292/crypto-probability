"""
Tests for utils module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend.utils import rate_of_change, diflen, exp_growth


def test_rate_of_change_basic():
    """Test that rate_of_change calculates percentage change correctly."""
    prices = pd.Series([100.0, 102.0, 105.0, 103.0, 107.0])
    result = rate_of_change(prices)
    
    assert len(result) == len(prices)
    assert pd.isna(result.iloc[0])  # First value should be NaN
    assert np.isclose(result.iloc[1], 0.02)  # (102-100)/100 = 0.02
    assert np.isclose(result.iloc[2], 0.0294117647, rtol=1e-5)  # (105-102)/102


def test_rate_of_change_empty_series():
    """Test that rate_of_change handles empty series."""
    prices = pd.Series([], dtype=float)
    result = rate_of_change(prices)
    
    assert len(result) == 0


def test_rate_of_change_single_value():
    """Test that rate_of_change handles single value."""
    prices = pd.Series([100.0])
    result = rate_of_change(prices)
    
    assert len(result) == 1
    assert pd.isna(result.iloc[0])


def test_rate_of_change_with_nan():
    """Test that rate_of_change handles NaN values."""
    prices = pd.Series([100.0, np.nan, 105.0, 103.0])
    result = rate_of_change(prices)
    
    assert len(result) == len(prices)
    assert pd.isna(result.iloc[0])  # First value is always NaN
    # pandas pct_change may fill NaN, so we just check it doesn't crash


def test_diflen_narrow():
    """Test diflen with Narrow robustness."""
    length = 20
    L1, L2, L3, L4, L_1, L_2, L_3, L_4 = diflen(length, robustness="Narrow")
    
    assert L1 == 21
    assert L2 == 22
    assert L3 == 23
    assert L4 == 24
    assert L_1 == 19
    assert L_2 == 18
    assert L_3 == 17
    assert L_4 == 16


def test_diflen_medium():
    """Test diflen with Medium robustness."""
    length = 20
    L1, L2, L3, L4, L_1, L_2, L_3, L_4 = diflen(length, robustness="Medium")
    
    assert L1 == 21
    assert L2 == 22
    assert L3 == 24
    assert L4 == 26
    assert L_1 == 19
    assert L_2 == 18
    assert L_3 == 16
    assert L_4 == 14


def test_diflen_wide():
    """Test diflen with Wide robustness."""
    length = 20
    L1, L2, L3, L4, L_1, L_2, L_3, L_4 = diflen(length, robustness="Wide")
    
    assert L1 == 21
    assert L2 == 23
    assert L3 == 25
    assert L4 == 27
    assert L_1 == 19
    assert L_2 == 17
    assert L_3 == 15
    assert L_4 == 13


def test_diflen_default_medium():
    """Test that diflen defaults to Medium robustness."""
    length = 20
    result_default = diflen(length)
    result_medium = diflen(length, robustness="Medium")
    
    assert result_default == result_medium


def test_diflen_none_robustness():
    """Test that diflen handles None robustness."""
    length = 20
    result = diflen(length, robustness=None)
    result_medium = diflen(length, robustness="Medium")
    
    assert result == result_medium


def test_diflen_invalid_robustness():
    """Test that diflen defaults to Medium for invalid robustness."""
    length = 20
    result = diflen(length, robustness="Invalid")
    result_medium = diflen(length, robustness="Medium")
    
    assert result == result_medium


def test_exp_growth_basic():
    """Test that exp_growth calculates exponential growth correctly."""
    index = pd.RangeIndex(0, 10)
    L = 0.02
    result = exp_growth(L=L, index=index, cutout=0)
    
    assert len(result) == 10
    # bar_index=0 -> bars=1.0, so e^(L * (1 - 0)) = e^0.02 ≈ 1.0202
    assert np.isclose(result.iloc[0], np.e ** (L * 1.0), rtol=1e-5)
    # bar_index=1 -> bars=1, e^(L * (1 - 0)) = e^0.02 (same as index 0)
    assert np.isclose(result.iloc[1], np.e ** (L * 1.0), rtol=1e-5)
    # bar_index=2 -> bars=2, e^(L * (2 - 0)) = e^0.04, should be > index 0
    assert result.iloc[2] > result.iloc[0]


def test_exp_growth_with_cutout():
    """Test that exp_growth respects cutout parameter."""
    index = pd.RangeIndex(0, 10)
    L = 0.02
    cutout = 3
    result = exp_growth(L=L, index=index, cutout=cutout)
    
    assert len(result) == 10
    # Values before cutout should be 1.0
    for i in range(cutout):
        assert np.isclose(result.iloc[i], 1.0)
    # At cutout: bars=3 >= 3, e^(L * (3 - 3)) = e^0 = 1.0
    assert np.isclose(result.iloc[cutout], 1.0)
    # After cutout: bars=4 >= 3, e^(L * (4 - 3)) = e^0.02 > 1.0
    assert result.iloc[cutout + 1] > 1.0
    assert result.iloc[cutout + 1] > result.iloc[cutout]  # Should increase


def test_exp_growth_empty_index():
    """Test that exp_growth handles empty index."""
    index = pd.RangeIndex(0, 0)
    result = exp_growth(L=0.02, index=index)
    
    assert len(result) == 0


def test_exp_growth_none_index():
    """Test that exp_growth handles None index."""
    result = exp_growth(L=0.02, index=None)
    
    assert len(result) == 0


def test_exp_growth_zero_lambda():
    """Test that exp_growth handles zero lambda."""
    index = pd.RangeIndex(0, 10)
    result = exp_growth(L=0.0, index=index, cutout=0)
    
    assert len(result) == 10
    # With L=0, e^(0 * bar_index) = 1.0 for all bars >= cutout
    assert np.isclose(result.iloc[0], 1.0)
    assert np.isclose(result.iloc[1], 1.0)


def test_exp_growth_negative_lambda():
    """Test that exp_growth handles negative lambda."""
    index = pd.RangeIndex(0, 10)
    L = -0.02
    result = exp_growth(L=L, index=index, cutout=0)
    
    assert len(result) == 10
    # With negative L, bar_index=0 -> bars=1.0, e^(-0.02 * 1) ≈ 0.9802
    # bar_index=1 -> bars=1, e^(-0.02 * 1) ≈ 0.9802 (same as index 0)
    # bar_index=2 -> bars=2, e^(-0.02 * 2) ≈ 0.9608
    # So values should decrease over time (index 2 < index 0)
    assert result.iloc[2] < result.iloc[0]
    assert result.iloc[1] == result.iloc[0]  # Same value (bars=1 for both)

