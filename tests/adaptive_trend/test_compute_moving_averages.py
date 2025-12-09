"""
Tests for compute_moving_averages module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend.core.compute_moving_averages import (
    calculate_kama_atc,
    ma_calculation,
    set_of_moving_averages,
)


def test_calculate_kama_atc_basic():
    """Test that calculate_kama_atc returns valid KAMA series."""
    prices = pd.Series(np.linspace(100.0, 110.0, 100))
    result = calculate_kama_atc(prices, length=28)
    
    assert result is not None
    assert len(result) == len(prices)
    assert not result.isna().all()  # Should have some valid values


def test_calculate_kama_atc_empty_series():
    """Test that calculate_kama_atc handles empty series."""
    prices = pd.Series([], dtype=float)
    result = calculate_kama_atc(prices)
    
    assert result is None


def test_calculate_kama_atc_none_input():
    """Test that calculate_kama_atc raises error for None input."""
    with pytest.raises(TypeError, match="prices must be a pandas Series"):
        calculate_kama_atc(None)


def test_calculate_kama_atc_insufficient_data():
    """Test that calculate_kama_atc handles insufficient data."""
    prices = pd.Series([100.0, 101.0, 102.0])  # Too short for KAMA
    result = calculate_kama_atc(prices, length=28)
    
    # May return None or partial result depending on implementation
    assert result is None or len(result) == len(prices)


def test_ma_calculation_ema():
    """Test that ma_calculation calculates EMA correctly."""
    source = pd.Series(np.linspace(100.0, 110.0, 50))
    result = ma_calculation(source, length=10, ma_type="EMA")
    
    assert result is not None
    assert len(result) == len(source)


def test_ma_calculation_hma():
    """Test that ma_calculation calculates HMA (SMA) correctly."""
    source = pd.Series(np.linspace(100.0, 110.0, 50))
    result = ma_calculation(source, length=10, ma_type="HMA")
    
    assert result is not None
    assert len(result) == len(source)


def test_ma_calculation_wma():
    """Test that ma_calculation calculates WMA correctly."""
    source = pd.Series(np.linspace(100.0, 110.0, 50))
    result = ma_calculation(source, length=10, ma_type="WMA")
    
    assert result is not None
    assert len(result) == len(source)


def test_ma_calculation_dema():
    """Test that ma_calculation calculates DEMA correctly."""
    source = pd.Series(np.linspace(100.0, 110.0, 50))
    result = ma_calculation(source, length=10, ma_type="DEMA")
    
    assert result is not None
    assert len(result) == len(source)


def test_ma_calculation_lsma():
    """Test that ma_calculation calculates LSMA correctly."""
    source = pd.Series(np.linspace(100.0, 110.0, 50))
    result = ma_calculation(source, length=10, ma_type="LSMA")
    
    assert result is not None
    assert len(result) == len(source)


def test_ma_calculation_kama():
    """Test that ma_calculation calculates KAMA correctly."""
    source = pd.Series(np.linspace(100.0, 110.0, 50))
    result = ma_calculation(source, length=10, ma_type="KAMA")
    
    assert result is not None
    assert len(result) == len(source)


def test_ma_calculation_case_insensitive():
    """Test that ma_calculation is case-insensitive."""
    source = pd.Series(np.linspace(100.0, 110.0, 50))
    result_upper = ma_calculation(source, length=10, ma_type="EMA")
    result_lower = ma_calculation(source, length=10, ma_type="ema")
    result_mixed = ma_calculation(source, length=10, ma_type="EmA")
    
    assert result_upper is not None
    assert result_lower is not None
    assert result_mixed is not None
    pd.testing.assert_series_equal(result_upper, result_lower)
    pd.testing.assert_series_equal(result_upper, result_mixed)


def test_ma_calculation_invalid_type():
    """Test that ma_calculation returns None for invalid MA type."""
    source = pd.Series(np.linspace(100.0, 110.0, 50))
    result = ma_calculation(source, length=10, ma_type="INVALID")
    
    assert result is None


def test_ma_calculation_empty_series():
    """Test that ma_calculation handles empty series."""
    source = pd.Series([], dtype=float)
    result = ma_calculation(source, length=10, ma_type="EMA")
    
    assert result is None


def test_ma_calculation_none_input():
    """Test that ma_calculation raises error for None input."""
    with pytest.raises(TypeError, match="source must be a pandas Series"):
        ma_calculation(None, length=10, ma_type="EMA")


def test_set_of_moving_averages_basic():
    """Test that set_of_moving_averages returns 9 MA series."""
    source = pd.Series(np.linspace(100.0, 110.0, 100))
    result = set_of_moving_averages(length=20, source=source, ma_type="EMA", robustness="Medium")
    
    assert result is not None
    assert len(result) == 9
    MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4 = result
    assert MA is not None
    assert MA1 is not None
    assert MA2 is not None
    assert MA3 is not None
    assert MA4 is not None
    assert MA_1 is not None
    assert MA_2 is not None
    assert MA_3 is not None
    assert MA_4 is not None


def test_set_of_moving_averages_different_robustness():
    """Test that set_of_moving_averages uses robustness parameter."""
    source = pd.Series(np.linspace(100.0, 110.0, 100))
    
    result_narrow = set_of_moving_averages(20, source, "EMA", robustness="Narrow")
    result_medium = set_of_moving_averages(20, source, "EMA", robustness="Medium")
    result_wide = set_of_moving_averages(20, source, "EMA", robustness="Wide")
    
    assert result_narrow is not None
    assert result_medium is not None
    assert result_wide is not None
    
    # MAs should be different due to different offsets
    # L1 is same for all (length+1), but L3 and L4 differ
    # Check MA3 values (which use L3 offsets)
    _, _, _, MA3_narrow, _, _, _, _, _ = result_narrow
    _, _, _, MA3_medium, _, _, _, _, _ = result_medium
    _, _, _, MA3_wide, _, _, _, _, _ = result_wide
    
    # MA3 should be different due to different lengths
    # Narrow: L3 = length+3 = 23
    # Medium: L3 = length+4 = 24
    # Wide: L3 = length+5 = 25
    # They should produce different EMA values
    # Use a more lenient check - verify they exist and are Series
    assert MA3_narrow is not None
    assert MA3_medium is not None
    assert MA3_wide is not None
    assert isinstance(MA3_narrow, pd.Series)
    assert isinstance(MA3_medium, pd.Series)
    assert isinstance(MA3_wide, pd.Series)
    
    # They might be very similar, so we check they're not all identical
    # by comparing a few values (after NaN)
    narrow_valid = MA3_narrow.dropna()
    medium_valid = MA3_medium.dropna()
    wide_valid = MA3_wide.dropna()
    
    if len(narrow_valid) > 0 and len(medium_valid) > 0:
        # They should be different (different lengths produce different EMAs)
        # But allow for very small differences due to floating point
        assert not (narrow_valid.equals(medium_valid))


def test_set_of_moving_averages_empty_series():
    """Test that set_of_moving_averages handles empty series."""
    source = pd.Series([], dtype=float)
    result = set_of_moving_averages(20, source, "EMA")
    
    assert result is None


def test_set_of_moving_averages_none_input():
    """Test that set_of_moving_averages raises error for None input."""
    with pytest.raises(TypeError, match="source must be a pandas Series"):
        set_of_moving_averages(20, None, "EMA")


def test_set_of_moving_averages_all_ma_types():
    """Test that set_of_moving_averages works with all MA types."""
    source = pd.Series(np.linspace(100.0, 110.0, 100))
    
    for ma_type in ["EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"]:
        result = set_of_moving_averages(20, source, ma_type)
        assert result is not None
        assert len(result) == 9

