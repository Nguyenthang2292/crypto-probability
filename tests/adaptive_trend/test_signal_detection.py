"""
Tests for signal_detection module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.adaptive_trend.core.signal_detection import crossover, crossunder, generate_signal_from_ma


def test_crossover_detects_upward_cross():
    """Test that crossover detects upward crossover correctly."""
    series_a = pd.Series([10.0, 10.0, 12.0, 13.0])  # Price
    series_b = pd.Series([11.0, 11.0, 11.0, 11.0])  # MA
    
    result = crossover(series_a, series_b)
    
    assert len(result) == len(series_a)
    assert result.iloc[0] == False  # 10 <= 11, 10 <= 11
    assert result.iloc[1] == False  # 10 <= 11, 10 <= 11
    assert result.iloc[2] == True   # 12 > 11 AND 10 <= 11 (crossover!)
    assert result.iloc[3] == False   # 13 > 11 BUT 12 > 11 (already crossed)


def test_crossover_no_crossover():
    """Test that crossover returns False when no crossover occurs."""
    series_a = pd.Series([10.0, 11.0, 12.0, 13.0])
    series_b = pd.Series([9.0, 9.5, 10.0, 10.5])  # Always below
    
    result = crossover(series_a, series_b)
    
    assert not result.any()  # No crossovers


def test_crossunder_detects_downward_cross():
    """Test that crossunder detects downward crossover correctly."""
    series_a = pd.Series([12.0, 12.0, 10.0, 9.0])  # Price
    series_b = pd.Series([11.0, 11.0, 11.0, 11.0])  # MA
    
    result = crossunder(series_a, series_b)
    
    assert len(result) == len(series_a)
    assert result.iloc[0] == False  # 12 >= 11, 12 >= 11
    assert result.iloc[1] == False  # 12 >= 11, 12 >= 11
    assert result.iloc[2] == True   # 10 < 11 AND 12 >= 11 (crossunder!)
    assert result.iloc[3] == False  # 9 < 11 BUT 10 < 11 (already crossed)


def test_crossunder_no_crossunder():
    """Test that crossunder returns False when no crossunder occurs."""
    series_a = pd.Series([10.0, 9.0, 8.0, 7.0])
    series_b = pd.Series([11.0, 11.5, 12.0, 12.5])  # Always above
    
    result = crossunder(series_a, series_b)
    
    assert not result.any()  # No crossunders


def test_generate_signal_from_ma_bullish():
    """Test that generate_signal_from_ma generates bullish signal."""
    price = pd.Series([10.0, 10.0, 12.0, 13.0])
    ma = pd.Series([11.0, 11.0, 11.0, 11.0])
    
    result = generate_signal_from_ma(price, ma)
    
    assert len(result) == len(price)
    assert result.iloc[0] == 0   # No signal
    assert result.iloc[1] == 0   # No signal
    assert result.iloc[2] == 1   # Bullish signal (crossover)
    assert result.iloc[3] == 0   # No new signal


def test_generate_signal_from_ma_bearish():
    """Test that generate_signal_from_ma generates bearish signal."""
    price = pd.Series([12.0, 12.0, 10.0, 9.0])
    ma = pd.Series([11.0, 11.0, 11.0, 11.0])
    
    result = generate_signal_from_ma(price, ma)
    
    assert len(result) == len(price)
    assert result.iloc[0] == 0   # No signal
    assert result.iloc[1] == 0   # No signal
    assert result.iloc[2] == -1 # Bearish signal (crossunder)
    assert result.iloc[3] == 0  # No new signal


def test_generate_signal_from_ma_no_signal():
    """Test that generate_signal_from_ma returns zeros when no crossover."""
    price = pd.Series([10.0, 11.0, 12.0, 13.0])
    ma = pd.Series([9.0, 9.5, 10.0, 10.5])  # Always below price
    
    result = generate_signal_from_ma(price, ma)
    
    assert (result == 0).all()


def test_generate_signal_from_ma_multiple_signals():
    """Test that generate_signal_from_ma handles multiple crossovers."""
    price = pd.Series([10.0, 12.0, 10.0, 12.0, 10.0])
    ma = pd.Series([11.0, 11.0, 11.0, 11.0, 11.0])
    
    result = generate_signal_from_ma(price, ma)
    
    assert result.iloc[0] == 0   # No signal
    assert result.iloc[1] == 1   # Bullish (10->12 crosses above 11)
    assert result.iloc[2] == -1  # Bearish (12->10 crosses below 11)
    assert result.iloc[3] == 1   # Bullish (10->12 crosses above 11)
    assert result.iloc[4] == -1  # Bearish (12->10 crosses below 11)


def test_generate_signal_from_ma_with_nan():
    """Test that generate_signal_from_ma handles NaN values."""
    price = pd.Series([10.0, np.nan, 12.0, 13.0])
    ma = pd.Series([11.0, 11.0, 11.0, 11.0])
    
    result = generate_signal_from_ma(price, ma)
    
    assert len(result) == len(price)
    # NaN handling depends on pandas behavior, but should not crash


def test_crossover_empty_series():
    """Test that crossover handles empty series."""
    series_a = pd.Series([], dtype=float)
    series_b = pd.Series([], dtype=float)
    
    result = crossover(series_a, series_b)
    
    assert len(result) == 0


def test_crossover_different_lengths():
    """Test that crossover handles series of different lengths."""
    series_a = pd.Series([10.0, 12.0])
    series_b = pd.Series([11.0, 11.0, 11.0])
    
    # pandas will align by index, so this may cause issues
    # In practice, series should have same length and index
    # This test checks that it doesn't crash, but behavior may be unexpected
    try:
        result = crossover(series_a, series_b)
        # If it doesn't crash, check basic properties
        assert isinstance(result, pd.Series)
    except (ValueError, IndexError):
        # It's acceptable if it raises an error for mismatched lengths
        pass

