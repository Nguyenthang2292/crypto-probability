"""Signal generation functions for Adaptive Trend Classification (ATC).

This module provides functions to generate trading signals from Moving Averages:
- crossover: Detect upward crossover (series_a crosses above series_b)
- crossunder: Detect downward crossover (series_a crosses below series_b)
- generate_signal_from_ma: Generate discrete signals {-1, 0, 1} from price/MA crossovers
"""

from __future__ import annotations

import pandas as pd

from modules.common.utils import log_warn, log_error


def crossover(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Detect upward crossover between two series.

    Equivalent to Pine Script `ta.crossover(a, b)`:
    Returns True when series_a currently > series_b AND series_a[1] <= series_b[1]

    Args:
        series_a: First series (typically price).
        series_b: Second series (typically Moving Average).

    Returns:
        Boolean Series: True at indices where crossover occurs.

    Raises:
        ValueError: If series are empty or have incompatible indices.
        TypeError: If inputs are not pandas Series.
    """
    if not isinstance(series_a, pd.Series):
        raise TypeError(f"series_a must be a pandas Series, got {type(series_a)}")
    
    if not isinstance(series_b, pd.Series):
        raise TypeError(f"series_b must be a pandas Series, got {type(series_b)}")
    
    if len(series_a) == 0 or len(series_b) == 0:
        log_warn("Empty series provided for crossover detection, returning empty boolean series")
        return pd.Series(dtype="bool", index=series_a.index if len(series_a) > 0 else series_b.index)
    
    try:
        # Align indices if needed
        if not series_a.index.equals(series_b.index):
            log_warn(
                f"series_a and series_b have different indices. "
                f"Aligning to common indices."
            )
            common_index = series_a.index.intersection(series_b.index)
            if len(common_index) == 0:
                log_warn("No common indices found between series_a and series_b")
                return pd.Series(dtype="bool", index=series_a.index)
            series_a = series_a.loc[common_index]
            series_b = series_b.loc[common_index]
        
        prev_a = series_a.shift(1)
        prev_b = series_b.shift(1)
        
        # Handle NaN values from shift(1) - first value will be NaN
        # NaN comparisons result in False, which is correct for our logic
        result = (series_a > series_b) & (prev_a <= prev_b)
        
        # Fill NaN values (from shift) with False
        result = result.fillna(False)
        
        return result
    
    except Exception as e:
        log_error(f"Error detecting crossover: {e}")
        raise


def crossunder(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Detect downward crossover between two series.

    Equivalent to Pine Script `ta.crossunder(a, b)`:
    Returns True when series_a currently < series_b AND series_a[1] >= series_b[1]

    Args:
        series_a: First series (typically price).
        series_b: Second series (typically Moving Average).

    Returns:
        Boolean Series: True at indices where crossunder occurs.

    Raises:
        ValueError: If series are empty or have incompatible indices.
        TypeError: If inputs are not pandas Series.
    """
    if not isinstance(series_a, pd.Series):
        raise TypeError(f"series_a must be a pandas Series, got {type(series_a)}")
    
    if not isinstance(series_b, pd.Series):
        raise TypeError(f"series_b must be a pandas Series, got {type(series_b)}")
    
    if len(series_a) == 0 or len(series_b) == 0:
        log_warn("Empty series provided for crossunder detection, returning empty boolean series")
        return pd.Series(dtype="bool", index=series_a.index if len(series_a) > 0 else series_b.index)
    
    try:
        # Align indices if needed
        if not series_a.index.equals(series_b.index):
            log_warn(
                f"series_a and series_b have different indices. "
                f"Aligning to common indices."
            )
            common_index = series_a.index.intersection(series_b.index)
            if len(common_index) == 0:
                log_warn("No common indices found between series_a and series_b")
                return pd.Series(dtype="bool", index=series_a.index)
            series_a = series_a.loc[common_index]
            series_b = series_b.loc[common_index]
        
        prev_a = series_a.shift(1)
        prev_b = series_b.shift(1)
        
        # Handle NaN values from shift(1) - first value will be NaN
        # NaN comparisons result in False, which is correct for our logic
        result = (series_a < series_b) & (prev_a >= prev_b)
        
        # Fill NaN values (from shift) with False
        result = result.fillna(False)
        
        return result
    
    except Exception as e:
        log_error(f"Error detecting crossunder: {e}")
        raise


def generate_signal_from_ma(
    price: pd.Series,
    ma: pd.Series,
) -> pd.Series:
    """Generate discrete trading signals from price/MA crossovers.

    Port of Pine Script function:
        signal(ma) =>
            var int sig = 0
            if ta.crossover(close, ma)
                sig := 1
            if ta.crossunder(close, ma)
                sig := -1
            sig

    Args:
        price: Price series (typically close prices).
        ma: Moving Average series.

    Returns:
        Series with discrete signal values:
        - 1: Bullish signal (price crosses above MA)
        - -1: Bearish signal (price crosses below MA)
        - 0: No signal (no crossover detected)

    Raises:
        ValueError: If price or ma are empty or have incompatible indices.
        TypeError: If inputs are not pandas Series.
    """
    if not isinstance(price, pd.Series):
        raise TypeError(f"price must be a pandas Series, got {type(price)}")
    
    if not isinstance(ma, pd.Series):
        raise TypeError(f"ma must be a pandas Series, got {type(ma)}")
    
    if len(price) == 0 or len(ma) == 0:
        log_warn("Empty price or MA series provided, returning empty signal series")
        return pd.Series(dtype="int8", index=price.index if len(price) > 0 else ma.index)
    
    try:
        # Align indices if needed
        if not price.index.equals(ma.index):
            log_warn(
                f"price and ma have different indices. "
                f"Aligning to common indices."
            )
            common_index = price.index.intersection(ma.index)
            if len(common_index) == 0:
                log_warn("No common indices found between price and ma")
                return pd.Series(dtype="int8", index=price.index)
            price = price.loc[common_index]
            ma = ma.loc[common_index]
        
        # Check for excessive NaN values
        price_nan_count = price.isna().sum()
        ma_nan_count = ma.isna().sum()
        total_bars = len(price)
        
        if price_nan_count > 0:
            nan_pct = (price_nan_count / total_bars) * 100
            if nan_pct > 10:  # Warn if more than 10% NaN
                log_warn(
                    f"Price series contains {price_nan_count} NaN values ({nan_pct:.1f}%). "
                    f"This may affect signal generation."
                )
        
        if ma_nan_count > 0:
            nan_pct = (ma_nan_count / total_bars) * 100
            if nan_pct > 10:  # Warn if more than 10% NaN
                log_warn(
                    f"MA series contains {ma_nan_count} NaN values ({nan_pct:.1f}%). "
                    f"This may affect signal generation."
                )
        
        # Generate signals
        sig = pd.Series(0, index=price.index, dtype="int8")
        up = crossover(price, ma)
        down = crossunder(price, ma)
        
        # Set signals (only where crossover/crossunder is True)
        sig.loc[up] = 1
        sig.loc[down] = -1
        
        # Check if both up and down are True at same index (shouldn't happen, but handle it)
        conflict_mask = up & down
        if conflict_mask.any():
            conflict_count = conflict_mask.sum()
            log_warn(
                f"Found {conflict_count} indices where both crossover and crossunder are True. "
                f"These will be set to 0 (no signal)."
            )
            sig.loc[conflict_mask] = 0
        
        return sig
    
    except Exception as e:
        log_error(f"Error generating signal from MA: {e}")
        raise


__all__ = [
    "crossover",
    "crossunder",
    "generate_signal_from_ma",
]

