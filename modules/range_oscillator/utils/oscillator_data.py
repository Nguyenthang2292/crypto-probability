"""
Oscillator data utility functions.

This module provides helper functions for oscillator data handling.
"""

from typing import Optional, Tuple
import pandas as pd

from modules.range_oscillator.core.oscillator import calculate_range_oscillator


def get_oscillator_data(
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    oscillator: Optional[pd.Series] = None,
    ma: Optional[pd.Series] = None,
    range_atr: Optional[pd.Series] = None,
    length: int = 50,
    mult: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Helper function to get oscillator data, either from pre-calculated values or by calculating.
    
    This function avoids redundant calculations when oscillator data is already available.
    
    Args:
        high: High price series (required if oscillator not provided)
        low: Low price series (required if oscillator not provided)
        close: Close price series (required if oscillator not provided)
        oscillator: Pre-calculated oscillator series (optional)
        ma: Pre-calculated moving average series (optional)
        range_atr: Pre-calculated range ATR series (optional)
        length: Minimum range length for oscillator calculation (default: 50)
        mult: Range width multiplier (default: 2.0)
    
    Returns:
        Tuple containing (oscillator, ma, range_atr)
    """
    if oscillator is not None and ma is not None and range_atr is not None:
        # Use pre-calculated values
        return oscillator, ma, range_atr
    elif high is not None and low is not None and close is not None:
        # Calculate oscillator
        oscillator, _, ma, range_atr = calculate_range_oscillator(
            high=high,
            low=low,
            close=close,
            length=length,
            mult=mult,
        )
        return oscillator, ma, range_atr
    else:
        raise ValueError("Either provide (oscillator, ma, range_atr) or (high, low, close) with length and mult")


__all__ = [
    "get_oscillator_data",
]

