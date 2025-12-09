"""Calculate percentage rate of change for price series."""

import pandas as pd

from modules.common.utils import log_warn, log_error


def rate_of_change(prices: pd.Series) -> pd.Series:
    """Calculate percentage rate of change for price series.

    Equivalent to Pine Script global variable:
        R = (close - close[1]) / close[1]

    Args:
        prices: Price series (typically close prices).

    Returns:
        Series containing percentage change values. First value will be NaN.

    Raises:
        ValueError: If prices is empty.
        TypeError: If prices is not a pandas Series.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pandas Series, got {type(prices)}")
    
    if prices is None or len(prices) == 0:
        log_warn("Empty prices series provided for rate_of_change, returning empty series")
        return pd.Series(dtype="float64", index=prices.index if hasattr(prices, 'index') else pd.RangeIndex(0, 0))
    
    try:
        result = prices.pct_change()
        
        # Check for excessive NaN values (should only be first value)
        nan_count = result.isna().sum()
        if nan_count > 1:
            log_warn(
                f"rate_of_change contains {nan_count} NaN values. "
                f"Expected only 1 (first value). This may indicate data quality issues."
            )
        
        return result
    
    except Exception as e:
        log_error(f"Error calculating rate_of_change: {e}")
        raise

