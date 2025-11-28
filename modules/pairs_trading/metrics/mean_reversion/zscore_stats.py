"""
Z-score statistics calculation for pairs trading analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

try:
    from modules.config import PAIRS_TRADING_ZSCORE_LOOKBACK
except ImportError:
    PAIRS_TRADING_ZSCORE_LOOKBACK = 60


def calculate_zscore_stats(
    spread: pd.Series, zscore_lookback: int = PAIRS_TRADING_ZSCORE_LOOKBACK
) -> Dict[str, Optional[float]]:
    """
    Calculate z-score statistics for the spread series.
    
    Computes rolling z-score (standardized spread) and calculates distribution statistics.
    Z-score measures how many standard deviations the spread is from its rolling mean.
    Useful for identifying mean-reversion opportunities in pairs trading.
    
    **Z-score Formula**: z = (spread - rolling_mean) / rolling_std
    
    **Interpretation**:
        - |z| > 2: Spread is far from mean (potential trading signal)
        - |z| < 1: Spread is close to mean (neutral)
        - Positive z: Spread above mean
        - Negative z: Spread below mean
    
    NaN values in the input spread are dropped before calculation to ensure
    clean statistical computations. Rolling window calculations require at least
    `zscore_lookback` valid (non-NaN) data points. If the spread never changes
    (all movements are zero), metrics cannot be calculated and all values return None.
    
    Args:
        spread: Spread series (pd.Series, price1 - hedge_ratio * price2).
               May contain NaN values, which will be dropped.
        zscore_lookback: Number of periods for rolling window (must be > 0). Default: 60
        
    Returns:
        Dictionary with z-score statistics (all Optional[float]):
        - mean_zscore: Mean of z-score values
        - std_zscore: Standard deviation of z-score values
        - skewness: Skewness of z-score distribution (requires >= 3 data points)
        - kurtosis: Kurtosis of z-score distribution (requires >= 3 data points)
        - current_zscore: Most recent z-score value
        
        All values are None if insufficient data or calculation fails.
        
    Example:
        >>> spread = pd.Series([0.1, -0.05, 0.15, -0.1, 0.08, ...])
        >>> stats = calculate_zscore_stats(spread, zscore_lookback=60)
        >>> # Returns dict with mean_zscore, std_zscore, skewness, kurtosis, current_zscore
    """
    result = {
        "mean_zscore": None,
        "std_zscore": None,
        "skewness": None,
        "kurtosis": None,
        "current_zscore": None,
    }

    if spread is None:
        return result
    
    if not isinstance(spread, pd.Series):
        return result
    
    # Validate zscore_lookback
    if zscore_lookback <= 0:
        return result
    
    if len(spread) < zscore_lookback:
        return result

    # Handle NaN values: drop NaN to ensure clean calculations
    # This ensures rolling window calculations are based on valid data only
    spread_clean = spread.dropna()
    
    # Check if we have enough valid data points after removing NaN
    if len(spread_clean) < zscore_lookback:
        return result
    
    # Validate spread_clean doesn't contain Inf
    if np.isinf(spread_clean.values).any():
        return result
    
    # Calculate rolling statistics on clean data
    rolling_mean = spread_clean.rolling(zscore_lookback, min_periods=zscore_lookback).mean()
    rolling_std = spread_clean.rolling(zscore_lookback, min_periods=zscore_lookback).std()
    
    # Replace zero std with NaN to avoid division by zero
    # Also ensure we don't divide by NaN (which would propagate NaN)
    rolling_std = rolling_std.replace(0, np.nan)
    
    # Validate rolling_std doesn't contain Inf (shouldn't happen, but safety check)
    if np.isinf(rolling_std.values).any():
        return result
    
    # Calculate z-score: (spread - mean) / std
    zscore = (spread_clean - rolling_mean) / rolling_std
    
    # Drop NaN values from z-score (from division by NaN std or missing rolling stats)
    zscore = zscore.dropna()
    
    # Validate we have enough z-score values for meaningful statistics
    if zscore.empty or len(zscore) < 2:
        return result
    
    # Validate zscore doesn't contain Inf (shouldn't happen if inputs are valid, but safety check)
    if np.isinf(zscore.values).any():
        return result

    try:
        # Calculate statistics with validation
        mean_val = zscore.mean()
        std_val = zscore.std()
        
        # Validate calculated values are finite (not NaN or inf)
        if pd.notna(mean_val) and np.isfinite(mean_val):
            result["mean_zscore"] = float(mean_val)
        
        if pd.notna(std_val) and np.isfinite(std_val):
            result["std_zscore"] = float(std_val)
        
        # Skewness and kurtosis require at least 3 data points
        if len(zscore) >= 3:
            skew_val = zscore.skew()
            if pd.notna(skew_val) and np.isfinite(skew_val):
                result["skewness"] = float(skew_val)
            
            kurt_val = zscore.kurtosis()
            if pd.notna(kurt_val) and np.isfinite(kurt_val):
                result["kurtosis"] = float(kurt_val)
        
        # Current z-score (most recent value)
        current_val = zscore.iloc[-1]
        if pd.notna(current_val) and np.isfinite(current_val):
            result["current_zscore"] = float(current_val)
            
    except (ValueError, TypeError, IndexError, AttributeError):
        # ValueError: Invalid values in statistical calculations (NaN, inf)
        # TypeError: Type conversion errors (e.g., float() on invalid types)
        # IndexError: Index access errors (e.g., empty Series after dropna)
        # AttributeError: Missing pandas methods (shouldn't happen, but safety)
        # Return partial results if some calculations succeed
        # Note: Exceptions are silently handled as partial results are already populated
        pass
    
    return result

