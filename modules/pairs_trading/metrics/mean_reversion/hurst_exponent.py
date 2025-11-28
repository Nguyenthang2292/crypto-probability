"""
Hurst exponent calculation for pairs trading analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    from modules.config import (
        PAIRS_TRADING_ZSCORE_LOOKBACK,
        PAIRS_TRADING_MIN_LAG,
        PAIRS_TRADING_MAX_LAG_DIVISOR,
        PAIRS_TRADING_HURST_EXPONENT_MULTIPLIER,
        PAIRS_TRADING_HURST_EXPONENT_MIN,
        PAIRS_TRADING_HURST_EXPONENT_MAX,
        PAIRS_TRADING_HURST_EXPONENT_MEAN_REVERTING_MAX,
    )
except ImportError:
    PAIRS_TRADING_ZSCORE_LOOKBACK = 60
    PAIRS_TRADING_MIN_LAG = 2
    PAIRS_TRADING_MAX_LAG_DIVISOR = 2
    PAIRS_TRADING_HURST_EXPONENT_MULTIPLIER = 2.0
    PAIRS_TRADING_HURST_EXPONENT_MIN = 0.0
    PAIRS_TRADING_HURST_EXPONENT_MAX = 2.0
    PAIRS_TRADING_HURST_EXPONENT_MEAN_REVERTING_MAX = 0.5


def calculate_hurst_exponent(
    spread: pd.Series,
    zscore_lookback: int = PAIRS_TRADING_ZSCORE_LOOKBACK,
    max_lag: int = 100,
) -> Optional[float]:
    """
    Calculate Hurst exponent using variance-based scaling analysis.
    
    Hurst exponent measures long-term memory of a time series:
    - **H < 0.5**: Mean-reverting (spread returns to mean) → Good for pairs trading
    - **H ≈ 0.5**: Random walk (no predictable pattern)
    - **H > 0.5**: Trending (persistent trends) → Less suitable for mean reversion
    
    **Method**: Variance-based scaling analysis. Computes variance scaling across lags
    by calculating the standard deviation of differences at different lag intervals.
    Fits log(tau) vs log(lag) regression and converts slope to Hurst exponent.
    
    Args:
        spread: Spread series (pd.Series, price1 - hedge_ratio * price2)
        zscore_lookback: Minimum data points required (must be > 0). Default: 60
        max_lag: Maximum lag for analysis (must be > 0, capped at series_length/2). Default: 100
        
    Returns:
        Hurst exponent in range [0, 2], or None if calculation fails.
        For pairs trading, values < PAIRS_TRADING_HURST_EXPONENT_MEAN_REVERTING_MAX (default: 0.5)
        indicate mean-reverting behavior suitable for pairs trading.
        
    Example:
        >>> spread = pd.Series([0.1, -0.05, 0.15, -0.1, ...])
        >>> hurst = calculate_hurst_exponent(spread)
        >>> # hurst = 0.42 means mean-reverting (good for pairs trading)
        >>> # hurst < PAIRS_TRADING_HURST_EXPONENT_MEAN_REVERTING_MAX indicates mean-reversion
    """
    if spread is None:
        return None
    
    if not isinstance(spread, pd.Series):
        return None
    
    # Validate zscore_lookback
    if zscore_lookback <= 0:
        return None
    
    # Handle NaN values: drop NaN to ensure clean calculations
    spread_clean = spread.dropna()
    if len(spread_clean) < zscore_lookback:
        return None

    # Validate max_lag
    if max_lag <= 0:
        return None
    
    series = spread_clean.values
    # Limit max_lag to half of series length for stability
    max_lag = min(max_lag, len(series) // PAIRS_TRADING_MAX_LAG_DIVISOR)
    # Ensure max_lag is still valid after capping
    if max_lag < PAIRS_TRADING_MIN_LAG:
        return None
    # Start from PAIRS_TRADING_MIN_LAG (2) to ensure meaningful variance calculation
    # Note: max_lag is already capped at len(series) // PAIRS_TRADING_MAX_LAG_DIVISOR, so lag < len(series) is guaranteed
    lags_list = list(range(PAIRS_TRADING_MIN_LAG, max_lag))
    if not lags_list:
        return None

    try:
        tau = []
        filtered_lags = []
        # Calculate variance-based scaling for each lag
        for lag in lags_list:
            # Difference between series at lag intervals
            diff = np.subtract(series[lag:], series[:-lag])
            
            # Validate diff has enough values for meaningful std calculation
            if len(diff) < 2:
                continue
            
            # Calculate standard deviation
            std_val = np.std(diff)
            
            # Validate std is finite and positive
            if np.isnan(std_val) or np.isinf(std_val) or std_val <= 0:
                continue
            
            # Square root of standard deviation approximates scaling behavior
            value = np.sqrt(std_val)
            
            # Validate value is positive and finite
            if value > 0 and np.isfinite(value):
                tau.append(value)
                filtered_lags.append(lag)
        if not tau:
            return None
        
        # Validate all values are positive before log operations
        # This prevents ValueError from log of non-positive values
        # Note: filtered_lags should all be >= PAIRS_TRADING_MIN_LAG (2), but check tau values
        if any(t <= 0 for t in tau):
            return None
        
        # Validate we have enough points for regression (at least 2 points)
        if len(filtered_lags) < 2 or len(tau) < 2:
            return None
        
        # Validate filtered_lags and tau have same length
        if len(filtered_lags) != len(tau):
            return None
        
        # Linear regression: log(tau) = H * log(lag) + constant
        # Using polyfit with degree 1 (linear)
        log_lags = np.log(filtered_lags)
        log_tau = np.log(tau)
        
        # Validate log values are finite
        if not np.all(np.isfinite(log_lags)) or not np.all(np.isfinite(log_tau)):
            return None
        
        poly = np.polyfit(log_lags, log_tau, 1)
        
        # Validate polyfit result
        if len(poly) < 1:
            return None
        
        # Validate slope (poly[0]) is finite
        if np.isnan(poly[0]) or np.isinf(poly[0]):
            return None
        
        # Convert slope to Hurst exponent (multiply by 2 based on R/S theory)
        hurst = poly[0] * PAIRS_TRADING_HURST_EXPONENT_MULTIPLIER
        
        # Validate result
        if np.isnan(hurst) or np.isinf(hurst):
            return None
        
        # Clamp to theoretical bounds [0, 2]
        hurst = max(PAIRS_TRADING_HURST_EXPONENT_MIN, min(PAIRS_TRADING_HURST_EXPONENT_MAX, float(hurst)))
        
        return hurst
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError, TypeError, OverflowError):
        # ValueError: Invalid values in log/sqrt operations
        # LinAlgError: Linear algebra error in polyfit (singular matrix, insufficient data)
        # ZeroDivisionError: Division by zero (shouldn't happen due to checks, but safety)
        # TypeError: Type conversion errors
        # OverflowError: Numerical overflow in calculations
        # Note: Exceptions are silently handled and return None for graceful failure
        return None

