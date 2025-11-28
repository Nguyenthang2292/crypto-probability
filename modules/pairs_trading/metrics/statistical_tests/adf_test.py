"""
Augmented Dickey-Fuller (ADF) test for pairs trading analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union

try:
    from statsmodels.tsa.stattools import adfuller  # type: ignore
except ImportError:
    adfuller = None

try:
    from modules.config import (
        PAIRS_TRADING_CORRELATION_MIN_POINTS,
        PAIRS_TRADING_ADF_MAXLAG,
    )
except ImportError:
    PAIRS_TRADING_CORRELATION_MIN_POINTS = 50
    PAIRS_TRADING_ADF_MAXLAG = 1


def calculate_adf_test(
    spread: pd.Series, min_points: int = PAIRS_TRADING_CORRELATION_MIN_POINTS
) -> Optional[Dict[str, Union[float, Dict[str, float]]]]:
    """
    Run Augmented Dickey-Fuller (ADF) test to check if spread is stationary (cointegrated).
    
    Tests whether spread has a unit root (non-stationary). Stationary spread = cointegrated assets
    suitable for pairs trading (will revert to mean). Non-stationary = not suitable.
    
    **Interpretation**:
    - **p-value < 0.05**: Reject H0 → Stationary (cointegrated) → Good for pairs trading
    - **p-value >= 0.05**: Fail to reject H0 → Non-stationary → Not suitable
    - More negative adf_statistic (vs critical values) = stronger evidence of stationarity
    
    Args:
        spread: Spread series (pd.Series, price1 - price2 * hedge_ratio). Missing values auto-removed.
        min_points: Minimum data points required (must be > 0). Default: 50.
        
    Returns:
        Dict with test results, or None if test fails:
        {
            'adf_statistic': float,  # Test statistic (more negative = more stationary)
            'adf_pvalue': float,     # p-value (p < 0.05 = stationary)
            'critical_values': dict  # {'1%': float, '5%': float, '10%': float}
        }
        
        Returns None if: statsmodels not installed, insufficient data, invalid input,
        or test execution fails.
        
    Example:
        >>> spread = pd.Series([0.1, -0.05, 0.15, -0.1, 0.08, ...])  # At least 50 points
        >>> result = calculate_adf_test(spread)
        >>> if result and result['adf_pvalue'] < 0.05:
        ...     print("✓ Spread is stationary - suitable for pairs trading")
    
    Note:
        - p-value < 0.05 (preferably < 0.01) indicates cointegrated pairs
        - Uses AIC lag selection with maxlag=PAIRS_TRADING_ADF_MAXLAG
    """
    if adfuller is None:
        return None
    
    if spread is None:
        return None
    
    if not isinstance(spread, pd.Series):
        return None
    
    # Validate min_points
    if min_points <= 0:
        return None

    spread = spread.dropna()
    if len(spread) < min_points:
        return None

    try:
        adf_result = adfuller(spread, maxlag=PAIRS_TRADING_ADF_MAXLAG, autolag="AIC")
        
        # Validate adf_result tuple structure
        # adfuller returns: (adf_stat, pvalue, usedlag, nobs, critical_values, icbest)
        if not isinstance(adf_result, tuple) or len(adf_result) < 5:
            return None
        
        adf_statistic = adf_result[0]
        adf_pvalue = adf_result[1]
        critical_values = adf_result[4]
        
        # Validate values are not None
        if adf_statistic is None or adf_pvalue is None or critical_values is None:
            return None
        
        # Validate and convert to float
        adf_statistic = float(adf_statistic)
        adf_pvalue = float(adf_pvalue)
        
        # Validate values are finite
        if np.isnan(adf_statistic) or np.isinf(adf_statistic):
            return None
        if np.isnan(adf_pvalue) or np.isinf(adf_pvalue) or adf_pvalue < 0 or adf_pvalue > 1:
            return None
        
        # Validate critical_values structure
        if not isinstance(critical_values, dict) or len(critical_values) == 0:
            return None
        
        # Validate and convert critical_values (accept any keys, but validate values)
        validated_critical_values: Dict[str, float] = {}
        for key, value in critical_values.items():
            if value is None or np.isnan(value) or np.isinf(value):
                return None
            validated_critical_values[str(key)] = float(value)
        
        return {
            "adf_statistic": adf_statistic,
            "adf_pvalue": adf_pvalue,
            "critical_values": validated_critical_values,
        }
    except (ValueError, TypeError, IndexError, KeyError):
        # Catch specific exceptions for better error handling
        return None
    except Exception:
        # Catch any other unexpected exceptions
        return None

