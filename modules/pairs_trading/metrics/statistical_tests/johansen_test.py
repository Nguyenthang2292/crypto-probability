"""
Johansen cointegration test for pairs trading analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union

try:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen  # type: ignore
except ImportError:
    coint_johansen = None

try:
    from modules.config import (
        PAIRS_TRADING_CORRELATION_MIN_POINTS,
        PAIRS_TRADING_JOHANSEN_CONFIDENCE,
        PAIRS_TRADING_JOHANSEN_DET_ORDER,
        PAIRS_TRADING_JOHANSEN_K_AR_DIFF,
    )
except ImportError:
    PAIRS_TRADING_CORRELATION_MIN_POINTS = 50
    PAIRS_TRADING_JOHANSEN_CONFIDENCE = 0.95
    PAIRS_TRADING_JOHANSEN_DET_ORDER = 0
    PAIRS_TRADING_JOHANSEN_K_AR_DIFF = 1


def calculate_johansen_test(
    price1: pd.Series,
    price2: pd.Series,
    min_points: int = PAIRS_TRADING_CORRELATION_MIN_POINTS,
    confidence: float = PAIRS_TRADING_JOHANSEN_CONFIDENCE,
    det_order: int = PAIRS_TRADING_JOHANSEN_DET_ORDER,
    k_ar_diff: int = PAIRS_TRADING_JOHANSEN_K_AR_DIFF,
) -> Optional[Dict[str, Union[float, bool]]]:
    """
    Run Johansen cointegration test to determine if two price series are cointegrated.
    
    Uses Vector Error Correction Model (VECM) framework to test for cointegrating relationships.
    More robust than Engle-Granger test. If trace_stat > critical_value, series are cointegrated
    and suitable for pairs trading (tend to revert to mean).
    
    Args:
        price1: First price series (pd.Series, should be aligned with price2)
        price2: Second price series (pd.Series, should be aligned with price1)
        min_points: Minimum data points required (must be > 0). Default: 50
        confidence: Confidence level. Options: 0.9 (90%), 0.95 (95%), 0.99 (99%). Default: 0.95
        det_order: Deterministic order in VECM model (must be in [-1, 0, 1]).
            0: No constant/trend (spread mean-reverts to zero) - most common for pairs trading
            1: Constant term (spread mean-reverts to constant level)
            -1: No constant but includes trend
            Default: 0
        k_ar_diff: Lag order for VAR model (must be >= 0). Typical: 1-4. Default: 1
        
    Returns:
        Dict with test results or None if test fails:
        {
            'johansen_trace_stat': float,        # Trace statistic (higher = stronger evidence)
            'johansen_critical_value': float,    # Critical value at confidence level
            'is_johansen_cointegrated': bool     # True if trace_stat > critical_value
        }
        
        Returns None if: statsmodels not installed, insufficient data, or test fails.
        
    Example:
        >>> result = calculate_johansen_test(price1, price2)
        >>> if result and result['is_johansen_cointegrated']:
        ...     print("✓ Assets are cointegrated - suitable for pairs trading")
    """
    if coint_johansen is None:
        return None
    
    if price1 is None or price2 is None:
        return None
    
    if not isinstance(price1, pd.Series) or not isinstance(price2, pd.Series):
        return None
    
    # Validate parameters
    if min_points <= 0:
        return None
    
    # Validate confidence level
    if confidence not in [0.9, 0.95, 0.99]:
        return None
    
    # Validate k_ar_diff
    if k_ar_diff < 0:
        return None
    
    # Validate det_order
    if det_order not in [-1, 0, 1]:
        return None
    
    # Check if price1 and price2 have same length
    if len(price1) != len(price2):
        return None
    
    # Handle NaN values: align and drop NaN to ensure clean calculations
    # Align indices first
    common_idx = price1.index.intersection(price2.index)
    if len(common_idx) < min_points:
        return None
    
    price1_aligned = price1.loc[common_idx]
    price2_aligned = price2.loc[common_idx]
    
    # Drop rows where either price1 or price2 is NaN
    valid_mask = price1_aligned.notna() & price2_aligned.notna()
    if valid_mask.sum() < min_points:
        return None
    
    price1_clean = price1_aligned[valid_mask]
    price2_clean = price2_aligned[valid_mask]

    data = np.column_stack([price1_clean.values, price2_clean.values])
    
    # Validate data shape
    if data.shape[0] < min_points or data.shape[1] != 2:
        return None
    
    # Validate data doesn't contain NaN or Inf
    if np.isnan(data).any() or np.isinf(data).any():
        return None

    try:
        confidence_map = {0.9: 0, 0.95: 1, 0.99: 2}
        confidence_key = round(confidence, 2)
        critical_idx = confidence_map.get(confidence_key, 1)
        
        # Johansen cointegration test with customizable parameters
        # det_order: deterministic order (0=no constant/trend, 1=constant, -1=no constant with trend)
        # k_ar_diff: lag order for VAR model (number of lags to include in the model)
        result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
        
        # Validate result structure
        if not hasattr(result, 'lr1') or not hasattr(result, 'cvt'):
            return None
        
        # Validate lr1 has at least one element
        if len(result.lr1) < 1:
            return None
        
        # Validate cvt has correct shape
        if result.cvt.shape[0] < 1 or result.cvt.shape[1] < (critical_idx + 1):
            return None
        
        trace_stat = result.lr1[0]
        critical_value = result.cvt[0, critical_idx]
        
        # Validate trace_stat and critical_value are finite
        if np.isnan(trace_stat) or np.isinf(trace_stat):
            return None
        if np.isnan(critical_value) or np.isinf(critical_value):
            return None
        
        is_cointegrated = trace_stat > critical_value
        
        return {
            "johansen_trace_stat": float(trace_stat),
            "johansen_critical_value": float(critical_value),
            "is_johansen_cointegrated": bool(is_cointegrated),
        }
    except (ValueError, TypeError, AttributeError, IndexError, np.linalg.LinAlgError):
        # ValueError: Invalid values in calculations
        # TypeError: Type conversion errors (e.g., float() on invalid types)
        # AttributeError: Missing attributes on result object (e.g., lr1, cvt)
        # IndexError: Index access errors (e.g., empty arrays)
        # LinAlgError: Linear algebra error in coint_johansen (singular matrix, insufficient data)
        return None
    except Exception:
        # Catch any other unexpected exceptions
        return None

