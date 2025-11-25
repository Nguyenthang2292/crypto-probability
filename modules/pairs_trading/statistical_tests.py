"""
Statistical tests for pairs trading analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

try:
    from statsmodels.tsa.stattools import adfuller  # type: ignore
except ImportError:
    adfuller = None

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    LinearRegression = None

try:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen  # type: ignore
except ImportError:
    coint_johansen = None

try:
    from modules.config import (
        PAIRS_TRADING_CORRELATION_MIN_POINTS,
        PAIRS_TRADING_JOHANSEN_CONFIDENCE,
    )
except ImportError:
    PAIRS_TRADING_CORRELATION_MIN_POINTS = 50
    PAIRS_TRADING_JOHANSEN_CONFIDENCE = 0.95


def calculate_adf_test(
    spread: pd.Series, min_points: int = PAIRS_TRADING_CORRELATION_MIN_POINTS
) -> Optional[Dict[str, float]]:
    """
    Run Augmented Dickey-Fuller test on the spread series.
    
    Args:
        spread: Spread series to test
        min_points: Minimum number of data points required
        
    Returns:
        Dictionary with ADF test results or None if test fails
    """
    if adfuller is None or spread is None:
        return None

    spread = spread.dropna()
    if len(spread) < min_points:
        return None

    try:
        adf_result = adfuller(spread, maxlag=1, autolag="AIC")
        return {
            "adf_statistic": float(adf_result[0]),
            "adf_pvalue": float(adf_result[1]),
            "critical_values": adf_result[4],
        }
    except Exception:
        return None


def calculate_half_life(spread: pd.Series) -> Optional[float]:
    """
    Calculate half-life of mean reversion for the spread.
    
    Args:
        spread: Spread series
        
    Returns:
        Half-life in periods, or None if calculation fails
    """
    if LinearRegression is None or spread is None:
        return None

    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag

    valid = spread_lag.notna() & spread_diff.notna()
    if valid.sum() < 10:
        return None

    try:
        X = spread_lag[valid].values.reshape(-1, 1)
        y = spread_diff[valid].values
        model = LinearRegression()
        model.fit(X, y)
        theta = model.coef_[0]
        if theta >= 0:
            return None
        half_life = -np.log(2) / theta
        if half_life < 0 or np.isinf(half_life):
            return None
        return float(half_life)
    except Exception:
        return None


def calculate_johansen_test(
    price1: pd.Series,
    price2: pd.Series,
    min_points: int = PAIRS_TRADING_CORRELATION_MIN_POINTS,
    confidence: float = PAIRS_TRADING_JOHANSEN_CONFIDENCE,
) -> Optional[Dict[str, Optional[float]]]:
    """
    Run Johansen cointegration test.
    
    Args:
        price1: First price series
        price2: Second price series
        min_points: Minimum number of data points required
        confidence: Confidence level (0.9, 0.95, 0.99)
        
    Returns:
        Dictionary with Johansen test results or None if test fails
    """
    if coint_johansen is None:
        return None

    data = np.column_stack([price1.values, price2.values])
    if data.shape[0] < min_points:
        return None

    try:
        confidence_map = {0.9: 0, 0.95: 1, 0.99: 2}
        confidence_key = round(confidence, 2)
        critical_idx = confidence_map.get(confidence_key, 1)
        result = coint_johansen(data, det_order=0, k_ar_diff=1)
        trace_stat = result.lr1[0]
        critical_value = result.cvt[0, critical_idx]
        is_cointegrated = trace_stat > critical_value
        return {
            "johansen_trace_stat": float(trace_stat),
            "johansen_critical_value": float(critical_value),
            "is_johansen_cointegrated": is_cointegrated,
        }
    except Exception:
        return None

