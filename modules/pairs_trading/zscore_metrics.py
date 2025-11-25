"""
Z-score and related metrics for pairs trading analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
except ImportError:
    f1_score = None
    precision_score = None
    recall_score = None
    accuracy_score = None

try:
    from modules.config import (
        PAIRS_TRADING_ZSCORE_LOOKBACK,
        PAIRS_TRADING_CLASSIFICATION_ZSCORE,
    )
except ImportError:
    PAIRS_TRADING_ZSCORE_LOOKBACK = 60
    PAIRS_TRADING_CLASSIFICATION_ZSCORE = 0.5


def calculate_zscore_stats(
    spread: pd.Series, zscore_lookback: int = PAIRS_TRADING_ZSCORE_LOOKBACK
) -> Dict[str, Optional[float]]:
    """
    Calculate z-score statistics for the spread series.
    
    Args:
        spread: Spread series
        zscore_lookback: Number of periods for rolling window
        
    Returns:
        Dictionary with z-score statistics
    """
    result = {
        "mean_zscore": None,
        "std_zscore": None,
        "skewness": None,
        "kurtosis": None,
        "current_zscore": None,
    }

    if spread is None or len(spread) < zscore_lookback:
        return result

    rolling_mean = spread.rolling(zscore_lookback).mean()
    rolling_std = spread.rolling(zscore_lookback).std()
    zscore = (spread - rolling_mean) / rolling_std.replace(0, np.nan)

    zscore = zscore.dropna()
    if zscore.empty:
        return result

    result.update(
        {
            "mean_zscore": float(zscore.mean()),
            "std_zscore": float(zscore.std()),
            "skewness": float(zscore.skew()) if hasattr(zscore, "skew") else None,
            "kurtosis": float(zscore.kurtosis())
            if hasattr(zscore, "kurtosis")
            else None,
            "current_zscore": float(zscore.iloc[-1]),
        }
    )
    return result


def calculate_hurst_exponent(
    spread: pd.Series,
    zscore_lookback: int = PAIRS_TRADING_ZSCORE_LOOKBACK,
    max_lag: int = 100,
) -> Optional[float]:
    """
    Calculate Hurst exponent (R/S analysis).
    
    Args:
        spread: Spread series
        zscore_lookback: Minimum data points required
        max_lag: Maximum lag for R/S analysis
        
    Returns:
        Hurst exponent or None if calculation fails
    """
    if spread is None or len(spread.dropna()) < zscore_lookback:
        return None

    series = spread.dropna().values
    max_lag = min(max_lag, len(series) // 2)
    lags_list = [lag for lag in range(2, max_lag) if lag < len(series)]
    if not lags_list:
        return None

    try:
        tau = []
        filtered_lags = []
        for lag in lags_list:
            diff = np.subtract(series[lag:], series[:-lag])
            value = np.sqrt(np.std(diff))
            if value > 0:
                tau.append(value)
                filtered_lags.append(lag)
        if not tau:
            return None
        poly = np.polyfit(np.log(filtered_lags), np.log(tau), 1)
        hurst = poly[0] * 2.0
        if np.isnan(hurst) or np.isinf(hurst):
            return None
        return float(hurst)
    except Exception:
        return None


def calculate_direction_metrics(
    spread: pd.Series,
    zscore_lookback: int = PAIRS_TRADING_ZSCORE_LOOKBACK,
    classification_zscore: float = PAIRS_TRADING_CLASSIFICATION_ZSCORE,
) -> Dict[str, Optional[float]]:
    """
    Calculate classification metrics for spread direction.
    
    Args:
        spread: Spread series
        zscore_lookback: Number of periods for rolling window
        classification_zscore: Z-score threshold for classification
        
    Returns:
        Dictionary with classification metrics (F1, precision, recall, accuracy)
    """
    result = {
        "classification_f1": None,
        "classification_precision": None,
        "classification_recall": None,
        "classification_accuracy": None,
    }

    if (
        f1_score is None
        or precision_score is None
        or recall_score is None
        or accuracy_score is None
    ):
        return result

    if spread is None or len(spread) < zscore_lookback:
        return result

    rolling_mean = spread.rolling(zscore_lookback).mean()
    rolling_std = spread.rolling(zscore_lookback).std().replace(0, np.nan)
    zscore = ((spread - rolling_mean) / rolling_std).dropna()
    future_return = spread.shift(-1) - spread
    actual = (future_return > 0).astype(int).dropna()

    common_idx = zscore.index.intersection(actual.index)
    if len(common_idx) < 20:
        return result

    zscore = zscore.loc[common_idx]
    actual = actual.loc[common_idx]

    threshold = classification_zscore
    predicted = pd.Series(
        np.where(zscore < -threshold, 1, 0), index=zscore.index, dtype=int
    )

    try:
        result["classification_f1"] = float(
            f1_score(actual, predicted, average="weighted")
        )
        result["classification_precision"] = float(
            precision_score(actual, predicted, average="weighted", zero_division=0)
        )
        result["classification_recall"] = float(
            recall_score(actual, predicted, average="weighted", zero_division=0)
        )
        result["classification_accuracy"] = float(accuracy_score(actual, predicted))
    except Exception:
        return {
            "classification_f1": None,
            "classification_precision": None,
            "classification_recall": None,
            "classification_accuracy": None,
        }

    return result

