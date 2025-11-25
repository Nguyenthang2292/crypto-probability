"""
Hedge ratio calculations for pairs trading.
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    LinearRegression = None

try:
    from pykalman import KalmanFilter  # type: ignore
except ImportError:
    KalmanFilter = None


def calculate_ols_hedge_ratio(
    price1: pd.Series, price2: pd.Series
) -> Optional[float]:
    """
    Calculate OLS hedge ratio (price1 ~ beta * price2).
    
    Args:
        price1: First price series (dependent variable)
        price2: Second price series (independent variable)
        
    Returns:
        Hedge ratio (beta) or None if calculation fails
    """
    if LinearRegression is None or len(price1) != len(price2) or len(price1) < 10:
        return None

    try:
        model = LinearRegression()
        model.fit(price2.values.reshape(-1, 1), price1.values)
        return float(model.coef_[0])
    except Exception:
        return None


def calculate_kalman_hedge_ratio(
    price1: pd.Series, price2: pd.Series
) -> Optional[float]:
    """
    Estimate dynamic hedge ratio using Kalman filter.
    
    Args:
        price1: First price series
        price2: Second price series
        
    Returns:
        Latest Kalman filter hedge ratio or None if calculation fails
    """
    if KalmanFilter is None or len(price1) != len(price2) or len(price1) < 10:
        return None

    try:
        delta = 1e-5
        trans_cov = delta / (1 - delta) * np.eye(2)
        obs_mat = np.vstack([price2.values, np.ones(len(price2))]).T[
            :, np.newaxis, :
        ]
        kf = KalmanFilter(
            transition_matrices=np.eye(2),
            observation_matrices=obs_mat,
            transition_covariance=trans_cov,
            observation_covariance=1.0,
        )
        state_means, _ = kf.filter(price1.values)
        beta_series = state_means[:, 0]
        if len(beta_series) == 0 or np.isnan(beta_series[-1]):
            return None
        return float(beta_series[-1])
    except Exception:
        return None

