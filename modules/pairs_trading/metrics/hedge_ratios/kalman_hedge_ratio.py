"""
Kalman filter hedge ratio calculation for pairs trading.
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    from modules.config import (
        PAIRS_TRADING_KALMAN_DELTA,
        PAIRS_TRADING_KALMAN_OBS_COV,
    )
except ImportError:
    PAIRS_TRADING_KALMAN_DELTA = 1e-5
    PAIRS_TRADING_KALMAN_OBS_COV = 1.0

try:
    from pykalman import KalmanFilter  # type: ignore
except ImportError:
    KalmanFilter = None


def calculate_kalman_hedge_ratio(
    price1: pd.Series,
    price2: pd.Series,
    delta: float = PAIRS_TRADING_KALMAN_DELTA,
    observation_covariance: float = PAIRS_TRADING_KALMAN_OBS_COV,
) -> Optional[float]:
    """
    Estimate dynamic (time-varying) hedge ratio using Kalman filter.
    
    The hedge ratio (β) determines how many units of asset 2 (price2) to short
    for each unit of asset 1 (price1) to long. Unlike OLS (static), Kalman filter
    estimates a time-varying ratio that adapts to changing market conditions.
    
    **Kalman (Dynamic) vs OLS (Static)**:
    - Kalman: Ratio evolves over time, adapting to regime changes and volatility shifts.
      Best for volatile markets or when relationships change.
    - OLS: Single constant ratio from all historical data. Simpler but may be outdated.
    
    **Parameters**:
    - delta: Adaptation speed (1e-6 = slow/stable, 1e-4 = fast/reactive). Default: 1e-5
    - observation_covariance: Trust in new observations vs. previous estimates. Default: 1.0
    
    Args:
        price1: First price series (dependent variable, asset to long). Must be pd.Series
        price2: Second price series (independent variable, asset to short). Must be pd.Series
        delta: Transition covariance parameter controlling adaptation speed (must be > 0). Default: 1e-5
        observation_covariance: Observation noise variance (must be > 0). Default: 1.0
        
    Returns:
        Latest (most recent) hedge ratio (β) as float, or None if calculation fails
        (insufficient data, pykalman not installed, invalid parameters, etc.).
        
    Example:
        >>> price1 = pd.Series([100, 102, 101, 103, 105, 104, 106])
        >>> price2 = pd.Series([50, 51, 50.5, 51.5, 52.5, 52, 53])
        >>> hedge_ratio = calculate_kalman_hedge_ratio(price1, price2)
        >>> # Returns current dynamic ratio, e.g., 1.92 = short 1.92 units price2 per 1 unit price1
    """
    if KalmanFilter is None:
        return None
    
    if price1 is None or price2 is None:
        return None
    
    if not isinstance(price1, pd.Series) or not isinstance(price2, pd.Series):
        return None
    
    # Check if price1 and price2 have same length
    if len(price1) != len(price2):
        return None
    
    # Align indices and handle NaN values
    common_idx = price1.index.intersection(price2.index)
    if len(common_idx) < 10:
        return None
    
    price1_aligned = price1.loc[common_idx]
    price2_aligned = price2.loc[common_idx]
    
    # Drop rows where either price1 or price2 is NaN
    valid_mask = price1_aligned.notna() & price2_aligned.notna()
    if valid_mask.sum() < 10:
        return None
    
    price1_clean = price1_aligned[valid_mask]
    price2_clean = price2_aligned[valid_mask]
    
    # Validate price1 and price2 don't contain Inf
    if np.isinf(price1_clean.values).any() or np.isinf(price2_clean.values).any():
        return None
    
    # Validate delta
    if delta <= 0 or delta >= 1:
        return None
    
    # Validate observation_covariance
    if observation_covariance <= 0:
        return None

    try:
        # Transition covariance heuristic:
        #   Q = (delta / (1 - delta)) * I
        # Derived from assuming random walk with drift where delta controls
        # process noise scaling. Larger delta -> more noise -> faster adaptation.
        trans_cov = delta / (1 - delta) * np.eye(2)
        
        # Validate trans_cov is finite
        if np.isnan(trans_cov).any() or np.isinf(trans_cov).any():
            return None

        # Observation matrix builds relationship: price1_t = beta_t * price2_t + alpha_t
        # Each row is [price2_t, 1], reshaped for KalmanFilter API expectations.
        obs_mat = np.vstack([price2_clean.values, np.ones(len(price2_clean))]).T[:, np.newaxis, :]
        
        # Validate obs_mat doesn't contain NaN/Inf
        if np.isnan(obs_mat).any() or np.isinf(obs_mat).any():
            return None
        
        kf = KalmanFilter(
            transition_matrices=np.eye(2),
            observation_matrices=obs_mat,
            transition_covariance=trans_cov,
            observation_covariance=observation_covariance,
        )
        state_means, _ = kf.filter(price1_clean.values)
        
        # Validate state_means shape and content
        if state_means is None or len(state_means) == 0:
            return None
        
        if state_means.shape[1] < 1:
            return None
        
        beta_series = state_means[:, 0]
        
        if len(beta_series) == 0:
            return None
        
        # Validate beta_series[-1] is finite before conversion
        beta_value = beta_series[-1]
        if np.isnan(beta_value) or np.isinf(beta_value):
            return None
        
        beta = float(beta_value)

        # Validate beta
        if np.isnan(beta) or np.isinf(beta):
            return None
        if abs(beta) > 1e6:
            return None

        return beta
    except (ValueError, TypeError, AttributeError, IndexError, np.linalg.LinAlgError):
        # ValueError: Invalid values in calculations (e.g., NaN in filter)
        # TypeError: Type conversion errors (e.g., float() on invalid types)
        # AttributeError: Missing attributes on KalmanFilter object
        # IndexError: Index access errors (e.g., empty arrays)
        # LinAlgError: Linear algebra error in Kalman filter (singular matrix, etc.)
        return None
    except Exception:
        # Catch any other unexpected exceptions
        return None

