"""
OLS hedge ratio calculation for pairs trading.
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    from modules.config import (
        PAIRS_TRADING_OLS_FIT_INTERCEPT,
    )
except ImportError:
    PAIRS_TRADING_OLS_FIT_INTERCEPT = True

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    LinearRegression = None



def calculate_ols_hedge_ratio(
    price1: pd.Series,
    price2: pd.Series,
    fit_intercept: bool = PAIRS_TRADING_OLS_FIT_INTERCEPT,
) -> Optional[float]:
    """
    Calculate OLS (Ordinary Least Squares) hedge ratio using linear regression.
    
    The hedge ratio (β) determines how many units of asset 2 (price2) to short
    for each unit of asset 1 (price1) to long. It minimizes spread variance:
    spread = price1 - β * price2.
    
    **OLS (Static) vs Kalman (Dynamic)**:
    - OLS: Single constant ratio from all historical data. Best for stable relationships.
    - Kalman: Time-varying ratio that adapts to market changes. Use for evolving relationships.
    
    **Formula**: β = argmin Σ(price1 - β * price2)²
    Regression model: price1 = β * price2 + α (if fit_intercept=True) or price1 = β * price2 (if False)
    
    Args:
        price1: First price series (pd.Series, dependent variable, asset to long)
        price2: Second price series (pd.Series, independent variable, asset to short)
        fit_intercept: Include intercept term. True = price1 = β*price2 + α, False = price1 = β*price2.
            Default: PAIRS_TRADING_OLS_FIT_INTERCEPT (True)
        
    Returns:
        Hedge ratio (β) as float, or None if calculation fails (insufficient data, sklearn not installed,
        invalid input, or calculation error).
        
    Example:
        >>> price1 = pd.Series([100, 102, 101, 103, 105])
        >>> price2 = pd.Series([50, 51, 50.5, 51.5, 52.5])
        >>> hedge_ratio = calculate_ols_hedge_ratio(price1, price2)
        >>> # hedge_ratio = 1.92 means: short 1.92 units of price2 per 1 unit of price1 long
    """
    if LinearRegression is None:
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

    try:
        # OLS regression: price1 = beta * price2 + alpha (if fit_intercept=True).
        # The hedge ratio is the slope beta. When fit_intercept=False, beta ties directly
        # to price1 ≈ beta * price2 (common for spread = price1 - beta * price2).
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(price2_clean.values.reshape(-1, 1), price1_clean.values)
        
        # Validate model has coefficient
        if not hasattr(model, 'coef_') or len(model.coef_) == 0:
            return None
        
        beta = float(model.coef_[0])

        # Validate beta (avoid NaN/Inf/unrealistic magnitudes).
        if np.isnan(beta) or np.isinf(beta):
            return None
        if abs(beta) > 1e6:
            return None

        return beta
    except (ValueError, TypeError, AttributeError, IndexError, np.linalg.LinAlgError):
        # ValueError: Invalid values in calculations (e.g., NaN in fit)
        # TypeError: Type conversion errors (e.g., float() on invalid types)
        # AttributeError: Missing attributes on model (e.g., coef_)
        # IndexError: Index access errors (e.g., empty coef_ array)
        # LinAlgError: Linear algebra error in regression (singular matrix, etc.)
        return None
    except Exception:
        # Catch any other unexpected exceptions
        return None

