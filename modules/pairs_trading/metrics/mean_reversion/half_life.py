"""
Half-life calculation for pairs trading analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    LinearRegression = None

try:
    from modules.config import (
        PAIRS_TRADING_MIN_HALF_LIFE_POINTS,
        PAIRS_TRADING_MAX_HALF_LIFE,
    )
except ImportError:
    PAIRS_TRADING_MIN_HALF_LIFE_POINTS = 10
    PAIRS_TRADING_MAX_HALF_LIFE = 50


def calculate_half_life(spread: pd.Series) -> Optional[float]:
    """
    Calculate half-life of mean reversion for the spread.
    
    Half-life is the expected number of periods for spread deviation to reduce to 50%.
    Shorter half-life = faster mean reversion = better for pairs trading.
    
    **Calculation**: Uses OLS regression Δy(t) = θ * y(t-1) + ε. Formula: half_life = -ln(2) / θ
    where θ (mean reversion coefficient) must be negative for mean reversion to occur.
    
    **Interpretation**:
    - < 10 periods: Very fast (excellent)
    - 10-30 periods: Fast (good)
    - 30-50 periods: Moderate (acceptable)
    - > 50 periods: Slow (not suitable)
    
    Args:
        spread: Spread series (pd.Series, price1 - price2 * hedge_ratio)
        
    Returns:
        Half-life in periods, or None if calculation fails (non-stationary, insufficient data,
        or invalid result exceeding PAIRS_TRADING_MAX_HALF_LIFE).
        
    Example:
        >>> spread = pd.Series([0.1, 0.08, 0.05, 0.02, -0.01, ...])
        >>> half_life = calculate_half_life(spread)
        >>> # Returns number of periods for deviation to halve, e.g., 9.5 periods
    """
    if LinearRegression is None:
        return None
    
    if spread is None:
        return None
    
    if not isinstance(spread, pd.Series):
        return None
    
    # Validate minimum length
    if len(spread) < 2:
        return None
    
    # Handle NaN values: drop NaN to ensure clean calculations
    spread_clean = spread.dropna()
    if len(spread_clean) < 2:
        return None

    spread_lag = spread_clean.shift(1)
    spread_diff = spread_clean - spread_lag

    valid = spread_lag.notna() & spread_diff.notna()
    if valid.sum() < PAIRS_TRADING_MIN_HALF_LIFE_POINTS:
        return None

    try:
        X = spread_lag[valid].values.reshape(-1, 1)
        y = spread_diff[valid].values
        
        # Validate X and y have valid values
        if len(X) == 0 or len(y) == 0:
            return None
        
        # Validate X and y don't contain only NaN/Inf
        if np.isnan(X).all() or np.isinf(X).any():
            return None
        if np.isnan(y).all() or np.isinf(y).any():
            return None
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Validate model has coefficient
        if not hasattr(model, 'coef_') or len(model.coef_) == 0:
            return None
        
        theta = model.coef_[0]
        
        # Validate theta is finite
        if np.isnan(theta) or np.isinf(theta):
            return None
        
        # theta must be negative for mean reversion to occur
        # A negative theta indicates that when spread is above mean, it will decrease back,
        # and when below mean, it will increase back (mean-reverting behavior)
        # If theta >= 0, the spread exhibits random walk or trending behavior (non-stationary)
        if theta >= 0:
            return None
        
        # Validate theta is not too close to zero (would cause very large half-life)
        # Use a small threshold to avoid division by very small numbers
        if abs(theta) < 1e-10:
            return None
        
        # Half-life formula: half_life = -ln(2) / theta
        # This formula comes from solving the mean-reverting OLS model:
        # Δy(t) = θ * y(t-1) + ε, where θ < 0
        # After half-life periods, the deviation reduces to 50% of original
        # ln(2) ≈ 0.693 is used because we want to find when deviation becomes 1/2
        half_life = -np.log(2) / theta
        
        # Validate half-life: must be positive, finite, and not NaN
        # Also check upper bound: very large half-life indicates slow/non mean-reverting behavior
        if (half_life < 0 or 
            np.isinf(half_life) or 
            np.isnan(half_life) or 
            half_life > PAIRS_TRADING_MAX_HALF_LIFE):
            return None
        
        return float(half_life)
    except (ValueError, TypeError, AttributeError, IndexError):
        # ValueError: Invalid values in calculations (e.g., NaN in reshape)
        # TypeError: Type conversion errors (e.g., float() on invalid types)
        # AttributeError: Missing attributes on model (e.g., coef_)
        # IndexError: Index access errors (e.g., empty coef_ array)
        return None
    except Exception:
        # Catch any other unexpected exceptions
        return None

