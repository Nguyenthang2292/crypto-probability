"""
Correlation calculation for pairs trading analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    from modules.config import PAIRS_TRADING_CORRELATION_MIN_POINTS
except ImportError:
    PAIRS_TRADING_CORRELATION_MIN_POINTS = 50

try:
    from modules.common.utils import log_warn
except ImportError:
    def log_warn(msg: str) -> None:
        print(f"[WARN] {msg}")


def calculate_correlation(
    price1: pd.Series,
    price2: pd.Series,
    min_points: int = PAIRS_TRADING_CORRELATION_MIN_POINTS,
) -> Optional[float]:
    """
    Calculate correlation between two price series based on returns.

    This function calculates the Pearson correlation coefficient between the returns
    of two price series. Correlation measures the linear relationship between two assets,
    which is important for pairs trading strategies.

    **Interpretation**:
    - **Correlation close to 1**: Strong positive relationship (prices move together)
    - **Correlation close to -1**: Strong negative relationship (prices move opposite)
    - **Correlation close to 0**: Weak or no linear relationship
    - For pairs trading, moderate positive correlation (0.3-0.9) is often preferred

    Args:
        price1: First price series (pd.Series). Missing values will be handled automatically.
        price2: Second price series (pd.Series). Must have same index as price1 or be aligned.
        min_points: Minimum data points required for calculation (must be >= 2). Default: 50.

    Returns:
        Correlation coefficient (float between -1 and 1), or None if calculation fails.
        
        Returns None if:
        - Input series are None or empty
        - Insufficient data points after alignment and cleaning
        - All returns are NaN or infinite
        - Correlation result is NaN or infinite
        - Correlation is outside valid range [-1, 1]

    Example:
        >>> import pandas as pd
        >>> price1 = pd.Series([100, 101, 102, 103, 104])
        >>> price2 = pd.Series([200, 201, 202, 203, 204])
        >>> corr = calculate_correlation(price1, price2, min_points=3)
        >>> print(f"Correlation: {corr:.3f}")
        Correlation: 1.000
    """
    # Validate inputs
    if price1 is None or price2 is None:
        return None
    
    if not isinstance(price1, pd.Series) or not isinstance(price2, pd.Series):
        return None
    
    if price1.empty or price2.empty:
        return None

    # Align indices and drop NaN values
    # Find common index
    common_index = price1.index.intersection(price2.index)
    if len(common_index) == 0:
        return None
    
    price1_aligned = price1.loc[common_index]
    price2_aligned = price2.loc[common_index]
    
    # Drop NaN values
    valid_mask = ~(price1_aligned.isna() | price2_aligned.isna())
    price1_clean = price1_aligned[valid_mask]
    price2_clean = price2_aligned[valid_mask]
    
    if len(price1_clean) < min_points:
        return None
    
    # Check for infinite values
    if np.isinf(price1_clean).any() or np.isinf(price2_clean).any():
        log_warn("Infinite values found in price series")
        return None

    try:
        # Calculate returns
        returns1 = price1_clean.pct_change().dropna()
        returns2 = price2_clean.pct_change().dropna()
        
        # Align returns by index (after pct_change, indices may differ by 1)
        returns_common = returns1.index.intersection(returns2.index)
        if len(returns_common) < min_points:
            return None
        
        returns1_aligned = returns1.loc[returns_common]
        returns2_aligned = returns2.loc[returns_common]
        
        # Validate returns data
        if returns1_aligned.isna().all() or returns2_aligned.isna().all():
            return None
        if np.isinf(returns1_aligned).any() or np.isinf(returns2_aligned).any():
            log_warn("Infinite values in returns")
            return None

        # Calculate correlation
        correlation = returns1_aligned.corr(returns2_aligned)

        if pd.isna(correlation) or np.isinf(correlation):
            return None

        # Validate correlation is in valid range [-1, 1]
        correlation = float(correlation)
        if not (-1 <= correlation <= 1):
            log_warn(f"Correlation out of valid range [-1, 1]: {correlation}")
            return None

        return correlation

    except (ValueError, AttributeError, KeyError) as e:
        log_warn(f"Error calculating correlation: {e}")
        return None
    except Exception as e:
        log_warn(f"Unexpected error calculating correlation: {e}")
        return None

