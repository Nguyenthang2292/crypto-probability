"""
Risk metrics calculations for pairs trading analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    from modules.config import PAIRS_TRADING_PERIODS_PER_YEAR
except ImportError:
    PAIRS_TRADING_PERIODS_PER_YEAR = 365 * 24


def calculate_spread_sharpe(
    spread: pd.Series, periods_per_year: int = PAIRS_TRADING_PERIODS_PER_YEAR
) -> Optional[float]:
    """
    Calculate Sharpe ratio of spread returns.
    
    Args:
        spread: Spread series
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Sharpe ratio or None if calculation fails
    """
    returns = spread.pct_change().dropna()
    if returns.empty or returns.std() == 0:
        return None
    mean_return = returns.mean()
    std_return = returns.std()
    try:
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        if np.isnan(sharpe) or np.isinf(sharpe):
            return None
        return float(sharpe)
    except Exception:
        return None


def calculate_max_drawdown(spread: pd.Series) -> Optional[float]:
    """
    Calculate maximum drawdown of the cumulative spread.
    
    Args:
        spread: Spread series
        
    Returns:
        Maximum drawdown (negative value) or None if calculation fails
    """
    if spread is None or len(spread) < 2:
        return None
    cumulative = spread.cumsum()
    running_max = cumulative.cummax().replace(0, np.nan)
    drawdown = (cumulative - running_max) / running_max
    if drawdown.isna().all():
        return None
    return float(drawdown.min())


def calculate_calmar_ratio(
    spread: pd.Series,
    periods_per_year: int = PAIRS_TRADING_PERIODS_PER_YEAR,
) -> Optional[float]:
    """
    Calculate Calmar ratio (annual return / max drawdown).
    
    Args:
        spread: Spread series
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Calmar ratio or None if calculation fails
    """
    returns = spread.pct_change().dropna()
    if returns.empty:
        return None
    annual_return = returns.mean() * periods_per_year
    # Use local calculate_max_drawdown function
    max_dd = calculate_max_drawdown(spread)
    if max_dd is None or max_dd == 0:
        return None
    max_dd = abs(max_dd)
    if max_dd == 0:
        return None
    calmar = annual_return / max_dd
    if np.isnan(calmar) or np.isinf(calmar):
        return None
    return float(calmar)

