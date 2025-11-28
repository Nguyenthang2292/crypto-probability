"""
Sharpe ratio calculation for pairs trading analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    from modules.config import PAIRS_TRADING_PERIODS_PER_YEAR
except ImportError:
    PAIRS_TRADING_PERIODS_PER_YEAR = 365 * 24


def calculate_spread_sharpe(
    pnl_series: pd.Series, 
    periods_per_year: int = PAIRS_TRADING_PERIODS_PER_YEAR,
    risk_free_rate: float = 0.0
) -> Optional[float]:
    """
    Calculate annualized Sharpe ratio of the strategy PnL.
    
    The Sharpe ratio measures risk-adjusted return by comparing excess return
    (above risk-free rate) to volatility. Higher values indicate better risk-adjusted performance.
    
    **Formula**:
        Period_RF_Rate = Risk_Free_Rate / periods_per_year
        Excess_Return = Mean_PnL - Period_RF_Rate
        Sharpe = (Excess_Return / Std_PnL) * sqrt(periods_per_year)
    
    **Interpretation**:
        - > 3.0: Excellent risk-adjusted return
        - 1.0-3.0: Good risk-adjusted return
        - 0.5-1.0: Acceptable
        - < 0.5: Poor risk-adjusted return
        - < 0: Negative (strategy underperforms risk-free rate)
    
    Args:
        pnl_series: Series of Profit/Loss per period (pd.Series, e.g. spread.diff()).
                   Should NOT be cumulative. Missing values will be dropped.
        periods_per_year: Number of periods per year for annualization (must be > 0). Default: 365*24
        risk_free_rate: Risk-free rate per year (default: 0.0). 
                       If pnl_series is in dollars, this should be 0 or dollar equivalent.
        
    Returns:
        Annualized Sharpe ratio as float, or None if calculation fails (insufficient data,
        zero volatility, invalid input, or calculation error).
        
    Example:
        >>> pnl_series = pd.Series([10, -5, 15, 8, 12, -3])  # Per-period PnL
        >>> sharpe = calculate_spread_sharpe(pnl_series, periods_per_year=252)
        >>> # Returns annualized Sharpe ratio, e.g., 1.85
    """
    if pnl_series is None:
        return None
    
    if not isinstance(pnl_series, pd.Series):
        return None
    
    if len(pnl_series) < 2:
        return None
    
    # Validate periods_per_year
    if periods_per_year <= 0:
        return None
    
    # Validate risk_free_rate is finite
    if np.isnan(risk_free_rate) or np.isinf(risk_free_rate):
        return None

    # Drop NaNs (e.g. first element from diff)
    pnl = pnl_series.dropna()
    
    # Validate we have enough data points after dropping NaN
    if len(pnl) < 2:
        return None
    
    # Validate pnl doesn't contain Inf
    if np.isinf(pnl.values).any():
        return None
    
    # Check for zero volatility (would cause division by zero)
    if pnl.std() == 0:
        return None

    try:
        # Convert annual risk-free rate to per-period rate
        period_rf_rate = risk_free_rate / periods_per_year
        
        # Validate period_rf_rate is finite
        if np.isnan(period_rf_rate) or np.isinf(period_rf_rate):
            return None
        
        # Calculate excess return
        mean_return = pnl.mean()
        std_return = pnl.std()
        
        # Validate mean_return and std_return are finite
        if np.isnan(mean_return) or np.isinf(mean_return):
            return None
        if np.isnan(std_return) or np.isinf(std_return) or std_return == 0:
            return None
        
        excess_return = mean_return - period_rf_rate
        
        # Annualized Sharpe ratio
        sharpe = (excess_return / std_return) * np.sqrt(periods_per_year)
        
        # Validate result is finite
        if np.isnan(sharpe) or np.isinf(sharpe):
            return None
        
        # Additional validation: check for unrealistic values
        # Sharpe ratio can be negative, but very extreme values might indicate calculation errors
        if abs(sharpe) > 1e6:
            return None
            
        return float(sharpe)
    except (ValueError, TypeError, ZeroDivisionError):
        # ValueError: Invalid values in calculations (e.g., NaN in sqrt)
        # TypeError: Type conversion errors (e.g., float() on invalid types)
        # ZeroDivisionError: Division by zero (shouldn't happen due to checks, but safety)
        return None
    except Exception:
        # Catch any other unexpected exceptions
        return None

