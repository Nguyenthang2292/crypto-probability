"""
Calmar ratio calculation for pairs trading analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    from modules.config import PAIRS_TRADING_PERIODS_PER_YEAR
except ImportError:
    PAIRS_TRADING_PERIODS_PER_YEAR = 365 * 24

from modules.pairs_trading.metrics.risk.max_drawdown import calculate_max_drawdown


def calculate_calmar_ratio(
    equity_curve: pd.Series,
    periods_per_year: int = PAIRS_TRADING_PERIODS_PER_YEAR,
) -> Optional[float]:
    """
    Calculate Calmar ratio (annualized return / absolute max drawdown).
    
    The Calmar ratio measures risk-adjusted return by comparing annualized return
    to maximum drawdown. Higher values indicate better risk-adjusted performance.
    
    **Formula**:
        Annualized_Return = Mean_PnL * periods_per_year
        Calmar = Annualized_Return / Absolute_Max_Drawdown
    
    **Interpretation**:
        - > 3.0: Excellent risk-adjusted return
        - 1.0-3.0: Good risk-adjusted return
        - 0.5-1.0: Acceptable
        - < 0.5: Poor risk-adjusted return
    
    Args:
        equity_curve: Cumulative PnL series (pd.Series, e.g. spread.diff().cumsum()).
                     Should be cumulative values, not per-period PnL.
        periods_per_year: Number of periods per year for annualization (must be > 0). Default: 365*24
        
    Returns:
        Calmar ratio as float, or None if calculation fails (insufficient data, zero drawdown,
        invalid input, or calculation error).
        
    Example:
        >>> equity_curve = pd.Series([0, 10, 5, 20, 15, 30])  # Cumulative PnL
        >>> calmar = calculate_calmar_ratio(equity_curve, periods_per_year=252)
        >>> # Returns annualized return / max drawdown ratio
    """
    if equity_curve is None:
        return None
    
    if not isinstance(equity_curve, pd.Series):
        return None
    
    if len(equity_curve) < 2:
        return None
    
    # Validate periods_per_year
    if periods_per_year <= 0:
        return None
    
    # Drop NaN values from equity curve
    equity_curve = equity_curve.dropna()
    if len(equity_curve) < 2:
        return None
        
    try:
        # Calculate PnL series from equity curve (period-to-period changes)
        pnl_series = equity_curve.diff().dropna()
        if pnl_series.empty:
            return None
        
        # Calculate annualized return: Mean PnL per period * periods per year
        # This is equivalent to: (Total Return / Number of Periods) * Periods Per Year
        mean_pnl = pnl_series.mean()
        
        # Validate mean_pnl is finite
        if np.isnan(mean_pnl) or np.isinf(mean_pnl):
            return None
            
        annual_return = mean_pnl * periods_per_year
        
        # Calculate max drawdown
        max_dd = calculate_max_drawdown(equity_curve)
        if max_dd is None:
            return None
        
        # Take absolute value of max drawdown (it's negative)
        abs_max_dd = abs(max_dd)
        
        # Avoid division by zero
        if abs_max_dd == 0:
            return None
        
        # Calculate Calmar ratio
        calmar = annual_return / abs_max_dd
        
        # Validate result
        if np.isnan(calmar) or np.isinf(calmar):
            return None
        
        # Additional validation: check for unrealistic values
        # Calmar ratio can be negative if annual return is negative
        # But very extreme values might indicate calculation errors
        if abs(calmar) > 1e6:
            return None
            
        return float(calmar)
        
    except (ValueError, TypeError, ZeroDivisionError):
        # Catch specific exceptions for better error handling
        return None
    except Exception:
        # Catch any other unexpected exceptions
        return None

