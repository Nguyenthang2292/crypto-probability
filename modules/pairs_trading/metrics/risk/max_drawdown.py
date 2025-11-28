"""
Maximum drawdown calculation for pairs trading analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional



def calculate_max_drawdown(equity_curve: pd.Series) -> Optional[float]:
    """
    Calculate maximum drawdown of the equity curve.
    
    Maximum drawdown measures the largest peak-to-trough decline in equity.
    It represents the worst loss from a peak to a subsequent trough.
    
    **Formula**:
        Drawdown(t) = Equity(t) - RunningMax(t)
        MaxDrawdown = min(Drawdown(t))
    
    **Interpretation**:
        - More negative values indicate larger drawdowns (worse risk)
        - A drawdown of -500 means equity dropped $500 from its peak
        - Lower (more negative) drawdown = higher risk
    
    Args:
        equity_curve: Cumulative PnL series (pd.Series, e.g. spread.diff().cumsum()).
                     Should be cumulative values, not per-period PnL.
        
    Returns:
        Maximum drawdown as a negative float (e.g., -500.0 for $500 drawdown),
        or None if calculation fails (insufficient data, invalid input, or calculation error).
        
    Example:
        >>> equity_curve = pd.Series([0, 10, 5, 20, 15, 30])  # Cumulative PnL
        >>> max_dd = calculate_max_drawdown(equity_curve)
        >>> # Returns most negative drawdown, e.g., -5.0
    """
    if equity_curve is None:
        return None
    
    if not isinstance(equity_curve, pd.Series):
        return None
    
    if len(equity_curve) < 2:
        return None
    
    # Handle NaN values: drop NaN to ensure clean calculations
    equity_curve_clean = equity_curve.dropna()
    if len(equity_curve_clean) < 2:
        return None
    
    # Validate equity_curve doesn't contain Inf
    if np.isinf(equity_curve_clean.values).any():
        return None
    
    try:
        # Calculate running maximum (cumulative maximum up to each point)
        running_max = equity_curve_clean.expanding().max()
        
        # Validate running_max is not all NaN
        if running_max.isna().all():
            return None
        
        # Calculate drawdown (difference from running maximum)
        # Negative values indicate drawdowns (equity below peak)
        drawdown = equity_curve_clean - running_max
        
        # Validate drawdown has valid values
        if drawdown.isna().all():
            return None
        
        # Return minimum (most negative) drawdown
        # This represents the worst peak-to-trough decline
        max_dd = float(drawdown.min())
        
        # Validate result is finite
        if np.isnan(max_dd) or np.isinf(max_dd):
            return None
        
        # Note: max_dd should typically be <= 0 (negative or zero)
        # Positive values are possible if equity never drops below initial value,
        # but this is unusual and might indicate calculation issues
        # We allow it but note it in validation
        
        return max_dd
        
    except (ValueError, TypeError, AttributeError, IndexError):
        # ValueError: Invalid values in calculations (e.g., NaN in min)
        # TypeError: Type conversion errors (e.g., float() on invalid types)
        # AttributeError: Missing attributes on pandas Series
        # IndexError: Index access errors (e.g., empty Series after operations)
        return None
    except Exception:
        # Catch any other unexpected exceptions
        return None

