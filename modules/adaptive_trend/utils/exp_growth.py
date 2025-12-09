"""Calculate exponential growth factor over time."""

from typing import Optional

import numpy as np
import pandas as pd

from modules.common.utils import log_warn, log_error


def exp_growth(
    L: float,
    index: Optional[pd.Index] = None,
    *,
    cutout: int = 0,
) -> pd.Series:
    """Calculate exponential growth factor over time.

    Port of Pine Script function:
        e(L) =>
            bars = bar_index == 0 ? 1 : bar_index
            x = 1.0
            if time >= cuttime
                x := math.pow(math.e, L * (bar_index - cutout))
            x

    In TradingView, `time` and `bar_index` are global environment variables.
    Here we approximate using positional indices (0, 1, 2, ...) of the Series.

    Args:
        L: Lambda (growth rate parameter, must be finite).
        index: Time/bar index of the data. If None, creates empty RangeIndex.
        cutout: Number of bars to skip at the beginning (bars before cutout
            will have value 1.0, must be >= 0).

    Returns:
        Series containing exponential growth factors e^(L * (bar_index - cutout))
        for bars >= cutout, and 1.0 for bars < cutout.

    Raises:
        ValueError: If L is invalid, cutout is invalid, or overflow occurs.
        TypeError: If L is not a number or cutout is not an integer.
    """
    if not isinstance(L, (int, float)) or np.isnan(L) or np.isinf(L):
        raise ValueError(f"L must be a finite number, got {L}")
    
    if not isinstance(cutout, int) or cutout < 0:
        raise ValueError(f"cutout must be a non-negative integer, got {cutout}")
    
    if index is None:
        index = pd.RangeIndex(0, 0)

    try:
        # Use position 0..n-1 as equivalent to `bar_index`
        bars = pd.Series(range(len(index)), index=index, dtype="float64")
        # In Pine: if bar_index == 0 then bars = 1, else = bar_index
        bars = bars.where(bars != 0, 1.0)

        # Condition "has passed cutout"
        active = bars >= cutout
        x = pd.Series(1.0, index=index, dtype="float64")
        
        # Calculate exponential growth for active bars
        if active.any():
            # Calculate exponent to check for overflow
            exponents = L * (bars[active] - cutout)
            
            # Check for potential overflow (exp > 700 will overflow float64)
            max_exponent = exponents.max() if len(exponents) > 0 else 0
            if max_exponent > 700:
                log_warn(
                    f"Potential overflow in exp_growth: max exponent = {max_exponent:.2f}. "
                    f"Values > 700 may result in inf. L={L}, max_bar={bars[active].max()}, cutout={cutout}"
                )
            
            # Calculate exponential growth
            growth_values = np.e ** exponents
            
            # Check for overflow/inf values
            inf_count = np.isinf(growth_values).sum()
            if inf_count > 0:
                log_warn(
                    f"exp_growth produced {inf_count} inf values. "
                    f"This may indicate overflow. Consider reducing L or cutout."
                )
                # Replace inf with a large but finite value
                growth_values = np.where(np.isinf(growth_values), np.finfo(np.float64).max, growth_values)
            
            x.loc[active] = growth_values.astype("float64")
        
        return x
    
    except OverflowError as e:
        log_error(f"Overflow error in exp_growth: {e}. L={L}, cutout={cutout}")
        raise ValueError(f"Overflow in exponential calculation. L={L} may be too large.") from e
    except Exception as e:
        log_error(f"Error calculating exp_growth: {e}")
        raise

