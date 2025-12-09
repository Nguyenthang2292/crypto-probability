"""Moving Average calculations for Adaptive Trend Classification (ATC).

This module provides functions to calculate various types of Moving Averages:
- calculate_kama_atc: KAMA (Kaufman Adaptive Moving Average) tuned for ATC
- ma_calculation: Calculate different MA types (EMA, HMA, WMA, DEMA, LSMA, KAMA)
- set_of_moving_averages: Generate a set of 9 MAs from a base length with offsets
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import pandas_ta as ta

from modules.common.utils import log_warn, log_error

from modules.common.indicators.momentum import calculate_kama_series
from modules.adaptive_trend.utils import diflen


def calculate_kama_atc(
    prices: pd.Series,
    length: int = 28,
) -> Optional[pd.Series]:
    """Calculate KAMA (Kaufman Adaptive Moving Average) for ATC.

    Uses KAMA formula from `momentum.calculate_kama_series` with parameters
    chosen to match Pine Script behavior:
    - length: Window length (default 28, equivalent to Pine `kama_len`)
    - fast: 2 → fast_sc ≈ 0.666 (matches Pine: 0.666)
    - slow: 30 → slow_sc ≈ 0.064 (matches Pine: 0.064)

    Args:
        prices: Price series (typically close prices).
        length: KAMA window length, equivalent to Pine `kama_len`.

    Returns:
        KAMA Series with same index as prices, or None if calculation fails.

    Raises:
        ValueError: If length is invalid or prices is empty.
        TypeError: If prices is not a pandas Series.
    """
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pandas Series, got {type(prices)}")
    
    if prices is None or len(prices) == 0:
        log_warn("Empty prices series provided for KAMA calculation")
        return None
    
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")
    
    if length > len(prices):
        log_warn(
            f"KAMA length ({length}) is greater than prices length ({len(prices)}). "
            f"This may result in insufficient data for calculation."
        )

    try:
        result = calculate_kama_series(
            prices=prices,
            period=length,
            fast=2,
            slow=30,
        )
        
        if result is None:
            log_warn(f"KAMA calculation returned None for length={length}")
        elif len(result) == 0:
            log_warn(f"KAMA calculation returned empty series for length={length}")
        
        return result
    
    except Exception as e:
        log_error(f"Error calculating KAMA: {e}")
        raise


def ma_calculation(
    source: pd.Series,
    length: int,
    ma_type: str,
) -> Optional[pd.Series]:
    """Calculate Moving Average based on specified type.

    Port of Pine Script function:
        ma_calculation(source, length, ma_type) =>
            if ma_type == "EMA"
                ta.ema(source, length)
            else if ma_type == "HMA"
                ta.sma(source, length)  # Note: Uses SMA, not classic Hull MA
            else if ma_type == "WMA"
                ta.wma(source, length)
            else if ma_type == "DEMA"
                ta.dema(source, length)
            else if ma_type == "LSMA"
                lsma(source, length)
            else if ma_type == "KAMA"
                kama(source, length)
            else
                na

    Notes:
    - HMA maps to SMA (not classic Hull MA) to match original script behavior.
    - LSMA uses `ta.linreg`, equivalent to `lsma()` in Pine.
    - KAMA calls `calculate_kama_atc` with normalized fast/slow parameters.

    Args:
        source: Source price series.
        length: Window length for Moving Average.
        ma_type: Type of MA: "EMA", "HMA", "WMA", "DEMA", "LSMA", or "KAMA"
            (case-insensitive).

    Returns:
        Moving Average Series, or None if calculation fails or invalid ma_type.

    Raises:
        ValueError: If length is invalid, source is empty, or ma_type is invalid.
        TypeError: If source is not a pandas Series.
    """
    if not isinstance(source, pd.Series):
        raise TypeError(f"source must be a pandas Series, got {type(source)}")
    
    if source is None or len(source) == 0:
        log_warn("Empty source series provided for MA calculation")
        return None
    
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")
    
    if length > len(source):
        log_warn(
            f"MA length ({length}) is greater than source length ({len(source)}). "
            f"This may result in insufficient data for calculation."
        )
    
    if not isinstance(ma_type, str) or not ma_type.strip():
        raise ValueError(f"ma_type must be a non-empty string, got {ma_type}")

    ma = ma_type.upper().strip()
    VALID_MA_TYPES = {"EMA", "HMA", "WMA", "DEMA", "LSMA", "KAMA"}
    
    if ma not in VALID_MA_TYPES:
        log_warn(
            f"Invalid ma_type '{ma_type}'. Valid types: {', '.join(VALID_MA_TYPES)}. "
            f"Returning None."
        )
        return None

    try:
        if ma == "EMA":
            result = ta.ema(source, length=length)
        elif ma == "HMA":
            # Pine: HMA branch uses ta.sma, not classic Hull MA.
            result = ta.sma(source, length=length)
        elif ma == "WMA":
            result = ta.wma(source, length=length)
        elif ma == "DEMA":
            result = ta.dema(source, length=length)
        elif ma == "LSMA":
            # LSMA ~ Linear Regression (Least Squares Moving Average)
            result = ta.linreg(source, length=length)
        elif ma == "KAMA":
            result = calculate_kama_atc(source, length=length)
        else:
            # This should never happen due to validation above, but kept for safety
            return None
        
        if result is None:
            log_warn(f"MA calculation ({ma}) returned None for length={length}")
        elif len(result) == 0:
            log_warn(f"MA calculation ({ma}) returned empty series for length={length}")
        elif not isinstance(result, pd.Series):
            log_warn(
                f"MA calculation ({ma}) returned unexpected type {type(result)}, "
                f"expected pandas Series"
            )
            return None
        
        return result
    
    except Exception as e:
        log_error(f"Error calculating {ma} MA with length={length}: {e}")
        raise


def set_of_moving_averages(
    length: int,
    source: pd.Series,
    ma_type: str,
    robustness: str = "Medium",
) -> Optional[Tuple[pd.Series, ...]]:
    """Generate a set of 9 Moving Averages with different length offsets.

    Port of Pine Script function:
        SetOfMovingAverages(length, source, ma_type) =>
            [L1,L2,L3,L4,L_1,L_2,L_3,L_4] = diflen(length)
            MA   = ma_calculation(source, length, ma_type)
            MA1  = ma_calculation(source, L1,     ma_type)
            ...
            [MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4]

    Calculates 9 MAs: one at base length, four with positive offsets (L1-L4),
    and four with negative offsets (L_1-L_4).

    Args:
        length: Base length for Moving Average.
        source: Source price series.
        ma_type: Type of MA: "EMA", "HMA", "WMA", "DEMA", "LSMA", or "KAMA".
        robustness: Robustness setting determining offset spread:
            "Narrow", "Medium", or "Wide".

    Returns:
        Tuple of 9 MA Series: (MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4),
        or None if source is empty or invalid.

    Raises:
        ValueError: If length is invalid, source is empty, or robustness is invalid.
        TypeError: If source is not a pandas Series.
    """
    # Input validation
    if not isinstance(source, pd.Series):
        raise TypeError(f"source must be a pandas Series, got {type(source)}")
    
    if source is None or len(source) == 0:
        log_warn("Empty source series provided for set_of_moving_averages")
        return None
    
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")
    
    if not isinstance(ma_type, str) or not ma_type.strip():
        raise ValueError(f"ma_type must be a non-empty string, got {ma_type}")
    
    VALID_ROBUSTNESS = {"Narrow", "Medium", "Wide"}
    if robustness not in VALID_ROBUSTNESS:
        log_warn(
            f"Invalid robustness '{robustness}'. Valid values: {', '.join(VALID_ROBUSTNESS)}. "
            f"Using default 'Medium'."
        )
        robustness = "Medium"

    try:
        # Calculate length offsets
        L1, L2, L3, L4, L_1, L_2, L_3, L_4 = diflen(length, robustness=robustness)
        
        # Validate offsets are positive (negative offsets from diflen should still be > 0)
        lengths = [length, L1, L2, L3, L4, L_1, L_2, L_3, L_4]
        if any(l <= 0 for l in lengths):
            invalid_lengths = [l for l in lengths if l <= 0]
            raise ValueError(
                f"Invalid length offsets calculated: {invalid_lengths}. "
                f"All lengths must be > 0."
            )

        # Calculate all MAs (optimized with list comprehension)
        ma_lengths = [length, L1, L2, L3, L4, L_1, L_2, L_3, L_4]
        ma_names = ["MA", "MA1", "MA2", "MA3", "MA4", "MA_1", "MA_2", "MA_3", "MA_4"]
        
        mas = []
        failed_calculations = []
        
        for ma_len, ma_name in zip(ma_lengths, ma_names):
            ma_result = ma_calculation(source, ma_len, ma_type)
            if ma_result is None:
                failed_calculations.append(f"{ma_name} (length={ma_len})")
                log_warn(
                    f"Failed to calculate {ma_name} ({ma_type}, length={ma_len})."
                )
            mas.append(ma_result)

        # Check if any MA calculation failed
        if all(ma is None for ma in mas):
            log_error(
                f"All MA calculations failed for ma_type={ma_type}, length={length}."
            )
            return None
        
        # Raise error if any MA calculation failed (don't return partial tuple)
        if failed_calculations:
            failed_list = ", ".join(failed_calculations)
            error_msg = (
                f"Failed to calculate {len(failed_calculations)} out of 9 MAs "
                f"for ma_type={ma_type}, length={length}. "
                f"Failed: {failed_list}. "
                f"Cannot proceed with partial MA set as it will cause TypeErrors downstream."
            )
            log_error(error_msg)
            raise ValueError(error_msg)

        # Unpack for return tuple (maintaining original variable names)
        MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4 = mas

        return MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4
    
    except Exception as e:
        log_error(f"Error calculating set of moving averages: {e}")
        raise


__all__ = [
    "calculate_kama_atc",
    "ma_calculation",
    "set_of_moving_averages",
]

