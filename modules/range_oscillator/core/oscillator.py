"""Range Oscillator main calculation.

This module provides the main Range Oscillator calculation function that
orchestrates the entire indicator computation process. It combines weighted MA,
ATR range, and trend direction to produce oscillator values with color coding.

The module uses helper functions from other core modules:
- weighted_ma: Weighted moving average calculation
- atr_range: ATR-based range bands calculation
- trend_direction: Trend direction determination

Port of Pine Script Range Oscillator (Zeiierman).
Original: https://creativecommons.org/licenses/by-nc-sa/4.0/
© Zeiierman
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from modules.range_oscillator.core.weighted_ma import calculate_weighted_ma
from modules.range_oscillator.core.atr_range import calculate_atr_range
from modules.range_oscillator.core.trend_direction import calculate_trend_direction


def calculate_range_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    length: int = 50,
    mult: float = 2.0,
    levels_inp: int = 2,
    heat_thresh: int = 1,
    strong_bullish_color: str = "#09ff00",
    strong_bearish_color: str = "#ff0000",
    weak_bearish_color: str = "#800000",
    weak_bullish_color: str = "#008000",
    transition_color: str = "#0000ff",
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate Range Oscillator indicator.

    This is the main function that orchestrates the entire Range Oscillator
    calculation process. It combines weighted MA, ATR range, trend direction,
    and heatmap colors to produce a comprehensive oscillator indicator.

    LUỒNG TÍNH TOÁN:
    ----------------
    1. Tính Weighted MA từ close prices
    2. Tính ATR Range từ high/low/close
    3. Xác định Trend Direction (bullish/bearish)
    4. Với mỗi bar:
       a. Tính Oscillator = 100 * (close - MA) / RangeATR
       b. Kiểm tra breakouts (upper/lower bounds)
       c. Tính heatmap color dựa trên historical touches
       d. Xác định final color (breakout > heatmap > transition)

    Port of Pine Script Range Oscillator (Zeiierman).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        length: Minimum range length (default: 50).
        mult: Range width multiplier (default: 2.0).
        levels_inp: Number of heat levels (default: 2).
        heat_thresh: Minimum touches per level (default: 1).
        strong_bullish_color: Hex color for strong bullish zones.
        strong_bearish_color: Hex color for strong bearish zones.
        weak_bearish_color: Hex color for weak bearish zones.
        weak_bullish_color: Hex color for weak bullish zones.
        transition_color: Hex color for transitions.

    Returns:
        Tuple containing:
        - oscillator: Oscillator values (ranges from -100 to +100)
        - oscillator_color: Hex color strings for each oscillator value
        - ma: Weighted moving average
        - range_atr: ATR-based range
    """
    # Step 1: Calculate weighted MA
    ma = calculate_weighted_ma(close, length=length)

    # Step 2: Calculate ATR range
    range_atr = calculate_atr_range(high, low, close, mult=mult)

    # Step 3: Calculate trend direction
    trend_dir = calculate_trend_direction(close, ma)

    # Step 4: Calculate oscillator and colors
    oscillator = pd.Series(np.nan, index=close.index, dtype="float64")
    oscillator_color = pd.Series(None, index=close.index, dtype="object")

    prev_trend_dir = 0

    for i in range(len(close)):
        if pd.isna(range_atr.iloc[i]) or range_atr.iloc[i] == 0:
            continue

        if pd.isna(ma.iloc[i]):
            continue

        # Step 4a: Calculate oscillator value
        osc_value = 100 * (close.iloc[i] - ma.iloc[i]) / range_atr.iloc[i]
        oscillator.iloc[i] = osc_value

        # Step 4b: Determine color
        current_trend_dir = trend_dir.iloc[i]
        no_color_on_flip = current_trend_dir != prev_trend_dir

        # Step 4c: Check for breakouts
        break_up = close.iloc[i] > ma.iloc[i] + range_atr.iloc[i]
        break_dn = close.iloc[i] < ma.iloc[i] - range_atr.iloc[i]

        if break_up:
            # Price broke above upper bound → strong bullish
            osc_color = strong_bullish_color
        elif break_dn:
            # Price broke below lower bound → strong bearish
            osc_color = strong_bearish_color
        else:
            # Step 4d: Use transition color when price is within range
            if no_color_on_flip:
                # Trend flip → transition color
                osc_color = transition_color
            else:
                # Use trend-based color
                if current_trend_dir == 1:
                    osc_color = weak_bullish_color
                else:
                    osc_color = weak_bearish_color

        oscillator_color.iloc[i] = osc_color
        prev_trend_dir = current_trend_dir

    return oscillator, oscillator_color, ma, range_atr


__all__ = [
    "calculate_range_oscillator",
]
