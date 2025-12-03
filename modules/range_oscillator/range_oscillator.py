"""Range Oscillator indicator (Zeiierman).

This module implements the Range Oscillator indicator ported from Pine Script.
The indicator calculates a weighted moving average based on price deltas and
uses ATR-based range bands to create an oscillator with dynamic heatmap colors.

LUỒNG HOẠT ĐỘNG:
================

1. HELPER FUNCTIONS (Color Utilities)
   - (removed)

2. CORE CALCULATIONS (Tính toán cơ bản)
   - calculate_weighted_ma: Tính weighted MA dựa trên price deltas
   - calculate_atr_range: Tính ATR-based range bands
   - calculate_trend_direction: Xác định trend direction (bullish/bearish)

3. VISUALIZATION (Màu sắc và heatmap)
   - (removed)

4. MAIN FUNCTION (Hàm chính)
   - calculate_range_oscillator: Orchestrates toàn bộ quá trình tính toán

CHI TIẾT LUỒNG HOẠT ĐỘNG:
==========================

Bước 1: Tính Weighted Moving Average
  - Với mỗi bar, tính delta = |close[i] - close[i+1]|
  - Weight w = delta / close[i+1]
  - Weighted MA = Σ(close[i] * w) / Σ(w)
  - Mục đích: Nhấn mạnh các bar có biến động lớn hơn

Bước 2: Tính ATR Range
  - Tính ATR với length 2000 (fallback 200 nếu không đủ data)
  - Range ATR = ATR * multiplier (default 2.0)
  - Mục đích: Xác định độ rộng của range bands

Bước 3: Xác định Trend Direction
  - So sánh close với MA:
    * close > MA → trend = 1 (bullish)
    * close < MA → trend = -1 (bearish)
    * close == MA → giữ giá trị trước đó
  - Mục đích: Xác định bias để chọn màu heatmap phù hợp

Bước 4: Tính Oscillator Value
  - Oscillator = 100 * (close - MA) / RangeATR
  - Giá trị từ -100 đến +100:
    * +100: Price ở upper bound của range
    * 0: Price ở equilibrium (MA)
    * -100: Price ở lower bound của range

Bước 5: Tính Heatmap Colors
  - Chia last 100 oscillator values thành N levels
  - Đếm số lần mỗi level được "touch"
  - Gradient màu dựa trên số touches:
    * < heat_thresh → cold color (weak)
    * >= heat_thresh + 10 → hot color (strong)
    * Giữa → gradient interpolation
  - Tìm level gần nhất với giá trị hiện tại và trả về màu

Bước 6: Xác định Final Color
  - Breakout lên trên (close > MA + RangeATR) → strong bullish color
  - Breakout xuống dưới (close < MA - RangeATR) → strong bearish color
  - Trend flip (trend_dir thay đổi) → transition color
  - Còn lại → heatmap color

Original Pine Script:
    https://creativecommons.org/licenses/by-nc-sa/4.0/
    © Zeiierman
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta


# ============================================================================
# 1. CORE CALCULATIONS - Basic Metrics
# ============================================================================


def calculate_weighted_ma(
    close: pd.Series,
    length: int = 50,
) -> pd.Series:
    """Calculate weighted moving average based on price deltas.

    This function calculates a weighted moving average where larger price
    movements receive higher weights. This emphasizes recent volatility and
    creates a more responsive equilibrium line compared to simple MA.

    Port of Pine Script logic:
        sumWeightedClose = 0.0
        sumWeights = 0.0
        for i = 0 to length - 1 by 1
            delta = math.abs(close[i] - close[i + 1])
            w = delta / close[i + 1]
            sumWeightedClose := sumWeightedClose + close[i] * w
            sumWeights := sumWeights + w
        ma = sumWeights != 0 ? sumWeightedClose / sumWeights : na

    Args:
        close: Close price series.
        length: Number of bars to use for calculation (default: 50).

    Returns:
        Series containing weighted moving average values.
        First `length` values are NaN.
    """
    if len(close) < length + 1:
        return pd.Series(np.nan, index=close.index, dtype="float64")

    ma_values = []
    for i in range(len(close)):
        if i < length:
            ma_values.append(np.nan)
            continue

        sum_weighted_close = 0.0
        sum_weights = 0.0

        for j in range(length):
            idx = i - j
            prev_idx = idx - 1
            if prev_idx < 0:
                break

            delta = abs(close.iloc[idx] - close.iloc[prev_idx])
            w = delta / close.iloc[prev_idx] if close.iloc[prev_idx] != 0 else 0.0

            sum_weighted_close += close.iloc[idx] * w
            sum_weights += w

        ma_value = sum_weighted_close / sum_weights if sum_weights != 0 else np.nan
        ma_values.append(ma_value)

    return pd.Series(ma_values, index=close.index, dtype="float64")


def calculate_atr_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    mult: float = 2.0,
    atr_length_primary: int = 2000,
    atr_length_fallback: int = 200,
) -> pd.Series:
    """Calculate ATR-based range bands.

    Calculates the Average True Range (ATR) and multiplies it by a factor
    to create dynamic range bands. These bands adapt to market volatility,
    expanding during volatile periods and contracting during quiet periods.

    Port of Pine Script logic:
        atrRaw = nz(ta.atr(2000), ta.atr(200))
        rangeATR = atrRaw * mult

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        mult: Multiplier for ATR (default: 2.0).
        atr_length_primary: Primary ATR length (default: 2000).
        atr_length_fallback: Fallback ATR length if primary fails (default: 200).

    Returns:
        Series containing ATR-based range values.
    """
    # Try primary ATR length first
    atr_raw = ta.atr(high, low, close, length=atr_length_primary)
    if atr_raw is None or atr_raw.isna().all():
        # Fallback to shorter ATR
        atr_raw = ta.atr(high, low, close, length=atr_length_fallback)

    if atr_raw is None:
        # If both fail, return NaN series
        return pd.Series(np.nan, index=close.index, dtype="float64")

    # Fill NaN values forward, then backward
    atr_raw = atr_raw.ffill().bfill()
    if atr_raw.isna().all():
        # If still all NaN, use a default value
        atr_raw = pd.Series(close * 0.01, index=close.index)

    range_atr = atr_raw * mult
    return range_atr


def calculate_trend_direction(
    close: pd.Series,
    ma: pd.Series,
) -> pd.Series:
    """Calculate trend direction based on close vs weighted MA.

    Determines whether the current price is above or below the weighted MA,
    indicating bullish or bearish bias. This is used to select appropriate
    heatmap colors (bullish colors vs bearish colors).

    Port of Pine Script logic:
        var int trendDir = 0
        trendDir := close > ma ? 1 : close < ma ? -1 : nz(trendDir[1])

    Args:
        close: Close price series.
        ma: Moving average series (typically from calculate_weighted_ma).

    Returns:
        Series with trend direction:
        - 1: Bullish (close > MA)
        - -1: Bearish (close < MA)
        - 0: Neutral (uses previous value if close == MA)
    """
    trend_dir = pd.Series(0, index=close.index, dtype="int8")

    for i in range(len(close)):
        if pd.isna(close.iloc[i]) or pd.isna(ma.iloc[i]):
            # Use previous value if available
            if i > 0:
                trend_dir.iloc[i] = trend_dir.iloc[i - 1]
            continue

        if close.iloc[i] > ma.iloc[i]:
            trend_dir.iloc[i] = 1
        elif close.iloc[i] < ma.iloc[i]:
            trend_dir.iloc[i] = -1
        else:
            # Use previous value
            if i > 0:
                trend_dir.iloc[i] = trend_dir.iloc[i - 1]

    return trend_dir


# ============================================================================
# 2. MAIN FUNCTION - Range Oscillator Calculation
# ============================================================================


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
    "calculate_weighted_ma",
    "calculate_atr_range",
    "calculate_trend_direction",
    "calculate_range_oscillator",
]
