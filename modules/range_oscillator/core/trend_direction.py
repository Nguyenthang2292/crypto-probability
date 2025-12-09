"""Trend Direction calculation.

This module provides trend direction calculation based on close vs weighted MA.
"""

from __future__ import annotations

import pandas as pd


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


__all__ = [
    "calculate_trend_direction",
]

