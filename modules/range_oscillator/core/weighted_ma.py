"""Weighted Moving Average calculation.

This module provides the weighted moving average calculation based on price deltas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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


__all__ = [
    "calculate_weighted_ma",
]

