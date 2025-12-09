"""ATR-based Range calculation.

This module provides ATR-based range bands calculation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta


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


__all__ = [
    "calculate_atr_range",
]

