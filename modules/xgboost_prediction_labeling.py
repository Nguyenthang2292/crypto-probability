"""
Labeling functions for xgboost_prediction_main.py
"""

import numpy as np
import pandas as pd
from .config import TARGET_HORIZON, TARGET_BASE_THRESHOLD, LABEL_TO_ID


def apply_directional_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels each row as UP/DOWN/NEUTRAL based on future price movement.
    
    Dynamic threshold is calculated using historical price from TARGET_HORIZON * 2 
    (48 candles) ago to better capture longer-term volatility patterns.
    """
    future_close = df["close"].shift(-TARGET_HORIZON)
    pct_change = (future_close - df["close"]) / df["close"]

    historical_ref = df["close"].shift(TARGET_HORIZON * 2)  # Lookback 48 candles for dynamic threshold
    historical_pct = (df["close"] - historical_ref) / historical_ref
    base_threshold = (
        historical_pct.abs()
        .fillna(TARGET_BASE_THRESHOLD)
        .clip(lower=TARGET_BASE_THRESHOLD)
    )
    atr_ratio = (
        df.get("ATR_RATIO_14_50", pd.Series(1.0, index=df.index))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=0.5, upper=2.0)
    )
    threshold_series = (base_threshold * atr_ratio).clip(lower=TARGET_BASE_THRESHOLD)
    df["DynamicThreshold"] = threshold_series

    df["TargetLabel"] = np.where(
        pct_change >= threshold_series,
        "UP",
        np.where(pct_change <= -threshold_series, "DOWN", "NEUTRAL"),
    )
    df.loc[future_close.isna(), "TargetLabel"] = np.nan
    df["Target"] = df["TargetLabel"].map(LABEL_TO_ID)
    return df
