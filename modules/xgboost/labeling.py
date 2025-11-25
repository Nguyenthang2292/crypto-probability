"""
Labeling functions for xgboost_prediction_main.py
"""

import numpy as np
import pandas as pd
from modules.config import (
    TARGET_HORIZON,
    TARGET_BASE_THRESHOLD,
    LABEL_TO_ID,
    DYNAMIC_LOOKBACK_SHORT_MULTIPLIER,
    DYNAMIC_LOOKBACK_MEDIUM_MULTIPLIER,
    DYNAMIC_LOOKBACK_LONG_MULTIPLIER,
    DYNAMIC_LOOKBACK_VOL_LOW_THRESHOLD,
    DYNAMIC_LOOKBACK_VOL_HIGH_THRESHOLD,
    DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL,
    DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL,
    DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL,
)


def _calculate_lookback_weights(
    volatility_multiplier: pd.Series,
    vol_low_threshold: pd.Series,
    vol_high_threshold: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate weights for short, medium, and long lookback periods based on volatility.
    
    Args:
        volatility_multiplier: Series of volatility multipliers
        vol_low_threshold: Series of low volatility thresholds
        vol_high_threshold: Series of high volatility thresholds
        
    Returns:
        Tuple of (weight_short, weight_medium, weight_long) Series
    """
    # Vectorized calculation: classify volatility regime for each row
    is_low_vol = volatility_multiplier < vol_low_threshold
    is_high_vol = volatility_multiplier > vol_high_threshold
    is_medium_vol = ~(is_low_vol | is_high_vol)
    
    # Calculate weights using vectorized operations
    weight_short = (
        is_low_vol * DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL[0] +
        is_medium_vol * DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL[0] +
        is_high_vol * DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL[0]
    )
    weight_medium = (
        is_low_vol * DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL[1] +
        is_medium_vol * DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL[1] +
        is_high_vol * DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL[1]
    )
    weight_long = (
        is_low_vol * DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL[2] +
        is_medium_vol * DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL[2] +
        is_high_vol * DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL[2]
    )
    
    # Normalize weights to ensure they sum to 1
    total_weight = weight_short + weight_medium + weight_long
    # Avoid division by zero
    total_weight = total_weight.replace(0, 1.0)
    
    weight_short = weight_short / total_weight
    weight_medium = weight_medium / total_weight
    weight_long = weight_long / total_weight
    
    return weight_short, weight_medium, weight_long


def _calculate_volatility_multiplier(df: pd.DataFrame) -> pd.Series:
    """
    Calculate volatility multiplier based on ATR or rolling volatility.
    
    Args:
        df: DataFrame with price data
        
    Returns:
        Series of volatility multipliers
    """
    if "ATR_14" in df.columns:
        # Normalize ATR relative to price to get volatility measure
        atr_pct = (df["ATR_14"] / df["close"]).fillna(0.01)
        # Scale lookback: higher volatility = longer lookback (1.5x to 3x TARGET_HORIZON)
        atr_median = atr_pct.rolling(window=50, min_periods=1).median()
        volatility_multiplier = (atr_pct / atr_median).fillna(2.0)
        volatility_multiplier = volatility_multiplier.clip(lower=1.5, upper=3.0)
    else:
        # Fallback: Use rolling volatility of returns
        returns = df["close"].pct_change(fill_method=None).fillna(0)
        rolling_vol = returns.rolling(window=20, min_periods=1).std().fillna(0.01)
        vol_median = rolling_vol.rolling(window=50, min_periods=1).median().fillna(0.01)
        volatility_multiplier = (rolling_vol / vol_median).fillna(2.0).clip(lower=1.5, upper=3.0)
    
    return volatility_multiplier


def apply_directional_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels each row as UP/DOWN/NEUTRAL based on future price movement.
    
    Dynamic threshold is calculated using adaptive historical lookback period
    that scales with market volatility and recent price movements.
    
    The lookback periods are dynamically adjusted based on volatility:
    - Higher volatility = longer lookback periods
    - Lower volatility = shorter lookback periods
    
    Args:
        df: DataFrame with OHLCV data and technical indicators
        
    Returns:
        DataFrame with added columns: TargetLabel, Target, DynamicThreshold
    """
    # Handle empty DataFrame
    if len(df) == 0:
        df["TargetLabel"] = pd.Series(dtype=object)
        df["Target"] = pd.Series(dtype=float)
        df["DynamicThreshold"] = pd.Series(dtype=float)
        return df
    
    # Calculate future price change
    future_close = df["close"].shift(-TARGET_HORIZON)
    pct_change = (future_close - df["close"]) / df["close"]

    # Calculate volatility multiplier
    volatility_multiplier = _calculate_volatility_multiplier(df)
    
    # Calculate base lookback periods
    base_short = TARGET_HORIZON * DYNAMIC_LOOKBACK_SHORT_MULTIPLIER
    base_medium = TARGET_HORIZON * DYNAMIC_LOOKBACK_MEDIUM_MULTIPLIER
    base_long = TARGET_HORIZON * DYNAMIC_LOOKBACK_LONG_MULTIPLIER
    
    # Calculate rolling quantiles to avoid data leakage
    # Use a rolling window (e.g., 500 periods) to define "low" and "high" volatility
    # relative to the recent market regime, rather than the entire future history.
    # Adaptive window size: use smaller window if dataset is small
    rolling_window = min(500, len(df))
    vol_low_rolling = volatility_multiplier.rolling(window=rolling_window, min_periods=1).quantile(0.33)
    vol_high_rolling = volatility_multiplier.rolling(window=rolling_window, min_periods=1).quantile(0.67)
    
    # Fill NaN values at the beginning using forward fill (more appropriate for time series)
    # Forward fill propagates the first valid value forward, which is correct for lookback
    vol_low_rolling = vol_low_rolling.ffill().fillna(1.5)
    vol_high_rolling = vol_high_rolling.ffill().fillna(2.5)
    
    # For lookback periods, we use FIXED anchors to allow vectorized shift operations.
    # Using dynamic lookbacks (from rolling quantiles) would prevent vectorization.
    # We use the typical range of the volatility multiplier (1.5 to 3.0).
    anchor_low = 1.5
    anchor_high = 3.0
    
    # Calculate lookback periods for fixed low and high volatility anchors
    max_lookback = min(len(df) - 1, int(TARGET_HORIZON * 5))
    max_lookback = max(1, max_lookback)
    
    lookback_short_low = max(1, min(int(base_short * anchor_low), max_lookback))
    lookback_short_high = max(1, min(int(base_short * anchor_high), max_lookback))
    lookback_medium_low = max(1, min(int(base_medium * anchor_low), max_lookback))
    lookback_medium_high = max(1, min(int(base_medium * anchor_high), max_lookback))
    lookback_long_low = max(1, min(int(base_long * anchor_low), max_lookback))
    lookback_long_high = max(1, min(int(base_long * anchor_high), max_lookback))
    
    # Get reference prices for different volatility-adjusted lookback periods
    # We calculate references for both low and high volatility scenarios
    ref_short_low = df["close"].shift(lookback_short_low)
    ref_short_high = df["close"].shift(lookback_short_high)
    ref_medium_low = df["close"].shift(lookback_medium_low)
    ref_medium_high = df["close"].shift(lookback_medium_high)
    ref_long_low = df["close"].shift(lookback_long_low)
    ref_long_high = df["close"].shift(lookback_long_high)
    
    # Interpolate between low and high volatility lookbacks based on current volatility
    # We normalize based on the fixed anchors used for the lookback calculation
    vol_normalized = (volatility_multiplier - anchor_low) / (anchor_high - anchor_low + 1e-8)
    vol_normalized = vol_normalized.clip(0, 1)  # Ensure between 0 and 1
    
    # Interpolate reference prices (handle NaN values)
    ref_short = (
        ref_short_low.bfill() * (1 - vol_normalized) + 
        ref_short_high.bfill() * vol_normalized
    )
    ref_medium = (
        ref_medium_low.bfill() * (1 - vol_normalized) + 
        ref_medium_high.bfill() * vol_normalized
    )
    ref_long = (
        ref_long_low.bfill() * (1 - vol_normalized) + 
        ref_long_high.bfill() * vol_normalized
    )
    
    # For per-row dynamic lookback, we use a weighted approach with adjusted periods
    # Calculate weights based on volatility using the rolling thresholds
    weight_short, weight_medium, weight_long = _calculate_lookback_weights(
        volatility_multiplier, vol_low_rolling, vol_high_rolling
    )
    
    # Calculate weighted historical reference
    historical_ref = (
        ref_short * weight_short + 
        ref_medium * weight_medium + 
        ref_long * weight_long
    )
    historical_ref = historical_ref.fillna(ref_medium)  # Fallback to medium lookback
    
    # Calculate dynamic threshold
    historical_pct = (df["close"] - historical_ref) / historical_ref
    base_threshold = (
        historical_pct.abs()
        .fillna(TARGET_BASE_THRESHOLD)
        .clip(lower=TARGET_BASE_THRESHOLD)
    )
    
    # Apply ATR ratio if available
    atr_ratio = (
        df.get("ATR_RATIO_14_50", pd.Series(1.0, index=df.index))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=0.5, upper=2.0)
    )
    threshold_series = (base_threshold * atr_ratio).clip(lower=TARGET_BASE_THRESHOLD)
    df["DynamicThreshold"] = threshold_series

    # Assign labels
    df["TargetLabel"] = np.where(
        pct_change >= threshold_series,
        "UP",
        np.where(pct_change <= -threshold_series, "DOWN", "NEUTRAL"),
    )
    df.loc[future_close.isna(), "TargetLabel"] = np.nan
    df["Target"] = df["TargetLabel"].map(LABEL_TO_ID)
    return df
