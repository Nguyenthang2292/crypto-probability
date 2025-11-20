"""
Technical indicators and candlestick pattern detection for xgboost_prediction_main.py
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from .utils import color_text
from colorama import Fore, Style


def add_candlestick_patterns(df):
    """Detect 10 reliable candlestick patterns and add binary columns."""
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    body = abs(c - o)
    upper_shadow = h - np.maximum(c, o)
    lower_shadow = np.minimum(c, o) - l
    range_hl = h - l
    range_hl = np.where(range_hl == 0, 0.0001, range_hl)
    range_hl = pd.Series(range_hl, index=df.index)
    df["DOJI"] = (body / range_hl < 0.1).astype(int)
    df["HAMMER"] = (
        (lower_shadow > 2 * body)
        & (upper_shadow < 0.3 * body)
        & (body / range_hl < 0.3)
    ).astype(int)
    df["INVERTED_HAMMER"] = (
        (upper_shadow > 2 * body)
        & (lower_shadow < 0.3 * body)
        & (body / range_hl < 0.3)
    ).astype(int)
    df["SHOOTING_STAR"] = (
        (upper_shadow > 2 * body) & (lower_shadow < 0.3 * body) & (c < o)
    ).astype(int)
    prev_bearish = o.shift(1) > c.shift(1)
    curr_bullish = c > o
    df["BULLISH_ENGULFING"] = (
        prev_bearish
        & curr_bullish
        & (c > o.shift(1))
        & (o < c.shift(1))
    ).astype(int)
    prev_bullish = c.shift(1) > o.shift(1)
    curr_bearish = o > c
    df["BEARISH_ENGULFING"] = (
        prev_bullish
        & curr_bearish
        & (o > c.shift(1))
        & (c < o.shift(1))
    ).astype(int)
    first_bearish = o.shift(2) > c.shift(2)
    second_small = body.shift(1) / range_hl.shift(1) < 0.3
    third_bullish = c > o
    df["MORNING_STAR"] = (
        first_bearish
        & second_small
        & third_bullish
        & (c > (o.shift(2) + c.shift(2)) / 2)
    ).astype(int)
    first_bullish_es = c.shift(2) > o.shift(2)
    second_small_es = body.shift(1) / range_hl.shift(1) < 0.3
    third_bearish_es = o > c
    df["EVENING_STAR"] = (
        first_bullish_es
        & second_small_es
        & third_bearish_es
        & (c < (o.shift(2) + c.shift(2)) / 2)
    ).astype(int)
    df["PIERCING"] = (
        (o.shift(1) > c.shift(1))
        & (c > o)
        & (o < c.shift(1))
        & (c > (o.shift(1) + c.shift(1)) / 2)
        & (c < o.shift(1))
    ).astype(int)
    df["DARK_CLOUD"] = (
        (c.shift(1) > o.shift(1))
        & (o > c)
        & (o > c.shift(1))
        & (c < (o.shift(1) + c.shift(1)) / 2)
        & (c > o.shift(1))
    ).astype(int)
    return df


def calculate_indicators(df, apply_labels=True):
    """
    Calculates technical indicators and candlestick patterns.
    Does NOT drop NaN values - caller should handle that.
    
    Args:
        df: DataFrame with OHLCV data
        apply_labels: If True, applies directional labels. If False, only calculates indicators.
    
    Returns:
        DataFrame with indicators added
    """
    # Trend - Moving Averages
    df["SMA_20"] = ta.sma(df["close"], length=20)
    df["SMA_50"] = ta.sma(df["close"], length=50)
    df["SMA_200"] = ta.sma(df["close"], length=200)

    # Momentum - RSI (neutral value is 50)
    rsi_9 = ta.rsi(df["close"], length=9)
    df["RSI_9"] = rsi_9.fillna(50.0) if rsi_9 is not None else pd.Series(50.0, index=df.index)
    
    rsi_14 = ta.rsi(df["close"], length=14)
    df["RSI_14"] = rsi_14.fillna(50.0) if rsi_14 is not None else pd.Series(50.0, index=df.index)
    
    rsi_25 = ta.rsi(df["close"], length=25)
    df["RSI_25"] = rsi_25.fillna(50.0) if rsi_25 is not None else pd.Series(50.0, index=df.index)

    # MACD (neutral values: MACD=0, MACDh=0, MACDs=0)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df = pd.concat([df, macd], axis=1)
        # Fill NaN with neutral values
        for col in ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
    else:
        print(
            color_text(
                "MACD calculation failed, using neutral values (0).", Fore.YELLOW
            )
        )
        df["MACD_12_26_9"] = 0.0
        df["MACDh_12_26_9"] = 0.0
        df["MACDs_12_26_9"] = 0.0

    # Bollinger Bands (neutral value for BBP is 0.5 - middle of band)
    bbands = ta.bbands(df["close"], length=20, std=2.0)
    if bbands is not None and not bbands.empty:
        bbp_cols = [c for c in bbands.columns if c.startswith("BBP")]
        if bbp_cols:
            df["BBP_5_2.0"] = bbands[bbp_cols[0]].fillna(0.5)
        else:
            print(
                color_text(
                    "BBP column not found in Bollinger Bands output, using neutral value (0.5).",
                    Fore.YELLOW,
                )
            )
            df["BBP_5_2.0"] = 0.5
    else:
        print(
            color_text(
                "Bollinger Bands calculation failed, using neutral value (0.5).", Fore.YELLOW
            )
        )
        df["BBP_5_2.0"] = 0.5

    # Stochastic RSI (neutral values: k=50, d=50)
    stochrsi = ta.stochrsi(df["close"], length=14, rsi_length=14, k=3, d=3)
    if stochrsi is not None and not stochrsi.empty:
        df = pd.concat([df, stochrsi], axis=1)
        # Fill NaN with neutral values
        if "STOCHRSIk_14_14_3_3" in df.columns:
            df["STOCHRSIk_14_14_3_3"] = df["STOCHRSIk_14_14_3_3"].fillna(50.0)
        if "STOCHRSId_14_14_3_3" in df.columns:
            df["STOCHRSId_14_14_3_3"] = df["STOCHRSId_14_14_3_3"].fillna(50.0)
    else:
        print(
            color_text(
                "Stochastic RSI calculation failed, using neutral values (50).", Fore.YELLOW
            )
        )
        df["STOCHRSIk_14_14_3_3"] = 50.0
        df["STOCHRSId_14_14_3_3"] = 50.0

    # On-Balance Volume (use forward fill for NaN)
    obv = ta.obv(df["close"], df["volume"])
    df["OBV"] = obv.ffill().fillna(0.0) if obv is not None else pd.Series(0.0, index=df.index)

    # Volatility - ATR (use forward fill for NaN)
    atr_14 = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ATR_14"] = atr_14.ffill().fillna(df["close"] * 0.01) if atr_14 is not None else pd.Series(df["close"] * 0.01, index=df.index)
    
    atr_50 = ta.atr(df["high"], df["low"], df["close"], length=50)
    df["ATR_50"] = atr_50.ffill().fillna(df["close"] * 0.01) if atr_50 is not None else pd.Series(df["close"] * 0.01, index=df.index)
    
    df["ATR_RATIO_14_50"] = df["ATR_14"] / df["ATR_50"]
    df["ATR_RATIO_14_50"] = df["ATR_RATIO_14_50"].fillna(1.0)  # Neutral ratio is 1.0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df["ATR_RATIO_14_50"] = df["ATR_RATIO_14_50"].fillna(1.0)

    # Candlestick patterns
    df = add_candlestick_patterns(df)

    # Apply directional labels if requested
    if apply_labels:
        from .xgboost_prediction_labeling import apply_directional_labels
        df = apply_directional_labels(df)

    return df


def add_indicators(df):
    """
    Adds technical indicators and candlestick patterns using pandas_ta.
    Also applies directional labels and drops NaN values.
    """
    # Calculate indicators with labels
    df = calculate_indicators(df, apply_labels=True)

    # Drop NaN values created by indicators
    initial_len = len(df)
    df.dropna(inplace=True)
    dropped = initial_len - len(df)
    
    # Warn about data loss due to SMA 200 (requires 200 periods)
    if dropped >= 200:
        print(
            color_text(
                f"WARNING: Dropped {dropped} rows (>=200 due to SMA_200 requirement). "
                f"Consider increasing --limit to at least {initial_len + 200} to maintain sufficient training data.",
                Fore.RED,
                Style.BRIGHT,
            )
        )
    elif dropped > 0:
        print(
            color_text(
                f"Dropped {dropped} rows due to NaN values from indicators.",
                Fore.BLUE,
            )
        )
    
    if len(df) < 200:
        print(
            color_text(
                f"WARNING: Only {len(df)} rows remaining after indicator calculation. "
                f"Consider increasing data limit (--limit) to at least {len(df) + 200} for better model performance.",
                Fore.YELLOW,
                Style.BRIGHT,
            )
        )
    
    return df

