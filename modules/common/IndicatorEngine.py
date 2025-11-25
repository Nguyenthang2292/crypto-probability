"""
Technical indicators and candlestick pattern detection utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
from colorama import Fore

from .utils import color_text


class IndicatorProfile(str, Enum):
    CORE = "core"
    XGBOOST = "xgboost"
    DEEP_LEARNING = "deep_learning"


@dataclass
class IndicatorConfig:
    include_trend: bool = True
    include_momentum: bool = True
    include_volatility: bool = True
    include_volume: bool = True
    include_candlestick: bool = False
    custom_indicators: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = field(
        default_factory=dict
    )

    @classmethod
    def for_profile(cls, profile: IndicatorProfile) -> "IndicatorConfig":
        if profile == IndicatorProfile.CORE:
            return cls(include_candlestick=False)
        if profile == IndicatorProfile.XGBOOST:
            return cls(include_candlestick=True)
        if profile == IndicatorProfile.DEEP_LEARNING:
            return cls(include_candlestick=False)
        return cls()


class IndicatorEngine:
    """
    Reusable indicator engine that can be shared across CLI, portfolio manager,
    or future deep-learning pipelines.
    """

    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or IndicatorConfig()
        self._custom_registry: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def register_indicator(
        self, name: str, func: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> None:
        """
        Register a custom indicator function that receives/returns a DataFrame.
        """
        self._custom_registry[name] = func

    def compute_features(
        self,
        df: pd.DataFrame,
        config: Optional[IndicatorConfig] = None,
        return_metadata: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, str]] | pd.DataFrame:
        """
        Calculates technical indicators/candlestick patterns based on the provided config.
        """
        cfg = config or self.config
        metadata: Dict[str, str] = {}

        def mark(category: str, before_cols: set) -> None:
            new_cols = [c for c in df.columns if c not in before_cols]
            for col in new_cols:
                metadata[col] = category

        if cfg.include_trend:
            before = set(df.columns)
            self._add_trend_features(df)
            mark("trend", before)

        if cfg.include_momentum:
            before = set(df.columns)
            self._add_momentum_features(df)
            mark("momentum", before)

        if cfg.include_volatility:
            before = set(df.columns)
            self._add_volatility_features(df)
            mark("volatility", before)

        if cfg.include_volume:
            before = set(df.columns)
            self._add_volume_features(df)
            mark("volume", before)

        if cfg.include_candlestick:
            before = set(df.columns)
            self._add_candlestick_patterns(df)
            mark("candlestick", before)

        combined_custom = {**self._custom_registry, **cfg.custom_indicators}
        for name, func in combined_custom.items():
            before = set(df.columns)
            result = func(df)
            if isinstance(result, pd.DataFrame):
                df = result
            mark(f"custom:{name}", before)

        if return_metadata:
            return df, metadata
        return df

    # ------------------------------------------------------------------ #
    # Built-in indicator blocks
    # ------------------------------------------------------------------ #
    @staticmethod
    def _add_trend_features(df: pd.DataFrame) -> None:
        df["SMA_20"] = ta.sma(df["close"], length=20)
        df["SMA_50"] = ta.sma(df["close"], length=50)
        df["SMA_200"] = ta.sma(df["close"], length=200)

    @staticmethod
    def _add_momentum_features(df: pd.DataFrame) -> None:
        rsi_9 = ta.rsi(df["close"], length=9)
        df["RSI_9"] = (
            rsi_9.fillna(50.0) if rsi_9 is not None else pd.Series(50.0, index=df.index)
        )
        rsi_14 = ta.rsi(df["close"], length=14)
        df["RSI_14"] = (
            rsi_14.fillna(50.0)
            if rsi_14 is not None
            else pd.Series(50.0, index=df.index)
        )
        rsi_25 = ta.rsi(df["close"], length=25)
        df["RSI_25"] = (
            rsi_25.fillna(50.0)
            if rsi_25 is not None
            else pd.Series(50.0, index=df.index)
        )

        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            macd = macd.copy()
            for col in ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"]:
                if col in macd.columns:
                    macd[col] = macd[col].fillna(0.0)
            df[macd.columns] = macd
        else:
            print(
                color_text(
                    "MACD calculation failed, using neutral values (0).", Fore.YELLOW
                )
            )
            df["MACD_12_26_9"] = 0.0
            df["MACDh_12_26_9"] = 0.0
            df["MACDs_12_26_9"] = 0.0

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
                    "Bollinger Bands calculation failed, using neutral value (0.5).",
                    Fore.YELLOW,
                )
            )
            df["BBP_5_2.0"] = 0.5

        stochrsi = ta.stochrsi(df["close"], length=14, rsi_length=14, k=3, d=3)
        if stochrsi is not None and not stochrsi.empty:
            stochrsi = stochrsi.copy()
            if "STOCHRSIk_14_14_3_3" in stochrsi.columns:
                stochrsi["STOCHRSIk_14_14_3_3"] = stochrsi[
                    "STOCHRSIk_14_14_3_3"
                ].fillna(50.0)
            if "STOCHRSId_14_14_3_3" in stochrsi.columns:
                stochrsi["STOCHRSId_14_14_3_3"] = stochrsi[
                    "STOCHRSId_14_14_3_3"
                ].fillna(50.0)
            df[stochrsi.columns] = stochrsi
        else:
            print(
                color_text(
                    "Stochastic RSI calculation failed, using neutral values (50).",
                    Fore.YELLOW,
                )
            )
            df["STOCHRSIk_14_14_3_3"] = 50.0
            df["STOCHRSId_14_14_3_3"] = 50.0

    @staticmethod
    def _add_volatility_features(df: pd.DataFrame) -> None:
        atr_14 = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["ATR_14"] = (
            atr_14.ffill().fillna(df["close"] * 0.01)
            if atr_14 is not None
            else pd.Series(df["close"] * 0.01, index=df.index)
        )

        atr_50 = ta.atr(df["high"], df["low"], df["close"], length=50)
        df["ATR_50"] = (
            atr_50.ffill().fillna(df["close"] * 0.01)
            if atr_50 is not None
            else pd.Series(df["close"] * 0.01, index=df.index)
        )

        df["ATR_RATIO_14_50"] = df["ATR_14"] / df["ATR_50"]
        df["ATR_RATIO_14_50"] = df["ATR_RATIO_14_50"].fillna(1.0)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df["ATR_RATIO_14_50"] = df["ATR_RATIO_14_50"].fillna(1.0)

    @staticmethod
    def _add_volume_features(df: pd.DataFrame) -> None:
        obv = ta.obv(df["close"], df["volume"])
        df["OBV"] = (
            obv.ffill().fillna(0.0)
            if obv is not None
            else pd.Series(0.0, index=df.index)
        )

    def _add_candlestick_patterns(self, df: pd.DataFrame) -> None:
        o = df["open"]
        h = df["high"]
        low = df["low"]
        c = df["close"]
        body = abs(c - o)
        upper_shadow = h - np.maximum(c, o)
        lower_shadow = np.minimum(c, o) - low
        range_hl = h - low
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
            prev_bearish & curr_bullish & (c > o.shift(1)) & (o < c.shift(1))
        ).astype(int)

        prev_bullish = c.shift(1) > o.shift(1)
        curr_bearish = o > c
        df["BEARISH_ENGULFING"] = (
            prev_bullish & curr_bearish & (o > c.shift(1)) & (c < o.shift(1))
        ).astype(int)

        body_prev = body.shift(1)
        range_prev = range_hl.shift(1)
        first_bearish = o.shift(2) > c.shift(2)
        second_small = (body_prev / range_prev) < 0.3
        third_bullish = c > o
        df["MORNING_STAR"] = (
            first_bearish
            & second_small
            & third_bullish
            & (c > (o.shift(2) + c.shift(2)) / 2)
        ).astype(int)

        first_bullish_es = c.shift(2) > o.shift(2)
        second_small_es = (body_prev / range_prev) < 0.3
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
