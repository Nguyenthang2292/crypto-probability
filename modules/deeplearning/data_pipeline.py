"""
Deep Learning Data Preparation Pipeline for Temporal Fusion Transformer (TFT).

This module provides comprehensive data preprocessing including:
- OHLCV fetching via DataFetcher
- Target engineering (Log Returns, % Change, Triple Barrier Method)
- Fractional Differentiation for stationarity
- Technical indicators and volatility metrics
- Known-future features (time-of-day, day-of-week, funding schedule)
- Per-symbol normalization with StandardScaler
- Chronological train/validation/test split
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from colorama import Fore, Style

from modules.common.DataFetcher import DataFetcher
from modules.common.IndicatorEngine import IndicatorEngine, IndicatorConfig, IndicatorProfile
from modules.config import (
    TARGET_HORIZON,
    DEEP_TRIPLE_BARRIER_TP_THRESHOLD,
    DEEP_TRIPLE_BARRIER_SL_THRESHOLD,
    DEEP_FRACTIONAL_DIFF_D,
    DEEP_FRACTIONAL_DIFF_WINDOW,
    DEEP_USE_FRACTIONAL_DIFF,
    DEEP_USE_TRIPLE_BARRIER,
    DEEP_SCALER_DIR,
    DEEP_TRAIN_RATIO,
    DEEP_VAL_RATIO,
    DEEP_TEST_RATIO,
    DEEP_USE_FEATURE_SELECTION,
    DEEP_FEATURE_SELECTION_METHOD,
    DEEP_FEATURE_SELECTION_TOP_K,
    DEEP_FEATURE_COLLINEARITY_THRESHOLD,
    DEEP_FEATURE_SELECTION_DIR,
)
from modules.deeplearning.feature_selection import FeatureSelector
from modules.common.utils import color_text, timeframe_to_minutes


class TripleBarrierLabeler:
    """
    Implements Triple Barrier Method for robust labeling:
    - Take Profit (TP): Upper barrier
    - Stop Loss (SL): Lower barrier
    - Time Limit: Maximum holding period
    """

    def __init__(
        self,
        tp_threshold: float = DEEP_TRIPLE_BARRIER_TP_THRESHOLD,
        sl_threshold: float = DEEP_TRIPLE_BARRIER_SL_THRESHOLD,
        time_limit: int = None,  # None means use TARGET_HORIZON
    ):
        self.tp_threshold = tp_threshold
        self.sl_threshold = sl_threshold
        self.time_limit = time_limit or TARGET_HORIZON

    def label(
        self, df: pd.DataFrame, price_col: str = "close"
    ) -> pd.DataFrame:
        """
        Apply Triple Barrier Method labeling.

        Returns:
            DataFrame with 'triple_barrier_label' column:
            - 1: TP hit (profit)
            - -1: SL hit (loss)
            - 0: Time limit reached (neutral)
            - np.nan: Insufficient future data
        """
        labels = pd.Series(index=df.index, dtype=float)
        prices = df[price_col].values

        for i in range(len(df)):
            if i + self.time_limit >= len(df):
                labels.iloc[i] = np.nan
                continue

            entry_price = prices[i]
            tp_price = entry_price * (1 + self.tp_threshold)
            sl_price = entry_price * (1 - self.sl_threshold)

            # Check future prices
            future_prices = prices[i + 1 : i + 1 + self.time_limit]
            tp_hit = np.any(future_prices >= tp_price)
            sl_hit = np.any(future_prices <= sl_price)

            if tp_hit and sl_hit:
                # Both hit - check which comes first
                tp_idx = np.where(future_prices >= tp_price)[0][0]
                sl_idx = np.where(future_prices <= sl_price)[0][0]
                labels.iloc[i] = 1 if tp_idx < sl_idx else -1
            elif tp_hit:
                labels.iloc[i] = 1
            elif sl_hit:
                labels.iloc[i] = -1
            else:
                # Time limit reached
                labels.iloc[i] = 0

        df["triple_barrier_label"] = labels
        return df


class FractionalDifferentiator:
    """
    Applies Fractional Differentiation to preserve memory while ensuring stationarity.
    Uses the fixed-width window fractional differentiation method.
    """

    def __init__(
        self, d: float = DEEP_FRACTIONAL_DIFF_D, window: int = DEEP_FRACTIONAL_DIFF_WINDOW
    ):
        """
        Args:
            d: Fractional differentiation order (0 < d < 1)
            window: Window size for computation
        """
        self.d = d
        self.window = window

    def differentiate(self, series: pd.Series) -> pd.Series:
        """
        Apply fractional differentiation.

        Formula: X_t^d = sum_{k=0}^{window} (-1)^k * C(d, k) * X_{t-k}
        where C(d, k) = d * (d-1) * ... * (d-k+1) / k!
        """
        result = pd.Series(index=series.index, dtype=float)
        values = series.values

        # Pre-compute binomial coefficients
        coeffs = self._compute_coefficients(self.d, self.window)

        for i in range(len(series)):
            if i < self.window:
                result.iloc[i] = np.nan
                continue

            diff_value = 0.0
            for k in range(self.window + 1):
                if i - k >= 0:
                    diff_value += coeffs[k] * values[i - k]

            result.iloc[i] = diff_value

        return result

    @staticmethod
    def _compute_coefficients(d: float, window: int) -> np.ndarray:
        """Compute binomial coefficients for fractional differentiation."""
        coeffs = np.zeros(window + 1)
        coeffs[0] = 1.0
        for k in range(1, window + 1):
            coeffs[k] = coeffs[k - 1] * (d - k + 1) / k
        return coeffs


class DeepLearningDataPipeline:
    """
    Comprehensive data preparation pipeline for deep learning models.
    """

    def __init__(
        self,
        data_fetcher: DataFetcher,
        indicator_engine: Optional[IndicatorEngine] = None,
        use_fractional_diff: bool = DEEP_USE_FRACTIONAL_DIFF,
        fractional_diff_d: float = DEEP_FRACTIONAL_DIFF_D,
        use_triple_barrier: bool = DEEP_USE_TRIPLE_BARRIER,
        triple_barrier_tp: float = DEEP_TRIPLE_BARRIER_TP_THRESHOLD,
        triple_barrier_sl: float = DEEP_TRIPLE_BARRIER_SL_THRESHOLD,
        scaler_dir: str = DEEP_SCALER_DIR,
        use_feature_selection: bool = DEEP_USE_FEATURE_SELECTION,
        feature_selection_method: str = DEEP_FEATURE_SELECTION_METHOD,
        feature_selection_top_k: int = DEEP_FEATURE_SELECTION_TOP_K,
        feature_collinearity_threshold: float = DEEP_FEATURE_COLLINEARITY_THRESHOLD,
        feature_selection_dir: str = DEEP_FEATURE_SELECTION_DIR,
    ):
        """
        Args:
            data_fetcher: DataFetcher instance for OHLCV data
            indicator_engine: IndicatorEngine instance (creates new if None)
            use_fractional_diff: Whether to apply fractional differentiation
            fractional_diff_d: Fractional differentiation order
            use_triple_barrier: Whether to use Triple Barrier Method for labeling
            triple_barrier_tp: Take profit threshold for triple barrier
            triple_barrier_sl: Stop loss threshold for triple barrier
            scaler_dir: Directory to save/load scaler parameters
            use_feature_selection: Whether to apply feature selection
            feature_selection_method: Feature selection method ('mutual_info', 'boruta', 'f_test', 'combined')
            feature_selection_top_k: Number of top features to select
            feature_collinearity_threshold: Correlation threshold for removing collinear features
            feature_selection_dir: Directory to save/load feature selection results
        """
        self.data_fetcher = data_fetcher
        self.indicator_engine = (
            indicator_engine
            or IndicatorEngine(
                IndicatorConfig.for_profile(IndicatorProfile.DEEP_LEARNING)
            )
        )
        self.use_fractional_diff = use_fractional_diff
        self.fractional_diff_d = fractional_diff_d
        self.use_triple_barrier = use_triple_barrier
        self.scaler_dir = Path(scaler_dir)
        self.scaler_dir.mkdir(parents=True, exist_ok=True)
        self.use_feature_selection = use_feature_selection

        self.fractional_diff = (
            FractionalDifferentiator(d=fractional_diff_d) if use_fractional_diff else None
        )
        self.triple_barrier = (
            TripleBarrierLabeler(
                tp_threshold=triple_barrier_tp, sl_threshold=triple_barrier_sl
            )
            if use_triple_barrier
            else None
        )

        # Initialize feature selector if enabled
        self.feature_selector = (
            FeatureSelector(
                method=feature_selection_method,
                top_k=feature_selection_top_k,
                collinearity_threshold=feature_collinearity_threshold,
                selection_dir=feature_selection_dir,
            )
            if use_feature_selection
            else None
        )

        # Store scalers per symbol
        self.scalers: Dict[str, StandardScaler] = {}

    def fetch_and_prepare(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        limit: int = 1500,
        check_freshness: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for multiple symbols and prepare for deep learning.

        Returns:
            DataFrame with multi-index (symbol, timestamp) or single symbol
        """
        all_dfs = []

        for symbol in symbols:
            print(
                color_text(
                    f"Fetching data for {symbol}...", Fore.CYAN, Style.BRIGHT
                )
            )
            df, exchange = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                check_freshness=check_freshness,
            )

            if df is None or df.empty:
                print(
                    color_text(
                        f"Warning: No data fetched for {symbol}", Fore.YELLOW
                    )
                )
                continue

            # Add symbol column
            df["symbol"] = symbol
            all_dfs.append(df)

        if not all_dfs:
            raise ValueError("No data fetched for any symbol")

        # Combine all symbols
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Process the combined dataframe
        return self.prepare_dataframe(combined_df, timeframe=timeframe)

    def prepare_dataframe(
        self, df: pd.DataFrame, timeframe: str = "1h"
    ) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline to a DataFrame.

        Steps:
        1. Target Engineering (Log Returns, % Change, Triple Barrier)
        2. Fractional Differentiation (if enabled)
        3. Technical Indicators
        4. Known-future Features
        5. Normalization (per symbol)
        6. Feature Selection (if enabled)
        """
        df = df.copy()

        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        # Step 1: Target Engineering
        df = self._engineer_targets(df)

        # Step 2: Fractional Differentiation (if enabled)
        if self.use_fractional_diff:
            df = self._apply_fractional_differentiation(df)

        # Step 3: Technical Indicators
        df = self._add_technical_indicators(df)

        # Step 4: Known-future Features
        df = self._add_known_future_features(df, timeframe)

        # Step 5: Normalization (per symbol)
        df = self._normalize_per_symbol(df)

        # Step 6: Feature Selection (if enabled)
        # Note: Feature selection should be applied after normalization
        # but we need the target column for selection, so we'll handle it separately
        # in a method that takes both X and y

        return df

    def _engineer_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer target variables: Log Returns, % Change, Triple Barrier."""
        # Log Returns
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # % Change
        df["pct_change"] = df["close"].pct_change()

        # Forward-looking returns (for prediction)
        df["future_log_return"] = (
            np.log(df["close"].shift(-TARGET_HORIZON) / df["close"])
            if "close" in df.columns
            else pd.Series(index=df.index, dtype=float)
        )
        df["future_pct_change"] = (
            (df["close"].shift(-TARGET_HORIZON) - df["close"]) / df["close"]
            if "close" in df.columns
            else pd.Series(index=df.index, dtype=float)
        )

        # Triple Barrier Method (if enabled)
        if self.use_triple_barrier and self.triple_barrier:
            df = self.triple_barrier.label(df, price_col="close")

        return df

    def _apply_fractional_differentiation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fractional differentiation to price columns."""
        if self.fractional_diff is None:
            return df

        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                # Apply per symbol if multiple symbols
                if "symbol" in df.columns:
                    for symbol in df["symbol"].unique():
                        mask = df["symbol"] == symbol
                        df.loc[mask, f"{col}_frac_diff"] = (
                            self.fractional_diff.differentiate(df.loc[mask, col])
                        )
                else:
                    df[f"{col}_frac_diff"] = self.fractional_diff.differentiate(
                        df[col]
                    )

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators and volatility metrics."""
        # Apply indicators per symbol
        if "symbol" in df.columns:
            processed_dfs = []
            for symbol in df["symbol"].unique():
                symbol_df = df[df["symbol"] == symbol].copy()
                symbol_df = self.indicator_engine.compute_features(symbol_df)
                processed_dfs.append(symbol_df)
            df = pd.concat(processed_dfs, ignore_index=True)
        else:
            df = self.indicator_engine.compute_features(df)

        # Additional volatility metrics
        if "close" in df.columns:
            # Rolling volatility
            if "symbol" in df.columns:
                for symbol in df["symbol"].unique():
                    mask = df["symbol"] == symbol
                    df.loc[mask, "volatility_20"] = (
                        df.loc[mask, "close"].rolling(20).std()
                    )
                    df.loc[mask, "volatility_50"] = (
                        df.loc[mask, "close"].rolling(50).std()
                    )
            else:
                df["volatility_20"] = df["close"].rolling(20).std()
                df["volatility_50"] = df["close"].rolling(50).std()

        return df

    def _add_known_future_features(
        self, df: pd.DataFrame, timeframe: str
    ) -> pd.DataFrame:
        """
        Generate known-future features:
        - time-of-day
        - day-of-week
        - funding schedule (for futures)
        """
        if "timestamp" not in df.columns:
            return df

        # Ensure timezone-aware
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

        # Time-of-day features (cyclical encoding)
        df["hour"] = df["timestamp"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Day-of-week features (cyclical encoding)
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Day-of-month
        df["day_of_month"] = df["timestamp"].dt.day
        df["day_of_month_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
        df["day_of_month_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)

        # Funding schedule (Binance Futures: 00:00, 08:00, 16:00 UTC)
        funding_hours = [0, 8, 16]
        df["hours_to_funding"] = df["hour"].apply(
            lambda h: min([(h - fh) % 24 for fh in funding_hours])
        )
        df["is_funding_time"] = df["hour"].isin(funding_hours).astype(int)

        # Timeframe-specific features
        timeframe_minutes = timeframe_to_minutes(timeframe)
        df["candle_index"] = (
            (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()
            / (timeframe_minutes * 60)
        ).astype(int)

        return df

    def _normalize_per_symbol(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numeric columns per symbol using StandardScaler.
        Persists scaler parameters for later use.
        """
        # Identify numeric columns to normalize (exclude targets, labels, timestamps)
        exclude_cols = {
            "timestamp",
            "symbol",
            "TargetLabel",
            "Target",
            "triple_barrier_label",
            "future_log_return",
            "future_pct_change",
            "hour",
            "day_of_week",
            "day_of_month",
            "candle_index",
        }

        numeric_cols = [
            col
            for col in df.columns
            if col not in exclude_cols
            and pd.api.types.is_numeric_dtype(df[col])
            and not col.startswith("is_")
            and not col.endswith("_sin")
            and not col.endswith("_cos")
        ]

        if not numeric_cols:
            return df

        # Normalize per symbol
        if "symbol" in df.columns:
            for symbol in df["symbol"].unique():
                mask = df["symbol"] == symbol
                symbol_df = df.loc[mask, numeric_cols].copy()

                # Fit scaler if not already fitted
                if symbol not in self.scalers:
                    scaler = StandardScaler()
                    scaler.fit(symbol_df)
                    self.scalers[symbol] = scaler
                    self._save_scaler(symbol, scaler)
                else:
                    scaler = self.scalers[symbol]

                # Transform
                df.loc[mask, numeric_cols] = scaler.transform(symbol_df)
        else:
            # Single symbol - use "default" as key
            symbol = "default"
            symbol_df = df[numeric_cols].copy()

            if symbol not in self.scalers:
                scaler = StandardScaler()
                scaler.fit(symbol_df)
                self.scalers[symbol] = scaler
                self._save_scaler(symbol, scaler)
            else:
                scaler = self.scalers[symbol]

            df[numeric_cols] = scaler.transform(symbol_df)

        return df

    def _save_scaler(self, symbol: str, scaler: StandardScaler) -> None:
        """Save scaler parameters to disk."""
        # Sanitize symbol name for filename (replace / with _)
        safe_symbol = symbol.replace("/", "_").replace("\\", "_")
        scaler_path = self.scaler_dir / f"{safe_symbol}_scaler.json"
        params = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "n_features_in_": scaler.n_features_in_,
        }
        with open(scaler_path, "w") as f:
            json.dump(params, f, indent=2)

    def load_scaler(self, symbol: str) -> Optional[StandardScaler]:
        """Load scaler parameters from disk."""
        # Sanitize symbol name for filename (replace / with _)
        safe_symbol = symbol.replace("/", "_").replace("\\", "_")
        scaler_path = self.scaler_dir / f"{safe_symbol}_scaler.json"
        if not scaler_path.exists():
            return None

        with open(scaler_path, "r") as f:
            params = json.load(f)

        scaler = StandardScaler()
        scaler.mean_ = np.array(params["mean"])
        scaler.scale_ = np.array(params["scale"])
        scaler.n_features_in_ = params["n_features_in_"]

        self.scalers[symbol] = scaler
        return scaler

    def apply_feature_selection(
        self,
        df: pd.DataFrame,
        target_col: str = "future_log_return",
        task_type: str = "regression",
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply feature selection to the preprocessed DataFrame.

        Args:
            df: Preprocessed DataFrame (after normalization)
            target_col: Name of the target column
            task_type: 'regression' or 'classification'
            symbol: Optional symbol name for per-symbol feature selection

        Returns:
            DataFrame with selected features only
        """
        if not self.use_feature_selection or self.feature_selector is None:
            return df

        if target_col not in df.columns:
            print(
                color_text(
                    f"Warning: Target column '{target_col}' not found. Skipping feature selection.",
                    Fore.YELLOW,
                )
            )
            return df

        # Separate features and target
        X = df.drop(columns=[target_col], errors="ignore")
        y = df[target_col]

        # Remove rows with NaN target values
        valid_mask = ~y.isna()
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()

        if len(X_clean) == 0:
            print(
                color_text(
                    "Warning: No valid target values. Skipping feature selection.",
                    Fore.YELLOW,
                )
            )
            return df

        # Apply feature selection
        try:
            X_selected = self.feature_selector.select_features(
                X_clean, y_clean, task_type=task_type, symbol=symbol
            )

            # Combine selected features with non-feature columns (target, timestamp, symbol, etc.)
            non_feature_cols = [
                col
                for col in df.columns
                if col not in X.columns or col in X_selected.columns
            ]
            result_df = pd.concat(
                [X_selected, df[non_feature_cols].loc[valid_mask]], axis=1
            )

            # Re-add rows that were removed due to NaN targets (with NaN in target)
            if not valid_mask.all():
                invalid_df = df[~valid_mask].copy()
                # Keep only selected features in invalid rows
                invalid_features = [
                    col for col in X_selected.columns if col in invalid_df.columns
                ]
                invalid_df = invalid_df[non_feature_cols + invalid_features]
                result_df = pd.concat([result_df, invalid_df], ignore_index=True)

            return result_df

        except Exception as e:
            print(
                color_text(
                    f"Error in feature selection: {e}. Returning original DataFrame.",
                    Fore.RED,
                )
            )
            return df

    def split_chronological(
        self,
        df: pd.DataFrame,
        train_ratio: float = DEEP_TRAIN_RATIO,
        val_ratio: float = DEEP_VAL_RATIO,
        test_ratio: float = DEEP_TEST_RATIO,
        gap: int = None,
        apply_feature_selection: bool = True,
        target_col: str = "future_log_return",
        task_type: str = "regression",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically into train/validation/test sets.
        Optionally applies feature selection on training set and applies to all splits.

        Args:
            df: Preprocessed DataFrame
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            gap: Gap between train and validation/test to prevent data leakage
                  (default: TARGET_HORIZON)
            apply_feature_selection: Whether to apply feature selection on training set
            target_col: Target column name for feature selection
            task_type: 'regression' or 'classification' for feature selection

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if gap is None:
            gap = TARGET_HORIZON

        # Ensure sorted by timestamp
        if "timestamp" in df.columns:
            df = df.sort_values(["symbol", "timestamp"] if "symbol" in df.columns else ["timestamp"]).reset_index(drop=True)

        total_len = len(df)
        train_end = int(total_len * train_ratio)
        val_end = train_end + int(total_len * val_ratio)

        # Apply gap to prevent data leakage
        train_end = max(0, train_end - gap)
        val_start = train_end + gap
        val_end = min(total_len, val_end)
        test_start = val_end + gap if val_end < total_len else val_end

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[val_start:val_end].copy() if val_start < total_len else pd.DataFrame()
        test_df = df.iloc[test_start:].copy() if test_start < total_len else pd.DataFrame()

        # Apply feature selection on training set, then apply to validation and test
        if apply_feature_selection and self.use_feature_selection and self.feature_selector:
            if not train_df.empty and target_col in train_df.columns:
                # Get symbol for feature selection (use first symbol if multiple)
                symbol = train_df["symbol"].iloc[0] if "symbol" in train_df.columns else None

                # Apply feature selection on training set
                train_df = self.apply_feature_selection(
                    train_df, target_col=target_col, task_type=task_type, symbol=symbol
                )

                # Get selected features and apply to validation and test sets
                if self.feature_selector.selected_features:
                    selected_features = self.feature_selector.selected_features
                    # Keep non-feature columns as well
                    non_feature_cols = [
                        col for col in df.columns if col not in selected_features
                    ]
                    all_cols = selected_features + non_feature_cols

                    if not val_df.empty:
                        val_cols = [col for col in all_cols if col in val_df.columns]
                        val_df = val_df[val_cols]
                    if not test_df.empty:
                        test_cols = [col for col in all_cols if col in test_df.columns]
                        test_df = test_df[test_cols]

        print(
            color_text(
                f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}",
                Fore.GREEN,
            )
        )

        return train_df, val_df, test_df

