"""
Dataset & DataModule for Temporal Fusion Transformer (TFT).

This module provides:
- TimeSeriesDataSet creation with proper time_idx handling
- Multi-asset support via group_ids
- Lightning DataModule wrapper for train/val/test DataLoaders
- Handling of missing candles (resample/ffill) to ensure no gaps in time_idx
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
# Import from lightning.pytorch to match pytorch-forecasting's imports
try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from colorama import Fore, Style

from modules.config import (
    TARGET_HORIZON,
    DEEP_MAX_ENCODER_LENGTH,
    DEEP_MAX_PREDICTION_LENGTH,
    DEEP_BATCH_SIZE,
    DEEP_NUM_WORKERS,
    DEEP_TARGET_COL,
    DEEP_TARGET_COL_CLASSIFICATION,
    DEEP_DATASET_DIR,
    DEEP_USE_TRIPLE_BARRIER,
)
from modules.common.utils import color_text


class TFTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Temporal Fusion Transformer.
    
    Handles:
    - TimeSeriesDataSet creation from preprocessed DataFrames
    - Train/validation/test DataLoaders
    - Proper time_idx handling with missing candle resampling
    - Multi-asset support via group_ids
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        target_col: str = DEEP_TARGET_COL,
        task_type: str = "regression",
        max_encoder_length: int = DEEP_MAX_ENCODER_LENGTH,
        max_prediction_length: int = DEEP_MAX_PREDICTION_LENGTH,
        batch_size: int = DEEP_BATCH_SIZE,
        num_workers: int = DEEP_NUM_WORKERS,
        dataset_dir: str = DEEP_DATASET_DIR,
        add_relative_time_idx: bool = True,
        add_target_scales: bool = True,
        allow_missing_timesteps: bool = False,
        timeframe: Optional[str] = None,
        max_ffill_limit: int = 5,
        use_interpolation: bool = True,
        max_gap_candles: int = 10,
    ):
        """
        Args:
            train_df: Training DataFrame (preprocessed)
            val_df: Validation DataFrame (preprocessed)
            test_df: Optional test DataFrame (preprocessed)
            target_col: Target column name
            task_type: 'regression' or 'classification'
            max_encoder_length: Lookback window length
            max_prediction_length: Prediction horizon length
            batch_size: Batch size for DataLoaders
            num_workers: Number of DataLoader workers
            dataset_dir: Directory to save/load dataset metadata
            add_relative_time_idx: Whether to add relative time index
            add_target_scales: Whether to add target scales for normalization
            allow_missing_timesteps: Whether to allow missing timesteps (if False, will resample)
            timeframe: Optional timeframe string (e.g., '1h', '4h') for accurate time_idx calculation.
                      If None, will be inferred from data.
            max_ffill_limit: Maximum number of consecutive candles to forward fill (default: 5).
                            Prevents infinite forward fill for large gaps.
            use_interpolation: Whether to use linear interpolation for short gaps (default: True).
                             If True, gaps <= max_gap_candles will use interpolation instead of ffill.
            max_gap_candles: Maximum gap size (in candles) to use interpolation (default: 10).
                            Gaps larger than this will use limited ffill or be dropped.
        """
        super().__init__()
        self.train_df = train_df.copy()
        self.val_df = val_df.copy()
        self.test_df = test_df.copy() if test_df is not None else None
        self.target_col = target_col
        self.task_type = task_type
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.add_relative_time_idx = add_relative_time_idx
        self.add_target_scales = add_target_scales
        self.allow_missing_timesteps = allow_missing_timesteps
        self.timeframe = timeframe
        self.max_ffill_limit = max_ffill_limit
        self.use_interpolation = use_interpolation
        self.max_gap_candles = max_gap_candles

        # Will be set in setup()
        self.training: Optional[TimeSeriesDataSet] = None
        self.validation: Optional[TimeSeriesDataSet] = None
        self.test: Optional[TimeSeriesDataSet] = None

    def prepare_data(self) -> None:
        """Prepare data - resample to handle missing candles and create time_idx."""
        # Prepare training data
        self.train_df = self._prepare_dataframe(self.train_df)
        
        # Prepare validation data
        self.val_df = self._prepare_dataframe(self.val_df)
        
        # Prepare test data if provided
        if self.test_df is not None:
            self.test_df = self._prepare_dataframe(self.test_df)

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for TimeSeriesDataSet:
        1. Ensure timestamp is datetime
        2. Resample to handle missing candles (if allow_missing_timesteps=False)
        3. Create monotonically increasing time_idx per symbol
        4. Ensure no gaps in time_idx
        """
        df = df.copy()

        # Ensure timestamp is datetime
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Ensure symbol column exists
        if "symbol" not in df.columns:
            df["symbol"] = "default"

        # Sort by symbol and timestamp
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        # Resample per symbol to handle missing candles
        if not self.allow_missing_timesteps:
            df = self._resample_missing_candles(df)

        # Create time_idx: use candle_index if available, otherwise calculate
        df = self._create_time_idx(df)

        return df

    def _resample_missing_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data to fill missing candles with intelligent gap handling.
        
        Strategy:
        1. For short gaps (<= max_gap_candles): Use linear interpolation if enabled
        2. For medium gaps: Use limited forward fill (max_ffill_limit)
        3. For large gaps: Drop or mark as NaN to avoid artificial flat data
        
        This prevents infinite forward fill that could create misleading flat data
        during exchange maintenance or delisting periods.
        """
        processed_dfs = []

        for symbol in df["symbol"].unique():
            symbol_df = df[df["symbol"] == symbol].copy()
            symbol_df = symbol_df.set_index("timestamp")

            # Infer frequency from data
            if len(symbol_df) > 1:
                freq = pd.infer_freq(symbol_df.index)
                if freq is None:
                    # Try to infer from median time difference
                    time_diffs = symbol_df.index.to_series().diff().dropna()
                    freq = time_diffs.median()
                    freq = pd.Timedelta(freq)

                # Resample to create full time index (including missing candles)
                symbol_df_resampled = symbol_df.resample(freq)
                
                # Fill missing values based on gap size
                if self.use_interpolation:
                    # Use interpolation for short gaps (smoother than ffill)
                    symbol_df = symbol_df_resampled.interpolate(
                        method="linear",
                        limit=self.max_gap_candles,
                        limit_direction="both"
                    )
                    
                    # For remaining gaps (medium size), use limited forward fill
                    symbol_df = symbol_df.ffill(limit=self.max_ffill_limit)
                else:
                    # Use limited forward fill only (prevents infinite fill)
                    symbol_df = symbol_df_resampled.ffill(limit=self.max_ffill_limit)
                
                # For large gaps that couldn't be filled, we keep them as NaN
                # This is better than creating artificial flat data
                
                # Drop rows that were completely NaN (before first valid data or after last valid data)
                # But keep rows with some NaN values (partial data is better than nothing)
                symbol_df = symbol_df.dropna(how="all")
                
                # Recalculate candle_index after resampling (if it existed)
                # New rows from resampling need updated candle_index
                if "candle_index" in symbol_df.columns and self.timeframe is not None:
                    from modules.common.utils import timeframe_to_minutes
                    timeframe_minutes = timeframe_to_minutes(self.timeframe)
                    timeframe_seconds = timeframe_minutes * 60
                    
                    timestamps = pd.to_datetime(symbol_df.index)
                    min_timestamp = timestamps.min()
                    time_deltas = (timestamps - min_timestamp).total_seconds()
                    symbol_df["candle_index"] = (time_deltas / timeframe_seconds).astype(int)

            symbol_df = symbol_df.reset_index()
            symbol_df["symbol"] = symbol
            processed_dfs.append(symbol_df)

        result_df = pd.concat(processed_dfs, ignore_index=True)
        return result_df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    def _create_time_idx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time_idx per symbol, reusing candle_index if available from pipeline.
        
        Strategy:
        1. If candle_index exists: Use it as time_idx (per symbol, starting from 0)
        2. If candle_index doesn't exist: Calculate time_idx based on timestamps
        
        This ensures consistency between the feature (candle_index) used for training
        and the index (time_idx) used for sequence ordering in TFT.
        
        Note: candle_index from pipeline is calculated globally, but time_idx needs
        to be per symbol (starting from 0 for each symbol).
        """
        df = df.copy()
        
        # Check if candle_index exists (from deeplearning_data_pipeline)
        if "candle_index" in df.columns:
            # Use candle_index as base, but make it per-symbol (starting from 0 for each symbol)
            df["time_idx"] = 0
            
            for symbol in df["symbol"].unique():
                mask = df["symbol"] == symbol
                symbol_df = df[mask].copy()
                
                if len(symbol_df) == 0:
                    continue
                
                # Get candle_index for this symbol
                symbol_candle_index = symbol_df["candle_index"].values
                
                # Normalize to start from 0 per symbol
                # This ensures each symbol's time_idx starts from 0
                min_candle_idx = symbol_candle_index.min()
                time_idx_values = symbol_candle_index - min_candle_idx
                
                df.loc[mask, "time_idx"] = time_idx_values
            
            print(
                color_text(
                    f"Using candle_index from pipeline as time_idx (per-symbol normalized)",
                    Fore.GREEN,
                )
            )
        else:
            # Fallback: Calculate time_idx from timestamps (original logic)
            df["time_idx"] = 0

            for symbol in df["symbol"].unique():
                mask = df["symbol"] == symbol
                symbol_df = df[mask].copy()
                
                if len(symbol_df) == 0:
                    continue
                
                # Get timestamps for this symbol
                timestamps = pd.to_datetime(symbol_df["timestamp"])
                min_timestamp = timestamps.min()
                
                # Calculate timeframe interval
                if self.timeframe is not None:
                    # Use provided timeframe
                    from modules.common.utils import timeframe_to_minutes
                    timeframe_minutes = timeframe_to_minutes(self.timeframe)
                    timeframe_seconds = timeframe_minutes * 60
                else:
                    # Infer timeframe from median time difference
                    if len(timestamps) > 1:
                        time_diffs = timestamps.diff().dropna()
                        median_diff = time_diffs.median()
                        timeframe_seconds = median_diff.total_seconds()
                        
                        # Round to nearest reasonable interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)
                        if timeframe_seconds < 90:  # < 1.5 minutes
                            timeframe_seconds = 60  # 1 minute
                        elif timeframe_seconds < 450:  # < 7.5 minutes
                            timeframe_seconds = 300  # 5 minutes
                        elif timeframe_seconds < 1350:  # < 22.5 minutes
                            timeframe_seconds = 900  # 15 minutes
                        elif timeframe_seconds < 2700:  # < 45 minutes
                            timeframe_seconds = 1800  # 30 minutes
                        elif timeframe_seconds < 5400:  # < 90 minutes
                            timeframe_seconds = 3600  # 1 hour
                        elif timeframe_seconds < 21600:  # < 6 hours
                            timeframe_seconds = 14400  # 4 hours
                        elif timeframe_seconds < 43200:  # < 12 hours
                            timeframe_seconds = 28800  # 8 hours
                        else:
                            timeframe_seconds = 86400  # 1 day
                    else:
                        # Fallback: use 1 hour if only one timestamp
                        timeframe_seconds = 3600
                
                # Calculate time_idx based on actual timestamp differences
                # This ensures gaps in time are properly reflected
                if isinstance(timestamps, pd.DatetimeIndex):
                    time_deltas = (timestamps - min_timestamp).total_seconds()
                else:
                    # If it's a Series, convert to Timedelta and get total_seconds
                    time_deltas = (timestamps - min_timestamp).dt.total_seconds()
                time_idx_values = (time_deltas / timeframe_seconds).astype(int)
                
                df.loc[mask, "time_idx"] = time_idx_values.values
            
            print(
                color_text(
                    f"Calculated time_idx from timestamps (candle_index not found)",
                    Fore.YELLOW,
                )
            )

        return df

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for training, validation, and testing."""
        if stage == "test" and self.test_df is not None:
            # Create test dataset
            self.test = self._create_dataset(
                self.test_df, training=False, predict_mode=True
            )
        elif stage == "fit" or stage is None:
            # Create training and validation datasets
            self.training = self._create_dataset(self.train_df, training=True)
            self.validation = self._create_dataset(
                self.val_df, training=False, predict_mode=False
            )

    def _create_dataset(
        self,
        df: pd.DataFrame,
        training: bool = True,
        predict_mode: bool = False,
    ) -> TimeSeriesDataSet:
        """
        Create TimeSeriesDataSet from DataFrame.

        Args:
            df: Preprocessed DataFrame
            training: Whether this is training dataset (affects normalization)
            predict_mode: Whether to create dataset for prediction (no target required)

        Returns:
            TimeSeriesDataSet instance
        """
        # TimeSeriesDataSet doesn't allow column names with '.' characters
        # Replace '.' with '_' in column names
        df = df.copy()
        df.columns = [col.replace('.', '_') for col in df.columns]
        
        # Identify feature columns
        exclude_cols = {
            "timestamp",
            "symbol",
            "time_idx",
            self.target_col,
            "future_pct_change",
            "triple_barrier_label",
            "TargetLabel",
            "Target",
        }

        # Separate known future features (time-based features we know in advance)
        known_future_keywords = [
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "day_of_month_sin",
            "day_of_month_cos",
            "hours_to_funding",
            "is_funding_time",
            "candle_index",
        ]
        known_future_reals = [
            col
            for col in df.columns
            if col in known_future_keywords
            and col not in exclude_cols
            and pd.api.types.is_numeric_dtype(df[col])
        ]

        # Time-varying unknown reals (features that change over time, not known in advance)
        time_varying_unknown_reals = [
            col
            for col in df.columns
            if col not in exclude_cols
            and col not in known_future_reals
            and pd.api.types.is_numeric_dtype(df[col])
            and not col.startswith("is_")
        ]

        # Static reals (features that don't change per time series, e.g., symbol-specific constants)
        # For crypto, we typically don't have static reals, but we can add them if needed
        static_reals: List[str] = []

        # Categorical features (if any)
        time_varying_known_categoricals: List[str] = []
        static_categoricals: List[str] = []

        # TimeSeriesDataSet doesn't allow NaN values in features
        # Fill NaN values with forward fill then backward fill
        all_feature_cols = known_future_reals + time_varying_unknown_reals + static_reals
        for col in all_feature_cols:
            if col in df.columns and df[col].isna().any():
                # Forward fill then backward fill (per symbol if symbol column exists)
                if "symbol" in df.columns:
                    df[col] = df.groupby("symbol")[col].fillna(method='ffill').fillna(method='bfill')
                else:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                # If still NaN (all values were NaN), fill with 0
                df[col] = df[col].fillna(0)

        # Ensure target column exists (unless in predict mode)
        if not predict_mode and self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        # Create dataset
        dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=self.target_col if not predict_mode else None,
            group_ids=["symbol"],
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=known_future_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            static_reals=static_reals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            static_categoricals=static_categoricals,
            add_relative_time_idx=self.add_relative_time_idx,
            add_target_scales=self.add_target_scales,
            target_normalizer=GroupNormalizer(groups=["symbol"], transformation="softplus")
            if self.task_type == "regression"
            else None,
            allow_missing_timesteps=self.allow_missing_timesteps,
        )

        return dataset

    def train_dataloader(self):
        """Return training DataLoader."""
        if self.training is None:
            raise RuntimeError("Must call setup('fit') first")
        return self.training.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        """Return validation DataLoader."""
        if self.validation is None:
            raise RuntimeError("Must call setup('fit') first")
        return self.validation.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        """Return test DataLoader."""
        if self.test is None:
            raise RuntimeError("Must call setup('test') first")
        return self.test.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def save_dataset_metadata(self, filepath: Optional[str] = None) -> None:
        """Save dataset metadata for later use in inference."""
        if filepath is None:
            filepath = self.dataset_dir / "dataset_metadata.pkl"

        metadata = {
            "target_col": self.target_col,
            "task_type": self.task_type,
            "max_encoder_length": self.max_encoder_length,
            "max_prediction_length": self.max_prediction_length,
            "training_dataset": self.training if self.training else None,
        }

        with open(filepath, "wb") as f:
            pickle.dump(metadata, f)

        print(
            color_text(
                f"Saved dataset metadata to {filepath}",
                Fore.GREEN,
            )
        )

    def load_dataset_metadata(self, filepath: Optional[str] = None) -> Dict:
        """Load dataset metadata."""
        if filepath is None:
            filepath = self.dataset_dir / "dataset_metadata.pkl"

        if not Path(filepath).exists():
            raise FileNotFoundError(f"Dataset metadata not found at {filepath}")

        with open(filepath, "rb") as f:
            metadata = pickle.load(f)

        return metadata

    def get_dataset_info(self) -> Dict:
        """Get information about the datasets."""
        info = {
            "train_samples": len(self.train_df) if self.train_df is not None else 0,
            "val_samples": len(self.val_df) if self.val_df is not None else 0,
            "test_samples": len(self.test_df) if self.test_df is not None else 0,
            "max_encoder_length": self.max_encoder_length,
            "max_prediction_length": self.max_prediction_length,
            "target_col": self.target_col,
            "task_type": self.task_type,
            "batch_size": self.batch_size,
        }

        if self.training is not None:
            info["training_dataset_size"] = len(self.training)
            info["training_num_features"] = len(
                self.training.reals + self.training.categoricals
            )

        if self.validation is not None:
            info["validation_dataset_size"] = len(self.validation)

        if self.test is not None:
            info["test_dataset_size"] = len(self.test)

        return info


def create_tft_datamodule(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    target_col: Optional[str] = None,
    task_type: str = "regression",
    max_encoder_length: int = DEEP_MAX_ENCODER_LENGTH,
    max_prediction_length: int = DEEP_MAX_PREDICTION_LENGTH,
    batch_size: int = DEEP_BATCH_SIZE,
    timeframe: Optional[str] = None,
    max_ffill_limit: int = 5,
    use_interpolation: bool = True,
    max_gap_candles: int = 10,
    **kwargs,
) -> TFTDataModule:
    """
    Convenience function to create TFTDataModule.

    Args:
        train_df: Training DataFrame (preprocessed)
        val_df: Validation DataFrame (preprocessed)
        test_df: Optional test DataFrame (preprocessed)
        target_col: Target column name (defaults based on task_type)
        task_type: 'regression' or 'classification'
        max_encoder_length: Lookback window length
        max_prediction_length: Prediction horizon length
        batch_size: Batch size for DataLoaders
        timeframe: Optional timeframe string (e.g., '1h', '4h') for accurate time_idx calculation.
                   If None, will be inferred from data.
        max_ffill_limit: Maximum number of consecutive candles to forward fill (default: 5)
        use_interpolation: Whether to use linear interpolation for short gaps (default: True)
        max_gap_candles: Maximum gap size (in candles) to use interpolation (default: 10)
        **kwargs: Additional arguments passed to TFTDataModule

    Returns:
        TFTDataModule instance
    """
    # Set default target column based on task type
    if target_col is None:
        if task_type == "classification" and DEEP_USE_TRIPLE_BARRIER:
            target_col = DEEP_TARGET_COL_CLASSIFICATION
        else:
            target_col = DEEP_TARGET_COL

    datamodule = TFTDataModule(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_col=target_col,
        task_type=task_type,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        batch_size=batch_size,
        timeframe=timeframe,
        max_ffill_limit=max_ffill_limit,
        use_interpolation=use_interpolation,
        max_gap_candles=max_gap_candles,
        **kwargs,
    )

    return datamodule

