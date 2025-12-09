"""
Feature calculations for Simplified Percentile Clustering.

Computes RSI, CCI, Fisher Transform, DMI, Z-Score, and MAR (Moving Average Ratio)
features with optional standardization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

# Try to import Numba for JIT compilation, fallback if not available
try:
    from numba import njit
    NUMBA_AVAILABLE = True
    # Create JIT-compiled version of the core function
    @njit(cache=True)
    def _fisher_transform_core_jit(
        hl2: np.ndarray, high_: np.ndarray, low_: np.ndarray, n: int
    ) -> np.ndarray:
        """
        Core Fisher Transform calculation using Numba JIT for performance.
        
        This function performs the recursive Fisher Transform calculation on numpy arrays.
        It's JIT-compiled for maximum performance.
        
        Args:
            hl2: Array of (high + low) / 2 values
            high_: Array of rolling max values
            low_: Array of rolling min values
            n: Length of arrays
            
        Returns:
            Array of Fisher Transform values
        """
        value = np.zeros(n, dtype=np.float64)
        fish1 = np.zeros(n, dtype=np.float64)
        
        for i in range(1, n):
            # Check for NaN or invalid values
            if np.isnan(high_[i]) or np.isnan(low_[i]) or np.isnan(hl2[i]):
                value[i] = value[i - 1] if i > 0 else 0.0
                fish1[i] = fish1[i - 1] if i > 0 else 0.0
                continue
            
            # Normalize
            if high_[i] == low_[i] or abs(high_[i] - low_[i]) < 1e-10:
                normalized = 0.0
            else:
                normalized = (hl2[i] - low_[i]) / (high_[i] - low_[i]) - 0.5
            
            # Update value with recursive smoothing
            prev_value = value[i - 1] if i > 0 and not np.isnan(value[i - 1]) else 0.0
            new_value = 0.66 * normalized + 0.67 * prev_value
            
            # Clamp to avoid infinite values
            if new_value > 0.99:
                new_value = 0.999
            elif new_value < -0.99:
                new_value = -0.999
            value[i] = new_value
            
            # Calculate Fisher Transform
            prev_fish = fish1[i - 1] if i > 0 and not np.isnan(fish1[i - 1]) else 0.0
            val_abs = abs(value[i])
            
            if val_abs >= 1.0 or not np.isfinite(value[i]):
                fish1[i] = prev_fish
            else:
                # Safe log calculation
                denominator = 1.0 - value[i]
                if abs(denominator) < 1e-10:
                    fish1[i] = prev_fish
                else:
                    ratio = (1.0 + value[i]) / denominator
                    if ratio > 0:
                        log_val = np.log(ratio)
                        fish1[i] = 0.5 * log_val + 0.5 * prev_fish
                    else:
                        fish1[i] = prev_fish
        
        return fish1
except ImportError:
    NUMBA_AVAILABLE = False
    _fisher_transform_core_jit = None


@dataclass
class FeatureConfig:
    """Configuration for feature calculations."""

    # RSI
    use_rsi: bool = True
    rsi_len: int = 14
    rsi_standardize: bool = True

    # CCI
    use_cci: bool = True
    cci_len: int = 20
    cci_standardize: bool = True

    # Fisher
    use_fisher: bool = True
    fisher_len: int = 9
    fisher_standardize: bool = True

    # DMI
    use_dmi: bool = True
    dmi_len: int = 9
    dmi_standardize: bool = True

    # Z-Score
    use_zscore: bool = True
    zscore_len: int = 20

    # MAR (Moving Average Ratio)
    use_mar: bool = True
    mar_len: int = 14
    mar_type: str = "SMA"  # "SMA" or "EMA"
    mar_standardize: bool = True


class FeatureCalculator:
    """Calculate technical features for clustering."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    @staticmethod
    def z_score(src: pd.Series, length: int) -> pd.Series:
        """Calculate rolling z-score standardization."""
        mean = src.rolling(window=length, min_periods=1).mean()
        stdev = src.rolling(window=length, min_periods=1).std()
        return (src - mean) / stdev.replace(0, np.nan)

    @staticmethod
    def round_fisher(val: float) -> float:
        """Safe clamp for Fisher transform to avoid infinite values."""
        return 0.999 if val > 0.99 else (-0.999 if val < -0.99 else val)


    @staticmethod
    def fisher_transform(
        high: pd.Series, low: pd.Series, close: pd.Series, length: int
    ) -> pd.Series:
        """
        Calculate Fisher Transform applied to hl2 over length bars.
        
        Uses Numba JIT compilation for the core recursive calculation if available,
        falling back to pure Python if Numba is not installed.
        """
        hl2 = (high + low) / 2
        high_ = hl2.rolling(window=length, min_periods=1).max()
        low_ = hl2.rolling(window=length, min_periods=1).min()

        # Convert to numpy arrays for JIT-compiled function
        hl2_arr = hl2.values
        high_arr = high_.values
        low_arr = low_.values
        n = len(close)

        # Use JIT-compiled core function if Numba is available
        if NUMBA_AVAILABLE and _fisher_transform_core_jit is not None:
            fish1_arr = _fisher_transform_core_jit(hl2_arr, high_arr, low_arr, n)
        else:
            # Fallback to original implementation if Numba is not available
            value = np.zeros(n, dtype=np.float64)
            fish1_arr = np.zeros(n, dtype=np.float64)
            
            for i in range(1, n):
                if np.isnan(high_arr[i]) or np.isnan(low_arr[i]) or np.isnan(hl2_arr[i]):
                    value[i] = value[i - 1] if i > 0 else 0.0
                    fish1_arr[i] = fish1_arr[i - 1] if i > 0 else 0.0
                    continue

                if high_arr[i] == low_arr[i] or abs(high_arr[i] - low_arr[i]) < 1e-10:
                    normalized = 0.0
                else:
                    normalized = (hl2_arr[i] - low_arr[i]) / (high_arr[i] - low_arr[i]) - 0.5

                prev_value = value[i - 1] if i > 0 and not np.isnan(value[i - 1]) else 0.0
                new_value = 0.66 * normalized + 0.67 * prev_value
                
                # Clamp
                if new_value > 0.99:
                    new_value = 0.999
                elif new_value < -0.99:
                    new_value = -0.999
                value[i] = new_value

                prev_fish = fish1_arr[i - 1] if i > 0 and not np.isnan(fish1_arr[i - 1]) else 0.0
                val_abs = abs(value[i])
                
                if val_abs >= 1.0 or not np.isfinite(value[i]):
                    fish1_arr[i] = prev_fish
                else:
                    denominator = 1.0 - value[i]
                    if abs(denominator) < 1e-10:
                        fish1_arr[i] = prev_fish
                    else:
                        ratio = (1.0 + value[i]) / denominator
                        if ratio > 0:
                            log_val = np.log(ratio)
                            fish1_arr[i] = 0.5 * log_val + 0.5 * prev_fish
                        else:
                            fish1_arr[i] = prev_fish

        # Convert back to pandas Series
        return pd.Series(fish1_arr, index=close.index)

    @staticmethod
    def dmi_difference(
        high: pd.Series, low: pd.Series, close: pd.Series, length: int
    ) -> pd.Series:
        """Calculate simplified DMI difference (plus - minus)."""
        up = high.diff()
        down = -low.diff()

        plus_dm = pd.Series(
            np.where((up > down) & (up > 0), up, 0.0),
            index=high.index,
        )
        minus_dm = pd.Series(
            np.where((down > up) & (down > 0), down, 0.0),
            index=low.index,
        )

        # True Range
        tr_components = pd.concat(
            [
                (high - low),
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        )
        tr = tr_components.max(axis=1)

        # RMA (Running Moving Average) - using EWM with alpha = 1/length
        trur = tr.ewm(alpha=1.0 / length, adjust=False).mean()
        plus = 100 * plus_dm.ewm(alpha=1.0 / length, adjust=False).mean() / trur.replace(0, np.nan)
        minus = (
            100
            * minus_dm.ewm(alpha=1.0 / length, adjust=False).mean()
            / trur.replace(0, np.nan)
        )

        diff = plus - minus
        return diff.fillna(0.0)

    def compute_rsi(self, close: pd.Series, lookback: int) -> tuple[pd.Series, pd.Series]:
        """Compute RSI and optionally standardized version."""
        rsi = ta.rsi(close, length=self.config.rsi_len)
        if rsi is None:
            rsi = pd.Series(50.0, index=close.index)
        rsi = rsi.fillna(50.0)

        rsi_z = self.z_score(rsi, lookback)
        rsi_val = rsi_z if self.config.rsi_standardize else rsi
        return rsi, rsi_val

    def compute_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, lookback: int) -> tuple[pd.Series, pd.Series]:
        """Compute CCI and optionally standardized version."""
        cci = ta.cci(high=high, low=low, close=close, length=self.config.cci_len)
        if cci is None:
            cci = pd.Series(0.0, index=close.index)
        cci = cci.fillna(0.0)

        cci_z = self.z_score(cci, lookback)
        cci_val = cci_z if self.config.cci_standardize else cci
        return cci, cci_val

    def compute_fisher(
        self, high: pd.Series, low: pd.Series, close: pd.Series, lookback: int
    ) -> tuple[pd.Series, pd.Series]:
        """Compute Fisher Transform and optionally standardized version."""
        fisher = self.fisher_transform(high, low, close, self.config.fisher_len)
        fisher_z = self.z_score(fisher, lookback)
        fisher_val = fisher_z if self.config.fisher_standardize else fisher
        return fisher, fisher_val

    def compute_dmi(
        self, high: pd.Series, low: pd.Series, close: pd.Series, lookback: int
    ) -> tuple[pd.Series, pd.Series]:
        """Compute DMI difference and optionally standardized version."""
        dmi = self.dmi_difference(high, low, close, self.config.dmi_len)
        dmi_z = self.z_score(dmi, lookback)
        dmi_val = dmi_z if self.config.dmi_standardize else dmi
        return dmi, dmi_val

    def compute_zscore(self, close: pd.Series) -> pd.Series:
        """Compute z-score of price itself."""
        return self.z_score(close, self.config.zscore_len)

    def compute_mar(self, close: pd.Series, lookback: int) -> tuple[pd.Series, pd.Series]:
        """Compute MAR (Moving Average Ratio) and optionally standardized version."""
        if self.config.mar_type == "SMA":
            ma = ta.sma(close, length=self.config.mar_len)
        else:  # EMA
            ma = ta.ema(close, length=self.config.mar_len)

        if ma is None:
            ma = close.copy()

        mar = close / ma.replace(0, np.nan)
        mar_z = self.z_score(mar, lookback)
        mar_val = mar_z if self.config.mar_standardize else mar
        return mar, mar_val

    def compute_all(
        self, high: pd.Series, low: pd.Series, close: pd.Series, lookback: int
    ) -> dict[str, pd.Series]:
        """Compute all enabled features."""
        results = {}

        if self.config.use_rsi:
            rsi, rsi_val = self.compute_rsi(close, lookback)
            results["rsi"] = rsi
            results["rsi_val"] = rsi_val

        if self.config.use_cci:
            cci, cci_val = self.compute_cci(high, low, close, lookback)
            results["cci"] = cci
            results["cci_val"] = cci_val

        if self.config.use_fisher:
            fisher, fisher_val = self.compute_fisher(high, low, close, lookback)
            results["fisher"] = fisher
            results["fisher_val"] = fisher_val

        if self.config.use_dmi:
            dmi, dmi_val = self.compute_dmi(high, low, close, lookback)
            results["dmi"] = dmi
            results["dmi_val"] = dmi_val

        if self.config.use_zscore:
            zsc_val = self.compute_zscore(close)
            results["zsc_val"] = zsc_val

        if self.config.use_mar:
            mar, mar_val = self.compute_mar(close, lookback)
            results["mar"] = mar
            results["mar_val"] = mar_val

        return results


def compute_features(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    config: Optional[FeatureConfig] = None,
    lookback: int = 1000,
) -> dict[str, pd.Series]:
    """Convenience function to compute all features."""
    calculator = FeatureCalculator(config)
    return calculator.compute_all(high, low, close, lookback)


__all__ = ["FeatureConfig", "FeatureCalculator", "compute_features"]

