"""
Cluster center calculation using percentiles and running mean.

Builds sliding arrays of historical values for each feature, sorts them,
computes lower/upper percentiles and a center near the mean. Then derives
k centers as:
  k=2 -> [avg(low_pct, mean), avg(high_pct, mean)]
  k=3 -> [avg(low_pct, mean), mean, avg(high_pct, mean)]
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import pandas as pd


class ClusterCenters:
    """Manages cluster centers calculation for a single feature."""

    def __init__(self, lookback: int, p_low: float = 5.0, p_high: float = 95.0, k: int = 2):
        """
        Initialize cluster centers calculator.

        Args:
            lookback: Number of historical values to keep for percentile calculations.
            p_low: Lower percentile (default: 5.0).
            p_high: Upper percentile (default: 95.0).
            k: Number of cluster centers (2 or 3).
        """
        if k not in [2, 3]:
            raise ValueError("k must be 2 or 3")
        if not (0 <= p_low < p_high <= 100):
            raise ValueError("p_low must be < p_high and both in [0, 100]")

        self.lookback = lookback
        self.p_low = p_low
        self.p_high = p_high
        self.k = k
        self._values = deque(maxlen=lookback)

    @staticmethod
    def get_percentile(arr: list[float], pct: float) -> float:
        """
        Returns the value at `pct` percentile from a sorted array.

        Implementation: sorts a copy and chooses index floor((n-1)*pct/100).
        This is a deterministic and inexpensive approximation.
        """
        if len(arr) == 0:
            return np.nan

        sorted_arr = sorted(arr)
        idx = int(np.floor((len(sorted_arr) - 1) * pct / 100))
        return sorted_arr[idx]

    def update(self, value: float) -> list[float]:
        """
        Update with new value and return current cluster centers.

        Args:
            value: New feature value to add.

        Returns:
            List of k cluster centers.
        """
        if not np.isfinite(value):
            # If we have previous values, return last centers
            if len(self._values) > 0:
                return self._get_centers()
            return [np.nan] * self.k

        self._values.append(float(value))
        return self._get_centers()

    def _get_centers(self) -> list[float]:
        """Calculate centers from current values."""
        if len(self._values) == 0:
            return [np.nan] * self.k

        values_list = list(self._values)

        # Calculate percentiles and mean
        x_high = self.get_percentile(values_list, self.p_high)
        x_low = self.get_percentile(values_list, self.p_low)
        x_mid = np.mean(values_list)

        # Calculate centers
        x_k0_center = (x_low + x_mid) / 2
        x_k1_center = (x_high + x_mid) / 2

        if self.k == 2:
            return [x_k0_center, x_k1_center]
        else:  # k == 3
            return [x_k0_center, x_mid, x_k1_center]

    def get_current_centers(self) -> list[float]:
        """Get current cluster centers without updating."""
        return self._get_centers()


def compute_centers(
    values: pd.Series,
    lookback: int = 1000,
    p_low: float = 5.0,
    p_high: float = 95.0,
    k: int = 2,
) -> pd.DataFrame:
    """
    Compute cluster centers for a time series using vectorized operations.

    This function uses Pandas rolling window operations for better performance
    compared to the iterative approach. It computes percentiles and mean using
    vectorized operations.

    Args:
        values: Feature values time series.
        lookback: Number of historical values to keep.
        p_low: Lower percentile (0-100).
        p_high: Upper percentile (0-100).
        k: Number of cluster centers (2 or 3).

    Returns:
        DataFrame with columns 'k0', 'k1', and optionally 'k2' containing
        cluster centers for each timestamp.
    """
    if len(values) == 0:
        if k == 2:
            return pd.DataFrame(columns=["k0", "k1"], index=values.index)
        else:
            return pd.DataFrame(columns=["k0", "k1", "k2"], index=values.index)

    # Convert percentiles to quantiles (0-1 range)
    q_low = p_low / 100.0
    q_high = p_high / 100.0

    # Vectorized calculation using rolling windows
    # Use min_periods=1 to handle initial values with insufficient data
    rolling = values.rolling(window=lookback, min_periods=1)

    # Calculate percentiles and mean using vectorized operations
    # Use "lower" interpolation to match the original floor-based percentile calculation
    x_low = rolling.quantile(q_low, interpolation="lower")
    x_high = rolling.quantile(q_high, interpolation="lower")
    x_mid = rolling.mean()

    # Calculate centers: k0 = (p_low + mean)/2, k1 = (p_high + mean)/2
    x_k0_center = (x_low + x_mid) / 2.0
    x_k1_center = (x_high + x_mid) / 2.0

    # Build result DataFrame
    if k == 2:
        result = pd.DataFrame(
            {
                "k0": x_k0_center,
                "k1": x_k1_center,
            },
            index=values.index,
        )
    else:  # k == 3
        result = pd.DataFrame(
            {
                "k0": x_k0_center,
                "k1": x_mid,  # For k=3, k1 is the mean
                "k2": x_k1_center,
            },
            index=values.index,
        )

    return result


__all__ = ["ClusterCenters", "compute_centers"]

