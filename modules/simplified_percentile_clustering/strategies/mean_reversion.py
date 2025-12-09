"""
Mean Reversion Strategy.

This strategy generates signals when the market is at cluster extremes
and expects mean reversion back to the center cluster.

Strategy Logic:
--------------
1. LONG Signal:
   - Market in k0 cluster (lower extreme)
   - Real_clust near 0 (far from center)
   - Expecting reversion to k1 or k2
   - Price showing signs of reversal

2. SHORT Signal:
   - Market in k2 or k1 cluster (upper extreme)
   - Real_clust near maximum (far from center)
   - Expecting reversion to k0 or k1
   - Price showing signs of reversal

3. NEUTRAL Signal:
   - Market near center cluster
   - No extreme conditions
   - Ambiguous signals
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from modules.simplified_percentile_clustering.core.clustering import (
    ClusteringResult,
    compute_clustering,
    ClusteringConfig,
)


@dataclass
class MeanReversionConfig:
    """Configuration for mean reversion strategy."""

    # Clustering configuration
    clustering_config: Optional[ClusteringConfig] = None

    # Signal generation parameters
    extreme_threshold: float = 0.2  # Real_clust threshold for extreme (0.0-1.0)
    min_extreme_duration: int = 3  # Minimum bars in extreme before signal
    require_reversal_signal: bool = True  # Require price reversal confirmation
    reversal_lookback: int = 3  # Bars to look back for reversal

    # Reversion targets
    bullish_reversion_target: float = 0.5  # Target real_clust for bullish reversion
    bearish_reversion_target: float = 0.5  # Target real_clust for bearish reversion

    # Signal strength parameters
    min_signal_strength: float = 0.4  # Minimum signal strength

    def __post_init__(self):
        """Set default reversion targets based on k."""
        if self.clustering_config:
            k = self.clustering_config.k
            if k == 3:
                self.bullish_reversion_target = 1.0  # Target middle cluster
                self.bearish_reversion_target = 1.0
            else:
                self.bullish_reversion_target = 0.5  # Target middle
                self.bearish_reversion_target = 0.5


def _detect_reversal(
    close: pd.Series, i: int, lookback: int, direction: str
) -> bool:
    """Detect price reversal signal."""
    if i < lookback:
        return False

    recent_prices = close.iloc[i - lookback : i + 1]
    if len(recent_prices) < 2:
        return False

    if direction == "bullish":
        # Check for bullish reversal: recent low followed by higher close
        recent_min_idx = recent_prices.idxmin()
        if recent_min_idx == recent_prices.index[-1]:
            return False
        min_idx_pos = recent_prices.index.get_loc(recent_min_idx)
        if min_idx_pos < len(recent_prices) - 1:
            # Price increased after the low
            return recent_prices.iloc[-1] > recent_prices.iloc[min_idx_pos]
        return False
    else:  # bearish
        # Check for bearish reversal: recent high followed by lower close
        recent_max_idx = recent_prices.idxmax()
        if recent_max_idx == recent_prices.index[-1]:
            return False
        max_idx_pos = recent_prices.index.get_loc(recent_max_idx)
        if max_idx_pos < len(recent_prices) - 1:
            # Price decreased after the high
            return recent_prices.iloc[-1] < recent_prices.iloc[max_idx_pos]
        return False


def generate_signals_mean_reversion(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    clustering_result: Optional[ClusteringResult] = None,
    config: Optional[MeanReversionConfig] = None,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Generate trading signals based on mean reversion.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        clustering_result: Pre-computed clustering result (optional).
        config: Strategy configuration.

    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
        - metadata: DataFrame with additional signal metadata
    """
    if config is None:
        config = MeanReversionConfig()

    # Compute clustering if not provided
    if clustering_result is None:
        clustering_result = compute_clustering(
            high, low, close, config=config.clustering_config
        )

    signals = pd.Series(0, index=close.index, dtype=int)
    signal_strength = pd.Series(0.0, index=close.index, dtype=float)

    # Determine max real_clust value based on k
    k = 2
    if config.clustering_config:
        k = config.clustering_config.k
    max_real_clust = 2.0 if k == 3 else 1.0

    # Track extreme duration
    extreme_duration = pd.Series(0, index=close.index, dtype=int)
    in_extreme = pd.Series(False, index=close.index, dtype=bool)

    for i in range(len(close)):
        real_clust = clustering_result.real_clust.iloc[i]
        cluster_val = clustering_result.cluster_val.iloc[i]

        if pd.isna(real_clust) or pd.isna(cluster_val):
            continue

        # Check if in extreme
        is_lower_extreme = real_clust <= config.extreme_threshold
        is_upper_extreme = real_clust >= (max_real_clust - config.extreme_threshold)

        if is_lower_extreme or is_upper_extreme:
            if i > 0 and in_extreme.iloc[i - 1]:
                extreme_duration.iloc[i] = extreme_duration.iloc[i - 1] + 1
            else:
                extreme_duration.iloc[i] = 1
            in_extreme.iloc[i] = True
        else:
            extreme_duration.iloc[i] = 0
            in_extreme.iloc[i] = False

    # Metadata columns
    metadata = {
        "cluster_val": clustering_result.cluster_val,
        "real_clust": clustering_result.real_clust,
        "extreme_duration": extreme_duration,
        "in_extreme": in_extreme,
        "price_change": close.pct_change(),
    }

    for i in range(len(close)):
        real_clust = metadata["real_clust"].iloc[i]
        cluster_val = metadata["cluster_val"].iloc[i]
        duration = metadata["extreme_duration"].iloc[i]
        is_extreme = metadata["in_extreme"].iloc[i]

        if pd.isna(real_clust) or pd.isna(cluster_val):
            continue

        # Check if in extreme long enough
        if not is_extreme or duration < config.min_extreme_duration:
            continue

        # Bullish reversion signal (from lower extreme)
        is_lower_extreme = real_clust <= config.extreme_threshold
        if is_lower_extreme:
            reversal_confirmed = True
            if config.require_reversal_signal:
                reversal_confirmed = _detect_reversal(
                    close, i, config.reversal_lookback, "bullish"
                )

            if reversal_confirmed:
                # Calculate distance to target
                distance_to_target = abs(real_clust - config.bullish_reversion_target)
                max_distance = max_real_clust
                strength = 1.0 - min(distance_to_target / max_distance, 1.0)

                # Adjust strength based on how extreme (lower = stronger)
                extreme_strength = 1.0 - (real_clust / config.extreme_threshold)
                combined_strength = (strength + extreme_strength) / 2.0

                if combined_strength >= config.min_signal_strength:
                    signals.iloc[i] = 1  # LONG
                    signal_strength.iloc[i] = combined_strength

        # Bearish reversion signal (from upper extreme)
        is_upper_extreme = real_clust >= (max_real_clust - config.extreme_threshold)
        if is_upper_extreme:
            reversal_confirmed = True
            if config.require_reversal_signal:
                reversal_confirmed = _detect_reversal(
                    close, i, config.reversal_lookback, "bearish"
                )

            if reversal_confirmed:
                # Calculate distance to target
                distance_to_target = abs(real_clust - config.bearish_reversion_target)
                max_distance = max_real_clust
                strength = 1.0 - min(distance_to_target / max_distance, 1.0)

                # Adjust strength based on how extreme (higher = stronger)
                extreme_strength = (
                    real_clust - (max_real_clust - config.extreme_threshold)
                ) / config.extreme_threshold
                combined_strength = (strength + extreme_strength) / 2.0

                if combined_strength >= config.min_signal_strength:
                    signals.iloc[i] = -1  # SHORT
                    signal_strength.iloc[i] = combined_strength

    metadata_df = pd.DataFrame(metadata, index=close.index)
    metadata_df["signal"] = signals
    metadata_df["signal_strength"] = signal_strength

    return signals, signal_strength, metadata_df


__all__ = ["MeanReversionConfig", "generate_signals_mean_reversion"]

