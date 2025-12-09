"""
Regime Following Strategy.

This strategy follows the current market regime (cluster) and generates
signals when the market is strongly in a particular regime.

Strategy Logic:
--------------
1. LONG Signal:
   - Market in k1 or k2 cluster (higher clusters)
   - Real_clust value consistently high
   - Strong regime (low rel_pos, meaning close to cluster center)
   - Price momentum confirming

2. SHORT Signal:
   - Market in k0 cluster (lower cluster)
   - Real_clust value consistently low
   - Strong regime (low rel_pos)
   - Price momentum confirming

3. NEUTRAL Signal:
   - Weak regime (high rel_pos, ambiguous)
   - Transitioning between clusters
   - Conflicting signals
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
class RegimeFollowingConfig:
    """Configuration for regime following strategy."""

    # Clustering configuration
    clustering_config: Optional[ClusteringConfig] = None

    # Signal generation parameters
    min_regime_strength: float = 0.7  # Minimum regime strength (1 - rel_pos)
    min_cluster_duration: int = 2  # Minimum bars in same cluster before signal
    require_momentum: bool = True  # Require price momentum confirmation
    momentum_period: int = 5  # Period for momentum calculation

    # Cluster preferences
    bullish_clusters: list[int] = None  # Clusters considered bullish (e.g., [1, 2])
    bearish_clusters: list[int] = None  # Clusters considered bearish (e.g., [0])

    # Real_clust thresholds
    bullish_real_clust_threshold: float = 0.5  # Minimum real_clust for bullish
    bearish_real_clust_threshold: float = 0.5  # Maximum real_clust for bearish

    def __post_init__(self):
        """Set default cluster preferences if not provided."""
        if self.bullish_clusters is None:
            self.bullish_clusters = [1, 2]
        if self.bearish_clusters is None:
            self.bearish_clusters = [0]


def generate_signals_regime_following(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    clustering_result: Optional[ClusteringResult] = None,
    config: Optional[RegimeFollowingConfig] = None,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Generate trading signals based on regime following.

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
        config = RegimeFollowingConfig()

    # Compute clustering if not provided
    if clustering_result is None:
        clustering_result = compute_clustering(
            high, low, close, config=config.clustering_config
        )

    signals = pd.Series(0, index=close.index, dtype=int)
    signal_strength = pd.Series(0.0, index=close.index, dtype=float)

    # Calculate momentum
    momentum = close.pct_change(periods=config.momentum_period)

    # Track cluster duration
    cluster_duration = pd.Series(0, index=close.index, dtype=int)
    prev_cluster = None
    for i in range(len(close)):
        curr_cluster = clustering_result.cluster_val.iloc[i]
        if pd.isna(curr_cluster):
            cluster_duration.iloc[i] = 0
            prev_cluster = None
        elif prev_cluster is not None and curr_cluster == prev_cluster:
            cluster_duration.iloc[i] = cluster_duration.iloc[i - 1] + 1
        else:
            cluster_duration.iloc[i] = 1
            prev_cluster = curr_cluster

    # Calculate regime strength (1 - rel_pos, higher = stronger)
    regime_strength = 1.0 - clustering_result.rel_pos.fillna(0.5)

    # Metadata columns
    metadata = {
        "cluster_val": clustering_result.cluster_val,
        "real_clust": clustering_result.real_clust,
        "regime_strength": regime_strength,
        "cluster_duration": cluster_duration,
        "momentum": momentum,
        "price_change": close.pct_change(),
    }

    for i in range(len(close)):
        cluster_val = metadata["cluster_val"].iloc[i]
        real_clust = metadata["real_clust"].iloc[i]
        strength = metadata["regime_strength"].iloc[i]
        duration = metadata["cluster_duration"].iloc[i]
        mom = metadata["momentum"].iloc[i]

        # Skip if missing data
        if pd.isna(cluster_val) or pd.isna(real_clust) or pd.isna(strength):
            continue

        cluster_int = int(cluster_val)

        # Check regime strength
        if strength < config.min_regime_strength:
            continue  # Weak regime, no signal

        # Check cluster duration
        if duration < config.min_cluster_duration:
            continue  # Not enough time in cluster

        # Check momentum if required
        momentum_confirmed = True
        if config.require_momentum:
            if pd.isna(mom):
                momentum_confirmed = False
            else:
                momentum_confirmed = True  # Will check direction below

        # Bullish signal
        if cluster_int in config.bullish_clusters:
            if real_clust >= config.bullish_real_clust_threshold:
                if not config.require_momentum or (momentum_confirmed and mom >= 0):
                    signals.iloc[i] = 1  # LONG
                    # Signal strength based on regime strength and real_clust position
                    if config.clustering_config and config.clustering_config.k == 3:
                        # Normalize real_clust to [0, 1] for k=3
                        normalized_real = (real_clust - 0) / 2.0
                    else:
                        # Normalize real_clust to [0, 1] for k=2
                        normalized_real = (real_clust - 0) / 1.0
                    signal_strength.iloc[i] = (strength + normalized_real) / 2.0

        # Bearish signal
        elif cluster_int in config.bearish_clusters:
            if real_clust <= config.bearish_real_clust_threshold:
                if not config.require_momentum or (momentum_confirmed and mom <= 0):
                    signals.iloc[i] = -1  # SHORT
                    # Signal strength based on regime strength and real_clust position
                    if config.clustering_config and config.clustering_config.k == 3:
                        # Normalize real_clust to [0, 1] for k=3 (inverted)
                        normalized_real = 1.0 - (real_clust - 0) / 2.0
                    else:
                        # Normalize real_clust to [0, 1] for k=2 (inverted)
                        normalized_real = 1.0 - (real_clust - 0) / 1.0
                    signal_strength.iloc[i] = (strength + normalized_real) / 2.0

    metadata_df = pd.DataFrame(metadata, index=close.index)
    metadata_df["signal"] = signals
    metadata_df["signal_strength"] = signal_strength

    return signals, signal_strength, metadata_df


__all__ = ["RegimeFollowingConfig", "generate_signals_regime_following"]

