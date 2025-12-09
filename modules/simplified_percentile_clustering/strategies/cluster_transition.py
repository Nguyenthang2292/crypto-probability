"""
Cluster Transition Strategy.

This strategy generates trading signals based on cluster transitions.
When the market transitions from one cluster to another, it may indicate
a regime change and potential trading opportunity.

Strategy Logic:
--------------
1. LONG Signal:
   - Transition from k0 (lower cluster) to k1 or k2 (higher clusters)
   - Real_clust value increasing and crossing cluster boundaries
   - Confirmation: price moving in same direction

2. SHORT Signal:
   - Transition from k2 or k1 (higher clusters) to k0 (lower cluster)
   - Real_clust value decreasing and crossing cluster boundaries
   - Confirmation: price moving in same direction

3. NEUTRAL Signal:
   - No cluster transition
   - Real_clust staying within same cluster
   - Ambiguous transitions (rel_pos near 0.5)
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
    FeatureConfig,
)


@dataclass
class ClusterTransitionConfig:
    """Configuration for cluster transition strategy."""

    # Clustering configuration
    clustering_config: Optional[ClusteringConfig] = None

    # Signal generation parameters
    require_price_confirmation: bool = True  # Require price to move in same direction
    min_rel_pos_change: float = 0.1  # Minimum relative position change for signal
    use_real_clust_cross: bool = True  # Use real_clust crossing cluster boundaries
    min_signal_strength: float = 0.3  # Minimum signal strength (0.0 to 1.0)

    # Cluster transition rules
    bullish_transitions: list[tuple[int, int]] = None  # e.g., [(0, 1), (0, 2), (1, 2)]
    bearish_transitions: list[tuple[int, int]] = None  # e.g., [(2, 1), (2, 0), (1, 0)]

    def __post_init__(self):
        """Set default transition rules if not provided."""
        if self.bullish_transitions is None:
            # Default: transitions to higher clusters are bullish
            self.bullish_transitions = [(0, 1), (0, 2), (1, 2)]
        if self.bearish_transitions is None:
            # Default: transitions to lower clusters are bearish
            self.bearish_transitions = [(2, 1), (2, 0), (1, 0)]


def generate_signals_cluster_transition(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    clustering_result: Optional[ClusteringResult] = None,
    config: Optional[ClusterTransitionConfig] = None,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Generate trading signals based on cluster transitions.

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
        config = ClusterTransitionConfig()

    # Compute clustering if not provided
    if clustering_result is None:
        clustering_result = compute_clustering(
            high, low, close, config=config.clustering_config
        )

    signals = pd.Series(0, index=close.index, dtype=int)
    signal_strength = pd.Series(0.0, index=close.index, dtype=float)

    # Metadata columns
    metadata = {
        "cluster_val": clustering_result.cluster_val,
        "prev_cluster_val": clustering_result.cluster_val.shift(1),
        "real_clust": clustering_result.real_clust,
        "prev_real_clust": clustering_result.real_clust.shift(1),
        "rel_pos": clustering_result.rel_pos,
        "price_change": close.pct_change(),
    }

    for i in range(1, len(close)):
        prev_cluster = metadata["prev_cluster_val"].iloc[i]
        curr_cluster = metadata["cluster_val"].iloc[i]
        prev_real = metadata["prev_real_clust"].iloc[i]
        curr_real = metadata["real_clust"].iloc[i]
        rel_pos = metadata["rel_pos"].iloc[i]
        price_change = metadata["price_change"].iloc[i]

        # Skip if missing data
        if (
            pd.isna(prev_cluster)
            or pd.isna(curr_cluster)
            or pd.isna(prev_real)
            or pd.isna(curr_real)
        ):
            continue

        prev_cluster_int = int(prev_cluster)
        curr_cluster_int = int(curr_cluster)

        # Check for cluster transition
        transition = (prev_cluster_int, curr_cluster_int)

        # Calculate signal strength based on real_clust movement
        real_clust_change = abs(curr_real - prev_real)
        max_possible_change = (
            2.0 if config.clustering_config and config.clustering_config.k == 3 else 1.0
        )
        strength_from_movement = min(real_clust_change / max_possible_change, 1.0)

        # Combine with rel_pos (lower rel_pos = stronger signal)
        rel_pos_strength = 1.0 - min(rel_pos, 1.0)
        combined_strength = (strength_from_movement + rel_pos_strength) / 2.0

        # Check for bullish transition
        if transition in config.bullish_transitions:
            # Price confirmation
            price_confirmed = True
            if config.require_price_confirmation:
                price_confirmed = price_change > 0 or pd.isna(price_change)

            if price_confirmed and combined_strength >= config.min_signal_strength:
                signals.iloc[i] = 1  # LONG
                signal_strength.iloc[i] = combined_strength

        # Check for bearish transition
        elif transition in config.bearish_transitions:
            # Price confirmation
            price_confirmed = True
            if config.require_price_confirmation:
                price_confirmed = price_change < 0 or pd.isna(price_change)

            if price_confirmed and combined_strength >= config.min_signal_strength:
                signals.iloc[i] = -1  # SHORT
                signal_strength.iloc[i] = combined_strength

        # Check for real_clust crossing cluster boundaries (if enabled)
        if config.use_real_clust_cross:
            # Crossing from below k0.5 to above (bullish)
            if prev_real < 0.5 and curr_real >= 0.5:
                if config.clustering_config and config.clustering_config.k == 3:
                    if prev_real < 1.5 and curr_real >= 1.5:
                        # Crossing to k2
                        if combined_strength >= config.min_signal_strength:
                            if signals.iloc[i] == 0:  # Only if no transition signal
                                signals.iloc[i] = 1
                                signal_strength.iloc[i] = combined_strength * 0.8
                else:
                    # Crossing to k1
                    if combined_strength >= config.min_signal_strength:
                        if signals.iloc[i] == 0:  # Only if no transition signal
                            signals.iloc[i] = 1
                            signal_strength.iloc[i] = combined_strength * 0.8

            # Crossing from above k0.5 to below (bearish)
            elif prev_real > 0.5 and curr_real <= 0.5:
                if config.clustering_config and config.clustering_config.k == 3:
                    if prev_real > 1.5 and curr_real <= 1.5:
                        # Crossing from k2
                        if combined_strength >= config.min_signal_strength:
                            if signals.iloc[i] == 0:  # Only if no transition signal
                                signals.iloc[i] = -1
                                signal_strength.iloc[i] = combined_strength * 0.8
                else:
                    # Crossing to k0
                    if combined_strength >= config.min_signal_strength:
                        if signals.iloc[i] == 0:  # Only if no transition signal
                            signals.iloc[i] = -1
                            signal_strength.iloc[i] = combined_strength * 0.8

    metadata_df = pd.DataFrame(metadata, index=close.index)
    metadata_df["signal"] = signals
    metadata_df["signal_strength"] = signal_strength

    return signals, signal_strength, metadata_df


__all__ = ["ClusterTransitionConfig", "generate_signals_cluster_transition"]

