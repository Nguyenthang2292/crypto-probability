"""
Configuration for Regime Following Strategy.

This strategy follows the current market regime (cluster) and generates
signals when the market is strongly in a particular regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from modules.simplified_percentile_clustering.core.clustering import ClusteringConfig
from modules.simplified_percentile_clustering.utils.validation import (
    validate_clustering_config,
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
        """Set default cluster preferences if not provided and validate config."""
        if self.bullish_clusters is None:
            self.bullish_clusters = [1, 2]
        if self.bearish_clusters is None:
            self.bearish_clusters = [0]
        # Validate configuration
        if not (0.0 <= self.min_regime_strength <= 1.0):
            raise ValueError(
                f"min_regime_strength must be in [0.0, 1.0], got {self.min_regime_strength}"
            )
        if self.min_cluster_duration < 1:
            raise ValueError(
                f"min_cluster_duration must be at least 1, got {self.min_cluster_duration}"
            )
        if self.momentum_period < 1:
            raise ValueError(
                f"momentum_period must be at least 1, got {self.momentum_period}"
            )
        if not (0.0 <= self.bullish_real_clust_threshold <= 1.0):
            raise ValueError(
                f"bullish_real_clust_threshold must be in [0.0, 1.0], got {self.bullish_real_clust_threshold}"
            )
        if not (0.0 <= self.bearish_real_clust_threshold <= 1.0):
            raise ValueError(
                f"bearish_real_clust_threshold must be in [0.0, 1.0], got {self.bearish_real_clust_threshold}"
            )
        if self.clustering_config is not None:
            validate_clustering_config(self.clustering_config)


__all__ = ["RegimeFollowingConfig"]

