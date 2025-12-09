"""
Trading strategies based on Simplified Percentile Clustering.

This module provides various trading strategies that utilize cluster
assignments and transitions to generate trading signals.
"""

from modules.simplified_percentile_clustering.strategies.cluster_transition import (
    generate_signals_cluster_transition,
    ClusterTransitionConfig,
)
from modules.simplified_percentile_clustering.strategies.regime_following import (
    generate_signals_regime_following,
    RegimeFollowingConfig,
)
from modules.simplified_percentile_clustering.strategies.mean_reversion import (
    generate_signals_mean_reversion,
    MeanReversionConfig,
)

__all__ = [
    "generate_signals_cluster_transition",
    "ClusterTransitionConfig",
    "generate_signals_regime_following",
    "RegimeFollowingConfig",
    "generate_signals_mean_reversion",
    "MeanReversionConfig",
]

