"""
Simplified Percentile Clustering Module.

A compact, streaming-friendly clustering heuristic intended for trend analysis.
It computes simple "cluster centers" per feature using percentiles and the
running mean, then assigns each bar to the closest center. Optionally, in
"Clusters" (combined) mode it fuses distances across multiple features to
compute a multi-dimensional proximity / cluster assignment.

Key design decisions:
  - K limited to 2 or 3 for stability and interpretability.
  - Uses percentiles (lower/upper) + running mean to form deterministic centers
    — a light-weight alternative to iterative clustering (k-means) that is
    streaming-friendly and cheap to compute.
  - Allows selective feature fusion (binary on/off) for combined distance.
  - Produces a `real_clust` value that smoothly interpolates between centers
    (useful for visualizing 'proximity-to-flip').

Notes:
  - This is NOT k-means. It's a percentile + mean center heuristic designed to
    favor stability and low compute on live series.
  - Good for feature engineering and visual regime detection.
"""

from modules.simplified_percentile_clustering.core.clustering import (
    SimplifiedPercentileClustering,
    compute_clustering,
)

__all__ = [
    "SimplifiedPercentileClustering",
    "compute_clustering",
]

