"""
Main clustering calculation for Simplified Percentile Clustering.

Combines feature calculations, center computation, and cluster assignment
to produce cluster assignments and interpolated cluster values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from modules.simplified_percentile_clustering.core.centers import (
    ClusterCenters,
    compute_centers,
)
from modules.simplified_percentile_clustering.core.features import (
    FeatureCalculator,
    FeatureConfig,
)


@dataclass
class ClusteringConfig:
    """Configuration for clustering calculation."""

    # Clustering parameters
    k: int = 2  # Number of clusters (2 or 3)
    lookback: int = 1000  # Historical bars for percentile/mean calculations
    p_low: float = 5.0  # Lower percentile
    p_high: float = 95.0  # Upper percentile

    # Feature configuration
    feature_config: Optional[FeatureConfig] = None

    # Main plot mode
    main_plot: str = "Clusters"  # "Clusters", "RSI", "CCI", "Fisher", "DMI", "Z-Score", "MAR"


@dataclass
class ClusteringResult:
    """Result of clustering calculation."""

    # Cluster assignment
    cluster_val: pd.Series  # Discrete cluster index (0, 1, or 2)
    curr_cluster: pd.Series  # Cluster name ("k0", "k1", "k2")
    real_clust: pd.Series  # Interpolated cluster value (continuous)

    # Distances
    min_dist: pd.Series  # Distance to closest center
    second_min_dist: pd.Series  # Distance to second closest center
    rel_pos: pd.Series  # Relative position between closest and second closest

    # Plot values
    plot_val: pd.Series  # Value to plot (feature value or real_clust)
    plot_k0_center: pd.Series  # k0 cluster center
    plot_k1_center: pd.Series  # k1 cluster center
    plot_k2_center: pd.Series  # k2 cluster center (if k=3)

    # Feature values (for reference)
    features: dict[str, pd.Series]


class SimplifiedPercentileClustering:
    """Main clustering calculator."""

    def __init__(self, config: Optional[ClusteringConfig] = None):
        self.config = config or ClusteringConfig()
        if self.config.feature_config is None:
            self.config.feature_config = FeatureConfig()

        self.feature_calc = FeatureCalculator(self.config.feature_config)
        self._centers_calculators: dict[str, ClusterCenters] = {}

    def _get_centers_calculator(self, feature_name: str) -> ClusterCenters:
        """Get or create centers calculator for a feature."""
        if feature_name not in self._centers_calculators:
            self._centers_calculators[feature_name] = ClusterCenters(
                lookback=self.config.lookback,
                p_low=self.config.p_low,
                p_high=self.config.p_high,
                k=self.config.k,
            )
        return self._centers_calculators[feature_name]

    def _compute_all_centers(
        self, features: dict[str, pd.Series]
    ) -> dict[str, pd.DataFrame]:
        """
        Compute cluster centers for all features using vectorized operations.
        
        Uses the vectorized compute_centers() function instead of iterative updates.
        """
        centers_dict = {}

        for feature_name, values in features.items():
            if feature_name.endswith("_val") or feature_name in ["zsc_val"]:
                # Use vectorized compute_centers instead of loop
                centers_df = compute_centers(
                    values,
                    lookback=self.config.lookback,
                    p_low=self.config.p_low,
                    p_high=self.config.p_high,
                    k=self.config.k,
                )
                centers_dict[feature_name] = centers_df

        return centers_dict

    def _compute_distance_single(
        self, feature_val: pd.Series, centers: pd.DataFrame
    ) -> pd.Series:
        """Compute distance for single-feature mode."""
        distances = []
        for i in range(len(feature_val)):
            if pd.isna(feature_val.iloc[i]):
                distances.append(np.nan)
                continue

            min_dist = np.inf
            for col in centers.columns:
                center_val = centers[col].iloc[i]
                if pd.isna(center_val):
                    continue
                dist = abs(feature_val.iloc[i] - center_val)
                if dist < min_dist:
                    min_dist = dist

            distances.append(min_dist if min_dist != np.inf else np.nan)

        return pd.Series(distances, index=feature_val.index)

    def _compute_distance_combined(
        self,
        features: dict[str, pd.Series],
        centers_dict: dict[str, pd.DataFrame],
        center_idx: int,
    ) -> pd.Series:
        """
        Compute combined distance across all enabled features using vectorized operations.
        
        Uses broadcasting to compute distances for all timestamps at once.
        """
        config = self.config.feature_config
        center_col = f"k{center_idx}"
        
        # Collect all feature distances using vectorized operations
        feature_distances = []
        
        # RSI
        if config.use_rsi and "rsi_val" in features:
            rsi_centers = centers_dict.get("rsi_val")
            if rsi_centers is not None and center_col in rsi_centers.columns:
                dist = (features["rsi_val"] - rsi_centers[center_col]).abs()
                feature_distances.append(dist)
        
        # CCI
        if config.use_cci and "cci_val" in features:
            cci_centers = centers_dict.get("cci_val")
            if cci_centers is not None and center_col in cci_centers.columns:
                dist = (features["cci_val"] - cci_centers[center_col]).abs()
                feature_distances.append(dist)
        
        # Fisher
        if config.use_fisher and "fisher_val" in features:
            fis_centers = centers_dict.get("fisher_val")
            if fis_centers is not None and center_col in fis_centers.columns:
                dist = (features["fisher_val"] - fis_centers[center_col]).abs()
                feature_distances.append(dist)
        
        # DMI
        if config.use_dmi and "dmi_val" in features:
            dmi_centers = centers_dict.get("dmi_val")
            if dmi_centers is not None and center_col in dmi_centers.columns:
                dist = (features["dmi_val"] - dmi_centers[center_col]).abs()
                feature_distances.append(dist)
        
        # Z-Score
        if config.use_zscore and "zsc_val" in features:
            zsc_centers = centers_dict.get("zsc_val")
            if zsc_centers is not None and center_col in zsc_centers.columns:
                dist = (features["zsc_val"] - zsc_centers[center_col]).abs()
                feature_distances.append(dist)
        
        # MAR
        if config.use_mar and "mar_val" in features:
            mar_centers = centers_dict.get("mar_val")
            if mar_centers is not None and center_col in mar_centers.columns:
                dist = (features["mar_val"] - mar_centers[center_col]).abs()
                feature_distances.append(dist)
        
        # Compute weighted average across all features
        if len(feature_distances) > 0:
            # Stack all distances into a DataFrame
            dist_df = pd.DataFrame(feature_distances).T
            # Average across columns (features), handling NaN
            combined_dist = dist_df.mean(axis=1, skipna=True)
            return combined_dist
        else:
            # Return NaN series if no features enabled
            index = next(iter(features.values())).index if features else pd.Index([])
            return pd.Series(np.nan, index=index)

    def compute(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> ClusteringResult:
        """
        Compute clustering for OHLCV data.

        Args:
            high: High price series.
            low: Low price series.
            close: Close price series.

        Returns:
            ClusteringResult with all computed values.
        """
        # Step 1: Compute all features
        features = self.feature_calc.compute_all(
            high, low, close, self.config.lookback
        )

        # Step 2: Compute centers for all features
        centers_dict = self._compute_all_centers(features)

        # Step 3: Determine which feature/centers to use based on main_plot
        main_plot = self.config.main_plot
        index = close.index
        n = len(close)

        # Step 4: Compute distances to all centers using vectorized operations
        # Build a matrix of distances: rows = timestamps, columns = centers (k0, k1, k2)
        distances_df = pd.DataFrame(index=index)
        
        # Map main_plot to feature key
        feature_map = {
            "RSI": "rsi_val",
            "CCI": "cci_val",
            "Fisher": "fisher_val",
            "DMI": "dmi_val",
            "Z-Score": "zsc_val",
            "MAR": "mar_val",
        }
        
        # Compute distances for each center using vectorized operations
        for center_idx in range(self.config.k):
            center_col = f"k{center_idx}"
            
            if main_plot in feature_map:
                # Single feature mode - use broadcasting
                feature_key = feature_map[main_plot]
                if feature_key in features and feature_key in centers_dict:
                    feature_vals = features[feature_key]
                    centers = centers_dict[feature_key]
                    if center_col in centers.columns:
                        # Vectorized distance calculation: abs(feature_series - center_series)
                        dist = (feature_vals - centers[center_col]).abs()
                        distances_df[center_col] = dist
                    else:
                        distances_df[center_col] = np.nan
                else:
                    distances_df[center_col] = np.nan
            else:
                # Combined mode - use vectorized combined distance
                dist_series = self._compute_distance_combined(
                    features, centers_dict, center_idx
                )
                distances_df[center_col] = dist_series

        # Replace inf with NaN for easier handling
        distances_df = distances_df.replace([np.inf, -np.inf], np.nan)
        
        # Step 5: Find min and second min distances using vectorized operations
        # Convert to numpy array for efficient operations
        dist_array = distances_df.values
        
        # Find indices of min and second min for each row
        # Use np.partition for efficient partial sort
        valid_mask = ~np.isnan(dist_array)
        
        # Initialize result arrays
        min_dist = np.full(n, np.nan)
        second_min_dist = np.full(n, np.nan)
        cluster_val = np.full(n, np.nan, dtype=float)
        curr_cluster = np.full(n, None, dtype=object)
        second_cluster = np.full(n, None, dtype=object)
        
        for i in range(n):
            row = dist_array[i, :]
            valid = valid_mask[i, :]
            
            if not np.any(valid):
                continue
            
            valid_distances = row[valid]
            valid_indices = np.where(valid)[0]
            
            if len(valid_distances) == 0:
                continue
            
            # Find min and second min
            if len(valid_distances) == 1:
                min_idx = valid_indices[0]
                min_dist[i] = valid_distances[0]
                cluster_val[i] = min_idx
                curr_cluster[i] = f"k{min_idx}"
                second_min_dist[i] = np.nan
            else:
                # Use argpartition for efficient partial sort
                sorted_indices = np.argsort(valid_distances)
                min_idx = valid_indices[sorted_indices[0]]
                second_min_idx = valid_indices[sorted_indices[1]]
                
                min_dist[i] = valid_distances[sorted_indices[0]]
                second_min_dist[i] = valid_distances[sorted_indices[1]]
                cluster_val[i] = min_idx
                curr_cluster[i] = f"k{min_idx}"
                second_cluster[i] = f"k{second_min_idx}"
        
        # Convert to Series
        min_dist = pd.Series(min_dist, index=index)
        second_min_dist = pd.Series(second_min_dist, index=index)
        cluster_val = pd.Series(cluster_val, index=index)
        
        # Convert curr_cluster to string Series, handling None values
        curr_cluster = pd.Series(
            [str(c) if c is not None else None for c in curr_cluster],
            index=index,
            dtype=object
        )
        second_cluster = pd.Series(
            [str(c) if c is not None else None for c in second_cluster],
            index=index,
            dtype=object
        )
        
        # Step 6: Compute relative position and real_clust using vectorized operations
        # Relative position: min_dist / (min_dist + second_min_dist)
        rel_pos = pd.Series(0.0, index=index)
        valid_rel = (second_min_dist > 0) & (second_min_dist != np.inf) & (~second_min_dist.isna())
        rel_pos[valid_rel] = min_dist[valid_rel] / (min_dist[valid_rel] + second_min_dist[valid_rel])
        
        # Second cluster value - convert cluster names to numeric values
        second_val = pd.Series(cluster_val.values, index=index)
        second_val_mask = second_cluster.notna()
        second_val[second_cluster == "k0"] = 0.0
        second_val[second_cluster == "k1"] = 1.0
        second_val[second_cluster == "k2"] = 2.0
        # If second_cluster is None/NaN, use cluster_val
        second_val[~second_val_mask] = cluster_val[~second_val_mask]
        
        # Real cluster (interpolated): cluster_val + (second_val - cluster_val) * rel_pos
        real_clust = cluster_val + (second_val - cluster_val) * rel_pos

        # Compute plot values
        if main_plot == "Clusters":
            plot_val = real_clust
        elif main_plot == "RSI" and "rsi_val" in features:
            plot_val = features["rsi_val"]
        elif main_plot == "CCI" and "cci_val" in features:
            plot_val = features["cci_val"]
        elif main_plot == "Fisher" and "fisher_val" in features:
            plot_val = features["fisher_val"]
        elif main_plot == "DMI" and "dmi_val" in features:
            plot_val = features["dmi_val"]
        elif main_plot == "Z-Score" and "zsc_val" in features:
            plot_val = features["zsc_val"]
        elif main_plot == "MAR" and "mar_val" in features:
            plot_val = features["mar_val"]
        else:
            plot_val = real_clust

        # Compute plot centers
        if main_plot == "Clusters":
            plot_k0_center = pd.Series(0.0, index=index)
            plot_k1_center = pd.Series(1.0, index=index)
            plot_k2_center = (
                pd.Series(2.0, index=index) if self.config.k == 3 else pd.Series(0.0, index=index)
            )
        else:
            # Use centers from the selected feature
            feature_key = {
                "RSI": "rsi_val",
                "CCI": "cci_val",
                "Fisher": "fisher_val",
                "DMI": "dmi_val",
                "Z-Score": "zsc_val",
                "MAR": "mar_val",
            }.get(main_plot)

            if feature_key and feature_key in centers_dict:
                centers = centers_dict[feature_key]
                plot_k0_center = centers["k0"]
                plot_k1_center = centers["k1"]
                plot_k2_center = (
                    centers["k2"] if self.config.k == 3 else pd.Series(0.0, index=index)
                )
            else:
                plot_k0_center = pd.Series(0.0, index=index)
                plot_k1_center = pd.Series(0.0, index=index)
                plot_k2_center = pd.Series(0.0, index=index)

        return ClusteringResult(
            cluster_val=cluster_val,
            curr_cluster=curr_cluster,
            real_clust=real_clust,
            min_dist=min_dist,
            second_min_dist=second_min_dist,
            rel_pos=rel_pos,
            plot_val=plot_val,
            plot_k0_center=plot_k0_center,
            plot_k1_center=plot_k1_center,
            plot_k2_center=plot_k2_center,
            features=features,
        )


def compute_clustering(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    config: Optional[ClusteringConfig] = None,
) -> ClusteringResult:
    """Convenience function to compute clustering."""
    clustering = SimplifiedPercentileClustering(config)
    return clustering.compute(high, low, close)


__all__ = [
    "ClusteringConfig",
    "ClusteringResult",
    "SimplifiedPercentileClustering",
    "compute_clustering",
]

