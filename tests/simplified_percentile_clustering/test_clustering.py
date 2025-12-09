"""
Tests for clustering module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.simplified_percentile_clustering.core.clustering import (
    SimplifiedPercentileClustering,
    ClusteringConfig,
    ClusteringResult,
    compute_clustering,
)
from modules.simplified_percentile_clustering.core.features import FeatureConfig


def _sample_ohlcv_data(length=200):
    """Generate sample OHLCV data."""
    np.random.seed(42)
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(length) * 0.5)
    
    high = prices + np.abs(np.random.randn(length) * 0.3)
    low = prices - np.abs(np.random.randn(length) * 0.3)
    close = prices
    
    return pd.Series(high), pd.Series(low), pd.Series(close)


def test_clustering_config_defaults():
    """Test ClusteringConfig default values."""
    config = ClusteringConfig()
    
    assert config.k == 2
    assert config.lookback == 1000
    assert config.p_low == 5.0
    assert config.p_high == 95.0
    assert config.main_plot == "Clusters"


def test_clustering_config_with_feature_config():
    """Test ClusteringConfig with FeatureConfig."""
    feature_config = FeatureConfig(use_rsi=True, use_cci=False)
    config = ClusteringConfig(feature_config=feature_config)
    
    assert config.feature_config == feature_config


def test_simplified_percentile_clustering_init():
    """Test SimplifiedPercentileClustering initialization."""
    config = ClusteringConfig(k=2, lookback=100)
    clustering = SimplifiedPercentileClustering(config)
    
    assert clustering.config == config
    assert clustering.feature_calc is not None


def test_simplified_percentile_clustering_compute_basic():
    """Test basic clustering computation."""
    high, low, close = _sample_ohlcv_data(200)
    config = ClusteringConfig(
        k=2,
        lookback=100,
        feature_config=FeatureConfig(
            use_rsi=True,
            use_cci=True,
            use_fisher=False,
            use_dmi=False,
            use_zscore=True,
            use_mar=False,
        ),
    )
    clustering = SimplifiedPercentileClustering(config)
    
    result = clustering.compute(high, low, close)
    
    assert isinstance(result, ClusteringResult)
    assert len(result.cluster_val) == len(close)
    assert len(result.real_clust) == len(close)
    assert len(result.plot_val) == len(close)


def test_simplified_percentile_clustering_compute_all_features():
    """Test clustering with all features enabled."""
    high, low, close = _sample_ohlcv_data(200)
    config = ClusteringConfig(
        k=2,
        lookback=100,
        feature_config=FeatureConfig(
            use_rsi=True,
            use_cci=True,
            use_fisher=True,
            use_dmi=True,
            use_zscore=True,
            use_mar=True,
        ),
    )
    clustering = SimplifiedPercentileClustering(config)
    
    result = clustering.compute(high, low, close)
    
    assert isinstance(result, ClusteringResult)
    assert "rsi_val" in result.features
    assert "cci_val" in result.features
    assert "fisher_val" in result.features


def test_simplified_percentile_clustering_k3():
    """Test clustering with k=3."""
    high, low, close = _sample_ohlcv_data(200)
    config = ClusteringConfig(k=3, lookback=100)
    clustering = SimplifiedPercentileClustering(config)
    
    result = clustering.compute(high, low, close)
    
    assert isinstance(result, ClusteringResult)
    # With k=3, cluster_val can be 0, 1, or 2
    assert result.cluster_val.max() <= 2
    assert result.cluster_val.min() >= 0


def test_simplified_percentile_clustering_main_plot_rsi():
    """Test clustering with main_plot='RSI'."""
    high, low, close = _sample_ohlcv_data(200)
    config = ClusteringConfig(
        k=2,
        lookback=100,
        main_plot="RSI",
        feature_config=FeatureConfig(use_rsi=True),
    )
    clustering = SimplifiedPercentileClustering(config)
    
    result = clustering.compute(high, low, close)
    
    assert isinstance(result, ClusteringResult)
    # plot_val should be RSI values
    assert "rsi_val" in result.features


def test_simplified_percentile_clustering_main_plot_clusters():
    """Test clustering with main_plot='Clusters' (combined mode)."""
    high, low, close = _sample_ohlcv_data(200)
    config = ClusteringConfig(
        k=2,
        lookback=100,
        main_plot="Clusters",
        feature_config=FeatureConfig(
            use_rsi=True,
            use_cci=True,
        ),
    )
    clustering = SimplifiedPercentileClustering(config)
    
    result = clustering.compute(high, low, close)
    
    assert isinstance(result, ClusteringResult)
    # plot_val should be real_clust in combined mode
    pd.testing.assert_series_equal(result.plot_val, result.real_clust)


def test_compute_clustering_convenience():
    """Test compute_clustering convenience function."""
    high, low, close = _sample_ohlcv_data(200)
    config = ClusteringConfig(k=2, lookback=100)
    
    result = compute_clustering(high, low, close, config=config)
    
    assert isinstance(result, ClusteringResult)
    assert len(result.cluster_val) == len(close)


def test_clustering_result_structure():
    """Test that ClusteringResult has all required fields."""
    high, low, close = _sample_ohlcv_data(200)
    config = ClusteringConfig(k=2, lookback=100)
    clustering = SimplifiedPercentileClustering(config)
    
    result = clustering.compute(high, low, close)
    
    # Check all required fields exist
    assert hasattr(result, "cluster_val")
    assert hasattr(result, "curr_cluster")
    assert hasattr(result, "real_clust")
    assert hasattr(result, "min_dist")
    assert hasattr(result, "second_min_dist")
    assert hasattr(result, "rel_pos")
    assert hasattr(result, "plot_val")
    assert hasattr(result, "plot_k0_center")
    assert hasattr(result, "plot_k1_center")
    assert hasattr(result, "plot_k2_center")
    assert hasattr(result, "features")


def test_clustering_result_cluster_val_range():
    """Test that cluster_val is in valid range."""
    high, low, close = _sample_ohlcv_data(200)
    config = ClusteringConfig(k=2, lookback=100)
    clustering = SimplifiedPercentileClustering(config)
    
    result = clustering.compute(high, low, close)
    
    # cluster_val should be 0, 1, or NaN
    valid_values = result.cluster_val.dropna()
    assert all(valid_values.isin([0, 1]))


def test_clustering_result_real_clust_range():
    """Test that real_clust is in valid range."""
    high, low, close = _sample_ohlcv_data(200)
    config = ClusteringConfig(k=2, lookback=100)
    clustering = SimplifiedPercentileClustering(config)
    
    result = clustering.compute(high, low, close)
    
    # real_clust should be between 0 and k-1 (or k for k=3)
    valid_values = result.real_clust.dropna()
    if config.k == 2:
        assert valid_values.min() >= 0
        assert valid_values.max() <= 1
    else:  # k=3
        assert valid_values.min() >= 0
        assert valid_values.max() <= 2


def test_clustering_result_rel_pos_range():
    """Test that rel_pos is in valid range [0, 1]."""
    high, low, close = _sample_ohlcv_data(200)
    config = ClusteringConfig(k=2, lookback=100)
    clustering = SimplifiedPercentileClustering(config)
    
    result = clustering.compute(high, low, close)
    
    # rel_pos should be between 0 and 1
    valid_values = result.rel_pos.dropna()
    assert valid_values.min() >= 0
    assert valid_values.max() <= 1


def test_clustering_centers_evolution():
    """Test that cluster centers evolve over time."""
    high, low, close = _sample_ohlcv_data(200)
    # Use single-feature mode to test center evolution
    config = ClusteringConfig(
        k=2,
        lookback=100,
        main_plot="RSI",
        feature_config=FeatureConfig(use_rsi=True),
    )
    clustering = SimplifiedPercentileClustering(config)
    
    result = clustering.compute(high, low, close)
    
    # Centers should change over time (not all same) in single-feature mode
    k0_centers = result.plot_k0_center.dropna()
    if len(k0_centers) > 1:
        # Centers should vary (not all identical)
        # Note: In "Clusters" mode, centers are constant (0, 1, 2)
        # So we test with single-feature mode where centers actually evolve
        assert k0_centers.std() > 0 or len(k0_centers.unique()) > 1

