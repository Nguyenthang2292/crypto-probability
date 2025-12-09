"""
Tests for regime_following strategy.
"""
import numpy as np
import pandas as pd
import pytest

from modules.simplified_percentile_clustering.strategies.regime_following import (
    RegimeFollowingConfig,
    generate_signals_regime_following,
)
from modules.simplified_percentile_clustering.core.clustering import (
    ClusteringConfig,
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


def test_regime_following_config_defaults():
    """Test RegimeFollowingConfig default values."""
    config = RegimeFollowingConfig()
    
    assert config.min_regime_strength == 0.7
    assert config.min_cluster_duration == 2
    assert config.require_momentum is True
    assert config.momentum_period == 5
    assert config.bullish_clusters == [1, 2]
    assert config.bearish_clusters == [0]


def test_regime_following_config_custom_clusters():
    """Test RegimeFollowingConfig with custom cluster preferences."""
    config = RegimeFollowingConfig(
        bullish_clusters=[2],
        bearish_clusters=[0, 1],
    )
    
    assert config.bullish_clusters == [2]
    assert config.bearish_clusters == [0, 1]


def test_generate_signals_regime_following_basic():
    """Test basic signal generation."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(
        k=2,
        lookback=100,
        feature_config=FeatureConfig(use_rsi=True, use_cci=True),
    )
    strategy_config = RegimeFollowingConfig(
        clustering_config=clustering_config,
        min_regime_strength=0.3,  # Lower threshold for testing
        min_cluster_duration=1,
    )
    
    signals, strength, metadata = generate_signals_regime_following(
        high=high,
        low=low,
        close=close,
        config=strategy_config,
    )
    
    assert len(signals) == len(close)
    assert len(strength) == len(close)
    assert isinstance(metadata, pd.DataFrame)
    
    # Signals should be -1, 0, or 1
    assert all(signals.isin([-1, 0, 1]))


def test_generate_signals_regime_following_with_clustering_result():
    """Test signal generation with pre-computed clustering result."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(
        k=2,
        lookback=100,
        feature_config=FeatureConfig(use_rsi=True),
    )
    
    # Pre-compute clustering
    clustering_result = compute_clustering(
        high, low, close, config=clustering_config
    )
    
    strategy_config = RegimeFollowingConfig(
        clustering_config=clustering_config,
        min_regime_strength=0.3,
        min_cluster_duration=1,
    )
    
    signals, strength, metadata = generate_signals_regime_following(
        high=high,
        low=low,
        close=close,
        clustering_result=clustering_result,
        config=strategy_config,
    )
    
    assert len(signals) == len(close)
    assert "cluster_val" in metadata.columns
    assert "regime_strength" in metadata.columns


def test_generate_signals_regime_following_no_momentum():
    """Test signal generation without momentum requirement."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(k=2, lookback=100)
    strategy_config = RegimeFollowingConfig(
        clustering_config=clustering_config,
        require_momentum=False,
        min_regime_strength=0.3,
        min_cluster_duration=1,
    )
    
    signals, strength, metadata = generate_signals_regime_following(
        high=high,
        low=low,
        close=close,
        config=strategy_config,
    )
    
    assert len(signals) == len(close)


def test_generate_signals_regime_following_metadata():
    """Test that metadata contains expected columns."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(k=2, lookback=100)
    strategy_config = RegimeFollowingConfig(
        clustering_config=clustering_config,
        min_regime_strength=0.3,
        min_cluster_duration=1,
    )
    
    signals, strength, metadata = generate_signals_regime_following(
        high=high,
        low=low,
        close=close,
        config=strategy_config,
    )
    
    expected_columns = [
        "cluster_val",
        "real_clust",
        "regime_strength",
        "cluster_duration",
        "momentum",
        "price_change",
        "signal",
        "signal_strength",
    ]
    
    for col in expected_columns:
        assert col in metadata.columns


def test_generate_signals_regime_following_k3():
    """Test signal generation with k=3."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(
        k=3,
        lookback=100,
        feature_config=FeatureConfig(use_rsi=True),
    )
    strategy_config = RegimeFollowingConfig(
        clustering_config=clustering_config,
        min_regime_strength=0.3,
        min_cluster_duration=1,
    )
    
    signals, strength, metadata = generate_signals_regime_following(
        high=high,
        low=low,
        close=close,
        config=strategy_config,
    )
    
    assert len(signals) == len(close)
    assert all(signals.isin([-1, 0, 1]))


def test_generate_signals_regime_following_cluster_duration():
    """Test that cluster duration is tracked correctly."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(k=2, lookback=100)
    strategy_config = RegimeFollowingConfig(
        clustering_config=clustering_config,
        min_regime_strength=0.3,
        min_cluster_duration=5,  # Require 5 bars
    )
    
    signals, strength, metadata = generate_signals_regime_following(
        high=high,
        low=low,
        close=close,
        config=strategy_config,
    )
    
    # Check that cluster_duration is tracked
    assert "cluster_duration" in metadata.columns
    assert metadata["cluster_duration"].min() >= 0


def test_generate_signals_regime_following_regime_strength():
    """Test that regime strength is calculated correctly."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(k=2, lookback=100)
    strategy_config = RegimeFollowingConfig(
        clustering_config=clustering_config,
        min_regime_strength=0.5,
        min_cluster_duration=1,
    )
    
    signals, strength, metadata = generate_signals_regime_following(
        high=high,
        low=low,
        close=close,
        config=strategy_config,
    )
    
    # Regime strength should be between 0 and 1
    valid_strength = metadata["regime_strength"].dropna()
    if len(valid_strength) > 0:
        assert valid_strength.min() >= 0
        assert valid_strength.max() <= 1

