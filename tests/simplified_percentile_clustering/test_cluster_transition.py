"""
Tests for cluster_transition strategy.
"""
import numpy as np
import pandas as pd
import pytest

from modules.simplified_percentile_clustering.strategies.cluster_transition import (
    ClusterTransitionConfig,
    generate_signals_cluster_transition,
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


def test_cluster_transition_config_defaults():
    """Test ClusterTransitionConfig default values."""
    config = ClusterTransitionConfig()
    
    assert config.require_price_confirmation is True
    assert config.min_rel_pos_change == 0.1
    assert config.use_real_clust_cross is True
    assert config.min_signal_strength == 0.3
    assert config.bullish_transitions == [(0, 1), (0, 2), (1, 2)]
    assert config.bearish_transitions == [(2, 1), (2, 0), (1, 0)]


def test_cluster_transition_config_custom_transitions():
    """Test ClusterTransitionConfig with custom transitions."""
    config = ClusterTransitionConfig(
        bullish_transitions=[(0, 1)],
        bearish_transitions=[(1, 0)],
    )
    
    assert config.bullish_transitions == [(0, 1)]
    assert config.bearish_transitions == [(1, 0)]


def test_generate_signals_cluster_transition_basic():
    """Test basic signal generation."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(
        k=2,
        lookback=100,
        feature_config=FeatureConfig(use_rsi=True, use_cci=True),
    )
    strategy_config = ClusterTransitionConfig(
        clustering_config=clustering_config,
        min_signal_strength=0.1,  # Lower threshold for testing
    )
    
    signals, strength, metadata = generate_signals_cluster_transition(
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
    
    # Strength should be between 0 and 1
    valid_strength = strength.dropna()
    if len(valid_strength) > 0:
        assert valid_strength.min() >= 0
        assert valid_strength.max() <= 1


def test_generate_signals_cluster_transition_with_clustering_result():
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
    
    strategy_config = ClusterTransitionConfig(
        clustering_config=clustering_config,
        min_signal_strength=0.1,
    )
    
    signals, strength, metadata = generate_signals_cluster_transition(
        high=high,
        low=low,
        close=close,
        clustering_result=clustering_result,
        config=strategy_config,
    )
    
    assert len(signals) == len(close)
    assert "cluster_val" in metadata.columns
    assert "signal" in metadata.columns


def test_generate_signals_cluster_transition_no_price_confirmation():
    """Test signal generation without price confirmation."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(k=2, lookback=100)
    strategy_config = ClusterTransitionConfig(
        clustering_config=clustering_config,
        require_price_confirmation=False,
        min_signal_strength=0.1,
    )
    
    signals, strength, metadata = generate_signals_cluster_transition(
        high=high,
        low=low,
        close=close,
        config=strategy_config,
    )
    
    assert len(signals) == len(close)


def test_generate_signals_cluster_transition_metadata():
    """Test that metadata contains expected columns."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(k=2, lookback=100)
    strategy_config = ClusterTransitionConfig(
        clustering_config=clustering_config,
        min_signal_strength=0.1,
    )
    
    signals, strength, metadata = generate_signals_cluster_transition(
        high=high,
        low=low,
        close=close,
        config=strategy_config,
    )
    
    expected_columns = [
        "cluster_val",
        "prev_cluster_val",
        "real_clust",
        "prev_real_clust",
        "rel_pos",
        "price_change",
        "signal",
        "signal_strength",
    ]
    
    for col in expected_columns:
        assert col in metadata.columns


def test_generate_signals_cluster_transition_k3():
    """Test signal generation with k=3."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(
        k=3,
        lookback=100,
        feature_config=FeatureConfig(use_rsi=True),
    )
    strategy_config = ClusterTransitionConfig(
        clustering_config=clustering_config,
        min_signal_strength=0.1,
    )
    
    signals, strength, metadata = generate_signals_cluster_transition(
        high=high,
        low=low,
        close=close,
        config=strategy_config,
    )
    
    assert len(signals) == len(close)
    # With k=3, transitions can include (0,2), (2,0), etc.
    assert all(signals.isin([-1, 0, 1]))


def test_generate_signals_cluster_transition_min_strength():
    """Test that min_signal_strength is respected."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(k=2, lookback=100)
    strategy_config = ClusterTransitionConfig(
        clustering_config=clustering_config,
        min_signal_strength=0.9,  # Very high threshold
    )
    
    signals, strength, metadata = generate_signals_cluster_transition(
        high=high,
        low=low,
        close=close,
        config=strategy_config,
    )
    
    # With high threshold, should have fewer signals
    # But still valid
    assert all(signals.isin([-1, 0, 1]))
    
    # All non-zero signals should have strength >= threshold
    non_zero_signals = signals[signals != 0]
    if len(non_zero_signals) > 0:
        corresponding_strength = strength[signals != 0]
        assert all(corresponding_strength >= 0.9)


def test_generate_signals_cluster_transition_real_clust_cross():
    """Test real_clust crossing detection."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(k=2, lookback=100)
    strategy_config = ClusterTransitionConfig(
        clustering_config=clustering_config,
        use_real_clust_cross=True,
        min_signal_strength=0.1,
    )
    
    signals, strength, metadata = generate_signals_cluster_transition(
        high=high,
        low=low,
        close=close,
        config=strategy_config,
    )
    
    assert len(signals) == len(close)

