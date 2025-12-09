"""
Tests for mean_reversion strategy.
"""
import numpy as np
import pandas as pd
import pytest

from modules.simplified_percentile_clustering.strategies.mean_reversion import (
    MeanReversionConfig,
    generate_signals_mean_reversion,
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


def test_mean_reversion_config_defaults():
    """Test MeanReversionConfig default values."""
    config = MeanReversionConfig()
    
    assert config.extreme_threshold == 0.2
    assert config.min_extreme_duration == 3
    assert config.require_reversal_signal is True
    assert config.reversal_lookback == 3
    assert config.min_signal_strength == 0.4


def test_mean_reversion_config_with_clustering_config():
    """Test MeanReversionConfig with ClusteringConfig."""
    clustering_config = ClusteringConfig(k=3)
    config = MeanReversionConfig(clustering_config=clustering_config)
    
    assert config.clustering_config == clustering_config
    # With k=3, targets should be adjusted
    assert config.bullish_reversion_target == 1.0
    assert config.bearish_reversion_target == 1.0


def test_generate_signals_mean_reversion_basic():
    """Test basic signal generation."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(
        k=2,
        lookback=100,
        feature_config=FeatureConfig(use_rsi=True, use_cci=True),
    )
    strategy_config = MeanReversionConfig(
        clustering_config=clustering_config,
        extreme_threshold=0.3,  # Higher threshold for testing
        min_extreme_duration=1,
        min_signal_strength=0.1,  # Lower threshold for testing
    )
    
    signals, strength, metadata = generate_signals_mean_reversion(
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


def test_generate_signals_mean_reversion_with_clustering_result():
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
    
    strategy_config = MeanReversionConfig(
        clustering_config=clustering_config,
        extreme_threshold=0.3,
        min_extreme_duration=1,
        min_signal_strength=0.1,
    )
    
    signals, strength, metadata = generate_signals_mean_reversion(
        high=high,
        low=low,
        close=close,
        clustering_result=clustering_result,
        config=strategy_config,
    )
    
    assert len(signals) == len(close)
    assert "cluster_val" in metadata.columns
    assert "extreme_duration" in metadata.columns


def test_generate_signals_mean_reversion_no_reversal():
    """Test signal generation without reversal requirement."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(k=2, lookback=100)
    strategy_config = MeanReversionConfig(
        clustering_config=clustering_config,
        require_reversal_signal=False,
        extreme_threshold=0.3,
        min_extreme_duration=1,
        min_signal_strength=0.1,
    )
    
    signals, strength, metadata = generate_signals_mean_reversion(
        high=high,
        low=low,
        close=close,
        config=strategy_config,
    )
    
    assert len(signals) == len(close)


def test_generate_signals_mean_reversion_metadata():
    """Test that metadata contains expected columns."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(k=2, lookback=100)
    strategy_config = MeanReversionConfig(
        clustering_config=clustering_config,
        extreme_threshold=0.3,
        min_extreme_duration=1,
        min_signal_strength=0.1,
    )
    
    signals, strength, metadata = generate_signals_mean_reversion(
        high=high,
        low=low,
        close=close,
        config=strategy_config,
    )
    
    expected_columns = [
        "cluster_val",
        "real_clust",
        "extreme_duration",
        "in_extreme",
        "price_change",
        "signal",
        "signal_strength",
    ]
    
    for col in expected_columns:
        assert col in metadata.columns


def test_generate_signals_mean_reversion_k3():
    """Test signal generation with k=3."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(
        k=3,
        lookback=100,
        feature_config=FeatureConfig(use_rsi=True),
    )
    strategy_config = MeanReversionConfig(
        clustering_config=clustering_config,
        extreme_threshold=0.3,
        min_extreme_duration=1,
        min_signal_strength=0.1,
    )
    
    signals, strength, metadata = generate_signals_mean_reversion(
        high=high,
        low=low,
        close=close,
        config=strategy_config,
    )
    
    assert len(signals) == len(close)
    assert all(signals.isin([-1, 0, 1]))


def test_generate_signals_mean_reversion_extreme_duration():
    """Test that extreme duration is tracked correctly."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(k=2, lookback=100)
    strategy_config = MeanReversionConfig(
        clustering_config=clustering_config,
        extreme_threshold=0.3,
        min_extreme_duration=5,  # Require 5 bars
        min_signal_strength=0.1,
    )
    
    signals, strength, metadata = generate_signals_mean_reversion(
        high=high,
        low=low,
        close=close,
        config=strategy_config,
    )
    
    # Check that extreme_duration is tracked
    assert "extreme_duration" in metadata.columns
    assert metadata["extreme_duration"].min() >= 0
    assert "in_extreme" in metadata.columns


def test_generate_signals_mean_reversion_extreme_detection():
    """Test that extreme conditions are detected."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(k=2, lookback=100)
    strategy_config = MeanReversionConfig(
        clustering_config=clustering_config,
        extreme_threshold=0.4,  # Large threshold
        min_extreme_duration=1,
        min_signal_strength=0.1,
    )
    
    signals, strength, metadata = generate_signals_mean_reversion(
        high=high,
        low=low,
        close=close,
        config=strategy_config,
    )
    
    # Check in_extreme flag
    assert "in_extreme" in metadata.columns
    assert metadata["in_extreme"].dtype == bool


def test_generate_signals_mean_reversion_signal_strength():
    """Test that signal strength is calculated correctly."""
    high, low, close = _sample_ohlcv_data(200)
    
    clustering_config = ClusteringConfig(k=2, lookback=100)
    strategy_config = MeanReversionConfig(
        clustering_config=clustering_config,
        extreme_threshold=0.3,
        min_extreme_duration=1,
        min_signal_strength=0.5,  # Higher threshold
    )
    
    signals, strength, metadata = generate_signals_mean_reversion(
        high=high,
        low=low,
        close=close,
        config=strategy_config,
    )
    
    # All non-zero signals should have strength >= threshold
    non_zero_signals = signals[signals != 0]
    if len(non_zero_signals) > 0:
        corresponding_strength = strength[signals != 0]
        assert all(corresponding_strength >= 0.5)

