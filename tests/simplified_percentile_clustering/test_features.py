"""
Tests for features module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.simplified_percentile_clustering.core.features import (
    FeatureCalculator,
    FeatureConfig,
    compute_features,
)


def _sample_ohlcv_data(length=100):
    """Generate sample OHLCV data."""
    np.random.seed(42)
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(length) * 0.5)
    
    high = prices + np.abs(np.random.randn(length) * 0.3)
    low = prices - np.abs(np.random.randn(length) * 0.3)
    close = prices
    
    return pd.Series(high), pd.Series(low), pd.Series(close)


def test_feature_calculator_z_score():
    """Test z_score calculation."""
    calculator = FeatureCalculator()
    prices = pd.Series([100.0, 102.0, 105.0, 103.0, 107.0])
    
    z_scores = calculator.z_score(prices, length=3)
    
    assert len(z_scores) == len(prices)
    assert not pd.isna(z_scores.iloc[-1])  # Last value should be calculated


def test_feature_calculator_round_fisher():
    """Test round_fisher clamping."""
    calculator = FeatureCalculator()
    
    assert calculator.round_fisher(0.5) == 0.5
    assert calculator.round_fisher(1.0) == 0.999
    assert calculator.round_fisher(-1.0) == -0.999
    assert calculator.round_fisher(0.99) == 0.99
    assert calculator.round_fisher(-0.99) == -0.99


def test_feature_calculator_fisher_transform():
    """Test Fisher Transform calculation."""
    calculator = FeatureCalculator()
    high, low, close = _sample_ohlcv_data(50)
    
    fisher = calculator.fisher_transform(high, low, close, length=9)
    
    assert len(fisher) == len(close)
    assert not fisher.isna().all()  # Should have some valid values


def test_feature_calculator_dmi_difference():
    """Test DMI difference calculation."""
    calculator = FeatureCalculator()
    high, low, close = _sample_ohlcv_data(50)
    
    dmi = calculator.dmi_difference(high, low, close, length=9)
    
    assert len(dmi) == len(close)
    assert not dmi.isna().all()  # Should have some valid values


def test_feature_calculator_compute_rsi():
    """Test RSI computation."""
    calculator = FeatureCalculator()
    close = _sample_ohlcv_data(100)[2]
    
    rsi, rsi_val = calculator.compute_rsi(close, lookback=50)
    
    assert len(rsi) == len(close)
    assert len(rsi_val) == len(close)
    assert not rsi.isna().all()


def test_feature_calculator_compute_cci():
    """Test CCI computation."""
    calculator = FeatureCalculator()
    high, low, close = _sample_ohlcv_data(100)
    
    cci, cci_val = calculator.compute_cci(high, low, close, lookback=50)
    
    assert len(cci) == len(close)
    assert len(cci_val) == len(close)


def test_feature_calculator_compute_fisher():
    """Test Fisher computation."""
    calculator = FeatureCalculator()
    high, low, close = _sample_ohlcv_data(100)
    
    fisher, fisher_val = calculator.compute_fisher(high, low, close, lookback=50)
    
    assert len(fisher) == len(close)
    assert len(fisher_val) == len(close)


def test_feature_calculator_compute_dmi():
    """Test DMI computation."""
    calculator = FeatureCalculator()
    high, low, close = _sample_ohlcv_data(100)
    
    dmi, dmi_val = calculator.compute_dmi(high, low, close, lookback=50)
    
    assert len(dmi) == len(close)
    assert len(dmi_val) == len(close)


def test_feature_calculator_compute_zscore():
    """Test Z-Score computation."""
    calculator = FeatureCalculator()
    close = _sample_ohlcv_data(100)[2]
    
    zsc_val = calculator.compute_zscore(close)
    
    assert len(zsc_val) == len(close)


def test_feature_calculator_compute_mar():
    """Test MAR computation."""
    calculator = FeatureCalculator()
    close = _sample_ohlcv_data(100)[2]
    
    mar, mar_val = calculator.compute_mar(close, lookback=50)
    
    assert len(mar) == len(close)
    assert len(mar_val) == len(close)


def test_feature_calculator_compute_all():
    """Test compute_all with all features enabled."""
    config = FeatureConfig(
        use_rsi=True,
        use_cci=True,
        use_fisher=True,
        use_dmi=True,
        use_zscore=True,
        use_mar=True,
    )
    calculator = FeatureCalculator(config)
    high, low, close = _sample_ohlcv_data(100)
    
    results = calculator.compute_all(high, low, close, lookback=50)
    
    assert "rsi_val" in results
    assert "cci_val" in results
    assert "fisher_val" in results
    assert "dmi_val" in results
    assert "zsc_val" in results
    assert "mar_val" in results


def test_feature_calculator_compute_all_selective():
    """Test compute_all with selective features."""
    config = FeatureConfig(
        use_rsi=True,
        use_cci=False,
        use_fisher=True,
        use_dmi=False,
        use_zscore=True,
        use_mar=False,
    )
    calculator = FeatureCalculator(config)
    high, low, close = _sample_ohlcv_data(100)
    
    results = calculator.compute_all(high, low, close, lookback=50)
    
    assert "rsi_val" in results
    assert "cci_val" not in results
    assert "fisher_val" in results
    assert "dmi_val" not in results
    assert "zsc_val" in results
    assert "mar_val" not in results


def test_feature_config_defaults():
    """Test FeatureConfig default values."""
    config = FeatureConfig()
    
    assert config.use_rsi is True
    assert config.rsi_len == 14
    assert config.rsi_standardize is True
    assert config.mar_type == "SMA"


def test_compute_features_convenience():
    """Test compute_features convenience function."""
    high, low, close = _sample_ohlcv_data(100)
    
    results = compute_features(high, low, close, lookback=50)
    
    assert isinstance(results, dict)
    assert len(results) > 0


def test_feature_calculator_standardization():
    """Test that standardization works correctly."""
    config = FeatureConfig(
        use_rsi=True,
        rsi_standardize=True,
    )
    calculator = FeatureCalculator(config)
    close = _sample_ohlcv_data(100)[2]
    
    rsi, rsi_val = calculator.compute_rsi(close, lookback=50)
    
    # Standardized values should have different scale than raw RSI
    assert not np.array_equal(rsi.values, rsi_val.values)


def test_feature_calculator_no_standardization():
    """Test that non-standardized features work."""
    config = FeatureConfig(
        use_rsi=True,
        rsi_standardize=False,
    )
    calculator = FeatureCalculator(config)
    close = _sample_ohlcv_data(100)[2]
    
    rsi, rsi_val = calculator.compute_rsi(close, lookback=50)
    
    # Non-standardized values should be same as raw
    pd.testing.assert_series_equal(rsi, rsi_val)


def test_feature_calculator_mar_sma():
    """Test MAR with SMA."""
    config = FeatureConfig(
        use_mar=True,
        mar_type="SMA",
    )
    calculator = FeatureCalculator(config)
    close = _sample_ohlcv_data(100)[2]
    
    mar, mar_val = calculator.compute_mar(close, lookback=50)
    
    assert len(mar) == len(close)
    assert len(mar_val) == len(close)


def test_feature_calculator_mar_ema():
    """Test MAR with EMA."""
    config = FeatureConfig(
        use_mar=True,
        mar_type="EMA",
    )
    calculator = FeatureCalculator(config)
    close = _sample_ohlcv_data(100)[2]
    
    mar, mar_val = calculator.compute_mar(close, lookback=50)
    
    assert len(mar) == len(close)
    assert len(mar_val) == len(close)

