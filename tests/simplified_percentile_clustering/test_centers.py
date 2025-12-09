"""
Tests for centers module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.simplified_percentile_clustering.core.centers import (
    ClusterCenters,
    compute_centers,
)


def _sample_values(length=100):
    """Generate sample values."""
    np.random.seed(42)
    return pd.Series(100.0 + np.random.randn(length) * 10.0)


def test_cluster_centers_init():
    """Test ClusterCenters initialization."""
    calculator = ClusterCenters(lookback=50, p_low=5.0, p_high=95.0, k=2)
    
    assert calculator.lookback == 50
    assert calculator.p_low == 5.0
    assert calculator.p_high == 95.0
    assert calculator.k == 2


def test_cluster_centers_init_invalid_k():
    """Test ClusterCenters with invalid k."""
    with pytest.raises(ValueError, match="k must be 2 or 3"):
        ClusterCenters(lookback=50, k=4)


def test_cluster_centers_init_invalid_percentiles():
    """Test ClusterCenters with invalid percentiles."""
    with pytest.raises(ValueError):
        ClusterCenters(lookback=50, p_low=95.0, p_high=5.0)  # p_low > p_high


def test_cluster_centers_get_percentile():
    """Test get_percentile method."""
    calculator = ClusterCenters(lookback=50, k=2)
    
    arr = [1.0, 2.0, 3.0, 4.0, 5.0]
    p50 = calculator.get_percentile(arr, 50.0)
    assert p50 == 3.0  # Middle value
    
    p0 = calculator.get_percentile(arr, 0.0)
    assert p0 == 1.0  # First value
    
    p100 = calculator.get_percentile(arr, 100.0)
    assert p100 == 5.0  # Last value


def test_cluster_centers_get_percentile_empty():
    """Test get_percentile with empty array."""
    calculator = ClusterCenters(lookback=50, k=2)
    
    result = calculator.get_percentile([], 50.0)
    assert np.isnan(result)


def test_cluster_centers_update_k2():
    """Test update method with k=2."""
    calculator = ClusterCenters(lookback=10, p_low=5.0, p_high=95.0, k=2)
    
    # Add some values
    for i in range(20):
        centers = calculator.update(float(100.0 + i))
    
    assert len(centers) == 2
    assert centers[0] < centers[1]  # k0 should be lower than k1


def test_cluster_centers_update_k3():
    """Test update method with k=3."""
    calculator = ClusterCenters(lookback=10, p_low=5.0, p_high=95.0, k=3)
    
    # Add some values
    for i in range(20):
        centers = calculator.update(float(100.0 + i))
    
    assert len(centers) == 3
    assert centers[0] < centers[1] < centers[2]  # Should be ordered


def test_cluster_centers_update_nan():
    """Test update with NaN values."""
    calculator = ClusterCenters(lookback=10, k=2)
    
    # First update with NaN
    centers = calculator.update(np.nan)
    assert all(np.isnan(c) for c in centers)
    
    # Then add valid values
    for i in range(10):
        centers = calculator.update(float(100.0 + i))
    
    # Should have valid centers now
    assert all(not np.isnan(c) for c in centers)


def test_cluster_centers_lookback_limit():
    """Test that lookback limit is respected."""
    calculator = ClusterCenters(lookback=5, k=2)
    
    # Add more values than lookback
    for i in range(20):
        calculator.update(float(100.0 + i))
    
    # Should only keep last 5 values
    centers = calculator.get_current_centers()
    assert len(centers) == 2
    assert all(not np.isnan(c) for c in centers)


def test_cluster_centers_get_current_centers():
    """Test get_current_centers without updating."""
    calculator = ClusterCenters(lookback=10, k=2)
    
    # Add some values
    for i in range(10):
        calculator.update(float(100.0 + i))
    
    # Get centers without updating
    centers1 = calculator.get_current_centers()
    centers2 = calculator.get_current_centers()
    
    assert centers1 == centers2  # Should be same


def test_compute_centers_k2():
    """Test compute_centers function with k=2."""
    values = _sample_values(100)
    
    centers_df = compute_centers(
        values,
        lookback=50,
        p_low=5.0,
        p_high=95.0,
        k=2,
    )
    
    assert isinstance(centers_df, pd.DataFrame)
    assert len(centers_df) == len(values)
    assert "k0" in centers_df.columns
    assert "k1" in centers_df.columns
    assert "k2" not in centers_df.columns


def test_compute_centers_k3():
    """Test compute_centers function with k=3."""
    values = _sample_values(100)
    
    centers_df = compute_centers(
        values,
        lookback=50,
        p_low=5.0,
        p_high=95.0,
        k=3,
    )
    
    assert isinstance(centers_df, pd.DataFrame)
    assert len(centers_df) == len(values)
    assert "k0" in centers_df.columns
    assert "k1" in centers_df.columns
    assert "k2" in centers_df.columns


def test_compute_centers_centers_ordered():
    """Test that centers are ordered correctly."""
    values = _sample_values(100)
    
    centers_df = compute_centers(values, lookback=50, k=2)
    
    # Last row should have k0 < k1
    last_row = centers_df.iloc[-1]
    assert last_row["k0"] < last_row["k1"]


def test_compute_centers_with_nan():
    """Test compute_centers with NaN values."""
    values = _sample_values(100)
    values.iloc[10:20] = np.nan  # Add some NaN values
    
    centers_df = compute_centers(values, lookback=50, k=2)
    
    assert len(centers_df) == len(values)
    # Should handle NaN gracefully


def test_cluster_centers_percentile_calculation():
    """Test that percentiles are calculated correctly."""
    calculator = ClusterCenters(lookback=100, p_low=10.0, p_high=90.0, k=2)
    
    # Add values from 0 to 99
    for i in range(100):
        calculator.update(float(i))
    
    centers = calculator.get_current_centers()
    
    # k0 should be around (10th percentile + mean) / 2
    # k1 should be around (90th percentile + mean) / 2
    assert centers[0] < centers[1]
    assert centers[0] < 50.0  # Should be below mean
    assert centers[1] > 50.0  # Should be above mean

