"""
Tests for direction_metrics module.
"""
import numpy as np
import pandas as pd

from modules.pairs_trading.metrics import calculate_direction_metrics


def test_calculate_direction_metrics_returns_dict_with_all_keys():
    """Test that calculate_direction_metrics returns dict with expected keys."""
    # Create a spread series that will generate active signals
    spread = pd.Series([0.1, -0.05, 0.15, -0.1, 0.2, -0.15, 0.05, -0.2] * 20)
    
    result = calculate_direction_metrics(spread)
    
    assert isinstance(result, dict)
    assert "classification_accuracy" in result
    assert "classification_precision" in result
    assert "classification_recall" in result
    assert "classification_f1" in result


def test_calculate_direction_metrics_insufficient_data():
    """Test that calculate_direction_metrics returns None values for insufficient data."""
    spread = pd.Series([0.1, -0.05, 0.15])  # Less than lookback
    
    result = calculate_direction_metrics(spread)
    
    assert result["classification_accuracy"] is None
    assert result["classification_precision"] is None
    assert result["classification_recall"] is None
    assert result["classification_f1"] is None


def test_calculate_direction_metrics_none_input():
    """Test that calculate_direction_metrics handles None input."""
    result = calculate_direction_metrics(None)
    
    assert result["classification_accuracy"] is None
    assert result["classification_precision"] is None
    assert result["classification_recall"] is None
    assert result["classification_f1"] is None


def test_calculate_direction_metrics_non_series_input():
    """Test that calculate_direction_metrics handles non-Series input."""
    result = calculate_direction_metrics([1, 2, 3, 4, 5])
    
    assert result["classification_accuracy"] is None
    assert result["classification_precision"] is None
    assert result["classification_recall"] is None
    assert result["classification_f1"] is None


def test_calculate_direction_metrics_default_parameters():
    """Test that calculate_direction_metrics works with default parameters."""
    spread = pd.Series([0.1, -0.05, 0.15, -0.1, 0.2, -0.15, 0.05, -0.2] * 20)
    
    result = calculate_direction_metrics(spread)
    
    # Should return a dict, values may be None if insufficient active signals
    assert isinstance(result, dict)

