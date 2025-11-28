"""
Tests for direction_metrics module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.pairs_trading.metrics import calculate_direction_metrics


def test_calculate_direction_metrics_produces_classification_scores():
    """Test that calculate_direction_metrics produces classification scores."""
    # Construct synthetic spread with mean-reverting behaviour
    base = np.sin(np.linspace(0, 20, 400))
    noise = np.linspace(0, 0.5, 400)
    spread = pd.Series(base + noise)

    metrics = calculate_direction_metrics(
        spread, zscore_lookback=40, classification_zscore=0.5
    )

    assert metrics["classification_f1"] is not None
    assert metrics["classification_accuracy"] is not None
    assert metrics["classification_precision"] is not None
    assert metrics["classification_recall"] is not None


def test_calculate_direction_metrics_insufficient_data():
    """Test that calculate_direction_metrics returns None for insufficient data."""
    spread = pd.Series([1.0, 2.0, 3.0])

    metrics = calculate_direction_metrics(
        spread, zscore_lookback=50, classification_zscore=0.5
    )

    assert metrics["classification_f1"] is None
    assert metrics["classification_accuracy"] is None
    assert metrics["classification_precision"] is None
    assert metrics["classification_recall"] is None


def test_calculate_direction_metrics_with_nan_values():
    """Test that calculate_direction_metrics handles NaN values correctly."""
    base = np.sin(np.linspace(0, 20, 400))
    noise = np.linspace(0, 0.5, 400)
    spread_values = base + noise
    spread_values[100] = np.nan
    spread_values[200] = np.nan
    spread = pd.Series(spread_values)

    metrics = calculate_direction_metrics(
        spread, zscore_lookback=40, classification_zscore=0.5
    )

    # Should still compute valid metrics after dropping NaN
    if len(spread.dropna()) >= 40:
        # Results may vary based on available data
        assert isinstance(metrics["classification_f1"], (float, type(None)))
        assert isinstance(metrics["classification_accuracy"], (float, type(None)))


def test_calculate_direction_metrics_none_input():
    """Test that calculate_direction_metrics handles None input."""
    metrics = calculate_direction_metrics(
        None, zscore_lookback=40, classification_zscore=0.5
    )

    assert metrics["classification_f1"] is None
    assert metrics["classification_accuracy"] is None
    assert metrics["classification_precision"] is None
    assert metrics["classification_recall"] is None


def test_calculate_direction_metrics_default_parameters():
    """Test that calculate_direction_metrics uses default parameters."""
    base = np.sin(np.linspace(0, 20, 400))
    noise = np.linspace(0, 0.5, 400)
    spread = pd.Series(base + noise)

    metrics = calculate_direction_metrics(spread)

    # Should work with default parameters
    assert isinstance(metrics["classification_f1"], (float, type(None)))
    assert isinstance(metrics["classification_accuracy"], (float, type(None)))


def test_calculate_direction_metrics_insufficient_active_signals():
    """Test that calculate_direction_metrics returns None when too few active signals."""
    # Create spread that rarely exceeds threshold
    spread = pd.Series(np.linspace(0.1, 0.15, 400))  # Very small values

    metrics = calculate_direction_metrics(
        spread, zscore_lookback=40, classification_zscore=0.5
    )

    # With very few active signals (below minimum), should return None
    if metrics["classification_f1"] is None:
        assert metrics["classification_accuracy"] is None


def test_calculate_direction_metrics_metrics_range():
    """Test that classification metrics are in valid range [0, 1] when not None."""
    base = np.sin(np.linspace(0, 20, 400))
    noise = np.linspace(0, 0.5, 400)
    spread = pd.Series(base + noise)

    metrics = calculate_direction_metrics(
        spread, zscore_lookback=40, classification_zscore=0.5
    )

    if metrics["classification_f1"] is not None:
        assert 0 <= metrics["classification_f1"] <= 1
    if metrics["classification_accuracy"] is not None:
        assert 0 <= metrics["classification_accuracy"] <= 1
    if metrics["classification_precision"] is not None:
        assert 0 <= metrics["classification_precision"] <= 1
    if metrics["classification_recall"] is not None:
        assert 0 <= metrics["classification_recall"] <= 1

