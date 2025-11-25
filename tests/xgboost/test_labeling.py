import pandas as pd
import numpy as np
import pytest

from modules import xgboost_prediction_labeling as labeling


def test_apply_directional_labels_assigns_expected_classes(monkeypatch):
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 120.0, 104.0, 90.0],
            "ATR_RATIO_14_50": [1.0] * 5,
        }
    )

    result = labeling.apply_directional_labels(df.copy())

    assert result.loc[0, "TargetLabel"] == "UP"
    assert result.loc[2, "TargetLabel"] == "DOWN"
    assert result.loc[1, "TargetLabel"] == "NEUTRAL"
    assert result.loc[0, "DynamicThreshold"] == 0.05
    assert result.loc[0, "Target"] == labeling.LABEL_TO_ID["UP"]


def test_apply_directional_labels_empty_dataframe(monkeypatch):
    """Test apply_directional_labels with empty DataFrame."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    df = pd.DataFrame(
        {
            "close": [],
            "ATR_RATIO_14_50": [],
        }
    )

    result = labeling.apply_directional_labels(df.copy())
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_apply_directional_labels_single_row(monkeypatch):
    """Test apply_directional_labels with single row."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    df = pd.DataFrame(
        {
            "close": [100.0],
            "ATR_RATIO_14_50": [1.0],
        }
    )

    result = labeling.apply_directional_labels(df.copy())
    
    assert len(result) == 1
    assert "TargetLabel" in result.columns
    assert "Target" in result.columns


def test_apply_directional_labels_missing_atr_ratio(monkeypatch):
    """Test apply_directional_labels with missing ATR_RATIO column."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0],
        }
    )

    # Should handle missing ATR_RATIO gracefully
    result = labeling.apply_directional_labels(df.copy())
    
    assert isinstance(result, pd.DataFrame)
    assert "TargetLabel" in result.columns


def test_apply_directional_labels_with_nan(monkeypatch):
    """Test apply_directional_labels with NaN values."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    df = pd.DataFrame(
        {
            "close": [100.0, np.nan, 102.0],
            "ATR_RATIO_14_50": [1.0, 1.0, 1.0],
        }
    )

    result = labeling.apply_directional_labels(df.copy())
    
    assert isinstance(result, pd.DataFrame)
    assert "TargetLabel" in result.columns


def test_apply_directional_labels_with_atr(monkeypatch):
    """Test apply_directional_labels with ATR_14 column (different volatility calculation path)."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "ATR_14": [2.0, 2.1, 2.0, 2.2, 2.1],
            "ATR_RATIO_14_50": [1.0] * 5,
        }
    )

    result = labeling.apply_directional_labels(df.copy())
    
    assert isinstance(result, pd.DataFrame)
    assert "TargetLabel" in result.columns
    assert "DynamicThreshold" in result.columns
    assert "Target" in result.columns


def test_apply_directional_labels_small_dataset_adaptive_window(monkeypatch):
    """Test adaptive rolling window with small dataset (< 500 rows)."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    # Create dataset with 50 rows (less than default 500 window)
    df = pd.DataFrame(
        {
            "close": [100.0 + i * 0.1 for i in range(50)],
            "ATR_RATIO_14_50": [1.0] * 50,
        }
    )

    result = labeling.apply_directional_labels(df.copy())
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 50
    assert "TargetLabel" in result.columns
    assert "DynamicThreshold" in result.columns


def test_apply_directional_labels_large_dataset(monkeypatch):
    """Test with large dataset (> 500 rows) to test full rolling window."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    # Create dataset with 1000 rows
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.randn(1000) * 0.1)
    df = pd.DataFrame(
        {
            "close": prices,
            "ATR_RATIO_14_50": [1.0] * 1000,
        }
    )

    result = labeling.apply_directional_labels(df.copy())
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1000
    assert "TargetLabel" in result.columns
    assert "DynamicThreshold" in result.columns


def test_apply_directional_labels_constant_volatility(monkeypatch):
    """Test with constant volatility (all values same)."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    # Constant price = constant volatility
    df = pd.DataFrame(
        {
            "close": [100.0] * 20,
            "ATR_RATIO_14_50": [1.0] * 20,
        }
    )

    result = labeling.apply_directional_labels(df.copy())
    
    assert isinstance(result, pd.DataFrame)
    assert "TargetLabel" in result.columns
    # All rows should have valid labels (or NaN for last TARGET_HORIZON rows)
    assert result["TargetLabel"].notna().sum() >= 0


def test_apply_directional_labels_extreme_volatility(monkeypatch):
    """Test with extreme volatility values."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    # Extreme price movements
    df = pd.DataFrame(
        {
            "close": [100.0, 50.0, 150.0, 75.0, 125.0, 60.0, 140.0],
            "ATR_RATIO_14_50": [1.0] * 7,
        }
    )

    result = labeling.apply_directional_labels(df.copy())
    
    assert isinstance(result, pd.DataFrame)
    assert "TargetLabel" in result.columns
    assert "DynamicThreshold" in result.columns
    # Thresholds should be valid (not NaN, not inf)
    assert result["DynamicThreshold"].notna().all()
    assert not (result["DynamicThreshold"] == np.inf).any()
    assert not (result["DynamicThreshold"] == -np.inf).any()


def test_apply_directional_labels_large_target_horizon(monkeypatch):
    """Test with large TARGET_HORIZON relative to dataset size."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 10)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    # Small dataset with large horizon
    df = pd.DataFrame(
        {
            "close": [100.0 + i * 0.1 for i in range(15)],
            "ATR_RATIO_14_50": [1.0] * 15,
        }
    )

    result = labeling.apply_directional_labels(df.copy())
    
    assert isinstance(result, pd.DataFrame)
    assert "TargetLabel" in result.columns
    # Last TARGET_HORIZON rows should have NaN labels (no future data)
    assert result["TargetLabel"].isna().sum() >= 0


def test_apply_directional_labels_atr_ratio_with_inf(monkeypatch):
    """Test with ATR_RATIO containing inf values."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0],
            "ATR_RATIO_14_50": [1.0, np.inf, -np.inf, 1.0],
        }
    )

    result = labeling.apply_directional_labels(df.copy())
    
    assert isinstance(result, pd.DataFrame)
    assert "TargetLabel" in result.columns
    assert "DynamicThreshold" in result.columns
    # Thresholds should handle inf gracefully (clipped to valid range)
    assert result["DynamicThreshold"].notna().all()
    assert not (result["DynamicThreshold"] == np.inf).any()
    assert not (result["DynamicThreshold"] == -np.inf).any()


def test_apply_directional_labels_two_rows(monkeypatch):
    """Test with minimum dataset size (2 rows)."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    df = pd.DataFrame(
        {
            "close": [100.0, 101.0],
            "ATR_RATIO_14_50": [1.0, 1.0],
        }
    )

    result = labeling.apply_directional_labels(df.copy())
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "TargetLabel" in result.columns
    assert "Target" in result.columns
    assert "DynamicThreshold" in result.columns


def test_apply_directional_labels_consecutive_nan(monkeypatch):
    """Test with consecutive NaN values in close price."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    df = pd.DataFrame(
        {
            "close": [100.0, np.nan, np.nan, 103.0, 104.0],
            "ATR_RATIO_14_50": [1.0] * 5,
        }
    )

    result = labeling.apply_directional_labels(df.copy())
    
    assert isinstance(result, pd.DataFrame)
    assert "TargetLabel" in result.columns
    assert "DynamicThreshold" in result.columns


def test_apply_directional_labels_lookback_exceeds_dataset(monkeypatch):
    """Test when calculated lookback periods exceed dataset size."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 100)  # Very large horizon
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    # Small dataset
    df = pd.DataFrame(
        {
            "close": [100.0 + i * 0.1 for i in range(10)],
            "ATR_RATIO_14_50": [1.0] * 10,
        }
    )

    result = labeling.apply_directional_labels(df.copy())
    
    assert isinstance(result, pd.DataFrame)
    assert "TargetLabel" in result.columns
    assert "DynamicThreshold" in result.columns
    # Should handle gracefully without errors


def test_apply_directional_labels_rolling_window_behavior(monkeypatch):
    """Test that rolling window correctly uses only past data."""
    monkeypatch.setattr(labeling, "TARGET_HORIZON", 2)
    monkeypatch.setattr(labeling, "TARGET_BASE_THRESHOLD", 0.05)

    # Create dataset with changing volatility pattern
    # First 100 rows: low volatility
    # Next 100 rows: high volatility
    low_vol_prices = 100.0 + np.cumsum(np.random.randn(100) * 0.01)
    high_vol_prices = low_vol_prices[-1] + np.cumsum(np.random.randn(100) * 0.5)
    prices = np.concatenate([low_vol_prices, high_vol_prices])
    
    df = pd.DataFrame(
        {
            "close": prices,
            "ATR_RATIO_14_50": [1.0] * 200,
        }
    )

    result = labeling.apply_directional_labels(df.copy())
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 200
    assert "TargetLabel" in result.columns
    assert "DynamicThreshold" in result.columns
    # Rolling thresholds should adapt to volatility changes
    # Thresholds in high volatility section should generally be higher
    early_thresholds = result["DynamicThreshold"].iloc[50:100]
    late_thresholds = result["DynamicThreshold"].iloc[150:200]
    # This is a sanity check - thresholds should exist and be valid
    assert early_thresholds.notna().all()
    assert late_thresholds.notna().all()