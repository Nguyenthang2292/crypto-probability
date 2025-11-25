"""
Test script for modules.feature_selection - FeatureSelector class.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
import tempfile
import shutil
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from modules.feature_selection import FeatureSelector

# Suppress warnings
warnings.filterwarnings("ignore")


def create_sample_data(n_samples=200, n_features=20, task_type="regression"):
    """Create sample data for testing."""
    np.random.seed(42)
    
    # Create features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    
    # Create target
    if task_type == "regression":
        y = pd.Series(
            X.iloc[:, 0] * 2 + X.iloc[:, 1] * 1.5 + np.random.randn(n_samples) * 0.1
        )
    else:
        # Classification: binary based on first feature
        y = pd.Series((X.iloc[:, 0] > 0).astype(int))
    
    return X, y


def test_feature_selector_initialization():
    """Test FeatureSelector initialization."""
    selector = FeatureSelector(
        method="mutual_info",
        top_k=10,
        collinearity_threshold=0.85,
    )
    
    assert selector.method == "mutual_info"
    assert selector.top_k == 10
    assert selector.collinearity_threshold == 0.85
    assert selector.selected_features == []
    assert selector.feature_scores == {}


def test_feature_selector_regression():
    """Test FeatureSelector with regression task."""
    X, y = create_sample_data(n_samples=100, n_features=15, task_type="regression")
    
    selector = FeatureSelector(method="mutual_info", top_k=5)
    X_selected = selector.select_features(X, y, task_type="regression")
    
    assert isinstance(X_selected, pd.DataFrame)
    assert len(X_selected.columns) == 5
    assert len(selector.selected_features) == 5
    assert len(selector.feature_scores) > 0


def test_feature_selector_classification():
    """Test FeatureSelector with classification task."""
    X, y = create_sample_data(n_samples=100, n_features=15, task_type="classification")
    
    selector = FeatureSelector(method="mutual_info", top_k=5)
    X_selected = selector.select_features(X, y, task_type="classification")
    
    assert isinstance(X_selected, pd.DataFrame)
    assert len(X_selected.columns) == 5


def test_feature_selector_boruta_method():
    """Test FeatureSelector with boruta method."""
    X, y = create_sample_data(n_samples=100, n_features=15, task_type="regression")
    
    selector = FeatureSelector(method="boruta", top_k=5)
    X_selected = selector.select_features(X, y, task_type="regression")
    
    assert isinstance(X_selected, pd.DataFrame)
    assert len(X_selected.columns) == 5


def test_feature_selector_f_test_method():
    """Test FeatureSelector with f_test method."""
    X, y = create_sample_data(n_samples=100, n_features=15, task_type="regression")
    
    selector = FeatureSelector(method="f_test", top_k=5)
    X_selected = selector.select_features(X, y, task_type="regression")
    
    assert isinstance(X_selected, pd.DataFrame)
    assert len(X_selected.columns) == 5


def test_feature_selector_combined_method():
    """Test FeatureSelector with combined method."""
    X, y = create_sample_data(n_samples=100, n_features=15, task_type="regression")
    
    selector = FeatureSelector(method="combined", top_k=5)
    X_selected = selector.select_features(X, y, task_type="regression")
    
    assert isinstance(X_selected, pd.DataFrame)
    assert len(X_selected.columns) == 5


def test_feature_selector_invalid_method():
    """Test FeatureSelector with invalid method."""
    X, y = create_sample_data(n_samples=100, n_features=15, task_type="regression")
    
    selector = FeatureSelector(method="invalid_method", top_k=5)
    
    with pytest.raises(ValueError, match="Unknown method"):
        selector.select_features(X, y, task_type="regression")


def test_filter_invalid_features():
    """Test _filter_invalid_features method."""
    X, y = create_sample_data(n_samples=100, n_features=10, task_type="regression")
    
    # Add invalid columns
    X["constant_col"] = 5  # Constant column
    X["nan_col"] = np.nan  # All NaN
    X["future_return"] = np.random.randn(100)  # Target leakage
    X["timestamp"] = pd.date_range("2023-01-01", periods=100)  # Non-numeric
    
    selector = FeatureSelector()
    X_clean = selector._filter_invalid_features(X, y)
    
    assert "constant_col" not in X_clean.columns
    assert "nan_col" not in X_clean.columns
    assert "future_return" not in X_clean.columns
    assert "timestamp" not in X_clean.columns


def test_remove_collinear_features():
    """Test _remove_collinear_features method."""
    X, y = create_sample_data(n_samples=100, n_features=10, task_type="regression")
    
    # Add highly correlated feature
    X["feature_0_copy"] = X["feature_0"] * 1.01  # 99%+ correlation
    
    selector = FeatureSelector(collinearity_threshold=0.9)
    X_clean = selector._remove_collinear_features(X)
    
    # Should remove one of the correlated features
    assert "feature_0" not in X_clean.columns or "feature_0_copy" not in X_clean.columns


def test_save_and_load_selection():
    """Test save_selection and load_selection methods."""
    X, y = create_sample_data(n_samples=100, n_features=15, task_type="regression")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        selector = FeatureSelector(
            method="mutual_info",
            top_k=5,
            selection_dir=tmpdir,
        )
        
        # Select features and save
        selector.select_features(X, y, task_type="regression", symbol="BTC/USDT")
        
        # Create new selector and load
        selector2 = FeatureSelector(
            method="mutual_info",
            top_k=5,
            selection_dir=tmpdir,
        )
        metadata = selector2.load_selection(symbol="BTC/USDT")
        
        assert metadata is not None
        assert "selected_features" in metadata
        assert len(metadata["selected_features"]) == 5
        assert selector2.selected_features == selector.selected_features


def test_apply_selection():
    """Test apply_selection method."""
    X, y = create_sample_data(n_samples=100, n_features=15, task_type="regression")
    
    selector = FeatureSelector(method="mutual_info", top_k=5)
    selector.select_features(X, y, task_type="regression")
    
    # Apply to new data
    X_new = X.copy()
    X_selected = selector.apply_selection(X_new)
    
    assert isinstance(X_selected, pd.DataFrame)
    assert len(X_selected.columns) == 5
    assert all(col in X_new.columns for col in X_selected.columns)


def test_apply_selection_without_selection():
    """Test apply_selection without prior selection raises error."""
    X, _ = create_sample_data(n_samples=100, n_features=15, task_type="regression")
    
    selector = FeatureSelector()
    
    with pytest.raises(ValueError, match="No features selected"):
        selector.apply_selection(X)


def test_get_feature_importance_report():
    """Test get_feature_importance_report method."""
    X, y = create_sample_data(n_samples=100, n_features=15, task_type="regression")
    
    selector = FeatureSelector(method="mutual_info", top_k=5)
    selector.select_features(X, y, task_type="regression")
    
    report = selector.get_feature_importance_report()
    
    assert isinstance(report, pd.DataFrame)
    assert "feature" in report.columns
    assert "score" in report.columns
    assert "selected" in report.columns
    assert len(report) > 0


def test_empty_dataframe():
    """Test FeatureSelector with empty DataFrame."""
    X = pd.DataFrame()
    y = pd.Series([])
    
    selector = FeatureSelector()
    
    with pytest.raises(ValueError, match="No valid features"):
        selector.select_features(X, y, task_type="regression")


def test_top_k_larger_than_features():
    """Test FeatureSelector when top_k is larger than available features."""
    X, y = create_sample_data(n_samples=100, n_features=5, task_type="regression")
    
    selector = FeatureSelector(method="mutual_info", top_k=10)
    X_selected = selector.select_features(X, y, task_type="regression")
    
    # Should select all available features (minus invalid ones)
    assert len(X_selected.columns) <= 5


def test_nan_handling():
    """Test FeatureSelector handles NaN values."""
    X, y = create_sample_data(n_samples=100, n_features=15, task_type="regression")
    
    # Add some NaN values
    X.loc[0:10, "feature_0"] = np.nan
    y.loc[0:5] = np.nan
    
    selector = FeatureSelector(method="mutual_info", top_k=5)
    # Should not raise error
    X_selected = selector.select_features(X, y, task_type="regression")
    
    assert isinstance(X_selected, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

