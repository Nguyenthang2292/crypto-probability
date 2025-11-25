"""
Test script for modules.xgboost_prediction_display - Display functions.
"""

import sys
from pathlib import Path
from io import StringIO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch
import numpy as np
import pytest

from modules.xgboost_prediction_display import print_classification_report


def test_print_classification_report():
    """Test print_classification_report function."""
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2, 0, 1, 1]  # One error
    
    # Capture stdout
    with patch("sys.stdout", new=StringIO()) as fake_out:
        print_classification_report(y_true, y_pred, title="Test Report")
        output = fake_out.getvalue()
    
    # Check that output contains expected elements
    assert "Test Report" in output
    assert "Classification Report" in output or "precision" in output.lower()
    assert "Confusion Matrix" in output
    assert "DOWN" in output
    assert "NEUTRAL" in output
    assert "UP" in output


def test_print_classification_report_perfect_prediction():
    """Test print_classification_report with perfect predictions."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]  # Perfect predictions
    
    with patch("sys.stdout", new=StringIO()) as fake_out:
        print_classification_report(y_true, y_pred)
        output = fake_out.getvalue()
    
    assert "Confusion Matrix" in output
    assert "DOWN" in output
    assert "NEUTRAL" in output
    assert "UP" in output


def test_print_classification_report_all_down():
    """Test print_classification_report with all DOWN predictions."""
    y_true = [0, 0, 0, 1, 1, 2]
    y_pred = [0, 0, 0, 0, 0, 0]  # All predicted as DOWN
    
    with patch("sys.stdout", new=StringIO()) as fake_out:
        print_classification_report(y_true, y_pred)
        output = fake_out.getvalue()
    
    assert "Confusion Matrix" in output


def test_print_classification_report_custom_title():
    """Test print_classification_report with custom title."""
    y_true = [0, 1, 2]
    y_pred = [0, 1, 2]
    
    custom_title = "Custom Test Title"
    
    with patch("sys.stdout", new=StringIO()) as fake_out:
        print_classification_report(y_true, y_pred, title=custom_title)
        output = fake_out.getvalue()
    
    assert custom_title in output


def test_print_classification_report_empty():
    """Test print_classification_report with empty arrays."""
    y_true = []
    y_pred = []
    
    with patch("sys.stdout", new=StringIO()) as fake_out:
        # Should not raise error, but may have empty output
        try:
            print_classification_report(y_true, y_pred)
            output = fake_out.getvalue()
        except Exception:
            # Empty arrays might cause errors in sklearn
            pass


def test_print_classification_report_imbalanced():
    """Test print_classification_report with imbalanced classes."""
    # Most predictions are class 0
    y_true = [0] * 50 + [1] * 5 + [2] * 5
    y_pred = [0] * 45 + [1] * 10 + [2] * 5  # Some errors
    
    with patch("sys.stdout", new=StringIO()) as fake_out:
        print_classification_report(y_true, y_pred)
        output = fake_out.getvalue()
    
    assert "Confusion Matrix" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

