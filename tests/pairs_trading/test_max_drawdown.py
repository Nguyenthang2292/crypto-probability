"""
Tests for max_drawdown module.
"""
import numpy as np
import pandas as pd

from modules.pairs_trading.metrics import calculate_max_drawdown


def test_calculate_max_drawdown_matches_manual():
    """Test that calculate_max_drawdown matches manual calculation."""
    spread = pd.Series([1, -2, -1, -3, 4, -2], dtype=float)
    cumulative = spread.cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    expected = float(drawdown.min())

    result = calculate_max_drawdown(cumulative)
    assert result == expected

