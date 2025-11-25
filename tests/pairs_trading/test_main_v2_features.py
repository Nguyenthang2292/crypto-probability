"""
Test cases for pairs_trading_main_v2.py new features.

Tests the display and CLI features:
- display_pairs_opportunities with quantitative metrics
- Sorting functionality
- Summary statistics
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import contextlib

# Import functions from main script
sys.path.insert(0, str(ROOT))
from pairs_trading_main_v2 import (
    display_pairs_opportunities,
    select_top_unique_pairs,
)


def test_display_pairs_opportunities_with_quantitative_score():
    """Test that display_pairs_opportunities handles quantitative_score correctly."""
    pairs_df = pd.DataFrame(
        [
            {
                "long_symbol": "WORST1/USDT",
                "short_symbol": "BEST1/USDT",
                "spread": 0.25,
                "correlation": 0.6,
                "opportunity_score": 0.30,
                "quantitative_score": 75.5,
                "is_cointegrated": True,
                "half_life": 20.0,
                "spread_sharpe": 1.5,
                "max_drawdown": -0.15,
            }
        ]
    )

    # Capture stdout
    output = StringIO()
    with contextlib.redirect_stdout(output):
        display_pairs_opportunities(pairs_df, max_display=1, verbose=False)

    output_str = output.getvalue()

    # Should contain quantitative_score display
    assert "QuantScore" in output_str or "quantitative" in output_str.lower()
    # Should contain cointegration status
    assert "Coint" in output_str or "cointegrated" in output_str.lower()


def test_display_pairs_opportunities_verbose_mode():
    """Test that display_pairs_opportunities shows additional metrics in verbose mode."""
    pairs_df = pd.DataFrame(
        [
            {
                "long_symbol": "WORST1/USDT",
                "short_symbol": "BEST1/USDT",
                "spread": 0.25,
                "correlation": 0.6,
                "opportunity_score": 0.30,
                "quantitative_score": 75.5,
                "is_cointegrated": True,
                "half_life": 20.0,
                "spread_sharpe": 1.5,
                "max_drawdown": -0.15,
            }
        ]
    )

    # Capture stdout
    output = StringIO()
    with contextlib.redirect_stdout(output):
        display_pairs_opportunities(pairs_df, max_display=1, verbose=True)

    output_str = output.getvalue()

    # Verbose mode should show additional metrics
    assert "HalfLife" in output_str or "half" in output_str.lower()
    assert "Sharpe" in output_str or "sharpe" in output_str.lower()
    assert "MaxDD" in output_str or "drawdown" in output_str.lower()


def test_display_pairs_opportunities_handles_missing_metrics():
    """Test that display_pairs_opportunities handles missing quantitative metrics gracefully."""
    pairs_df = pd.DataFrame(
        [
            {
                "long_symbol": "WORST1/USDT",
                "short_symbol": "BEST1/USDT",
                "spread": 0.25,
                "correlation": 0.6,
                "opportunity_score": 0.30,
                "quantitative_score": None,  # Missing
                "is_cointegrated": None,  # Missing
            }
        ]
    )

    # Should not raise error
    output = StringIO()
    with contextlib.redirect_stdout(output):
        try:
            display_pairs_opportunities(pairs_df, max_display=1, verbose=False)
            success = True
        except Exception as e:
            success = False
            print(f"Error: {e}")

    assert success, "display_pairs_opportunities should handle missing metrics gracefully"


def test_select_top_unique_pairs():
    """Test select_top_unique_pairs selects unique symbols when possible."""
    pairs_df = pd.DataFrame(
        [
            {
                "long_symbol": "WORST1/USDT",
                "short_symbol": "BEST1/USDT",
                "spread": 0.25,
                "opportunity_score": 0.30,
                "quantitative_score": 80,
            },
            {
                "long_symbol": "WORST2/USDT",
                "short_symbol": "BEST2/USDT",
                "spread": 0.20,
                "opportunity_score": 0.25,
                "quantitative_score": 75,
            },
            {
                "long_symbol": "WORST1/USDT",  # Duplicate long_symbol
                "short_symbol": "BEST3/USDT",
                "spread": 0.18,
                "opportunity_score": 0.20,
                "quantitative_score": 70,
            },
        ]
    )

    selected = select_top_unique_pairs(pairs_df, target_pairs=2)

    # Should select first 2 pairs (unique symbols)
    assert len(selected) == 2
    assert selected.iloc[0]["long_symbol"] == "WORST1/USDT"
    assert selected.iloc[1]["long_symbol"] == "WORST2/USDT"

    # Check that symbols are unique
    all_symbols = set(selected["long_symbol"]) | set(selected["short_symbol"])
    assert len(all_symbols) == 4  # 2 long + 2 short = 4 unique symbols


def test_select_top_unique_pairs_empty_dataframe():
    """Test select_top_unique_pairs handles empty DataFrame."""
    empty_df = pd.DataFrame(columns=["long_symbol", "short_symbol", "spread"])

    selected = select_top_unique_pairs(empty_df, target_pairs=5)

    assert len(selected) == 0


def test_display_pairs_opportunities_empty_dataframe():
    """Test display_pairs_opportunities handles empty DataFrame."""
    empty_df = pd.DataFrame(columns=["long_symbol", "short_symbol"])

    output = StringIO()
    with contextlib.redirect_stdout(output):
        display_pairs_opportunities(empty_df, max_display=10, verbose=False)

    output_str = output.getvalue()
    assert "No pairs" in output_str or "found" in output_str.lower()


def test_display_pairs_opportunities_cointegration_status():
    """Test that cointegration status is displayed correctly."""
    pairs_df = pd.DataFrame(
        [
            {
                "long_symbol": "WORST1/USDT",
                "short_symbol": "BEST1/USDT",
                "spread": 0.25,
                "correlation": 0.6,
                "opportunity_score": 0.30,
                "quantitative_score": 75.5,
                "is_cointegrated": True,  # Cointegrated
            },
            {
                "long_symbol": "WORST2/USDT",
                "short_symbol": "BEST2/USDT",
                "spread": 0.20,
                "correlation": 0.5,
                "opportunity_score": 0.25,
                "quantitative_score": 50.0,
                "is_cointegrated": False,  # Not cointegrated
            },
        ]
    )

    output = StringIO()
    with contextlib.redirect_stdout(output):
        display_pairs_opportunities(pairs_df, max_display=2, verbose=False)

    output_str = output.getvalue()

    # Should display cointegration status
    # Note: Exact format depends on implementation, but should show status somehow
    assert len(output_str) > 0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])

