"""
Tests for ensure_symbols_in_pools module.
"""
import pandas as pd
import pytest
from modules.pairs_trading.utils.ensure_symbols_in_pools import (
    ensure_symbols_in_candidate_pools,
)


def test_ensure_symbols_in_candidate_pools_basic():
    """Test basic functionality of ensure_symbols_in_candidate_pools."""
    performance_df = pd.DataFrame({
        "symbol": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        "score": [0.5, -0.3, 0.2]
    })
    best_df = pd.DataFrame({"symbol": ["BTC/USDT"], "score": [0.5]})
    worst_df = pd.DataFrame({"symbol": ["ETH/USDT"], "score": [-0.3]})
    
    # SOL/USDT is positive score (0.2), should be added to best_df
    best_updated, worst_updated = ensure_symbols_in_candidate_pools(
        performance_df, best_df, worst_df, ["SOL/USDT"]
    )
    
    assert "SOL/USDT" in best_updated["symbol"].values
    assert len(best_updated) == 2
    # Check sorting (descending score for best)
    assert best_updated.iloc[0]["symbol"] == "BTC/USDT"  # 0.5
    assert best_updated.iloc[1]["symbol"] == "SOL/USDT"  # 0.2
    
    # Worst df should be unchanged
    assert len(worst_updated) == 1
    assert "ETH/USDT" in worst_updated["symbol"].values


def test_ensure_symbols_in_candidate_pools_negative_score():
    """Test adding symbol with negative score to worst pool."""
    performance_df = pd.DataFrame({
        "symbol": ["BTC/USDT", "ADA/USDT"],
        "score": [0.5, -0.4]
    })
    best_df = pd.DataFrame({"symbol": ["BTC/USDT"], "score": [0.5]})
    worst_df = pd.DataFrame(columns=["symbol", "score"])
    
    # ADA/USDT is negative score (-0.4), should be added to worst_df
    best_updated, worst_updated = ensure_symbols_in_candidate_pools(
        performance_df, best_df, worst_df, ["ADA/USDT"]
    )
    
    assert "ADA/USDT" in worst_updated["symbol"].values
    assert len(worst_updated) == 1
    assert len(best_updated) == 1


def test_ensure_symbols_in_candidate_pools_already_present():
    """Test that symbols already in pools are not duplicated."""
    performance_df = pd.DataFrame({
        "symbol": ["BTC/USDT"],
        "score": [0.5]
    })
    best_df = pd.DataFrame({"symbol": ["BTC/USDT"], "score": [0.5]})
    worst_df = pd.DataFrame(columns=["symbol", "score"])
    
    # BTC/USDT already in best_df
    best_updated, worst_updated = ensure_symbols_in_candidate_pools(
        performance_df, best_df, worst_df, ["BTC/USDT"]
    )
    
    assert len(best_updated) == 1
    assert best_updated.iloc[0]["symbol"] == "BTC/USDT"


def test_ensure_symbols_in_candidate_pools_missing_symbol():
    """Test handling of symbols not found in performance data."""
    performance_df = pd.DataFrame({
        "symbol": ["BTC/USDT"],
        "score": [0.5]
    })
    best_df = pd.DataFrame({"symbol": ["BTC/USDT"], "score": [0.5]})
    worst_df = pd.DataFrame(columns=["symbol", "score"])
    
    # UNKNOWN/USDT not in performance_df
    best_updated, worst_updated = ensure_symbols_in_candidate_pools(
        performance_df, best_df, worst_df, ["UNKNOWN/USDT"]
    )
    
    # Should be unchanged
    assert len(best_updated) == 1
    assert len(worst_updated) == 0


def test_ensure_symbols_in_candidate_pools_empty_target():
    """Test handling of empty target symbols list."""
    performance_df = pd.DataFrame({
        "symbol": ["BTC/USDT"],
        "score": [0.5]
    })
    best_df = pd.DataFrame({"symbol": ["BTC/USDT"], "score": [0.5]})
    worst_df = pd.DataFrame(columns=["symbol", "score"])
    
    best_updated, worst_updated = ensure_symbols_in_candidate_pools(
        performance_df, best_df, worst_df, []
    )
    
    assert best_updated.equals(best_df)
    assert worst_updated.equals(worst_df)
