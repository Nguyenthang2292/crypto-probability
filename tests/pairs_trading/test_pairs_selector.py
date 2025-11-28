"""
Tests for pairs_selector module.
"""
import pandas as pd
import pytest
from modules.pairs_trading.utils.pairs_selector import (
    select_top_unique_pairs,
    select_pairs_for_symbols,
)


def test_select_top_unique_pairs_basic():
    """Test basic functionality of select_top_unique_pairs."""
    pairs_df = pd.DataFrame({
        "long_symbol": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        "short_symbol": ["ETH/USDT", "BNB/USDT", "ADA/USDT"],
        "score": [0.9, 0.8, 0.7]
    })
    
    # Select 2 pairs
    selected = select_top_unique_pairs(pairs_df, target_pairs=2)
    
    assert len(selected) == 2
    # First pair: BTC/ETH (unique)
    assert selected.iloc[0]["long_symbol"] == "BTC/USDT"
    # Second pair: SOL/ADA (unique, ETH used in first pair so skipped ETH/BNB in first pass)
    # Wait, ETH is in short_symbol of first pair, and long_symbol of second pair.
    # First pass:
    # 1. BTC/ETH -> Selected. Used: {BTC, ETH}
    # 2. ETH/BNB -> ETH used. Skipped.
    # 3. SOL/ADA -> Selected. Used: {BTC, ETH, SOL, ADA}
    
    assert selected.iloc[1]["long_symbol"] == "SOL/USDT"


def test_select_top_unique_pairs_fallback_to_second_pass():
    """Test fallback to second pass when not enough unique pairs."""
    pairs_df = pd.DataFrame({
        "long_symbol": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        "short_symbol": ["ETH/USDT", "BNB/USDT", "BTC/USDT"],
        "score": [0.9, 0.8, 0.7]
    })
    
    # Target 3 pairs, but all share symbols
    selected = select_top_unique_pairs(pairs_df, target_pairs=3)
    
    assert len(selected) == 3
    # First pass:
    # 1. BTC/ETH -> Selected. Used: {BTC, ETH}
    # 2. ETH/BNB -> ETH used. Skipped.
    # 3. BNB/BTC -> BNB used (no), BTC used (yes). Skipped.
    
    # Second pass:
    # Fill remaining slots with skipped pairs
    assert selected.iloc[0]["long_symbol"] == "BTC/USDT"
    assert selected.iloc[1]["long_symbol"] == "ETH/USDT"
    assert selected.iloc[2]["long_symbol"] == "BNB/USDT"


def test_select_top_unique_pairs_empty_input():
    """Test handling of empty input DataFrame."""
    empty_df = pd.DataFrame(columns=["long_symbol", "short_symbol", "score"])
    selected = select_top_unique_pairs(empty_df, target_pairs=5)
    assert selected.empty
    
    selected_none = select_top_unique_pairs(None, target_pairs=5)
    assert selected_none is None


def test_select_pairs_for_symbols_basic():
    """Test basic functionality of select_pairs_for_symbols."""
    pairs_df = pd.DataFrame({
        "long_symbol": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        "short_symbol": ["ETH/USDT", "BNB/USDT", "ADA/USDT"],
        "score": [0.9, 0.8, 0.7]
    })
    
    # Find pair for ETH/USDT (appears in first pair as short, second as long)
    # Should pick first occurrence (highest score)
    selected = select_pairs_for_symbols(pairs_df, ["ETH/USDT"])
    
    assert len(selected) == 1
    assert selected.iloc[0]["long_symbol"] == "BTC/USDT"
    assert selected.iloc[0]["short_symbol"] == "ETH/USDT"


def test_select_pairs_for_symbols_multiple_targets():
    """Test selecting pairs for multiple target symbols."""
    pairs_df = pd.DataFrame({
        "long_symbol": ["BTC/USDT", "SOL/USDT", "XRP/USDT"],
        "short_symbol": ["ETH/USDT", "ADA/USDT", "DOGE/USDT"],
        "score": [0.9, 0.8, 0.7]
    })
    
    selected = select_pairs_for_symbols(pairs_df, ["BTC/USDT", "SOL/USDT"])
    
    assert len(selected) == 2
    assert "BTC/USDT" in selected["long_symbol"].values
    assert "SOL/USDT" in selected["long_symbol"].values


def test_select_pairs_for_symbols_no_match():
    """Test handling when target symbol is not found."""
    pairs_df = pd.DataFrame({
        "long_symbol": ["BTC/USDT"],
        "short_symbol": ["ETH/USDT"],
        "score": [0.9]
    })
    
    selected = select_pairs_for_symbols(pairs_df, ["SOL/USDT"])
    assert selected.empty


def test_select_pairs_for_symbols_max_pairs():
    """Test max_pairs limit."""
    pairs_df = pd.DataFrame({
        "long_symbol": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        "short_symbol": ["ETH/USDT", "BNB/USDT", "ADA/USDT"],
        "score": [0.9, 0.8, 0.7]
    })
    
    # Request 3 symbols but limit to 2 pairs
    selected = select_pairs_for_symbols(
        pairs_df, 
        ["BTC/USDT", "ETH/USDT", "SOL/USDT"], 
        max_pairs=2
    )
    
    assert len(selected) == 2
