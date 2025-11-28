"""
Tests for input_parsers module.
"""
import pytest
from modules.pairs_trading.cli.input_parsers import (
    standardize_symbol_input,
    parse_weights,
    parse_symbols,
)


def test_standardize_symbol_input():
    """Test standardizing symbol input."""
    # Basic cases
    assert standardize_symbol_input("BTC/USDT") == "BTC/USDT"
    assert standardize_symbol_input("btc/usdt") == "BTC/USDT"
    
    # Missing quote
    assert standardize_symbol_input("BTC") == "BTC/USDT"
    assert standardize_symbol_input("btc") == "BTC/USDT"
    
    # Suffix style
    assert standardize_symbol_input("BTCUSDT") == "BTC/USDT"
    assert standardize_symbol_input("btcusdt") == "BTC/USDT"
    
    # Edge cases
    assert standardize_symbol_input("") == ""
    assert standardize_symbol_input("  btc  ") == "BTC/USDT"
    
    # Custom quote
    assert standardize_symbol_input("ETH/BTC") == "ETH/BTC"


def test_parse_weights_default():
    """Test parsing weights with default values."""
    weights = parse_weights(None)
    assert weights["1d"] == 0.5
    assert weights["3d"] == 0.3
    assert weights["1w"] == 0.2
    assert sum(weights.values()) == 1.0


def test_parse_weights_preset():
    """Test parsing weights from presets."""
    # Balanced preset
    weights = parse_weights(None, preset_key="balanced")
    assert weights["1d"] == 0.3
    assert weights["3d"] == 0.4
    assert weights["1w"] == 0.3
    
    # Momentum preset
    weights = parse_weights(None, preset_key="momentum")
    assert weights["1d"] == 0.5
    assert weights["3d"] == 0.3
    assert weights["1w"] == 0.2


def test_parse_weights_custom_string():
    """Test parsing custom weights string."""
    weights_str = "1d:0.8, 3d:0.1, 1w:0.1"
    weights = parse_weights(weights_str)
    
    assert weights["1d"] == 0.8
    assert weights["3d"] == 0.1
    assert weights["1w"] == 0.1
    assert sum(weights.values()) == 1.0


def test_parse_weights_normalization():
    """Test that weights are normalized to sum to 1.0."""
    # Sum = 2.0
    weights_str = "1d:1.0, 3d:0.6, 1w:0.4"
    weights = parse_weights(weights_str)
    
    # Should be normalized by dividing by 2.0
    assert weights["1d"] == 0.5
    assert weights["3d"] == 0.3
    assert weights["1w"] == 0.2
    assert sum(weights.values()) == 1.0


def test_parse_weights_invalid_input():
    """Test handling of invalid weight strings."""
    # Invalid format
    weights = parse_weights("invalid_string")
    # Should fallback to default
    assert weights["1d"] == 0.5
    assert weights["3d"] == 0.3
    assert weights["1w"] == 0.2


def test_parse_symbols():
    """Test parsing symbols string."""
    # Comma separated
    inputs, parsed = parse_symbols("BTC, ETH")
    assert "BTC" in inputs
    assert "ETH" in inputs
    assert "BTC/USDT" in parsed
    assert "ETH/USDT" in parsed
    
    # Space separated
    inputs, parsed = parse_symbols("BTC ETH")
    assert len(inputs) == 2
    
    # Mixed separators
    inputs, parsed = parse_symbols("BTC, ETH; SOL|ADA")
    assert len(inputs) == 4
    assert "SOL" in inputs
    assert "ADA" in inputs
    
    # Empty input
    inputs, parsed = parse_symbols("")
    assert len(inputs) == 0
    assert len(parsed) == 0
    
    # Duplicates
    inputs, parsed = parse_symbols("BTC, BTC, btc")
    assert len(inputs) == 1
    assert len(parsed) == 1
