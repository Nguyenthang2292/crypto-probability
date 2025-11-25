"""
Test script for modules.xgboost_prediction_cli - CLI parsing functions.
"""

import sys
from pathlib import Path
from io import StringIO
from unittest.mock import patch, Mock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from modules.xgboost_prediction_cli import (
    prompt_with_default,
    resolve_input,
    parse_args,
)


def test_prompt_with_default_with_input():
    """Test prompt_with_default with user input."""
    with patch("builtins.input", return_value="BTC/USDT"):
        result = prompt_with_default("Enter symbol", "ETH/USDT", str)
        assert result == "BTC/USDT"


def test_prompt_with_default_no_input():
    """Test prompt_with_default with no input (uses default)."""
    with patch("builtins.input", return_value=""):
        result = prompt_with_default("Enter symbol", "ETH/USDT", str)
        assert result == "ETH/USDT"


def test_prompt_with_default_int_cast():
    """Test prompt_with_default with int casting."""
    with patch("builtins.input", return_value="100"):
        result = prompt_with_default("Enter limit", 50, int)
        assert result == 100


def test_prompt_with_default_int_cast_default():
    """Test prompt_with_default with int casting and default."""
    with patch("builtins.input", return_value=""):
        result = prompt_with_default("Enter limit", 50, int)
        assert result == 50


def test_prompt_with_default_invalid_input():
    """Test prompt_with_default with invalid input."""
    # Mock input to return invalid value first, then valid
    with patch("builtins.input", side_effect=["invalid", "100"]):
        result = prompt_with_default("Enter limit", 50, int)
        assert result == 100


def test_resolve_input_with_cli_value():
    """Test resolve_input with CLI value."""
    result = resolve_input("BTC/USDT", "ETH/USDT", "Enter symbol", str, allow_prompt=False)
    assert result == "BTC/USDT"


def test_resolve_input_no_cli_with_prompt():
    """Test resolve_input without CLI value, with prompt."""
    with patch("builtins.input", return_value="BTC/USDT"):
        result = resolve_input(None, "ETH/USDT", "Enter symbol", str, allow_prompt=True)
        assert result == "BTC/USDT"


def test_resolve_input_no_cli_no_prompt():
    """Test resolve_input without CLI value, without prompt."""
    result = resolve_input(None, "ETH/USDT", "Enter symbol", str, allow_prompt=False)
    assert result == "ETH/USDT"


def test_resolve_input_int_cast():
    """Test resolve_input with int casting."""
    result = resolve_input("100", 50, "Enter limit", int, allow_prompt=False)
    assert result == 100


def test_parse_args_defaults():
    """Test parse_args with default values."""
    with patch("sys.argv", ["xgboost_prediction_cli.py"]):
        args = parse_args()
        
        # All should be None with default values
        assert args.symbol is None
        assert args.quote is None
        assert args.timeframe is None
        assert args.limit is None
        assert args.exchanges is None
        assert args.no_prompt is False


def test_parse_args_with_symbol():
    """Test parse_args with symbol argument."""
    with patch("sys.argv", ["xgboost_prediction_cli.py", "--symbol", "BTC/USDT"]):
        args = parse_args()
        assert args.symbol == "BTC/USDT"


def test_parse_args_with_all_args():
    """Test parse_args with all arguments."""
    with patch("sys.argv", [
        "xgboost_prediction_cli.py",
        "--symbol", "ETH/USDT",
        "--quote", "USDT",
        "--timeframe", "4h",
        "--limit", "500",
        "--exchanges", "binance,kraken",
        "--no-prompt",
    ]):
        args = parse_args()
        
        assert args.symbol == "ETH/USDT"
        assert args.quote == "USDT"
        assert args.timeframe == "4h"
        assert args.limit == 500
        assert args.exchanges == "binance,kraken"
        assert args.no_prompt is True


def test_parse_args_short_flags():
    """Test parse_args with short flag arguments."""
    with patch("sys.argv", [
        "xgboost_prediction_cli.py",
        "-s", "BTC/USDT",
        "-t", "1h",
        "-l", "1000",
        "-e", "binance",
    ]):
        args = parse_args()
        
        assert args.symbol == "BTC/USDT"
        assert args.timeframe == "1h"
        assert args.limit == 1000
        assert args.exchanges == "binance"


def test_parse_args_no_prompt_flag():
    """Test parse_args with --no-prompt flag."""
    with patch("sys.argv", ["xgboost_prediction_cli.py", "--no-prompt"]):
        args = parse_args()
        assert args.no_prompt is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

