"""
Common utility functions for all algorithms.
"""

import re
import pandas as pd
from colorama import Fore, Style
from modules.config import DEFAULT_QUOTE


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Converts a timeframe string like '30m', '1h', '1d' into minutes.
    """
    match = re.match(r"^\s*(\d+)\s*([mhdw])\s*$", timeframe.lower())
    if not match:
        return 60  # default 1h

    value, unit = match.groups()
    value = int(value)

    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 60 * 24
    if unit == "w":
        return value * 60 * 24 * 7
    return 60


def color_text(text: str, color: str = Fore.WHITE, style: str = Style.NORMAL) -> str:
    """
    Applies color and style to text using colorama.
    """
    return f"{style}{color}{text}{Style.RESET_ALL}"


def format_price(value: float) -> str:
    """
    Formats prices/indicators with adaptive precision so tiny values remain readable.
    """
    if value is None or pd.isna(value):
        return "N/A"

    abs_val = abs(value)
    if abs_val >= 1:
        precision = 2
    elif abs_val >= 0.01:
        precision = 4
    elif abs_val >= 0.0001:
        precision = 6
    else:
        precision = 8

    return f"{value:.{precision}f}"


def normalize_symbol(user_input: str, quote: str = DEFAULT_QUOTE) -> str:
    """
    Converts user input like 'xmr' into 'XMR/USDT'. Keeps existing slash pairs.

    Args:
        user_input: User input symbol (e.g., 'btc', 'BTC/USDT', 'btcusdt')
        quote: Quote currency (default: DEFAULT_QUOTE)

    Returns:
        Normalized symbol in format 'BASE/QUOTE' (e.g., 'BTC/USDT')
    """
    if not user_input:
        return f"BTC/{quote}"

    norm = user_input.strip().upper()
    if "/" in norm:
        return norm

    if norm.endswith(quote):
        return f"{norm[:-len(quote)]}/{quote}"

    return f"{norm}/{quote}"


def normalize_symbol_key(symbol: str) -> str:
    """
    Generates a compare-friendly key by uppercasing and stripping separators.
    """
    if not symbol:
        return ""
    return "".join(ch for ch in symbol.upper() if ch.isalnum())