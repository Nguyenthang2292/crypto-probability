"""
Common utility functions for all algorithms.

This module provides utilities organized into the following categories:
- System: Platform-specific configuration
- Data: DataFrame/Series manipulation
- Domain: Trading-specific utilities (symbols, timeframes)
- CLI/UI: Text formatting, user input, and logging
"""

import re
import sys
import io
import os
import pandas as pd
from typing import Optional
from colorama import Fore, Style
from modules.config import DEFAULT_QUOTE

# ============================================================================
# SYSTEM UTILITIES
# ============================================================================

def configure_windows_stdio() -> None:
    """
    Configure Windows stdio encoding for UTF-8 support.
    
    Only applies to interactive CLI runs, not during pytest.
    This function fixes encoding issues on Windows by wrapping
    stdout and stderr with UTF-8 encoding.
    
    Note:
        - Only runs on Windows (win32 platform)
        - Skips configuration during pytest runs
        - Only configures if stdout/stderr have buffer attribute
    """
    if sys.platform != "win32":
        return
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    if not hasattr(sys.stdout, "buffer") or isinstance(sys.stdout, io.TextIOWrapper):
        return
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ============================================================================
# DATA UTILITIES
# ============================================================================

def dataframe_to_close_series(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    """
    Converts a fetched OHLCV DataFrame into a pandas Series of closing prices indexed by timestamp.

    Args:
        df: OHLCV DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    Returns:
        pandas Series of closing prices indexed by timestamp, or None if input is invalid
    """
    if df is None or df.empty:
        return None
    if "timestamp" not in df.columns or "close" not in df.columns:
        return None
    series = df.set_index("timestamp")["close"].copy()
    series.name = "close"
    return series


# ============================================================================
# DOMAIN-SPECIFIC UTILITIES (Trading)
# ============================================================================

# --- Timeframe Utilities ---

def timeframe_to_minutes(timeframe: str) -> int:
    """
    Converts a timeframe string like '30m', '1h', '1d' into minutes.

    Args:
        timeframe: Timeframe string (e.g., '30m', '1h', '1d', '1w')

    Returns:
        Number of minutes
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


# --- Symbol Normalization Utilities ---

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

    Args:
        symbol: Symbol string (e.g., 'BTC/USDT', 'ETH-USDT')

    Returns:
        Normalized key string (e.g., 'BTCUSDT', 'ETHUSDT')
    """
    if not symbol:
        return ""
    return "".join(ch for ch in symbol.upper() if ch.isalnum())


# ============================================================================
# CLI/UI UTILITIES
# ============================================================================

# --- Text Formatting ---

def color_text(text: str, color: str = Fore.WHITE, style: str = Style.NORMAL) -> str:
    """
    Applies color and style to text using colorama.

    Args:
        text: Text to format
        color: Colorama Fore color (default: Fore.WHITE)
        style: Colorama Style (default: Style.NORMAL)

    Returns:
        Formatted text string with color and style codes
    """
    return f"{style}{color}{text}{Style.RESET_ALL}"


def format_price(value: float) -> str:
    """
    Formats prices/indicators with adaptive precision so tiny values remain readable.

    Args:
        value: Numeric value to format

    Returns:
        Formatted price string with appropriate precision, or "N/A" if invalid
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


# --- User Input ---

def prompt_user_input(
    prompt: str,
    default: Optional[str] = None,
    color: str = Fore.YELLOW,
) -> str:
    """
    Prompt user for input with optional default value and colored prompt.
    
    Args:
        prompt: Prompt message to display
        default: Default value if user enters empty string
        color: Colorama Fore color for prompt (default: Fore.YELLOW)
        
    Returns:
        User input string, or default if empty input provided
    """
    user_input = input(color_text(prompt, color)).strip()
    return user_input if user_input else (default or "")


def extract_dict_from_namespace(namespace, keys: list) -> dict:
    """
    Extract a dictionary from a namespace object using specified keys.
    
    Args:
        namespace: Namespace object (e.g., from argparse)
        keys: List of attribute names to extract
        
    Returns:
        Dictionary with extracted key-value pairs
    """
    return {key: getattr(namespace, key, None) for key in keys}


# --- Logging Functions ---
# Organized by severity level and purpose

# Standard severity levels
def log_info(message: str) -> None:
    """Print informational message with blue color."""
    print(color_text(message, Fore.BLUE))


def log_success(message: str) -> None:
    """Print success message with green color."""
    print(color_text(message, Fore.GREEN))


def log_error(message: str) -> None:
    """Print error message with red color and bright style."""
    print(color_text(message, Fore.RED, Style.BRIGHT))


def log_warn(message: str) -> None:
    """Print warning message with yellow color."""
    print(color_text(message, Fore.YELLOW))


def log_debug(message: str) -> None:
    """Print debug message with white color."""
    print(color_text(message, Fore.WHITE))


# Domain-specific logging
def log_data(message: str) -> None:
    """Print data-related message with cyan color."""
    print(color_text(message, Fore.CYAN))


def log_analysis(message: str) -> None:
    """Print analysis-related message with magenta color."""
    print(color_text(message, Fore.MAGENTA))


def log_model(message: str) -> None:
    """Print model-related message with magenta color."""
    print(color_text(message, Fore.MAGENTA))


def log_exchange(message: str) -> None:
    """Print exchange-related message with cyan color."""
    print(color_text(message, Fore.CYAN))


def log_system(message: str) -> None:
    """Print system-level message with white color."""
    print(color_text(message, Fore.WHITE))


def log_progress(message: str) -> None:
    """Print progress update message with yellow color."""
    print(color_text(message, Fore.YELLOW))