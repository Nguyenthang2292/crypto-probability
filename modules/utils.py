"""
Utility functions for timeframe conversion, formatting, and text coloring.
"""
import re
import pandas as pd
from colorama import Fore, Style
from .config import PREDICTION_WINDOWS


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


def get_prediction_window(timeframe: str) -> str:
    """
    Returns a textual description of the prediction horizon based on timeframe.
    """
    timeframe = timeframe.lower()
    return PREDICTION_WINDOWS.get(timeframe, "next sessions")


def color_text(text: str, color: str = Fore.WHITE, style: str = Style.NORMAL) -> str:
    """Applies color and style to text using colorama."""
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

