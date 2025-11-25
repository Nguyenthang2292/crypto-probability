"""
Utility functions specific to xgboost_prediction_main.py
"""

from modules.config import PREDICTION_WINDOWS
from modules.common.utils import color_text, format_price, timeframe_to_minutes

# Re-export common utilities for backward compatibility
__all__ = [
    "get_prediction_window",
    "color_text",
    "format_price",
    "timeframe_to_minutes",
]


def get_prediction_window(timeframe: str) -> str:
    """
    Returns a textual description of the prediction horizon based on timeframe.
    """
    timeframe = timeframe.lower()
    return PREDICTION_WINDOWS.get(timeframe, "next sessions")
