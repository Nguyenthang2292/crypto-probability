"""
Position data class for portfolio management.
"""
from dataclasses import dataclass


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    direction: str
    entry_price: float
    size_usdt: float

