"""Configuration for dynamic strategy selection."""

from dataclasses import dataclass


@dataclass
class DynamicSelectionConfig:
    """Configuration for dynamic strategy selection based on market conditions.
    
    Attributes:
        enabled: Whether dynamic selection is enabled.
        lookback: Number of bars to look back for market condition analysis.
        volatility_threshold: Threshold for determining high volatility (0-1).
        trend_threshold: Threshold for determining trending market (0-1).
    """
    enabled: bool = False
    lookback: int = 20
    volatility_threshold: float = 0.6
    trend_threshold: float = 0.5

