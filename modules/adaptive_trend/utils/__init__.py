"""
Utility functions for Adaptive Trend Classification (ATC).

This package provides core utility functions used throughout the ATC system:
- rate_of_change: Calculate percentage price change
- diflen: Calculate length offsets for Moving Averages based on robustness
- exp_growth: Calculate exponential growth factor over time
- ATCConfig: Configuration dataclass for ATC analysis
- create_atc_config_from_dict: Helper to create ATCConfig from dictionary
"""

from modules.adaptive_trend.utils.rate_of_change import rate_of_change
from modules.adaptive_trend.utils.diflen import diflen
from modules.adaptive_trend.utils.exp_growth import exp_growth
from modules.adaptive_trend.utils.config import ATCConfig, create_atc_config_from_dict

__all__ = [
    "rate_of_change",
    "diflen",
    "exp_growth",
    "ATCConfig",
    "create_atc_config_from_dict",
]

