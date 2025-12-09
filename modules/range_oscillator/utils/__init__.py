"""
Utilities for Range Oscillator module.

This module provides utility functions for the Range Oscillator.
Configuration classes have been moved to modules.range_oscillator.config.
"""

from .oscillator_data import get_oscillator_data

# Re-export config classes from config module for backward compatibility
from modules.range_oscillator.config import (
    DynamicSelectionConfig,
    ConsensusConfig,
    StrategySpecificConfig,
    CombinedStrategyConfig,
)

__all__ = [
    "DynamicSelectionConfig",
    "ConsensusConfig",
    "StrategySpecificConfig",
    "CombinedStrategyConfig",
    "get_oscillator_data",
]

