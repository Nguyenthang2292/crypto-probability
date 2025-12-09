"""
Configuration classes for Range Oscillator.

This module provides configuration classes for the Range Oscillator combined strategy,
including consensus settings, dynamic selection, strategy-specific parameters, and
the main combined strategy configuration.
"""

from .dynamic_selection_config import DynamicSelectionConfig
from .consensus_config import ConsensusConfig
from .strategy_specific_config import StrategySpecificConfig
from .strategy_combine_config import CombinedStrategyConfig

__all__ = [
    "DynamicSelectionConfig",
    "ConsensusConfig",
    "StrategySpecificConfig",
    "CombinedStrategyConfig",
]

