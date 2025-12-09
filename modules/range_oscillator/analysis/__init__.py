"""
Analysis tools for Range Oscillator.

This module provides signal analysis and performance evaluation tools.
"""

from modules.range_oscillator.analysis.summary import get_signal_summary
from modules.range_oscillator.analysis.combined import (
    generate_signals_combined_all_strategy,
    CombinedStrategy,
    STRATEGY_FUNCTIONS,
)
from modules.range_oscillator.config import (
    CombinedStrategyConfig,
    ConsensusConfig,
    DynamicSelectionConfig,
    StrategySpecificConfig,
)

__all__ = [
    "get_signal_summary",
    "generate_signals_combined_all_strategy",
    "CombinedStrategy",
    "CombinedStrategyConfig",
    "ConsensusConfig",
    "DynamicSelectionConfig",
    "StrategySpecificConfig",
    "STRATEGY_FUNCTIONS",
]

