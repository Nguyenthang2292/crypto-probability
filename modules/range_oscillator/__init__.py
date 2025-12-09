"""
Range Oscillator module.

This module provides Range Oscillator indicator calculations and signal strategies.
"""

from modules.range_oscillator.core.weighted_ma import calculate_weighted_ma
from modules.range_oscillator.core.atr_range import calculate_atr_range
from modules.range_oscillator.core.trend_direction import calculate_trend_direction
from modules.range_oscillator.core.oscillator import calculate_range_oscillator

from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
from modules.range_oscillator.strategies.sustained import generate_signals_sustained_strategy
from modules.range_oscillator.strategies.crossover import generate_signals_crossover_strategy
from modules.range_oscillator.strategies.momentum import generate_signals_momentum_strategy
from modules.range_oscillator.analysis.combined import generate_signals_combined_all_strategy
from modules.range_oscillator.strategies.breakout import generate_signals_breakout_strategy
from modules.range_oscillator.strategies.divergence import generate_signals_divergence_strategy
from modules.range_oscillator.strategies.trend_following import generate_signals_trend_following_strategy
from modules.range_oscillator.strategies.mean_reversion import generate_signals_mean_reversion_strategy
from modules.range_oscillator.analysis.summary import get_signal_summary

__all__ = [
    # Core calculations
    "calculate_weighted_ma",
    "calculate_atr_range",
    "calculate_trend_direction",
    "calculate_range_oscillator",
    # Signal strategies
    "generate_signals_basic_strategy",
    "generate_signals_sustained_strategy",
    "generate_signals_crossover_strategy",
    "generate_signals_momentum_strategy",
    "generate_signals_combined_all_strategy",
    "generate_signals_breakout_strategy",
    "generate_signals_divergence_strategy",
    "generate_signals_trend_following_strategy",
    "generate_signals_mean_reversion_strategy",
    "get_signal_summary",
]
