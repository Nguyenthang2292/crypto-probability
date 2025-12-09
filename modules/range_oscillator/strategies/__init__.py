"""
Range Oscillator Strategies.

This module provides various signal generation strategies for the Range Oscillator.
"""

from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
from modules.range_oscillator.strategies.sustained import generate_signals_sustained_strategy
from modules.range_oscillator.strategies.crossover import generate_signals_crossover_strategy
from modules.range_oscillator.strategies.momentum import generate_signals_momentum_strategy
from modules.range_oscillator.analysis.combined import generate_signals_combined_all_strategy
from modules.range_oscillator.strategies.breakout import generate_signals_breakout_strategy
from modules.range_oscillator.strategies.divergence import generate_signals_divergence_strategy
from modules.range_oscillator.strategies.trend_following import generate_signals_trend_following_strategy
from modules.range_oscillator.strategies.mean_reversion import generate_signals_mean_reversion_strategy

__all__ = [
    "generate_signals_basic_strategy",
    "generate_signals_sustained_strategy",
    "generate_signals_crossover_strategy",
    "generate_signals_momentum_strategy",
    "generate_signals_combined_all_strategy",
    "generate_signals_breakout_strategy",
    "generate_signals_divergence_strategy",
    "generate_signals_trend_following_strategy",
    "generate_signals_mean_reversion_strategy",
]

