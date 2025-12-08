"""
Range Oscillator Strategies.

This module provides various signal generation strategies for the Range Oscillator.
"""

from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
from modules.range_oscillator.strategies.sustained import generate_signals_strategy2_sustained
from modules.range_oscillator.strategies.crossover import generate_signals_strategy3_crossover
from modules.range_oscillator.strategies.momentum import generate_signals_strategy4_momentum
from modules.range_oscillator.strategies.combined import generate_signals_strategy5_combined
from modules.range_oscillator.strategies.breakout import generate_signals_breakout_strategy
from modules.range_oscillator.strategies.divergence import generate_signals_strategy7_divergence
from modules.range_oscillator.strategies.trend_following import generate_signals_strategy8_trend_following
from modules.range_oscillator.strategies.mean_reversion import generate_signals_strategy9_mean_reversion

__all__ = [
    "generate_signals_basic_strategy",
    "generate_signals_strategy2_sustained",
    "generate_signals_strategy3_crossover",
    "generate_signals_strategy4_momentum",
    "generate_signals_strategy5_combined",
    "generate_signals_breakout_strategy",
    "generate_signals_strategy7_divergence",
    "generate_signals_strategy8_trend_following",
    "generate_signals_strategy9_mean_reversion",
]

