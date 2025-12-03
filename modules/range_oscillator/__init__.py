"""
Range Oscillator module.

This module provides Range Oscillator indicator calculations and signal strategies.
"""

from modules.range_oscillator.range_oscillator import (
    calculate_weighted_ma,
    calculate_atr_range,
    calculate_trend_direction,
    calculate_range_oscillator,
)

from modules.range_oscillator.strategy import (
    generate_signals_strategy1,
    generate_signals_strategy2_sustained,
    generate_signals_strategy3_crossover,
    generate_signals_strategy4_momentum,
    generate_signals_strategy5_combined,
    generate_signals_strategy6_breakout,
    generate_signals_strategy7_divergence,
    generate_signals_strategy8_trend_following,
    generate_signals_strategy9_mean_reversion,
    get_signal_summary,
)

__all__ = [
    # Core calculations
    "calculate_weighted_ma",
    "calculate_atr_range",
    "calculate_trend_direction",
    "calculate_range_oscillator",
    # Signal strategies
    "generate_signals_strategy1",
    "generate_signals_strategy2_sustained",
    "generate_signals_strategy3_crossover",
    "generate_signals_strategy4_momentum",
    "generate_signals_strategy5_combined",
    "generate_signals_strategy6_breakout",
    "generate_signals_strategy7_divergence",
    "generate_signals_strategy8_trend_following",
    "generate_signals_strategy9_mean_reversion",
    "get_signal_summary",
]

