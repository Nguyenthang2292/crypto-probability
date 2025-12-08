"""
Range Oscillator Compatibility Layer.

This module provides backward compatibility by re-exporting all strategy functions
from their individual modules. Individual strategies have been moved to separate files:
- strategy1.py: Basic oscillator signals
- strategy2.py: Sustained pressure
- strategy3.py: Zero line crossover
- strategy4.py: Momentum
- strategy5.py: Combined
- strategy6.py: Range breakouts
- strategy7.py: Divergence detection
- strategy8.py: Trend following
- strategy9.py: Mean reversion
- summary.py: Signal summary utilities

This file exists to maintain backward compatibility with existing code that imports
from `modules.range_oscillator._compat`. New code should import directly from
the individual strategy modules in `modules.range_oscillator.strategies` or from 
`modules.range_oscillator`.
"""

# Re-export all strategies for backward compatibility
from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
from modules.range_oscillator.strategies.sustained import generate_signals_strategy2_sustained
from modules.range_oscillator.strategies.crossover import generate_signals_strategy3_crossover
from modules.range_oscillator.strategies.momentum import generate_signals_strategy4_momentum
from modules.range_oscillator.strategies.combined import generate_signals_strategy5_combined
from modules.range_oscillator.strategies.breakout import generate_signals_breakout_strategy
from modules.range_oscillator.strategies.divergence import generate_signals_strategy7_divergence
from modules.range_oscillator.strategies.trend_following import generate_signals_strategy8_trend_following
from modules.range_oscillator.strategies.mean_reversion import generate_signals_strategy9_mean_reversion
from modules.range_oscillator.analysis.summary import get_signal_summary

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
    "get_signal_summary",
]

