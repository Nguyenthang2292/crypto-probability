"""
Core computation modules for Adaptive Trend Classification (ATC).

This package provides the core computational functions for ATC:
- Signal computation and processing
- Moving average calculations
- Equity curve calculations
- Symbol analysis and scanning
- Signal detection and generation
"""

from modules.adaptive_trend.core.analyzer import analyze_symbol
from modules.adaptive_trend.core.scanner import scan_all_symbols
from modules.adaptive_trend.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend.core.compute_equity import equity_series
from modules.adaptive_trend.core.compute_moving_averages import (
    calculate_kama_atc,
    ma_calculation,
    set_of_moving_averages,
)
from modules.adaptive_trend.core.process_layer1 import (
    weighted_signal,
    cut_signal,
    trend_sign,
    _layer1_signal_for_ma,
)
from modules.adaptive_trend.core.signal_detection import (
    crossover,
    crossunder,
    generate_signal_from_ma,
)

__all__ = [
    # Analysis
    "analyze_symbol",
    "scan_all_symbols",
    # Signal computation
    "compute_atc_signals",
    # Equity calculations
    "equity_series",
    # Moving averages
    "calculate_kama_atc",
    "ma_calculation",
    "set_of_moving_averages",
    # Layer 1 processing
    "weighted_signal",
    "cut_signal",
    "trend_sign",
    "_layer1_signal_for_ma",
    # Signal detection
    "crossover",
    "crossunder",
    "generate_signal_from_ma",
]

