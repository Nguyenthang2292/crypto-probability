"""
Adaptive Trend Classification (ATC) Module.

This module provides Adaptive Trend Classification analysis for cryptocurrency trading,
including signal computation, symbol analysis, scanning, and CLI utilities.

Core Components:
- Signal Computation: Calculate ATC signals from price data
- Symbol Analysis: Analyze individual symbols
- Scanner: Scan multiple symbols for trading signals
- Configuration: ATCConfig and configuration utilities
- CLI: Command-line interface components
"""

# Core analysis functions
from modules.adaptive_trend.core.analyzer import analyze_symbol
from modules.adaptive_trend.core.scanner import scan_all_symbols
from modules.adaptive_trend.core.compute_atc_signals import compute_atc_signals

# Configuration
from modules.adaptive_trend.utils.config import (
    ATCConfig,
    create_atc_config_from_dict,
)

# Utility functions
from modules.adaptive_trend.utils import (
    rate_of_change,
    diflen,
    exp_growth,
)

# CLI components
from modules.adaptive_trend.cli import (
    parse_args,
    prompt_timeframe,
    prompt_interactive_mode,
    display_atc_signals,
    display_scan_results,
    list_futures_symbols,
)

__all__ = [
    # Core analysis
    "analyze_symbol",
    "scan_all_symbols",
    "compute_atc_signals",
    # Configuration
    "ATCConfig",
    "create_atc_config_from_dict",
    # Utilities
    "rate_of_change",
    "diflen",
    "exp_growth",
    # CLI
    "parse_args",
    "prompt_timeframe",
    "prompt_interactive_mode",
    "display_atc_signals",
    "display_scan_results",
    "list_futures_symbols",
]

