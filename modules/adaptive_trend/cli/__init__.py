"""
Command-line interface components for ATC analysis.

This package provides CLI utilities including argument parsing, interactive prompts,
and formatted display functions.
"""

# Argument parsing
from modules.adaptive_trend.cli.argument_parser import parse_args

# Interactive prompts
from modules.adaptive_trend.cli.interactive_prompts import (
    prompt_timeframe,
    prompt_interactive_mode,
)

# Display utilities
from modules.adaptive_trend.cli.display import (
    display_atc_signals,
    display_scan_results,
    list_futures_symbols,
)

__all__ = [
    # Argument parsing
    'parse_args',
    # Interactive prompts
    'prompt_timeframe',
    'prompt_interactive_mode',
    # Display utilities
    'display_atc_signals',
    'display_scan_results',
    'list_futures_symbols',
]

