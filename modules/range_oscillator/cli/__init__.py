"""
CLI tools for Range Oscillator module.

This module provides command-line interface tools for Range Oscillator operations.
"""

from modules.range_oscillator.cli.argument_parser import parse_args
from modules.range_oscillator.cli.display import (
    display_configuration,
    display_final_results,
)

__all__ = [
    "parse_args",
    "display_configuration",
    "display_final_results",
]

