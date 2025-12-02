"""
Command-line argument parser for ATC analysis.

This module provides the main argument parser for the ATC CLI,
defining all command-line options and their default values.
"""

import argparse

try:
    from modules.config import (
        DEFAULT_SYMBOL,
        DEFAULT_QUOTE,
        DEFAULT_TIMEFRAME,
        DEFAULT_LIMIT,
    )
except ImportError:
    DEFAULT_SYMBOL = "BTC/USDT"
    DEFAULT_QUOTE = "USDT"
    DEFAULT_TIMEFRAME = "1h"
    DEFAULT_LIMIT = 1500


def parse_args():
    """Parse command-line arguments for ATC analysis."""
    parser = argparse.ArgumentParser(
        description="Adaptive Trend Classification (ATC) Analysis for Binance Futures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help=f"Symbol pair to analyze (default: {DEFAULT_SYMBOL})",
    )
    parser.add_argument(
        "--quote",
        type=str,
        default=DEFAULT_QUOTE,
        help=f"Quote currency (default: {DEFAULT_QUOTE})",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=DEFAULT_TIMEFRAME,
        help=f"Timeframe for analysis (default: {DEFAULT_TIMEFRAME})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of candles to fetch (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--ema-len",
        type=int,
        default=28,
        help="EMA length (default: 28)",
    )
    parser.add_argument(
        "--hma-len",
        type=int,
        default=28,
        help="HMA length (default: 28)",
    )
    parser.add_argument(
        "--wma-len",
        type=int,
        default=28,
        help="WMA length (default: 28)",
    )
    parser.add_argument(
        "--dema-len",
        type=int,
        default=28,
        help="DEMA length (default: 28)",
    )
    parser.add_argument(
        "--lsma-len",
        type=int,
        default=28,
        help="LSMA length (default: 28)",
    )
    parser.add_argument(
        "--kama-len",
        type=int,
        default=28,
        help="KAMA length (default: 28)",
    )
    parser.add_argument(
        "--robustness",
        type=str,
        choices=["Narrow", "Medium", "Wide"],
        default="Medium",
        help="Robustness setting (default: Medium)",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=0.02,
        dest="lambda_param",
        help="Lambda parameter for exponential growth (default: 0.02)",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.03,
        help="Decay rate (default: 0.03)",
    )
    parser.add_argument(
        "--cutout",
        type=int,
        default=0,
        help="Number of bars to skip at start (default: 0)",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disable interactive prompts",
    )
    parser.add_argument(
        "--no-menu",
        action="store_true",
        help="Disable interactive menu",
    )
    parser.add_argument(
        "--list-symbols",
        action="store_true",
        help="List available futures symbols and exit",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Maximum number of symbols to scan in auto mode",
    )
    parser.add_argument(
        "--min-signal",
        type=float,
        default=0.01,
        help="Minimum signal strength to display (default: 0.01)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Force auto mode (scan all symbols)",
    )

    return parser.parse_args()

