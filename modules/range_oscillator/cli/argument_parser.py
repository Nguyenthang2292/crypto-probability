"""
Command-line argument parser for ATC + Range Oscillator combined signal filter.

This module provides the main argument parser for the ATC + Range Oscillator CLI,
defining all command-line options and their default values.
"""

import argparse

try:
    from modules.config import DEFAULT_TIMEFRAME
except ImportError:
    DEFAULT_TIMEFRAME = "15m"


def parse_args():
    """Parse command-line arguments for ATC + Range Oscillator combined signal filter."""
    parser = argparse.ArgumentParser(
        description="ATC + Range Oscillator Combined Signal Filter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=DEFAULT_TIMEFRAME,
        help=f"Timeframe for analysis (default: {DEFAULT_TIMEFRAME})",
    )
    parser.add_argument(
        "--no-menu",
        action="store_true",
        help="Disable interactive timeframe menu",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of candles to fetch (default: 500)",
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
        default=0.5,
        dest="lambda_param",
        help="Lambda parameter (default: 0.5)",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.1,
        help="Decay rate (default: 0.1)",
    )
    parser.add_argument(
        "--cutout",
        type=int,
        default=5,
        help="Number of bars to skip at start (default: 5)",
    )
    parser.add_argument(
        "--min-signal",
        type=float,
        default=0.01,
        help="Minimum signal strength to display (default: 0.01)",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Maximum number of symbols to scan (default: None = all)",
    )
    parser.add_argument(
        "--osc-length",
        type=int,
        default=50,
        help="Range Oscillator length parameter (default: 50)",
    )
    parser.add_argument(
        "--osc-mult",
        type=float,
        default=2.0,
        help="Range Oscillator multiplier (default: 2.0)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of parallel workers for Range Oscillator filtering (default: 10)",
    )
    parser.add_argument(
        "--osc-strategies",
        type=int,
        nargs="+",
        default=None,
        help="Range Oscillator strategies to use (e.g., --osc-strategies 5 6 7 8 9). Default: all [5, 6, 7, 8, 9]",
    )
    
    return parser.parse_args()
