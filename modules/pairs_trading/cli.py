"""
Command-line interface for pairs trading analysis.
"""

import sys
import argparse
from typing import Dict, Optional

from colorama import Fore, Style

try:
    from modules.common.utils import color_text
except ImportError:
    def color_text(text, color=None, style=None):
        return text

try:
    from modules.config import (
        PAIRS_TRADING_WEIGHTS,
        PAIRS_TRADING_TOP_N,
        PAIRS_TRADING_MIN_SPREAD,
        PAIRS_TRADING_MAX_SPREAD,
        PAIRS_TRADING_MIN_CORRELATION,
        PAIRS_TRADING_MAX_CORRELATION,
        PAIRS_TRADING_MAX_HALF_LIFE,
    )
except ImportError:
    PAIRS_TRADING_WEIGHTS = {"1d": 0.5, "3d": 0.3, "1w": 0.2}
    PAIRS_TRADING_TOP_N = 5
    PAIRS_TRADING_MIN_SPREAD = 0.01
    PAIRS_TRADING_MAX_SPREAD = 0.50
    PAIRS_TRADING_MIN_CORRELATION = 0.3
    PAIRS_TRADING_MAX_CORRELATION = 0.9
    PAIRS_TRADING_MAX_HALF_LIFE = 50


def standardize_symbol_input(symbol: str) -> str:
    """Convert raw user input into f'{base}/USDT' style if needed."""
    if not symbol:
        return ""
    cleaned = symbol.strip().upper()
    if "/" in cleaned:
        base, quote = cleaned.split("/", 1)
        base = base.strip()
        quote = quote.strip() or "USDT"
        return f"{base}/{quote}"
    if cleaned.endswith("USDT"):
        base = cleaned[:-4]
        base = base.strip()
        return f"{base}/USDT"
    return f"{cleaned}/USDT"


def prompt_interactive_mode() -> Dict[str, Optional[str]]:
    """Interactive launcher for selecting analysis mode and symbol source."""
    print(color_text("\n" + "=" * 60, Fore.CYAN, Style.BRIGHT))
    print(color_text("Pairs Trading Analysis - Interactive Launcher", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 60, Fore.CYAN, Style.BRIGHT))
    print("1) Auto mode  - analyze entire market to surface opportunities")
    print("2) Manual mode - focus on specific symbols you provide")
    print("3) Exit")

    while True:
        choice = input(color_text("\nSelect option [1-3]: ", Fore.YELLOW)).strip() or "1"
        if choice in {"1", "2", "3"}:
            break
        print(color_text("Invalid selection. Please enter 1, 2, or 3.", Fore.RED))

    if choice == "3":
        print(color_text("\nExiting...", Fore.YELLOW))
        sys.exit(0)

    manual_symbols = None
    if choice == "2":
        manual_symbols = input(
            color_text(
                "Enter symbols separated by comma/space (e.g., BTC/USDT, ETH/USDT): ",
                Fore.YELLOW,
            )
        ).strip()

    return {
        "mode": "manual" if choice == "2" else "auto",
        "symbols_raw": manual_symbols or None,
    }


def parse_weights(weights_str: Optional[str]) -> Dict[str, float]:
    """Parse weights string into dictionary.
    
    Args:
        weights_str: Weights in format '1d:0.5,3d:0.3,1w:0.2'
        
    Returns:
        Dictionary with weights, normalized to sum to 1.0
    """
    weights = PAIRS_TRADING_WEIGHTS.copy()
    if not weights_str:
        return weights
    
    try:
        weight_parts = weights_str.split(",")
        weights = {}
        for part in weight_parts:
            key, value = part.split(":")
            weights[key.strip()] = float(value.strip())
        # Validate weights sum to 1.0
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            print(
                color_text(
                    f"Warning: Weights sum to {total:.3f}, not 1.0. Normalizing...",
                    Fore.YELLOW,
                )
            )
            weights = {k: v / total for k, v in weights.items()}
    except Exception as e:
        print(
            color_text(
                f"Error parsing weights: {e}. Using default weights.",
                Fore.RED,
            )
        )
        weights = PAIRS_TRADING_WEIGHTS.copy()
    
    return weights


def parse_symbols(symbols_str: Optional[str]):
    """Parse symbols string into display and parsed lists.
    
    Args:
        symbols_str: Comma/space separated symbols
        
    Returns:
        Tuple of (target_symbol_inputs, parsed_target_symbols)
    """
    target_symbol_inputs = []
    parsed_target_symbols = []
    
    if not symbols_str:
        return target_symbol_inputs, parsed_target_symbols
    
    raw_parts = (
        symbols_str.replace(",", " ")
        .replace(";", " ")
        .replace("|", " ")
        .split()
    )
    seen_display = set()
    seen_parsed = set()
    for part in raw_parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        display_value = cleaned.upper()
        parsed_value = standardize_symbol_input(cleaned)
        if display_value not in seen_display:
            seen_display.add(display_value)
            target_symbol_inputs.append(display_value)
        parsed_key = parsed_value.upper()
        if parsed_key not in seen_parsed:
            seen_parsed.add(parsed_key)
            parsed_target_symbols.append(parsed_value)
    
    return target_symbol_inputs, parsed_target_symbols


def parse_args():
    """Parse command-line arguments for pairs trading analysis."""
    parser = argparse.ArgumentParser(
        description="Pairs Trading Analysis - Identify trading opportunities from best/worst performers"
    )
    parser.add_argument(
        "--pairs-count",
        type=int,
        default=PAIRS_TRADING_TOP_N,
        help=f"Number of tradeable pairs to return (default: {PAIRS_TRADING_TOP_N})",
    )
    parser.add_argument(
        "--candidate-depth",
        type=int,
        default=50,
        help="Number of top/bottom symbols to consider per side when forming pairs",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Maximum number of symbols to analyze (default: all available)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Weights for timeframes in format '1d:0.5,3d:0.3,1w:0.2' (default: from config)",
    )
    parser.add_argument(
        "--min-spread",
        type=float,
        default=PAIRS_TRADING_MIN_SPREAD,
        help=f"Minimum spread percentage (default: {PAIRS_TRADING_MIN_SPREAD*100:.2f}%)",
    )
    parser.add_argument(
        "--max-spread",
        type=float,
        default=PAIRS_TRADING_MAX_SPREAD,
        help=f"Maximum spread percentage (default: {PAIRS_TRADING_MAX_SPREAD*100:.2f}%)",
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=PAIRS_TRADING_MIN_CORRELATION,
        help=f"Minimum correlation (default: {PAIRS_TRADING_MIN_CORRELATION:.2f})",
    )
    parser.add_argument(
        "--max-correlation",
        type=float,
        default=PAIRS_TRADING_MAX_CORRELATION,
        help=f"Maximum correlation (default: {PAIRS_TRADING_MAX_CORRELATION:.2f})",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=10,
        help="Maximum number of pairs to display (default: 10)",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip pairs validation (show all opportunities)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Manual mode: comma/space separated symbols to focus on (e.g., 'BTC/USDT,ETH/USDT')",
    )
    parser.add_argument(
        "--no-menu",
        action="store_true",
        help="Skip interactive launcher (retain legacy CLI flag workflow)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        choices=["opportunity_score", "quantitative_score"],
        default="opportunity_score",
        help="Sort pairs by opportunity_score or quantitative_score (default: opportunity_score)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show detailed quantitative metrics (half-life, Sharpe, MaxDD, etc.) in output (default: True)",
    )
    parser.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="Disable detailed quantitative metrics",
    )
    parser.add_argument(
        "--require-cointegration",
        action="store_true",
        help="Only accept cointegrated pairs (filter out non-cointegrated pairs)",
    )
    parser.add_argument(
        "--max-half-life",
        type=float,
        default=PAIRS_TRADING_MAX_HALF_LIFE,
        help=f"Maximum acceptable half-life for mean reversion (default: {PAIRS_TRADING_MAX_HALF_LIFE})",
    )
    parser.add_argument(
        "--min-quantitative-score",
        type=float,
        default=None,
        help="Minimum quantitative score (0-100) to accept a pair (default: no threshold)",
    )
    
    return parser.parse_args()

