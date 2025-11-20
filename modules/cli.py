"""
Command-line interface and input handling functions.
"""
import argparse
from .utils import color_text
from colorama import Fore
from .config import (
    DEFAULT_SYMBOL,
    DEFAULT_QUOTE,
    DEFAULT_TIMEFRAME,
    DEFAULT_LIMIT,
    DEFAULT_EXCHANGE_STRING,
)


def prompt_with_default(message: str, default, cast=str):
    """Prompts user for input with a default value."""
    while True:
        raw = input(color_text(f"{message} (default {default}): ", Fore.CYAN))
        value = raw.strip()
        if not value:
            return default
        try:
            return cast(value)
        except ValueError:
            print(color_text("Invalid input. Please try again.", Fore.RED))


def resolve_input(cli_value, default, prompt_message, cast=str, allow_prompt=True):
    """Resolves input from CLI argument or user prompt."""
    if cli_value is not None:
        return cast(cli_value)
    if allow_prompt:
        return prompt_with_default(prompt_message, default, cast)
    return default


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Crypto movement predictor using technical indicators and XGBoost."
    )
    parser.add_argument(
        "-s",
        "--symbol",
        help=f"Trading pair symbol (default: {DEFAULT_SYMBOL}). Accepts formats like 'BTC/USDT' or 'btc'.",
    )
    parser.add_argument(
        "-q",
        "--quote",
        help=f"Quote currency when symbol is given without slash (default: {DEFAULT_QUOTE}).",
    )
    parser.add_argument(
        "-t",
        "--timeframe",
        help=f"Timeframe for OHLCV data (default: {DEFAULT_TIMEFRAME}, e.g., 30m, 1h, 4h).",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        help=f"Number of candles to fetch (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "-e",
        "--exchanges",
        help=f"Comma-separated list of exchanges to try (default: {DEFAULT_EXCHANGE_STRING}).",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disable interactive prompts; rely only on CLI arguments.",
    )
    return parser.parse_args()

