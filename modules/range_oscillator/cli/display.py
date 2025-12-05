"""
Display utilities for ATC + Range Oscillator CLI.

This module provides formatted display functions for combined signal results,
configuration, and summary information.
"""

import pandas as pd
from typing import Optional

from colorama import Fore, Style

from modules.common.utils import (
    color_text,
    format_price,
    log_progress,
)


def display_configuration(
    timeframe: str,
    limit: int,
    min_signal: float,
    max_workers: int,
    strategies: Optional[list],
    max_symbols: Optional[int] = None,
):
    """
    Display configuration information.
    
    Args:
        timeframe: Selected timeframe
        limit: Number of candles
        min_signal: Minimum signal strength
        max_workers: Number of parallel workers
        strategies: List of strategy numbers
        max_symbols: Maximum number of symbols to scan (optional)
    """
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("ATC + RANGE OSCILLATOR COMBINED SIGNAL FILTER", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("Configuration:", Fore.WHITE))
    print(color_text(f"  Timeframe: {timeframe}", Fore.WHITE))
    print(color_text(f"  Limit: {limit} candles", Fore.WHITE))
    print(color_text(f"  Min Signal: {min_signal}", Fore.WHITE))
    print(color_text(f"  Parallel Workers: {max_workers}", Fore.WHITE))
    strategies_str = ", ".join(map(str, strategies)) if strategies else "5, 6, 7, 8, 9 (all)"
    print(color_text(f"  Oscillator Strategies: {strategies_str}", Fore.WHITE))
    print(color_text(f"  Oscillator Mode: Any Strategy Mode", Fore.WHITE))
    if max_symbols:
        print(color_text(f"  Max Symbols: {max_symbols}", Fore.WHITE))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))


def display_final_results(
    long_signals: pd.DataFrame,
    short_signals: pd.DataFrame,
    original_long_count: int,
    original_short_count: int,
):
    """
    Display final filtered results.
    
    Args:
        long_signals: Filtered LONG signals DataFrame
        short_signals: Filtered SHORT signals DataFrame
        original_long_count: Original number of LONG signals from ATC
        original_short_count: Original number of SHORT signals from ATC
    """
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("FINAL CONFIRMED SIGNALS (ATC + Range Oscillator)", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))

    # LONG Signals
    print("\n" + color_text("CONFIRMED LONG SIGNALS (BULLISH)", Fore.GREEN, Style.BRIGHT))
    print(color_text("-" * 80, Fore.CYAN))
    
    if long_signals.empty:
        print(color_text("  No confirmed LONG signals found", Fore.YELLOW))
    else:
        print(color_text(f"  Found {len(long_signals)} confirmed LONG signals (from {original_long_count} ATC signals)", Fore.WHITE))
        print()
        # Check if osc_strategies column exists
        has_strategies = 'osc_strategies' in long_signals.columns
        if has_strategies:
            print(color_text(f"{'Symbol':<15} {'ATC Signal':>12} {'Price':>15} {'Exchange':<10} {'Strategy':<12}", Fore.MAGENTA))
        else:
            print(color_text(f"{'Symbol':<15} {'ATC Signal':>12} {'Price':>15} {'Exchange':<10}", Fore.MAGENTA))
        print(color_text("-" * 80, Fore.CYAN))
        
        for _, row in long_signals.iterrows():
            signal_str = f"{row['signal']:+.6f}"
            price_str = format_price(row['price'])
            if has_strategies:
                strategies = row.get('osc_strategies', [])
                if isinstance(strategies, list) and strategies:
                    strategy_str = ",".join(map(str, strategies))
                else:
                    strategy_str = "N/A"
                print(
                    color_text(
                        f"{row['symbol']:<15} {signal_str:>12} {price_str:>15} {row['exchange']:<10} {strategy_str:<12}",
                        Fore.GREEN,
                    )
                )
            else:
                print(
                    color_text(
                        f"{row['symbol']:<15} {signal_str:>12} {price_str:>15} {row['exchange']:<10}",
                        Fore.GREEN,
                    )
                )

    # SHORT Signals
    print("\n" + color_text("CONFIRMED SHORT SIGNALS (BEARISH)", Fore.RED, Style.BRIGHT))
    print(color_text("-" * 80, Fore.CYAN))
    
    if short_signals.empty:
        print(color_text("  No confirmed SHORT signals found", Fore.YELLOW))
    else:
        print(color_text(f"  Found {len(short_signals)} confirmed SHORT signals (from {original_short_count} ATC signals)", Fore.WHITE))
        print()
        # Check if osc_strategies column exists
        has_strategies = 'osc_strategies' in short_signals.columns
        if has_strategies:
            print(color_text(f"{'Symbol':<15} {'ATC Signal':>12} {'Price':>15} {'Exchange':<10} {'Strategy':<12}", Fore.MAGENTA))
        else:
            print(color_text(f"{'Symbol':<15} {'ATC Signal':>12} {'Price':>15} {'Exchange':<10}", Fore.MAGENTA))
        print(color_text("-" * 80, Fore.CYAN))
        
        for _, row in short_signals.iterrows():
            signal_str = f"{row['signal']:+.6f}"
            price_str = format_price(row['price'])
            if has_strategies:
                strategies = row.get('osc_strategies', [])
                if isinstance(strategies, list) and strategies:
                    strategy_str = ",".join(map(str, strategies))
                else:
                    strategy_str = "N/A"
                print(
                    color_text(
                        f"{row['symbol']:<15} {signal_str:>12} {price_str:>15} {row['exchange']:<10} {strategy_str:<12}",
                        Fore.RED,
                    )
                )
            else:
                print(
                    color_text(
                        f"{row['symbol']:<15} {signal_str:>12} {price_str:>15} {row['exchange']:<10}",
                        Fore.RED,
                    )
                )

    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text(f"Summary:", Fore.WHITE, Style.BRIGHT))
    print(color_text(f"  ATC Signals: {original_long_count} LONG + {original_short_count} SHORT = {original_long_count + original_short_count}", Fore.WHITE))
    print(color_text(f"  Confirmed Signals: {len(long_signals)} LONG + {len(short_signals)} SHORT = {len(long_signals) + len(short_signals)}", Fore.WHITE, Style.BRIGHT))
    print(color_text(f"  Confirmation Rate: {(len(long_signals) + len(short_signals)) / (original_long_count + original_short_count) * 100:.1f}%" if (original_long_count + original_short_count) > 0 else "N/A", Fore.YELLOW))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
