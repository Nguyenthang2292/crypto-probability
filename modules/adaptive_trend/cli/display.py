"""
Display utilities for ATC CLI.

This module provides formatted display functions for ATC signals,
scan results, and symbol listings.
"""

import pandas as pd
from typing import Optional

from colorama import Fore, Style

from modules.common.utils import (
    color_text,
    format_price,
    log_error,
    log_warn,
    log_success,
    log_progress,
)

from modules.adaptive_trend.core.process_layer1 import trend_sign


def display_atc_signals(
    symbol: str,
    df: pd.DataFrame,
    atc_results: dict,
    current_price: float,
    exchange_label: str,
):
    """
    Display ATC signals and analysis results.

    Args:
        symbol: Symbol being analyzed
        df: DataFrame with OHLCV data
        atc_results: Dictionary with ATC signal results
        current_price: Current price
        exchange_label: Exchange name label
    """
    average_signal = atc_results.get("Average_Signal")
    if average_signal is None or len(average_signal) == 0:
        log_error("No ATC signals available")
        return

    # Get latest signal values
    latest_signal = average_signal.iloc[-1] if not average_signal.empty else 0.0
    latest_trend = trend_sign(average_signal)
    latest_trend_value = latest_trend.iloc[-1] if not latest_trend.empty else 0

    # Get individual MA signals
    ema_signal = atc_results.get("EMA_Signal", pd.Series())
    hma_signal = atc_results.get("HMA_Signal", pd.Series())
    wma_signal = atc_results.get("WMA_Signal", pd.Series())
    dema_signal = atc_results.get("DEMA_Signal", pd.Series())
    lsma_signal = atc_results.get("LSMA_Signal", pd.Series())
    kama_signal = atc_results.get("KAMA_Signal", pd.Series())

    # Get equity weights
    ema_s = atc_results.get("EMA_S", pd.Series())
    hma_s = atc_results.get("HMA_S", pd.Series())
    wma_s = atc_results.get("WMA_S", pd.Series())
    dema_s = atc_results.get("DEMA_S", pd.Series())
    lsma_s = atc_results.get("LSMA_S", pd.Series())
    kama_s = atc_results.get("KAMA_S", pd.Series())

    # Determine trend direction
    if latest_trend_value > 0:
        trend_direction = "BULLISH"
        trend_color = Fore.GREEN
    elif latest_trend_value < 0:
        trend_direction = "BEARISH"
        trend_color = Fore.RED
    else:
        trend_direction = "NEUTRAL"
        trend_color = Fore.YELLOW

    # Display header
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(
        color_text(
            f"ADAPTIVE TREND CLASSIFICATION (ATC) - {symbol} | {exchange_label}",
            Fore.CYAN,
            Style.BRIGHT,
        )
    )
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))

    # Current price
    print(color_text(f"Current Price: {format_price(current_price)}", Fore.WHITE))
    print(color_text("-" * 80, Fore.CYAN))

    # Average Signal
    print(
        color_text(
            f"Average Signal: {latest_signal:.4f}",
            trend_color,
            Style.BRIGHT,
        )
    )
    print(
        color_text(
            f"Trend Direction: {trend_direction}",
            trend_color,
            Style.BRIGHT,
        )
    )
    print(color_text("-" * 80, Fore.CYAN))

    # Individual MA Signals
    print(color_text("Individual MA Signals:", Fore.MAGENTA, Style.BRIGHT))
    ma_signals = [
        ("EMA", ema_signal),
        ("HMA", hma_signal),
        ("WMA", wma_signal),
        ("DEMA", dema_signal),
        ("LSMA", lsma_signal),
        ("KAMA", kama_signal),
    ]

    for ma_name, ma_sig in ma_signals:
        if not ma_sig.empty:
            latest_ma_sig = ma_sig.iloc[-1]
            ma_trend = trend_sign(ma_sig)
            ma_trend_value = ma_trend.iloc[-1] if not ma_trend.empty else 0

            if ma_trend_value > 0:
                ma_color = Fore.GREEN
                ma_dir = "^"
            elif ma_trend_value < 0:
                ma_color = Fore.RED
                ma_dir = "v"
            else:
                ma_color = Fore.YELLOW
                ma_dir = "-"

            print(
                color_text(
                    f"  {ma_name:6s}: {latest_ma_sig:8.4f} {ma_dir}",
                    ma_color,
                )
            )

    print(color_text("-" * 80, Fore.CYAN))

    # Equity Weights (Layer 2)
    print(color_text("Equity Weights (Layer 2):", Fore.MAGENTA, Style.BRIGHT))
    ma_weights = [
        ("EMA", ema_s),
        ("HMA", hma_s),
        ("WMA", wma_s),
        ("DEMA", dema_s),
        ("LSMA", lsma_s),
        ("KAMA", kama_s),
    ]

    for ma_name, ma_weight in ma_weights:
        if not ma_weight.empty:
            latest_weight = ma_weight.iloc[-1]
            if pd.notna(latest_weight):
                print(
                    color_text(
                        f"  {ma_name:6s}: {latest_weight:8.4f}",
                        Fore.WHITE,
                    )
                )

    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))


def display_scan_results(long_signals: pd.DataFrame, short_signals: pd.DataFrame, min_signal: float):
    """
    Display scan results for LONG and SHORT signals.
    
    Args:
        long_signals: DataFrame with LONG signals
        short_signals: DataFrame with SHORT signals
        min_signal: Minimum signal threshold used
    """
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("ATC SIGNAL SCAN RESULTS", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))

    # LONG Signals
    print("\n" + color_text("LONG SIGNALS (BULLISH)", Fore.GREEN, Style.BRIGHT))
    print(color_text("-" * 80, Fore.CYAN))
    
    if long_signals.empty:
        print(color_text("  No LONG signals found", Fore.YELLOW))
    else:
        print(color_text(f"  Found {len(long_signals)} symbols with LONG signals (min: {min_signal:.4f})", Fore.WHITE))
        print()
        print(color_text(f"{'Symbol':<15} {'Signal':>12} {'Price':>15} {'Exchange':<10}", Fore.MAGENTA))
        print(color_text("-" * 80, Fore.CYAN))
        
        for _, row in long_signals.iterrows():
            signal_str = f"{row['signal']:+.6f}"
            price_str = format_price(row['price'])
            print(
                color_text(
                    f"{row['symbol']:<15} {signal_str:>12} {price_str:>15} {row['exchange']:<10}",
                    Fore.GREEN,
                )
            )

    # SHORT Signals
    print("\n" + color_text("SHORT SIGNALS (BEARISH)", Fore.RED, Style.BRIGHT))
    print(color_text("-" * 80, Fore.CYAN))
    
    if short_signals.empty:
        print(color_text("  No SHORT signals found", Fore.YELLOW))
    else:
        print(color_text(f"  Found {len(short_signals)} symbols with SHORT signals (min: {min_signal:.4f})", Fore.WHITE))
        print()
        print(color_text(f"{'Symbol':<15} {'Signal':>12} {'Price':>15} {'Exchange':<10}", Fore.MAGENTA))
        print(color_text("-" * 80, Fore.CYAN))
        
        for _, row in short_signals.iterrows():
            signal_str = f"{row['signal']:+.6f}"
            price_str = format_price(row['price'])
            print(
                color_text(
                    f"{row['symbol']:<15} {signal_str:>12} {price_str:>15} {row['exchange']:<10}",
                    Fore.RED,
                )
            )

    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text(f"Total: {len(long_signals)} LONG + {len(short_signals)} SHORT = {len(long_signals) + len(short_signals)} signals", Fore.WHITE))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))


def list_futures_symbols(data_fetcher, max_symbols: Optional[int] = None):
    """
    List available futures symbols from Binance.

    Args:
        data_fetcher: DataFetcher instance
        max_symbols: Maximum number of symbols to display
    """
    try:
        log_progress("Fetching futures symbols from Binance...")
        symbols = data_fetcher.list_binance_futures_symbols(
            max_candidates=max_symbols,
            progress_label="Symbol Discovery",
        )

        if not symbols:
            log_error("No symbols found")
            return

        log_success(f"Found {len(symbols)} futures symbols")

        print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
        print(color_text("AVAILABLE FUTURES SYMBOLS", Fore.CYAN, Style.BRIGHT))
        print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))

        # Display symbols in columns
        cols = 4
        for i in range(0, len(symbols), cols):
            row_symbols = symbols[i : i + cols]
            row_text = "  ".join(f"{sym:15s}" for sym in row_symbols)
            print(color_text(row_text, Fore.WHITE))

        print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))

    except Exception as e:
        log_error(f"Error listing symbols: {type(e).__name__}: {e}")

