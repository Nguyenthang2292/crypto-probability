"""
ATC + Range Oscillator Combined Signal Filter.

This program combines signals from Adaptive Trend Classification (ATC) and Range Oscillator:
1. Runs ATC auto scan to find LONG/SHORT signals
2. Filters symbols by checking if Range Oscillator signals match ATC signals
3. Returns final list of symbols with confirmed signals from both indicators
"""

import warnings
import sys
import threading
from typing import Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from modules.common.utils import configure_windows_stdio

# Fix encoding issues on Windows for interactive CLI runs only
configure_windows_stdio()

from colorama import Fore, Style, init as colorama_init

from modules.config import DEFAULT_TIMEFRAME
from modules.common.utils import (
    color_text,
    log_error,
    log_analysis,
    log_data,
    log_progress,
    log_success,
    log_warn,
    format_price,
)
from modules.common.ExchangeManager import ExchangeManager
from modules.common.DataFetcher import DataFetcher
from modules.adaptive_trend.scanner import scan_all_symbols
from modules.adaptive_trend.cli import prompt_timeframe
from modules.common.utils import prompt_user_input
from modules.range_oscillator.range_oscillator import calculate_range_oscillator
from modules.range_oscillator.strategy import (
    generate_signals_strategy5_combined,
    generate_signals_strategy6_breakout,
    generate_signals_strategy7_divergence,
    generate_signals_strategy8_trend_following,
    generate_signals_strategy9_mean_reversion,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
colorama_init(autoreset=True)


def prompt_oscillator_mode(default_mode: str = "voting") -> str:
    """
    Interactive menu for selecting Range Oscillator signal combination mode.
    
    Args:
        default_mode: Default mode to use ("voting" or "any")
        
    Returns:
        Selected mode string ("voting" or "any")
    """
    print("\n" + color_text("=" * 60, Fore.CYAN))
    print(color_text("RANGE OSCILLATOR SIGNAL MODE SELECTION", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 60, Fore.CYAN))
    
    modes = [
        ("voting", "Voting Mode - Requires consensus from multiple strategies (more conservative, fewer signals)"),
        ("any", "Any Strategy Mode - Accepts signal from any single strategy (more aggressive, more signals)"),
    ]
    
    # Find default index
    default_idx = 0
    for idx, (mode, _) in enumerate(modes):
        if mode == default_mode:
            default_idx = idx
            break
    
    for idx, (mode, desc) in enumerate(modes, 1):
        marker = " <-- default" if mode == default_mode else ""
        print(f"{idx}) {mode.upper():8s} - {desc}{marker}")
    
    print(f"{len(modes) + 1}) Use default ({default_mode})")
    
    while True:
        choice = prompt_user_input(
            f"\nSelect mode [1-{len(modes) + 1}] (default {default_idx + 1}): ",
            default=str(default_idx + 1),
        )
        
        if not choice:
            return default_mode
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(modes):
                return modes[choice_num - 1][0]
            elif choice_num == len(modes) + 1:
                return default_mode
            else:
                log_error(f"Invalid choice. Please enter 1-{len(modes) + 1}.")
        except ValueError:
            log_error("Invalid input. Please enter a number.")


def get_range_oscillator_signal(
    data_fetcher: DataFetcher,
    symbol: str,
    timeframe: str,
    limit: int,
    osc_length: int = 50,
    osc_mult: float = 2.0,
    strategies: Optional[list] = None,
    consensus_threshold: float = 0.5,
    mode: str = "voting",
) -> Optional[int]:
    """
    Calculate Range Oscillator signal for a symbol using multiple strategies.
    
    Args:
        data_fetcher: DataFetcher instance
        symbol: Symbol to analyze
        timeframe: Timeframe for data
        limit: Number of candles
        osc_length: Range Oscillator length parameter
        osc_mult: Range Oscillator multiplier
        strategies: List of strategy numbers to use (e.g., [5, 6, 7, 8, 9]). 
                    If None, uses all strategies [5, 6, 7, 8, 9]
        consensus_threshold: Minimum fraction of strategies that must agree (0.0-1.0, default: 0.5)
                            Only used in "voting" mode
        mode: Signal combination mode:
              - "voting": Voting mechanism (default) - requires consensus threshold
              - "any": Any strategy mode - returns signal if ANY strategy gives LONG/SHORT
        
    Returns:
        Signal value: 1 (LONG), -1 (SHORT), 0 (NEUTRAL), or None if error
    """
    try:
        # Fetch OHLCV data
        df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol,
            limit=limit,
            timeframe=timeframe,
            check_freshness=True,
        )

        if df is None or df.empty:
            return None

        if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns:
            return None

        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Default to all strategies if not specified
        if strategies is None:
            strategies = [5, 6, 7, 8, 9]

        # Calculate signals from each strategy
        strategy_signals = []
        
        for strategy_num in strategies:
            try:
                if strategy_num == 5:
                    signals, _ = generate_signals_strategy5_combined(
                        high=high,
                        low=low,
                        close=close,
                        length=osc_length,
                        mult=osc_mult,
                        use_sustained=True,
                        use_crossover=True,
                        use_momentum=True,
                    )
                elif strategy_num == 6:
                    signals, _ = generate_signals_strategy6_breakout(
                        high=high,
                        low=low,
                        close=close,
                        length=osc_length,
                        mult=osc_mult,
                    )
                elif strategy_num == 7:
                    signals, _ = generate_signals_strategy7_divergence(
                        high=high,
                        low=low,
                        close=close,
                        length=osc_length,
                        mult=osc_mult,
                    )
                elif strategy_num == 8:
                    signals, _ = generate_signals_strategy8_trend_following(
                        high=high,
                        low=low,
                        close=close,
                        length=osc_length,
                        mult=osc_mult,
                    )
                elif strategy_num == 9:
                    signals, _ = generate_signals_strategy9_mean_reversion(
                        high=high,
                        low=low,
                        close=close,
                        length=osc_length,
                        mult=osc_mult,
                    )
                else:
                    continue  # Skip unknown strategy numbers
                
                if signals is not None and not signals.empty:
                    # Get latest signal (last non-NaN value)
                    non_nan_signals = signals.dropna()
                    if len(non_nan_signals) > 0:
                        latest_signal = int(non_nan_signals.iloc[-1])
                        strategy_signals.append(latest_signal)
            except Exception:
                # Skip strategies that fail for this symbol
                continue

        if not strategy_signals:
            return None

        # Mode 1: Voting mechanism (default)
        if mode == "voting":
            # Count LONG and SHORT votes
            long_votes = sum(1 for s in strategy_signals if s == 1)
            short_votes = sum(1 for s in strategy_signals if s == -1)
            total_votes = len(strategy_signals)
            
            # Calculate consensus thresholds
            long_consensus = long_votes / total_votes if total_votes > 0 else 0
            short_consensus = short_votes / total_votes if total_votes > 0 else 0
            
            # Return signal only if consensus threshold is met
            if long_consensus >= consensus_threshold:
                return 1  # LONG
            elif short_consensus >= consensus_threshold:
                return -1  # SHORT
            else:
                return 0  # NEUTRAL (no consensus)
        
        # Mode 2: Any strategy mode - return signal if ANY strategy gives LONG/SHORT
        elif mode == "any":
            # Check if any strategy gives LONG signal
            if any(s == 1 for s in strategy_signals):
                return 1  # LONG (at least one strategy says LONG)
            # Check if any strategy gives SHORT signal
            elif any(s == -1 for s in strategy_signals):
                return -1  # SHORT (at least one strategy says SHORT)
            else:
                return 0  # NEUTRAL (all strategies are neutral or no signals)
        
        else:
            # Invalid mode, fallback to voting
            log_warn(f"Invalid mode '{mode}', using 'voting' mode")
            # Use voting logic directly
            long_votes = sum(1 for s in strategy_signals if s == 1)
            short_votes = sum(1 for s in strategy_signals if s == -1)
            total_votes = len(strategy_signals)
            
            long_consensus = long_votes / total_votes if total_votes > 0 else 0
            short_consensus = short_votes / total_votes if total_votes > 0 else 0
            
            if long_consensus >= consensus_threshold:
                return 1
            elif short_consensus >= consensus_threshold:
                return -1
            else:
                return 0

    except Exception as e:
        # Skip symbols with errors
        return None


def _process_symbol_for_oscillator(
    symbol_data: Dict[str, Any],
    exchange_manager: ExchangeManager,
    timeframe: str,
    limit: int,
    expected_osc_signal: int,
    osc_length: int,
    osc_mult: float,
    strategies: Optional[list] = None,
    consensus_threshold: float = 0.5,
    mode: str = "voting",
) -> Optional[Dict[str, Any]]:
    """
    Worker function to process a single symbol for Range Oscillator confirmation.
    
    This function is designed to be thread-safe by creating its own DataFetcher instance.
    
    Args:
        symbol_data: Dictionary with symbol information (symbol, signal, trend, price, exchange)
        exchange_manager: ExchangeManager instance (shared, thread-safe)
        timeframe: Timeframe for data
        limit: Number of candles
        expected_osc_signal: Expected signal value (1 for LONG, -1 for SHORT)
        osc_length: Range Oscillator length parameter
        osc_mult: Range Oscillator multiplier
        
    Returns:
        Dictionary with confirmed signal data if signals match, None otherwise
    """
    try:
        # Create a new DataFetcher instance for this thread (thread-safe)
        data_fetcher = DataFetcher(exchange_manager)
        
        symbol = symbol_data["symbol"]
        
        # Calculate Range Oscillator signal
        osc_signal = get_range_oscillator_signal(
            data_fetcher=data_fetcher,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            osc_length=osc_length,
            osc_mult=osc_mult,
            strategies=strategies,
            consensus_threshold=consensus_threshold,
            mode=mode,
        )

        # Check if signals match
        if osc_signal == expected_osc_signal:
            # Signals match - return confirmed signal data
            return {
                "symbol": symbol,
                "signal": symbol_data["signal"],
                "trend": symbol_data["trend"],
                "price": symbol_data["price"],
                "exchange": symbol_data["exchange"],
                "osc_signal": osc_signal,
            }
        
        return None
        
    except Exception as e:
        # Skip symbols with errors
        return None


def filter_signals_by_range_oscillator(
    data_fetcher: DataFetcher,
    atc_signals_df: pd.DataFrame,
    timeframe: str,
    limit: int,
    signal_type: str,  # "LONG" or "SHORT"
    osc_length: int = 50,
    osc_mult: float = 2.0,
    max_workers: int = 10,
    strategies: Optional[list] = None,
    consensus_threshold: float = 0.5,
    mode: str = "voting",
) -> pd.DataFrame:
    """
    Filter ATC signals by checking Range Oscillator confirmation using parallel processing.
    
    Args:
        data_fetcher: DataFetcher instance (used to get ExchangeManager)
        atc_signals_df: DataFrame with ATC signals (columns: symbol, signal, trend, price, exchange)
        timeframe: Timeframe for Range Oscillator calculation
        limit: Number of candles
        signal_type: "LONG" or "SHORT"
        osc_length: Range Oscillator length parameter
        osc_mult: Range Oscillator multiplier
        max_workers: Maximum number of parallel workers (default: 10)
        strategies: List of strategy numbers to use (e.g., [5, 6, 7, 8, 9]). 
                   If None, uses all strategies [5, 6, 7, 8, 9]
        consensus_threshold: Minimum fraction of strategies that must agree (0.0-1.0, default: 0.5)
                            Only used in "voting" mode
        mode: Signal combination mode:
              - "voting": Voting mechanism (default) - requires consensus threshold
              - "any": Any strategy mode - returns signal if ANY strategy gives LONG/SHORT
        
    Returns:
        DataFrame with filtered signals that match Range Oscillator
    """
    if atc_signals_df.empty:
        return pd.DataFrame()

    expected_osc_signal = 1 if signal_type == "LONG" else -1
    total = len(atc_signals_df)
    
    strategies_str = ", ".join(map(str, strategies)) if strategies else "5, 6, 7, 8, 9 (all)"
    mode_str = f"mode: {mode}" + (f", consensus: {consensus_threshold:.0%}" if mode == "voting" else "")
    log_progress(
        f"Checking Range Oscillator signals for {total} {signal_type} symbols "
        f"(strategies: {strategies_str}, {mode_str}, workers: {max_workers})..."
    )

    # Get ExchangeManager from DataFetcher (shared, thread-safe)
    exchange_manager = data_fetcher.exchange_manager
    
    # Convert DataFrame rows to list of dictionaries for parallel processing
    symbol_data_list = [
        {
            "symbol": row["symbol"],
            "signal": row["signal"],
            "trend": row["trend"],
            "price": row["price"],
            "exchange": row["exchange"],
        }
        for _, row in atc_signals_df.iterrows()
    ]

    # Thread-safe progress tracking
    progress_lock = threading.Lock()
    checked_count = [0]  # Use list to allow modification in nested function
    confirmed_count = [0]

    # Process symbols in parallel using ThreadPoolExecutor
    filtered_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(
                _process_symbol_for_oscillator,
                symbol_data,
                exchange_manager,
                timeframe,
                limit,
                expected_osc_signal,
                osc_length,
                osc_mult,
                strategies,
                consensus_threshold,
                mode,
            ): symbol_data["symbol"]
            for symbol_data in symbol_data_list
        }

        # Process completed tasks as they finish
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result is not None:
                    with progress_lock:
                        confirmed_count[0] += 1
                        filtered_results.append(result)
            except Exception as e:
                # Skip symbols with errors
                pass
            finally:
                # Update progress (thread-safe)
                with progress_lock:
                    checked_count[0] += 1
                    current_checked = checked_count[0]
                    current_confirmed = confirmed_count[0]
                    
                    # Update progress every 10 symbols or at completion
                    if current_checked % 10 == 0 or current_checked == total:
                        log_progress(
                            f"Checked {current_checked}/{total} symbols... "
                            f"Found {current_confirmed} confirmed {signal_type} signals"
                        )

    if not filtered_results:
        return pd.DataFrame()

    filtered_df = pd.DataFrame(filtered_results)
    
    # Sort by signal strength (absolute value)
    if signal_type == "LONG":
        filtered_df = filtered_df.sort_values("signal", ascending=False).reset_index(drop=True)
    else:
        filtered_df = filtered_df.sort_values("signal", ascending=True).reset_index(drop=True)

    return filtered_df


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
        print(color_text(f"{'Symbol':<15} {'ATC Signal':>12} {'Price':>15} {'Exchange':<10}", Fore.MAGENTA))
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
    print("\n" + color_text("CONFIRMED SHORT SIGNALS (BEARISH)", Fore.RED, Style.BRIGHT))
    print(color_text("-" * 80, Fore.CYAN))
    
    if short_signals.empty:
        print(color_text("  No confirmed SHORT signals found", Fore.YELLOW))
    else:
        print(color_text(f"  Found {len(short_signals)} confirmed SHORT signals (from {original_short_count} ATC signals)", Fore.WHITE))
        print()
        print(color_text(f"{'Symbol':<15} {'ATC Signal':>12} {'Price':>15} {'Exchange':<10}", Fore.MAGENTA))
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
    print(color_text(f"Summary:", Fore.WHITE, Style.BRIGHT))
    print(color_text(f"  ATC Signals: {original_long_count} LONG + {original_short_count} SHORT = {original_long_count + original_short_count}", Fore.WHITE))
    print(color_text(f"  Confirmed Signals: {len(long_signals)} LONG + {len(short_signals)} SHORT = {len(long_signals) + len(short_signals)}", Fore.WHITE, Style.BRIGHT))
    print(color_text(f"  Confirmation Rate: {(len(long_signals) + len(short_signals)) / (original_long_count + original_short_count) * 100:.1f}%" if (original_long_count + original_short_count) > 0 else "N/A", Fore.YELLOW))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))


def main() -> None:
    """
    Main function for ATC + Range Oscillator combined signal filtering.
    
    Workflow:
    1. Run ATC auto scan to get LONG/SHORT signals
    2. For each symbol with ATC signal, check Range Oscillator signal
    3. Filter to keep only symbols where both indicators agree
    4. Display final confirmed signals
    """
    import argparse
    
    # Parse command-line arguments
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
    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.5,
        help="Minimum fraction of strategies that must agree (0.0-1.0, default: 0.5 = 50%%). Only used in 'voting' mode",
    )
    parser.add_argument(
        "--osc-mode",
        type=str,
        choices=["voting", "any"],
        default="voting",
        help="Signal combination mode: 'voting' (requires consensus) or 'any' (any strategy gives signal). Default: voting",
    )
    
    args = parser.parse_args()
    
    # Configuration from arguments
    timeframe = args.timeframe
    osc_mode = args.osc_mode  # Initialize osc_mode from args first
    
    # Prompt for timeframe and oscillator mode selection if menu is enabled
    if not args.no_menu:
        print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
        print(color_text("ATC PHASE - TIMEFRAME SELECTION", Fore.CYAN, Style.BRIGHT))
        print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
        timeframe = prompt_timeframe(default_timeframe=timeframe)
        print(color_text(f"\nSelected timeframe for ATC analysis: {timeframe}", Fore.GREEN))
        
        # Prompt for oscillator mode
        osc_mode = prompt_oscillator_mode(default_mode=osc_mode)
        print(color_text(f"Selected Range Oscillator mode: {osc_mode}", Fore.GREEN))
    limit = args.limit
    ema_len = args.ema_len
    hma_len = args.hma_len
    wma_len = args.wma_len
    dema_len = args.dema_len
    lsma_len = args.lsma_len
    kama_len = args.kama_len
    robustness = args.robustness
    lambda_param = args.lambda_param
    decay = args.decay
    cutout = args.cutout
    min_signal = args.min_signal
    max_symbols = args.max_symbols
    osc_length = args.osc_length
    osc_mult = args.osc_mult
    max_workers = args.max_workers
    strategies = args.osc_strategies
    consensus_threshold = args.consensus_threshold
    # osc_mode already initialized above (line 699)

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
    print(color_text(f"  Oscillator Mode: {osc_mode}", Fore.WHITE))
    if osc_mode == "voting":
        print(color_text(f"  Consensus Threshold: {consensus_threshold:.0%}", Fore.WHITE))
    if max_symbols:
        print(color_text(f"  Max Symbols: {max_symbols}", Fore.WHITE))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))

    # Initialize components
    log_progress("Initializing components...")
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)

    # Step 1: Run ATC auto scan
    log_progress("\nStep 1: Running ATC auto scan...")
    log_progress("=" * 80)
    
    long_signals_atc, short_signals_atc = scan_all_symbols(
        data_fetcher=data_fetcher,
        timeframe=timeframe,
        limit=limit,
        ema_len=ema_len,
        hma_len=hma_len,
        wma_len=wma_len,
        dema_len=dema_len,
        lsma_len=lsma_len,
        kama_len=kama_len,
        robustness=robustness,
        lambda_param=lambda_param,
        decay=decay,
        cutout=cutout,
        max_symbols=max_symbols,
        min_signal=min_signal,
    )

    original_long_count = len(long_signals_atc)
    original_short_count = len(short_signals_atc)

    log_success(f"\nATC Scan Complete: Found {original_long_count} LONG + {original_short_count} SHORT signals")

    if long_signals_atc.empty and short_signals_atc.empty:
        log_warn("No ATC signals found. Exiting.")
        return

    # Step 2: Filter by Range Oscillator confirmation
    log_progress("\nStep 2: Filtering by Range Oscillator confirmation...")
    log_progress("=" * 80)

    # Filter LONG signals (parallel processing)
    long_signals_confirmed = pd.DataFrame()
    if not long_signals_atc.empty:
        long_signals_confirmed = filter_signals_by_range_oscillator(
            data_fetcher=data_fetcher,
            atc_signals_df=long_signals_atc,
            timeframe=timeframe,
            limit=limit,
            signal_type="LONG",
            osc_length=osc_length,
            osc_mult=osc_mult,
            max_workers=max_workers,
            strategies=strategies,
            consensus_threshold=consensus_threshold,
            mode=osc_mode,
        )

    # Filter SHORT signals (parallel processing)
    short_signals_confirmed = pd.DataFrame()
    if not short_signals_atc.empty:
        short_signals_confirmed = filter_signals_by_range_oscillator(
            data_fetcher=data_fetcher,
            atc_signals_df=short_signals_atc,
            timeframe=timeframe,
            limit=limit,
            signal_type="SHORT",
            osc_length=osc_length,
            osc_mult=osc_mult,
            max_workers=max_workers,
            strategies=strategies,
            consensus_threshold=consensus_threshold,
            mode=osc_mode,
        )

    # Step 3: Display final results
    log_progress("\nStep 3: Displaying final results...")
    display_final_results(
        long_signals=long_signals_confirmed,
        short_signals=short_signals_confirmed,
        original_long_count=original_long_count,
        original_short_count=original_short_count,
    )

    log_success("\nAnalysis complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(color_text("\nExiting program by user request.", Fore.YELLOW))
        sys.exit(0)
    except Exception as e:
        log_error(f"Error: {type(e).__name__}: {e}")
        import traceback
        log_error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

