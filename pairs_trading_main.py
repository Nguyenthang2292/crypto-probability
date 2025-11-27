"""
Pairs Trading Analysis Main Program

Analyzes futures pairs on Binance to identify pairs trading opportunities:
- Loads 1h candle data from all futures pairs
- Calculates top 5 best and worst performers
- Identifies pairs trading opportunities (long worst, short best)
"""

import warnings
import sys
import io

# Fix encoding issues on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from colorama import Fore, Style, init as colorama_init

from modules.config import (
    PAIRS_TRADING_WEIGHTS,
    PAIRS_TRADING_TOP_N,
    PAIRS_TRADING_MIN_VOLUME,
    PAIRS_TRADING_MIN_SPREAD,
    PAIRS_TRADING_MAX_SPREAD,
    PAIRS_TRADING_MIN_CORRELATION,
    PAIRS_TRADING_MAX_CORRELATION,
)
from modules.common.utils import color_text, format_price
from modules.common.ExchangeManager import ExchangeManager
from modules.common.DataFetcher import DataFetcher
from modules.pairs_trading.performance_analyzer import PerformanceAnalyzer
from modules.pairs_trading.pairs_analyzer import PairsTradingAnalyzer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
colorama_init(autoreset=True)


def display_performers(df, title, color):
    """Display top/worst performers in a formatted table."""
    if df is None or df.empty:
        print(color_text(f"No {title.lower()} found.", Fore.YELLOW))
        return

    print(color_text(f"\n{'=' * 80}", color, Style.BRIGHT))
    print(color_text(f"{title}", color, Style.BRIGHT))
    print(color_text(f"{'=' * 80}", color, Style.BRIGHT))

    print(
        f"{'Rank':<6} {'Symbol':<15} {'Score':<12} {'1d Return':<12} {'3d Return':<12} {'1w Return':<12} {'Price':<15}"
    )
    print("-" * 80)

    for idx, row in df.iterrows():
        rank = idx + 1
        symbol = row["symbol"]
        score = row["score"] * 100
        return_1d = row["1d_return"] * 100
        return_3d = row["3d_return"] * 100
        return_1w = row["1w_return"] * 100
        price = row["current_price"]

        score_color = Fore.GREEN if score > 0 else Fore.RED
        print(
            f"{rank:<6} {symbol:<15} "
            f"{color_text(f'{score:+.2f}%', score_color):<20} "
            f"{return_1d:+.2f}%{'':<6} {return_3d:+.2f}%{'':<6} {return_1w:+.2f}%{'':<6} {format_price(price):<15}"
        )

    print(color_text(f"{'=' * 80}", color, Style.BRIGHT))


def display_pairs_opportunities(pairs_df, max_display=10):
    """Display pairs trading opportunities in a formatted table."""
    import pandas as pd

    if pairs_df is None or pairs_df.empty:
        print(color_text("\nNo pairs trading opportunities found.", Fore.YELLOW))
        return

    print(color_text(f"\n{'=' * 100}", Fore.MAGENTA, Style.BRIGHT))
    print(color_text("PAIRS TRADING OPPORTUNITIES", Fore.MAGENTA, Style.BRIGHT))
    print(color_text(f"{'=' * 100}", Fore.MAGENTA, Style.BRIGHT))

    print(
        f"{'Rank':<6} {'Long (Worst)':<18} {'Short (Best)':<18} {'Spread':<12} {'Correlation':<15} {'Score':<12}"
    )
    print("-" * 100)

    display_count = min(len(pairs_df), max_display)
    for idx in range(display_count):
        row = pairs_df.iloc[idx]
        rank = idx + 1
        long_symbol = row["long_symbol"]
        short_symbol = row["short_symbol"]
        spread = row["spread"] * 100
        correlation = row.get("correlation")
        opportunity_score = row["opportunity_score"] * 100

        # Color code based on opportunity score
        if opportunity_score > 20:
            score_color = Fore.GREEN
        elif opportunity_score > 10:
            score_color = Fore.YELLOW
        else:
            score_color = Fore.WHITE

        # Color code correlation
        if correlation is not None and not pd.isna(correlation):
            abs_corr = abs(correlation)
            if abs_corr > 0.7:
                corr_color = Fore.GREEN
            elif abs_corr > 0.4:
                corr_color = Fore.YELLOW
            else:
                corr_color = Fore.RED
            corr_text = f"{correlation:+.3f}"
        else:
            corr_color = Fore.WHITE
            corr_text = "N/A"

        print(
            f"{rank:<6} {long_symbol:<18} {short_symbol:<18} "
            f"{spread:+.2f}%{'':<6} {color_text(corr_text, corr_color):<20} "
            f"{color_text(f'{opportunity_score:+.2f}%', score_color):<20}"
        )

    print(color_text(f"{'=' * 100}", Fore.MAGENTA, Style.BRIGHT))

    if len(pairs_df) > max_display:
        print(
            color_text(
                f"\nShowing top {max_display} of {len(pairs_df)} opportunities. "
                f"Use --max-pairs to see more.",
                Fore.CYAN,
            )
        )


def main():
    """Main function for pairs trading analysis."""
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Pairs Trading Analysis - Identify trading opportunities from best/worst performers"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=PAIRS_TRADING_TOP_N,
        help=f"Number of top/bottom performers to analyze (default: {PAIRS_TRADING_TOP_N})",
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
        "--min-volume",
        type=float,
        default=PAIRS_TRADING_MIN_VOLUME,
        help=f"Minimum volume in USDT (default: {PAIRS_TRADING_MIN_VOLUME:,.0f})",
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
        "--use-hardcoded-symbols",
        action="store_true",
        help="Use hardcoded popular symbols instead of fetching from Binance (for testing without API)",
    )

    args = parser.parse_args()

    # Parse weights if provided
    weights = PAIRS_TRADING_WEIGHTS.copy()
    if args.weights:
        try:
            weight_parts = args.weights.split(",")
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

    print(color_text("\n" + "=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("PAIRS TRADING ANALYSIS", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(
        color_text(
            f"\nConfiguration:", Fore.CYAN,
        )
    )
    print(f"  Top N performers: {args.top_n}")
    print(f"  Weights: 1d={weights['1d']:.2f}, 3d={weights['3d']:.2f}, 1w={weights['1w']:.2f}")
    print(f"  Min volume: {args.min_volume:,.0f} USDT")
    print(f"  Spread range: {args.min_spread*100:.2f}% - {args.max_spread*100:.2f}%")
    print(f"  Correlation range: {args.min_correlation:.2f} - {args.max_correlation:.2f}")

    # Initialize components
    print(color_text("\nInitializing components...", Fore.YELLOW))
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)
    performance_analyzer = PerformanceAnalyzer(weights=weights)
    pairs_analyzer = PairsTradingAnalyzer(
        min_volume=args.min_volume,
        min_spread=args.min_spread,
        max_spread=args.max_spread,
        min_correlation=args.min_correlation,
        max_correlation=args.max_correlation,
    )

    # Step 1: Get list of futures symbols
    print(color_text("\n[1/4] Fetching futures symbols from Binance...", Fore.CYAN, Style.BRIGHT))
    
    if args.use_hardcoded_symbols:
        # Use hardcoded popular symbols for testing without API
        symbols = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
            "ADA/USDT", "DOGE/USDT", "DOT/USDT", "MATIC/USDT", "AVAX/USDT",
            "LINK/USDT", "UNI/USDT", "ATOM/USDT", "ETC/USDT", "LTC/USDT",
            "NEAR/USDT", "ALGO/USDT", "FIL/USDT", "APT/USDT", "ARB/USDT",
        ]
        if args.max_symbols:
            symbols = symbols[:args.max_symbols]
        print(color_text(f"Using {len(symbols)} hardcoded symbols for testing.", Fore.YELLOW))
    else:
        try:
            symbols = data_fetcher.list_binance_futures_symbols(
                max_candidates=args.max_symbols,
                progress_label="Symbol Discovery",
            )
            if not symbols:
                print(
                    color_text(
                        "No symbols found. Please check your API connection.\n"
                        "Tip: Use --use-hardcoded-symbols to test with popular symbols.",
                        Fore.RED,
                        Style.BRIGHT,
                    )
                )
                return
            print(color_text(f"Found {len(symbols)} futures symbols.", Fore.GREEN))
        except Exception as e:
            print(
                color_text(
                    f"Error fetching symbols: {e}\n"
                    "Tip: Use --use-hardcoded-symbols to test with popular symbols.",
                    Fore.RED,
                    Style.BRIGHT,
                )
            )
            return

    # Step 2: Analyze performance
    print(
        color_text(
            f"\n[2/4] Analyzing performance for {len(symbols)} symbols...",
            Fore.CYAN,
            Style.BRIGHT,
        )
    )
    try:
        performance_df = performance_analyzer.analyze_all_symbols(
            symbols, data_fetcher, verbose=True
        )
        if performance_df.empty:
            print(
                color_text(
                    "No valid performance data found. Please try again later.",
                    Fore.RED,
                    Style.BRIGHT,
                )
            )
            return
    except KeyboardInterrupt:
        print(color_text("\nAnalysis interrupted by user.", Fore.YELLOW))
        return
    except Exception as e:
        print(
            color_text(
                f"Error during performance analysis: {e}",
                Fore.RED,
                Style.BRIGHT,
            )
        )
        return

    # Step 3: Get top and worst performers
    print(
        color_text(
            f"\n[3/4] Identifying top {args.top_n} best and worst performers...",
            Fore.CYAN,
            Style.BRIGHT,
        )
    )
    best_performers = performance_analyzer.get_top_performers(performance_df, top_n=args.top_n)
    worst_performers = performance_analyzer.get_worst_performers(performance_df, top_n=args.top_n)

    display_performers(best_performers, "TOP PERFORMERS (Best)", Fore.GREEN)
    display_performers(worst_performers, "WORST PERFORMERS (Worst)", Fore.RED)

    # Step 4: Analyze pairs trading opportunities
    print(
        color_text(
            "\n[4/4] Analyzing pairs trading opportunities...",
            Fore.CYAN,
            Style.BRIGHT,
        )
    )
    try:
        pairs_df = pairs_analyzer.analyze_pairs_opportunity(
            best_performers, worst_performers, data_fetcher=data_fetcher, verbose=True
        )

        if pairs_df.empty:
            print(
                color_text(
                    "No pairs opportunities found.",
                    Fore.YELLOW,
                )
            )
            return

        # Validate pairs if requested
        if not args.no_validation:
            print(color_text("\nValidating pairs...", Fore.CYAN))
            pairs_df = pairs_analyzer.validate_pairs(pairs_df, data_fetcher, verbose=True)

        # Display results
        display_pairs_opportunities(pairs_df, max_display=args.max_pairs)

        # Summary
        print(color_text("\n" + "=" * 80, Fore.CYAN, Style.BRIGHT))
        print(color_text("SUMMARY", Fore.CYAN, Style.BRIGHT))
        print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
        print(f"Total symbols analyzed: {len(performance_df)}")
        print(f"Top performers identified: {len(best_performers)}")
        print(f"Worst performers identified: {len(worst_performers)}")
        print(f"Pairs opportunities found: {len(pairs_df)}")
        if not pairs_df.empty:
            avg_spread = pairs_df["spread"].mean() * 100
            print(f"Average spread: {avg_spread:.2f}%")
            correlations = pairs_df["correlation"].dropna()
            if not correlations.empty:
                avg_correlation = correlations.mean()
                print(f"Average correlation: {avg_correlation:.3f}")
        print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))

    except KeyboardInterrupt:
        print(color_text("\nPairs analysis interrupted by user.", Fore.YELLOW))
    except Exception as e:
        print(
            color_text(
                f"Error during pairs analysis: {e}",
                Fore.RED,
                Style.BRIGHT,
            )
        )


if __name__ == "__main__":
    main()

