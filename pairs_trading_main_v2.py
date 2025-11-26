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
import pandas as pd

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
    PAIRS_TRADING_MAX_HALF_LIFE,
)
from modules.common.utils import color_text, format_price, normalize_symbol_key
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


def display_pairs_opportunities(pairs_df, max_display=10, verbose=False):
    """Display pairs trading opportunities in a formatted table.
    
    Args:
        pairs_df: DataFrame with pairs data
        max_display: Maximum number of pairs to display
        verbose: If True, show additional quantitative metrics
    """

    def _pad_colored(text: str, width: int, color, style=None) -> str:
        """Pad text to fixed width before applying ANSI colors to avoid misalignment."""
        padded = text.ljust(width)
        if style is None:
            return color_text(padded, color)
        return color_text(padded, color, style)

    if pairs_df is None or pairs_df.empty:
        print(color_text("\nNo pairs trading opportunities found.", Fore.YELLOW))
        return

    print(color_text(f"\n{'=' * 120}", Fore.MAGENTA, Style.BRIGHT))
    print(color_text("PAIRS TRADING OPPORTUNITIES", Fore.MAGENTA, Style.BRIGHT))
    print(color_text(f"{'=' * 120}", Fore.MAGENTA, Style.BRIGHT))

    if verbose:
        print(
            f"{'Rank':<6} {'Long':<15} {'Short':<15} {'Spread':<10} {'Corr':<8} {'OppScore':<10} "
            f"{'QuantScore':<12} {'Coint':<7} {'HalfLife':<10} {'Sharpe':<10} {'MaxDD':<10}"
        )
        print("-" * 120)
    else:
        print(
            f"{'Rank':<6} {'Long (Worst)':<18} {'Short (Best)':<18} {'Spread':<12} {'Correlation':<15} "
            f"{'OppScore':<12} {'QuantScore':<12} {'Coint':<7}"
        )
        print("-" * 120)

    display_count = min(len(pairs_df), max_display)
    for idx in range(display_count):
        row = pairs_df.iloc[idx]
        rank = idx + 1
        long_symbol = row["long_symbol"]
        short_symbol = row["short_symbol"]
        spread = row["spread"] * 100
        correlation = row.get("correlation")
        opportunity_score = row["opportunity_score"] * 100
        quantitative_score = row.get("quantitative_score")
        is_cointegrated = row.get("is_cointegrated")
        if (is_cointegrated is None or pd.isna(is_cointegrated)) and "is_johansen_cointegrated" in row:
            alt_coint = row.get("is_johansen_cointegrated")
            if alt_coint is not None and not pd.isna(alt_coint):
                is_cointegrated = bool(alt_coint)
        
        # Get verbose metrics if available
        half_life = row.get("half_life")
        spread_sharpe = row.get("spread_sharpe")
        max_drawdown = row.get("max_drawdown")

        # Prepare spread text
        spread_text = f"{spread:+.2f}%"

        # Color code based on opportunity score
        if opportunity_score > 20:
            score_color = Fore.GREEN
        elif opportunity_score > 10:
            score_color = Fore.YELLOW
        else:
            score_color = Fore.WHITE
        opp_text = f"{opportunity_score:+.1f}%"
        opp_display = _pad_colored(opp_text, 12, score_color)

        # Color code quantitative score
        if quantitative_score is not None and not pd.isna(quantitative_score):
            if quantitative_score >= 70:
                quant_color = Fore.GREEN
            elif quantitative_score >= 50:
                quant_color = Fore.YELLOW
            else:
                quant_color = Fore.RED
            quant_text = f"{quantitative_score:.1f}"
        else:
            quant_color = Fore.WHITE
            quant_text = "N/A"
        quant_display = _pad_colored(quant_text, 12, quant_color)

        # Cointegration status
        if is_cointegrated is not None and not pd.isna(is_cointegrated):
            coint_status = "✅" if is_cointegrated else "❌"
            coint_color = Fore.GREEN if is_cointegrated else Fore.RED
        else:
            coint_status = "?"
            coint_color = Fore.WHITE
        coint_display = _pad_colored(coint_status, 7, coint_color)
        coint_display_verbose = _pad_colored(coint_status, 9, coint_color)

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
        corr_display = _pad_colored(corr_text, 12, corr_color)

        if verbose:
            # Format verbose metrics
            half_life_text = f"{half_life:.1f}" if half_life is not None and not pd.isna(half_life) else "N/A"
            sharpe_text = f"{spread_sharpe:.2f}" if spread_sharpe is not None and not pd.isna(spread_sharpe) else "N/A"
            maxdd_text = f"{max_drawdown*100:.1f}%" if max_drawdown is not None and not pd.isna(max_drawdown) else "N/A"
            
            print(
                f"{rank:<6} {long_symbol:<15} {short_symbol:<15} "
                f"{spread_text:<10} {corr_display} "
                f"{opp_display} "
                f"{quant_display} "
                f"{coint_display_verbose}"
                f"{half_life_text:<12} {sharpe_text:<12} {maxdd_text:<12}"
            )
        else:
            print(
                f"{rank:<6} {long_symbol:<18} {short_symbol:<18} "
                f"{spread_text:<12} {corr_display} "
                f"{opp_display} "
                f"{quant_display} "
                f"{coint_display}"
            )

    print(color_text(f"{'=' * 120}", Fore.MAGENTA, Style.BRIGHT))

    if len(pairs_df) > max_display:
        print(
            color_text(
                f"\nShowing top {max_display} of {len(pairs_df)} opportunities. "
                f"Use --max-pairs to see more.",
                Fore.CYAN,
            )
        )


def select_top_unique_pairs(pairs_df, target_pairs):
    """Pick up to target_pairs rows ensuring unique symbols when possible."""
    if pairs_df is None or pairs_df.empty:
        return pairs_df

    selected_indices = []
    used_symbols = set()

    for idx, row in pairs_df.iterrows():
        long_symbol = row["long_symbol"]
        short_symbol = row["short_symbol"]
        if long_symbol in used_symbols or short_symbol in used_symbols:
            continue
        selected_indices.append(idx)
        used_symbols.update([long_symbol, short_symbol])
        if len(selected_indices) == target_pairs:
            break

    if len(selected_indices) < target_pairs:
        for idx in pairs_df.index:
            if idx in selected_indices:
                continue
            selected_indices.append(idx)
            if len(selected_indices) == target_pairs:
                break

    if not selected_indices:
        return pairs_df.head(target_pairs).reset_index(drop=True)

    return pairs_df.loc[selected_indices].reset_index(drop=True)


def ensure_symbols_in_candidate_pools(performance_df, best_df, worst_df, target_symbols):
    """Ensure target symbols are present in candidate pools based on their score direction."""
    if not target_symbols:
        return best_df, worst_df

    best_symbols = set(best_df["symbol"].tolist())
    worst_symbols = set(worst_df["symbol"].tolist())

    for symbol in target_symbols:
        row = performance_df[performance_df["symbol"] == symbol]
        if row.empty:
            continue
        score = row.iloc[0]["score"]
        if score >= 0:
            if symbol not in best_symbols:
                best_df = pd.concat([best_df, row], ignore_index=True)
                best_symbols.add(symbol)
        else:
            if symbol not in worst_symbols:
                worst_df = pd.concat([worst_df, row], ignore_index=True)
                worst_symbols.add(symbol)

    best_df = (
        best_df.sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    worst_df = (
        worst_df.sort_values("score", ascending=True)
        .reset_index(drop=True)
    )
    return best_df, worst_df


def select_pairs_for_symbols(pairs_df, target_symbols, max_pairs=None):
    """Select the best pair (highest score) for each requested symbol."""
    if pairs_df is None or pairs_df.empty or not target_symbols:
        return pd.DataFrame(columns=pairs_df.columns if pairs_df is not None else [])

    selected_rows = []
    for symbol in target_symbols:
        matches = pairs_df[
            (pairs_df["long_symbol"] == symbol) | (pairs_df["short_symbol"] == symbol)
        ]
        if matches.empty:
            continue
        selected_rows.append(matches.iloc[0])
        if max_pairs is not None and len(selected_rows) >= max_pairs:
            break

    if not selected_rows:
        return pd.DataFrame(columns=pairs_df.columns)

    return pd.DataFrame(selected_rows).reset_index(drop=True)
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


def prompt_interactive_mode():
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


def main():
    """Main function for pairs trading analysis."""
    import argparse
    import pandas as pd

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
        default=20,
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
        help="Weights for timeframes in format '1d:0.3,3d:0.4,1w:0.3' (default: from config)",
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
        help="Show detailed quantitative metrics (half-life, Sharpe, MaxDD, etc.) in output",
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

    args = parser.parse_args()

    if not args.no_menu:
        menu_result = prompt_interactive_mode()
        if menu_result["mode"] == "auto":
            args.symbols = None
        else:
            args.symbols = menu_result["symbols_raw"]

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

    # Parse target symbols if provided
    target_symbol_inputs = []
    parsed_target_symbols = []
    target_symbols = []
    if args.symbols:
        raw_parts = (
            args.symbols.replace(",", " ")
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
        if target_symbol_inputs:
            print(
                color_text(
                    f"\nManual mode enabled for symbols: {', '.join(target_symbol_inputs)}",
                    Fore.MAGENTA,
                    Style.BRIGHT,
                )
            )

    print(color_text("\n" + "=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("PAIRS TRADING ANALYSIS", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(
        color_text(
            f"\nConfiguration:", Fore.CYAN,
        )
    )
    print(f"  Target pairs: {args.pairs_count}")
    print(f"  Candidate depth per side: {args.candidate_depth}")
    print(f"  Weights: 1d={weights['1d']:.2f}, 3d={weights['3d']:.2f}, 1w={weights['1w']:.2f}")
    print(f"  Min volume: {args.min_volume:,.0f} USDT")
    print(f"  Spread range: {args.min_spread*100:.2f}% - {args.max_spread*100:.2f}%")
    print(f"  Correlation range: {args.min_correlation:.2f} - {args.max_correlation:.2f}")
    if target_symbol_inputs:
        print(f"  Mode: MANUAL (requested {', '.join(target_symbol_inputs)})")
    else:
        print("  Mode: AUTO (optimize across all symbols)")

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
        require_cointegration=args.require_cointegration,
        max_half_life=args.max_half_life,
        min_quantitative_score=args.min_quantitative_score,
    )

    # Step 1: Get list of futures symbols
    print(color_text("\n[1/4] Fetching futures symbols from Binance...", Fore.CYAN, Style.BRIGHT))
    
    try:
        symbols = data_fetcher.list_binance_futures_symbols(
            max_candidates=args.max_symbols,
            progress_label="Symbol Discovery",
        )
        if not symbols:
            print(
                color_text(
                    "No symbols found. Please check your API connection.",
                    Fore.RED,
                    Style.BRIGHT,
                )
            )
            return
        print(color_text(f"Found {len(symbols)} futures symbols.", Fore.GREEN))
    except Exception as e:
        print(
            color_text(
                f"Error fetching symbols: {e}",
                Fore.RED,
                Style.BRIGHT,
            )
        )
        return

    if target_symbol_inputs:
        available_lookup = {normalize_symbol_key(sym): sym for sym in symbols}
        missing_targets = []
        mapped_targets = []
        newly_added_symbols = []
        for sym in parsed_target_symbols:
            normalized_key = normalize_symbol_key(sym)
            actual_symbol = available_lookup.get(normalized_key)
            if actual_symbol:
                mapped_targets.append(actual_symbol)
            else:
                missing_targets.append(sym)
                mapped_targets.append(sym)
                if normalized_key not in available_lookup:
                    newly_added_symbols.append(sym)
                    available_lookup[normalized_key] = sym
        if newly_added_symbols:
            symbols = list(dict.fromkeys(symbols + newly_added_symbols))
            print(
                color_text(
                    f"Added manual symbols to analysis universe: {', '.join(newly_added_symbols)}",
                    Fore.CYAN,
                )
            )
        if missing_targets:
            print(
                color_text(
                    f"These symbols were not discovered automatically but will be fetched manually: {', '.join(missing_targets)}",
                    Fore.YELLOW,
                )
            )
        target_symbols = mapped_targets
        if target_symbols:
            print(
                color_text(
                    f"Tracking manual symbols: {', '.join(target_symbols)}",
                    Fore.MAGENTA,
                    Style.BRIGHT,
                )
            )
        if not target_symbols:
            print(
                color_text(
                    "All requested symbols were unavailable. Reverting to AUTO mode.",
                    Fore.YELLOW,
                )
            )
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

    # Step 3: Build candidate pools for long/short sides
    print(
        color_text(
            f"\n[3/4] Building candidate pools for auto pair selection...",
            Fore.CYAN,
            Style.BRIGHT,
        )
    )
    candidate_depth = max(args.candidate_depth, args.pairs_count * 2)
    best_performers = (
        performance_df.sort_values("score", ascending=False)
        .head(candidate_depth)
        .reset_index(drop=True)
    )
    worst_performers = (
        performance_df.sort_values("score", ascending=True)
        .head(candidate_depth)
        .reset_index(drop=True)
    )

    if target_symbols:
        best_performers, worst_performers = ensure_symbols_in_candidate_pools(
            performance_df, best_performers, worst_performers, target_symbols
        )

    display_performers(best_performers, "SHORT CANDIDATES (Strong performers)", Fore.GREEN)
    display_performers(worst_performers, "LONG CANDIDATES (Weak performers)", Fore.RED)

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

        # Sort pairs by selected criteria
        sort_column = args.sort_by if args.sort_by in pairs_df.columns else "opportunity_score"
        if sort_column in pairs_df.columns:
            pairs_df = pairs_df.sort_values(sort_column, ascending=False).reset_index(drop=True)
            sort_display = "quantitative score" if sort_column == "quantitative_score" else "opportunity score"
            print(color_text(f"\nSorted pairs by {sort_display} (descending).", Fore.CYAN))

        selected_pairs = None
        manual_pairs = None
        displayed_manual = False
        if target_symbols:
            manual_pairs = select_pairs_for_symbols(
                pairs_df,
                target_symbols,
                max_pairs=None,
            )
            matched_symbols = {
                sym
                for sym in target_symbols
                if not manual_pairs[
                    (manual_pairs["long_symbol"] == sym)
                    | (manual_pairs["short_symbol"] == sym)
                ].empty
            }
            missing_matches = [sym for sym in target_symbols if sym not in matched_symbols]
            if missing_matches:
                print(
                    color_text(
                        f"No valid pairs found for: {', '.join(missing_matches)}",
                        Fore.YELLOW,
                    )
                )
            if not manual_pairs.empty:
                print(
                    color_text(
                        "\nBest pairs for requested symbols:",
                        Fore.MAGENTA,
                        Style.BRIGHT,
                    )
                )
                display_pairs_opportunities(
                    manual_pairs,
                    max_display=min(args.max_pairs, len(manual_pairs)),
                    verbose=args.verbose,
                )
                selected_pairs = manual_pairs
                displayed_manual = True

        if selected_pairs is None:
            # Select target number of tradeable pairs (unique symbols when possible)
            selected_pairs = select_top_unique_pairs(pairs_df, args.pairs_count)

        if selected_pairs is None or selected_pairs.empty:
            print(color_text("No qualifying pairs after selection.", Fore.YELLOW))
            return

        if not displayed_manual:
            display_pairs_opportunities(
                selected_pairs,
                max_display=min(args.max_pairs, len(selected_pairs)),
                verbose=args.verbose,
            )

        # Summary
        print(color_text("\n" + "=" * 80, Fore.CYAN, Style.BRIGHT))
        print(color_text("SUMMARY", Fore.CYAN, Style.BRIGHT))
        print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
        print(f"Total symbols analyzed: {len(performance_df)}")
        print(f"Short candidates considered: {len(best_performers)}")
        print(f"Long candidates considered: {len(worst_performers)}")
        print(f"Valid pairs available: {len(pairs_df)}")
        print(f"Selected tradeable pairs: {len(selected_pairs)}")
        if not selected_pairs.empty:
            avg_spread = selected_pairs["spread"].mean() * 100
            print(f"Average spread: {avg_spread:.2f}%")
            correlations = selected_pairs["correlation"].dropna()
            if not correlations.empty:
                avg_correlation = correlations.mean()
                print(f"Average correlation: {avg_correlation:.3f}")
            
            # Quantitative metrics statistics
            print(color_text("\nQuantitative Metrics:", Fore.CYAN, Style.BRIGHT))
            
            # Quantitative score
            quant_scores = selected_pairs["quantitative_score"].dropna()
            if not quant_scores.empty:
                avg_quant_score = quant_scores.mean()
                print(f"  Average quantitative score: {avg_quant_score:.1f}/100")
            
            # Cointegration rate
            is_cointegrated_col = selected_pairs.get("is_cointegrated")
            if is_cointegrated_col is not None:
                cointegrated_count = is_cointegrated_col.fillna(False).sum()
                cointegration_rate = (cointegrated_count / len(selected_pairs)) * 100
                print(f"  Cointegration rate: {cointegration_rate:.1f}% ({cointegrated_count}/{len(selected_pairs)})")
            
            # Average half-life
            half_lives = selected_pairs.get("half_life").dropna() if "half_life" in selected_pairs.columns else pd.Series()
            if not half_lives.empty:
                avg_half_life = half_lives.mean()
                print(f"  Average half-life: {avg_half_life:.1f} periods")
            
            # Average Sharpe ratio
            sharpe_ratios = selected_pairs.get("spread_sharpe").dropna() if "spread_sharpe" in selected_pairs.columns else pd.Series()
            if not sharpe_ratios.empty:
                avg_sharpe = sharpe_ratios.mean()
                print(f"  Average Sharpe ratio: {avg_sharpe:.2f}")
            
            # Average max drawdown
            max_dds = selected_pairs.get("max_drawdown").dropna() if "max_drawdown" in selected_pairs.columns else pd.Series()
            if not max_dds.empty:
                avg_max_dd = max_dds.mean() * 100
                print(f"  Average max drawdown: {avg_max_dd:.2f}%")
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

