"""
Hedge finder for discovering and analyzing hedge candidates.
"""

import os
from math import ceil
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from colorama import Fore, Style

try:
    from modules.common.Position import Position
    from modules.common.utils import normalize_symbol, color_text
    from modules.common.ProgressBar import ProgressBar
    from modules.common.ExchangeManager import ExchangeManager
    from modules.common.DataFetcher import DataFetcher
    from modules.portfolio.correlation_analyzer import PortfolioCorrelationAnalyzer
    from modules.portfolio.risk_calculator import PortfolioRiskCalculator
    from modules.config import BENCHMARK_SYMBOL
except ImportError:
    Position = None
    normalize_symbol = None
    color_text = None
    ProgressBar = None
    ExchangeManager = None
    DataFetcher = None
    PortfolioCorrelationAnalyzer = None
    PortfolioRiskCalculator = None
    BENCHMARK_SYMBOL = "BTC/USDT"


class HedgeFinder:
    """Finds and analyzes hedge candidates."""

    def __init__(
        self,
        exchange_manager: ExchangeManager,
        correlation_analyzer: PortfolioCorrelationAnalyzer,
        risk_calculator: PortfolioRiskCalculator,
        positions: List[Position],
        benchmark_symbol: str = BENCHMARK_SYMBOL,
        shutdown_event=None,
        data_fetcher: Optional["DataFetcher"] = None,
    ):
        self.exchange_manager = exchange_manager
        self.correlation_analyzer = correlation_analyzer
        self.risk_calculator = risk_calculator
        self.positions = positions
        self.benchmark_symbol = benchmark_symbol
        self.shutdown_event = shutdown_event
        if data_fetcher is not None:
            self.data_fetcher = data_fetcher
        elif DataFetcher is not None:
            self.data_fetcher = DataFetcher(exchange_manager, shutdown_event)
        else:
            self.data_fetcher = None

    def should_stop(self) -> bool:
        """Check if shutdown was requested."""
        if self.shutdown_event:
            return self.shutdown_event.is_set()
        return False

    def _list_candidate_symbols(
        self,
        exclude_symbols: Optional[set] = None,
        max_candidates: Optional[int] = None,
    ) -> List[str]:
        """Fetches potential hedge symbols from Binance Futures."""
        if self.data_fetcher is None:
            raise ImportError("DataFetcher is required to list candidate symbols.")
        progress_label = "Symbol Discovery"
        return self.data_fetcher.list_binance_futures_symbols(
            exclude_symbols=exclude_symbols,
            max_candidates=max_candidates,
            progress_label=progress_label,
        )

    def _score_candidate(self, symbol: str) -> Optional[Dict[str, float]]:
        """Score a hedge candidate based on correlation."""
        weighted_corr, _ = self.correlation_analyzer.calculate_weighted_correlation_with_new_symbol(
            symbol, verbose=False
        )
        return_corr, _ = (
            self.correlation_analyzer.calculate_portfolio_return_correlation(
                symbol, verbose=False
            )
        )

        if weighted_corr is None and return_corr is None:
            return None

        score_components = [
            abs(x) for x in [weighted_corr, return_corr] if x is not None
        ]
        if not score_components:
            return None

        score = sum(score_components) / len(score_components)
        return {
            "symbol": symbol,
            "weighted_corr": weighted_corr,
            "return_corr": return_corr,
            "score": score,
        }

    def find_best_hedge_candidate(
        self,
        total_delta: float,
        total_beta_delta: float,
        max_candidates: Optional[int] = None,
    ) -> Optional[Dict]:
        """Automatically scans Binance futures symbols to find the best hedge candidate."""
        if not self.positions:
            print(
                color_text(
                    "No positions loaded. Cannot search for hedge candidates.",
                    Fore.YELLOW,
                )
            )
            return None

        existing_symbols = {normalize_symbol(p.symbol) for p in self.positions}
        existing_symbols.add(normalize_symbol(self.benchmark_symbol))
        candidate_symbols = self._list_candidate_symbols(
            existing_symbols, max_candidates=None
        )

        if not candidate_symbols:
            print(
                color_text(
                    "Could not find candidate symbols from Binance.", Fore.YELLOW
                )
            )
            return None
        if self.should_stop():
            print(color_text("Hedge scan aborted before start.", Fore.YELLOW))
            return None

        if max_candidates is not None:
            candidate_symbols = candidate_symbols[:max_candidates]
        scan_count = len(candidate_symbols)
        print(
            color_text(
                f"\nScanning {scan_count} candidate(s) for optimal hedge...", Fore.CYAN
            )
        )

        core_count = max(1, int((os.cpu_count() or 1) * 0.8))
        batch_size = ceil(scan_count / core_count) if scan_count else 0
        batches = [
            candidate_symbols[i : i + batch_size]
            for i in range(0, scan_count, batch_size)
        ] or [[]]
        total_batches = len([b for b in batches if b])
        progress_bar = ProgressBar(total_batches or 1, "Batch Progress")

        best_candidate = None

        def process_batch(batch_symbols: List[str]) -> Optional[Dict]:
            local_best = None
            for sym in batch_symbols:
                if self.should_stop():
                    return None
                result = self._score_candidate(sym)
                if result is None:
                    continue
                if local_best is None or result["score"] > local_best["score"]:
                    local_best = result
            return local_best

        with ThreadPoolExecutor(max_workers=core_count) as executor:
            futures = {}
            for idx, batch in enumerate(batches, start=1):
                if not batch:
                    continue
                print(
                    color_text(
                        f"Starting batch {idx}/{total_batches} (size {len(batch)})",
                        Fore.BLUE,
                    )
                )
                futures[executor.submit(process_batch, batch)] = idx
            for future in as_completed(futures):
                if self.should_stop():
                    break
                batch_id = futures[future]
                try:
                    batch_best = future.result()
                except Exception as exc:
                    print(color_text(f"Batch {batch_id} failed: {exc}", Fore.RED))
                    continue
                if batch_best is None:
                    print(
                        color_text(
                            f"Batch {batch_id}: no viable candidate.", Fore.YELLOW
                        )
                    )
                    progress_bar.update()
                    continue
                print(
                    color_text(
                        f"Batch {batch_id}: best {batch_best['symbol']} (score {batch_best['score']:.4f})",
                        Fore.WHITE,
                    )
                )
                if (
                    best_candidate is None
                    or batch_best["score"] > best_candidate["score"]
                ):
                    best_candidate = batch_best
                progress_bar.update()
        if total_batches:
            progress_bar.finish()

        if best_candidate is None:
            print(
                color_text(
                    "No suitable hedge candidate found (insufficient data).",
                    Fore.YELLOW,
                )
            )
        else:
            print(
                color_text(
                    f"\nBest candidate: {best_candidate['symbol']} (score {best_candidate['score']:.4f})",
                    Fore.MAGENTA,
                    Style.BRIGHT,
                )
            )
            if best_candidate["weighted_corr"] is not None:
                print(
                    color_text(
                        f"  Weighted Correlation: {best_candidate['weighted_corr']:.4f}",
                        Fore.WHITE,
                    )
                )
            if best_candidate["return_corr"] is not None:
                print(
                    color_text(
                        f"  Portfolio Return Correlation: {best_candidate['return_corr']:.4f}",
                        Fore.WHITE,
                    )
                )

        return best_candidate

    def analyze_new_trade(
        self,
        new_symbol: str,
        total_delta: float,
        total_beta_delta: float,
        last_var_value: Optional[float] = None,
        last_var_confidence: Optional[float] = None,
        correlation_mode: str = "weighted",
    ):
        """Analyzes a potential new trade and automatically recommends direction for beta-weighted hedging."""
        normalized_symbol = normalize_symbol(new_symbol)
        if normalized_symbol != new_symbol:
            print(
                color_text(
                    f"Symbol normalized: '{new_symbol}' -> '{normalized_symbol}'",
                    Fore.CYAN,
                )
            )

        new_symbol = normalized_symbol
        print(
            color_text(
                f"\nAnalyzing potential trade on {new_symbol}...",
                Fore.CYAN,
                Style.BRIGHT,
            )
        )
        print(color_text(f"Current Total Delta: {total_delta:+.2f} USDT", Fore.WHITE))
        print(
            color_text(
                f"Current Total Beta Delta (vs {self.benchmark_symbol}): {total_beta_delta:+.2f} USDT",
                Fore.WHITE,
            )
        )

        new_symbol_beta = self.risk_calculator.calculate_beta(new_symbol)
        beta_available = new_symbol_beta is not None and abs(new_symbol_beta) > 1e-6
        if beta_available:
            print(
                color_text(
                    f"{new_symbol} beta vs {self.benchmark_symbol}: {new_symbol_beta:.4f}",
                    Fore.CYAN,
                )
            )
        else:
            print(
                color_text(
                    f"Could not compute beta for {new_symbol}. Falling back to simple delta hedging.",
                    Fore.YELLOW,
                )
            )

        hedge_mode = "beta" if beta_available else "delta"
        metric_label = "Beta Delta" if beta_available else "Delta"
        current_metric = total_beta_delta if beta_available else total_delta
        target_metric = -current_metric

        recommended_direction = None
        recommended_size = None

        if abs(current_metric) < 0.01:
            print(
                color_text(
                    f"Portfolio is already {metric_label} Neutral ({metric_label} ≈ 0).",
                    Fore.GREEN,
                )
            )
        else:
            if beta_available:
                beta_sign = np.sign(new_symbol_beta)
                if beta_sign == 0:
                    beta_available = False
                    hedge_mode = "delta"
                    metric_label = "Delta"
                    current_metric = total_delta
                    target_metric = -current_metric
                else:
                    direction_multiplier = -np.sign(current_metric) * beta_sign
                    recommended_direction = (
                        "LONG" if direction_multiplier >= 0 else "SHORT"
                    )
                    recommended_size = abs(current_metric) / max(
                        abs(new_symbol_beta), 1e-6
                    )
                    print(
                        color_text(
                            f"Targeting Beta Neutrality using {metric_label}.",
                            Fore.CYAN,
                        )
                    )
            if not beta_available:
                if current_metric > 0:
                    recommended_direction = "SHORT"
                    recommended_size = abs(target_metric)
                    print(
                        color_text(
                            "Portfolio has excess LONG delta exposure.", Fore.YELLOW
                        )
                    )
                else:
                    recommended_direction = "LONG"
                    recommended_size = abs(target_metric)
                    print(
                        color_text(
                            "Portfolio has excess SHORT delta exposure.", Fore.YELLOW
                        )
                    )
                print(color_text("Targeting simple Delta Neutrality.", Fore.CYAN))

        if recommended_direction and recommended_size is not None:
            print(
                color_text(
                    f"\nRecommended {hedge_mode.upper()} hedge:",
                    Fore.CYAN,
                    Style.BRIGHT,
                )
            )
            print(color_text(f"  Direction: {recommended_direction}", Fore.WHITE))
            print(
                color_text(
                    f"  Size: {recommended_size:.2f} USDT", Fore.GREEN, Style.BRIGHT
                )
            )

        if not self.positions:
            print(
                color_text(
                    "\nNo existing positions for correlation analysis.", Fore.WHITE
                )
            )
            return (
                recommended_direction,
                recommended_size if recommended_direction else None,
                None,
            )

        print(color_text("\n" + "=" * 70, Fore.CYAN))
        print(
            color_text(
                "CORRELATION ANALYSIS - COMPARING BOTH METHODS", Fore.CYAN, Style.BRIGHT
            )
        )
        print(color_text("=" * 70, Fore.CYAN))

        weighted_corr, weighted_details = (
            self.correlation_analyzer.calculate_weighted_correlation_with_new_symbol(new_symbol)
        )
        portfolio_return_corr, portfolio_return_details = (
            self.correlation_analyzer.calculate_portfolio_return_correlation(new_symbol)
        )

        print(color_text("\n" + "=" * 70, Fore.CYAN))
        print(color_text("CORRELATION SUMMARY", Fore.MAGENTA, Style.BRIGHT))
        print(color_text("=" * 70, Fore.CYAN))

        if weighted_corr is not None:
            weighted_color = (
                Fore.GREEN
                if abs(weighted_corr) > 0.7
                else (Fore.YELLOW if abs(weighted_corr) > 0.4 else Fore.RED)
            )
            print("1. Weighted Correlation (by Position Size):")
            print(
                f"   {new_symbol} vs Portfolio: {color_text(f'{weighted_corr:>6.4f}', weighted_color, Style.BRIGHT)}"
            )

            if abs(weighted_corr) > 0.7:
                print(
                    color_text("   → High correlation. Good for hedging.", Fore.GREEN)
                )
            elif abs(weighted_corr) > 0.4:
                print(
                    color_text(
                        "   → Moderate correlation. Partial hedging effect.",
                        Fore.YELLOW,
                    )
                )
            else:
                print(
                    color_text(
                        "   → Low correlation. Limited hedging effectiveness.", Fore.RED
                    )
                )
        else:
            print(
                f"1. Weighted Correlation: {color_text('N/A (insufficient data)', Fore.YELLOW)}"
            )

        if portfolio_return_corr is not None:
            portfolio_color = (
                Fore.GREEN
                if abs(portfolio_return_corr) > 0.7
                else (Fore.YELLOW if abs(portfolio_return_corr) > 0.4 else Fore.RED)
            )
            samples_info = (
                portfolio_return_details.get("samples", "N/A")
                if isinstance(portfolio_return_details, dict)
                else "N/A"
            )
            print("\n2. Portfolio Return Correlation (includes direction):")
            print(
                f"   {new_symbol} vs Portfolio Return: {color_text(f'{portfolio_return_corr:>6.4f}', portfolio_color, Style.BRIGHT)}"
            )
            print(f"   Samples used: {samples_info}")

            if abs(portfolio_return_corr) > 0.7:
                print(
                    color_text(
                        "   → High correlation. Excellent for hedging.", Fore.GREEN
                    )
                )
            elif abs(portfolio_return_corr) > 0.4:
                print(
                    color_text(
                        "   → Moderate correlation. Acceptable hedging effect.",
                        Fore.YELLOW,
                    )
                )
            else:
                print(
                    color_text(
                        "   → Low correlation. Poor hedging effectiveness.", Fore.RED
                    )
                )
        else:
            print(
                f"\n2. Portfolio Return Correlation: {color_text('N/A (insufficient data)', Fore.YELLOW)}"
            )

        print(color_text("\n" + "-" * 70, Fore.WHITE))
        print(color_text("OVERALL ASSESSMENT:", Fore.CYAN, Style.BRIGHT))

        if weighted_corr is not None and portfolio_return_corr is not None:
            diff = abs(weighted_corr - portfolio_return_corr)
            if diff < 0.1:
                print(
                    color_text(
                        "   ✓ Both methods show similar correlation → Consistent result",
                        Fore.GREEN,
                    )
                )
            else:
                print(
                    color_text(
                        f"   ⚠ Methods differ by {diff:.4f} → Check if portfolio has SHORT positions",
                        Fore.YELLOW,
                    )
                )

            avg_corr = (abs(weighted_corr) + abs(portfolio_return_corr)) / 2
            if avg_corr > 0.7:
                print(
                    color_text(
                        "   [OK] High correlation detected. This pair is suitable for statistical hedging.",
                        Fore.GREEN,
                        Style.BRIGHT,
                    )
                )
            elif avg_corr > 0.4:
                print(
                    color_text(
                        "   [!] Moderate correlation. Hedge may be partially effective.",
                        Fore.YELLOW,
                    )
                )
            else:
                print(
                    color_text(
                        "   [X] Low correlation. This hedge might be less effective systematically.",
                        Fore.RED,
                    )
                )

        elif weighted_corr is not None:
            if abs(weighted_corr) > 0.7:
                print(
                    color_text(
                        "   [OK] High correlation detected. This pair is suitable for statistical hedging.",
                        Fore.GREEN,
                        Style.BRIGHT,
                    )
                )
            elif abs(weighted_corr) > 0.4:
                print(
                    color_text(
                        "   [!] Moderate correlation. Hedge may be partially effective.",
                        Fore.YELLOW,
                    )
                )
            else:
                print(
                    color_text(
                        "   [X] Low correlation. This hedge might be less effective systematically.",
                        Fore.RED,
                    )
                )

        elif portfolio_return_corr is not None:
            if abs(portfolio_return_corr) > 0.7:
                print(
                    color_text(
                        "   [OK] High correlation detected. This pair is suitable for statistical hedging.",
                        Fore.GREEN,
                        Style.BRIGHT,
                    )
                )
            elif abs(portfolio_return_corr) > 0.4:
                print(
                    color_text(
                        "   [!] Moderate correlation. Hedge may be partially effective.",
                        Fore.YELLOW,
                    )
                )
            else:
                print(
                    color_text(
                        "   [X] Low correlation. This hedge might be less effective systematically.",
                        Fore.RED,
                    )
                )

        if last_var_value is not None and last_var_confidence is not None:
            conf_pct = int(last_var_confidence * 100)
            print(color_text("\nVaR INSIGHT:", Fore.MAGENTA, Style.BRIGHT))
            print(
                color_text(
                    f"  With {conf_pct}% confidence, daily loss is unlikely to exceed {last_var_value:.2f} USDT.",
                    Fore.WHITE,
                )
            )
            print(
                color_text(
                    "  Use this ceiling to judge whether the proposed hedge keeps risk tolerable.",
                    Fore.WHITE,
                )
            )
        else:
            print(
                color_text(
                    "\nVaR INSIGHT: N/A (insufficient historical data for VaR).",
                    Fore.YELLOW,
                )
            )

        print(color_text("=" * 70 + "\n", Fore.CYAN))

        final_corr = (
            weighted_corr if weighted_corr is not None else portfolio_return_corr
        )

        return (
            recommended_direction,
            recommended_size if recommended_direction else None,
            final_corr,
        )
