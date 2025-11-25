"""
Correlation analyzer for portfolio correlation calculations.
"""

import pandas as pd
import numpy as np
from typing import List
from colorama import Fore, Style

try:
    from modules.common.Position import Position
    from modules.common.utils import color_text, normalize_symbol
    from modules.config import (
        DEFAULT_CORRELATION_MIN_POINTS,
        DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS,
    )
except ImportError:
    Position = None
    color_text = None
    normalize_symbol = None
    DEFAULT_CORRELATION_MIN_POINTS = 10
    DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS = 10


class PortfolioCorrelationAnalyzer:
    """Analyzes correlation between portfolio and new symbols."""

    def __init__(self, data_fetcher, positions: List[Position]):
        self.data_fetcher = data_fetcher
        self.positions = positions
        self._series_cache: dict[str, pd.Series] = {}

    def _fetch_symbol_series(self, symbol: str) -> pd.Series | None:
        """
        Helper method to fetch price series for a symbol.
        Uses cache to avoid redundant fetches.
        """
        if symbol in self._series_cache:
            return self._series_cache[symbol]

        df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(symbol)
        series = self.data_fetcher.dataframe_to_close_series(df)
        if series is not None:
            self._series_cache[symbol] = series
        return series

    def _get_portfolio_series_dict(self) -> dict[str, pd.Series]:
        """Get all price series for portfolio positions."""
        symbol_series = {}
        for pos in self.positions:
            if pos.symbol not in symbol_series:
                series = self._fetch_symbol_series(pos.symbol)
                if series is not None:
                    symbol_series[pos.symbol] = series
        return symbol_series

    def calculate_weighted_correlation(
        self, verbose: bool = True
    ) -> tuple[float | None, list]:
        """
        Calculate weighted internal correlation of the current portfolio (between positions).
        
        Returns:
            (weighted_correlation, position_correlations_list)
            - weighted_correlation: Average weighted correlation between all position pairs
            - position_correlations_list: List of correlation details for each pair
        """
        if len(self.positions) < 2:
            if verbose:
                print(
                    color_text(
                        "Need at least 2 positions to calculate internal correlation.",
                        Fore.YELLOW,
                    )
                )
            return None, []

        if verbose:
            print(
                color_text(
                    "\nPortfolio Internal Correlation Analysis:", Fore.CYAN, Style.BRIGHT
                )
            )

        symbol_series = self._get_portfolio_series_dict()
        if len(symbol_series) < 2:
            if verbose:
                print(
                    color_text(
                        "Insufficient data for internal correlation analysis.", Fore.YELLOW
                    )
                )
            return None, []

        correlations = []
        weights = []
        position_pairs = []

        # Calculate correlation between all pairs
        symbols = list(symbol_series.keys())
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i + 1 :]:
                series1 = symbol_series[symbol1]
                series2 = symbol_series[symbol2]

                df = pd.concat([series1, series2], axis=1, join="inner")
                if len(df) < DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS:
                    continue

                # Calculate correlation on returns (pct_change) instead of prices
                # to avoid spurious correlation from non-stationary price series
                returns_df = df.pct_change().dropna()
                if len(returns_df) < DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS:
                    continue

                # Weight by average position size
                pos1 = next((p for p in self.positions if p.symbol == symbol1), None)
                pos2 = next((p for p in self.positions if p.symbol == symbol2), None)

                # Adjust returns based on position direction (LONG/SHORT)
                # SHORT positions have inverted returns for PnL correlation
                adjusted_returns = returns_df.copy()
                if pos1 and pos1.direction == "SHORT":
                    adjusted_returns.iloc[:, 0] = -adjusted_returns.iloc[:, 0]
                if pos2 and pos2.direction == "SHORT":
                    adjusted_returns.iloc[:, 1] = -adjusted_returns.iloc[:, 1]

                corr = adjusted_returns.iloc[:, 0].corr(adjusted_returns.iloc[:, 1])
                if pd.isna(corr):
                    continue
                weight = (
                    (pos1.size_usdt if pos1 else 0) + (pos2.size_usdt if pos2 else 0)
                ) / 2

                correlations.append(corr)
                weights.append(weight)
                position_pairs.append(
                    {
                        "symbol1": symbol1,
                        "symbol2": symbol2,
                        "direction1": pos1.direction if pos1 else "UNKNOWN",
                        "direction2": pos2.direction if pos2 else "UNKNOWN",
                        "correlation": corr,  # PnL correlation (adjusted for LONG/SHORT)
                        "weight": weight,
                    }
                )

        if not correlations:
            if verbose:
                print(
                    color_text(
                        "Insufficient overlapping data for correlation analysis.",
                        Fore.YELLOW,
                    )
                )
            return None, []

        total_weight = sum(weights)
        weighted_corr = (
            sum(c * w for c, w in zip(correlations, weights)) / total_weight
            if total_weight > 0
            else sum(correlations) / len(correlations)
        )

        if verbose:
            print("\nPosition Pair Correlations (PnL-adjusted):")
            for pair in position_pairs:
                corr_color = (
                    Fore.GREEN
                    if abs(pair["correlation"]) > 0.7
                    else (Fore.YELLOW if abs(pair["correlation"]) > 0.4 else Fore.RED)
                )
                weight_pct = (pair["weight"] / total_weight * 100) if total_weight > 0 else 0
                direction1 = pair.get("direction1", "UNKNOWN")
                direction2 = pair.get("direction2", "UNKNOWN")
                print(
                    f"  {pair['symbol1']:12} ({direction1:5}) <-> "
                    f"{pair['symbol2']:12} ({direction2:5}) "
                    f"({pair['weight']:>8.2f} USDT, {weight_pct:>5.1f}%): "
                    + color_text(f"{pair['correlation']:>6.4f}", corr_color)
                )

            print(
                f"\n{color_text('Weighted Internal Correlation:', Fore.CYAN, Style.BRIGHT)}"
            )
            corr_color = (
                Fore.GREEN
                if abs(weighted_corr) > 0.7
                else (Fore.YELLOW if abs(weighted_corr) > 0.4 else Fore.RED)
            )
            print(
                f"  Portfolio Internal: {color_text(f'{weighted_corr:>6.4f}', corr_color, Style.BRIGHT)}"
            )

        return weighted_corr, position_pairs

    def analyze_correlation_with_new_symbol(
        self,
        new_symbol: str,
        new_position_size: float = 0.0,
        new_direction: str = "LONG",
        verbose: bool = True,
    ) -> dict:
        """
        Analyze correlation impact of adding a new symbol to the portfolio.
        
        Args:
            new_symbol: Symbol to analyze
            new_position_size: Size of new position in USDT (for weighted calculation)
            new_direction: Direction of new position (LONG/SHORT)
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary with before/after correlation metrics
        """
        result = {
            "before": {},
            "after": {},
            "impact": {},
        }

        # Calculate current portfolio internal correlation
        if verbose:
            print(
                color_text(
                    "\n=== Analyzing Correlation Impact of Adding New Symbol ===",
                    Fore.CYAN,
                    Style.BRIGHT,
                )
            )

        internal_corr_before, _ = self.calculate_weighted_correlation(
            verbose=False
        )
        result["before"]["internal_correlation"] = internal_corr_before

        # Calculate correlation with new symbol
        weighted_corr, position_details = self.calculate_weighted_correlation_with_new_symbol(
            new_symbol, verbose=False
        )
        result["after"]["new_symbol_correlation"] = weighted_corr

        # Calculate portfolio return correlation with new symbol
        portfolio_return_corr, return_metadata = (
            self.calculate_portfolio_return_correlation(new_symbol, verbose=False)
        )
        result["after"]["portfolio_return_correlation"] = portfolio_return_corr

        # Simulate adding new position and recalculate internal correlation
        if new_position_size > 0:
            from modules.common.Position import Position

            temp_positions = self.positions + [
                Position(new_symbol, new_direction, 0.0, new_position_size)
            ]
            temp_analyzer = PortfolioCorrelationAnalyzer(
                self.data_fetcher, temp_positions
            )
            internal_corr_after, _ = temp_analyzer.calculate_weighted_correlation(
                verbose=False
            )
            result["after"]["internal_correlation"] = internal_corr_after

            # Calculate impact
            if internal_corr_before is not None and internal_corr_after is not None:
                correlation_change = internal_corr_after - internal_corr_before
                result["impact"]["correlation_change"] = correlation_change
                result["impact"]["diversification_improvement"] = (
                    abs(internal_corr_after) < abs(internal_corr_before)
                )

        if verbose:
            print("\n=== Summary ===")
            if internal_corr_before is not None:
                print(
                    f"Current Portfolio Internal Correlation: {internal_corr_before:.4f}"
                )
            if weighted_corr is not None:
                print(f"New Symbol vs Portfolio Correlation: {weighted_corr:.4f}")
            if portfolio_return_corr is not None:
                print(
                    f"Portfolio Return vs New Symbol Correlation: {portfolio_return_corr:.4f}"
                )
            if "internal_correlation" in result["after"]:
                print(
                    f"Portfolio Internal Correlation After: {result['after']['internal_correlation']:.4f}"
                )
            if "correlation_change" in result["impact"]:
                change = result["impact"]["correlation_change"]
                improvement = result["impact"]["diversification_improvement"]
                change_color = Fore.GREEN if improvement else Fore.YELLOW
                print(
                    f"Correlation Change: {color_text(f'{change:+.4f}', change_color)}"
                )
                print(
                    f"Diversification Improvement: {color_text(str(improvement), Fore.GREEN if improvement else Fore.RED)}"
                )

        return result

    def calculate_weighted_correlation_with_new_symbol(self, new_symbol: str, verbose: bool = True):
        """Calculates weighted correlation between a new symbol and the entire portfolio based on position sizes."""
        correlations = []
        weights = []
        position_details = []

        if verbose:
            print(
                color_text(
                    "\nCorrelation Analysis (Weighted by Position Size):", Fore.CYAN
                )
            )

        new_df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(new_symbol)
        new_series = self.data_fetcher.dataframe_to_close_series(new_df)
        if new_series is None:
            if verbose:
                print(
                    color_text(
                        f"Could not fetch price history for {new_symbol}", Fore.RED
                    )
                )
            return None, []

        for pos in self.positions:
            pos_series = self._fetch_symbol_series(pos.symbol)

            if pos_series is not None:
                df = pd.concat([pos_series, new_series], axis=1, join="inner")
                if len(df) < DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS:
                    continue

                # Calculate correlation on returns (pct_change) instead of prices
                # to avoid spurious correlation from non-stationary price series
                returns_df = df.pct_change().dropna()
                if len(returns_df) < DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS:
                    continue

                # Adjust returns based on position direction (LONG/SHORT)
                # SHORT positions have inverted returns for PnL correlation
                adjusted_returns = returns_df.copy()
                if pos.direction == "SHORT":
                    adjusted_returns.iloc[:, 0] = -adjusted_returns.iloc[:, 0]

                corr = adjusted_returns.iloc[:, 0].corr(adjusted_returns.iloc[:, 1])
                if pd.isna(corr):
                    continue

                weight = pos.size_usdt

                correlations.append(corr)
                weights.append(weight)

                position_details.append(
                    {
                        "symbol": pos.symbol,
                        "direction": pos.direction,
                        "size": pos.size_usdt,
                        "correlation": corr,
                        "weight": weight,
                    }
                )

        if not correlations:
            if verbose:
                print(
                    color_text(
                        "Insufficient data for correlation analysis.", Fore.YELLOW
                    )
                )
            return None, []

        total_weight = sum(weights)
        weighted_corr = sum(c * w for c, w in zip(correlations, weights)) / total_weight

        if verbose:
            print("\nIndividual Correlations:")
            for detail in position_details:
                corr_color = (
                    Fore.GREEN
                    if abs(detail["correlation"]) > 0.7
                    else (Fore.YELLOW if abs(detail["correlation"]) > 0.4 else Fore.RED)
                )
                weight_pct = (detail["weight"] / total_weight) * 100
                print(
                    f"  {detail['symbol']:12} ({detail['direction']:5}, {detail['size']:>8.2f} USDT, {weight_pct:>5.1f}%): "
                    + color_text(f"{detail['correlation']:>6.4f}", corr_color)
                )

            print(
                f"\n{color_text('Weighted Portfolio Correlation:', Fore.CYAN, Style.BRIGHT)}"
            )
            weighted_corr_color = (
                Fore.GREEN
                if abs(weighted_corr) > 0.7
                else (Fore.YELLOW if abs(weighted_corr) > 0.4 else Fore.RED)
            )
            print(
                f"  {new_symbol} vs Portfolio: {color_text(f'{weighted_corr:>6.4f}', weighted_corr_color, Style.BRIGHT)}"
            )

        return weighted_corr, position_details

    def calculate_portfolio_return_correlation(
        self,
        new_symbol: str,
        min_points: int = DEFAULT_CORRELATION_MIN_POINTS,
        verbose: bool = True,
    ):
        """Calculates correlation between the portfolio's aggregated return and the new symbol."""
        if verbose:
            print(color_text("\nPortfolio Return Correlation Analysis:", Fore.CYAN))

        if not self.positions:
            if verbose:
                print(
                    color_text(
                        "No positions in portfolio to compare against.", Fore.YELLOW
                    )
                )
            return None, {}

        new_df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(new_symbol)
        new_series = self.data_fetcher.dataframe_to_close_series(new_df)
        if new_series is None:
            if verbose:
                print(
                    color_text(
                        f"Could not fetch price history for {new_symbol}", Fore.RED
                    )
                )
            return None, {}

        symbol_series = self._get_portfolio_series_dict()

        if not symbol_series:
            if verbose:
                print(
                    color_text(
                        "Unable to fetch history for existing positions.", Fore.YELLOW
                    )
                )
            return None, {}

        price_df = pd.DataFrame(symbol_series).dropna(how="all")
        if price_df.empty:
            if verbose:
                print(
                    color_text(
                        "Insufficient overlapping data among current positions.",
                        Fore.YELLOW,
                    )
                )
            return None, {}

        portfolio_returns_df = price_df.pct_change().dropna(how="all")
        new_returns = new_series.pct_change().dropna()

        if portfolio_returns_df.empty or new_returns.empty:
            if verbose:
                print(
                    color_text(
                        "Insufficient price history to compute returns.", Fore.YELLOW
                    )
                )
            return None, {}

        common_index = portfolio_returns_df.index.intersection(new_returns.index)
        if len(common_index) < min_points:
            if verbose:
                print(
                    color_text(
                        f"Need at least {min_points} overlapping points, found {len(common_index)}.",
                        Fore.YELLOW,
                    )
                )
            return None, {}

        # Vectorized approach: Adjust returns for LONG/SHORT and calculate weighted portfolio returns
        # Create adjusted returns DataFrame (invert SHORT positions)
        adjusted_returns_df = portfolio_returns_df.copy()
        for pos in self.positions:
            if pos.symbol in adjusted_returns_df.columns and pos.direction == "SHORT":
                adjusted_returns_df[pos.symbol] = -adjusted_returns_df[pos.symbol]

        # Create weights Series for each position
        position_weights = {}
        for pos in self.positions:
            if pos.symbol in adjusted_returns_df.columns:
                position_weights[pos.symbol] = abs(pos.size_usdt)

        # Filter to only symbols that exist in both adjusted_returns_df and position_weights
        valid_symbols = [
            sym for sym in adjusted_returns_df.columns if sym in position_weights
        ]
        if not valid_symbols:
            if verbose:
                print(
                    color_text(
                        "No valid positions for portfolio return calculation.",
                        Fore.YELLOW,
                    )
                )
            return None, {}

        # Calculate weighted portfolio returns using vectorization
        # Select only valid symbols and common index
        adjusted_common = adjusted_returns_df.loc[common_index, valid_symbols]
        weights_array = np.array([position_weights[sym] for sym in valid_symbols])

        # Calculate weighted sum for each row (index)
        weighted_sums = (adjusted_common * weights_array).sum(axis=1)
        total_weights = adjusted_common.notna().dot(weights_array)

        # Calculate weighted average portfolio returns (avoid division by zero)
        portfolio_return_series = weighted_sums / total_weights.replace(0, np.nan)

        # Align new returns with common index
        new_return_series = new_returns.loc[common_index]

        # Remove rows where either series has NaN
        valid_mask = portfolio_return_series.notna() & new_return_series.notna()
        portfolio_return_series = portfolio_return_series[valid_mask]
        new_return_series = new_return_series[valid_mask]

        if len(portfolio_return_series) < min_points:
            if verbose:
                print(
                    color_text(
                        "Not enough aligned return samples for correlation.",
                        Fore.YELLOW,
                    )
                )
            return None, {"samples": len(portfolio_return_series)}

        correlation = portfolio_return_series.corr(new_return_series)

        if pd.isna(correlation):
            if verbose:
                print(
                    color_text(
                        "Unable to compute correlation (insufficient variance).",
                        Fore.YELLOW,
                    )
                )
            return None, {"samples": len(portfolio_return_series)}

        if verbose:
            corr_color = (
                Fore.GREEN
                if abs(correlation) > 0.7
                else (Fore.YELLOW if abs(correlation) > 0.4 else Fore.RED)
            )
            print(
                f"  Portfolio Return vs {new_symbol}: {color_text(f'{correlation:>6.4f}', corr_color, Style.BRIGHT)}"
            )
            print(color_text(f"  Samples used: {len(portfolio_return_series)}", Fore.WHITE))

        return correlation, {"samples": len(portfolio_return_series)}
