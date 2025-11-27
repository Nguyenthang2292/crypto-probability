"""
Performance analyzer for calculating symbol performance scores across multiple timeframes.
"""

import math
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from colorama import Fore, Style

try:
    from modules.config import (
        PAIRS_TRADING_WEIGHTS,
        PAIRS_TRADING_TOP_N,
        PAIRS_TRADING_MIN_CANDLES,
        PAIRS_TRADING_TIMEFRAME,
        PAIRS_TRADING_LIMIT,
    )
    from modules.common.utils import color_text, normalize_symbol, timeframe_to_minutes
    from modules.common.ProgressBar import ProgressBar
except ImportError:
    PAIRS_TRADING_WEIGHTS = {'1d': 0.5, '3d': 0.3, '1w': 0.2}
    PAIRS_TRADING_TOP_N = 5
    PAIRS_TRADING_MIN_CANDLES = 168
    PAIRS_TRADING_TIMEFRAME = "1h"
    PAIRS_TRADING_LIMIT = 200
    color_text = None
    normalize_symbol = None

    def timeframe_to_minutes(timeframe: str) -> int:
        match = re.match(r"^\s*(\d+)\s*([mhdw])\s*$", timeframe.lower())
        if not match:
            return 60
        value, unit = match.groups()
        value = int(value)
        if unit == "m":
            return value
        if unit == "h":
            return value * 60
        if unit == "d":
            return value * 60 * 24
        if unit == "w":
            return value * 60 * 24 * 7
        return 60

    ProgressBar = None


class PerformanceAnalyzer:
    """
    Analyzes performance of trading symbols across multiple timeframes.
    
    Calculates weighted performance scores from 1 day, 3 days, and 1 week returns.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        min_candles: int = PAIRS_TRADING_MIN_CANDLES,
        timeframe: str = PAIRS_TRADING_TIMEFRAME,
        limit: int = PAIRS_TRADING_LIMIT,
    ):
        """
        Initialize PerformanceAnalyzer.

        Args:
            weights: Dictionary with weights for '1d', '3d', '1w' timeframes.
                     Default: {'1d': 0.5, '3d': 0.3, '1w': 0.2}
            min_candles: Minimum number of candles required for analysis (default: 168)
            timeframe: Timeframe for OHLCV data (default: '1h')
            limit: Number of candles to fetch (default: 200)
        """
        self.weights = weights or PAIRS_TRADING_WEIGHTS.copy()
        self.min_candles = min_candles
        self.timeframe = timeframe
        self.limit = limit
        
        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight}. "
                f"Current weights: {self.weights}"
            )

    def calculate_performance_score(
        self, symbol: str, df: pd.DataFrame
    ) -> Optional[Dict[str, float]]:
        """
        Calculate performance score for a symbol from OHLCV DataFrame.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            df: DataFrame with OHLCV data, must have 'close' column and be sorted by timestamp

        Returns:
            Dictionary with keys:
                - 'symbol': Symbol name
                - 'score': Weighted performance score
                - '1d_return': 1-day return percentage
                - '3d_return': 3-day return percentage
                - '1w_return': 1-week return percentage
                - 'current_price': Current closing price
            Returns None if insufficient data or calculation fails.
        """
        if df is None or df.empty:
            return None

        if 'close' not in df.columns:
            return None

        # Ensure DataFrame is sorted by timestamp (ascending)
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        # Check if we have enough data
        if len(df) < self.min_candles:
            return None

        try:
            close_prices = df['close'].values
            current_price = float(close_prices[-1])

            # Calculate returns for different timeframes
            minutes_per_candle = max(1, timeframe_to_minutes(self.timeframe))

            def _candles_for_days(days: int) -> int:
                period_minutes = days * 24 * 60
                return max(1, math.ceil(period_minutes / minutes_per_candle))

            candles_1d = _candles_for_days(1)
            candles_3d = _candles_for_days(3)
            candles_1w = _candles_for_days(7)

            # Calculate returns
            returns = {}
            
            # 1-day return
            if len(close_prices) >= candles_1d + 1:
                price_1d_ago = float(close_prices[-(candles_1d + 1)])
                if price_1d_ago > 0:
                    returns['1d'] = (current_price - price_1d_ago) / price_1d_ago
                else:
                    returns['1d'] = 0.0
            else:
                returns['1d'] = 0.0

            # 3-day return
            if len(close_prices) >= candles_3d + 1:
                price_3d_ago = float(close_prices[-(candles_3d + 1)])
                if price_3d_ago > 0:
                    returns['3d'] = (current_price - price_3d_ago) / price_3d_ago
                else:
                    returns['3d'] = 0.0
            else:
                returns['3d'] = 0.0

            # 1-week return
            if len(close_prices) >= candles_1w + 1:
                price_1w_ago = float(close_prices[-(candles_1w + 1)])
                if price_1w_ago > 0:
                    returns['1w'] = (current_price - price_1w_ago) / price_1w_ago
                else:
                    returns['1w'] = 0.0
            else:
                returns['1w'] = 0.0

            # Calculate weighted score
            score = (
                returns['1d'] * self.weights.get('1d', 0.0) +
                returns['3d'] * self.weights.get('3d', 0.0) +
                returns['1w'] * self.weights.get('1w', 0.0)
            )

            return {
                'symbol': symbol,
                'score': float(score),
                '1d_return': float(returns['1d']),
                '3d_return': float(returns['3d']),
                '1w_return': float(returns['1w']),
                'current_price': current_price,
            }

        except (ValueError, IndexError, KeyError) as e:
            # Return None on any calculation error
            return None

    def analyze_all_symbols(
        self,
        symbols: List[str],
        data_fetcher,
        verbose: bool = True,
        shutdown_event=None,
    ) -> pd.DataFrame:
        """
        Analyze performance for all symbols.

        Args:
            symbols: List of trading symbols to analyze
            data_fetcher: DataFetcher instance for fetching OHLCV data
            verbose: If True, print progress messages
            shutdown_event: Optional threading.Event to signal shutdown

        Returns:
            DataFrame with columns: ['symbol', 'score', '1d_return', '3d_return', 
                                   '1w_return', 'current_price']
            Sorted by score (descending). Symbols with insufficient data are excluded.
        """
        if not symbols:
            return pd.DataFrame(
                columns=['symbol', 'score', '1d_return', '3d_return', '1w_return', 'current_price']
            )

        results = []
        progress = None

        if verbose and ProgressBar:
            progress = ProgressBar(len(symbols), "Performance Analysis")

        for symbol in symbols:
            # Check for shutdown signal
            if shutdown_event and shutdown_event.is_set():
                if verbose:
                    print(color_text("\nAnalysis aborted due to shutdown signal.", Fore.YELLOW))
                break

            try:
                # Fetch OHLCV data
                df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                    symbol,
                    limit=self.limit,
                    timeframe=self.timeframe,
                    check_freshness=False,
                )

                if df is None or df.empty:
                    if verbose:
                        print(
                            color_text(
                                f"  Skipping {symbol}: No data available",
                                Fore.YELLOW,
                            )
                        )
                    if progress:
                        progress.update()
                    continue

                # Calculate performance score
                result = self.calculate_performance_score(symbol, df)

                if result is None:
                    if verbose:
                        print(
                            color_text(
                                f"  Skipping {symbol}: Insufficient data or calculation failed",
                                Fore.YELLOW,
                            )
                        )
                    if progress:
                        progress.update()
                    continue

                results.append(result)

                if verbose:
                    score_pct = result['score'] * 100
                    try:
                        print(
                            color_text(
                                f"  {symbol}: Score {score_pct:+.2f}% "
                                f"(1d: {result['1d_return']*100:+.2f}%, "
                                f"3d: {result['3d_return']*100:+.2f}%, "
                                f"1w: {result['1w_return']*100:+.2f}%)",
                                Fore.GREEN if score_pct > 0 else Fore.RED,
                            )
                        )
                    except UnicodeEncodeError:
                        # Fallback for encoding issues
                        print(
                            f"  {symbol}: Score {score_pct:+.2f}% "
                            f"(1d: {result['1d_return']*100:+.2f}%, "
                            f"3d: {result['3d_return']*100:+.2f}%, "
                            f"1w: {result['1w_return']*100:+.2f}%)"
                        )

            except Exception as e:
                if verbose:
                    print(
                        color_text(
                            f"  Error analyzing {symbol}: {e}",
                            Fore.RED,
                        )
                    )
            finally:
                if progress:
                    progress.update()

        if progress:
            progress.finish()

        if not results:
            if verbose:
                print(
                    color_text(
                        "No valid performance data found for any symbols.",
                        Fore.YELLOW,
                    )
                )
            return pd.DataFrame(
                columns=['symbol', 'score', '1d_return', '3d_return', '1w_return', 'current_price']
            )

        # Create DataFrame and sort by score (descending)
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('score', ascending=False).reset_index(drop=True)

        if verbose:
            print(
                color_text(
                    f"\nSuccessfully analyzed {len(df_results)}/{len(symbols)} symbols.",
                    Fore.GREEN,
                    Style.BRIGHT,
                )
            )

        return df_results

    def get_top_performers(
        self, df: pd.DataFrame, top_n: int = PAIRS_TRADING_TOP_N
    ) -> pd.DataFrame:
        """
        Get top N best performing symbols.

        Args:
            df: DataFrame from analyze_all_symbols()
            top_n: Number of top performers to return (default: 5)

        Returns:
            DataFrame with top N performers (sorted by score, descending)
        """
        if df is None or df.empty:
            return pd.DataFrame(
                columns=['symbol', 'score', '1d_return', '3d_return', '1w_return', 'current_price']
            )

        # DataFrame is already sorted by score descending
        return df.head(top_n).copy()

    def get_worst_performers(
        self, df: pd.DataFrame, top_n: int = PAIRS_TRADING_TOP_N
    ) -> pd.DataFrame:
        """
        Get top N worst performing symbols.

        Args:
            df: DataFrame from analyze_all_symbols()
            top_n: Number of worst performers to return (default: 5)

        Returns:
            DataFrame with top N worst performers (sorted by score, ascending)
        """
        if df is None or df.empty:
            return pd.DataFrame(
                columns=['symbol', 'score', '1d_return', '3d_return', '1w_return', 'current_price']
            )

        # Sort by score ascending and take top N
        df_sorted = df.sort_values('score', ascending=True).reset_index(drop=True)
        return df_sorted.head(top_n).copy()

