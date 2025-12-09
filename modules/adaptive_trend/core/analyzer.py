"""
ATC Symbol Analyzer.

This module provides functions for analyzing individual symbols using
Adaptive Trend Classification (ATC).
"""

import traceback
from typing import TYPE_CHECKING, Optional, Dict, Any

if TYPE_CHECKING:
    from modules.common.DataFetcher import DataFetcher
    import pandas as pd

try:
    from modules.common.utils import (
        log_error,
        log_progress,
    )
except ImportError:
    def log_error(message: str) -> None:
        print(f"[ERROR] {message}")
    
    def log_progress(message: str) -> None:
        print(f"[PROGRESS] {message}")

from modules.adaptive_trend.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend.utils.config import ATCConfig

__all__ = ["analyze_symbol"]


def analyze_symbol(
    symbol: str,
    data_fetcher: "DataFetcher",
    config: ATCConfig,
) -> Optional[Dict[str, Any]]:
    """
    Analyze a single symbol using ATC.

    This function computes ATC signals and returns the results. It does not
    handle display - that should be done by the calling code.

    Args:
        symbol: Symbol to analyze
        data_fetcher: DataFetcher instance
        config: ATCConfig containing all ATC parameters

    Returns:
        Dictionary containing analysis results with keys:
            - symbol: Symbol name
            - df: OHLCV DataFrame
            - atc_results: ATC signals dictionary
            - current_price: Current price
            - exchange_label: Exchange identifier
        Returns None if analysis failed.
    """
    try:
        # Fetch OHLCV data
        df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol,
            limit=config.limit,
            timeframe=config.timeframe,
            check_freshness=True,
        )

        if df is None or df.empty:
            log_error(f"No data available for {symbol}")
            return None

        exchange_label = exchange_id.upper() if exchange_id else "UNKNOWN"

        # Get close prices
        if "close" not in df.columns:
            log_error(f"No 'close' column in data for {symbol}")
            return None

        close_prices = df["close"]
        current_price = close_prices.iloc[-1]

        # Calculate ATC signals
        log_progress(f"Calculating ATC signals for {symbol}...")

        atc_results = compute_atc_signals(
            prices=close_prices,
            src=None,  # Use close prices as source
            ema_len=config.ema_len,
            hull_len=config.hma_len,
            wma_len=config.wma_len,
            dema_len=config.dema_len,
            lsma_len=config.lsma_len,
            kama_len=config.kama_len,
            ema_w=1.0,
            hma_w=1.0,
            wma_w=1.0,
            dema_w=1.0,
            lsma_w=1.0,
            kama_w=1.0,
            robustness=config.robustness,
            La=config.lambda_param,
            De=config.decay,
            cutout=config.cutout,
        )

        # Return results instead of displaying
        return {
            "symbol": symbol,
            "df": df,
            "atc_results": atc_results,
            "current_price": current_price,
            "exchange_label": exchange_label,
        }

    except Exception as e:
        log_error(f"Error analyzing {symbol}: {type(e).__name__}: {e}")
        log_error(f"Traceback: {traceback.format_exc()}")
        return None

