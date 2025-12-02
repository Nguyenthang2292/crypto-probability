"""
ATC Symbol Analyzer.

This module provides functions for analyzing individual symbols using
Adaptive Trend Classification (ATC).
"""

import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.common.DataFetcher import DataFetcher

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

from modules.adaptive_trend.atc import compute_atc_signals
from modules.adaptive_trend.cli.display import display_atc_signals


def analyze_symbol(
    symbol: str,
    data_fetcher: "DataFetcher",
    timeframe: str,
    limit: int,
    ema_len: int,
    hma_len: int,
    wma_len: int,
    dema_len: int,
    lsma_len: int,
    kama_len: int,
    robustness: str,
    lambda_param: float,
    decay: float,
    cutout: int,
) -> bool:
    """
    Analyze a single symbol using ATC.

    Args:
        symbol: Symbol to analyze
        data_fetcher: DataFetcher instance
        timeframe: Timeframe for data
        limit: Number of candles
        ema_len: EMA length
        hma_len: HMA length
        wma_len: WMA length
        dema_len: DEMA length
        lsma_len: LSMA length
        kama_len: KAMA length
        robustness: Robustness setting
        lambda_param: Lambda parameter
        decay: Decay rate
        cutout: Cutout period

    Returns:
        bool: True if analysis succeeded, False otherwise
    """
    try:
        # Fetch OHLCV data
        df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol,
            limit=limit,
            timeframe=timeframe,
            check_freshness=True,
        )

        if df is None or df.empty:
            log_error(f"No data available for {symbol}")
            return False

        exchange_label = exchange_id.upper() if exchange_id else "UNKNOWN"

        # Get close prices
        if "close" not in df.columns:
            log_error(f"No 'close' column in data for {symbol}")
            return False

        close_prices = df["close"]
        current_price = close_prices.iloc[-1]

        # Calculate ATC signals
        log_progress(f"Calculating ATC signals for {symbol}...")

        atc_results = compute_atc_signals(
            prices=close_prices,
            src=None,  # Use close prices as source
            ema_len=ema_len,
            hull_len=hma_len,
            wma_len=wma_len,
            dema_len=dema_len,
            lsma_len=lsma_len,
            kama_len=kama_len,
            ema_w=1.0,
            hma_w=1.0,
            wma_w=1.0,
            dema_w=1.0,
            lsma_w=1.0,
            kama_w=1.0,
            robustness=robustness,
            La=lambda_param,
            De=decay,
            cutout=cutout,
        )

        # Display results
        display_atc_signals(
            symbol=symbol,
            df=df,
            atc_results=atc_results,
            current_price=current_price,
            exchange_label=exchange_label,
        )

        return True

    except Exception as e:
        log_error(f"Error analyzing {symbol}: {type(e).__name__}: {e}")
        log_error(f"Traceback: {traceback.format_exc()}")
        return False

