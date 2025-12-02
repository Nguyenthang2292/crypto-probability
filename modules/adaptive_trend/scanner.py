"""
ATC Symbol Scanner.

This module provides functions for scanning multiple symbols using
Adaptive Trend Classification (ATC) to find LONG/SHORT signals.
"""

import pandas as pd
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.common.DataFetcher import DataFetcher

try:
    from modules.common.utils import (
        log_error,
        log_warn,
        log_success,
        log_progress,
    )
except ImportError:
    def log_error(message: str) -> None:
        print(f"[ERROR] {message}")
    
    def log_warn(message: str) -> None:
        print(f"[WARN] {message}")
    
    def log_success(message: str) -> None:
        print(f"[SUCCESS] {message}")
    
    def log_progress(message: str) -> None:
        print(f"[PROGRESS] {message}")

from modules.adaptive_trend.atc import compute_atc_signals
from modules.adaptive_trend.layer1 import trend_sign


def scan_all_symbols(
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
    max_symbols: Optional[int] = None,
    min_signal: float = 0.01,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan all futures symbols and filter those with LONG/SHORT signals.
    
    Args:
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
        max_symbols: Maximum number of symbols to scan
        min_signal: Minimum signal strength to display
        
    Returns:
        tuple: (long_signals_df, short_signals_df) DataFrames with symbol, signal, price
    """
    try:
        log_progress("Fetching futures symbols from Binance...")
        all_symbols = data_fetcher.list_binance_futures_symbols(
            max_candidates=None,  # Get all symbols first
            progress_label="Symbol Discovery",
        )

        if not all_symbols:
            log_error("No symbols found")
            return pd.DataFrame(), pd.DataFrame()

        # Limit symbols if max_symbols specified
        if max_symbols and max_symbols > 0:
            symbols = all_symbols[:max_symbols]
            log_success(f"Found {len(all_symbols)} futures symbols, scanning first {len(symbols)} symbols")
        else:
            symbols = all_symbols
            log_success(f"Found {len(symbols)} futures symbols")
        
        log_progress(f"Scanning {len(symbols)} symbols for ATC signals...")

        results = []
        total = len(symbols)
        
        for idx, symbol in enumerate(symbols, 1):
            try:
                # Fetch OHLCV data
                df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                    symbol,
                    limit=limit,
                    timeframe=timeframe,
                    check_freshness=True,
                )

                if df is None or df.empty or "close" not in df.columns:
                    continue

                close_prices = df["close"]
                current_price = close_prices.iloc[-1]

                # Calculate ATC signals
                atc_results = compute_atc_signals(
                    prices=close_prices,
                    src=None,
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

                average_signal = atc_results.get("Average_Signal")
                if average_signal is None or average_signal.empty:
                    continue

                latest_signal = average_signal.iloc[-1]
                latest_trend = trend_sign(average_signal)
                latest_trend_value = latest_trend.iloc[-1] if not latest_trend.empty else 0

                # Only include signals above threshold
                if abs(latest_signal) < min_signal:
                    continue

                results.append({
                    "symbol": symbol,
                    "signal": latest_signal,
                    "trend": latest_trend_value,
                    "price": current_price,
                    "exchange": exchange_id or "UNKNOWN",
                })

                # Progress update every 10 symbols
                if idx % 10 == 0 or idx == total:
                    log_progress(f"Scanned {idx}/{total} symbols... Found {len(results)} signals")

            except KeyboardInterrupt:
                log_warn("Scan interrupted by user")
                break
            except Exception as e:
                # Skip symbols with errors, continue scanning
                continue

        if not results:
            log_warn("No signals found above threshold")
            return pd.DataFrame(), pd.DataFrame()

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Filter LONG and SHORT signals
        long_signals = results_df[results_df["trend"] > 0].copy()
        short_signals = results_df[results_df["trend"] < 0].copy()

        # Sort by signal strength (absolute value)
        long_signals = long_signals.sort_values("signal", ascending=False).reset_index(drop=True)
        short_signals = short_signals.sort_values("signal", ascending=True).reset_index(drop=True)

        return long_signals, short_signals

    except Exception as e:
        log_error(f"Error scanning symbols: {type(e).__name__}: {e}")
        import traceback
        log_error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), pd.DataFrame()

