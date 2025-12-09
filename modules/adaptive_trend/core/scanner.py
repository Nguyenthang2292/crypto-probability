"""ATC Symbol Scanner.

This module provides functions for scanning multiple symbols using
Adaptive Trend Classification (ATC) to find LONG/SHORT signals.

The scanner fetches data for multiple symbols, calculates ATC signals,
and filters results based on signal strength and trend direction.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

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

from modules.adaptive_trend.core.compute_atc_signals import compute_atc_signals
from modules.adaptive_trend.core.process_layer1 import trend_sign
from modules.adaptive_trend.utils.config import ATCConfig


def _scan_sequential(
    symbols: list,
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
    min_signal: float,
) -> Tuple[list, int, int, list]:
    """Scan symbols sequentially (one by one)."""
    results = []
    skipped_count = 0
    error_count = 0
    skipped_symbols = []
    total = len(symbols)
    
    for idx, symbol in enumerate(symbols, 1):
        try:
            result = _process_symbol(symbol, data_fetcher, atc_config, min_signal)
            
            if result is None:
                skipped_count += 1
                skipped_symbols.append(symbol)
            else:
                results.append(result)
            
            # Progress update every 10 symbols
            if idx % 10 == 0 or idx == total:
                log_progress(
                    f"Scanned {idx}/{total} symbols... "
                    f"Found {len(results)} signals, "
                    f"Skipped {skipped_count}, Errors {error_count}"
                )
        except KeyboardInterrupt:
            log_warn("Scan interrupted by user")
            break
        except Exception as e:
            error_count += 1
            skipped_symbols.append(symbol)
            log_warn(
                f"Error processing symbol {symbol}: {type(e).__name__}: {e}. "
                f"Skipping and continuing..."
            )
    
    return results, skipped_count, error_count, skipped_symbols


def _scan_threadpool(
    symbols: list,
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
    min_signal: float,
    max_workers: Optional[int],
) -> Tuple[list, int, int, list]:
    """Scan symbols using ThreadPoolExecutor for parallel data fetching."""
    if max_workers is None:
        max_workers = min(32, len(symbols) + 4)
    
    results = []
    skipped_count = 0
    error_count = 0
    skipped_symbols = []
    total = len(symbols)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(_process_symbol, symbol, data_fetcher, atc_config, min_signal): symbol
            for symbol in symbols
        }
        
        # Process completed tasks
        try:
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result is None:
                        skipped_count += 1
                        skipped_symbols.append(symbol)
                    else:
                        results.append(result)
                except Exception as e:
                    error_count += 1
                    skipped_symbols.append(symbol)
                    log_warn(
                        f"Error processing symbol {symbol}: {type(e).__name__}: {e}. "
                        f"Skipping and continuing..."
                    )
                
                # Progress update every 10 symbols
                if completed % 10 == 0 or completed == total:
                    log_progress(
                        f"Scanned {completed}/{total} symbols... "
                        f"Found {len(results)} signals, "
                        f"Skipped {skipped_count}, Errors {error_count}"
                    )
        except KeyboardInterrupt:
            log_warn("Scan interrupted by user")
            # Cancel remaining tasks
            for future in future_to_symbol:
                future.cancel()
    
    return results, skipped_count, error_count, skipped_symbols


async def _process_symbol_async(
    symbol: str,
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
    min_signal: float,
    loop: asyncio.AbstractEventLoop,
) -> Optional[Dict[str, Any]]:
    """Async wrapper for _process_symbol using asyncio.to_thread."""
    try:
        result = await loop.run_in_executor(
            None,
            _process_symbol,
            symbol,
            data_fetcher,
            atc_config,
            min_signal,
        )
        return result
    except Exception:
        return None


def _scan_asyncio(
    symbols: list,
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
    min_signal: float,
    max_workers: Optional[int],
) -> Tuple[list, int, int, list]:
    """Scan symbols using asyncio for parallel data fetching."""
    async def _async_scan():
        loop = asyncio.get_event_loop()
        if max_workers is not None:
            # Use semaphore to limit concurrent tasks
            semaphore = asyncio.Semaphore(max_workers)
            
            async def _process_with_semaphore(symbol):
                async with semaphore:
                    result = await _process_symbol_async(symbol, data_fetcher, atc_config, min_signal, loop)
                    return symbol, result
            
            tasks = [_process_with_semaphore(symbol) for symbol in symbols]
        else:
            # Wrap to include symbol
            async def _wrap_with_symbol(symbol):
                result = await _process_symbol_async(symbol, data_fetcher, atc_config, min_signal, loop)
                return symbol, result
            tasks = [_wrap_with_symbol(symbol) for symbol in symbols]
        
        results = []
        skipped_count = 0
        error_count = 0
        skipped_symbols = []
        total = len(symbols)
        completed = 0
        
        try:
            # Process results as they complete
            for coro in asyncio.as_completed(tasks):
                try:
                    symbol, result = await coro
                    completed += 1
                    
                    if result is None:
                        skipped_count += 1
                        skipped_symbols.append(symbol)
                    else:
                        results.append(result)
                    
                    # Progress update every 10 symbols
                    if completed % 10 == 0 or completed == total:
                        log_progress(
                            f"Scanned {completed}/{total} symbols... "
                            f"Found {len(results)} signals, "
                            f"Skipped {skipped_count}, Errors {error_count}"
                        )
                except Exception as e:
                    error_count += 1
                    completed += 1
                    # Try to get symbol from task if possible
                    skipped_symbols.append("UNKNOWN")
                    log_warn(
                        f"Error processing symbol: {type(e).__name__}: {e}. "
                        f"Skipping and continuing..."
                    )
        except KeyboardInterrupt:
            log_warn("Scan interrupted by user")
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
        
        return results, skipped_count, error_count, skipped_symbols
    
    # Run async function
    try:
        return asyncio.run(_async_scan())
    except RuntimeError:
        # If we're already in an event loop, use nest_asyncio or create new loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_async_scan())
        finally:
            loop.close()


def _process_symbol(
    symbol: str,
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
    min_signal: float,
) -> Optional[Dict[str, Any]]:
    """
    Process a single symbol: fetch data and calculate ATC signals.
    
    Args:
        symbol: Symbol to process
        data_fetcher: DataFetcher instance
        atc_config: ATCConfig object
        min_signal: Minimum signal strength threshold
        
    Returns:
        Dictionary with symbol data if signal found, None otherwise
    """
    try:
        # Fetch OHLCV data
        df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol,
            limit=atc_config.limit,
            timeframe=atc_config.timeframe,
            check_freshness=True,
        )

        if df is None or df.empty:
            return None
        
        if "close" not in df.columns:
            return None

        close_prices = df["close"]
        
        # Validate we have enough data
        if len(close_prices) < atc_config.limit:
            return None
        
        current_price = close_prices.iloc[-1]
        
        # Validate price is valid
        if pd.isna(current_price) or current_price <= 0:
            return None

        # Calculate ATC signals
        atc_results = compute_atc_signals(
            prices=close_prices,
            src=None,
            ema_len=atc_config.ema_len,
            hull_len=atc_config.hma_len,
            wma_len=atc_config.wma_len,
            dema_len=atc_config.dema_len,
            lsma_len=atc_config.lsma_len,
            kama_len=atc_config.kama_len,
            ema_w=1.0,
            hma_w=1.0,
            wma_w=1.0,
            dema_w=1.0,
            lsma_w=1.0,
            kama_w=1.0,
            robustness=atc_config.robustness,
            La=atc_config.lambda_param,
            De=atc_config.decay,
            cutout=atc_config.cutout,
        )

        average_signal = atc_results.get("Average_Signal")
        if average_signal is None or average_signal.empty:
            return None

        latest_signal = average_signal.iloc[-1]
        
        # Validate signal is not NaN
        if pd.isna(latest_signal):
            return None
        
        latest_trend = trend_sign(average_signal)
        latest_trend_value = latest_trend.iloc[-1] if not latest_trend.empty else 0

        # Only include signals above threshold
        if abs(latest_signal) < min_signal:
            return None

        return {
            "symbol": symbol,
            "signal": latest_signal,
            "trend": latest_trend_value,
            "price": current_price,
            "exchange": exchange_id or "UNKNOWN",
        }
    except Exception:
        # Return None on any error - errors are logged in the calling function
        return None


def scan_all_symbols(
    data_fetcher: "DataFetcher",
    atc_config: ATCConfig,
    max_symbols: Optional[int] = None,
    min_signal: float = 0.01,
    execution_mode: str = "threadpool",
    max_workers: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scan all futures symbols and filter those with LONG/SHORT signals.
    
    Fetches OHLCV data for multiple symbols, calculates ATC signals for each,
    and returns DataFrames containing symbols with signals above the threshold,
    separated into LONG (trend > 0) and SHORT (trend < 0) signals.
    
    Execution modes:
    - "sequential": Process symbols one by one (safest, avoids rate limits)
    - "threadpool": Use ThreadPoolExecutor for parallel data fetching (default, faster)
    - "asyncio": Use asyncio for parallel data fetching (fastest, but requires async support)
    
    Args:
        data_fetcher: DataFetcher instance for fetching market data.
        atc_config: ATCConfig object containing all ATC parameters.
        max_symbols: Maximum number of symbols to scan (None = all symbols).
        min_signal: Minimum signal strength to include (must be >= 0).
        execution_mode: Execution mode - "sequential", "threadpool", or "asyncio" (default: "threadpool").
        max_workers: Maximum number of worker threads/processes for parallel execution.
                    If None, uses default (min(32, num_symbols + 4) for threadpool).
        
    Returns:
        Tuple of two DataFrames:
        - long_signals_df: Symbols with bullish signals (trend > 0), sorted by signal strength
        - short_signals_df: Symbols with bearish signals (trend < 0), sorted by signal strength
        
        Each DataFrame contains columns: symbol, signal, trend, price, exchange.
        
    Raises:
        ValueError: If any parameter is invalid.
        TypeError: If data_fetcher is None or missing required methods.
        AttributeError: If data_fetcher doesn't have required methods.
    """
    # Input validation
    if data_fetcher is None:
        raise ValueError("data_fetcher cannot be None")
    
    if not isinstance(atc_config, ATCConfig):
        raise ValueError(f"atc_config must be an ATCConfig instance, got {type(atc_config)}")
    
    # Validate data_fetcher has required methods
    required_methods = ["list_binance_futures_symbols", "fetch_ohlcv_with_fallback_exchange"]
    for method_name in required_methods:
        if not hasattr(data_fetcher, method_name):
            raise AttributeError(
                f"data_fetcher must have method '{method_name}', "
                f"got {type(data_fetcher)}"
            )
    
    if not isinstance(atc_config.timeframe, str) or not atc_config.timeframe.strip():
        raise ValueError(f"atc_config.timeframe must be a non-empty string, got {atc_config.timeframe}")
    
    if not isinstance(atc_config.limit, int) or atc_config.limit <= 0:
        raise ValueError(f"atc_config.limit must be a positive integer, got {atc_config.limit}")
    
    # Validate all MA lengths
    ma_lengths = {
        "ema_len": atc_config.ema_len,
        "hma_len": atc_config.hma_len,
        "wma_len": atc_config.wma_len,
        "dema_len": atc_config.dema_len,
        "lsma_len": atc_config.lsma_len,
        "kama_len": atc_config.kama_len,
    }
    for name, length in ma_lengths.items():
        if not isinstance(length, int) or length <= 0:
            raise ValueError(f"atc_config.{name} must be a positive integer, got {length}")
    
    # Validate robustness
    VALID_ROBUSTNESS = {"Narrow", "Medium", "Wide"}
    if atc_config.robustness not in VALID_ROBUSTNESS:
        raise ValueError(
            f"atc_config.robustness must be one of {VALID_ROBUSTNESS}, got {atc_config.robustness}"
        )
    
    # Validate lambda_param
    if not isinstance(atc_config.lambda_param, (int, float)) or np.isnan(atc_config.lambda_param) or np.isinf(atc_config.lambda_param):
        raise ValueError(f"atc_config.lambda_param must be a finite number, got {atc_config.lambda_param}")
    
    # Validate decay
    if not isinstance(atc_config.decay, (int, float)) or not (0 <= atc_config.decay <= 1):
        raise ValueError(f"atc_config.decay must be between 0 and 1, got {atc_config.decay}")
    
    # Validate cutout
    if not isinstance(atc_config.cutout, int) or atc_config.cutout < 0:
        raise ValueError(f"atc_config.cutout must be a non-negative integer, got {atc_config.cutout}")
    
    # Validate max_symbols
    if max_symbols is not None and (not isinstance(max_symbols, int) or max_symbols <= 0):
        raise ValueError(f"max_symbols must be a positive integer or None, got {max_symbols}")
    
    # Validate min_signal
    if not isinstance(min_signal, (int, float)) or min_signal < 0:
        raise ValueError(f"min_signal must be a non-negative number, got {min_signal}")
    
    # Validate execution_mode
    VALID_MODES = {"sequential", "threadpool", "asyncio"}
    if execution_mode not in VALID_MODES:
        raise ValueError(
            f"execution_mode must be one of {VALID_MODES}, got {execution_mode}"
        )
    
    # Validate max_workers
    if max_workers is not None and (not isinstance(max_workers, int) or max_workers <= 0):
        raise ValueError(f"max_workers must be a positive integer or None, got {max_workers}")

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
            log_success(
                f"Found {len(all_symbols)} futures symbols, "
                f"scanning first {len(symbols)} symbols"
            )
        else:
            symbols = all_symbols
            log_success(f"Found {len(symbols)} futures symbols")
        
        log_progress(f"Scanning {len(symbols)} symbols for ATC signals using {execution_mode} mode...")

        # Route to appropriate execution method
        if execution_mode == "sequential":
            results, skipped_count, error_count, skipped_symbols = _scan_sequential(
                symbols, data_fetcher, atc_config, min_signal
            )
        elif execution_mode == "threadpool":
            results, skipped_count, error_count, skipped_symbols = _scan_threadpool(
                symbols, data_fetcher, atc_config, min_signal, max_workers
            )
        elif execution_mode == "asyncio":
            results, skipped_count, error_count, skipped_symbols = _scan_asyncio(
                symbols, data_fetcher, atc_config, min_signal, max_workers
            )
        else:
            # Fallback to sequential
            results, skipped_count, error_count, skipped_symbols = _scan_sequential(
                symbols, data_fetcher, atc_config, min_signal
            )
        
        total = len(symbols)

        # Summary logging
        log_progress(
            f"Scan complete: {total} total, {len(results)} signals found, "
            f"{skipped_count} skipped, {error_count} errors"
        )
        
        if skipped_count > 0 and len(skipped_symbols) <= 10:
            log_warn(f"Skipped symbols: {', '.join(skipped_symbols)}")
        elif skipped_count > 10:
            log_warn(f"Skipped {skipped_count} symbols (first 10: {', '.join(skipped_symbols[:10])}...)")

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

        log_success(
            f"Found {len(long_signals)} LONG signals and {len(short_signals)} SHORT signals"
        )

        return long_signals, short_signals

    except KeyboardInterrupt:
        log_warn("Scan interrupted by user")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        log_error(f"Fatal error scanning symbols: {type(e).__name__}: {e}")
        import traceback
        log_error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), pd.DataFrame()

