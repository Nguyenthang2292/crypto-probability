"""
Data fetcher for retrieving market data from exchanges.
"""
import pandas as pd
from typing import Dict, Tuple, Optional
from colorama import Fore

try:
    from modules.utils import normalize_symbol, color_text
    from modules.ProgressBar import ProgressBar
    from modules.ExchangeManager import ExchangeManager
except ImportError:
    def normalize_symbol(user_input: str, quote: str = "USDT") -> str:
        if not user_input:
            return f"BTC/{quote}"
        norm = user_input.strip().upper()
        if "/" in norm:
            return norm
        if norm.endswith(quote):
            return f"{norm[:-len(quote)]}/{quote}"
        return f"{norm}/{quote}"
    
    def color_text(text: str, color=None, style=None) -> str:
        return text
    
    ProgressBar = None
    ExchangeManager = None


class DataFetcher:
    """Fetches market data (prices, OHLCV) from exchanges."""
    
    def __init__(self, exchange_manager: ExchangeManager, shutdown_event=None):
        self.exchange_manager = exchange_manager
        self.shutdown_event = shutdown_event
        self._ohlcv_cache: Dict[Tuple[str, str, int], pd.Series] = {}
        self.market_prices: Dict[str, float] = {}
    
    def should_stop(self) -> bool:
        """Check if shutdown was requested."""
        if self.shutdown_event:
            return self.shutdown_event.is_set()
        return False
    
    def fetch_prices(self, symbols: list):
        """Fetches current prices for all symbols from Binance."""
        if not symbols:
            return
        
        print(color_text("Fetching current prices from Binance...", Fore.CYAN))
        progress = ProgressBar(len(symbols), "Price Fetch")
        
        try:
            # Use authenticated manager for authenticated calls
            exchange = self.exchange_manager.authenticated.connect_to_binance_with_credentials()
        except ValueError as e:
            print(color_text(f"Error: {e}", Fore.RED))
            return
        
        fetched_count = 0
        failed_symbols = []
        
        for symbol in symbols:
            if self.should_stop():
                print(color_text("Price fetch aborted due to shutdown signal.", Fore.YELLOW))
                break
            normalized_symbol = normalize_symbol(symbol)
            try:
                # Use authenticated manager's throttled_call
                ticker = self.exchange_manager.authenticated.throttled_call(exchange.fetch_ticker, normalized_symbol)
                if ticker and 'last' in ticker:
                    self.market_prices[symbol] = ticker['last']
                    fetched_count += 1
                    print(color_text(f"  [BINANCE] {normalized_symbol}: {ticker['last']:.8f}", Fore.GREEN))
                else:
                    failed_symbols.append(symbol)
                    print(color_text(f"  {normalized_symbol}: No price data available", Fore.YELLOW))
            except Exception as e:
                failed_symbols.append(symbol)
                print(color_text(f"  Error fetching {normalized_symbol}: {e}", Fore.YELLOW))
            finally:
                progress.update()
        progress.finish()
        
        if failed_symbols:
            print(color_text(f"\nWarning: Could not fetch prices for {len(failed_symbols)} symbol(s): {', '.join(failed_symbols)}", Fore.YELLOW))
        
        if fetched_count > 0:
            print(color_text(f"\nSuccessfully fetched prices for {fetched_count}/{len(symbols)} symbols", Fore.GREEN))
    
    def fetch_ohlcv(self, symbol, limit=1500, timeframe='1h'):
        """Fetches OHLCV data using ccxt with fallback exchanges (with caching)."""
        normalized_symbol = normalize_symbol(symbol)
        cache_key = (normalized_symbol.upper(), timeframe, int(limit))
        if cache_key in self._ohlcv_cache:
            return self._ohlcv_cache[cache_key].copy()
        
        last_error = None
        for exchange_id in self.exchange_manager.public.exchange_priority_for_fallback:
            if self.should_stop():
                print(color_text("OHLCV fetch cancelled by shutdown.", Fore.YELLOW))
                return None
            exchange_id = exchange_id.strip()
            if not exchange_id:
                continue
            try:
                # Use public manager for public OHLCV data (no credentials needed)
                exchange = self.exchange_manager.public.connect_to_exchange_with_no_credentials(exchange_id)
            except Exception as exc:
                last_error = exc
                continue
            try:
                # Use public manager's throttled_call
                ohlcv = self.exchange_manager.public.throttled_call(
                    exchange.fetch_ohlcv,
                    normalized_symbol,
                    timeframe=timeframe,
                    limit=limit
                )
            except Exception as exc:
                last_error = exc
                continue
            if not ohlcv:
                last_error = ValueError(f"{exchange_id}: empty OHLCV")
                continue
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if df.empty:
                last_error = ValueError(f"{exchange_id}: OHLCV dataframe empty")
                continue
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            print(color_text(f"  [OHLCV] {normalized_symbol} loaded from {exchange_id} ({len(df)} bars)", Fore.GREEN))
            series = df['close']
            self._ohlcv_cache[cache_key] = series
            return series.copy()
        print(color_text(f"Failed to fetch OHLCV for {normalized_symbol}: {last_error}", Fore.RED))
        return None

