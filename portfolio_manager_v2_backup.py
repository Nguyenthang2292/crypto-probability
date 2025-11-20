import ccxt
import pandas as pd
import numpy as np
import os
import time
import threading
import signal
import sys
from math import ceil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from colorama import Fore, Style, init as colorama_init

try:
    from modules.config import (
        DEFAULT_EXCHANGE_STRING,
        DEFAULT_QUOTE,
        BENCHMARK_SYMBOL,
        DEFAULT_REQUEST_PAUSE,
        DEFAULT_BETA_MIN_POINTS,
        DEFAULT_BETA_LIMIT,
        DEFAULT_BETA_TIMEFRAME,
        DEFAULT_CORRELATION_MIN_POINTS,
        DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS,
        DEFAULT_VAR_CONFIDENCE,
        DEFAULT_VAR_LOOKBACK_DAYS,
        DEFAULT_VAR_MIN_HISTORY_DAYS,
        DEFAULT_VAR_MIN_PNL_SAMPLES,
    )
except ImportError:
    DEFAULT_EXCHANGE_STRING = "binance,kraken,kucoin,gate,okx,bybit,mexc,huobi"
    DEFAULT_QUOTE = "USDT"
    BENCHMARK_SYMBOL = "BTC/USDT"
    DEFAULT_REQUEST_PAUSE = 0.2
    DEFAULT_BETA_MIN_POINTS = 50
    DEFAULT_BETA_LIMIT = 1000
    DEFAULT_BETA_TIMEFRAME = "1h"
    DEFAULT_CORRELATION_MIN_POINTS = 10
    DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS = 10
    DEFAULT_VAR_CONFIDENCE = 0.95
    DEFAULT_VAR_LOOKBACK_DAYS = 90
    DEFAULT_VAR_MIN_HISTORY_DAYS = 20
    DEFAULT_VAR_MIN_PNL_SAMPLES = 10

try:
    from modules.binance_positions import get_binance_futures_positions
except ImportError:
    print("Warning: Could not import get_binance_futures_positions from modules.binance_positions")
    get_binance_futures_positions = None

try:
    from modules.config_api import BINANCE_API_KEY, BINANCE_API_SECRET
except ImportError:
    BINANCE_API_KEY = None
    BINANCE_API_SECRET = None

try:
    from modules.utils import normalize_symbol, color_text
    from modules.ProgressBar import ProgressBar
    from modules.Position import Position
except ImportError:
    def normalize_symbol(user_input: str, quote: str = "USDT") -> str:
        """Converts user input like 'eth' into 'ETH/USDT'. Keeps existing slash pairs."""
        if not user_input:
            return f"BTC/{quote}"
        
        norm = user_input.strip().upper()
        if "/" in norm:
            return norm
        
        if norm.endswith(quote):
            return f"{norm[:-len(quote)]}/{quote}"
        
        return f"{norm}/{quote}"
    
    def color_text(text: str, color: str = Fore.WHITE, style: str = Style.NORMAL) -> str:
        """Fallback color_text function."""
        return f"{style}{color}{text}{Style.RESET_ALL}"
    
    class ProgressBar:
        def __init__(self, total: int, label: str = "Progress", width: int = 30):
            self.total = max(total, 1)
            self.label = label
            self.width = width
            self.current = 0
            self._lock = threading.Lock()

        def update(self, step: int = 1):
            with self._lock:
                self.current = min(self.total, self.current + step)
                ratio = self.current / self.total
                filled = int(self.width * ratio)
                bar = "█" * filled + "-" * (self.width - filled)
                percent = ratio * 100
                print(f"\r{color_text(f'{self.label}: [{bar}] {self.current}/{self.total} ({percent:5.1f}%)', Fore.CYAN)}",
                      end='',
                      flush=True)

        def finish(self):
            self.update(0)
            print()
    
    @dataclass
    class Position:
        symbol: str
        direction: str
        entry_price: float
        size_usdt: float

colorama_init(autoreset=True)

class PortfolioManager:
    def __init__(self, api_key=None, api_secret=None, testnet=False):
        self.positions: List[Position] = []
        self.market_prices = {}
        self.api_key = api_key or os.getenv('BINANCE_API_KEY') or BINANCE_API_KEY
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET') or BINANCE_API_SECRET
        self.testnet = testnet
        self._binance_exchange = None
        self.benchmark_symbol = BENCHMARK_SYMBOL
        self._beta_cache: Dict[str, float] = {}
        self.last_var_value: Optional[float] = None
        self.last_var_confidence: Optional[float] = None
        self.request_pause = float(os.getenv("BINANCE_REQUEST_SLEEP", DEFAULT_REQUEST_PAUSE))
        self._request_lock = threading.Lock()
        self._last_request_ts = 0.0
        self._ohlcv_clients: Dict[str, ccxt.Exchange] = {}
        self._ohlcv_cache: Dict[Tuple[str, str, int], pd.Series] = {}
        fallback_string = os.getenv("OHLCV_FALLBACKS", DEFAULT_EXCHANGE_STRING)
        self.ohlcv_exchange_priority = fallback_string.split(",")
        self.shutdown_event = threading.Event()
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def add_position(self, symbol: str, direction: str, entry_price: float, size_usdt: float):
        self.positions.append(Position(symbol.upper(), direction.upper(), entry_price, size_usdt))

    def _handle_shutdown(self, signum, frame):
        if not self.shutdown_event.is_set():
            print(color_text("\nInterrupt received. Cancelling ongoing tasks...", Fore.YELLOW))
            self.shutdown_event.set()
        sys.exit(0)

    def _should_stop(self) -> bool:
        return self.shutdown_event.is_set()

    def _get_binance_exchange(self):
        """Get Binance exchange instance, creating if needed."""
        if self._binance_exchange is None:
            if not self.api_key or not self.api_secret:
                raise ValueError(
                    "API Key và API Secret là bắt buộc!\n"
                    "Cung cấp qua một trong các cách sau:\n"
                    "  1. Tham số khi khởi tạo PortfolioManager\n"
                    "  2. Biến môi trường: BINANCE_API_KEY và BINANCE_API_SECRET\n"
                    "  3. File config: modules/config_api.py (BINANCE_API_KEY và BINANCE_API_SECRET)"
                )
            
            options = {
                'defaultType': 'future',
                'options': {
                    'defaultType': 'future',
                }
            }
            
            if self.testnet:
                self._binance_exchange = ccxt.binance({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'options': options,
                    'enableRateLimit': True,
                    'sandbox': True,
                })
            else:
                self._binance_exchange = ccxt.binance({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'options': options,
                    'enableRateLimit': True,
                })
        
        return self._binance_exchange

    def _throttled_call(self, func, *args, **kwargs):
        """Ensures a minimum delay between REST calls to respect rate limits."""
        with self._request_lock:
            wait = self.request_pause - (time.time() - self._last_request_ts)
            if wait > 0:
                time.sleep(wait)
            result = func(*args, **kwargs)
            self._last_request_ts = time.time()
            return result

    def _get_ohlcv_exchange(self, exchange_id: str) -> ccxt.Exchange:
        exchange_id = exchange_id.strip().lower()
        if exchange_id == "binance":
            return self._get_binance_exchange()
        if exchange_id not in self._ohlcv_clients:
            if not hasattr(ccxt, exchange_id):
                raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt.")
            exchange_class = getattr(ccxt, exchange_id)
            params = {
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                }
            }
            try:
                self._ohlcv_clients[exchange_id] = exchange_class(params)
            except Exception as exc:
                raise ValueError(f"Cannot initialize exchange {exchange_id}: {exc}")
        return self._ohlcv_clients[exchange_id]

    def load_from_binance(self, api_key=None, api_secret=None, testnet=None, debug=False):
        """Load positions directly from Binance Futures USDT-M."""
        if get_binance_futures_positions is None:
            raise ImportError("Cannot import get_binance_futures_positions from modules.binance_positions")
        
        if api_key is not None:
            self.api_key = api_key
        if api_secret is not None:
            self.api_secret = api_secret
        if testnet is not None:
            self.testnet = testnet
        
        print(color_text("Loading positions from Binance Futures USDT-M...", Fore.CYAN, Style.BRIGHT))
        
        try:
            binance_positions = get_binance_futures_positions(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
                debug=debug
            )
            
            if not binance_positions:
                print(color_text("No open positions found on Binance.", Fore.YELLOW))
                return
            
            self.positions.clear()
            
            for pos in binance_positions:
                self.add_position(
                    symbol=pos['symbol'],
                    direction=pos['direction'],
                    entry_price=pos['entry_price'],
                    size_usdt=pos['size_usdt']
                )
            
            print(color_text(f"✓ Loaded {len(binance_positions)} position(s) from Binance", Fore.GREEN))
            
            print("\n" + color_text("Loaded Positions:", Fore.CYAN))
            for pos in binance_positions:
                print(f"  {pos['symbol']:<15} {pos['direction']:<5} Entry: {pos['entry_price']:>12.8f} Size: {pos['size_usdt']:>12.2f} USDT")
            
        except Exception as e:
            raise ValueError(f"Error loading positions from Binance: {e}")

    def fetch_prices(self):
        """Fetches current prices for all symbols from Binance."""
        symbols = list(set([p.symbol for p in self.positions]))
        if not symbols:
            return
        
        print(color_text("Fetching current prices from Binance...", Fore.CYAN))
        progress = ProgressBar(len(symbols), "Price Fetch")
        
        try:
            exchange = self._get_binance_exchange()
        except ValueError as e:
            print(color_text(f"Error: {e}", Fore.RED))
            return
        
        fetched_count = 0
        failed_symbols = []
        
        for symbol in symbols:
            if self._should_stop():
                print(color_text("Price fetch aborted due to shutdown signal.", Fore.YELLOW))
                break
            normalized_symbol = normalize_symbol(symbol)
            try:
                ticker = self._throttled_call(exchange.fetch_ticker, normalized_symbol)
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

    def calculate_stats(self):
        """Calculates PnL, simple delta, and beta-weighted delta for the portfolio."""
        total_pnl = 0
        total_delta = 0
        total_beta_delta = 0
        
        results = []
        
        for p in self.positions:
            current_price = self.market_prices.get(p.symbol)
            if current_price is None:
                continue
                
            if p.direction == 'LONG':
                pnl_pct = (current_price - p.entry_price) / p.entry_price
                delta = p.size_usdt
            else:
                pnl_pct = (p.entry_price - current_price) / p.entry_price
                delta = -p.size_usdt
                
            pnl_usdt = pnl_pct * p.size_usdt
            
            total_pnl += pnl_usdt
            total_delta += delta
            
            beta = self.calculate_beta(p.symbol)
            beta_delta = None
            if beta is not None:
                beta_delta = delta * beta
                total_beta_delta += beta_delta
            
            results.append({
                'Symbol': p.symbol,
                'Direction': p.direction,
                'Entry': p.entry_price,
                'Current': current_price,
                'Size': p.size_usdt,
                'PnL': pnl_usdt,
                'Delta': delta,
                'Beta': beta,
                'Beta Delta': beta_delta
            })
            
        return pd.DataFrame(results), total_pnl, total_delta, total_beta_delta

    def fetch_ohlcv(self, symbol, limit=1500, timeframe='1h'):
        """Fetches OHLCV data using ccxt with fallback exchanges (with caching)."""
        normalized_symbol = normalize_symbol(symbol)
        cache_key = (normalized_symbol.upper(), timeframe, int(limit))
        if cache_key in self._ohlcv_cache:
            return self._ohlcv_cache[cache_key].copy()
        last_error = None
        for exchange_id in self.ohlcv_exchange_priority:
            if self._should_stop():
                print(color_text("OHLCV fetch cancelled by shutdown.", Fore.YELLOW))
                return None
            exchange_id = exchange_id.strip()
            if not exchange_id:
                continue
            try:
                exchange = self._get_ohlcv_exchange(exchange_id)
            except Exception as exc:
                last_error = exc
                continue
            try:
                ohlcv = self._throttled_call(
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

    def calculate_beta(self, symbol: str, benchmark_symbol: Optional[str] = None, min_points: int = DEFAULT_BETA_MIN_POINTS,
                       limit: int = DEFAULT_BETA_LIMIT, timeframe: str = DEFAULT_BETA_TIMEFRAME) -> Optional[float]:
        """Calculates beta of a symbol versus a benchmark (default BTC/USDT)."""
        benchmark_symbol = benchmark_symbol or self.benchmark_symbol
        normalized_symbol = normalize_symbol(symbol)
        normalized_benchmark = normalize_symbol(benchmark_symbol)
        cache_key = f"{normalized_symbol}|{normalized_benchmark}|{timeframe}|{limit}"
        
        if normalized_symbol == normalized_benchmark:
            return 1.0
        
        if cache_key in self._beta_cache:
            return self._beta_cache[cache_key]
        
        asset_series = self.fetch_ohlcv(normalized_symbol, limit=limit, timeframe=timeframe)
        benchmark_series = self.fetch_ohlcv(normalized_benchmark, limit=limit, timeframe=timeframe)
        
        if asset_series is None or benchmark_series is None:
            return None
        
        df = pd.concat([asset_series, benchmark_series], axis=1, join='inner').dropna()
        if len(df) < min_points:
            return None
        
        returns = df.pct_change().dropna()
        if returns.empty:
            return None
        
        benchmark_var = returns.iloc[:, 1].var()
        if benchmark_var is None or benchmark_var <= 0:
            return None
        
        covariance = returns.iloc[:, 0].cov(returns.iloc[:, 1])
        if covariance is None:
            return None
        
        beta = covariance / benchmark_var
        if pd.isna(beta):
            return None
        
        self._beta_cache[cache_key] = beta
        return beta

    def calculate_portfolio_var(self, confidence: float = DEFAULT_VAR_CONFIDENCE, lookback_days: int = DEFAULT_VAR_LOOKBACK_DAYS) -> Optional[float]:
        """Calculates Historical Simulation VaR for the current portfolio."""
        self.last_var_value = None
        self.last_var_confidence = None
        if not self.positions:
            print(color_text("No positions available for VaR calculation.", Fore.YELLOW))
            return None
        
        confidence_pct = int(confidence * 100)
        print(color_text(f"\nCalculating Historical VaR ({confidence_pct}% confidence, {lookback_days}d lookback)...", Fore.CYAN))
        
        price_history = {}
        fetch_limit = max(lookback_days * 2, lookback_days + 50)
        for pos in self.positions:
            series = self.fetch_ohlcv(pos.symbol, limit=fetch_limit, timeframe='1d')
            if series is not None:
                price_history[pos.symbol] = series
        
        if not price_history:
            print(color_text("Unable to fetch historical data for VaR.", Fore.YELLOW))
            return None
        
        price_df = pd.DataFrame(price_history).dropna(how="all")
        if price_df.empty:
            print(color_text("No overlapping history found for VaR.", Fore.YELLOW))
            return None
        
        if len(price_df) < lookback_days:
            print(color_text(f"Only {len(price_df)} daily points available (requested {lookback_days}). Using available history.", Fore.YELLOW))
        price_df = price_df.tail(lookback_days)
        
        if len(price_df) < DEFAULT_VAR_MIN_HISTORY_DAYS:
            print(color_text(f"Insufficient history (<{DEFAULT_VAR_MIN_HISTORY_DAYS} days) for reliable VaR.", Fore.YELLOW))
            return None
        
        returns_df = price_df.pct_change().dropna(how="all")
        if returns_df.empty:
            print(color_text("Unable to compute returns for VaR.", Fore.YELLOW))
            return None
        
        daily_pnls = []
        for idx in returns_df.index:
            daily_pnl = 0.0
            has_data = False
            for pos in self.positions:
                if pos.symbol not in returns_df.columns:
                    continue
                ret = returns_df.at[idx, pos.symbol]
                if pd.isna(ret):
                    continue
                exposure = pos.size_usdt if pos.direction == 'LONG' else -pos.size_usdt
                daily_pnl += exposure * ret
                has_data = True
            if has_data:
                daily_pnls.append(daily_pnl)
        
        if len(daily_pnls) < DEFAULT_VAR_MIN_PNL_SAMPLES:
            print(color_text(f"Not enough historical PnL samples for VaR (need at least {DEFAULT_VAR_MIN_PNL_SAMPLES}).", Fore.YELLOW))
            return None
        
        percentile = max(0, min(100, (1 - confidence) * 100))
        loss_percentile = np.percentile(daily_pnls, percentile)
        var_amount = max(0.0, -loss_percentile)
        
        print(color_text(f"Historical VaR ({confidence_pct}%): {var_amount:.2f} USDT", Fore.MAGENTA, Style.BRIGHT))
        self.last_var_value = var_amount
        self.last_var_confidence = confidence
        return var_amount

    def _normalize_market_symbol(self, market_symbol: str) -> str:
        """Converts Binance futures symbols like BTC/USDT:USDT into BTC/USDT."""
        if ":" in market_symbol:
            market_symbol = market_symbol.split(":")[0]
        return normalize_symbol(market_symbol)

    def list_candidate_symbols(self, exclude_symbols: Optional[set] = None,
                               max_candidates: Optional[int] = None) -> List[str]:
        """Fetches potential hedge symbols from Binance Futures."""
        exclude_symbols = exclude_symbols or set()
        candidates: List[Tuple[str, float]] = []
        try:
            exchange = self._get_binance_exchange()
        except ValueError as exc:
            print(color_text(f"Unable to list hedge candidates: {exc}", Fore.RED))
            return []
        
        markets = exchange.load_markets()
        progress = ProgressBar(len(markets), "Symbol Discovery")
        seen = set()
        for market in markets.values():
            if self._should_stop():
                print(color_text("Symbol discovery aborted due to shutdown.", Fore.YELLOW))
                break
            if not market.get('contract'):
                progress.update()
                continue
            if market.get('quote') != 'USDT':
                progress.update()
                continue
            if not market.get('active', True):
                progress.update()
                continue
            symbol = self._normalize_market_symbol(market['symbol'])
            if symbol in exclude_symbols or symbol in seen:
                progress.update()
                continue
            volume = 0.0
            info = market.get('info', {})
            volume_str = info.get('volume') or info.get('quoteVolume') or info.get('turnover')
            try:
                volume = float(volume_str)
            except (TypeError, ValueError):
                volume = 0.0
            candidates.append((symbol, volume))
            seen.add(symbol)
            progress.update()
        progress.finish()
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        if max_candidates is not None:
            candidates = candidates[:max_candidates]
        symbol_list = [symbol for symbol, _ in candidates]
        print(color_text(f"Discovered {len(symbol_list)} futures symbols from Binance.", Fore.CYAN))
        return symbol_list

    def _score_candidate(self, symbol: str) -> Optional[Dict[str, float]]:
        weighted_corr, _ = self.calculate_weighted_correlation(symbol, verbose=False)
        return_corr, _ = self.calculate_portfolio_return_correlation(symbol, verbose=False)
        
        if weighted_corr is None and return_corr is None:
            return None
        
        score_components = [abs(x) for x in [weighted_corr, return_corr] if x is not None]
        if not score_components:
            return None
        
        score = sum(score_components) / len(score_components)
        return {
            'symbol': symbol,
            'weighted_corr': weighted_corr,
            'return_corr': return_corr,
            'score': score
        }

    def find_best_hedge_candidate(self, total_delta: float, total_beta_delta: float,
                                  max_candidates: Optional[int] = None) -> Optional[Dict]:
        """Automatically scans Binance futures symbols to find the best hedge candidate."""
        if not self.positions:
            print(color_text("No positions loaded. Cannot search for hedge candidates.", Fore.YELLOW))
            return None
        
        existing_symbols = {normalize_symbol(p.symbol) for p in self.positions}
        existing_symbols.add(normalize_symbol(self.benchmark_symbol))
        candidate_symbols = self.list_candidate_symbols(existing_symbols, max_candidates=None)
        
        if not candidate_symbols:
            print(color_text("Could not find candidate symbols from Binance.", Fore.YELLOW))
            return None
        if self._should_stop():
            print(color_text("Hedge scan aborted before start.", Fore.YELLOW))
            return None
        
        if max_candidates is not None:
            candidate_symbols = candidate_symbols[:max_candidates]
        scan_count = len(candidate_symbols)
        print(color_text(f"\nScanning {scan_count} candidate(s) for optimal hedge...", Fore.CYAN))
        
        core_count = max(1, int((os.cpu_count() or 1) * 0.8))
        batch_size = ceil(scan_count / core_count) if scan_count else 0
        batches = [candidate_symbols[i:i + batch_size] for i in range(0, scan_count, batch_size)] or [[]]
        total_batches = len([b for b in batches if b])
        progress_bar = ProgressBar(total_batches or 1, "Batch Progress")
        
        best_candidate = None
        
        def process_batch(batch_symbols: List[str]) -> Optional[Dict]:
            local_best = None
            for sym in batch_symbols:
                if self._should_stop():
                    return None
                result = self._score_candidate(sym)
                if result is None:
                    continue
                if local_best is None or result['score'] > local_best['score']:
                    local_best = result
            return local_best
        
        with ThreadPoolExecutor(max_workers=core_count) as executor:
            futures = {}
            for idx, batch in enumerate(batches, start=1):
                if not batch:
                    continue
                print(color_text(f"Starting batch {idx}/{total_batches} (size {len(batch)})", Fore.BLUE))
                futures[executor.submit(process_batch, batch)] = idx
            for future in as_completed(futures):
                if self._should_stop():
                    break
                batch_id = futures[future]
                try:
                    batch_best = future.result()
                except Exception as exc:
                    print(color_text(f"Batch {batch_id} failed: {exc}", Fore.RED))
                    continue
                if batch_best is None:
                    print(color_text(f"Batch {batch_id}: no viable candidate.", Fore.YELLOW))
                    progress_bar.update()
                    continue
                print(color_text(f"Batch {batch_id}: best {batch_best['symbol']} (score {batch_best['score']:.4f})", Fore.WHITE))
                if best_candidate is None or batch_best['score'] > best_candidate['score']:
                    best_candidate = batch_best
                progress_bar.update()
        if total_batches:
            progress_bar.finish()
        
        if best_candidate is None:
            print(color_text("No suitable hedge candidate found (insufficient data).", Fore.YELLOW))
        else:
            print(color_text(f"\nBest candidate: {best_candidate['symbol']} (score {best_candidate['score']:.4f})", Fore.MAGENTA, Style.BRIGHT))
            if best_candidate['weighted_corr'] is not None:
                print(color_text(f"  Weighted Correlation: {best_candidate['weighted_corr']:.4f}", Fore.WHITE))
            if best_candidate['return_corr'] is not None:
                print(color_text(f"  Portfolio Return Correlation: {best_candidate['return_corr']:.4f}", Fore.WHITE))
        
        return best_candidate

    def calculate_weighted_correlation(self, new_symbol: str, verbose: bool = True):
        """Calculates weighted correlation with entire portfolio based on position sizes."""
        correlations = []
        weights = []
        position_details = []
        
        if verbose:
            print(color_text(f"\nCorrelation Analysis (Weighted by Position Size):", Fore.CYAN))
        
        new_series = self.fetch_ohlcv(new_symbol)
        if new_series is None:
            if verbose:
                print(color_text(f"Could not fetch price history for {new_symbol}", Fore.RED))
            return None, []
        
        for pos in self.positions:
            pos_series = self.fetch_ohlcv(pos.symbol)
            
            if pos_series is not None:
                df = pd.concat([pos_series, new_series], axis=1, join='inner')
                if len(df) < DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS:
                    continue
                
                corr = df.iloc[:, 0].corr(df.iloc[:, 1])
                weight = pos.size_usdt
                
                correlations.append(corr)
                weights.append(weight)
                
                position_details.append({
                    'symbol': pos.symbol,
                    'direction': pos.direction,
                    'size': pos.size_usdt,
                    'correlation': corr,
                    'weight': weight
                })
        
        if not correlations:
            if verbose:
                print(color_text("Insufficient data for correlation analysis.", Fore.YELLOW))
            return None, []
        
        total_weight = sum(weights)
        weighted_corr = sum(c * w for c, w in zip(correlations, weights)) / total_weight
        
        if verbose:
            print(f"\nIndividual Correlations:")
            for detail in position_details:
                corr_color = Fore.GREEN if abs(detail['correlation']) > 0.7 else (Fore.YELLOW if abs(detail['correlation']) > 0.4 else Fore.RED)
                weight_pct = (detail['weight'] / total_weight) * 100
                print(f"  {detail['symbol']:12} ({detail['direction']:5}, {detail['size']:>8.2f} USDT, {weight_pct:>5.1f}%): "
                      + color_text(f"{detail['correlation']:>6.4f}", corr_color))
            
            print(f"\n{color_text('Weighted Portfolio Correlation:', Fore.CYAN, Style.BRIGHT)}")
            weighted_corr_color = Fore.GREEN if abs(weighted_corr) > 0.7 else (Fore.YELLOW if abs(weighted_corr) > 0.4 else Fore.RED)
            print(f"  {new_symbol} vs Portfolio: {color_text(f'{weighted_corr:>6.4f}', weighted_corr_color, Style.BRIGHT)}")
        
        return weighted_corr, position_details

    def calculate_portfolio_return_correlation(self, new_symbol: str, min_points: int = DEFAULT_CORRELATION_MIN_POINTS, verbose: bool = True):
        """Calculates correlation between the portfolio's aggregated return and the new symbol."""
        if verbose:
            print(color_text(f"\nPortfolio Return Correlation Analysis:", Fore.CYAN))

        if not self.positions:
            if verbose:
                print(color_text("No positions in portfolio to compare against.", Fore.YELLOW))
            return None, {}

        new_series = self.fetch_ohlcv(new_symbol)
        if new_series is None:
            if verbose:
                print(color_text(f"Could not fetch price history for {new_symbol}", Fore.RED))
            return None, {}

        symbol_series = {}
        for pos in self.positions:
            if pos.symbol not in symbol_series:
                series = self.fetch_ohlcv(pos.symbol)
                if series is not None:
                    symbol_series[pos.symbol] = series

        if not symbol_series:
            if verbose:
                print(color_text("Unable to fetch history for existing positions.", Fore.YELLOW))
            return None, {}

        price_df = pd.DataFrame(symbol_series).dropna(how="all")
        if price_df.empty:
            if verbose:
                print(color_text("Insufficient overlapping data among current positions.", Fore.YELLOW))
            return None, {}

        portfolio_returns_df = price_df.pct_change().dropna(how="all")
        new_returns = new_series.pct_change().dropna()

        if portfolio_returns_df.empty or new_returns.empty:
            if verbose:
                print(color_text("Insufficient price history to compute returns.", Fore.YELLOW))
            return None, {}

        common_index = portfolio_returns_df.index.intersection(new_returns.index)
        if len(common_index) < min_points:
            if verbose:
                print(color_text(f"Need at least {min_points} overlapping points, found {len(common_index)}.", Fore.YELLOW))
            return None, {}

        portfolio_returns = []
        aligned_new_returns = []

        for idx in common_index:
            total_weight = 0.0
            weighted_return = 0.0
            for pos in self.positions:
                if pos.symbol not in portfolio_returns_df.columns:
                    continue
                ret = portfolio_returns_df.at[idx, pos.symbol]
                if pd.isna(ret):
                    continue
                if pos.direction == "SHORT":
                    ret = -ret
                weight = abs(pos.size_usdt)
                if weight <= 0:
                    continue
                weighted_return += ret * weight
                total_weight += weight

            new_ret = new_returns.loc[idx]
            if total_weight > 0 and not pd.isna(new_ret):
                portfolio_returns.append(weighted_return / total_weight)
                aligned_new_returns.append(new_ret)

        if len(portfolio_returns) < min_points:
            if verbose:
                print(color_text("Not enough aligned return samples for correlation.", Fore.YELLOW))
            return None, {"samples": len(portfolio_returns)}

        portfolio_return_series = pd.Series(portfolio_returns)
        new_return_series = pd.Series(aligned_new_returns)
        correlation = portfolio_return_series.corr(new_return_series)

        if pd.isna(correlation):
            if verbose:
                print(color_text("Unable to compute correlation (insufficient variance).", Fore.YELLOW))
            return None, {"samples": len(portfolio_returns)}

        if verbose:
            corr_color = Fore.GREEN if abs(correlation) > 0.7 else (Fore.YELLOW if abs(correlation) > 0.4 else Fore.RED)
            print(f"  Portfolio Return vs {new_symbol}: {color_text(f'{correlation:>6.4f}', corr_color, Style.BRIGHT)}")
            print(color_text(f"  Samples used: {len(portfolio_returns)}", Fore.WHITE))

        return correlation, {"samples": len(portfolio_returns)}

    def analyze_new_trade(self, new_symbol: str, total_delta: float, total_beta_delta: float,
                          correlation_mode: str = "weighted"):
        """Analyzes a potential new trade and automatically recommends direction for beta-weighted hedging."""
        normalized_symbol = normalize_symbol(new_symbol)
        if normalized_symbol != new_symbol:
            print(color_text(f"Symbol normalized: '{new_symbol}' -> '{normalized_symbol}'", Fore.CYAN))
        
        new_symbol = normalized_symbol
        print(color_text(f"\nAnalyzing potential trade on {new_symbol}...", Fore.CYAN, Style.BRIGHT))
        print(color_text(f"Current Total Delta: {total_delta:+.2f} USDT", Fore.WHITE))
        print(color_text(f"Current Total Beta Delta (vs {self.benchmark_symbol}): {total_beta_delta:+.2f} USDT", Fore.WHITE))
        
        new_symbol_beta = self.calculate_beta(new_symbol)
        beta_available = new_symbol_beta is not None and abs(new_symbol_beta) > 1e-6
        if beta_available:
            print(color_text(f"{new_symbol} beta vs {self.benchmark_symbol}: {new_symbol_beta:.4f}", Fore.CYAN))
        else:
            print(color_text(f"Could not compute beta for {new_symbol}. Falling back to simple delta hedging.", Fore.YELLOW))
        
        hedge_mode = "beta" if beta_available else "delta"
        metric_label = "Beta Delta" if beta_available else "Delta"
        current_metric = total_beta_delta if beta_available else total_delta
        target_metric = -current_metric
        
        recommended_direction = None
        recommended_size = None
        
        if abs(current_metric) < 0.01:
            print(color_text(f"Portfolio is already {metric_label} Neutral ({metric_label} ≈ 0).", Fore.GREEN))
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
                    recommended_direction = 'LONG' if direction_multiplier >= 0 else 'SHORT'
                    recommended_size = abs(current_metric) / max(abs(new_symbol_beta), 1e-6)
                    print(color_text(f"Targeting Beta Neutrality using {metric_label}.", Fore.CYAN))
            if not beta_available:
                if current_metric > 0:
                    recommended_direction = 'SHORT'
                    recommended_size = abs(target_metric)
                    print(color_text("Portfolio has excess LONG delta exposure.", Fore.YELLOW))
                else:
                    recommended_direction = 'LONG'
                    recommended_size = abs(target_metric)
                    print(color_text("Portfolio has excess SHORT delta exposure.", Fore.YELLOW))
                print(color_text("Targeting simple Delta Neutrality.", Fore.CYAN))
        
        if recommended_direction and recommended_size is not None:
            print(color_text(f"\nRecommended {hedge_mode.upper()} hedge:", Fore.CYAN, Style.BRIGHT))
            print(color_text(f"  Direction: {recommended_direction}", Fore.WHITE))
            print(color_text(f"  Size: {recommended_size:.2f} USDT", Fore.GREEN, Style.BRIGHT))

        if not self.positions:
            print(color_text("\nNo existing positions for correlation analysis.", Fore.WHITE))
            return recommended_direction, recommended_size if recommended_direction else None, None

        print(color_text("\n" + "="*70, Fore.CYAN))
        print(color_text("CORRELATION ANALYSIS - COMPARING BOTH METHODS", Fore.CYAN, Style.BRIGHT))
        print(color_text("="*70, Fore.CYAN))
        
        weighted_corr, weighted_details = self.calculate_weighted_correlation(new_symbol)
        portfolio_return_corr, portfolio_return_details = self.calculate_portfolio_return_correlation(new_symbol)
        
        print(color_text("\n" + "="*70, Fore.CYAN))
        print(color_text("CORRELATION SUMMARY", Fore.MAGENTA, Style.BRIGHT))
        print(color_text("="*70, Fore.CYAN))
        
        if weighted_corr is not None:
            weighted_color = Fore.GREEN if abs(weighted_corr) > 0.7 else (Fore.YELLOW if abs(weighted_corr) > 0.4 else Fore.RED)
            print(f"1. Weighted Correlation (by Position Size):")
            print(f"   {new_symbol} vs Portfolio: {color_text(f'{weighted_corr:>6.4f}', weighted_color, Style.BRIGHT)}")
            
            if abs(weighted_corr) > 0.7:
                print(color_text("   → High correlation. Good for hedging.", Fore.GREEN))
            elif abs(weighted_corr) > 0.4:
                print(color_text("   → Moderate correlation. Partial hedging effect.", Fore.YELLOW))
            else:
                print(color_text("   → Low correlation. Limited hedging effectiveness.", Fore.RED))
        else:
            print(f"1. Weighted Correlation: {color_text('N/A (insufficient data)', Fore.YELLOW)}")
        
        if portfolio_return_corr is not None:
            portfolio_color = Fore.GREEN if abs(portfolio_return_corr) > 0.7 else (Fore.YELLOW if abs(portfolio_return_corr) > 0.4 else Fore.RED)
            samples_info = portfolio_return_details.get('samples', 'N/A') if isinstance(portfolio_return_details, dict) else 'N/A'
            print(f"\n2. Portfolio Return Correlation (includes direction):")
            print(f"   {new_symbol} vs Portfolio Return: {color_text(f'{portfolio_return_corr:>6.4f}', portfolio_color, Style.BRIGHT)}")
            print(f"   Samples used: {samples_info}")
            
            if abs(portfolio_return_corr) > 0.7:
                print(color_text("   → High correlation. Excellent for hedging.", Fore.GREEN))
            elif abs(portfolio_return_corr) > 0.4:
                print(color_text("   → Moderate correlation. Acceptable hedging effect.", Fore.YELLOW))
            else:
                print(color_text("   → Low correlation. Poor hedging effectiveness.", Fore.RED))
        else:
            print(f"\n2. Portfolio Return Correlation: {color_text('N/A (insufficient data)', Fore.YELLOW)}")
        
        print(color_text("\n" + "-"*70, Fore.WHITE))
        print(color_text("OVERALL ASSESSMENT:", Fore.CYAN, Style.BRIGHT))
        
        if weighted_corr is not None and portfolio_return_corr is not None:
            diff = abs(weighted_corr - portfolio_return_corr)
            if diff < 0.1:
                print(color_text("   ✓ Both methods show similar correlation → Consistent result", Fore.GREEN))
            else:
                print(color_text(f"   ⚠ Methods differ by {diff:.4f} → Check if portfolio has SHORT positions", Fore.YELLOW))
            
            avg_corr = (abs(weighted_corr) + abs(portfolio_return_corr)) / 2
            if avg_corr > 0.7:
                print(color_text("   [OK] High correlation detected. This pair is suitable for statistical hedging.", Fore.GREEN, Style.BRIGHT))
            elif avg_corr > 0.4:
                print(color_text("   [!] Moderate correlation. Hedge may be partially effective.", Fore.YELLOW))
            else:
                print(color_text("   [X] Low correlation. This hedge might be less effective systematically.", Fore.RED))
        
        elif weighted_corr is not None:
            if abs(weighted_corr) > 0.7:
                print(color_text("   [OK] High correlation detected. This pair is suitable for statistical hedging.", Fore.GREEN, Style.BRIGHT))
            elif abs(weighted_corr) > 0.4:
                print(color_text("   [!] Moderate correlation. Hedge may be partially effective.", Fore.YELLOW))
            else:
                print(color_text("   [X] Low correlation. This hedge might be less effective systematically.", Fore.RED))
        
        elif portfolio_return_corr is not None:
            if abs(portfolio_return_corr) > 0.7:
                print(color_text("   [OK] High correlation detected. This pair is suitable for statistical hedging.", Fore.GREEN, Style.BRIGHT))
            elif abs(portfolio_return_corr) > 0.4:
                print(color_text("   [!] Moderate correlation. Hedge may be partially effective.", Fore.YELLOW))
            else:
                print(color_text("   [X] Low correlation. This hedge might be less effective systematically.", Fore.RED))

        if self.last_var_value is not None and self.last_var_confidence is not None:
            conf_pct = int(self.last_var_confidence * 100)
            print(color_text("\nVaR INSIGHT:", Fore.MAGENTA, Style.BRIGHT))
            print(color_text(f"  With {conf_pct}% confidence, daily loss is unlikely to exceed {self.last_var_value:.2f} USDT.", Fore.WHITE))
            print(color_text("  Use this ceiling to judge whether the proposed hedge keeps risk tolerable.", Fore.WHITE))
        else:
            print(color_text("\nVaR INSIGHT: N/A (insufficient historical data for VaR).", Fore.YELLOW))
        
        print(color_text("="*70 + "\n", Fore.CYAN))
        
        final_corr = weighted_corr if weighted_corr is not None else portfolio_return_corr
        
        return recommended_direction, recommended_size if recommended_direction else None, final_corr

def main():
    print(color_text("=== Crypto Portfolio Manager (Binance Integration) ===", Fore.MAGENTA, Style.BRIGHT))
    
    try:
        pm = PortfolioManager()
    except Exception as e:
        print(color_text(f"Error initializing PortfolioManager: {e}", Fore.RED))
        return
    
    print("\n" + color_text("Loading positions from Binance...", Fore.CYAN))
    try:
        pm.load_from_binance()
    except Exception as e:
        print(color_text(f"Error loading from Binance: {e}", Fore.RED))
        print(color_text("Please check your API credentials and try again.", Fore.YELLOW))
        return
            
    if not pm.positions:
        print(color_text("No positions available. Exiting.", Fore.YELLOW))
        return

    pm.fetch_prices()
    df, total_pnl, total_delta, total_beta_delta = pm.calculate_stats()
    
    print("\n" + color_text("=== PORTFOLIO STATUS ===", Fore.WHITE, Style.BRIGHT))
    print(df.to_string(index=False))
    print("-" * 50)
    print(f"Total PnL: {color_text(f'{total_pnl:.2f} USDT', Fore.GREEN if total_pnl >= 0 else Fore.RED)}")
    print(f"Total Delta: {color_text(f'{total_delta:.2f} USDT', Fore.YELLOW)}")
    print(f"Total Beta Delta (vs {pm.benchmark_symbol}): {color_text(f'{total_beta_delta:.2f} USDT', Fore.YELLOW)}")
    
    var_value = pm.calculate_portfolio_var(confidence=DEFAULT_VAR_CONFIDENCE, lookback_days=DEFAULT_VAR_LOOKBACK_DAYS)
    if var_value is not None:
        conf_pct = int((pm.last_var_confidence or 0) * 100)
        print(color_text(f"Interpretation: With {conf_pct}% confidence, daily loss should stay within {var_value:.2f} USDT.", Fore.WHITE))
    else:
        print(color_text("VaR Interpretation: Not enough history for a reliable estimate.", Fore.YELLOW))
    
    best_candidate = pm.find_best_hedge_candidate(total_delta, total_beta_delta)
    if best_candidate:
        symbol = best_candidate['symbol']
        print("\n" + color_text("=== AUTO HEDGE ANALYSIS ===", Fore.MAGENTA, Style.BRIGHT))
        recommended_direction, recommended_size, correlation = pm.analyze_new_trade(symbol, total_delta, total_beta_delta)
        if recommended_direction and recommended_size is not None:
            print(color_text(f"\n✓ Auto-selected hedge: {symbol} | {recommended_direction} {recommended_size:.2f} USDT", Fore.GREEN, Style.BRIGHT))
        else:
            print(color_text(f"\n{symbol}: Portfolio already neutral, no trade required.", Fore.WHITE))
    else:
        print(color_text("\nCould not determine a suitable hedge candidate automatically.", Fore.YELLOW))

if __name__ == "__main__":
    main()
