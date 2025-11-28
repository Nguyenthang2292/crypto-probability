"""
Data fetcher for retrieving market data from exchanges.
"""

import pandas as pd
from typing import Dict, Tuple, Optional, List

from modules.common.utils import (
    normalize_symbol,
    timeframe_to_minutes,
    dataframe_to_close_series,
    log_exchange,
    log_error,
    log_warn,
    log_success,
    log_debug,
    log_info,
    log_data,
)
from modules.common.ProgressBar import ProgressBar
from modules.common.ExchangeManager import ExchangeManager


class DataFetcher:
    """Fetches market data (prices, OHLCV) from exchanges."""

    def __init__(self, exchange_manager: ExchangeManager, shutdown_event=None):
        self.exchange_manager = exchange_manager
        self.shutdown_event = shutdown_event
        self._ohlcv_dataframe_cache: Dict[
            Tuple[str, str, int], Tuple[pd.DataFrame, Optional[str]]
        ] = {}
        self.market_prices: Dict[str, float] = {}

    def should_stop(self) -> bool:
        """Check if shutdown was requested."""
        if self.shutdown_event:
            return self.shutdown_event.is_set()
        return False

    def fetch_current_prices_from_binance(self, symbols: list):
        """Fetches current prices for all symbols from Binance."""
        if not symbols:
            return

        log_exchange("Fetching current prices from Binance...")
        progress = ProgressBar(len(symbols), "Price Fetch")

        try:
            # Use authenticated manager for authenticated calls
            exchange = (
                self.exchange_manager.authenticated.connect_to_binance_with_credentials()
            )
        except ValueError as e:
            log_error(f"Error: {e}")
            return

        fetched_count = 0
        failed_symbols = []

        for symbol in symbols:
            if self.should_stop():
                log_warn("Price fetch aborted due to shutdown signal.")
                break
            normalized_symbol = normalize_symbol(symbol)
            try:
                # Use authenticated manager's throttled_call
                ticker = self.exchange_manager.authenticated.throttled_call(
                    exchange.fetch_ticker, normalized_symbol
                )
                if ticker and "last" in ticker:
                    self.market_prices[symbol] = ticker["last"]
                    fetched_count += 1
                    log_exchange(f"[BINANCE] {normalized_symbol}: {ticker['last']:.8f}")
                else:
                    failed_symbols.append(symbol)
                    log_warn(f"{normalized_symbol}: No price data available")
            except Exception as e:
                failed_symbols.append(symbol)
                log_error(f"Error fetching {normalized_symbol}: {e}")
            finally:
                progress.update()
        progress.finish()

        if failed_symbols:
            log_warn(
                f"Warning: Could not fetch prices for {len(failed_symbols)} symbol(s): {', '.join(failed_symbols)}"
            )

        if fetched_count > 0:
            log_success(
                f"Successfully fetched prices for {fetched_count}/{len(symbols)} symbols"
            )

    def fetch_binance_futures_positions(
        self,
        api_key: str = None,
        api_secret: str = None,
        testnet: bool = False,
        debug: bool = False,
    ) -> List[Dict]:
        """
        Fetches open positions from Binance Futures USDT-M.

        Args:
            api_key: API Key from Binance. Priority:
                1. This parameter (if provided)
                2. Environment variable BINANCE_API_KEY
                3. From ExchangeManager's default credentials
            api_secret: API Secret from Binance. Priority:
                1. This parameter (if provided)
                2. Environment variable BINANCE_API_SECRET
                3. From ExchangeManager's default credentials
            testnet: Use testnet if True (default: False)
            debug: Show debug info if True (default: False)

        Returns:
            List of dictionaries containing position information with keys:
            - symbol: Normalized symbol (e.g., 'BTC/USDT')
            - size_usdt: Position size in USDT
            - entry_price: Entry price
            - direction: 'LONG' or 'SHORT'
            - contracts: Absolute number of contracts
        """
        api_key, api_secret = self._resolve_binance_credentials(api_key, api_secret)
        exchange = self._connect_binance_futures(api_key, api_secret, testnet)

        try:
            # Fetch all positions using throttled_call
            positions = self.exchange_manager.authenticated.throttled_call(
                exchange.fetch_positions
            )

            # Filter only open positions (size != 0) and USDT-M
            open_positions = []
            for pos in positions:
                contracts = self._extract_position_contracts(pos)
                if contracts is None or contracts == 0:
                    continue

                normalized_symbol = self._normalize_position_symbol(
                    pos.get("symbol", "")
                )
                if not self._is_usdtm_symbol(normalized_symbol):
                    continue

                entry_price = float(pos.get("entryPrice", 0) or 0)
                if debug:
                    self._debug_position(pos, normalized_symbol, contracts)

                direction = self._determine_position_direction(pos, contracts)
                size_usdt = self._calculate_position_size(
                    pos, contracts, entry_price, exchange
                )

                if size_usdt <= 0:
                    continue

                open_positions.append(
                    {
                        "symbol": normalized_symbol,
                        "size_usdt": size_usdt,
                        "entry_price": entry_price,
                        "direction": direction,
                        "contracts": abs(contracts),
                    }
                )

            return open_positions

        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "api" in error_msg.lower():
                raise ValueError(
                    f"Lỗi xác thực API: {e}\nVui lòng kiểm tra lại API Key và Secret"
                )
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                raise ValueError(f"Lỗi kết nối mạng: {e}")
            else:
                raise ValueError(f"Lỗi khi lấy positions: {e}")

    def _resolve_binance_credentials(
        self, api_key: Optional[str], api_secret: Optional[str]
    ) -> Tuple[str, str]:
        import os

        resolved_key = (
            api_key
            or os.getenv("BINANCE_API_KEY")
            or self.exchange_manager.authenticated.default_api_key
        )
        resolved_secret = (
            api_secret
            or os.getenv("BINANCE_API_SECRET")
            or self.exchange_manager.authenticated.default_api_secret
        )

        if not resolved_key or not resolved_secret:
            raise ValueError(
                "API Key và API Secret là bắt buộc!\n"
                "Cung cấp qua một trong các cách sau:\n"
                "  1. Tham số method: api_key và api_secret\n"
                "  2. Biến môi trường: BINANCE_API_KEY và BINANCE_API_SECRET\n"
                "  3. ExchangeManager credentials (khi khởi tạo ExchangeManager với api_key/api_secret)"
            )
        return resolved_key, resolved_secret

    def _connect_binance_futures(self, api_key: str, api_secret: str, testnet: bool):
        try:
            return self.exchange_manager.authenticated.connect_to_exchange_with_credentials(
                "binance",
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
                contract_type="future",
            )
        except ValueError as exc:
            raise ValueError(f"Error connecting to Binance: {exc}")

    @staticmethod
    def _extract_position_contracts(position: Dict) -> Optional[float]:
        contracts = position.get("contracts")
        if contracts is not None:
            try:
                value = float(contracts or 0)
                if value != 0:
                    return value
            except (ValueError, TypeError):
                pass

        position_amt = position.get("positionAmt", 0)
        if position_amt:
            try:
                value = float(position_amt)
                if value != 0:
                    return value
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def _normalize_position_symbol(symbol: str) -> str:
        if ":" in symbol:
            symbol = symbol.split(":")[0]
        return normalize_symbol(symbol, quote="USDT")

    @staticmethod
    def _is_usdtm_symbol(symbol: str) -> bool:
        return "/USDT" in symbol or symbol.endswith("USDT")

    def _determine_position_direction(self, position: Dict, contracts: float) -> str:
        candidates = [
            position.get("positionSide"),
            position.get("side"),
            (
                (position.get("info") or {}).get("positionSide")
                if isinstance(position.get("info"), dict)
                else None
            ),
        ]

        for candidate in candidates:
            if candidate:
                upper = str(candidate).upper()
                if upper in ["LONG", "SHORT"]:
                    return upper

        # Inspect raw position amount if available
        info = position.get("info")
        if isinstance(info, dict):
            raw_amt = info.get("positionAmt")
            if raw_amt:
                try:
                    amt = float(raw_amt)
                    if amt != 0:
                        return "LONG" if amt > 0 else "SHORT"
                except (ValueError, TypeError):
                    pass

        return "LONG" if contracts > 0 else "SHORT"

    def _calculate_position_size(
        self, position: Dict, contracts: float, entry_price: float, exchange
    ) -> float:
        notional = position.get("notional")
        if notional is not None:
            try:
                value = float(notional)
                if value != 0:
                    return abs(value)
            except (ValueError, TypeError):
                pass

        size_usdt = abs(contracts * entry_price)

        if size_usdt == 0 and entry_price > 0:
            try:
                pos_detail = self.exchange_manager.authenticated.throttled_call(
                    exchange.fetch_position, position.get("symbol", "")
                )
                notional = pos_detail.get("notional", None)
                if notional is not None and notional != 0:
                    size_usdt = abs(float(notional))
            except Exception:
                pass
        return size_usdt

    @staticmethod
    def _debug_position(position: Dict, symbol: str, contracts: float):
        info = position.get("info", {})
        log_debug(f"Position data for {symbol}:")
        log_debug(f"  contracts: {contracts}")
        log_debug(f"  positionSide: {position.get('positionSide')}")
        log_debug(f"  side: {position.get('side')}")
        log_debug(
            f"  info.positionSide: {info.get('positionSide', 'N/A') if isinstance(info, dict) else 'N/A'}"
        )
        log_debug(
            f"  info.positionAmt: {info.get('positionAmt', 'N/A') if isinstance(info, dict) else 'N/A'}"
        )

    def list_binance_futures_symbols(
        self,
        exclude_symbols: Optional[set] = None,
        max_candidates: Optional[int] = None,
        progress_label: str = "Symbol Discovery",
    ) -> List[str]:
        """
        Lists available Binance USDT-M futures symbols sorted by quote volume.

        Args:
            exclude_symbols: Symbols to exclude (normalized format, e.g., 'BTC/USDT').
            max_candidates: Optional maximum number of symbols to return.
            progress_label: Label for the progress bar (default: "Symbol Discovery").

        Returns:
            List of symbol strings sorted by descending volume.
        """
        exclude_symbols = exclude_symbols or set()
        candidates: List[Tuple[str, float]] = []

        try:
            exchange = (
                self.exchange_manager.authenticated.connect_to_binance_with_credentials()
            )
        except ValueError as exc:
            log_error(f"Unable to list hedge candidates: {exc}")
            return []

        try:
            markets = self.exchange_manager.authenticated.throttled_call(
                exchange.load_markets
            )
        except Exception as exc:
            log_error(f"Failed to load Binance markets: {exc}")
            return []

        progress = ProgressBar(len(markets), progress_label)
        seen = set()

        for market in markets.values():
            if self.should_stop():
                log_warn("Symbol discovery aborted due to shutdown.")
                break
            if not market.get("contract"):
                progress.update()
                continue
            if market.get("quote") != "USDT":
                progress.update()
                continue
            if not market.get("active", True):
                progress.update()
                continue

            symbol = self.exchange_manager.normalize_symbol(market.get("symbol", ""))
            if symbol in exclude_symbols or symbol in seen:
                progress.update()
                continue

            info = market.get("info", {})
            volume_str = (
                info.get("volume") or info.get("quoteVolume") or info.get("turnover")
            )
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
        log_exchange(f"Discovered {len(symbol_list)} futures symbols from Binance.")
        return symbol_list

    @staticmethod
    def dataframe_to_close_series(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
        """
        Converts a fetched OHLCV DataFrame into a pandas Series of closing prices indexed by timestamp.
        
        This is a wrapper method for backward compatibility. The actual implementation
        is in modules.common.utils.dataframe_to_close_series().
        """
        return dataframe_to_close_series(df)

    def fetch_ohlcv_with_fallback_exchange(
        self,
        symbol,
        limit=1500,
        timeframe="1h",
        check_freshness=False,
        exchanges=None,
    ):
        """
        Fetches OHLCV data using ccxt with fallback exchanges (with caching).

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            limit: Number of candles to fetch (default: 1500)
            timeframe: Timeframe string (e.g., '1h', '1d') (default: '1h')
            check_freshness: If True, checks data freshness and tries multiple exchanges (default: False)
            exchanges: Optional list of exchange IDs to try. If None, uses exchange_manager's priority list

        Returns:
            Tuple[pd.DataFrame, str]: DataFrame contains full OHLCV data with columns
            ['timestamp', 'open', 'high', 'low', 'close', 'volume'] and exchange_id string.
            Returns (None, None) if data cannot be fetched.
        """
        normalized_symbol = normalize_symbol(symbol)
        cache_key = (normalized_symbol.upper(), timeframe, int(limit))

        # Check cache (only if not checking freshness, as freshness requires fresh data)
        if not check_freshness:
            if cache_key in self._ohlcv_dataframe_cache:
                cached_df, cached_exchange = self._ohlcv_dataframe_cache[cache_key]
                return cached_df.copy(), cached_exchange

        # Determine which exchanges to try
        exchange_list = (
            exchanges or self.exchange_manager.public.exchange_priority_for_fallback
        )

        # Freshness checking setup
        freshness_minutes = None
        fallback = None
        if check_freshness:
            freshness_minutes = max(timeframe_to_minutes(timeframe) * 1.5, 5)
            log_data(f"Fetching {limit} candles for {normalized_symbol} ({timeframe})...")

        last_error = None
        for exchange_id in exchange_list:
            if self.should_stop():
                log_warn("OHLCV fetch cancelled by shutdown.")
                return None, None

            exchange_id = exchange_id.strip()
            if not exchange_id:
                continue

            try:
                # Use public manager for public OHLCV data (no credentials needed)
                exchange = self.exchange_manager.public.connect_to_exchange_with_no_credentials(
                    exchange_id
                )
            except Exception as exc:
                last_error = exc
                if check_freshness:
                    log_warn(f"[{exchange_id.upper()}] Error connecting: {exc}")
                continue

            try:
                # Use public manager's throttled_call
                ohlcv = self.exchange_manager.public.throttled_call(
                    exchange.fetch_ohlcv,
                    normalized_symbol,
                    timeframe=timeframe,
                    limit=limit,
                )
            except Exception as exc:
                last_error = exc
                if check_freshness:
                    log_error(f"[{exchange_id.upper()}] Error fetching data: {exc}")
                continue

            if not ohlcv:
                last_error = ValueError(f"{exchange_id}: empty OHLCV")
                if check_freshness:
                    log_warn(f"[{exchange_id.upper()}] No data retrieved.")
                continue

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            if df.empty:
                last_error = ValueError(f"{exchange_id}: OHLCV dataframe empty")
                if check_freshness:
                    log_warn(f"[{exchange_id.upper()}] No data retrieved.")
                continue

            # Convert timestamp and ensure ordering
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

            # Check freshness if requested
            if check_freshness:
                last_ts = df["timestamp"].iloc[-1]
                now = pd.Timestamp.now(tz="UTC")
                age_minutes = (now - last_ts).total_seconds() / 60.0

                if age_minutes <= freshness_minutes:
                    log_success(f"[{exchange_id.upper()}] Data age {age_minutes:.1f}m (fresh).")
                    self._ohlcv_dataframe_cache[cache_key] = (df.copy(), exchange_id)
                    # When check_freshness=True, always return tuple
                    return df, exchange_id

                log_warn(f"[{exchange_id.upper()}] Data age {age_minutes:.1f}m (stale). Trying next exchange...")
                fallback = (df, exchange_id)
                continue

            # No freshness check - use first successful result
            log_success(f"[OHLCV] {normalized_symbol} loaded from {exchange_id} ({len(df)} bars)")

            self._ohlcv_dataframe_cache[cache_key] = (df.copy(), exchange_id)
            return df, exchange_id

        # Handle fallback for stale data
        if check_freshness and fallback:
            df, exchange_id = fallback
            log_info(f"Using latest available data from {exchange_id.upper()} despite staleness.")
            self._ohlcv_dataframe_cache[cache_key] = (df.copy(), exchange_id)
            # When check_freshness=True, always return tuple
            return df, exchange_id

        # Failed to fetch
        log_error(f"Failed to fetch OHLCV for {normalized_symbol}: {last_error}")
        return None, None
