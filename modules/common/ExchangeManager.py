"""
Exchange manager for handling exchange connections and API calls.
Refactored: Separated into AuthenticatedExchangeManager and PublicExchangeManager.
"""

import ccxt
import os
import time
import threading
from typing import Dict, Optional

# Import normalize_symbol from utils (core module, should always be available)
from modules.common.utils import normalize_symbol

try:
    from modules.config import (
        DEFAULT_EXCHANGE_STRING,
        DEFAULT_REQUEST_PAUSE,
        DEFAULT_CONTRACT_TYPE,
        DEFAULT_EXCHANGES,
    )
    from modules.config_api import BINANCE_API_KEY, BINANCE_API_SECRET
except ImportError:
    DEFAULT_EXCHANGE_STRING = "binance,kraken,kucoin,gate,okx,bybit,mexc,huobi"
    DEFAULT_REQUEST_PAUSE = 0.2
    DEFAULT_CONTRACT_TYPE = "future"
    DEFAULT_EXCHANGES = [
        "binance",
        "kraken",
        "kucoin",
        "gate",
        "okx",
        "bybit",
        "mexc",
        "huobi",
    ]
    BINANCE_API_KEY = None
    BINANCE_API_SECRET = None


class AuthenticatedExchangeManager:
    """Manages authenticated exchange connections (requires API credentials)."""

    def __init__(
        self,
        api_key=None,
        api_secret=None,
        testnet=False,
        request_pause=None,
        contract_type=None,
    ):
        """
        Initialize AuthenticatedExchangeManager.

        Args:
            api_key: Default API key (for Binance backward compatibility)
            api_secret: Default API secret (for Binance backward compatibility)
            testnet: Use testnet if True
            request_pause: Pause between requests in seconds
            contract_type: Contract type ('spot', 'margin', 'future'). Defaults to DEFAULT_CONTRACT_TYPE
        """
        # Store default credentials for Binance (backward compatibility)
        self.default_api_key = (
            api_key or os.getenv("BINANCE_API_KEY") or BINANCE_API_KEY
        )
        self.default_api_secret = (
            api_secret or os.getenv("BINANCE_API_SECRET") or BINANCE_API_SECRET
        )
        self.testnet = testnet
        self.contract_type = contract_type or os.getenv(
            "DEFAULT_CONTRACT_TYPE", DEFAULT_CONTRACT_TYPE
        )

        # Cache for authenticated exchanges (key: exchange_id)
        self._authenticated_exchanges: Dict[str, ccxt.Exchange] = {}
        # Store credentials per exchange (key: exchange_id)
        self._exchange_credentials: Dict[str, Dict[str, str]] = {}

        self.request_pause = float(
            request_pause or os.getenv("BINANCE_REQUEST_SLEEP", DEFAULT_REQUEST_PAUSE)
        )
        self._request_lock = threading.Lock()
        self._last_request_ts = 0.0

    def connect_to_exchange_with_credentials(
        self,
        exchange_id: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated exchange instance (REQUIRES credentials).

        Supports multiple exchanges: binance, okx, kucoin, bybit, gate, mexc, huobi, etc.

        Args:
            exchange_id: Exchange name (e.g., 'binance', 'okx', 'kucoin', 'bybit')
            api_key: API key for this exchange (optional, uses default if not provided)
            api_secret: API secret for this exchange (optional, uses default if not provided)
            testnet: Use testnet if True (optional, uses instance default if not provided)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated exchange instance

        Raises:
            ValueError: If exchange not supported or credentials not provided
        """
        exchange_id = exchange_id.strip().lower()
        testnet = testnet if testnet is not None else self.testnet
        contract_type = contract_type or self.contract_type

        # Check cache
        cache_key = f"{exchange_id}_{testnet}_{contract_type}"
        if cache_key in self._authenticated_exchanges:
            return self._authenticated_exchanges[cache_key]

        # Get credentials (per-exchange or default)
        if exchange_id == "binance":
            # Use default credentials for Binance (backward compatibility)
            cred_key = api_key or self.default_api_key
            cred_secret = api_secret or self.default_api_secret
        else:
            # For other exchanges, try per-exchange credentials or default
            exchange_creds = self._exchange_credentials.get(exchange_id, {})
            cred_key = api_key or exchange_creds.get("api_key") or self.default_api_key
            cred_secret = (
                api_secret
                or exchange_creds.get("api_secret")
                or self.default_api_secret
            )

        if not cred_key or not cred_secret:
            raise ValueError(
                f"API Key và API Secret là bắt buộc cho {exchange_id}!\n"
                f"Cung cấp qua một trong các cách sau:\n"
                f"  1. Tham số khi gọi connect_to_exchange_with_credentials()\n"
                f"  2. Sử dụng set_exchange_credentials() để set credentials cho exchange\n"
                f"  3. Biến môi trường: {exchange_id.upper()}_API_KEY và {exchange_id.upper()}_API_SECRET\n"
                f"  4. File config: modules/config_api.py"
            )

        # Check if exchange is supported
        if not hasattr(ccxt, exchange_id):
            raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt.")

        exchange_class = getattr(ccxt, exchange_id)

        # Build options with configurable contract type
        options = {
            "defaultType": contract_type,
            "options": {
                "defaultType": contract_type,
            },
        }

        # Build params
        params = {
            "apiKey": cred_key,
            "secret": cred_secret,
            "options": options,
            "enableRateLimit": True,
        }

        # Add testnet/sandbox for supported exchanges
        if testnet:
            if exchange_id == "binance":
                params["sandbox"] = True
            elif exchange_id in ["okx", "kucoin", "bybit"]:
                params["sandbox"] = True
            # Some exchanges use 'test' instead of 'sandbox'
            elif exchange_id in ["gate"]:
                params["options"]["sandboxMode"] = True

        try:
            exchange_instance = exchange_class(params)
            self._authenticated_exchanges[cache_key] = exchange_instance
            return exchange_instance
        except Exception as exc:
            raise ValueError(
                f"Cannot initialize authenticated {exchange_id} exchange: {exc}"
            )

    def set_exchange_credentials(self, exchange_id: str, api_key: str, api_secret: str):
        """
        Set credentials for a specific exchange.

        Args:
            exchange_id: Exchange name (e.g., 'okx', 'kucoin', 'bybit')
            api_key: API key for this exchange
            api_secret: API secret for this exchange
        """
        exchange_id = exchange_id.strip().lower()
        self._exchange_credentials[exchange_id] = {
            "api_key": api_key,
            "api_secret": api_secret,
        }
        # Clear cached exchange if exists to force reconnection with new credentials
        keys_to_remove = [
            k
            for k in self._authenticated_exchanges.keys()
            if k.startswith(f"{exchange_id}_")
        ]
        for key in keys_to_remove:
            del self._authenticated_exchanges[key]

    def update_default_credentials(
        self, api_key: Optional[str] = None, api_secret: Optional[str] = None
    ):
        """
        Update default credentials used for authenticated exchanges and clear caches.

        Args:
            api_key: New default API key (if None, keep existing)
            api_secret: New default API secret (if None, keep existing)
        """
        updated = False
        if api_key is not None:
            self.default_api_key = api_key
            updated = True
        if api_secret is not None:
            self.default_api_secret = api_secret
            updated = True
        if updated:
            self._authenticated_exchanges.clear()

    def connect_to_binance_with_credentials(self) -> ccxt.Exchange:
        """
        Connect to authenticated Binance exchange instance (REQUIRES credentials).

        Use this for:
        - fetch_ticker() - Get current prices
        - load_markets() - List available symbols
        - fetch_positions() - Get account positions
        - Any authenticated API calls

        Returns:
            ccxt.Exchange: Authenticated Binance exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return self.connect_to_exchange_with_credentials("binance")

    def connect_to_kraken_with_credentials(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated Kraken exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('kraken').

        Args:
            api_key: API key for Kraken (optional, uses set credentials or default)
            api_secret: API secret for Kraken (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated Kraken exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return self.connect_to_exchange_with_credentials(
            "kraken", api_key, api_secret, testnet, contract_type
        )

    def connect_to_kucoin_with_credentials(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated KuCoin exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('kucoin').

        Args:
            api_key: API key for KuCoin (optional, uses set credentials or default)
            api_secret: API secret for KuCoin (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated KuCoin exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return self.connect_to_exchange_with_credentials(
            "kucoin", api_key, api_secret, testnet, contract_type
        )

    def connect_to_gate_with_credentials(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated Gate.io exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('gate').

        Args:
            api_key: API key for Gate.io (optional, uses set credentials or default)
            api_secret: API secret for Gate.io (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated Gate.io exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return self.connect_to_exchange_with_credentials(
            "gate", api_key, api_secret, testnet, contract_type
        )

    def connect_to_okx_with_credentials(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated OKX exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('okx').

        Args:
            api_key: API key for OKX (optional, uses set credentials or default)
            api_secret: API secret for OKX (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated OKX exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return self.connect_to_exchange_with_credentials(
            "okx", api_key, api_secret, testnet, contract_type
        )

    def connect_to_bybit_with_credentials(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated Bybit exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('bybit').

        Args:
            api_key: API key for Bybit (optional, uses set credentials or default)
            api_secret: API secret for Bybit (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated Bybit exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return self.connect_to_exchange_with_credentials(
            "bybit", api_key, api_secret, testnet, contract_type
        )

    def connect_to_mexc_with_credentials(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated MEXC exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('mexc').

        Args:
            api_key: API key for MEXC (optional, uses set credentials or default)
            api_secret: API secret for MEXC (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated MEXC exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return self.connect_to_exchange_with_credentials(
            "mexc", api_key, api_secret, testnet, contract_type
        )

    def connect_to_huobi_with_credentials(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
        contract_type: Optional[str] = None,
    ) -> ccxt.Exchange:
        """
        Connect to authenticated Huobi exchange instance (REQUIRES credentials).

        Convenience method for connect_to_exchange_with_credentials('huobi').

        Args:
            api_key: API key for Huobi (optional, uses set credentials or default)
            api_secret: API secret for Huobi (optional, uses set credentials or default)
            testnet: Use testnet if True (optional, uses instance default)
            contract_type: Contract type ('spot', 'margin', 'future') (optional, uses config default)

        Returns:
            ccxt.Exchange: Authenticated Huobi exchange instance

        Raises:
            ValueError: If API key/secret not provided
        """
        return self.connect_to_exchange_with_credentials(
            "huobi", api_key, api_secret, testnet, contract_type
        )

    def throttled_call(self, func, *args, **kwargs):
        """
        Ensures a minimum delay between REST calls to respect rate limits.

        Args:
            func: Function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func(*args, **kwargs)
        """
        with self._request_lock:
            wait = self.request_pause - (time.time() - self._last_request_ts)
            if wait > 0:
                time.sleep(wait)
            result = func(*args, **kwargs)
            self._last_request_ts = time.time()
            return result


class PublicExchangeManager:
    """Manages public exchange connections (no credentials required)."""

    def __init__(self, request_pause=None):
        self.request_pause = float(
            request_pause or os.getenv("BINANCE_REQUEST_SLEEP", DEFAULT_REQUEST_PAUSE)
        )
        self._request_lock = threading.Lock()
        self._last_request_ts = 0.0
        self._public_exchanges: Dict[str, ccxt.Exchange] = {}
        fallback_string = os.getenv("OHLCV_FALLBACKS", DEFAULT_EXCHANGE_STRING)
        self._exchange_priority_for_fallback = fallback_string.split(",")

    def connect_to_exchange_with_no_credentials(
        self, exchange_id: str
    ) -> ccxt.Exchange:
        """
        Connect to public exchange instance (NO credentials required).

        Use this for:
        - fetch_ohlcv() - Get historical OHLCV data (public data)
        - fetch_ticker() - Get current prices (public data, if supported)
        - Any public API calls that don't require authentication

        Args:
            exchange_id: Exchange name (e.g., 'binance', 'kraken', 'kucoin')

        Returns:
            ccxt.Exchange: Public exchange instance (no authentication)

        Raises:
            ValueError: If exchange is not supported by ccxt or cannot be initialized
        """
        exchange_id = exchange_id.strip().lower()

        # Cache check
        if exchange_id not in self._public_exchanges:
            if not hasattr(ccxt, exchange_id):
                raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt.")

            exchange_class = getattr(ccxt, exchange_id)
            contract_type = os.getenv("DEFAULT_CONTRACT_TYPE", DEFAULT_CONTRACT_TYPE)
            params = {
                "enableRateLimit": True,
                "options": {
                    "defaultType": contract_type,
                },
            }

            try:
                self._public_exchanges[exchange_id] = exchange_class(params)
            except Exception as exc:
                raise ValueError(f"Cannot initialize exchange {exchange_id}: {exc}")

        return self._public_exchanges[exchange_id]

    def throttled_call(self, func, *args, **kwargs):
        """
        Ensures a minimum delay between REST calls to respect rate limits.

        Args:
            func: Function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func(*args, **kwargs)
        """
        with self._request_lock:
            wait = self.request_pause - (time.time() - self._last_request_ts)
            if wait > 0:
                time.sleep(wait)
            result = func(*args, **kwargs)
            self._last_request_ts = time.time()
            return result

    @property
    def exchange_priority_for_fallback(self):
        """Get list of exchange IDs in priority order for fallback."""
        return self._exchange_priority_for_fallback

    @exchange_priority_for_fallback.setter
    def exchange_priority_for_fallback(self, value):
        """Set list of exchange IDs in priority order for fallback."""
        self._exchange_priority_for_fallback = value


class ExchangeManager:
    """
    Composite manager that combines AuthenticatedExchangeManager and PublicExchangeManager.

    This class provides a unified interface while maintaining clear separation
    between authenticated and public exchange operations.
    """

    def __init__(self, api_key=None, api_secret=None, testnet=False):
        """
        Initialize ExchangeManager with optional credentials.

        Args:
            api_key: Binance API key (optional, can be set via env or config)
            api_secret: Binance API secret (optional, can be set via env or config)
            testnet: Use Binance testnet if True
        """
        # Initialize sub-managers
        self.authenticated = AuthenticatedExchangeManager(api_key, api_secret, testnet)
        self.public = PublicExchangeManager()

        # Store credentials for backward compatibility
        self.api_key = api_key or os.getenv("BINANCE_API_KEY") or BINANCE_API_KEY
        self.api_secret = (
            api_secret or os.getenv("BINANCE_API_SECRET") or BINANCE_API_SECRET
        )
        self.testnet = testnet

    def normalize_symbol(self, market_symbol: str) -> str:
        """
        Normalize market symbol by converting Binance futures symbols like BTC/USDT:USDT into BTC/USDT.

        This method handles the conversion of exchange-specific symbol formats to a standardized format.
        It first removes contract markers (e.g., ':USDT') and then uses the utility normalize_symbol function.

        Args:
            market_symbol: Market symbol from exchange (e.g., 'BTC/USDT:USDT', 'ETHUSDT')

        Returns:
            str: Normalized symbol in format 'BASE/QUOTE' (e.g., 'BTC/USDT')
        """
        if ":" in market_symbol:
            market_symbol = market_symbol.split(":")[0]
        return normalize_symbol(market_symbol)

    @property
    def exchange_priority_for_fallback(self):
        """Get list of exchange IDs in priority order for OHLCV fallback."""
        return self.public.exchange_priority_for_fallback

    @exchange_priority_for_fallback.setter
    def exchange_priority_for_fallback(self, value):
        """Set list of exchange IDs in priority order for OHLCV fallback."""
        self.public.exchange_priority_for_fallback = value
