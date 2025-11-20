"""
Data fetching functions for cryptocurrency market data.
"""
import ccxt
import pandas as pd
from .utils import color_text, timeframe_to_minutes
from colorama import Fore, Style
from .config import DEFAULT_EXCHANGES, DEFAULT_QUOTE


def normalize_symbol(user_input: str, quote: str = DEFAULT_QUOTE) -> str:
    """
    Converts user input like 'xmr' into 'XMR/USDT'. Keeps existing slash pairs.
    """
    if not user_input:
        return f"BTC/{quote}"

    norm = user_input.strip().upper()
    if "/" in norm:
        return norm

    if norm.endswith(quote):
        return f"{norm[:-len(quote)]}/{quote}"

    return f"{norm}/{quote}"


def fetch_data(symbol="BTC/USDT", timeframe="1h", limit=1000, exchanges=None):
    """
    Fetches OHLCV data trying multiple exchanges until fresh data is returned.
    """
    exchanges = exchanges or DEFAULT_EXCHANGES
    freshness_minutes = max(timeframe_to_minutes(timeframe) * 1.5, 5)
    fallback = None

    print(
        color_text(
            f"Fetching {limit} candles for {symbol} ({timeframe})...",
            Fore.CYAN,
            Style.BRIGHT,
        )
    )
    for exchange_id in exchanges:
        exchange_cls = getattr(ccxt, exchange_id)
        exchange = exchange_cls()
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            if df.empty:
                print(
                    color_text(
                        f"[{exchange_id.upper()}] No data retrieved.", Fore.YELLOW
                    )
                )
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            last_ts = df["timestamp"].iloc[-1]
            now = pd.Timestamp.now(tz="UTC")
            age_minutes = (now - last_ts).total_seconds() / 60.0

            if age_minutes <= freshness_minutes:
                print(
                    color_text(
                        f"[{exchange_id.upper()}] Data age {age_minutes:.1f}m (fresh).",
                        Fore.GREEN,
                    )
                )
                return df, exchange_id

            print(
                color_text(
                    f"[{exchange_id.upper()}] Data age {age_minutes:.1f}m (stale). Trying next exchange...",
                    Fore.YELLOW,
                )
            )
            fallback = (df, exchange_id)
        except Exception as e:
            print(
                color_text(
                    f"[{exchange_id.upper()}] Error fetching data: {e}", Fore.RED
                )
            )
            continue

    if fallback:
        df, exchange_id = fallback
        print(
            color_text(
                f"Using latest available data from {exchange_id.upper()} despite staleness.",
                Fore.MAGENTA,
            )
        )
        return df, exchange_id

    print(
        color_text("Failed to fetch data from all exchanges.", Fore.RED, Style.BRIGHT)
    )
    return None, None

