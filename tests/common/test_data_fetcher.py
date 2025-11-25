from types import SimpleNamespace

import pandas as pd

from modules.DataFetcher import DataFetcher


def _build_ohlcv(last_timestamp_ms: int):
    step = 60_000
    return [
        [last_timestamp_ms - 2 * step, 1, 2, 0.5, 1.5, 10],
        [last_timestamp_ms - step, 1.6, 2.1, 1.0, 1.8, 11],
        [last_timestamp_ms, 1.9, 2.3, 1.7, 2.0, 12],
    ]


class DummyExchange:
    def __init__(self, data, call_tracker=None):
        self._data = data
        self._call_tracker = call_tracker

    def fetch_ohlcv(self, *args, **kwargs):
        if isinstance(self._data, Exception):
            raise self._data
        if self._call_tracker is not None:
            self._call_tracker["calls"] += 1
        return self._data


class DummyPublic:
    def __init__(self, priority, responses):
        self.exchange_priority_for_fallback = priority
        self._responses = responses

    def connect_to_exchange_with_no_credentials(self, exchange_id: str):
        response = self._responses[exchange_id]
        if isinstance(response, Exception):
            raise response
        data = response if not isinstance(response, tuple) else response[0]
        tracker = None if not isinstance(response, tuple) else response[1]
        return DummyExchange(data, tracker)

    @staticmethod
    def throttled_call(func, *args, **kwargs):
        return func(*args, **kwargs)


def test_fetch_ohlcv_with_fallback_prefers_fresh_data(monkeypatch):
    now = pd.Timestamp.now(tz="UTC")
    stale_last = int((now - pd.Timedelta(minutes=180)).timestamp() * 1000)
    fresh_last = int((now - pd.Timedelta(minutes=10)).timestamp() * 1000)

    responses = {
        "binance": _build_ohlcv(stale_last),
        "kraken": _build_ohlcv(fresh_last),
    }
    public = DummyPublic(["binance", "kraken"], responses)
    exchange_manager = SimpleNamespace(public=public)
    fetcher = DataFetcher(exchange_manager)

    df, exchange_id = fetcher.fetch_ohlcv_with_fallback_exchange(
        "eth/usdt", limit=3, timeframe="1h", check_freshness=True
    )

    assert exchange_id == "kraken"
    assert len(df) == 3
    assert df["timestamp"].iloc[-1] == pd.to_datetime(fresh_last, unit="ms", utc=True)


def test_fetch_ohlcv_uses_cache_when_not_checking_freshness():
    now = pd.Timestamp.now(tz="UTC")
    last_ts = int(now.timestamp() * 1000)

    tracker = {"calls": 0}
    responses = {
        "binance": (_build_ohlcv(last_ts), tracker),
    }
    public = DummyPublic(["binance"], responses)
    exchange_manager = SimpleNamespace(public=public)
    fetcher = DataFetcher(exchange_manager)

    df1, _ = fetcher.fetch_ohlcv_with_fallback_exchange(
        "btc/usdt", limit=3, timeframe="1h", check_freshness=False
    )
    df2, _ = fetcher.fetch_ohlcv_with_fallback_exchange(
        "btc/usdt", limit=3, timeframe="1h", check_freshness=False
    )

    assert tracker["calls"] == 1  # second call served from cache
    assert df1.equals(df2)


def test_fetch_ohlcv_returns_stale_fallback_when_no_fresh_data():
    now = pd.Timestamp.now(tz="UTC")
    stale_last = int((now - pd.Timedelta(minutes=300)).timestamp() * 1000)
    responses = {
        "binance": _build_ohlcv(stale_last),
    }
    public = DummyPublic(["binance"], responses)
    exchange_manager = SimpleNamespace(public=public)
    fetcher = DataFetcher(exchange_manager)

    df, exchange_id = fetcher.fetch_ohlcv_with_fallback_exchange(
        "ada/usdt", limit=3, timeframe="1h", check_freshness=True
    )

    assert exchange_id == "binance"
    assert df is not None


def test_dataframe_to_close_series_converts_dataframe():
    now = pd.Timestamp.now(tz="UTC")
    data = pd.DataFrame(
        {
            "timestamp": [now - pd.Timedelta(minutes=i) for i in range(3)],
            "close": [10.0, 11.0, 12.0],
        }
    )

    series = DataFetcher.dataframe_to_close_series(data)

    assert series is not None
    assert list(series.values) == [10.0, 11.0, 12.0]
