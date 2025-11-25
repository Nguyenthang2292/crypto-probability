import math
from dataclasses import dataclass

import pandas as pd

from modules.common.DataFetcher import DataFetcher
from modules.pairs_trading.performance_analyzer import PerformanceAnalyzer
from modules.pairs_trading.pairs_analyzer import PairsTradingAnalyzer


def _generate_ohlcv_series(base_price: float, drift: float, length: int = 240):
    """Create deterministic OHLCV rows with hourly spacing."""
    rows = []
    timestamp = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    price = base_price
    for idx in range(length):
        # Smooth deterministic drift plus small oscillation to avoid flat returns
        price = max(price + drift + math.sin(idx / 10.0) * 0.2, 1.0)
        open_price = price * 0.999
        high_price = price * 1.001
        low_price = price * 0.999
        volume = 50000 + idx * 10
        rows.append(
            [
                int(timestamp.value / 1_000_000) + idx * 3600 * 1000,
                open_price,
                high_price,
                low_price,
                price,
                volume,
            ]
        )
    return rows


class StubExchange:
    def __init__(self, data_map):
        self.data_map = data_map

    def fetch_ohlcv(self, symbol, timeframe="1h", limit=200):
        series = self.data_map[symbol]
        return series[-limit:]


class StubPublicManager:
    def __init__(self, data_map):
        self.exchange_priority_for_fallback = ["stub"]
        self.data_map = data_map

    def connect_to_exchange_with_no_credentials(self, exchange_id):
        if exchange_id != "stub":
            raise ValueError("Unexpected exchange id")
        return StubExchange(self.data_map)

    def throttled_call(self, func, *args, **kwargs):
        return func(*args, **kwargs)


@dataclass
class StubExchangeManager:
    data_map: dict

    def __post_init__(self):
        self.public = StubPublicManager(self.data_map)
        self.authenticated = None  # Not required for these integration tests

    def normalize_symbol(self, symbol: str) -> str:
        return symbol


def _build_data_map():
    return {
        "AAA/USDT": _generate_ohlcv_series(50.0, drift=0.3),
        "BBB/USDT": _generate_ohlcv_series(25.0, drift=-0.2),
        "CCC/USDT": _generate_ohlcv_series(10.0, drift=0.05),
    }


def test_performance_and_pairs_pipeline_with_real_data_fetcher(monkeypatch):
    data_map = _build_data_map()
    exchange_manager = StubExchangeManager(data_map)
    data_fetcher = DataFetcher(exchange_manager)

    performance_analyzer = PerformanceAnalyzer(limit=200, timeframe="1h")
    symbols = ["AAA/USDT", "BBB/USDT", "CCC/USDT"]
    performance_df = performance_analyzer.analyze_all_symbols(
        symbols, data_fetcher, verbose=False
    )

    assert not performance_df.empty
    assert list(performance_df["symbol"]) == ["AAA/USDT", "CCC/USDT", "BBB/USDT"]

    best = performance_analyzer.get_top_performers(performance_df, top_n=1)
    worst = performance_analyzer.get_worst_performers(performance_df, top_n=1)

    # Simplify heavy metrics while still exercising analyzer logic
    monkeypatch.setattr(
        "modules.pairs_trading.pairs_analyzer.PairMetricsComputer.compute_pair_metrics",
        lambda self, price1, price2: {
            "is_cointegrated": True,
            "half_life": 10.0,
            "current_zscore": 1.2,
            "spread_sharpe": 1.0,
            "classification_f1": 0.7,
        },
    )

    analyzer = PairsTradingAnalyzer(
        min_volume=0,
        min_spread=0.0,
        max_spread=2.0,
        min_correlation=0.0,
        max_correlation=1.0,
        correlation_min_points=50,
    )

    pairs_df = analyzer.analyze_pairs_opportunity(
        best, worst, data_fetcher=data_fetcher, verbose=False
    )

    assert not pairs_df.empty
    assert pairs_df.iloc[0]["long_symbol"] == worst.iloc[0]["symbol"]
    assert pairs_df.iloc[0]["short_symbol"] == best.iloc[0]["symbol"]
    assert pairs_df.iloc[0]["correlation"] is not None

    validated = analyzer.validate_pairs(pairs_df, data_fetcher, verbose=False)
    assert len(validated) == len(pairs_df)


def test_data_fetcher_cache_reuse_integration():
    data_map = _build_data_map()
    exchange_manager = StubExchangeManager(data_map)
    data_fetcher = DataFetcher(exchange_manager)

    df_first, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
        "AAA/USDT", limit=100, timeframe="1h"
    )
    df_second, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
        "AAA/USDT", limit=100, timeframe="1h"
    )

    assert df_first.equals(df_second)
    # Ensure no mutation occurs between cached copies
    df_first.iloc[0, df_first.columns.get_loc("close")] = 999.0
    df_third, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
        "AAA/USDT", limit=100, timeframe="1h"
    )
    assert df_third.iloc[0]["close"] != 999.0

