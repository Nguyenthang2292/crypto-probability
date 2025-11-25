import pandas as pd
import numpy as np
from types import SimpleNamespace

from modules.Position import Position
from modules.PortfolioRiskCalculator import PortfolioRiskCalculator


def test_calculate_stats_handles_long_and_short_positions():
    positions = [
        Position("BTC/USDT", "LONG", entry_price=100.0, size_usdt=1000.0),
        Position("ETH/USDT", "SHORT", entry_price=200.0, size_usdt=500.0),
    ]
    prices = {"BTC/USDT": 110.0, "ETH/USDT": 190.0}
    dummy_fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=lambda *a, **k: (None, None),
        dataframe_to_close_series=lambda df: None,
    )

    calc = PortfolioRiskCalculator(dummy_fetcher)
    calc.calculate_beta = lambda *args, **kwargs: None
    df, total_pnl, total_delta, total_beta_delta = calc.calculate_stats(
        positions, prices
    )

    assert len(df) == 2
    assert total_pnl > 0
    assert total_delta == 500.0  # 1000 long - 500 short
    # Beta data missing -> total_beta_delta stays 0
    assert total_beta_delta == 0


def test_calculate_beta_returns_none_when_not_enough_data(monkeypatch):
    close_series = pd.Series(
        [100, 101, 102],
        index=pd.date_range("2023-01-01", periods=3, freq="h"),
        name="close",
    )

    def fake_fetch(symbol, **kwargs):
        df = pd.DataFrame(
            {"timestamp": close_series.index, "close": close_series.values}
        )
        return df, "binance"

    fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
        dataframe_to_close_series=lambda df: df.set_index("timestamp")["close"],
    )

    calc = PortfolioRiskCalculator(fetcher)
    beta = calc.calculate_beta("XRP/USDT", benchmark_symbol="BTC/USDT", min_points=10)

    assert beta is None


def test_calculate_beta_uses_cache(monkeypatch):
    timestamps = pd.date_range("2023-01-01", periods=20, freq="h")

    def fake_fetch(symbol, **kwargs):
        rng = np.random.default_rng(42 if symbol.startswith("BTC") else 43)
        prices = rng.normal(100, 1, size=len(timestamps)).cumsum()
        df = pd.DataFrame({"timestamp": timestamps, "close": prices})
        return df, "binance"

    fetcher = SimpleNamespace(
        fetch_ohlcv_with_fallback_exchange=fake_fetch,
        dataframe_to_close_series=lambda df: df.set_index("timestamp")["close"],
    )

    calc = PortfolioRiskCalculator(fetcher)
    beta1 = calc.calculate_beta("BTC/USDT", benchmark_symbol="ETH/USDT", min_points=5)
    beta2 = calc.calculate_beta("BTC/USDT", benchmark_symbol="ETH/USDT", min_points=5)

    assert beta1 == beta2


def test_calculate_portfolio_var_returns_none_without_positions(capfd):
    fetcher = SimpleNamespace()
    calc = PortfolioRiskCalculator(fetcher)

    result = calc.calculate_portfolio_var([])

    assert result is None
    captured = capfd.readouterr()
    assert "No positions available for VaR" in captured.out
