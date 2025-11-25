import pandas as pd

from modules.pairs_trading.pair_metrics_computer import PairMetricsComputer


def test_compute_pair_metrics_collects_all_sources(monkeypatch):
    monkeypatch.setattr(
        "modules.pairs_trading.pair_metrics_computer.calculate_ols_hedge_ratio",
        lambda p1, p2: 2.0,
    )
    monkeypatch.setattr(
        "modules.pairs_trading.pair_metrics_computer.calculate_adf_test",
        lambda spread, min_points: {"adf_pvalue": 0.02},
    )
    monkeypatch.setattr(
        "modules.pairs_trading.pair_metrics_computer.calculate_half_life",
        lambda spread: 15.0,
    )
    monkeypatch.setattr(
        "modules.pairs_trading.pair_metrics_computer.calculate_zscore_stats",
        lambda spread, lookback: {
            "mean_zscore": 0.0,
            "std_zscore": 1.0,
            "skewness": 0.1,
            "kurtosis": 3.0,
            "current_zscore": 1.5,
        },
    )
    monkeypatch.setattr(
        "modules.pairs_trading.pair_metrics_computer.calculate_hurst_exponent",
        lambda spread, lookback: 0.4,
    )
    monkeypatch.setattr(
        "modules.pairs_trading.pair_metrics_computer.calculate_spread_sharpe",
        lambda spread, periods: 1.2,
    )
    monkeypatch.setattr(
        "modules.pairs_trading.pair_metrics_computer.calculate_max_drawdown",
        lambda spread: -0.2,
    )
    monkeypatch.setattr(
        "modules.pairs_trading.pair_metrics_computer.calculate_calmar_ratio",
        lambda spread, periods: 1.1,
    )
    monkeypatch.setattr(
        "modules.pairs_trading.pair_metrics_computer.calculate_johansen_test",
        lambda p1, p2, min_pt, conf: {
            "johansen_trace_stat": 15.0,
            "johansen_critical_value": 14.0,
            "is_johansen_cointegrated": True,
        },
    )
    monkeypatch.setattr(
        "modules.pairs_trading.pair_metrics_computer.calculate_kalman_hedge_ratio",
        lambda p1, p2: 1.8,
    )
    monkeypatch.setattr(
        "modules.pairs_trading.pair_metrics_computer.calculate_direction_metrics",
        lambda spread, lookback, threshold: {
            "classification_f1": 0.7,
            "classification_precision": 0.75,
            "classification_recall": 0.65,
            "classification_accuracy": 0.8,
        },
    )

    price1 = pd.Series([10 + i for i in range(100)], dtype=float)
    price2 = pd.Series([5 + i * 0.5 for i in range(100)], dtype=float)

    computer = PairMetricsComputer()
    metrics = computer.compute_pair_metrics(price1, price2)

    assert metrics["hedge_ratio"] == 2.0
    assert metrics["is_cointegrated"] is True
    assert metrics["half_life"] == 15.0
    assert metrics["hurst_exponent"] == 0.4
    assert metrics["spread_sharpe"] == 1.2
    assert metrics["johansen_trace_stat"] == 15.0
    assert metrics["kalman_hedge_ratio"] == 1.8
    assert metrics["classification_f1"] == 0.7

