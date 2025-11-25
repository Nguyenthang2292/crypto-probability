import pandas as pd
import numpy as np

from modules.pairs_trading import statistical_tests


def test_calculate_adf_test_uses_stub(monkeypatch):
    called = {}

    def fake_adfuller(series, maxlag, autolag):
        called["data"] = series.tolist()
        return (-3.2, 0.01, None, None, {"5%": -2.9})

    monkeypatch.setattr(statistical_tests, "adfuller", fake_adfuller)
    spread = pd.Series(np.linspace(1, 100, 60))

    result = statistical_tests.calculate_adf_test(spread, min_points=30)

    assert called["data"][0] == 1.0
    assert result == {
        "adf_statistic": -3.2,
        "adf_pvalue": 0.01,
        "critical_values": {"5%": -2.9},
    }


def test_calculate_half_life_with_stubbed_regression(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.coef_ = [-0.1]

        def fit(self, X, y):
            pass

    monkeypatch.setattr(statistical_tests, "LinearRegression", FakeModel)

    spread = pd.Series(np.linspace(100, 80, 50))
    result = statistical_tests.calculate_half_life(spread)
    expected = -np.log(2) / -0.1

    assert result == expected


def test_calculate_johansen_test_with_stub(monkeypatch):
    class DummyResult:
        def __init__(self):
            self.lr1 = np.array([20.0])
            self.cvt = np.array([[15.0, 18.0, 25.0]])

    def fake_coint_johansen(data, det_order, k_ar_diff):
        return DummyResult()

    monkeypatch.setattr(statistical_tests, "coint_johansen", fake_coint_johansen)

    price1 = pd.Series(np.arange(60, dtype=float))
    price2 = pd.Series(np.arange(60, dtype=float) * 0.5)

    result = statistical_tests.calculate_johansen_test(price1, price2, min_points=30, confidence=0.95)

    assert result == {
        "johansen_trace_stat": 20.0,
        "johansen_critical_value": 18.0,
        "is_johansen_cointegrated": True,
    }

