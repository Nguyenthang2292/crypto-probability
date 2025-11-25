import numpy as np
import pandas as pd

from modules.pairs_trading import hedge_ratio


def test_calculate_ols_hedge_ratio_recovers_linear_beta():
    price2 = pd.Series(np.linspace(1, 20, 50))
    price1 = 2.5 * price2 + 5

    beta = hedge_ratio.calculate_ols_hedge_ratio(price1, price2)
    assert beta is not None
    assert np.isclose(beta, 2.5, atol=1e-2)


def test_calculate_kalman_hedge_ratio_uses_stubbed_filter(monkeypatch):
    class DummyKalman:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def filter(self, observations):
            steps = len(observations)
            means = np.column_stack([np.linspace(0.5, 1.5, steps), np.zeros(steps)])
            cov = np.zeros((steps, 2))
            return means, cov

    monkeypatch.setattr(hedge_ratio, "KalmanFilter", DummyKalman)

    price2 = pd.Series(np.linspace(1, 10, 40))
    price1 = pd.Series(np.linspace(2, 12, 40))

    beta = hedge_ratio.calculate_kalman_hedge_ratio(price1, price2)

    assert np.isclose(beta, 1.5)

