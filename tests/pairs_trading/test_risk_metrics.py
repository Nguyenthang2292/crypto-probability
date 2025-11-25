import numpy as np
import pandas as pd

from modules.pairs_trading import risk_metrics


def test_calculate_spread_sharpe_matches_manual_computation():
    spread = pd.Series([100, 102, 101, 105, 107, 110], dtype=float)
    periods = 4

    expected_returns = spread.pct_change().dropna()
    expected_sharpe = (expected_returns.mean() / expected_returns.std()) * np.sqrt(periods)

    result = risk_metrics.calculate_spread_sharpe(spread, periods)

    assert result is not None
    assert np.isclose(result, expected_sharpe)


def test_calculate_spread_sharpe_handles_zero_std():
    spread = pd.Series([100] * 10, dtype=float)
    assert risk_metrics.calculate_spread_sharpe(spread, 4) is None


def test_calculate_max_drawdown_matches_manual():
    spread = pd.Series([1, -2, -1, -3, 4, -2], dtype=float)
    cumulative = spread.cumsum()
    running_max = cumulative.cummax().replace(0, np.nan)
    drawdown = (cumulative - running_max) / running_max
    expected = float(drawdown.min())

    result = risk_metrics.calculate_max_drawdown(spread)
    assert result == expected


def test_calculate_calmar_ratio_uses_annual_return_and_drawdown():
    spread = pd.Series([1.0, -2.0, 3.0, -1.5, 2.5, -0.5, 4.0], dtype=float)
    periods = 12

    returns = spread.pct_change().dropna()
    annual_return = returns.mean() * periods
    cumulative = spread.cumsum()
    running_max = cumulative.cummax().replace(0, np.nan)
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(drawdown.min())
    expected = annual_return / max_dd

    result = risk_metrics.calculate_calmar_ratio(spread, periods)

    assert result is not None
    assert np.isclose(result, expected)

