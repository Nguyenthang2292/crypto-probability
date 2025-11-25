import numpy as np
import pandas as pd

from modules.pairs_trading import zscore_metrics


def test_calculate_zscore_stats_matches_manual_computation():
    spread = pd.Series(np.linspace(1.0, 3.0, 200))
    lookback = 50

    result = zscore_metrics.calculate_zscore_stats(spread, lookback)

    rolling_mean = spread.rolling(lookback).mean()
    rolling_std = spread.rolling(lookback).std().replace(0, np.nan)
    zscore = ((spread - rolling_mean) / rolling_std).dropna()

    assert result["mean_zscore"] == float(zscore.mean())
    assert result["std_zscore"] == float(zscore.std())
    assert result["current_zscore"] == float(zscore.iloc[-1])


def test_calculate_hurst_exponent_returns_value_for_long_series():
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(size=500).cumsum())

    hurst = zscore_metrics.calculate_hurst_exponent(series, zscore_lookback=50, max_lag=20)
    assert hurst is not None
    assert 0 <= hurst <= 2


def test_calculate_direction_metrics_produces_classification_scores():
    # Construct synthetic spread with mean-reverting behaviour
    base = np.sin(np.linspace(0, 20, 400))
    noise = np.linspace(0, 0.5, 400)
    spread = pd.Series(base + noise)

    metrics = zscore_metrics.calculate_direction_metrics(
        spread, zscore_lookback=40, classification_zscore=0.5
    )

    assert metrics["classification_f1"] is not None
    assert metrics["classification_accuracy"] is not None

