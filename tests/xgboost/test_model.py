import numpy as np
import pandas as pd

from modules.xgboost_prediction_model import predict_next_move, train_and_predict
from modules.config import TARGET_LABELS


def _synthetic_df(rows=300):
    rng = np.random.default_rng(42)
    data = {
        "open": rng.normal(100, 1, rows).cumsum(),
        "high": rng.normal(101, 1, rows).cumsum(),
        "low": rng.normal(99, 1, rows).cumsum(),
        "close": rng.normal(100.5, 1, rows).cumsum(),
        "volume": rng.integers(1000, 2000, rows),
        "SMA_20": rng.normal(100, 1, rows),
        "SMA_50": rng.normal(100, 1, rows),
        "SMA_200": rng.normal(100, 1, rows),
        "RSI_9": rng.uniform(0, 100, rows),
        "RSI_14": rng.uniform(0, 100, rows),
        "RSI_25": rng.uniform(0, 100, rows),
        "ATR_14": rng.uniform(0.5, 2, rows),
        "MACD_12_26_9": rng.normal(0, 1, rows),
        "MACDh_12_26_9": rng.normal(0, 1, rows),
        "MACDs_12_26_9": rng.normal(0, 1, rows),
        "BBP_5_2.0": rng.uniform(0, 1, rows),
        "STOCHRSIk_14_14_3_3": rng.uniform(0, 100, rows),
        "STOCHRSId_14_14_3_3": rng.uniform(0, 100, rows),
        "OBV": rng.normal(0, 1, rows).cumsum(),
        "DOJI": rng.integers(0, 2, rows),
        "HAMMER": rng.integers(0, 2, rows),
        "INVERTED_HAMMER": rng.integers(0, 2, rows),
        "SHOOTING_STAR": rng.integers(0, 2, rows),
        "BULLISH_ENGULFING": rng.integers(0, 2, rows),
        "BEARISH_ENGULFING": rng.integers(0, 2, rows),
        "MORNING_STAR": rng.integers(0, 2, rows),
        "EVENING_STAR": rng.integers(0, 2, rows),
        "PIERCING": rng.integers(0, 2, rows),
        "DARK_CLOUD": rng.integers(0, 2, rows),
        "ATR_RATIO_14_50": rng.uniform(0.5, 1.5, rows),
        "DynamicThreshold": rng.uniform(0.01, 0.05, rows),
    }
    df = pd.DataFrame(data)
    df["Target"] = rng.integers(0, len(TARGET_LABELS), rows)
    return df


def test_train_and_predict_returns_model():
    df = _synthetic_df()
    model = train_and_predict(df)

    assert hasattr(model, "predict")


def test_predict_next_move_returns_valid_probabilities():
    df = _synthetic_df()
    model = train_and_predict(df)
    last_row = df.iloc[-1:]
    proba = predict_next_move(model, last_row)

    assert len(proba) == len(TARGET_LABELS)
    assert np.isclose(proba.sum(), 1.0, atol=1e-3)


def test_train_and_predict_handles_small_dataset(monkeypatch):
    df = _synthetic_df(rows=60)

    # Force TARGET_HORIZON to be larger to trigger warning branch
    monkeypatch.setattr("modules.xgboost_prediction_model.TARGET_HORIZON", 50)
    model = train_and_predict(df)

    assert hasattr(model, "predict")
