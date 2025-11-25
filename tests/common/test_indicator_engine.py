import pandas as pd

from modules.IndicatorEngine import (
    IndicatorConfig,
    IndicatorEngine,
    IndicatorProfile,
)


def _sample_df():
    data = {
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [101.0, 102.0, 103.0, 104.0, 105.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0],
        "close": [100.5, 101.5, 102.5, 103.5, 104.5],
        "volume": [1000, 1100, 1200, 1300, 1400],
    }
    return pd.DataFrame(data)


def test_core_profile_generates_trend_and_momentum_metadata():
    df = _sample_df()
    engine = IndicatorEngine(IndicatorConfig.for_profile(IndicatorProfile.CORE))

    features, metadata = engine.compute_features(df.copy(), return_metadata=True)

    assert {"SMA_20", "RSI_9"}.issubset(features.columns)
    assert metadata["SMA_20"] == "trend"
    assert metadata["RSI_9"] == "momentum"
    assert all(not name.startswith("candlestick") for name in metadata.values())


def test_xgboost_profile_includes_candlestick_features():
    df = _sample_df()
    engine = IndicatorEngine(IndicatorConfig.for_profile(IndicatorProfile.XGBOOST))

    features = engine.compute_features(df.copy())

    assert "DOJI" in features.columns
    assert "ATR_RATIO_14_50" in features.columns


def test_register_indicator_adds_custom_columns():
    df = _sample_df()
    engine = IndicatorEngine()

    def add_mid_price(dataframe: pd.DataFrame):
        dataframe["MID"] = (dataframe["high"] + dataframe["low"]) / 2
        return dataframe

    engine.register_indicator("mid", add_mid_price)
    features = engine.compute_features(df.copy())

    assert "MID" in features.columns
    assert all(features["MID"] == (df["high"] + df["low"]) / 2)
