import argparse
import re
import warnings

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
from colorama import Fore, Style, init as colorama_init
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import TimeSeriesSplit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
colorama_init(autoreset=True)

DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_QUOTE = "USDT"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_LIMIT = 1500
DEFAULT_EXCHANGES = [
    "binance",
    "kraken",
    "kucoin",
    "gate",
    "okx",
    "bybit",
    "mexc",
    "huobi",
]
DEFAULT_EXCHANGE_STRING = ",".join(DEFAULT_EXCHANGES)
PREDICTION_WINDOWS = {
    "30m": "12h",
    "45m": "18h",
    "1h": "24h",
    "2h": "36h",
    "4h": "48h",
    "6h": "72h",
    "12h": "7d",
    "1d": "7d",
}
TARGET_HORIZON = 24
TARGET_BASE_THRESHOLD = 0.01
TARGET_LABELS = ["DOWN", "NEUTRAL", "UP"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(TARGET_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def print_classification_report(y_true, y_pred, title="Classification Report"):
    """
    Prints a formatted classification report with color coding.
    """
    print("\n" + color_text("=" * 60, Fore.CYAN, Style.BRIGHT))
    print(color_text(title, Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 60, Fore.CYAN, Style.BRIGHT))
    
    # Get classification report as string
    report = classification_report(
        y_true,
        y_pred,
        target_names=TARGET_LABELS,
        output_dict=False,
    )
    print(report)
    
    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(color_text("\nConfusion Matrix:", Fore.MAGENTA, Style.BRIGHT))
    print(color_text("(Rows = True, Columns = Predicted)", Fore.WHITE))
    print(" " * 12, end="")
    for label in TARGET_LABELS:
        print(f"{label:>12}", end="")
    print()
    for i, label in enumerate(TARGET_LABELS):
        print(f"{label:>12}", end="")
        for j in range(len(TARGET_LABELS)):
            value = cm[i, j]
            # Color code: green for correct predictions (diagonal), red for major errors
            if i == j:
                color = Fore.GREEN
            elif abs(i - j) == 2:  # UP vs DOWN or vice versa
                color = Fore.RED
            else:
                color = Fore.YELLOW
            print(color_text(f"{value:>12}", color), end="")
        print()
    
    print(color_text("=" * 60, Fore.CYAN, Style.BRIGHT) + "\n")


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Converts a timeframe string like '30m', '1h', '1d' into minutes.
    """
    match = re.match(r"^\s*(\d+)\s*([mhdw])\s*$", timeframe.lower())
    if not match:
        return 60  # default 1h

    value, unit = match.groups()
    value = int(value)

    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 60 * 24
    if unit == "w":
        return value * 60 * 24 * 7
    return 60


def get_prediction_window(timeframe: str) -> str:
    """
    Returns a textual description of the prediction horizon based on timeframe.
    """
    timeframe = timeframe.lower()
    return PREDICTION_WINDOWS.get(timeframe, "next sessions")


def color_text(text: str, color: str = Fore.WHITE, style: str = Style.NORMAL) -> str:
    return f"{style}{color}{text}{Style.RESET_ALL}"


def format_price(value: float) -> str:
    """
    Formats prices/indicators with adaptive precision so tiny values remain readable.
    """
    if value is None or pd.isna(value):
        return "N/A"

    abs_val = abs(value)
    if abs_val >= 1:
        precision = 2
    elif abs_val >= 0.01:
        precision = 4
    elif abs_val >= 0.0001:
        precision = 6
    else:
        precision = 8

    return f"{value:.{precision}f}"


def apply_directional_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels each row as UP/DOWN/NEUTRAL based on future price movement.
    """
    future_close = df["close"].shift(-TARGET_HORIZON)
    pct_change = (future_close - df["close"]) / df["close"]

    historical_ref = df["close"].shift(TARGET_HORIZON)
    historical_pct = (df["close"] - historical_ref) / historical_ref
    base_threshold = (
        historical_pct.abs()
        .fillna(TARGET_BASE_THRESHOLD)
        .clip(lower=TARGET_BASE_THRESHOLD)
    )
    atr_ratio = (
        df.get("ATR_RATIO_14_50", pd.Series(1.0, index=df.index))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(lower=0.5, upper=2.0)
    )
    threshold_series = (base_threshold * atr_ratio).clip(lower=TARGET_BASE_THRESHOLD)
    df["DynamicThreshold"] = threshold_series

    df["TargetLabel"] = np.where(
        pct_change >= threshold_series,
        "UP",
        np.where(pct_change <= -threshold_series, "DOWN", "NEUTRAL"),
    )
    df.loc[future_close.isna(), "TargetLabel"] = np.nan
    df["Target"] = df["TargetLabel"].map(LABEL_TO_ID)
    return df


def normalize_symbol(user_input: str, quote: str = DEFAULT_QUOTE) -> str:
    """
    Converts user input like 'xmr' into 'XMR/USDT'. Keeps existing slash pairs.
    """
    if not user_input:
        return f"BTC/{quote}"

    norm = user_input.strip().upper()
    if "/" in norm:
        return norm

    if norm.endswith(quote):
        return f"{norm[:-len(quote)]}/{quote}"

    return f"{norm}/{quote}"


def prompt_with_default(message: str, default, cast=str):
    while True:
        raw = input(color_text(f"{message} (default {default}): ", Fore.CYAN))
        value = raw.strip()
        if not value:
            return default
        try:
            return cast(value)
        except ValueError:
            print(color_text("Invalid input. Please try again.", Fore.RED))


def resolve_input(cli_value, default, prompt_message, cast=str, allow_prompt=True):
    if cli_value is not None:
        return cast(cli_value)
    if allow_prompt:
        return prompt_with_default(prompt_message, default, cast)
    return default


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crypto movement predictor using technical indicators and XGBoost."
    )
    parser.add_argument(
        "-s",
        "--symbol",
        help=f"Trading pair symbol (default: {DEFAULT_SYMBOL}). Accepts formats like 'BTC/USDT' or 'btc'.",
    )
    parser.add_argument(
        "-q",
        "--quote",
        help=f"Quote currency when symbol is given without slash (default: {DEFAULT_QUOTE}).",
    )
    parser.add_argument(
        "-t",
        "--timeframe",
        help=f"Timeframe for OHLCV data (default: {DEFAULT_TIMEFRAME}, e.g., 30m, 1h, 4h).",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        help=f"Number of candles to fetch (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "-e",
        "--exchanges",
        help=f"Comma-separated list of exchanges to try (default: {DEFAULT_EXCHANGE_STRING}).",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disable interactive prompts; rely only on CLI arguments.",
    )
    return parser.parse_args()


def fetch_data(symbol="BTC/USDT", timeframe="1h", limit=1000, exchanges=None):
    """
    Fetches OHLCV data trying multiple exchanges until fresh data is returned.
    """
    exchanges = exchanges or DEFAULT_EXCHANGES
    freshness_minutes = max(timeframe_to_minutes(timeframe) * 1.5, 5)
    fallback = None

    print(
        color_text(
            f"Fetching {limit} candles for {symbol} ({timeframe})...",
            Fore.CYAN,
            Style.BRIGHT,
        )
    )
    for exchange_id in exchanges:
        exchange_cls = getattr(ccxt, exchange_id)
        exchange = exchange_cls()
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            if df.empty:
                print(
                    color_text(
                        f"[{exchange_id.upper()}] No data retrieved.", Fore.YELLOW
                    )
                )
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            last_ts = df["timestamp"].iloc[-1]
            now = pd.Timestamp.now(tz="UTC")
            age_minutes = (now - last_ts).total_seconds() / 60.0

            if age_minutes <= freshness_minutes:
                print(
                    color_text(
                        f"[{exchange_id.upper()}] Data age {age_minutes:.1f}m (fresh).",
                        Fore.GREEN,
                    )
                )
                return df, exchange_id

            print(
                color_text(
                    f"[{exchange_id.upper()}] Data age {age_minutes:.1f}m (stale). Trying next exchange...",
                    Fore.YELLOW,
                )
            )
            fallback = (df, exchange_id)
        except Exception as e:
            print(
                color_text(
                    f"[{exchange_id.upper()}] Error fetching data: {e}", Fore.RED
                )
            )
            continue

    if fallback:
        df, exchange_id = fallback
        print(
            color_text(
                f"Using latest available data from {exchange_id.upper()} despite staleness.",
                Fore.MAGENTA,
            )
        )
        return df, exchange_id

    print(
        color_text("Failed to fetch data from all exchanges.", Fore.RED, Style.BRIGHT)
    )
    return None, None


def add_candlestick_patterns(df):
    """Detect 10 reliable candlestick patterns and add binary columns."""
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    body = abs(c - o)
    upper_shadow = h - np.maximum(c, o)
    lower_shadow = np.minimum(c, o) - l
    range_hl = h - l
    range_hl = np.where(range_hl == 0, 0.0001, range_hl)
    range_hl = pd.Series(range_hl, index=df.index)
    df["DOJI"] = (body / range_hl < 0.1).astype(int)
    df["HAMMER"] = (
        (lower_shadow > 2 * body)
        & (upper_shadow < 0.3 * body)
        & (body / range_hl < 0.3)
    ).astype(int)
    df["INVERTED_HAMMER"] = (
        (upper_shadow > 2 * body)
        & (lower_shadow < 0.3 * body)
        & (body / range_hl < 0.3)
    ).astype(int)
    df["SHOOTING_STAR"] = (
        (upper_shadow > 2 * body) & (lower_shadow < 0.3 * body) & (c < o)
    ).astype(int)
    prev_bearish = o.shift(1) > c.shift(1)
    curr_bullish = c > o
    df["BULLISH_ENGULFING"] = (
        prev_bearish
        & curr_bullish
        & (c > o.shift(1))
        & (o < c.shift(1))
    ).astype(int)
    prev_bullish = c.shift(1) > o.shift(1)
    curr_bearish = o > c
    df["BEARISH_ENGULFING"] = (
        prev_bullish
        & curr_bearish
        & (o > c.shift(1))
        & (c < o.shift(1))
    ).astype(int)
    first_bearish = o.shift(2) > c.shift(2)
    second_small = body.shift(1) / range_hl.shift(1) < 0.3
    third_bullish = c > o
    df["MORNING_STAR"] = (
        first_bearish
        & second_small
        & third_bullish
        & (c > (o.shift(2) + c.shift(2)) / 2)
    ).astype(int)
    first_bullish_es = c.shift(2) > o.shift(2)
    second_small_es = body.shift(1) / range_hl.shift(1) < 0.3
    third_bearish_es = o > c
    df["EVENING_STAR"] = (
        first_bullish_es
        & second_small_es
        & third_bearish_es
        & (c < (o.shift(2) + c.shift(2)) / 2)
    ).astype(int)
    df["PIERCING"] = (
        (o.shift(1) > c.shift(1))
        & (c > o)
        & (o < c.shift(1))
        & (c > (o.shift(1) + c.shift(1)) / 2)
        & (c < o.shift(1))
    ).astype(int)
    df["DARK_CLOUD"] = (
        (c.shift(1) > o.shift(1))
        & (o > c)
        & (o > c.shift(1))
        & (c < (o.shift(1) + c.shift(1)) / 2)
        & (c > o.shift(1))
    ).astype(int)
    return df


def calculate_indicators(df, apply_labels=True):
    """
    Calculates technical indicators and candlestick patterns.
    Does NOT drop NaN values - caller should handle that.
    
    Args:
        df: DataFrame with OHLCV data
        apply_labels: If True, applies directional labels. If False, only calculates indicators.
    
    Returns:
        DataFrame with indicators added
    """
    # Trend - Moving Averages
    df["SMA_20"] = ta.sma(df["close"], length=20)
    df["SMA_50"] = ta.sma(df["close"], length=50)
    df["SMA_200"] = ta.sma(df["close"], length=200)

    # Momentum - RSI (neutral value is 50)
    rsi_9 = ta.rsi(df["close"], length=9)
    df["RSI_9"] = rsi_9.fillna(50.0) if rsi_9 is not None else pd.Series(50.0, index=df.index)
    
    rsi_14 = ta.rsi(df["close"], length=14)
    df["RSI_14"] = rsi_14.fillna(50.0) if rsi_14 is not None else pd.Series(50.0, index=df.index)
    
    rsi_25 = ta.rsi(df["close"], length=25)
    df["RSI_25"] = rsi_25.fillna(50.0) if rsi_25 is not None else pd.Series(50.0, index=df.index)

    # MACD (neutral values: MACD=0, MACDh=0, MACDs=0)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df = pd.concat([df, macd], axis=1)
        # Fill NaN with neutral values
        for col in ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
    else:
        print(
            color_text(
                "MACD calculation failed, using neutral values (0).", Fore.YELLOW
            )
        )
        df["MACD_12_26_9"] = 0.0
        df["MACDh_12_26_9"] = 0.0
        df["MACDs_12_26_9"] = 0.0

    # Bollinger Bands (neutral value for BBP is 0.5 - middle of band)
    bbands = ta.bbands(df["close"], length=20, std=2.0)
    if bbands is not None and not bbands.empty:
        bbp_cols = [c for c in bbands.columns if c.startswith("BBP")]
        if bbp_cols:
            df["BBP_5_2.0"] = bbands[bbp_cols[0]].fillna(0.5)
        else:
            print(
                color_text(
                    "BBP column not found in Bollinger Bands output, using neutral value (0.5).",
                    Fore.YELLOW,
                )
            )
            df["BBP_5_2.0"] = 0.5
    else:
        print(
            color_text(
                "Bollinger Bands calculation failed, using neutral value (0.5).", Fore.YELLOW
            )
        )
        df["BBP_5_2.0"] = 0.5

    # Stochastic RSI (neutral values: k=50, d=50)
    stochrsi = ta.stochrsi(df["close"], length=14, rsi_length=14, k=3, d=3)
    if stochrsi is not None and not stochrsi.empty:
        df = pd.concat([df, stochrsi], axis=1)
        # Fill NaN with neutral values
        if "STOCHRSIk_14_14_3_3" in df.columns:
            df["STOCHRSIk_14_14_3_3"] = df["STOCHRSIk_14_14_3_3"].fillna(50.0)
        if "STOCHRSId_14_14_3_3" in df.columns:
            df["STOCHRSId_14_14_3_3"] = df["STOCHRSId_14_14_3_3"].fillna(50.0)
    else:
        print(
            color_text(
                "Stochastic RSI calculation failed, using neutral values (50).", Fore.YELLOW
            )
        )
        df["STOCHRSIk_14_14_3_3"] = 50.0
        df["STOCHRSId_14_14_3_3"] = 50.0

    # On-Balance Volume (use forward fill for NaN)
    obv = ta.obv(df["close"], df["volume"])
    df["OBV"] = obv.ffill().fillna(0.0) if obv is not None else pd.Series(0.0, index=df.index)

    # Volatility - ATR (use forward fill for NaN)
    atr_14 = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ATR_14"] = atr_14.ffill().fillna(df["close"] * 0.01) if atr_14 is not None else pd.Series(df["close"] * 0.01, index=df.index)
    
    atr_50 = ta.atr(df["high"], df["low"], df["close"], length=50)
    df["ATR_50"] = atr_50.ffill().fillna(df["close"] * 0.01) if atr_50 is not None else pd.Series(df["close"] * 0.01, index=df.index)
    
    df["ATR_RATIO_14_50"] = df["ATR_14"] / df["ATR_50"]
    df["ATR_RATIO_14_50"] = df["ATR_RATIO_14_50"].fillna(1.0)  # Neutral ratio is 1.0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df["ATR_RATIO_14_50"] = df["ATR_RATIO_14_50"].fillna(1.0)

    # Candlestick patterns
    df = add_candlestick_patterns(df)

    # Apply directional labels if requested
    if apply_labels:
        df = apply_directional_labels(df)

    return df


def add_indicators(df):
    """
    Adds technical indicators and candlestick patterns using pandas_ta.
    Also applies directional labels and drops NaN values.
    """
    # Calculate indicators with labels
    df = calculate_indicators(df, apply_labels=True)

    # Drop NaN values created by indicators
    initial_len = len(df)
    df.dropna(inplace=True)
    dropped = initial_len - len(df)
    
    # Warn about data loss due to SMA 200 (requires 200 periods)
    if dropped >= 200:
        print(
            color_text(
                f"WARNING: Dropped {dropped} rows (>=200 due to SMA_200 requirement). "
                f"Consider increasing --limit to at least {initial_len + 200} to maintain sufficient training data.",
                Fore.RED,
                Style.BRIGHT,
            )
        )
    elif dropped > 0:
        print(
            color_text(
                f"Dropped {dropped} rows due to NaN values from indicators.",
                Fore.BLUE,
            )
        )
    
    if len(df) < 200:
        print(
            color_text(
                f"WARNING: Only {len(df)} rows remaining after indicator calculation. "
                f"Consider increasing data limit (--limit) to at least {len(df) + 200} for better model performance.",
                Fore.YELLOW,
                Style.BRIGHT,
            )
        )
    
    return df


def train_and_predict(df):
    """
    Trains XGBoost model and predicts the next movement.
    """
    # Features to use for prediction
    features = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "SMA_20",
        "SMA_50",
        "SMA_200",
        "RSI_9",
        "RSI_14",
        "RSI_25",
        "ATR_14",
        "MACD_12_26_9",
        "MACDh_12_26_9",
        "MACDs_12_26_9",
        "BBP_5_2.0",
        "STOCHRSIk_14_14_3_3",
        "STOCHRSId_14_14_3_3",
        "OBV",
        # Candlestick patterns
        "DOJI",
        "HAMMER",
        "INVERTED_HAMMER",
        "SHOOTING_STAR",
        "BULLISH_ENGULFING",
        "BEARISH_ENGULFING",
        "MORNING_STAR",
        "EVENING_STAR",
        "PIERCING",
        "DARK_CLOUD",
    ]

    # Split data: Train on all except the last row (which has no target yet for validation,
    # but in this live scenario we train on everything available up to the second to last candle
    # to predict the movement for the very last known candle, OR we train on history to predict the FUTURE).

    # To predict the FUTURE (next candle after the latest one):
    # We need to train on data where we KNOW the outcome.
    # So we use the dataset where 'Target' is valid.
    # The last row of df currently has Target based on a future candle that doesn't exist yet (shift(-1)).
    # So the last row's Target is False/NaN (pandas shift fills with NaN, but we did dropna).
    # Wait, if we shift(-1), the last row gets NaN. dropna() removes it.
    # So 'df' now contains only historical data where we know the outcome.

    X = df[features]
    y = df["Target"].astype(int)

    def build_model():
        return xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=0.1,
            min_child_weight=3,
            random_state=42,
            objective="multi:softprob",
            num_class=len(TARGET_LABELS),
            eval_metric="mlogloss",
            n_jobs=-1,
        )

    # Train/Test split (80/20) for evaluation metrics
    # IMPORTANT: Create a gap of TARGET_HORIZON between train and test to prevent data leakage
    # The last TARGET_HORIZON rows of train set use future prices from test set to create labels
    split = int(len(df) * 0.8)
    train_end = split - TARGET_HORIZON
    test_start = split
    
    # Ensure we have enough data after creating the gap
    if train_end < len(df) * 0.5:
        train_end = int(len(df) * 0.5)
        test_start = train_end + TARGET_HORIZON
        if test_start >= len(df):
            # Not enough data for proper train/test split with gap
            print(
                color_text(
                    f"WARNING: Insufficient data for train/test split with gap. "
                    f"Need at least {len(df) + TARGET_HORIZON} rows. Using all data for training.",
                    Fore.YELLOW,
                )
            )
            train_end = len(df)
            test_start = len(df)
    
    X_train, X_test = X.iloc[:train_end], X.iloc[test_start:]
    y_train, y_test = y.iloc[:train_end], y.iloc[test_start:]
    
    gap_size = test_start - train_end
    if gap_size > 0:
        print(
            color_text(
                f"Train/Test split: {len(X_train)} train, {gap_size} gap (to prevent leakage), {len(X_test)} test",
                Fore.CYAN,
            )
        )

    model = build_model()
    model.fit(X_train, y_train)

    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        score = model.score(X_test, y_test)
        print(color_text(f"\nHoldout Accuracy: {score:.4f}", Fore.YELLOW, Style.BRIGHT))
        print_classification_report(y_test, y_pred, "Holdout Test Set Evaluation")
    else:
        print(
            color_text(
                "Skipping holdout evaluation (insufficient test data after gap).",
                Fore.YELLOW,
            )
        )

    # Time-series cross validation with gap to prevent data leakage
    max_splits = min(5, len(df) - 1)
    if max_splits >= 2:
        tscv = TimeSeriesSplit(n_splits=max_splits)
        cv_scores = []
        all_y_true = []
        all_y_pred = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            # Apply gap to prevent data leakage: remove last TARGET_HORIZON indices from train
            train_idx_array = np.array(train_idx)
            if len(train_idx_array) > TARGET_HORIZON:
                # Remove the last TARGET_HORIZON indices from train to create gap
                train_idx_filtered = train_idx_array[:-TARGET_HORIZON]
            else:
                # Not enough data for gap, skip this fold
                print(
                    color_text(
                        f"CV Fold {fold}: Skipped (insufficient train data for gap)",
                        Fore.YELLOW,
                    )
                )
                continue
            
            # Ensure test set doesn't overlap with gap
            # Gap is sufficient when: test_start > train_end + TARGET_HORIZON
            test_idx_array = np.array(test_idx)
            if len(train_idx_filtered) > 0 and len(test_idx_array) > 0:
                min_test_start = train_idx_filtered[-1] + TARGET_HORIZON + 1
                if test_idx_array[0] < min_test_start:
                    # Adjust test start to create proper gap
                    test_idx_array = test_idx_array[test_idx_array >= min_test_start]
                    if len(test_idx_array) == 0:
                        print(
                            color_text(
                                f"CV Fold {fold}: Skipped (no valid test data after gap)",
                                Fore.YELLOW,
                            )
                        )
                        continue
            
            # Check if filtered training data contains all required classes
            y_train_fold = y.iloc[train_idx_filtered]
            unique_classes = sorted(y_train_fold.unique())
            
            # XGBoost requires at least 2 classes, but we need all 3 for proper multi-class
            # If we don't have all classes, skip this fold
            if len(unique_classes) < 2:
                print(
                    color_text(
                        f"CV Fold {fold}: Skipped (insufficient class diversity: {unique_classes})",
                        Fore.YELLOW,
                    )
                )
                continue
            
            # If we have all 3 classes, proceed normally
            # If we only have 2 classes, we can still train but need to handle it
            # For now, we'll skip folds that don't have all 3 classes to maintain consistency
            if len(unique_classes) < len(TARGET_LABELS):
                print(
                    color_text(
                        f"CV Fold {fold}: Skipped (missing classes: expected {TARGET_LABELS}, got {[ID_TO_LABEL[c] for c in unique_classes]})",
                        Fore.YELLOW,
                    )
                )
                continue
            
            cv_model = build_model()
            cv_model.fit(X.iloc[train_idx_filtered], y.iloc[train_idx_filtered])
            if len(test_idx_array) > 0:
                y_test_fold = y.iloc[test_idx_array]
                preds = cv_model.predict(X.iloc[test_idx_array])
                acc = accuracy_score(y_test_fold, preds)
                cv_scores.append(acc)
                
                # Collect predictions for aggregated report
                all_y_true.extend(y_test_fold.tolist())
                all_y_pred.extend(preds.tolist())
                
                print(
                    color_text(
                        f"CV Fold {fold} Accuracy: {acc:.4f} (train: {len(train_idx_filtered)}, gap: {TARGET_HORIZON}, test: {len(test_idx_array)})",
                        Fore.BLUE,
                    )
                )
        
        if len(cv_scores) > 0:
            mean_cv = sum(cv_scores) / len(cv_scores)
            print(
                color_text(
                    f"\nCV Mean Accuracy ({len(cv_scores)} folds): {mean_cv:.4f}",
                    Fore.GREEN,
                    Style.BRIGHT,
                )
            )
            
            # Print aggregated classification report across all CV folds
            if len(all_y_true) > 0 and len(all_y_pred) > 0:
                print_classification_report(
                    np.array(all_y_true),
                    np.array(all_y_pred),
                    "Cross-Validation Aggregated Report (All Folds)",
                )
        else:
            print(
                color_text(
                    "CV: No valid folds after applying gap. Consider increasing data limit.",
                    Fore.YELLOW,
                )
            )
    else:
        print(
            color_text(
                "Not enough data for cross-validation (requires >=3 samples).",
                Fore.YELLOW,
            )
        )

    model.fit(X, y)
    return model


def predict_next_move(model, last_row):
    """
    Predicts the probability for the next candle.
    """
    # Prepare the single row of features
    features = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "SMA_20",
        "SMA_50",
        "SMA_200",
        "RSI_9",
        "RSI_14",
        "RSI_25",
        "ATR_14",
        "MACD_12_26_9",
        "MACDh_12_26_9",
        "MACDs_12_26_9",
        "BBP_5_2.0",
        "STOCHRSIk_14_14_3_3",
        "STOCHRSId_14_14_3_3",
        "OBV",
        # Candlestick patterns
        "DOJI",
        "HAMMER",
        "INVERTED_HAMMER",
        "SHOOTING_STAR",
        "BULLISH_ENGULFING",
        "BEARISH_ENGULFING",
        "MORNING_STAR",
        "EVENING_STAR",
        "PIERCING",
        "DARK_CLOUD",
    ]
    X_new = last_row[features].values.reshape(1, -1)

    # Predict probability
    # proba[0] is prob of class 0 (Down), proba[1] is prob of class 1 (Up)
    proba = model.predict_proba(X_new)[0]

    return proba


def main():
    args = parse_args()
    allow_prompt = not args.no_prompt

    quote = args.quote.upper() if args.quote else DEFAULT_QUOTE
    timeframe = resolve_input(
        args.timeframe, DEFAULT_TIMEFRAME, "Enter timeframe", str, allow_prompt
    ).lower()
    limit = args.limit if args.limit is not None else DEFAULT_LIMIT
    exchanges_input = args.exchanges if args.exchanges else DEFAULT_EXCHANGE_STRING
    exchanges = [
        ex.strip() for ex in exchanges_input.split(",") if ex.strip()
    ] or DEFAULT_EXCHANGES

    def run_once(raw_symbol):
        symbol = normalize_symbol(raw_symbol, quote)
        df, exchange_id = fetch_data(
            symbol, timeframe, limit=limit, exchanges=exchanges
        )
        exchange_label = exchange_id.upper() if exchange_id else "UNKNOWN"

        if df is not None:
            # Calculate indicators without labels first (to preserve latest_data)
            df = calculate_indicators(df, apply_labels=False)
            
            # Save latest data before applying labels and dropping NaN
            latest_data = df.iloc[-1:].copy()
            # Fill any remaining NaN in latest_data with forward fill then backward fill
            latest_data = latest_data.ffill().bfill()

            # Apply directional labels and drop NaN for training data
            df = apply_directional_labels(df)
            latest_threshold = df["DynamicThreshold"].iloc[-1] if len(df) > 0 else TARGET_BASE_THRESHOLD
            df.dropna(inplace=True)
            latest_data["DynamicThreshold"] = latest_threshold

            print(color_text(f"Training on {len(df)} samples...", Fore.CYAN))
            model = train_and_predict(df)

            proba = predict_next_move(model, latest_data)
            proba_percent = {
                label: proba[LABEL_TO_ID[label]] * 100 for label in TARGET_LABELS
            }
            best_idx = int(np.argmax(proba))
            direction = ID_TO_LABEL[best_idx]
            probability = proba_percent[direction]

            current_price = latest_data["close"].values[0]
            atr = latest_data["ATR_14"].values[0]
            prediction_window = get_prediction_window(timeframe)
            threshold_value = latest_data["DynamicThreshold"].iloc[0]
            prediction_context = f"{prediction_window} | {TARGET_HORIZON} candles >={threshold_value*100:.2f}% move"

            print("\n" + color_text("=" * 40, Fore.BLUE, Style.BRIGHT))
            print(
                color_text(
                    f"ANALYSIS FOR {symbol} | TF {timeframe} | {exchange_label}",
                    Fore.CYAN,
                    Style.BRIGHT,
                )
            )
            print(
                color_text(f"Current Price: {format_price(current_price)}", Fore.WHITE)
            )
            print(
                color_text(f"Market Volatility (ATR): {format_price(atr)}", Fore.WHITE)
            )
            print(color_text("-" * 40, Fore.BLUE))

            if direction == "UP":
                direction_color = Fore.GREEN
                atr_sign = 1
            elif direction == "DOWN":
                direction_color = Fore.RED
                atr_sign = -1
            else:
                direction_color = Fore.YELLOW
                atr_sign = 0

            print(
                color_text(
                    f"PREDICTION ({prediction_context}): {direction}",
                    direction_color,
                    Style.BRIGHT,
                )
            )
            print(color_text(f"Confidence: {probability:.2f}%", direction_color))

            prob_summary = " | ".join(
                f"{label}: {value:.2f}%" for label, value in proba_percent.items()
            )
            print(color_text(f"Probabilities -> {prob_summary}", Fore.WHITE))

            if direction == "NEUTRAL":
                print(
                    color_text(
                        "Market expected to stay within +/-{:.2f}% over the next {} candles.".format(
                            threshold_value * 100, TARGET_HORIZON
                        ),
                        Fore.YELLOW,
                    )
                )
            else:
                print(
                    color_text(
                        "Estimated Targets via ATR multiples:",
                        Fore.MAGENTA,
                        Style.BRIGHT,
                    )
                )
                for multiple in (1, 2, 3):
                    target_price = current_price + atr_sign * multiple * atr
                    move_abs = abs(target_price - current_price)
                    move_pct = (
                        (move_abs / current_price) * 100 if current_price else None
                    )
                    move_pct_text = (
                        f"{move_pct:.2f}%" if move_pct is not None else "N/A"
                    )
                    print(
                        color_text(
                            f"  ATR x{multiple}: {format_price(target_price)} | Delta {format_price(move_abs)} ({move_pct_text})",
                            Fore.MAGENTA,
                        )
                    )
            print(color_text("=" * 40, Fore.BLUE, Style.BRIGHT))
        else:
            print(
                color_text(
                    "Unable to proceed without market data. Please try again later.",
                    Fore.RED,
                    Style.BRIGHT,
                )
            )

    try:
        while True:
            raw_symbol = resolve_input(
                args.symbol, DEFAULT_SYMBOL, "Enter symbol pair", str, allow_prompt
            )
            run_once(raw_symbol)
            args.symbol = None  # force prompt next iteration
            if not allow_prompt:
                break
            print(
                color_text(
                    "\nPress Ctrl+C to exit. Provide a new symbol to continue.",
                    Fore.YELLOW,
                )
            )
    except KeyboardInterrupt:
        print(color_text("\nExiting program by user request.", Fore.YELLOW))


if __name__ == "__main__":
    main()
