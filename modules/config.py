"""
Configuration constants for the crypto prediction system.
"""

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

# Model features list
MODEL_FEATURES = [
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

