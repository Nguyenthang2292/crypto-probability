"""
Configuration constants for xgboost_prediction_main.py and portfolio_manager_v2.py
"""

DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_QUOTE = "USDT"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_LIMIT = 1500

# Portfolio Manager Configuration
BENCHMARK_SYMBOL = "BTC/USDT"  # Default benchmark for beta calculation
DEFAULT_REQUEST_PAUSE = 0.2  # Default pause between API requests (seconds)
DEFAULT_CONTRACT_TYPE = 'future'  # Default contract type: 'spot', 'margin', or 'future'
DEFAULT_BETA_MIN_POINTS = 50  # Minimum data points required for beta calculation
DEFAULT_BETA_LIMIT = 1000  # Default limit for beta calculation OHLCV fetch
DEFAULT_BETA_TIMEFRAME = "1h"  # Default timeframe for beta calculation
DEFAULT_CORRELATION_MIN_POINTS = 10  # Minimum data points for correlation analysis
DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS = 10  # Minimum data points for weighted correlation
DEFAULT_VAR_CONFIDENCE = 0.95  # Default confidence level for VaR calculation (95%)
DEFAULT_VAR_LOOKBACK_DAYS = 60  # Default lookback period for VaR calculation (days)
DEFAULT_VAR_MIN_HISTORY_DAYS = 20  # Minimum history required for reliable VaR
DEFAULT_VAR_MIN_PNL_SAMPLES = 10  # Minimum PnL samples required for VaR
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


# XGBoost Model Hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 200,  # Số lượng cây quyết định (trees). Càng nhiều càng phức tạp, dễ overfitting.
    "learning_rate": 0.05,  # Tốc độ học (eta). Kiểm soát mức đóng góp của mỗi cây.
    "max_depth": 5,  # Độ sâu tối đa của mỗi cây. Kiểm soát độ phức tạp của model.
    "subsample": 0.9,  # Tỷ lệ mẫu dữ liệu dùng cho mỗi cây (giảm overfitting).
    "colsample_bytree": 0.9,  # Tỷ lệ đặc trưng (features) dùng cho mỗi cây.
    "gamma": 0.1,  # Mức giảm loss tối thiểu để chia nút (cắt tỉa cây).
    "min_child_weight": 3,  # Tổng trọng số tối thiểu tại nút con (tránh học nhiễu).
    "random_state": 42,  # Hạt giống ngẫu nhiên để tái lập kết quả.
    "objective": "multi:softprob",  # Hàm mục tiêu: phân loại đa lớp trả về xác suất.
    "eval_metric": "mlogloss",  # Thước đo đánh giá lỗi: Multi-class Log Loss.
    "n_jobs": -1,  # Sử dụng tất cả lõi CPU.
}
