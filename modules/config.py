"""
Configuration constants for all components.

Organized by component:
1. Common/Shared Configuration
2. XGBoost Prediction Configuration
3. Portfolio Manager Configuration
4. Deep Learning Configuration
5. Pairs Trading Configuration
"""

# ============================================================================
# COMMON / SHARED CONFIGURATION
# ============================================================================

# Default exchange settings
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
DEFAULT_REQUEST_PAUSE = 0.2  # Pause between API requests (seconds)
DEFAULT_CONTRACT_TYPE = "future"  # Contract type: 'spot', 'margin', or 'future'

# Default data fetching settings
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_QUOTE = "USDT"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_LIMIT = 1500


# ============================================================================
# XGBOOST PREDICTION CONFIGURATION
# ============================================================================

# Prediction target configuration
TARGET_HORIZON = 24  # Number of candles to predict ahead
TARGET_BASE_THRESHOLD = 0.01  # Base threshold for directional labeling (1%)
TARGET_LABELS = ["DOWN", "NEUTRAL", "UP"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(TARGET_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

# Prediction windows mapping
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

# Dynamic Lookback Weight Configuration
# Controls how historical reference prices are weighted based on volatility
DYNAMIC_LOOKBACK_SHORT_MULTIPLIER = 1.5  # Short lookback: TARGET_HORIZON * 1.5
DYNAMIC_LOOKBACK_MEDIUM_MULTIPLIER = 2.0  # Medium lookback: TARGET_HORIZON * 2.0 (original)
DYNAMIC_LOOKBACK_LONG_MULTIPLIER = 2.5  # Long lookback: TARGET_HORIZON * 2.5

# Volatility thresholds for weight adjustment
DYNAMIC_LOOKBACK_VOL_LOW_THRESHOLD = 1.8  # Below this = low volatility
DYNAMIC_LOOKBACK_VOL_HIGH_THRESHOLD = 2.2  # Above this = high volatility

# Weight configuration for different volatility regimes
# Format: [weight_short, weight_medium, weight_long]
DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL = [0.4, 0.4, 0.2]  # Low volatility: favor short-medium
DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL = [0.2, 0.5, 0.3]  # Medium volatility: balanced
DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL = [0.2, 0.3, 0.5]  # High volatility: favor medium-long

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


# ============================================================================
# PORTFOLIO MANAGER CONFIGURATION
# ============================================================================

# Benchmark configuration
BENCHMARK_SYMBOL = "BTC/USDT"  # Default benchmark for beta calculation

# Beta calculation configuration
DEFAULT_BETA_MIN_POINTS = 50  # Minimum data points required for beta calculation
DEFAULT_BETA_LIMIT = 1000  # Default limit for beta calculation OHLCV fetch
DEFAULT_BETA_TIMEFRAME = "1h"  # Default timeframe for beta calculation

# Correlation analysis configuration
DEFAULT_CORRELATION_MIN_POINTS = 10  # Minimum data points for correlation analysis
DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS = (
    10  # Minimum data points for weighted correlation
)

# VaR (Value at Risk) configuration
DEFAULT_VAR_CONFIDENCE = 0.95  # Default confidence level for VaR calculation (95%)
DEFAULT_VAR_LOOKBACK_DAYS = 60  # Default lookback period for VaR calculation (days)
DEFAULT_VAR_MIN_HISTORY_DAYS = 20  # Minimum history required for reliable VaR
DEFAULT_VAR_MIN_PNL_SAMPLES = 10  # Minimum PnL samples required for VaR


# ============================================================================
# DEEP LEARNING CONFIGURATION
# ============================================================================

# Triple Barrier Method defaults
DEEP_TRIPLE_BARRIER_TP_THRESHOLD = 0.02  # 2% take profit threshold
DEEP_TRIPLE_BARRIER_SL_THRESHOLD = 0.01  # 1% stop loss threshold

# Fractional Differentiation defaults
DEEP_FRACTIONAL_DIFF_D = 0.5  # Fractional differentiation order (0 < d < 1)
DEEP_FRACTIONAL_DIFF_WINDOW = 100  # Window size for fractional differentiation

# Deep Learning Pipeline defaults
DEEP_USE_FRACTIONAL_DIFF = True  # Whether to apply fractional differentiation
DEEP_USE_TRIPLE_BARRIER = False  # Whether to use Triple Barrier Method for labeling
DEEP_SCALER_DIR = "artifacts/deep/scalers"  # Directory to save/load scaler parameters

# Data split defaults
DEEP_TRAIN_RATIO = 0.7  # Proportion for training set
DEEP_VAL_RATIO = 0.15  # Proportion for validation set
DEEP_TEST_RATIO = 0.15  # Proportion for test set

# Feature Selection & Engineering defaults
DEEP_FEATURE_SELECTION_METHOD = "combined"  # 'mutual_info', 'boruta', 'f_test', or 'combined'
DEEP_FEATURE_SELECTION_TOP_K = 25  # Number of top features to select (20-30 recommended)
DEEP_FEATURE_COLLINEARITY_THRESHOLD = 0.85  # Correlation threshold for removing collinear features (0.8-0.95)
DEEP_FEATURE_SELECTION_DIR = "artifacts/deep/feature_selection"  # Directory to save/load feature selection results
DEEP_USE_FEATURE_SELECTION = True  # Whether to apply feature selection

# Dataset & DataModule Configuration
DEEP_MAX_ENCODER_LENGTH = 64  # Lookback window (64-128 bars recommended)
DEEP_MAX_PREDICTION_LENGTH = TARGET_HORIZON  # Prediction horizon (align with TARGET_HORIZON)
DEEP_BATCH_SIZE = 64  # Batch size for training
DEEP_NUM_WORKERS = 4  # Number of workers for DataLoader
DEEP_TARGET_COL = "future_log_return"  # Default target column for regression
DEEP_TARGET_COL_CLASSIFICATION = "triple_barrier_label"  # Target column for classification
DEEP_DATASET_DIR = "artifacts/deep/datasets"  # Directory to save/load dataset metadata

# Model Configuration - Phase 1: Vanilla TFT (MVP)
DEEP_MODEL_HIDDEN_SIZE = 16  # Hidden size of TFT model
DEEP_MODEL_ATTENTION_HEAD_SIZE = 4  # Size of attention heads
DEEP_MODEL_DROPOUT = 0.1  # Dropout rate
DEEP_MODEL_LEARNING_RATE = 0.03  # Learning rate
DEEP_MODEL_QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]  # Quantiles for QuantileLoss
DEEP_MODEL_REDUCE_ON_PLATEAU_PATIENCE = 4  # Patience for learning rate reduction
DEEP_CHECKPOINT_DIR = "artifacts/deep/checkpoints"  # Directory to save model checkpoints
DEEP_EARLY_STOPPING_PATIENCE = 10  # Early stopping patience
DEEP_CHECKPOINT_SAVE_TOP_K = 3  # Number of top checkpoints to save

# Model Configuration - Phase 2: Optuna Optimization
DEEP_OPTUNA_N_TRIALS = 20  # Number of optimization trials
DEEP_OPTUNA_TIMEOUT = None  # Timeout in seconds (None = no timeout)
DEEP_OPTUNA_N_JOBS = 1  # Number of parallel jobs
DEEP_OPTUNA_DIR = "artifacts/deep/optuna"  # Directory to save Optuna results
DEEP_OPTUNA_MAX_EPOCHS = 50  # Maximum epochs per trial

# Model Configuration - Phase 3: Hybrid LSTM + TFT (Advanced)
DEEP_HYBRID_LSTM_HIDDEN_SIZE = 32  # Hidden size for LSTM branch
DEEP_HYBRID_LSTM_NUM_LAYERS = 2  # Number of LSTM layers
DEEP_HYBRID_FUSION_SIZE = 64  # Size of fused representation
DEEP_HYBRID_NUM_CLASSES = 3  # Number of classes for classification (UP, NEUTRAL, DOWN)
DEEP_HYBRID_LAMBDA_CLASS = 1.0  # Weight for classification loss
DEEP_HYBRID_LAMBDA_REG = 1.0  # Weight for regression loss
DEEP_HYBRID_LEARNING_RATE = 0.001  # Learning rate for hybrid model

# Training Configuration
DEEP_MAX_EPOCHS = 100  # Maximum number of training epochs
DEEP_ACCELERATOR = "auto"  # Accelerator: 'auto', 'gpu', 'cpu'
DEEP_DEVICES = 1  # Number of devices to use
DEEP_PRECISION = 32  # Precision: 16, 32, or 'bf16'
DEEP_GRADIENT_CLIP_VAL = 0.5  # Gradient clipping value (None to disable)


# ============================================================================
# PAIRS TRADING CONFIGURATION
# ============================================================================

# Performance analysis weights
PAIRS_TRADING_WEIGHTS = {
    '1d': 0.5,   # Trọng số cho 1 ngày (24 candles)
    '3d': 0.3,   # Trọng số cho 3 ngày (72 candles)
    '1w': 0.2    # Trọng số cho 1 tuần (168 candles)
}

# Performance analysis settings
PAIRS_TRADING_TOP_N = 5  # Số lượng top/bottom performers
PAIRS_TRADING_MIN_CANDLES = 168  # Số candles tối thiểu (1 tuần = 168h)
PAIRS_TRADING_TIMEFRAME = "1h"  # Timeframe cho phân tích
PAIRS_TRADING_LIMIT = 200  # Số candles cần fetch (đủ cho 1 tuần + buffer)

# Pairs validation settings
PAIRS_TRADING_MIN_VOLUME = 1000000  # Volume tối thiểu (USDT) để xem xét
PAIRS_TRADING_MIN_SPREAD = 0.01  # Spread tối thiểu (%)
PAIRS_TRADING_MAX_SPREAD = 0.50  # Spread tối đa (%)
PAIRS_TRADING_MIN_CORRELATION = 0.3  # Correlation tối thiểu để xem xét
PAIRS_TRADING_MAX_CORRELATION = 0.9  # Correlation tối đa (tránh quá tương quan)
PAIRS_TRADING_CORRELATION_MIN_POINTS = 50  # Số điểm dữ liệu tối thiểu để tính correlation
PAIRS_TRADING_ADF_PVALUE_THRESHOLD = 0.05  # Ngưỡng p-value để xác nhận cointegration
PAIRS_TRADING_MAX_HALF_LIFE = 50  # Số candles tối đa cho half-life (mean reversion)
PAIRS_TRADING_ZSCORE_LOOKBACK = 60  # Số candles để tính rolling z-score
PAIRS_TRADING_HURST_THRESHOLD = 0.5  # Hurst exponent threshold (mean reversion < 0.5)
PAIRS_TRADING_MIN_SPREAD_SHARPE = 1.0  # Sharpe tối thiểu cho spread
PAIRS_TRADING_MAX_DRAWDOWN = 0.3  # Drawdown tối đa (30%)
PAIRS_TRADING_MIN_CALMAR = 1.0  # Calmar ratio tối thiểu
PAIRS_TRADING_JOHANSEN_CONFIDENCE = 0.95  # Mức confidence cho Johansen test
PAIRS_TRADING_PERIODS_PER_YEAR = 365 * 24  # Số kỳ trong năm (timeframe 1h)
PAIRS_TRADING_CLASSIFICATION_ZSCORE = 0.5  # Ngưỡng z-score cho phân loại direction
