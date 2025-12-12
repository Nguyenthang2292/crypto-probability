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

# Exchange Settings
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

# Data Fetching Settings
DEFAULT_SYMBOL = "BTC/USDT"  # Default trading pair
DEFAULT_QUOTE = "USDT"  # Default quote currency
DEFAULT_TIMEFRAME = "15m"  # Default timeframe
DEFAULT_LIMIT = 1500  # Default number of candles to fetch


# ============================================================================
# XGBOOST PREDICTION CONFIGURATION
# ============================================================================

# Prediction Target Configuration
TARGET_HORIZON = 24  # Number of candles to predict ahead
TARGET_BASE_THRESHOLD = 0.01  # Base threshold for directional labeling (1%)
TARGET_LABELS = ["DOWN", "NEUTRAL", "UP"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(TARGET_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

# Prediction Windows Mapping
# Maps input timeframes to prediction horizons
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

# Dynamic Lookback Configuration
# Controls how historical reference prices are weighted based on volatility
DYNAMIC_LOOKBACK_SHORT_MULTIPLIER = 1.5  # Short lookback: TARGET_HORIZON * 1.5
DYNAMIC_LOOKBACK_MEDIUM_MULTIPLIER = 2.0  # Medium lookback: TARGET_HORIZON * 2.0 (original)
DYNAMIC_LOOKBACK_LONG_MULTIPLIER = 2.5  # Long lookback: TARGET_HORIZON * 2.5

# Volatility Thresholds for Weight Adjustment
DYNAMIC_LOOKBACK_VOL_LOW_THRESHOLD = 1.8  # Below this = low volatility
DYNAMIC_LOOKBACK_VOL_HIGH_THRESHOLD = 2.2  # Above this = high volatility

# Weight Configuration for Different Volatility Regimes
# Format: [weight_short, weight_medium, weight_long]
DYNAMIC_LOOKBACK_WEIGHTS_LOW_VOL = [0.4, 0.4, 0.2]  # Low volatility: favor short-medium
DYNAMIC_LOOKBACK_WEIGHTS_MEDIUM_VOL = [0.2, 0.5, 0.3]  # Medium volatility: balanced
DYNAMIC_LOOKBACK_WEIGHTS_HIGH_VOL = [0.2, 0.3, 0.5]  # High volatility: favor medium-long

# Model Features List
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
    "MARUBOZU_BULL",
    "MARUBOZU_BEAR",
    "SPINNING_TOP",
    "DRAGONFLY_DOJI",
    "GRAVESTONE_DOJI",
    "BULLISH_ENGULFING",
    "BEARISH_ENGULFING",
    "BULLISH_HARAMI",
    "BEARISH_HARAMI",
    "HARAMI_CROSS_BULL",
    "HARAMI_CROSS_BEAR",
    "MORNING_STAR",
    "EVENING_STAR",
    "PIERCING",
    "DARK_CLOUD",
    "THREE_WHITE_SOLDIERS",
    "THREE_BLACK_CROWS",
    "THREE_INSIDE_UP",
    "THREE_INSIDE_DOWN",
    "TWEEZER_TOP",
    "TWEEZER_BOTTOM",
    "RISING_WINDOW",
    "FALLING_WINDOW",
    "TASUKI_GAP_BULL",
    "TASUKI_GAP_BEAR",
    "MAT_HOLD_BULL",
    "MAT_HOLD_BEAR",
    "ADVANCE_BLOCK",
    "STALLED_PATTERN",
    "BELT_HOLD_BULL",
    "BELT_HOLD_BEAR",
    "KICKER_BULL",
    "KICKER_BEAR",
    "HANGING_MAN",
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

# Benchmark Configuration
BENCHMARK_SYMBOL = "BTC/USDT"  # Default benchmark for beta calculation

# Beta Calculation Configuration
DEFAULT_BETA_MIN_POINTS = 50  # Minimum data points required for beta calculation
DEFAULT_BETA_LIMIT = 1000  # Default limit for beta calculation OHLCV fetch
DEFAULT_BETA_TIMEFRAME = "1h"  # Default timeframe for beta calculation

# Correlation Analysis Configuration
DEFAULT_CORRELATION_MIN_POINTS = 10  # Minimum data points for correlation analysis
DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS = (
    10  # Minimum data points for weighted correlation
)

# Hedge Correlation Thresholds
HEDGE_CORRELATION_HIGH_THRESHOLD = 0.7  # High correlation threshold (>= 0.7 = excellent for hedging)
HEDGE_CORRELATION_MEDIUM_THRESHOLD = 0.4  # Medium correlation threshold (0.4-0.7 = moderate hedging effect)
HEDGE_CORRELATION_DIFF_THRESHOLD = 0.1  # Maximum difference between methods for consistency check

# VaR (Value at Risk) Configuration
DEFAULT_VAR_CONFIDENCE = 0.95  # Default confidence level for VaR calculation (95%)
DEFAULT_VAR_LOOKBACK_DAYS = 60  # Default lookback period for VaR calculation (days)
DEFAULT_VAR_MIN_HISTORY_DAYS = 20  # Minimum history required for reliable VaR
DEFAULT_VAR_MIN_PNL_SAMPLES = 10  # Minimum PnL samples required for VaR


# ============================================================================
# DEEP LEARNING CONFIGURATION
# ============================================================================

# Triple Barrier Method Configuration
DEEP_TRIPLE_BARRIER_TP_THRESHOLD = 0.02  # 2% take profit threshold
DEEP_TRIPLE_BARRIER_SL_THRESHOLD = 0.01  # 1% stop loss threshold

# Fractional Differentiation Configuration
DEEP_FRACTIONAL_DIFF_D = 0.5  # Fractional differentiation order (0 < d < 1)
DEEP_FRACTIONAL_DIFF_WINDOW = 100  # Window size for fractional differentiation

# Pipeline Configuration
DEEP_USE_FRACTIONAL_DIFF = True  # Whether to apply fractional differentiation
DEEP_USE_TRIPLE_BARRIER = False  # Whether to use Triple Barrier Method for labeling
DEEP_SCALER_DIR = "artifacts/deep/scalers"  # Directory to save/load scaler parameters

# Data Split Configuration
DEEP_TRAIN_RATIO = 0.7  # Proportion for training set
DEEP_VAL_RATIO = 0.15  # Proportion for validation set
DEEP_TEST_RATIO = 0.15  # Proportion for test set

# Feature Selection & Engineering Configuration
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
# HMM CONFIGURATION
# ============================================================================

# KAMA (Kaufman Adaptive Moving Average) Configuration
HMM_WINDOW_KAMA_DEFAULT = 10  # Default window size for KAMA calculation
HMM_FAST_KAMA_DEFAULT = 2  # Default fast period for KAMA
HMM_SLOW_KAMA_DEFAULT = 30  # Default slow period for KAMA
HMM_WINDOW_SIZE_DEFAULT = 100  # Default window size for HMM analysis

# High-Order HMM Configuration
HMM_HIGH_ORDER_ORDERS_ARGRELEXTREMA_DEFAULT = 5  # Order parameter for argrelextrema swing detection
HMM_HIGH_ORDER_STRICT_MODE_DEFAULT = False  # Whether to use strict mode for swing-to-state conversion

# Signal Configuration
# Note: Signal values (LONG=1, HOLD=0, SHORT=-1) are now constants in modules.hmm.signal_resolution
HMM_PROBABILITY_THRESHOLD = 0.5  # Minimum probability threshold for signal generation

# Signal Scoring Configuration
HMM_SIGNAL_PRIMARY_WEIGHT = 2  # Weight for primary signal (next_state_with_hmm_kama)
HMM_SIGNAL_TRANSITION_WEIGHT = 1  # Weight for transition states
HMM_SIGNAL_ARM_WEIGHT = 1  # Weight for ARM-based states
HMM_SIGNAL_MIN_THRESHOLD = 3  # Minimum score threshold for signal generation

# Confidence & Normalization Configuration
HMM_HIGH_ORDER_MAX_SCORE = 1.0  # Max score from High-Order HMM (normalized)
HMM_HIGH_ORDER_WEIGHT = 0.4  # Weight for High-Order HMM in combined confidence
HMM_KAMA_WEIGHT = 1.0 - HMM_HIGH_ORDER_WEIGHT  # Weight for KAMA (calculated automatically)
HMM_AGREEMENT_BONUS = 1.2  # Bonus multiplier when signals agree

# Feature Flags
HMM_FEATURES = {
    "confidence_enabled": True,
    "normalization_enabled": True,
    "combined_confidence_enabled": True,
    "high_order_scoring_enabled": True,
    "conflict_resolution_enabled": True,
    "dynamic_threshold_enabled": True,
    "state_strength_enabled": True,
}

# High-Order HMM Scoring Configuration
HMM_HIGH_ORDER_STRENGTH = {
    "bearish": 1.0,  # Strength multiplier for bearish signals
    "bullish": 1.0,  # Strength multiplier for bullish signals
}

# Conflict Resolution Configuration
HMM_CONFLICT_RESOLUTION_THRESHOLD = 1.2  # Ratio to prioritize High-Order over KAMA

# Dynamic Threshold Configuration
HMM_VOLATILITY_CONFIG = {
    "high_threshold": 0.03,  # Volatility threshold (3% std)
    "adjustments": {
        "high": 1.2,   # Multiplier for high volatility (conservative)
        "low": 0.9,   # Multiplier for low volatility (aggressive)
    }
}

# State Strength Multipliers Configuration
HMM_STATE_STRENGTH = {
    "strong": 1.0,  # Multiplier for strong states (0, 3)
    "weak": 0.7,   # Multiplier for weak states (1, 2)
}


# ============================================================================
# PAIRS TRADING CONFIGURATION
# ============================================================================

# Performance Analysis Weights (Default)
PAIRS_TRADING_WEIGHTS = {
    '1d': 0.5,   # Weight for 1 day (24 candles)
    '3d': 0.3,   # Weight for 3 days (72 candles)
    '1w': 0.2    # Weight for 1 week (168 candles)
}

# Performance Analysis Weight Presets
# Named presets for CLI selection with different trading strategies
PAIRS_TRADING_WEIGHT_PRESETS = {
    "momentum": {'1d': 0.5, '3d': 0.3, '1w': 0.2},          # Favor short-term signals
    "balanced": {'1d': 0.3, '3d': 0.4, '1w': 0.3},          # Balanced short-medium-long term
    "short_term_bounce": {'1d': 0.7, '3d': 0.2, '1w': 0.1},  # Very sensitive to 1d volatility
    "trend_follower": {'1d': 0.2, '3d': 0.3, '1w': 0.5},     # Follow longer trends
    "mean_reversion": {'1d': 0.25, '3d': 0.5, '1w': 0.25},   # Emphasize medium-term for mean reversion
    "volatility_buffer": {'1d': 0.2, '3d': 0.4, '1w': 0.4},  # Reduce short-term noise, increase stability
}

# Hedge Ratio Calculation Configuration
PAIRS_TRADING_OLS_FIT_INTERCEPT = True  # Whether to fit intercept in OLS regression
PAIRS_TRADING_KALMAN_DELTA = 1e-5  # Default delta for Kalman filter
PAIRS_TRADING_KALMAN_OBS_COV = 1.0  # Default observation covariance for Kalman filter

# Kalman Filter Presets
PAIRS_TRADING_KALMAN_PRESETS = {
    "fast_react": {
        "description": "Fast reaction – beta changes quickly, suitable for volatile markets",
        "delta": 5e-5,
        "obs_cov": 0.5,
    },
    "balanced": {
        "description": "Balanced – default setting, moderate reaction",
        "delta": 1e-5,
        "obs_cov": 1.0,
    },
    "stable": {
        "description": "Stable – beta changes slowly, reduces noise",
        "delta": 5e-6,
        "obs_cov": 2.0,
    },
}

# Opportunity Scoring Presets
# Multipliers for different scoring factors
PAIRS_TRADING_OPPORTUNITY_PRESETS = {
    "balanced": {
        "description": "Default balanced between rewards/penalties",
        "hedge_ratio_strategy": "best",  # 'ols', 'kalman', 'best', or 'avg'
        "corr_good_bonus": 1.20,
        "corr_low_penalty": 0.80,
        "corr_high_penalty": 0.90,
        "cointegration_bonus": 1.15,
        "weak_cointegration_bonus": 1.05,
        "half_life_bonus": 1.10,
        "zscore_divisor": 5.0,
        "zscore_cap": 0.20,
        "hurst_good_bonus": 1.08,
        "hurst_ok_bonus": 1.02,
        "hurst_ok_threshold": 0.60,
        "sharpe_good_bonus": 1.08,
        "sharpe_ok_bonus": 1.03,
        "maxdd_bonus": 1.05,
        "calmar_bonus": 1.05,
        "johansen_bonus": 1.08,
        "f1_high_bonus": 1.05,
        "f1_mid_bonus": 1.02,
        # Momentum-specific bonuses
        "momentum_cointegration_penalty": 0.95,
        "momentum_zscore_high_bonus": 1.15,  # Bonus for |z-score| > 2.0
        "momentum_zscore_moderate_bonus": 1.08,  # Bonus for |z-score| > 1.0
        "momentum_zscore_high_threshold": 2.0,  # Threshold for high z-score bonus
        "momentum_zscore_moderate_threshold": 1.0,  # Threshold for moderate z-score bonus
    },
    "aggressive": {
        "description": "Large rewards for strong signals, accept volatility",
        "hedge_ratio_strategy": "best",  # Use best of OLS/Kalman
        "corr_good_bonus": 1.30,
        "corr_low_penalty": 0.70,
        "corr_high_penalty": 0.85,
        "cointegration_bonus": 1.25,
        "weak_cointegration_bonus": 1.10,
        "half_life_bonus": 1.15,
        "zscore_divisor": 4.0,
        "zscore_cap": 0.30,
        "hurst_good_bonus": 1.12,
        "hurst_ok_bonus": 1.05,
        "hurst_ok_threshold": 0.65,
        "sharpe_good_bonus": 1.12,
        "sharpe_ok_bonus": 1.05,
        "maxdd_bonus": 1.02,
        "calmar_bonus": 1.02,
        "johansen_bonus": 1.12,
        "f1_high_bonus": 1.08,
        "f1_mid_bonus": 1.04,
        # Momentum-specific bonuses
        "momentum_cointegration_penalty": 0.90,
        "momentum_zscore_high_bonus": 1.20,  # Bonus for |z-score| > 2.0
        "momentum_zscore_moderate_bonus": 1.12,  # Bonus for |z-score| > 1.0
        "momentum_zscore_high_threshold": 2.0,
        "momentum_zscore_moderate_threshold": 1.0,
    },
    "conservative": {
        "description": "Light rewards, prioritize stable pairs",
        "hedge_ratio_strategy": "ols",  # Prefer stable OLS metrics
        "corr_good_bonus": 1.10,
        "corr_low_penalty": 0.90,
        "corr_high_penalty": 0.95,
        "cointegration_bonus": 1.10,
        "weak_cointegration_bonus": 1.02,
        "half_life_bonus": 1.05,
        "zscore_divisor": 6.0,
        "zscore_cap": 0.15,
        "hurst_good_bonus": 1.04,
        "hurst_ok_bonus": 1.01,
        "hurst_ok_threshold": 0.55,
        "sharpe_good_bonus": 1.05,
        "sharpe_ok_bonus": 1.02,
        "maxdd_bonus": 1.08,
        "calmar_bonus": 1.08,
        "johansen_bonus": 1.05,
        "f1_high_bonus": 1.03,
        "f1_mid_bonus": 1.01,
        # Momentum-specific bonuses
        "momentum_cointegration_penalty": 0.98,
        "momentum_zscore_high_bonus": 1.10,  # Bonus for |z-score| > 2.0
        "momentum_zscore_moderate_bonus": 1.05,  # Bonus for |z-score| > 1.0
        "momentum_zscore_high_threshold": 2.0,
        "momentum_zscore_moderate_threshold": 1.0,
    },
}

# Quantitative Score Weights Configuration
# Default weights and thresholds for calculate_quantitative_score (0-100 scale)
PAIRS_TRADING_QUANTITATIVE_SCORE_WEIGHTS = {
    # Cointegration weights
    "cointegration_full_weight": 30.0,  # Full weight if cointegrated
    "cointegration_weak_weight": 15.0,  # Weak weight if pvalue < 0.1
    "cointegration_weak_pvalue_threshold": 0.1,  # Threshold for weak cointegration
    
    # Half-life weights and thresholds
    "half_life_excellent_weight": 20.0,  # Weight if half_life < excellent_threshold
    "half_life_good_weight": 10.0,  # Weight if half_life < good_threshold
    "half_life_excellent_threshold": 20.0,  # Excellent threshold (periods)
    "half_life_good_threshold": 50.0,  # Good threshold (periods)
    
    # Hurst exponent weights and thresholds
    "hurst_excellent_weight": 15.0,  # Weight if hurst < excellent_threshold
    "hurst_good_weight": 8.0,  # Weight if hurst < good_threshold
    "hurst_excellent_threshold": 0.4,  # Excellent threshold
    "hurst_good_threshold": 0.5,  # Good threshold
    
    # Sharpe ratio weights and thresholds
    "sharpe_excellent_weight": 15.0,  # Weight if sharpe > excellent_threshold
    "sharpe_good_weight": 8.0,  # Weight if sharpe > good_threshold
    "sharpe_excellent_threshold": 2.0,  # Excellent threshold
    "sharpe_good_threshold": 1.0,  # Good threshold
    
    # F1-score weights and thresholds
    "f1_excellent_weight": 10.0,  # Weight if f1 > excellent_threshold
    "f1_good_weight": 5.0,  # Weight if f1 > good_threshold
    "f1_excellent_threshold": 0.7,  # Excellent threshold
    "f1_good_threshold": 0.6,  # Good threshold
    
    # Max drawdown weights and thresholds
    "maxdd_excellent_weight": 10.0,  # Weight if abs(maxdd) < excellent_threshold
    "maxdd_good_weight": 5.0,  # Weight if abs(maxdd) < good_threshold
    "maxdd_excellent_threshold": 0.2,  # Excellent threshold (20%)
    "maxdd_good_threshold": 0.3,  # Good threshold (30%)
    
    # Calmar ratio weights and thresholds
    "calmar_excellent_weight": 5.0,  # Weight if calmar >= excellent_threshold
    "calmar_good_weight": 2.5,  # Weight if calmar >= good_threshold
    "calmar_excellent_threshold": 1.0,  # Excellent threshold
    "calmar_good_threshold": 0.5,  # Good threshold
    
    # Momentum extensions
    "momentum_adx_strong_weight": 10.0,
    "momentum_adx_moderate_weight": 5.0,

    # Maximum score cap
    "max_score": 100.0,  # Maximum quantitative score (capped at 100)
}

# Momentum-specific filters and thresholds
PAIRS_TRADING_ADX_PERIOD = 14
PAIRS_TRADING_MOMENTUM_FILTERS = {
    "min_adx": 18.0,            # Minimum ADX required to consider a leg trending
    "strong_adx": 25.0,         # Strong trend confirmation threshold
    "adx_base_bonus": 1.03,     # Bonus when both legs pass min_adx
    "adx_strong_bonus": 1.08,   # Bonus when both legs exceed strong_adx
    "adx_weak_penalty_factor": 0.5,  # Penalty scaling factor when ADX < min_adx (0.0-1.0, lower = more penalty)
    "adx_very_weak_threshold": 10.0,  # ADX below this gets severe penalty
    "adx_very_weak_penalty": 0.3,  # Severe penalty multiplier for very weak ADX
    "low_corr_threshold": 0.30, # Prefer divergence / low correlation
    "high_corr_threshold": 0.75,# Penalize highly correlated legs
    "low_corr_bonus": 1.05,     # Bonus if |corr| below low_corr_threshold
    "negative_corr_bonus": 1.10,# Bonus if correlation is negative
    "high_corr_penalty": 0.90,  # Penalty if |corr| above high_corr_threshold
}

# Performance Analysis Settings
PAIRS_TRADING_TOP_N = 5  # Number of top/bottom performers to display
PAIRS_TRADING_MIN_CANDLES = 168  # Minimum candles required (1 week = 168h)
PAIRS_TRADING_TIMEFRAME = "1h"  # Timeframe for analysis
PAIRS_TRADING_LIMIT = 200  # Number of candles to fetch (enough for 1 week + buffer)

# Pairs Validation Settings
PAIRS_TRADING_MIN_VOLUME = 1000000  # Minimum volume (USDT) to consider
PAIRS_TRADING_MIN_SPREAD = 0.01  # Minimum spread (%)
PAIRS_TRADING_MAX_SPREAD = 0.50  # Maximum spread (%)
PAIRS_TRADING_MIN_CORRELATION = 0.3  # Minimum correlation to consider
PAIRS_TRADING_MAX_CORRELATION = 0.9  # Maximum correlation (avoid over-correlation)
PAIRS_TRADING_CORRELATION_MIN_POINTS = 50  # Minimum data points for correlation calculation
PAIRS_TRADING_ADF_PVALUE_THRESHOLD = 0.05  # P-value threshold for cointegration confirmation
PAIRS_TRADING_ADF_MAXLAG = 1  # Maximum lag for Augmented Dickey-Fuller test (used with autolag="AIC")
PAIRS_TRADING_MAX_HALF_LIFE = 50  # Maximum candles for half-life (mean reversion)
PAIRS_TRADING_MIN_HALF_LIFE_POINTS = 10  # Minimum number of valid data points required for half-life calculation
PAIRS_TRADING_ZSCORE_LOOKBACK = 60  # Number of candles for rolling z-score calculation
PAIRS_TRADING_HURST_THRESHOLD = 0.5  # Hurst exponent threshold (mean reversion < 0.5)
PAIRS_TRADING_MIN_SPREAD_SHARPE = 1.0  # Minimum Sharpe ratio for spread
PAIRS_TRADING_MAX_DRAWDOWN = 0.3  # Maximum drawdown (30%)
PAIRS_TRADING_MIN_CALMAR = 1.0  # Minimum Calmar ratio
PAIRS_TRADING_JOHANSEN_CONFIDENCE = 0.95  # Confidence level for Johansen test
PAIRS_TRADING_JOHANSEN_DET_ORDER = 0  # Deterministic order: 0=no constant/trend, 1=constant, -1=no constant with trend
PAIRS_TRADING_JOHANSEN_K_AR_DIFF = 1  # Lag order for VAR model (k_ar_diff = number of lags)
PAIRS_TRADING_PERIODS_PER_YEAR = 365 * 24  # Number of periods per year (for 1h timeframe)
PAIRS_TRADING_CLASSIFICATION_ZSCORE = 0.5  # Z-score threshold for direction classification

# Z-score Metrics Configuration
PAIRS_TRADING_MIN_LAG = 2  # Minimum lag for R/S analysis (must be >= 2 for meaningful variance calculation)
PAIRS_TRADING_MAX_LAG_DIVISOR = 2  # Maximum lag is limited to half of series length for stability
PAIRS_TRADING_HURST_EXPONENT_MULTIPLIER = 2.0  # Multiplier to convert R/S slope to Hurst exponent
PAIRS_TRADING_MIN_CLASSIFICATION_SAMPLES = 20  # Minimum number of samples required for reliable classification metrics
PAIRS_TRADING_HURST_EXPONENT_MIN = 0.0  # Theoretical minimum for Hurst exponent
PAIRS_TRADING_HURST_EXPONENT_MAX = 2.0  # Theoretical maximum for Hurst exponent
PAIRS_TRADING_HURST_EXPONENT_MEAN_REVERTING_MAX = 0.5  # Maximum Hurst for mean-reverting behavior

# Pairs DataFrame Column Names
# Complete list of columns for pairs trading DataFrame results
# This includes core pair information and all quantitative metrics
PAIRS_TRADING_PAIR_COLUMNS = [
    # Core pair information
    'long_symbol',
    'short_symbol',
    'long_score',
    'short_score',
    'spread',
    'correlation',
    'opportunity_score',
    'quantitative_score',
    # OLS-based metrics
    'hedge_ratio',
    'adf_pvalue',
    'is_cointegrated',
    'half_life',
    'mean_zscore',
    'std_zscore',
    'skewness',
    'kurtosis',
    'current_zscore',
    'hurst_exponent',
    'spread_sharpe',
    'max_drawdown',
    'calmar_ratio',
    'classification_f1',
    'classification_precision',
    'classification_recall',
    'classification_accuracy',
    # Johansen test (independent of hedge ratio method)
    'johansen_trace_stat',
    'johansen_critical_value',
    'is_johansen_cointegrated',
    # Kalman hedge ratio
    'kalman_hedge_ratio',
    # Kalman-based metrics
    'kalman_half_life',
    'kalman_mean_zscore',
    'kalman_std_zscore',
    'kalman_skewness',
    'kalman_kurtosis',
    'kalman_current_zscore',
    'kalman_hurst_exponent',
    'kalman_spread_sharpe',
    'kalman_max_drawdown',
    'kalman_calmar_ratio',
    'kalman_classification_f1',
    'kalman_classification_precision',
    'kalman_classification_recall',
    'kalman_classification_accuracy',
]


# ============================================================================
# RANGE OSCILLATOR CONFIGURATION
# ============================================================================

# Strategy categories for dynamic selection
TRENDING_STRATEGIES = [3, 4, 6, 8]
RANGE_BOUND_STRATEGIES = [2, 7, 9]
VOLATILE_STRATEGIES = [6, 7]
STABLE_STRATEGIES = [2, 3, 9]

# Constants for performance scoring weights
AGREEMENT_WEIGHT = 0.6
STRENGTH_WEIGHT = 0.4

# Normalization constant for oscillator extreme calculation
OSCILLATOR_NORMALIZATION = 100.0

# Valid strategy IDs
VALID_STRATEGY_IDS = {2, 3, 4, 6, 7, 8, 9}

# Range Oscillator Default Parameters
# These parameters control Range Oscillator signal generation
# Adjust based on backtesting results or market conditions
RANGE_OSCILLATOR_LENGTH = 50  # Oscillator length parameter
RANGE_OSCILLATOR_MULTIPLIER = 2.0  # Oscillator multiplier


# ============================================================================
# DECISION MATRIX CONFIGURATION
# ============================================================================

# Indicator Accuracy Values (for Decision Matrix voting system)
# These values represent historical accuracy/performance of each indicator
# Adjust based on backtesting results or actual performance data

# Main Indicators Accuracy
DECISION_MATRIX_ATC_ACCURACY = 0.65  # Adaptive Trend Classification accuracy
DECISION_MATRIX_OSCILLATOR_ACCURACY = 0.70  # Range Oscillator accuracy (highest)

# SPC Strategy Accuracies (for weighted aggregation)
DECISION_MATRIX_SPC_CLUSTER_TRANSITION_ACCURACY = 0.68  # Cluster Transition strategy accuracy
DECISION_MATRIX_SPC_REGIME_FOLLOWING_ACCURACY = 0.66  # Regime Following strategy accuracy
DECISION_MATRIX_SPC_MEAN_REVERSION_ACCURACY = 0.64  # Mean Reversion strategy accuracy

# SPC Aggregated Accuracy (weighted average of 3 strategies)
# Formula: (0.68 + 0.66 + 0.64) / 3 ≈ 0.66
DECISION_MATRIX_SPC_AGGREGATED_ACCURACY = 0.66

# Dictionary for easy access to SPC strategy accuracies
DECISION_MATRIX_SPC_STRATEGY_ACCURACIES = {
    'cluster_transition': DECISION_MATRIX_SPC_CLUSTER_TRANSITION_ACCURACY,
    'regime_following': DECISION_MATRIX_SPC_REGIME_FOLLOWING_ACCURACY,
    'mean_reversion': DECISION_MATRIX_SPC_MEAN_REVERSION_ACCURACY,
}

# Dictionary for all indicator accuracies
DECISION_MATRIX_INDICATOR_ACCURACIES = {
    'atc': DECISION_MATRIX_ATC_ACCURACY,
    'oscillator': DECISION_MATRIX_OSCILLATOR_ACCURACY,
    'spc': DECISION_MATRIX_SPC_AGGREGATED_ACCURACY,
}

# SPC Strategy-Specific Parameters
# These parameters control signal generation for each SPC strategy
# Adjust based on backtesting results or market conditions

# Cluster Transition Strategy Parameters
SPC_CLUSTER_TRANSITION_MIN_SIGNAL_STRENGTH = 0.3  # Minimum signal strength threshold
SPC_CLUSTER_TRANSITION_MIN_REL_POS_CHANGE = 0.1  # Minimum relative position change

# Regime Following Strategy Parameters
SPC_REGIME_FOLLOWING_MIN_REGIME_STRENGTH = 0.7  # Minimum regime strength threshold
SPC_REGIME_FOLLOWING_MIN_CLUSTER_DURATION = 2  # Minimum bars in same cluster

# Mean Reversion Strategy Parameters
SPC_MEAN_REVERSION_EXTREME_THRESHOLD = 0.2  # Real_clust threshold for extreme detection
SPC_MEAN_REVERSION_MIN_EXTREME_DURATION = 3  # Minimum bars in extreme state

# Dictionary for easy access to SPC strategy parameters
SPC_STRATEGY_PARAMETERS = {
    'cluster_transition': {
        'min_signal_strength': SPC_CLUSTER_TRANSITION_MIN_SIGNAL_STRENGTH,
        'min_rel_pos_change': SPC_CLUSTER_TRANSITION_MIN_REL_POS_CHANGE,
    },
    'regime_following': {
        'min_regime_strength': SPC_REGIME_FOLLOWING_MIN_REGIME_STRENGTH,
        'min_cluster_duration': SPC_REGIME_FOLLOWING_MIN_CLUSTER_DURATION,
    },
    'mean_reversion': {
        'extreme_threshold': SPC_MEAN_REVERSION_EXTREME_THRESHOLD,
        'min_extreme_duration': SPC_MEAN_REVERSION_MIN_EXTREME_DURATION,
    },
}