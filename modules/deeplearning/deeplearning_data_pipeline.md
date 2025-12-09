# 📚 Deep Learning Data Pipeline Documentation

## Mục lục
1. [Tổng quan](#tổng-quan)
2. [Khởi tạo](#khởi-tạo)
3. [Phương thức chính](#phương-thức-chính)
4. [Các thành phần](#các-thành-phần)
5. [Ví dụ sử dụng](#ví-dụ-sử-dụng)
6. [Best Practices](#best-practices)
7. [Pipeline Steps](#pipeline-steps)

---

## Tổng quan

`DeepLearningDataPipeline` là một pipeline toàn diện để chuẩn bị dữ liệu cho deep learning models, đặc biệt là Temporal Fusion Transformer (TFT). Pipeline này cung cấp:

- ✅ **OHLCV Fetching** - Lấy dữ liệu từ DataFetcher với fallback tự động
- ✅ **Target Engineering** - Log Returns, % Change, Triple Barrier Method
- ✅ **Fractional Differentiation** - Đảm bảo stationarity mà vẫn giữ memory
- ✅ **Technical Indicators** - Tự động tính toán indicators qua IndicatorEngine
- ✅ **Known-future Features** - Time-of-day, day-of-week, funding schedule
- ✅ **Per-symbol Normalization** - StandardScaler per symbol để xử lý scale differences
- ✅ **Feature Selection** - Tích hợp FeatureSelector để chọn top features
- ✅ **Chronological Split** - Train/validation/test split với gap để tránh data leakage

### Khi nào dùng DeepLearningDataPipeline?

| Mục đích | Dùng Pipeline? | Phương thức |
|----------|----------------|-------------|
| Chuẩn bị data cho TFT | ✅ Có | `fetch_and_prepare()` + `split_chronological()` |
| Cần target engineering (log returns, triple barrier) | ✅ Có | Tự động trong `prepare_dataframe()` |
| Cần fractional differentiation | ✅ Có | Set `use_fractional_diff=True` |
| Cần technical indicators | ✅ Có | Tự động qua IndicatorEngine |
| Cần normalization per symbol | ✅ Có | Tự động trong `_normalize_per_symbol()` |
| Cần feature selection | ✅ Có | Tích hợp FeatureSelector |
| Cần train/val/test split | ✅ Có | `split_chronological()` |

---

## Khởi tạo

### Cú pháp

```python
from modules.deeplearning_data_pipeline import DeepLearningDataPipeline
from modules.DataFetcher import DataFetcher

# Khởi tạo DataFetcher trước
data_fetcher = DataFetcher(exchange_manager)

# Khởi tạo Pipeline
pipeline = DeepLearningDataPipeline(
    data_fetcher=data_fetcher,
    use_fractional_diff=True,
    use_triple_barrier=False,
    use_feature_selection=True
)
```

### Tham số chính

- `data_fetcher` (DataFetcher, **bắt buộc**): Instance của DataFetcher để lấy OHLCV data
- `indicator_engine` (IndicatorEngine, **tùy chọn**): Instance của IndicatorEngine (tạo mới nếu None)
- `use_fractional_diff` (bool, **mặc định**: `True`): Có áp dụng fractional differentiation không
- `fractional_diff_d` (float, **mặc định**: `0.5`): Order của fractional differentiation (0 < d < 1)
- `use_triple_barrier` (bool, **mặc định**: `False`): Có dùng Triple Barrier Method không
- `triple_barrier_tp` (float, **mặc định**: `0.02`): Take profit threshold (2%)
- `triple_barrier_sl` (float, **mặc định**: `0.01`): Stop loss threshold (1%)
- `use_feature_selection` (bool, **mặc định**: `True`): Có áp dụng feature selection không
- `feature_selection_method` (str, **mặc định**: `"mutual_info"`): Phương pháp feature selection
- `feature_selection_top_k` (int, **mặc định**: `25`): Số lượng features cần chọn

### Ví dụ khởi tạo

```python
from modules.deeplearning_data_pipeline import DeepLearningDataPipeline
from modules.DataFetcher import DataFetcher
from modules.ExchangeManager import ExchangeManager

# Setup
em = ExchangeManager()
data_fetcher = DataFetcher(em)

# Cách 1: Mặc định (fractional diff ON, triple barrier OFF, feature selection ON)
pipeline = DeepLearningDataPipeline(data_fetcher)

# Cách 2: Tùy chỉnh
pipeline = DeepLearningDataPipeline(
    data_fetcher=data_fetcher,
    use_fractional_diff=True,
    use_triple_barrier=True,  # Bật triple barrier
    triple_barrier_tp=0.03,  # 3% TP
    triple_barrier_sl=0.015,  # 1.5% SL
    use_feature_selection=True,
    feature_selection_method="boruta",
    feature_selection_top_k=30
)

# Cách 3: Không dùng fractional diff (nhanh hơn)
pipeline = DeepLearningDataPipeline(
    data_fetcher=data_fetcher,
    use_fractional_diff=False
)
```

---

## Phương thức chính

### `fetch_and_prepare(symbols, timeframe="1h", limit=1500, check_freshness=False) -> pd.DataFrame`

Lấy OHLCV data cho nhiều symbols và chuẩn bị cho deep learning.

**Tham số:**
- `symbols` (List[str]): Danh sách symbols (ví dụ: `["BTC/USDT", "ETH/USDT"]`)
- `timeframe` (str, **mặc định**: `"1h"`): Timeframe (ví dụ: `"1h"`, `"4h"`, `"1d"`)
- `limit` (int, **mặc định**: `1500`): Số lượng candles cần lấy
- `check_freshness` (bool, **mặc định**: `False`): Có kiểm tra độ tươi của data không

**Trả về:**
- `pd.DataFrame`: DataFrame đã được preprocess với tất cả features

**Ví dụ:**

```python
# Lấy data cho nhiều symbols
df = pipeline.fetch_and_prepare(
    symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
    timeframe="1h",
    limit=2000
)

print(df.columns)  # Xem tất cả features đã được tạo
print(df.head())
```

### `prepare_dataframe(df, timeframe="1h") -> pd.DataFrame`

Áp dụng full preprocessing pipeline cho một DataFrame đã có.

**Tham số:**
- `df` (pd.DataFrame): DataFrame với OHLCV data (phải có columns: `open`, `high`, `low`, `close`, `volume`, `timestamp`)
- `timeframe` (str, **mặc định**: `"1h"`): Timeframe để tính known-future features

**Trả về:**
- `pd.DataFrame`: DataFrame đã được preprocess

**Ví dụ:**

```python
# Nếu đã có DataFrame từ nguồn khác
df_raw = get_data_from_other_source()

# Preprocess
df_processed = pipeline.prepare_dataframe(df_raw, timeframe="4h")
```

### `split_chronological(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, gap=None, apply_feature_selection=True, target_col="future_log_return", task_type="regression") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`

Chia data chronologically thành train/validation/test sets.

**Tham số:**
- `df` (pd.DataFrame): Preprocessed DataFrame
- `train_ratio` (float, **mặc định**: `0.7`): Tỷ lệ training set
- `val_ratio` (float, **mặc định**: `0.15`): Tỷ lệ validation set
- `test_ratio` (float, **mặc định**: `0.15`): Tỷ lệ test set
- `gap` (int, **tùy chọn**): Gap giữa train và val/test (mặc định: `TARGET_HORIZON`)
- `apply_feature_selection` (bool, **mặc định**: `True`): Có áp dụng feature selection không
- `target_col` (str, **mặc định**: `"future_log_return"`): Target column name
- `task_type` (str, **mặc định**: `"regression"`): `"regression"` hoặc `"classification"`

**Trả về:**
- `Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`: (train_df, val_df, test_df)

**Ví dụ:**

```python
# Split với feature selection tự động
train_df, val_df, test_df = pipeline.split_chronological(
    df,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    apply_feature_selection=True,
    target_col="future_log_return",
    task_type="regression"
)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
```

### `apply_feature_selection(df, target_col="future_log_return", task_type="regression", symbol=None) -> pd.DataFrame`

Áp dụng feature selection cho DataFrame đã preprocess.

**Tham số:**
- `df` (pd.DataFrame): Preprocessed DataFrame
- `target_col` (str, **mặc định**: `"future_log_return"`): Target column name
- `task_type` (str, **mặc định**: `"regression"`): `"regression"` hoặc `"classification"`
- `symbol` (str, **tùy chọn**): Symbol name cho per-symbol selection

**Trả về:**
- `pd.DataFrame`: DataFrame chỉ chứa selected features

**Ví dụ:**

```python
# Áp dụng feature selection
df_selected = pipeline.apply_feature_selection(
    df,
    target_col="future_log_return",
    task_type="regression",
    symbol="BTC/USDT"
)
```

---

## Các thành phần

### 1. TripleBarrierLabeler

Triple Barrier Method cho robust labeling.

**Labels:**
- `1`: Take Profit hit (profit)
- `-1`: Stop Loss hit (loss)
- `0`: Time limit reached (neutral)
- `np.nan`: Insufficient future data

**Ví dụ:**

```python
from modules.deeplearning_data_pipeline import TripleBarrierLabeler

labeler = TripleBarrierLabeler(
    tp_threshold=0.02,  # 2% TP
    sl_threshold=0.01,  # 1% SL
    time_limit=24  # 24 candles
)

df = labeler.label(df, price_col="close")
print(df["triple_barrier_label"].value_counts())
```

### 2. FractionalDifferentiator

Fractional Differentiation để đảm bảo stationarity mà vẫn giữ memory.

**Công thức:**
```
X_t^d = sum_{k=0}^{window} (-1)^k * C(d, k) * X_{t-k}
```

**Ví dụ:**

```python
from modules.deeplearning_data_pipeline import FractionalDifferentiator

diff = FractionalDifferentiator(d=0.5, window=100)
df["close_frac_diff"] = diff.differentiate(df["close"])
```

### 3. Target Engineering

Pipeline tự động tạo các target variables:

- `log_return`: Log return giữa các candles
- `pct_change`: Percentage change
- `future_log_return`: Forward-looking log return (cho prediction)
- `future_pct_change`: Forward-looking percentage change
- `triple_barrier_label`: Triple barrier label (nếu enabled)

---

## Ví dụ sử dụng

### Ví dụ 1: Basic Workflow

```python
from modules.deeplearning_data_pipeline import DeepLearningDataPipeline
from modules.DataFetcher import DataFetcher
from modules.ExchangeManager import ExchangeManager

# Setup
em = ExchangeManager()
data_fetcher = DataFetcher(em)
pipeline = DeepLearningDataPipeline(data_fetcher)

# Fetch và prepare
df = pipeline.fetch_and_prepare(
    symbols=["BTC/USDT"],
    timeframe="1h",
    limit=2000
)

# Split
train_df, val_df, test_df = pipeline.split_chronological(df)

print(f"Features: {len(train_df.columns)}")
print(f"Train samples: {len(train_df)}")
```

### Ví dụ 2: Với Triple Barrier

```python
# Bật triple barrier
pipeline = DeepLearningDataPipeline(
    data_fetcher=data_fetcher,
    use_triple_barrier=True,
    triple_barrier_tp=0.03,  # 3% TP
    triple_barrier_sl=0.015   # 1.5% SL
)

df = pipeline.fetch_and_prepare(symbols=["BTC/USDT"], timeframe="1h")
train_df, val_df, test_df = pipeline.split_chronological(
    df,
    target_col="triple_barrier_label",  # Dùng triple barrier label
    task_type="classification"
)
```

### Ví dụ 3: Multi-asset Training

```python
# Fetch nhiều symbols
df = pipeline.fetch_and_prepare(
    symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"],
    timeframe="4h",
    limit=1500
)

# Split (normalization đã được áp dụng per symbol)
train_df, val_df, test_df = pipeline.split_chronological(df)

# Kiểm tra per symbol
for symbol in df["symbol"].unique():
    symbol_train = train_df[train_df["symbol"] == symbol]
    print(f"{symbol}: {len(symbol_train)} samples")
```

### Ví dụ 4: Tùy chỉnh Feature Selection

```python
# Tùy chỉnh feature selection
pipeline = DeepLearningDataPipeline(
    data_fetcher=data_fetcher,
    use_feature_selection=True,
    feature_selection_method="boruta",
    feature_selection_top_k=30,
    feature_collinearity_threshold=0.8
)

df = pipeline.fetch_and_prepare(symbols=["BTC/USDT"])
train_df, val_df, test_df = pipeline.split_chronological(df)

# Xem selected features
if pipeline.feature_selector:
    print(f"Selected {len(pipeline.feature_selector.selected_features)} features")
    print(pipeline.feature_selector.selected_features)
```

---

## Best Practices

### 1. Data Quality

- **Kiểm tra data trước khi split:**
```python
df = pipeline.fetch_and_prepare(symbols=["BTC/USDT"])
print(df.isna().sum())  # Kiểm tra missing values
print(df.describe())     # Kiểm tra statistics
```

### 2. Normalization

- Normalization được áp dụng **per symbol** tự động
- Scaler parameters được lưu vào `artifacts/deep/scalers/`
- Có thể load lại bằng `pipeline.load_scaler(symbol)`

### 3. Feature Selection

- Feature selection được áp dụng trên **training set** only
- Kết quả được áp dụng cho validation và test sets
- Selection được lưu để tái sử dụng

### 4. Chronological Split

- **Luôn dùng chronological split** cho time series
- Gap được tự động thêm để tránh data leakage
- Gap = `TARGET_HORIZON` (mặc định 24 candles)

### 5. Multi-asset Training

- Pipeline hỗ trợ multi-asset training
- Normalization per symbol đảm bảo scale consistency
- Feature selection có thể per-symbol hoặc global

---

## Pipeline Steps

Pipeline thực hiện các bước sau (trong `prepare_dataframe()`):

### Step 1: Target Engineering
- Tính `log_return`, `pct_change`
- Tính `future_log_return`, `future_pct_change`
- Áp dụng Triple Barrier Method (nếu enabled)

### Step 2: Fractional Differentiation
- Áp dụng cho price columns (`open`, `high`, `low`, `close`)
- Tạo columns: `{col}_frac_diff`

### Step 3: Technical Indicators
- Sử dụng IndicatorEngine với `DEEP_LEARNING` profile
- Thêm volatility metrics (`volatility_20`, `volatility_50`)

### Step 4: Known-future Features
- Time-of-day: `hour_sin`, `hour_cos`
- Day-of-week: `day_sin`, `day_cos`
- Day-of-month: `day_of_month_sin`, `day_of_month_cos`
- Funding schedule: `hours_to_funding`, `is_funding_time`
- Candle index: `candle_index`

### Step 5: Normalization
- Per-symbol StandardScaler
- Lưu scaler parameters
- Exclude: targets, labels, timestamps, cyclical features

### Step 6: Feature Selection (trong split_chronological)
- Áp dụng trên training set
- Lưu kết quả
- Áp dụng cho validation và test sets

---

## Configuration

Các config constants trong `modules/config.py`:

```python
# Triple Barrier
DEEP_TRIPLE_BARRIER_TP_THRESHOLD = 0.02  # 2%
DEEP_TRIPLE_BARRIER_SL_THRESHOLD = 0.01  # 1%

# Fractional Differentiation
DEEP_FRACTIONAL_DIFF_D = 0.5
DEEP_FRACTIONAL_DIFF_WINDOW = 100
DEEP_USE_FRACTIONAL_DIFF = True

# Feature Selection
DEEP_USE_FEATURE_SELECTION = True
DEEP_FEATURE_SELECTION_METHOD = "mutual_info"
DEEP_FEATURE_SELECTION_TOP_K = 25
DEEP_FEATURE_COLLINEARITY_THRESHOLD = 0.85

# Data Split
DEEP_TRAIN_RATIO = 0.7
DEEP_VAL_RATIO = 0.15
DEEP_TEST_RATIO = 0.15
```

---

## Troubleshooting

### Lỗi: "No data fetched for any symbol"

**Nguyên nhân:** Không fetch được data từ bất kỳ symbol nào

**Giải pháp:**
- Kiểm tra kết nối internet
- Kiểm tra symbol names (format: `"BTC/USDT"`)
- Kiểm tra ExchangeManager có hoạt động không

### Lỗi: "Target column not found"

**Nguyên nhân:** Target column không tồn tại trong DataFrame

**Giải pháp:**
- Kiểm tra `target_col` parameter
- Đảm bảo đã gọi `prepare_dataframe()` trước
- Với triple barrier, dùng `"triple_barrier_label"`

### Normalization issues

**Nguyên nhân:** NaN values sau normalization

**Giải pháp:**
- Kiểm tra data quality trước normalization
- Đảm bảo có đủ data per symbol
- Kiểm tra có constant columns không

---

## Tham khảo

- [Temporal Fusion Transformer Paper](https://arxiv.org/abs/1912.09363)
- [Fractional Differentiation](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
- [Triple Barrier Method](https://www.quantresearch.org/TripleBarrierMethod.pdf)

