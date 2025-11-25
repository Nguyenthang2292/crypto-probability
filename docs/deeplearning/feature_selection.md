# 📚 Feature Selection Documentation

## Mục lục
1. [Tổng quan](#tổng-quan)
2. [Khởi tạo](#khởi-tạo)
3. [Phương thức](#phương-thức)
4. [Ví dụ sử dụng](#ví-dụ-sử-dụng)
5. [Best Practices](#best-practices)
6. [Các phương pháp Feature Selection](#các-phương-pháp-feature-selection)

---

## Tổng quan

`FeatureSelector` là một module toàn diện để chọn lọc và kỹ thuật hóa features cho deep learning models. Module này cung cấp:

- ✅ **Mutual Information Selection** - Chọn features dựa trên mutual information với target
- ✅ **Boruta-like Selection** - Sử dụng Random Forest importance để chọn features
- ✅ **F-test Selection** - Sử dụng ANOVA F-statistic
- ✅ **Combined Method** - Kết hợp Mutual Information và Boruta
- ✅ **Collinearity Removal** - Loại bỏ features có correlation cao để cải thiện tính ổn định của model
- ✅ **Feature Filtering** - Tự động loại bỏ invalid features (non-numeric, constant, target leakage)
- ✅ **Persistent Storage** - Lưu và tải kết quả feature selection để tái sử dụng

### Khi nào dùng FeatureSelector?

| Mục đích | Dùng FeatureSelector? | Phương thức |
|----------|----------------------|-------------|
| Chọn top 20-30 features quan trọng nhất | ✅ Có | `select_features()` |
| Loại bỏ features có correlation cao | ✅ Có | Tự động trong `select_features()` |
| Tránh "Garbage In, Garbage Out" | ✅ Có | Tự động filter invalid features |
| Lưu kết quả để tái sử dụng | ✅ Có | `_save_selection()` / `load_selection()` |
| Áp dụng selection đã lưu cho data mới | ✅ Có | `apply_selection()` |
| Xem feature importance scores | ✅ Có | `get_feature_importance_report()` |

---

## Khởi tạo

### Cú pháp

```python
from modules.feature_selection import FeatureSelector

selector = FeatureSelector(
    method="mutual_info",  # 'mutual_info', 'boruta', 'f_test', or 'combined'
    top_k=25,  # Số lượng features cần chọn
    collinearity_threshold=0.85,  # Ngưỡng correlation để loại bỏ
    selection_dir="artifacts/deep/feature_selection"  # Thư mục lưu kết quả
)
```

### Tham số

- `method` (str, **mặc định**: `"mutual_info"`): Phương pháp chọn features
  - `"mutual_info"`: Mutual Information (khuyến nghị cho hầu hết trường hợp)
  - `"boruta"`: Random Forest importance (tốt cho non-linear relationships)
  - `"f_test"`: ANOVA F-statistic (nhanh, tốt cho linear relationships)
  - `"combined"`: Kết hợp Mutual Information và Boruta
- `top_k` (int, **mặc định**: `25`): Số lượng top features cần chọn (20-30 khuyến nghị)
- `collinearity_threshold` (float, **mặc định**: `0.85`): Ngưỡng correlation để loại bỏ collinear features (0.8-0.95)
- `selection_dir` (str, **mặc định**: `"artifacts/deep/feature_selection"`): Thư mục lưu/load kết quả

### Ví dụ khởi tạo

```python
from modules.feature_selection import FeatureSelector

# Cách 1: Sử dụng mặc định (mutual_info, top 25)
selector = FeatureSelector()

# Cách 2: Tùy chỉnh phương pháp và số lượng
selector = FeatureSelector(
    method="boruta",
    top_k=30,
    collinearity_threshold=0.9
)

# Cách 3: Sử dụng combined method
selector = FeatureSelector(
    method="combined",
    top_k=20
)
```

### Thuộc tính

Sau khi khởi tạo, `FeatureSelector` có các thuộc tính:

- `method`: Phương pháp chọn features đã chọn
- `top_k`: Số lượng features cần chọn
- `collinearity_threshold`: Ngưỡng correlation
- `selected_features`: Danh sách features đã chọn (sau khi gọi `select_features()`)
- `feature_scores`: Dictionary chứa scores của tất cả features
- `selection_metadata`: Metadata của selection (method, top_k, etc.)

---

## Phương thức

### `select_features(X, y, task_type="regression", symbol=None) -> pd.DataFrame`

Chọn top features sử dụng phương pháp đã chỉ định.

**Tham số:**
- `X` (pd.DataFrame): DataFrame chứa features
- `y` (pd.Series): Target Series (continuous cho regression, discrete cho classification)
- `task_type` (str, **mặc định**: `"regression"`): Loại task - `"regression"` hoặc `"classification"`
- `symbol` (str, **tùy chọn**): Tên symbol để lưu selection per-symbol

**Trả về:**
- `pd.DataFrame`: DataFrame chỉ chứa các features đã chọn

**Quy trình:**
1. **Filter invalid features**: Loại bỏ non-numeric, constant, target leakage columns
2. **Remove collinear features**: Loại bỏ features có correlation > threshold
3. **Apply selection method**: Chọn top K features
4. **Save results**: Lưu kết quả vào disk

**Ví dụ:**

```python
import pandas as pd
from modules.feature_selection import FeatureSelector

# Tạo selector
selector = FeatureSelector(method="mutual_info", top_k=25)

# Chọn features
X_selected = selector.select_features(
    X=train_features_df,
    y=train_target_series,
    task_type="regression",
    symbol="BTC/USDT"
)

print(f"Selected {len(selector.selected_features)} features")
print(selector.selected_features)
```

### `apply_selection(X) -> pd.DataFrame`

Áp dụng selection đã có (từ `select_features()` hoặc `load_selection()`) cho DataFrame mới.

**Tham số:**
- `X` (pd.DataFrame): DataFrame mới cần áp dụng selection

**Trả về:**
- `pd.DataFrame`: DataFrame chỉ chứa selected features

**Lưu ý:** Phải gọi `select_features()` hoặc `load_selection()` trước.

**Ví dụ:**

```python
# Đã có selection từ trước
selector.load_selection(symbol="BTC/USDT")

# Áp dụng cho validation/test data
X_val_selected = selector.apply_selection(X_val)
X_test_selected = selector.apply_selection(X_test)
```

### `load_selection(symbol=None) -> Optional[Dict]`

Load kết quả feature selection đã lưu từ disk.

**Tham số:**
- `symbol` (str, **tùy chọn**): Tên symbol (nếu lưu per-symbol)

**Trả về:**
- `Optional[Dict]`: Metadata của selection hoặc `None` nếu không tìm thấy

**Ví dụ:**

```python
metadata = selector.load_selection(symbol="BTC/USDT")
if metadata:
    print(f"Loaded {len(selector.selected_features)} features")
    print(f"Method: {metadata['method']}")
```

### `get_feature_importance_report() -> pd.DataFrame`

Lấy báo cáo feature importance scores.

**Trả về:**
- `pd.DataFrame`: DataFrame với columns: `feature`, `score`, `selected`

**Ví dụ:**

```python
report = selector.get_feature_importance_report()
print(report.head(10))  # Top 10 features

# Lọc chỉ selected features
selected_report = report[report["selected"] == True]
print(selected_report)
```

---

## Ví dụ sử dụng

### Ví dụ 1: Basic Feature Selection

```python
from modules.feature_selection import FeatureSelector
import pandas as pd

# Khởi tạo
selector = FeatureSelector(
    method="mutual_info",
    top_k=25,
    collinearity_threshold=0.85
)

# Chọn features
X_selected = selector.select_features(
    X=train_X,
    y=train_y,
    task_type="regression"
)

# Xem kết quả
print(f"Selected {len(selector.selected_features)} features")
print(selector.selected_features[:10])  # Top 10
```

### Ví dụ 2: Sử dụng với Classification

```python
selector = FeatureSelector(method="boruta", top_k=30)

# Classification task
X_selected = selector.select_features(
    X=train_X,
    y=train_labels,  # Categorical labels
    task_type="classification",
    symbol="BTC/USDT"
)

# Xem importance report
report = selector.get_feature_importance_report()
print(report.sort_values("score", ascending=False).head(15))
```

### Ví dụ 3: Load và Apply Selection

```python
# Load selection đã lưu
selector = FeatureSelector()
metadata = selector.load_selection(symbol="BTC/USDT")

if metadata:
    # Áp dụng cho validation và test sets
    X_val_selected = selector.apply_selection(X_val)
    X_test_selected = selector.apply_selection(X_test)
else:
    print("No saved selection found. Run select_features() first.")
```

### Ví dụ 4: Combined Method

```python
# Sử dụng combined method (Mutual Info + Boruta)
selector = FeatureSelector(
    method="combined",
    top_k=20
)

X_selected = selector.select_features(
    X=train_X,
    y=train_y,
    task_type="regression"
)

# Combined method sẽ normalize và average scores từ cả 2 methods
```

---

## Best Practices

### 1. Chọn phương pháp phù hợp

| Phương pháp | Khi nào dùng | Ưu điểm | Nhược điểm |
|------------|--------------|---------|------------|
| `mutual_info` | Hầu hết trường hợp | Không giả định linear, nhanh | Có thể miss non-linear patterns phức tạp |
| `boruta` | Non-linear relationships | Phát hiện interactions tốt | Chậm hơn (cần train RF) |
| `f_test` | Linear relationships | Rất nhanh | Chỉ phát hiện linear relationships |
| `combined` | Muốn kết hợp ưu điểm | Cân bằng giữa các methods | Chậm nhất |

### 2. Số lượng features (top_k)

- **20-30 features**: Khuyến nghị cho hầu hết trường hợp
- **< 20**: Có thể miss important features
- **> 30**: Có thể gây overfitting, tăng training time

### 3. Collinearity threshold

- **0.85 (mặc định)**: Cân bằng tốt
- **0.8**: Loại bỏ nhiều hơn (nếu có quá nhiều correlated features)
- **0.9-0.95**: Chỉ loại bỏ highly correlated (giữ lại nhiều features hơn)

### 4. Per-symbol selection

Nếu training multi-asset, nên lưu selection per-symbol:

```python
for symbol in symbols:
    selector.select_features(
        X=symbol_X,
        y=symbol_y,
        symbol=symbol  # Lưu per-symbol
    )
```

### 5. Validation workflow

```python
# 1. Select trên training set
X_train_selected = selector.select_features(X_train, y_train)

# 2. Lưu selection
# (Tự động trong select_features)

# 3. Load và apply cho validation/test
selector.load_selection()
X_val_selected = selector.apply_selection(X_val)
X_test_selected = selector.apply_selection(X_test)
```

---

## Các phương pháp Feature Selection

### 1. Mutual Information

**Cách hoạt động:**
- Đo lường mutual information giữa mỗi feature và target
- Chọn K features có mutual information cao nhất
- Không giả định linear relationship

**Ưu điểm:**
- Nhanh
- Phát hiện cả linear và non-linear relationships
- Không cần train model

**Nhược điểm:**
- Có thể miss complex interactions

### 2. Boruta-like (Random Forest)

**Cách hoạt động:**
- Train Random Forest model
- Sử dụng feature importance scores
- Chọn K features có importance cao nhất

**Ưu điểm:**
- Phát hiện non-linear relationships tốt
- Phát hiện feature interactions
- Robust với noise

**Nhược điểm:**
- Chậm hơn (cần train RF)
- Có thể overfit nếu RF overfit

### 3. F-test (ANOVA)

**Cách hoạt động:**
- Tính F-statistic (ANOVA) giữa features và target
- Chọn K features có F-statistic cao nhất
- Giả định linear relationship

**Ưu điểm:**
- Rất nhanh
- Tốt cho linear relationships

**Nhược điểm:**
- Chỉ phát hiện linear relationships
- Miss non-linear patterns

### 4. Combined

**Cách hoạt động:**
- Chạy cả Mutual Information và Boruta
- Normalize scores từ cả 2 methods
- Average normalized scores
- Chọn K features có combined score cao nhất

**Ưu điểm:**
- Kết hợp ưu điểm của cả 2 methods
- Robust hơn

**Nhược điểm:**
- Chậm nhất (cần train RF + tính MI)

---

## Tự động Filter Invalid Features

Module tự động loại bỏ:

1. **Non-numeric columns**: Chỉ giữ numeric features
2. **Columns với >50% NaN**: Loại bỏ columns có quá nhiều missing values
3. **Constant columns**: Loại bỏ columns có zero variance
4. **Target leakage columns**: Loại bỏ columns chứa future information
   - Columns có chứa: `future_`, `target`, `label`, `triple_barrier`
5. **Metadata columns**: Loại bỏ `timestamp`, `symbol`, `time_idx`

---

## Lưu và Load Selection

### Lưu tự động

Khi gọi `select_features()`, kết quả tự động được lưu vào:
```
artifacts/deep/feature_selection/feature_selection_{symbol}.json
```

### Load selection

```python
selector = FeatureSelector()
metadata = selector.load_selection(symbol="BTC/USDT")

# Sau khi load, có thể dùng apply_selection()
X_new_selected = selector.apply_selection(X_new)
```

### Format của saved file

```json
{
  "method": "mutual_info",
  "top_k": 25,
  "collinearity_threshold": 0.85,
  "selected_features": ["feature1", "feature2", ...],
  "feature_scores": {
    "feature1": 0.85,
    "feature2": 0.72,
    ...
  }
}
```

---

## Tích hợp với DeepLearningDataPipeline

Feature selection được tích hợp tự động trong `DeepLearningDataPipeline`:

```python
from modules.deeplearning_data_pipeline import DeepLearningDataPipeline

# Feature selection tự động được áp dụng trong split_chronological()
pipeline = DeepLearningDataPipeline(data_fetcher)
df = pipeline.fetch_and_prepare(symbols=["BTC/USDT"])
train_df, val_df, test_df = pipeline.split_chronological(
    df,
    apply_feature_selection=True  # Mặc định True
)
```

---

## Troubleshooting

### Lỗi: "No valid features after filtering"

**Nguyên nhân:** Tất cả features đều bị loại bỏ (constant, NaN, etc.)

**Giải pháp:**
- Kiểm tra data quality
- Giảm `nan_threshold` trong `_filter_invalid_features()`
- Kiểm tra có features nào valid không

### Lỗi: "No features selected"

**Nguyên nhân:** Chưa gọi `select_features()` hoặc `load_selection()`

**Giải pháp:**
```python
# Phải gọi select_features() trước
selector.select_features(X, y)
# Sau đó mới dùng apply_selection()
X_selected = selector.apply_selection(X_new)
```

### Selection quá ít features

**Nguyên nhân:** `top_k` quá nhỏ hoặc `collinearity_threshold` quá cao

**Giải pháp:**
- Tăng `top_k` (ví dụ: 25 → 30)
- Giảm `collinearity_threshold` (ví dụ: 0.85 → 0.8)

---

## Tham khảo

- [scikit-learn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Mutual Information](https://en.wikipedia.org/wiki/Mutual_information)
- [Boruta Algorithm](https://www.jstatsoft.org/article/view/v036i11)

