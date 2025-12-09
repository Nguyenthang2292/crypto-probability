# Giải thích TARGET_HORIZON

## Định nghĩa

`TARGET_HORIZON = 24` (được định nghĩa trong `modules/config.py`)

Đây là số lượng **candles (nến)** trong tương lai mà model sẽ dự đoán. Giá trị mặc định là **24 candles**.

## Cách sử dụng trong hệ thống

### 1. **Tạo Labels (Labeling) - `modules/labeling.py`**

`TARGET_HORIZON` được sử dụng để tạo target labels dựa trên giá tương lai:

```python
# Lấy giá đóng cửa sau TARGET_HORIZON candles
future_close = df["close"].shift(-TARGET_HORIZON)  # shift(-24) = lùi về sau 24 candles

# Tính phần trăm thay đổi giá
pct_change = (future_close - df["close"]) / df["close"]

# Lấy giá đóng cửa trước đó TARGET_HORIZON candles để tính threshold động
historical_ref = df["close"].shift(TARGET_HORIZON)  # shift(24) = lùi về trước 24 candles
```

**Ví dụ với TARGET_HORIZON = 24:**
- Tại candle thứ 100, model sẽ xem giá tại candle thứ 124 (100 + 24) để tạo label
- Nếu giá tại candle 124 > giá tại candle 100 + threshold → Label = "UP"
- Nếu giá tại candle 124 < giá tại candle 100 - threshold → Label = "DOWN"
- Ngược lại → Label = "NEUTRAL"

### 2. **Ngăn chặn Data Leakage (Model Training) - `modules/model.py`**

`TARGET_HORIZON` được sử dụng để tạo **gap (khoảng trống)** giữa training set và test set:

```python
# Train/Test split với gap
split = int(len(df) * 0.8)  # 80% data
train_end = split - TARGET_HORIZON  # Kết thúc train trước TARGET_HORIZON candles
test_start = split  # Bắt đầu test sau gap
```

**Tại sao cần gap?**

Khi tạo labels, mỗi row trong training set sử dụng giá từ **TARGET_HORIZON candles sau đó** để tạo label. Nếu không có gap:

```
❌ SAI (Data Leakage):
Train: [1, 2, 3, ..., 100]
Test:  [101, 102, 103, ...]
→ Row 100 trong train sử dụng giá từ row 124 để tạo label
→ Row 101-124 trong test đã được "nhìn thấy" khi tạo label cho row 100
→ Model đã "biết trước" dữ liệu test → Data Leakage!

✅ ĐÚNG (Có gap):
Train: [1, 2, 3, ..., 76]  (kết thúc tại 80 - 24 = 76)
Gap:   [77, 78, 79, ..., 100]  (24 candles gap)
Test:  [101, 102, 103, ...]
→ Row 76 trong train sử dụng giá từ row 100 để tạo label
→ Row 101+ trong test hoàn toàn độc lập, không bị leak
```

### 3. **Cross-Validation với Gap - `modules/model.py`**

Trong cross-validation, cũng cần tạo gap tương tự:

```python
# Loại bỏ TARGET_HORIZON indices cuối cùng từ training set
train_idx_filtered = train_idx_array[:-TARGET_HORIZON]

# Đảm bảo test set bắt đầu sau gap
min_test_start = train_idx_filtered[-1] + TARGET_HORIZON + 1
```

### 4. **Hiển thị trong Output - `main.py`**

`TARGET_HORIZON` được hiển thị trong prediction context:

```python
prediction_context = f"{prediction_window} | {TARGET_HORIZON} candles >={threshold_value*100:.2f}% move"
# Ví dụ: "24h | 24 candles >=1.50% move"
```

## Ví dụ cụ thể

### Với timeframe = "1h" và TARGET_HORIZON = 24:

- Model dự đoán giá sau **24 giờ** (24 candles × 1h = 24h)
- Tại thời điểm hiện tại, model sẽ dự đoán giá sau 24 giờ nữa
- Labels được tạo bằng cách so sánh giá hiện tại với giá sau 24 candles

### Với timeframe = "4h" và TARGET_HORIZON = 24:

- Model dự đoán giá sau **96 giờ** (24 candles × 4h = 96h = 4 ngày)
- Tại thời điểm hiện tại, model sẽ dự đoán giá sau 4 ngày nữa

## Tác động của việc thay đổi TARGET_HORIZON

### Tăng TARGET_HORIZON (ví dụ: 24 → 48):
- ✅ Dự đoán xa hơn trong tương lai
- ❌ Cần nhiều dữ liệu hơn (mất thêm 48 rows do gap)
- ❌ Labels có thể kém chính xác hơn (dự đoán xa hơn = khó hơn)
- ❌ Mất nhiều dữ liệu hơn ở cuối dataset (không thể tạo label cho 48 candles cuối)

### Giảm TARGET_HORIZON (ví dụ: 24 → 12):
- ✅ Cần ít dữ liệu hơn
- ✅ Dự đoán gần hơn (dễ dự đoán hơn)
- ❌ Dự đoán ngắn hạn hơn
- ✅ Ít mất dữ liệu hơn (chỉ mất 12 rows cuối)

## Khuyến nghị

- **TARGET_HORIZON = 24** là giá trị hợp lý cho hầu hết các timeframe
- Với timeframe ngắn (30m, 1h): có thể giữ 24 hoặc tăng lên 48
- Với timeframe dài (4h, 1d): có thể giảm xuống 12 hoặc 6
- Đảm bảo có đủ dữ liệu: cần ít nhất `TARGET_HORIZON * 2 + 200` rows để có đủ data sau khi tạo gap và tính indicators

## Công thức tính số dữ liệu tối thiểu

```
Minimum rows needed = TARGET_HORIZON + TARGET_HORIZON + 200
                    = 2 * TARGET_HORIZON + 200
                    = Gap + Training data + Indicators requirement
```

Với `TARGET_HORIZON = 24`:
- Minimum = 2 × 24 + 200 = **248 rows**
- Khuyến nghị: **500+ rows** để có đủ dữ liệu training tốt

