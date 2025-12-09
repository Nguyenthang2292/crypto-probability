# Cải Thiện Strategy 5: Combined Strategy

## Tổng Quan

File `modules/range_oscillator/strategies/combined.py` đã được cải thiện và mở rộng với nhiều tính năng mới, giữ nguyên backward compatibility.

## Các Tính Năng Mới

### 1. **Hỗ Trợ Tất Cả Strategies (2-9)**
   - **Trước**: Chỉ hỗ trợ 3 strategies (Sustained, Crossover, Momentum)
   - **Sau**: Hỗ trợ tất cả 7 strategies:
     - Strategy 2: Sustained Pressure
     - Strategy 3: Zero Line Crossover
     - Strategy 4: Momentum
     - Strategy 6: Range Breakouts (MỚI)
     - Strategy 7: Divergence Detection (MỚI)
     - Strategy 8: Trend Following (MỚI)
     - Strategy 9: Mean Reversion (MỚI)

### 2. **Nhiều Chế Độ Consensus**
   - **"threshold"** (mặc định): Yêu cầu một phần trăm nhất định strategies đồng ý (theo `consensus_threshold`)
     - Default `consensus_threshold=0.5` (ít nhất 50% strategies phải đồng ý)
     - Có thể điều chỉnh từ 0.0 đến 1.0 để yêu cầu ít/nhiều strategies hơn
   - **"weighted"**: Bỏ phiếu có trọng số dựa trên `strategy_weights`

### 3. **Hệ Thống Trọng Số (Weighting System)**
   - Cho phép đặt trọng số khác nhau cho từng strategy
   - Tự động normalize weights
   - Sử dụng với consensus mode "weighted"

### 4. **Lọc Signal Theo Strength**
   - Tham số `min_signal_strength`: Chỉ chấp nhận signals có strength >= threshold
   - Giúp loại bỏ signals yếu, chỉ giữ lại signals mạnh

### 5. **Lựa Chọn Strategies Linh Hoạt**
   - Cách 1: Sử dụng `enabled_strategies` list (ví dụ: `[2, 3, 4, 6]`)
   - Cách 2: Sử dụng các flags `use_*` (backward compatible)
   - Có thể kết hợp cả hai cách

### 6. **Thống Kê Strategy Contributions**
   - Tham số `return_strategy_stats=True`: Trả về thống kê về contribution của từng strategy
   - Bao gồm số lượng LONG/SHORT signals từ mỗi strategy
   - Hữu ích cho phân tích và tối ưu hóa

### 7. **Parameters Tùy Chỉnh Cho Từng Strategy**
   - Tất cả parameters của các strategies đều có thể tùy chỉnh
   - Ví dụ: `breakout_upper_threshold`, `divergence_lookback_period`, etc.

### 8. **Dynamic Strategy Selection** (MỚI)
   - Tự động chọn strategies dựa trên market conditions
   - Phân tích volatility, trend strength, range-bound vs trending
   - High volatility → Breakout, Divergence strategies
   - Trending market → Crossover, Momentum, Breakout, Trend Following
   - Range-bound market → Sustained, Divergence, Mean Reversion
   - ✅ **Có thể sử dụng cùng Adaptive Weights**: Dynamic Selection chọn strategies, sau đó Adaptive Weights điều chỉnh weights của chúng

### 9. **Adaptive Weights** (MỚI - Đã Cải Thiện)
   - Tự động điều chỉnh weights dựa trên **actual price movement accuracy** (không phải agreement với consensus)
   - Tránh circular logic và groupthink: Đánh giá dựa trên độ chính xác thực tế so với thị trường
   - Logic:
     - **Accuracy (70%)**: Nếu strategy tạo LONG signal, kiểm tra xem giá có thực sự tăng trong N bars tiếp theo không
     - **Strength (30%)**: Average strength của các signals đã đúng
   - Chỉ hoạt động với `consensus_mode="weighted"` và yêu cầu `close` prices
   - Tự động normalize weights
   - ✅ **Có thể sử dụng cùng Dynamic Selection**: Adaptive weights sẽ điều chỉnh weights của các strategies đã được Dynamic Selection chọn

### 10. **Signal Confidence Score** (MỚI)
   - Tính toán confidence score (0.0 đến 1.0) dựa trên:
     - Agreement level: fraction của strategies đồng ý (60% weight)
     - Signal strength: average strength của agreeing strategies (40% weight)
   - Trả về thêm confidence score series khi `return_confidence_score=True`

## Ví Dụ Sử Dụng

### Ví Dụ 1: Sử Dụng Cơ Bản (Backward Compatible)
```python
signals, strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    use_sustained=True,
    use_crossover=True,
    use_momentum=True
)
```

### Ví Dụ 2: Sử Dụng Tất Cả Strategies (Threshold Mode - Default)
```python
signals, strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6, 7, 8, 9],  # Tất cả strategies
    # consensus_mode="threshold" là mặc định, consensus_threshold=0.5
)
```

### Ví Dụ 3: Threshold Mode với Custom Threshold
```python
signals, strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6, 7],
    consensus_mode="threshold",
    consensus_threshold=0.6  # Ít nhất 60% strategies phải đồng ý
)
```

### Ví Dụ 4: Weighted Voting
```python
signals, strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6],
    consensus_mode="weighted",
    strategy_weights={
        2: 1.0,  # Sustained: weight 1.0
        3: 1.5,  # Crossover: weight 1.5 (quan trọng hơn)
        4: 0.8,  # Momentum: weight 0.8
        6: 1.2,  # Breakout: weight 1.2
    }
)
```

### Ví Dụ 5: Threshold Mode - Strict (Yêu Cầu Nhiều Strategies)
```python
signals, strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6, 7, 8],
    consensus_mode="threshold",
    consensus_threshold=0.75  # Ít nhất 75% strategies phải đồng ý (rất strict)
)
```

### Ví Dụ 6: Lọc Signal Strength
```python
signals, strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4],
    min_signal_strength=0.3  # Chỉ chấp nhận signals có strength >= 0.3
)
```

### Ví Dụ 7: Lấy Thống Kê
```python
signals, strength, stats = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6],
    return_strategy_stats=True
)

# stats sẽ chứa:
# {
#     2: {"name": "Sustained", "long_count": 10, "short_count": 5},
#     3: {"name": "Crossover", "long_count": 8, "short_count": 7},
#     ...
# }
```

### Ví Dụ 8: Tùy Chỉnh Parameters Cho Strategies
```python
signals, strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6],
    # Strategy 2 parameters
    min_bars_sustained=5,
    # Strategy 3 parameters
    confirmation_bars=3,
    # Strategy 4 parameters
    momentum_period=5,
    momentum_threshold=7.0,
    # Strategy 6 parameters
    breakout_upper_threshold=120.0,
    breakout_lower_threshold=-120.0,
)
```

### Ví Dụ 9: Dynamic Strategy Selection
```python
signals, strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6, 7, 8, 9],  # Tất cả strategies
    enable_dynamic_selection=True,  # Bật dynamic selection
    dynamic_selection_lookback=20,  # Phân tích 20 bars gần nhất
    dynamic_volatility_threshold=0.6,  # Threshold cho high volatility
    dynamic_trend_threshold=0.5,  # Threshold cho trending market
)
```

### Ví Dụ 10: Adaptive Weights
```python
signals, strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6],
    consensus_mode="weighted",
    enable_adaptive_weights=True,  # Bật adaptive weights
    adaptive_performance_window=10,  # Tính performance từ 10 bars gần nhất
)
```

### Ví Dụ 11: Confidence Score
```python
signals, strength, confidence = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6],
    return_confidence_score=True,  # Trả về confidence score
)

# confidence là Series với giá trị 0.0 đến 1.0
# Giá trị cao = nhiều strategies đồng ý và signal strength cao
```

### Ví Dụ 12: Kết Hợp Tất Cả Tính Năng (Including Dynamic Selection + Adaptive Weights)
```python
signals, strength, stats, confidence = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6, 7, 8, 9],
    # Dynamic selection: Tự động chọn strategies dựa trên market conditions
    enable_dynamic_selection=True,
    dynamic_selection_lookback=20,
    # Adaptive weights: Tự động điều chỉnh weights của strategies đã được chọn
    consensus_mode="weighted",
    enable_adaptive_weights=True,
    adaptive_performance_window=10,
    # Confidence score
    return_confidence_score=True,
    # Stats
    return_strategy_stats=True,
)
```

**Lưu ý quan trọng:**
- ✅ **Dynamic Selection và Adaptive Weights CÓ THỂ sử dụng cùng nhau**
- Dynamic Selection chọn strategies nào sẽ được sử dụng (dựa trên market conditions)
- Adaptive Weights điều chỉnh weights của các strategies đã được chọn (dựa trên performance)
- Flow: Market Analysis → Dynamic Selection → Generate Signals → Adaptive Weights → Final Signals

## Backward Compatibility

✅ **Hoàn toàn tương thích ngược**:
- Tất cả code cũ vẫn hoạt động bình thường
- Các tham số mặc định giữ nguyên behavior cũ
- Function signature không thay đổi (trừ khi sử dụng `return_strategy_stats=True`)

## Cải Thiện Performance

- Sử dụng vectorized operations với NumPy
- Tối ưu hóa memory usage
- Xử lý lỗi tốt hơn (graceful fallback khi strategy fail)

## Error Handling

- Validation parameters tốt hơn
- Graceful handling khi strategy fail (skip và tiếp tục)
- Fallback về basic strategy nếu tất cả strategies fail

## Documentation

- Docstring chi tiết và đầy đủ
- Comments giải thích logic
- Type hints đầy đủ

## Migration Guide

### Thay Đổi Gần Đây (Simplification)

**Đã loại bỏ hoàn toàn các Consensus Modes cũ:**
- ❌ **Đã xóa**: `consensus_mode="majority"` (không còn hỗ trợ)
- ❌ **Đã xóa**: `consensus_mode="unanimous"` (không còn hỗ trợ)
- ✅ **Giữ lại**: `consensus_mode="threshold"` (mặc định, với `consensus_threshold=0.5`)
- ✅ **Giữ lại**: `consensus_mode="weighted"`

**Migration cho code cũ:**
```python
# Code cũ (sẽ raise ValueError)
consensus_mode="majority"  # ❌ Không còn hoạt động

# Code mới (khuyến nghị)
consensus_mode="threshold"  # mặc định
consensus_threshold=0.5     # = 50% strategies phải đồng ý

# Hoặc cho unanimous behavior:
consensus_mode="threshold"
consensus_threshold=1.0     # = 100% strategies phải đồng ý
```

**Breaking Changes:**
- Code sử dụng `consensus_mode="majority"` hoặc `"unanimous"` sẽ raise `ValueError`
- Cần cập nhật code để sử dụng `consensus_mode="threshold"` với `consensus_threshold` phù hợp

## Các Cải Thiện Gần Đây (Latest Updates)

### 1. **Loại Bỏ Deprecated Consensus Modes**
   - ✅ Đã loại bỏ hoàn toàn `"majority"` và `"unanimous"` (không còn backward compatibility)
   - ✅ Chỉ hỗ trợ `"threshold"` và `"weighted"`
   - ✅ Validation rõ ràng với error messages khi sử dụng giá trị không hợp lệ
   - ✅ Code sử dụng deprecated values sẽ raise `ValueError` ngay lập tức

### 2. **Cải Thiện Threshold Voting Logic**
   - ✅ Sử dụng `ceil(n * threshold)` để tính min_agreement
   - ✅ Với 4 strategies và threshold=0.5: cần >= 2 votes
   - ✅ Giữ check `long_votes > short_votes` để đảm bảo NO_SIGNAL khi votes bằng nhau

### 3. **Cải Thiện Python Compatibility**
   - ✅ Loại bỏ `strict=True` trong `zip()` để tương thích với Python < 3.10
   - ✅ Code giờ chạy được trên Python 3.8+

### 4. **Cải Thiện Error Handling và Validation**
   - ✅ **adaptive_trend/equity.py**: Thêm validation đầy đủ, logging, xử lý NaN
   - ✅ **adaptive_trend/layer1.py**: 
     - Sửa `weighted_signal` để preserve tất cả indices (union thay vì intersection)
     - Thêm validation, logging, xử lý edge cases
   - ✅ **adaptive_trend/moving_averages.py**: 
     - Raise error ngay khi có MA calculation thất bại (không return partial tuple)
     - Thêm validation, logging, xử lý lỗi
   - ✅ **adaptive_trend/signals.py**: Thêm validation, logging, xử lý NaN và index alignment
   - ✅ **adaptive_trend/utils.py**: Thêm validation, logging, xử lý overflow
   - ✅ **adaptive_trend/scanner.py**: Thêm validation đầy đủ, tracking errors, summary logging

### 5. **Cải Thiện Code Quality**
   - ✅ Tất cả modules có validation đầu vào đầy đủ
   - ✅ Logging nhất quán từ `modules.common.utils`
   - ✅ Error messages rõ ràng và bằng tiếng Anh
   - ✅ Documentation đầy đủ với `Raises` sections
   - ✅ Xử lý edge cases tốt hơn (NaN, empty series, index mismatches)

### 6. **Cải Thiện Performance và Reliability**
   - ✅ Tối ưu code với list comprehensions thay vì duplication
   - ✅ Xử lý overflow trong exponential calculations
   - ✅ Tự động align indices khi cần thiết
   - ✅ Early error detection và reporting

## Bug Fixes và Improvements

### Fixed Issues

1. **Weighted Signal Index Preservation** (`layer1.py`)
   - ✅ Sửa logic để preserve tất cả indices từ tất cả pairs (union thay vì intersection)
   - ✅ Tránh mất indices hợp lệ khi các pairs có indices khác nhau

2. **Threshold Voting Logic** (`combined.py`)
   - ✅ Sử dụng `ceil(n * threshold)` để tính min_agreement
   - ✅ Logic rõ ràng và đơn giản hơn

3. **Partial MA Tuple Handling** (`moving_averages.py`)
   - ✅ Raise error ngay khi có MA calculation thất bại
   - ✅ Tránh return tuple chứa None values gây TypeError downstream

4. **Python Version Compatibility**
   - ✅ Loại bỏ `strict=True` để tương thích với Python < 3.10

5. **Loại Bỏ Deprecated Values**
   - ✅ Đã loại bỏ hoàn toàn `"majority"` và `"unanimous"`
   - ✅ Code sử dụng các giá trị này sẽ raise `ValueError` ngay lập tức

## Future Enhancements (Gợi Ý)

1. **Dynamic Strategy Selection**: Tự động chọn strategies dựa trên market conditions
2. **Strategy Performance Tracking**: Theo dõi performance của từng strategy qua thời gian
3. **Adaptive Weights**: Tự động điều chỉnh weights dựa trên performance
4. **Strategy Ensembles**: Kết hợp nhiều consensus modes
5. **Signal Confidence Score**: Tính toán confidence score dựa trên agreement level

## Modules Đã Được Cải Thiện

### Adaptive Trend Classification (ATC) Modules

#### 1. **equity.py**
- ✅ Validation đầy đủ cho tất cả parameters
- ✅ Logging khi có NaN values, floor hits
- ✅ Xử lý index alignment tự động
- ✅ Error handling với try-except

#### 2. **layer1.py**
- ✅ Sửa `weighted_signal`: preserve tất cả indices (union)
- ✅ Validation cho tất cả functions
- ✅ Logging cho warnings và errors
- ✅ Xử lý NaN và edge cases

#### 3. **moving_averages.py**
- ✅ Raise error ngay khi MA calculation thất bại
- ✅ Validation cho lengths, robustness, ma_type
- ✅ Logging cho warnings và errors
- ✅ Tối ưu code với list comprehensions

#### 4. **signals.py**
- ✅ Validation và index alignment
- ✅ Xử lý NaN values
- ✅ Logging cho warnings
- ✅ Xử lý conflict (cả crossover và crossunder cùng True)

#### 5. **utils.py**
- ✅ Validation cho rate_of_change, diflen, exp_growth
- ✅ Xử lý overflow trong exp_growth
- ✅ Đảm bảo diflen không trả về length <= 0
- ✅ Logging cho warnings và errors

#### 6. **scanner.py**
- ✅ Validation đầy đủ cho tất cả parameters
- ✅ Tracking errors và skipped symbols
- ✅ Summary logging cuối cùng
- ✅ Xử lý data quality issues

### Range Oscillator Strategy Modules

#### 1. **combined.py**
- ✅ Loại bỏ hoàn toàn "majority" và "unanimous" (không còn backward compatibility)
- ✅ Cải thiện threshold voting logic
- ✅ Validation và error handling tốt hơn
- ✅ Python compatibility (loại bỏ strict=True)

## Technical Details

### Code Quality Improvements

- **Validation**: Tất cả functions có input validation đầy đủ
- **Error Handling**: Try-except blocks với logging chi tiết
- **Type Safety**: Type hints đầy đủ và validation
- **Documentation**: Docstrings với `Raises` sections
- **Logging**: Consistent logging từ `modules.common.utils`
- **Edge Cases**: Xử lý NaN, empty series, index mismatches

### Performance Optimizations

- **Vectorization**: Sử dụng NumPy operations
- **Memory**: Tối ưu memory usage với proper dtype
- **Code Duplication**: Giảm duplication với list comprehensions
- **Early Validation**: Fail fast với validation sớm

### Compatibility

- **Python Version**: Tương thích với Python 3.8+ (loại bỏ strict=True)
- **Breaking Changes**: `consensus_mode="majority"` và `"unanimous"` không còn được hỗ trợ
- **Migration**: Sử dụng `consensus_mode="threshold"` với `consensus_threshold` phù hợp
