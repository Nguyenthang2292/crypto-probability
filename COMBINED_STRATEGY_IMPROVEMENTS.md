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
   - **"majority"** (mặc định): Đa số strategies phải đồng ý
   - **"unanimous"**: Tất cả strategies phải đồng ý
   - **"weighted"**: Bỏ phiếu có trọng số dựa trên `strategy_weights`
   - **"threshold"**: Yêu cầu một phần trăm nhất định strategies đồng ý (theo `consensus_threshold`)

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

## Ví Dụ Sử Dụng

### Ví Dụ 1: Sử Dụng Cơ Bản (Backward Compatible)
```python
signals, strength = generate_signals_strategy5_combined(
    high=high, low=low, close=close,
    use_sustained=True,
    use_crossover=True,
    use_momentum=True
)
```

### Ví Dụ 2: Sử Dụng Tất Cả Strategies
```python
signals, strength = generate_signals_strategy5_combined(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6, 7, 8, 9],  # Tất cả strategies
    consensus_mode="majority"
)
```

### Ví Dụ 3: Consensus Mode "Unanimous"
```python
signals, strength = generate_signals_strategy5_combined(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4],
    consensus_mode="unanimous"  # Tất cả phải đồng ý
)
```

### Ví Dụ 4: Weighted Voting
```python
signals, strength = generate_signals_strategy5_combined(
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

### Ví Dụ 5: Threshold Mode
```python
signals, strength = generate_signals_strategy5_combined(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4, 6, 7],
    consensus_mode="threshold",
    consensus_threshold=0.6  # Ít nhất 60% strategies phải đồng ý
)
```

### Ví Dụ 6: Lọc Signal Strength
```python
signals, strength = generate_signals_strategy5_combined(
    high=high, low=low, close=close,
    enabled_strategies=[2, 3, 4],
    min_signal_strength=0.3  # Chỉ chấp nhận signals có strength >= 0.3
)
```

### Ví Dụ 7: Lấy Thống Kê
```python
signals, strength, stats = generate_signals_strategy5_combined(
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
signals, strength = generate_signals_strategy5_combined(
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

Không cần migration! Code cũ vẫn hoạt động. Chỉ cần thêm các tính năng mới khi cần.

## Future Enhancements (Gợi Ý)

1. **Dynamic Strategy Selection**: Tự động chọn strategies dựa trên market conditions
2. **Strategy Performance Tracking**: Theo dõi performance của từng strategy qua thời gian
3. **Adaptive Weights**: Tự động điều chỉnh weights dựa trên performance
4. **Strategy Ensembles**: Kết hợp nhiều consensus modes
5. **Signal Confidence Score**: Tính toán confidence score dựa trên agreement level
