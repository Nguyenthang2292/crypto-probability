# Cấu trúc Module - Crypto Prediction System

File `crypto_simple_enhance_2.py` đã được tách thành các module nhỏ để dễ bảo trì và mở rộng.

## Cấu trúc Thư mục

```
crypto-probability-/
├── main.py                    # Entry point chính
├── modules/                   # Package chứa tất cả modules
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── cli.py
│   ├── data_fetcher.py
│   ├── indicators.py
│   ├── labeling.py
│   ├── display.py
│   └── model.py
└── ...
```

## Cấu trúc Module

### 1. `config.py`
Chứa tất cả các constants và cấu hình:
- Default values (symbol, timeframe, limit, exchanges)
- Prediction windows mapping
- Target labels và mappings (DOWN, NEUTRAL, UP)
- Model features list

### 2. `utils.py`
Các utility functions:
- `timeframe_to_minutes()` - Chuyển đổi timeframe sang phút
- `get_prediction_window()` - Lấy mô tả prediction window
- `color_text()` - Format text với màu sắc
- `format_price()` - Format giá với precision phù hợp

### 3. `data_fetcher.py`
Xử lý việc fetch dữ liệu từ exchanges:
- `normalize_symbol()` - Chuẩn hóa symbol input
- `fetch_data()` - Fetch OHLCV data từ nhiều exchanges

### 4. `indicators.py`
Tính toán technical indicators và candlestick patterns:
- `add_candlestick_patterns()` - Phát hiện 10 candlestick patterns
- `calculate_indicators()` - Tính toán tất cả indicators (SMA, RSI, MACD, Bollinger Bands, etc.)
- `add_indicators()` - Wrapper function với dropna và warnings

### 5. `labeling.py`
Tạo labels cho training data:
- `apply_directional_labels()` - Tạo UP/DOWN/NEUTRAL labels dựa trên future price movement

### 6. `model.py`
Model training và prediction:
- `build_model()` - Tạo XGBoost classifier với config
- `train_and_predict()` - Train model với cross-validation
- `predict_next_move()` - Predict probability cho next candle

### 7. `display.py`
Hiển thị kết quả và reports:
- `print_classification_report()` - In classification report với confusion matrix có màu sắc

### 8. `cli.py`
Command-line interface:
- `parse_args()` - Parse command-line arguments
- `prompt_with_default()` - Prompt user với default value
- `resolve_input()` - Resolve input từ CLI hoặc prompt

### 9. `main.py`
Entry point chính của ứng dụng:
- `main()` - Orchestrate toàn bộ workflow
- `run_once()` - Xử lý prediction cho một symbol

## Cách sử dụng

Chạy chương trình như trước:
```bash
python main.py
```

Hoặc với arguments:
```bash
python main.py -s BTC/USDT -t 1h -l 1500
```

## Lợi ích của cấu trúc mới

1. **Dễ bảo trì**: Mỗi module có trách nhiệm rõ ràng
2. **Dễ test**: Có thể test từng module riêng biệt
3. **Dễ mở rộng**: Thêm features mới không ảnh hưởng đến code cũ
4. **Tái sử dụng**: Các functions có thể được import và sử dụng ở nơi khác
5. **Dễ đọc**: Code ngắn gọn, dễ hiểu hơn

## Dependency Graph

```
main.py
└── modules/
    ├── config.py (base - no dependencies)
    ├── utils.py
    │   └── config.py
    ├── cli.py
    │   ├── utils.py
    │   └── config.py
    ├── data_fetcher.py
    │   ├── utils.py
    │   └── config.py
    ├── labeling.py
    │   └── config.py
    ├── indicators.py
    │   ├── utils.py
    │   └── labeling.py (lazy import)
    ├── display.py
    │   ├── utils.py
    │   └── config.py
    └── model.py
        ├── config.py
        ├── utils.py
        └── display.py
```

## Import Pattern

Tất cả các imports trong modules sử dụng relative imports (`.`):
- `from .config import ...` - import từ cùng package
- `from .utils import ...` - import từ cùng package

Trong `main.py` sử dụng absolute imports:
- `from modules.config import ...`
- `from modules.utils import ...`

## Notes

- File gốc `crypto_simple_enhance_2.py` vẫn được giữ nguyên để tham khảo
- Tất cả functionality giữ nguyên, chỉ được tổ chức lại
- Import statements đã được tối ưu để tránh circular dependencies

