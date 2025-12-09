# Adaptive Trend Classification (ATC)

Module Adaptive Trend Classification (ATC) cung cấp hệ thống phân tích xu hướng thích ứng sử dụng nhiều loại Moving Averages với adaptive weighting dựa trên equity curves.

## Tổng quan

ATC là một hệ thống phân loại xu hướng thích ứng sử dụng:

- **6 loại Moving Averages**: EMA, HMA, WMA, DEMA, LSMA, KAMA
- **2-layer architecture**: 
  - Layer 1: Tính signals cho từng MA type dựa trên equity curves
  - Layer 2: Tính weights và kết hợp tất cả để tạo Average_Signal
- **Adaptive weighting**: Sử dụng equity curves để tự động điều chỉnh trọng số của từng MA
- **Robustness modes**: "Narrow", "Medium", "Wide" để điều chỉnh độ nhạy

## Cấu trúc Module

```text
adaptive_trend/
├── __init__.py              # Module exports
├── README.md                # Tài liệu này
├── core/
│   ├── __init__.py          # Core exports
│   ├── analyzer.py          # Phân tích symbol đơn lẻ
│   ├── scanner.py          # Scan nhiều symbols
│   ├── compute_atc_signals.py    # Tính toán ATC signals chính
│   ├── compute_equity.py         # Tính toán equity curves
│   ├── compute_moving_averages.py # Tính toán các loại MA
│   ├── process_layer1.py         # Xử lý Layer 1
│   └── signal_detection.py       # Phát hiện signals (crossover/crossunder)
├── cli/
│   ├── __init__.py          # CLI exports
│   ├── argument_parser.py   # Parse command-line arguments
│   ├── display.py           # Hiển thị kết quả
│   └── interactive_prompts.py # Interactive prompts
└── utils/
    ├── __init__.py          # Utils exports
    ├── config.py            # ATCConfig và configuration utilities
    ├── rate_of_change.py   # Tính toán rate of change
    ├── diflen.py            # Tính toán độ dài khác biệt
    └── exp_growth.py        # Exponential growth factor
```

## Cách hoạt động

### Layer 1: Individual MA Signals

Với mỗi loại MA (EMA, HMA, WMA, DEMA, LSMA, KAMA):

1. Tính toán 9 MAs với các độ dài khác nhau (base length ± offsets dựa trên robustness)
2. Tính signals cho từng MA dựa trên price/MA crossovers
3. Tính equity curves cho từng signal sử dụng exponential growth
4. Weighted average của 9 signals dựa trên equity curves → Layer 1 signal cho MA type đó

### Layer 2: Combined Signal

1. Tính weights cho từng MA type dựa trên Layer 1 signals
2. Weighted average của tất cả Layer 1 signals → **Average_Signal** (final output)

### Equity Curves

Equity curves mô phỏng performance của trading strategy:
- Sử dụng exponential growth factor (La) và decay rate (De)
- Equity cao hơn → weight cao hơn → MA đó có ảnh hưởng lớn hơn
- Adaptive: Tự động điều chỉnh weights dựa trên performance

## Sử dụng

### Ví dụ cơ bản

```python
import pandas as pd
from modules.adaptive_trend import compute_atc_signals, ATCConfig

# Chuẩn bị dữ liệu
prices = pd.Series([...])  # Close prices

# Cấu hình
config = ATCConfig(
    ema_len=28,
    hma_len=28,
    wma_len=28,
    dema_len=28,
    lsma_len=28,
    kama_len=28,
    robustness="Medium",  # "Narrow", "Medium", "Wide"
    lambda_param=0.02,     # Growth rate
    decay=0.03,            # Decay rate
    cutout=0,              # Bars to skip
)

# Tính toán ATC signals
results = compute_atc_signals(
    prices=prices,
    ema_len=config.ema_len,
    hull_len=config.hma_len,
    wma_len=config.wma_len,
    dema_len=config.dema_len,
    lsma_len=config.lsma_len,
    kama_len=config.kama_len,
    robustness=config.robustness,
    La=config.lambda_param,
    De=config.decay,
    cutout=config.cutout,
)

# Kết quả
average_signal = results["Average_Signal"]  # Final combined signal
ema_signal = results["EMA_Signal"]         # Layer 1: EMA signal
hma_signal = results["HMA_Signal"]         # Layer 1: HMA signal
# ... các signals khác
```

### Phân tích một symbol

```python
from modules.adaptive_trend import analyze_symbol, ATCConfig
from modules.common.DataFetcher import DataFetcher
from modules.common.ExchangeManager import ExchangeManager

# Khởi tạo
exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager)

# Cấu hình
config = ATCConfig(
    timeframe="15m",
    limit=1500,
    ema_len=28,
    # ... các parameters khác
)

# Phân tích
result = analyze_symbol(
    symbol="BTC/USDT",
    data_fetcher=data_fetcher,
    config=config,
)

if result:
    print(f"Symbol: {result['symbol']}")
    print(f"Current Price: {result['current_price']}")
    print(f"ATC Results: {result['atc_results']}")
```

### Scan nhiều symbols

```python
from modules.adaptive_trend import scan_all_symbols, ATCConfig
from modules.common.DataFetcher import DataFetcher
from modules.common.ExchangeManager import ExchangeManager

# Khởi tạo
exchange_manager = ExchangeManager()
data_fetcher = DataFetcher(exchange_manager)

# Cấu hình
config = ATCConfig(
    timeframe="15m",
    limit=1500,
    # ... các parameters khác
)

# Scan
symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
results, stats = scan_all_symbols(
    symbols=symbols,
    data_fetcher=data_fetcher,
    atc_config=config,
    min_signal=0.5,  # Minimum signal strength
    parallel=True,   # Use parallel processing
)

# Kết quả
for result in results:
    print(f"{result['symbol']}: Signal = {result['signal']}")
```

### Sử dụng CLI

```bash
# Phân tích một symbol
python main_atc.py BTC/USDT

# Scan tất cả futures symbols
python main_atc.py --auto --scan

# Interactive mode
python main_atc.py --interactive

# Custom timeframe
python main_atc.py BTC/USDT --timeframe 1h
```

## Cấu hình

### ATCConfig

```python
@dataclass
class ATCConfig:
    # Moving Average lengths
    ema_len: int = 28
    hma_len: int = 28
    wma_len: int = 28
    dema_len: int = 28
    lsma_len: int = 28
    kama_len: int = 28
    
    # ATC parameters
    robustness: str = "Medium"  # "Narrow", "Medium", or "Wide"
    lambda_param: float = 0.02  # Growth rate for equity
    decay: float = 0.03         # Decay rate for equity
    cutout: int = 0            # Bars to skip at beginning
    
    # Data parameters
    limit: int = 1500          # Number of candles to fetch
    timeframe: str = "15m"     # Timeframe
```

### Robustness Modes

- **Narrow**: Offsets nhỏ → ít variation trong MA lengths → nhạy cảm hơn
- **Medium**: Offsets trung bình → cân bằng
- **Wide**: Offsets lớn → nhiều variation → ổn định hơn, ít nhạy cảm hơn

## Kết quả

`compute_atc_signals` trả về dictionary chứa:

- **Average_Signal**: Signal cuối cùng (kết hợp tất cả MAs)
- **EMA_Signal**, **HMA_Signal**, **WMA_Signal**, **DEMA_Signal**, **LSMA_Signal**, **KAMA_Signal**: Layer 1 signals cho từng MA type
- **EMA_Weight**, **HMA_Weight**, **WMA_Weight**, **DEMA_Weight**, **LSMA_Weight**, **KAMA_Weight**: Weights cho từng MA type
- **EMA_Equity**, **HMA_Equity**, ...: Equity curves cho từng MA type

Tất cả đều là `pd.Series` với cùng index như input prices.

## Signal Interpretation

- **Positive values (> 0)**: Bullish signal, giá trên MA
- **Negative values (< 0)**: Bearish signal, giá dưới MA
- **Zero (0)**: Neutral, không có signal rõ ràng
- **Magnitude**: Độ mạnh của signal (cao hơn = mạnh hơn)

## Utilities

### rate_of_change

Tính toán rate of change (tỷ lệ thay đổi) của một series:

```python
from modules.adaptive_trend.utils import rate_of_change

roc = rate_of_change(prices, period=1)
```

### diflen

Tính toán độ dài khác biệt dựa trên robustness mode:

```python
from modules.adaptive_trend.utils import diflen

offset = diflen(robustness="Medium")  # Returns offset value
```

### exp_growth

Tính toán exponential growth factor:

```python
from modules.adaptive_trend.utils import exp_growth

growth = exp_growth(La=0.02, period=1)
```

## Performance Optimization

- **Numba JIT compilation**: Equity calculations được compile với Numba để tăng tốc
- **Vectorized operations**: Sử dụng NumPy cho các phép tính cuối cùng
- **Caching**: Rate of change được cache để tránh tính toán lại
- **Parallel scanning**: Scanner hỗ trợ parallel processing cho nhiều symbols

## Lưu ý

1. **Data quality**: ATC cần dữ liệu OHLCV chất lượng cao. Đảm bảo data không có gaps lớn.

2. **Timeframe**: ATC hoạt động tốt trên nhiều timeframes, nhưng parameters có thể cần điều chỉnh:
   - Timeframe ngắn (1m, 5m): Có thể cần giảm lengths
   - Timeframe dài (4h, 1d): Có thể cần tăng lengths

3. **Robustness**: 
   - "Narrow" cho thị trường trending mạnh
   - "Medium" cho thị trường cân bằng
   - "Wide" cho thị trường volatile

4. **Lambda và Decay**: 
   - Lambda cao → equity tăng nhanh → weights thay đổi nhanh
   - Decay cao → equity giảm nhanh → weights giảm nhanh

5. **Cutout**: Bỏ qua một số bars đầu tiên để tránh initialization artifacts.

## CLI Commands

Module cung cấp CLI interface qua `main_atc.py`:

```bash
# Basic usage
python main_atc.py <SYMBOL>

# Options
--timeframe TIMEFRAME    # Set timeframe (default: 15m)
--auto                   # Auto mode (no interactive menu)
--scan                   # Scan all futures symbols
--min-signal FLOAT       # Minimum signal strength for scan
--no-menu                # Skip interactive menu
--parallel               # Use parallel processing for scan
```

## Ví dụ nâng cao

### Custom configuration từ dictionary

```python
from modules.adaptive_trend.utils.config import create_atc_config_from_dict

params = {
    "ema_len": 21,
    "hma_len": 21,
    "wma_len": 21,
    "dema_len": 21,
    "lsma_len": 21,
    "kama_len": 21,
    "robustness": "Narrow",
    "lambda_param": 0.03,
    "decay": 0.02,
    "limit": 2000,
}

config = create_atc_config_from_dict(params, timeframe="1h")
```

### Kết hợp với các indicators khác

```python
from modules.adaptive_trend import compute_atc_signals
from modules.common.IndicatorEngine import IndicatorEngine

# Tính ATC signals
atc_results = compute_atc_signals(prices=df['close'], ...)

# Tính các indicators khác
engine = IndicatorEngine()
df_with_indicators, metadata = engine.compute(df)

# Kết hợp signals
combined_signal = (
    atc_results['Average_Signal'] * 0.6 +
    (df_with_indicators['RSI_14'] - 50) / 50 * 0.4
)
```

## Tài liệu tham khảo

- Port từ Pine Script indicator "Adaptive Trend Classification"
- Sử dụng multiple Moving Averages với adaptive weighting
- Equity-based weighting để tự động điều chỉnh trọng số

