# Pairs Trading Documentation

Tài liệu cho pairs trading analysis component.

## Overview

Pairs Trading component phân tích performance của các futures pairs trên Binance để xác định cơ hội pairs trading:
- Phân tích performance của tất cả futures pairs
- Xác định top best và worst performers
- Tìm pairs trading opportunities (long worst, short best)
- Validate pairs dựa trên correlation và spread
- Tính toán quantitative metrics để đánh giá chất lượng pairs

## Components

### Performance Analyzer
- **Location:** `modules/pairs_trading/performance_analyzer.py`
- Tính toán performance score từ 3 khung thời gian (1 ngày, 3 ngày, 1 tuần)
- Xác định top N best và worst performers
- Weighted scoring với configurable weights

### Pairs Analyzer
- **Location:** `modules/pairs_trading/pairs_analyzer.py`
- Phân tích pairs trading opportunities
- Tính correlation giữa pairs
- Validate pairs (spread, correlation, volume)
- Sử dụng các quantitative metrics để đánh giá pairs

## Module Structure

Module đã được tách thành các file nhỏ để dễ test và debug:

### 1. `pairs_analyzer.py` (Main Analyzer)
- **Mục đích**: Class chính để phân tích và validate pairs trading opportunities
- **Trách nhiệm**:
  - Fetch và align price data
  - Tính correlation giữa các symbols
  - Tính spread giữa long và short symbols
  - Analyze pairs opportunities
  - Validate pairs
- **Dependencies**: Sử dụng `PairMetricsComputer` và `OpportunityScorer`

### 2. `statistical_tests.py` (Statistical Tests)
- **Mục đích**: Các test thống kê cho pairs trading
- **Functions**:
  - `calculate_adf_test()`: Augmented Dickey-Fuller test để kiểm tra cointegration
  - `calculate_half_life()`: Tính half-life của mean reversion
  - `calculate_johansen_test()`: Johansen cointegration test
- **Dependencies**: statsmodels, sklearn

### 3. `risk_metrics.py` (Risk Metrics)
- **Mục đích**: Tính toán các chỉ số rủi ro
- **Functions**:
  - `calculate_spread_sharpe()`: Sharpe ratio của spread returns
  - `calculate_max_drawdown()`: Maximum drawdown của cumulative spread
  - `calculate_calmar_ratio()`: Calmar ratio (annual return / max drawdown)
- **Dependencies**: pandas, numpy

### 4. `hedge_ratio.py` (Hedge Ratio Calculations)
- **Mục đích**: Tính toán hedge ratio giữa hai assets
- **Functions**:
  - `calculate_ols_hedge_ratio()`: OLS hedge ratio
  - `calculate_kalman_hedge_ratio()`: Dynamic hedge ratio sử dụng Kalman filter
- **Dependencies**: sklearn, pykalman

### 5. `zscore_metrics.py` (Z-Score Metrics)
- **Mục đích**: Tính toán các metrics liên quan đến z-score
- **Functions**:
  - `calculate_zscore_stats()`: Z-score statistics (mean, std, skewness, kurtosis, current)
  - `calculate_hurst_exponent()`: Hurst exponent (R/S analysis)
  - `calculate_direction_metrics()`: Classification metrics (F1, precision, recall, accuracy)
- **Dependencies**: sklearn.metrics

### 6. `pair_metrics_computer.py` (Metrics Computer)
- **Mục đích**: Orchestrate tất cả metrics calculations
- **Class**: `PairMetricsComputer`
- **Method**: `compute_pair_metrics()` - Tính tất cả quantitative metrics cho một pair
- **Dependencies**: Tất cả các module trên

### 7. `opportunity_scorer.py` (Opportunity Scorer)
- **Mục đích**: Tính toán opportunity score cho pairs trading opportunities
- **Class**: `OpportunityScorer`
- **Method**: `calculate_opportunity_score()` - Tính score dựa trên spread, correlation và quant metrics
- **Dependencies**: numpy

### 8. `performance_analyzer.py` (Performance Analyzer)
- **Mục đích**: Phân tích performance của các symbols (không thay đổi)

## Lợi ích của cấu trúc module

1. **Dễ test**: Mỗi module có thể được test độc lập
2. **Dễ debug**: Dễ dàng tìm và sửa lỗi trong từng module cụ thể
3. **Tái sử dụng**: Các functions có thể được sử dụng trong các context khác
4. **Maintainability**: Code dễ đọc và maintain hơn
5. **Separation of concerns**: Mỗi module có một trách nhiệm rõ ràng

## Usage

### Command Line

```bash
python pairs_trading_main.py
```

### Options

```bash
# Basic usage
python pairs_trading_main.py

# With custom parameters
python pairs_trading_main.py --top-n 10 --weights "1d:0.4,3d:0.4,1w:0.2"

# Test mode (without API)
python pairs_trading_main.py --use-hardcoded-symbols --max-symbols 20
```

### Python API

```python
from modules.pairs_trading import PairsTradingAnalyzer

# Khởi tạo analyzer
analyzer = PairsTradingAnalyzer()

# Analyze pairs
pairs_df = analyzer.analyze_pairs_opportunity(
    best_performers=best_df,
    worst_performers=worst_df,
    data_fetcher=data_fetcher
)

# Validate pairs
validated_pairs = analyzer.validate_pairs(pairs_df, data_fetcher)
```

### Import các module riêng lẻ

Bạn cũng có thể import và sử dụng các functions riêng lẻ:

```python
from modules.pairs_trading.statistical_tests import calculate_adf_test
from modules.pairs_trading.risk_metrics import calculate_spread_sharpe
from modules.pairs_trading.hedge_ratio import calculate_ols_hedge_ratio
from modules.pairs_trading.zscore_metrics import calculate_zscore_stats
```

## Configuration

Tất cả config được định nghĩa trong `modules/config.py` section **Pairs Trading Configuration**:
- `PAIRS_TRADING_WEIGHTS` - Weights cho timeframes (1d, 3d, 1w)
- `PAIRS_TRADING_TOP_N` - Số lượng top/bottom performers
- `PAIRS_TRADING_MIN_SPREAD` - Spread tối thiểu
- `PAIRS_TRADING_MAX_SPREAD` - Spread tối đa
- `PAIRS_TRADING_MIN_CORRELATION` - Correlation tối thiểu
- `PAIRS_TRADING_MAX_CORRELATION` - Correlation tối đa
- `PAIRS_TRADING_ADF_PVALUE_THRESHOLD` - Threshold cho ADF test
- `PAIRS_TRADING_MAX_HALF_LIFE` - Half-life tối đa
- `PAIRS_TRADING_MIN_SPREAD_SHARPE` - Sharpe ratio tối thiểu
- Và nhiều config khác cho quantitative metrics

## Strategy

Pairs trading strategy:
- **Long:** Worst performers (expect mean reversion upward)
- **Short:** Best performers (expect mean reversion downward)
- **Ideal correlation:** 0.3 - 0.9 (moderate correlation)
- **Ideal spread:** 1% - 50%
- **Cointegration:** Spread phải stationary (mean-reverting)
- **Half-life:** Càng ngắn càng tốt (< 50 periods)

## Features

- Performance analysis với weighted scoring
- Correlation calculation với caching
- Pairs validation (spread, correlation, volume)
- Opportunity scoring với quantitative metrics
- Real-time analysis với Binance data
- Quantitative metrics:
  - Cointegration tests (ADF, Johansen)
  - Half-life calculation
  - Z-score statistics
  - Hurst exponent
  - Risk metrics (Sharpe, drawdown, Calmar)
  - Dynamic hedge ratio (Kalman filter)
  - Classification metrics

## Workflow

1. Fetch futures symbols từ Binance
2. Analyze performance cho tất cả symbols
3. Identify top N best và worst performers
4. Analyze pairs trading opportunities
   - Tính correlation giữa pairs
   - Tính quantitative metrics cho mỗi pair
   - Tính opportunity score
5. Validate pairs (spread, correlation, volume)
6. Display recommended pairs với đầy đủ metrics

## Related Documentation

- [Quantitative Metrics Proposal](./QUANT_METRICS_PROPOSAL.md) - Chi tiết về các quantitative metrics đã implement
- [Common Utilities](../common/) - DataFetcher, ExchangeManager
- [Config](../../modules/config.py) - Pairs trading configuration
