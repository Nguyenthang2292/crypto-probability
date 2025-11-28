# Pairs Trading Documentation

Tài liệu cho pairs trading analysis component.

## Overview

Pairs Trading component phân tích performance của các futures pairs trên Binance để xác định cơ hội pairs trading:
- Phân tích performance của tất cả futures pairs
- Xác định top best và worst performers
- Tìm pairs trading opportunities (long worst, short best)
- Validate pairs dựa trên correlation và spread
- Tính toán quantitative metrics để đánh giá chất lượng pairs

Module này cung cấp một framework hoàn chỉnh để xác định, phân tích và validate các cơ hội pairs trading dựa trên nguyên tắc statistical arbitrage.

## Architecture

```text
pairs_trading/
├── core/                   # Core analysis components
│   ├── pairs_analyzer.py          # Main pairs trading analyzer
│   ├── pair_metrics_computer.py   # Quantitative metrics computation
│   └── opportunity_scorer.py      # Opportunity scoring logic
│
├── metrics/                # Statistical and quantitative metrics (organized into sub-packages)
│   ├── __init__.py                # Re-exports all functions (backward compatible)
│   ├── statistical_tests/        # Statistical tests for cointegration
│   │   ├── adf_test.py           # Augmented Dickey-Fuller test
│   │   └── johansen_test.py      # Johansen cointegration test
│   ├── mean_reversion/            # Mean reversion metrics
│   │   ├── half_life.py          # Half-life of mean reversion
│   │   ├── hurst_exponent.py     # Hurst exponent calculation
│   │   └── zscore_stats.py       # Z-score statistics
│   ├── hedge_ratios/             # Hedge ratio calculations
│   │   ├── ols_hedge_ratio.py    # OLS (static) hedge ratio
│   │   └── kalman_hedge_ratio.py # Kalman (dynamic) hedge ratio
│   ├── risk/                     # Risk metrics
│   │   ├── sharpe_ratio.py       # Sharpe ratio calculation
│   │   ├── max_drawdown.py       # Maximum drawdown calculation
│   │   └── calmar_ratio.py       # Calmar ratio calculation
│   └── classification/           # Classification/prediction metrics
│       └── direction_metrics.py  # Direction prediction metrics
│
├── analysis/               # Performance analysis
│   └── performance_analyzer.py    # Multi-timeframe performance analysis
│
├── utils/                  # Business logic utilities
│   ├── pairs_selector.py           # Pair selection algorithms
│   └── ensure_symbols_in_pools.py  # Candidate pool management
│
└── cli/                    # Command-line interface
    ├── argument_parser.py         # CLI argument parsing
    ├── interactive_prompts.py     # Interactive user prompts
    ├── input_parsers.py           # Input validation and parsing
    └── display/                   # Display utilities
        ├── display_performers.py
        └── display_pairs_opportunities.py
```

## Key Components

### Core Components

#### PairsTradingAnalyzer
Main class để phân tích pairs trading opportunities từ best và worst performing symbols.

**Location:** `modules/pairs_trading/core/pairs_analyzer.py`

```python
from modules.pairs_trading import PairsTradingAnalyzer

analyzer = PairsTradingAnalyzer(
    min_spread=0.01,
    max_spread=0.50,
    min_correlation=0.3,
    max_correlation=0.9
)

pairs = analyzer.analyze_pairs_opportunity(
    best_performers=best_df,
    worst_performers=worst_df,
    data_fetcher=fetcher
)
```

**Trách nhiệm:**
- Fetch và align price data
- Tính correlation giữa các symbols
- Tính spread giữa long và short symbols
- Analyze pairs opportunities
- Validate pairs

#### PairMetricsComputer
Tính toán comprehensive quantitative metrics cho trading pairs.

**Location:** `modules/pairs_trading/core/pair_metrics_computer.py`

```python
from modules.pairs_trading import PairMetricsComputer

computer = PairMetricsComputer(
    adf_pvalue_threshold=0.05,
    zscore_lookback=60
)

metrics = computer.compute_pair_metrics(price1, price2)
# Returns: hedge_ratio, adf_pvalue, is_cointegrated, half_life,
#          zscore stats, Hurst exponent, Sharpe ratio, etc.
```

#### OpportunityScorer
Tính toán opportunity scores dựa trên spread, correlation và quantitative metrics.

**Location:** `modules/pairs_trading/core/opportunity_scorer.py`

```python
from modules.pairs_trading import OpportunityScorer

scorer = OpportunityScorer(
    min_correlation=0.3,
    max_correlation=0.9
)

score = scorer.calculate_opportunity_score(
    spread=0.15,
    correlation=0.75,
    quant_metrics=metrics
)
```

#### PerformanceAnalyzer
Tính toán performance score từ 3 khung thời gian (1 ngày, 3 ngày, 1 tuần).

**Location:** `modules/pairs_trading/analysis/performance_analyzer.py`

- Xác định top N best và worst performers
- Weighted scoring với configurable weights

### Metrics Modules

Metrics được tổ chức thành các sub-packages logic để dễ quản lý và mở rộng.

#### Cấu trúc Sub-packages

```text
metrics/
├── statistical_tests/    # Statistical tests for cointegration
├── mean_reversion/       # Mean reversion metrics
├── hedge_ratios/         # Hedge ratio calculations
├── risk/                 # Risk metrics
└── classification/       # Classification/prediction metrics
```

#### Statistical Tests
**Location:** `modules/pairs_trading/metrics/statistical_tests/`

- **ADF Test**: Augmented Dickey-Fuller test để kiểm tra cointegration
- **Johansen Test**: Johansen cointegration test

```python
# Backward compatible import
from modules.pairs_trading.metrics import (
    calculate_adf_test,
    calculate_johansen_test,
)

# Hoặc import từ sub-package
from modules.pairs_trading.metrics.statistical_tests import (
    calculate_adf_test,
    calculate_johansen_test,
)

adf_result = calculate_adf_test(spread_series)
johansen_result = calculate_johansen_test(price1, price2)
```

#### Mean Reversion Metrics
**Location:** `modules/pairs_trading/metrics/mean_reversion/`

- **Half-life**: Mean reversion half-life
- **Hurst Exponent**: Mean reversion indicator (H < 0.5 = mean-reverting)
- **Z-score Statistics**: Mean, std, skewness, kurtosis, current z-score

```python
# Backward compatible import
from modules.pairs_trading.metrics import (
    calculate_half_life,
    calculate_hurst_exponent,
    calculate_zscore_stats,
)

# Hoặc import từ sub-package
from modules.pairs_trading.metrics.mean_reversion import (
    calculate_half_life,
    calculate_hurst_exponent,
    calculate_zscore_stats,
)

half_life = calculate_half_life(spread_series)
hurst = calculate_hurst_exponent(spread_series)
zscore_stats = calculate_zscore_stats(spread_series)
```

#### Hedge Ratios
**Location:** `modules/pairs_trading/metrics/hedge_ratios/`

- **OLS Hedge Ratio**: Ordinary Least Squares regression (static)
- **Kalman Hedge Ratio**: Kalman filter cho time-varying hedge ratio (dynamic)

```python
# Backward compatible import
from modules.pairs_trading.metrics import (
    calculate_ols_hedge_ratio,
    calculate_kalman_hedge_ratio,
)

# Hoặc import từ sub-package
from modules.pairs_trading.metrics.hedge_ratios import (
    calculate_ols_hedge_ratio,
    calculate_kalman_hedge_ratio,
)

ols_ratio = calculate_ols_hedge_ratio(price1, price2, fit_intercept=True)
kalman_ratio = calculate_kalman_hedge_ratio(price1, price2, delta=1e-5)
```

#### Risk Metrics
**Location:** `modules/pairs_trading/metrics/risk/`

- **Spread Sharpe Ratio**: Risk-adjusted return của spread
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return / max drawdown

```python
# Backward compatible import
from modules.pairs_trading.metrics import (
    calculate_spread_sharpe,
    calculate_max_drawdown,
    calculate_calmar_ratio,
)

# Hoặc import từ sub-package
from modules.pairs_trading.metrics.risk import (
    calculate_spread_sharpe,
    calculate_max_drawdown,
    calculate_calmar_ratio,
)

sharpe = calculate_spread_sharpe(pnl_series, periods_per_year=365*24)
max_dd = calculate_max_drawdown(equity_curve)
calmar = calculate_calmar_ratio(equity_curve, periods_per_year=365*24)
```

#### Classification Metrics
**Location:** `modules/pairs_trading/metrics/classification/`

- **Direction Metrics**: Classification metrics cho spread direction prediction (accuracy, precision, recall, F1)

```python
# Backward compatible import
from modules.pairs_trading.metrics import calculate_direction_metrics

# Hoặc import từ sub-package
from modules.pairs_trading.metrics.classification import calculate_direction_metrics

direction_metrics = calculate_direction_metrics(spread_series)
```

#### Lợi ích của cấu trúc Sub-packages

1. **Tổ chức tốt hơn**: Metrics liên quan được nhóm lại với nhau
2. **Dễ tìm kiếm**: Cấu trúc rõ ràng giúp dễ tìm metrics cụ thể
3. **Dễ mở rộng**: Dễ dàng thêm metrics mới vào sub-package phù hợp
4. **Backward Compatible**: Code hiện tại vẫn hoạt động không cần thay đổi
5. **Dependencies rõ ràng**: Dependencies nội bộ (ví dụ: calmar_ratio → max_drawdown) nằm trong cùng package

### Utilities

#### Pair Selection
**Location:** `modules/pairs_trading/utils/pairs_selector.py`

```python
from modules.pairs_trading import (
    select_top_unique_pairs,
    select_pairs_for_symbols
)

# Select top N pairs với unique symbols
unique_pairs = select_top_unique_pairs(pairs_df, target_pairs=10)

# Select best pairs cho specific symbols
symbol_pairs = select_pairs_for_symbols(
    pairs_df, 
    target_symbols=['BTC/USDT', 'ETH/USDT']
)
```

#### Candidate Pool Management
**Location:** `modules/pairs_trading/utils/ensure_symbols_in_pools.py`

```python
from modules.pairs_trading import ensure_symbols_in_candidate_pools

# Ensure target symbols trong appropriate pools
best_df, worst_df = ensure_symbols_in_candidate_pools(
    performance_df,
    best_df,
    worst_df,
    target_symbols=['BTC/USDT', 'ETH/USDT']
)
```

### CLI Tools

#### Display Formatters
**Location:** `modules/pairs_trading/cli/display/`

```python
from modules.pairs_trading import (
    display_performers,
    display_pairs_opportunities
)

# Display top/worst performers
display_performers(best_df, "Top Performers", color=Fore.GREEN)
display_performers(worst_df, "Worst Performers", color=Fore.RED)

# Display pairs opportunities
display_pairs_opportunities(pairs_df, max_display=10)
```

#### Argument Parsing
**Location:** `modules/pairs_trading/cli/argument_parser.py`

```python
from modules.pairs_trading import parse_args

args = parse_args()
# Parses all CLI arguments: --pairs-count, --weights, --min-spread, etc.
```

#### Interactive Prompts
**Location:** `modules/pairs_trading/cli/interactive_prompts.py`

```python
from modules.pairs_trading import (
    prompt_interactive_mode,
    prompt_weight_preset_selection,
    prompt_kalman_preset_selection,
    prompt_opportunity_preset_selection,
    prompt_target_pairs,
    prompt_candidate_depth
)

mode, symbols_str = prompt_interactive_mode()
preset = prompt_weight_preset_selection(current_preset='balanced')
kalman_preset = prompt_kalman_preset_selection()
opportunity_preset = prompt_opportunity_preset_selection()
target_pairs = prompt_target_pairs(default_count=10)
candidate_depth = prompt_candidate_depth(default=50)
```

#### Input Parsers
**Location:** `modules/pairs_trading/cli/input_parsers.py`

```python
from modules.pairs_trading import (
    parse_weights,
    parse_symbols,
    standardize_symbol_input
)

weights = parse_weights("1d:0.5,3d:0.3,1w:0.2")
symbols = parse_symbols("BTC/USDT,ETH/USDT")
standardized = standardize_symbol_input("btc/usdt")
```

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

Bạn có thể import từ main package (backward compatible) hoặc từ sub-packages:

```python
# Backward compatible - import từ main package
from modules.pairs_trading.metrics import (
    calculate_adf_test,
    calculate_spread_sharpe,
    calculate_ols_hedge_ratio,
    calculate_zscore_stats,
    calculate_direction_metrics,
)

# Hoặc import từ sub-packages (rõ ràng hơn)
from modules.pairs_trading.metrics.statistical_tests import calculate_adf_test
from modules.pairs_trading.metrics.risk import calculate_spread_sharpe
from modules.pairs_trading.metrics.hedge_ratios import calculate_ols_hedge_ratio
from modules.pairs_trading.metrics.mean_reversion import calculate_zscore_stats
from modules.pairs_trading.metrics.classification import calculate_direction_metrics
```

## Workflow

### Typical Usage Flow

1. **Analyze Performance**
   ```python
   from modules.pairs_trading import PerformanceAnalyzer
   
   analyzer = PerformanceAnalyzer(weights={'1d': 0.5, '3d': 0.3, '1w': 0.2})
   performance_df = analyzer.analyze_all_symbols(symbols, data_fetcher)
   best_df = analyzer.get_top_performers(performance_df, top_n=50)
   worst_df = analyzer.get_worst_performers(performance_df, top_n=50)
   ```

2. **Analyze Pairs**
   ```python
   from modules.pairs_trading import PairsTradingAnalyzer
   
   pairs_analyzer = PairsTradingAnalyzer()
   pairs_df = pairs_analyzer.analyze_pairs_opportunity(
       best_performers=best_df,
       worst_performers=worst_df,
       data_fetcher=data_fetcher
   )
   ```

3. **Validate Pairs**
   ```python
   validated_pairs = pairs_analyzer.validate_pairs(
       pairs_df,
       data_fetcher=data_fetcher
   )
   ```

4. **Select and Display**
   ```python
   from modules.pairs_trading import (
       select_top_unique_pairs,
       display_pairs_opportunities
   )
   
   final_pairs = select_top_unique_pairs(validated_pairs, target_pairs=10)
   display_pairs_opportunities(final_pairs, max_display=10)
   ```

### Detailed Workflow Steps

1. Fetch futures symbols từ Binance
2. Analyze performance cho tất cả symbols
3. Identify top N best và worst performers
4. Analyze pairs trading opportunities
   - Tính correlation giữa pairs
   - Tính quantitative metrics cho mỗi pair
   - Tính opportunity score
5. Validate pairs (spread, correlation, volume)
6. Display recommended pairs với đầy đủ metrics

## Strategy

Pairs trading strategy:
- **Long:** Worst performers (expect mean reversion upward)
- **Short:** Best performers (expect mean reversion downward)
- **Ideal correlation:** 0.3 - 0.9 (moderate correlation)
- **Ideal spread:** 1% - 50%
- **Cointegration:** Spread phải stationary (mean-reverting)
- **Half-life:** Càng ngắn càng tốt (< 50 periods)

## Configuration

Tất cả config được định nghĩa trong `modules/config.py` section **Pairs Trading Configuration**:

### Key Parameters

```python
# Spread thresholds
PAIRS_TRADING_MIN_SPREAD = 0.01  # 1%
PAIRS_TRADING_MAX_SPREAD = 0.50  # 50%

# Correlation thresholds
PAIRS_TRADING_MIN_CORRELATION = 0.3
PAIRS_TRADING_MAX_CORRELATION = 0.9

# Statistical tests
PAIRS_TRADING_ADF_PVALUE_THRESHOLD = 0.05
PAIRS_TRADING_MAX_HALF_LIFE = 50

# Z-score parameters
PAIRS_TRADING_ZSCORE_LOOKBACK = 60
PAIRS_TRADING_CLASSIFICATION_ZSCORE = 0.5

# Hedge ratio parameters
PAIRS_TRADING_OLS_FIT_INTERCEPT = True
PAIRS_TRADING_KALMAN_DELTA = 1e-5
PAIRS_TRADING_KALMAN_OBS_COV = 1.0

# Performance analysis
PAIRS_TRADING_WEIGHTS = {'1d': 0.5, '3d': 0.3, '1w': 0.2}
PAIRS_TRADING_TOP_N = 5
PAIRS_TRADING_MIN_SPREAD_SHARPE = 1.0
```

Xem `modules/config.py` để biết thêm chi tiết về tất cả các config parameters.

## Metrics Interpretation

### Opportunity Score
- **> 20%**: Excellent opportunity (green)
- **10-20%**: Good opportunity (yellow)
- **< 10%**: Weak opportunity (white)

### Quantitative Score (0-100)
- **>= 70**: Strong quantitative metrics (green)
- **50-70**: Moderate metrics (yellow)
- **< 50**: Weak metrics (red)

### Hurst Exponent
- **H < 0.5**: Mean-reverting (good for pairs trading)
- **H ≈ 0.5**: Random walk
- **H > 0.5**: Trending (less suitable)

### Correlation
- **|r| > 0.7**: Strong correlation (green)
- **0.4 < |r| <= 0.7**: Moderate correlation (yellow)
- **|r| <= 0.4**: Weak correlation (red)

## Best Practices

1. **Always validate pairs** với statistical tests (ADF, Johansen)
2. **Check Hurst exponent** - prefer H < 0.5 cho mean reversion
3. **Monitor half-life** - shorter is better cho faster mean reversion
4. **Use appropriate hedge ratios** - Kalman cho time-varying relationships
5. **Diversify** - select pairs với unique symbols
6. **Review quantitative scores** - aim for >= 70 cho robust opportunities

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

## Dependencies

- pandas
- numpy
- scipy (for statistical tests)
- sklearn (for classification metrics)
- statsmodels (for ADF, Johansen tests)
- pykalman (for Kalman filter)
- colorama (for CLI color output)

## Related Documentation

- [Quantitative Metrics Proposal](./QUANT_METRICS_PROPOSAL.md) - Chi tiết về các quantitative metrics đã implement
- [Common Utilities](../common/) - DataFetcher, ExchangeManager
- [Config](../../modules/config.py) - Pairs trading configuration
