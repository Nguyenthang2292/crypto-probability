# Portfolio Manager Documentation

Tài liệu cho portfolio management component.

## Overview

Portfolio Manager component cung cấp các công cụ để quản lý và phân tích portfolio:
- Correlation analysis
- Risk calculation (PnL, Delta, Beta, VaR)
- Hedge finding

## Components

### Correlation Analyzer
**[PortfolioCorrelationAnalyzer.md](./PortfolioCorrelationAnalyzer.md)**

Phân tích correlation giữa:
- Portfolio internal correlation (giữa các positions)
- Portfolio với new symbols
- Weighted correlation dựa trên position sizes

### Risk Calculator
- **Location:** `modules/portfolio/risk_calculator.py`
- Tính toán PnL, Delta, Beta-weighted Delta
- Value at Risk (VaR) calculation
- Beta calculation so với benchmark

### Hedge Finder
- **Location:** `modules/portfolio/hedge_finder.py`
- Tìm hedge candidates dựa trên correlation
- Phân tích impact của hedge lên portfolio risk

## Usage

```bash
python portfolio_manager_main.py
```

## Configuration

Tất cả config được định nghĩa trong `modules/config.py` section **Portfolio Manager Configuration**:
- `BENCHMARK_SYMBOL` - Default benchmark cho beta calculation
- `DEFAULT_BETA_MIN_POINTS` - Minimum data points cho beta
- `DEFAULT_VAR_CONFIDENCE` - VaR confidence level
- `DEFAULT_CORRELATION_MIN_POINTS` - Minimum points cho correlation

## Features

- Real-time portfolio monitoring
- Correlation-based hedging
- Risk metrics calculation
- Multi-position analysis

## Related Documentation

- [Common Utilities](../common/) - DataFetcher, ExchangeManager
- [Config](../../modules/config.py) - Portfolio configuration

