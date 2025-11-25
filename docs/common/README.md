# Common Utilities Documentation

Tài liệu cho các modules dùng chung (shared utilities) được sử dụng bởi tất cả components.

## Modules

### DataFetcher
**[DataFetcher.md](./DataFetcher.md)**

Class để fetch dữ liệu OHLCV và giá hiện tại từ các exchanges:
- Fetch OHLCV data với fallback exchanges
- Fetch current prices từ Binance
- List futures symbols từ Binance
- Cache OHLCV data để tối ưu performance

**Sử dụng bởi:**
- XGBoost prediction
- Portfolio manager
- Deep learning
- Pairs trading

### ExchangeManager
**[ExchangeManager.md](./ExchangeManager.md)**

Quản lý kết nối với các exchanges:
- Public connections (không cần credentials)
- Authenticated connections (cần API keys)
- Rate limiting và throttling
- Fallback mechanism

**Sử dụng bởi:**
- Tất cả components cần fetch data từ exchanges

## Related Modules

Các modules khác trong `modules/common/`:
- `IndicatorEngine` - Tính toán technical indicators
- `utils` - Utility functions (normalize_symbol, format_price, etc.)
- `ProgressBar` - Hiển thị progress bar
- `Position` - Position data structure

