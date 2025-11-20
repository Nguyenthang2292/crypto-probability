# 🚀 ExchangeManager Improvements

## Tổng quan

Đã cải tiến `AuthenticatedExchangeManager` để:
1. ✅ Hỗ trợ nhiều exchanges cần credentials (OKX, KuCoin, Bybit, Gate, MEXC, Huobi, ...)
2. ✅ Cấu hình hóa `defaultType` thay vì hardcode 'future'

---

## Thay đổi chính

### 1. Thêm config `DEFAULT_CONTRACT_TYPE` vào `config.py`

```python
DEFAULT_CONTRACT_TYPE = 'future'  # Options: 'spot', 'margin', 'future'
```

Có thể override qua environment variable:
```bash
export DEFAULT_CONTRACT_TYPE=spot
```

### 2. Method mới: `connect_to_exchange_with_credentials()`

Hỗ trợ kết nối đến bất kỳ exchange nào với credentials:

```python
from modules.ExchangeManager import AuthenticatedExchangeManager

# Khởi tạo
auth_manager = AuthenticatedExchangeManager(
    api_key="your_binance_key",  # Default cho Binance
    api_secret="your_binance_secret",
    contract_type='future'  # hoặc 'spot', 'margin'
)

# Kết nối Binance (backward compatible)
binance = auth_manager.connect_to_binance_with_credentials()

# Kết nối OKX
okx = auth_manager.connect_to_exchange_with_credentials(
    'okx',
    api_key='your_okx_key',
    api_secret='your_okx_secret'
)

# Kết nối KuCoin
kucoin = auth_manager.connect_to_exchange_with_credentials(
    'kucoin',
    api_key='your_kucoin_key',
    api_secret='your_kucoin_secret'
)

# Kết nối Bybit
bybit = auth_manager.connect_to_exchange_with_credentials(
    'bybit',
    api_key='your_bybit_key',
    api_secret='your_bybit_secret'
)
```

### 3. Quản lý credentials per-exchange

Có thể set credentials cho từng exchange một lần:

```python
# Set credentials cho OKX
auth_manager.set_exchange_credentials(
    'okx',
    api_key='your_okx_key',
    api_secret='your_okx_secret'
)

# Sau đó chỉ cần gọi
okx = auth_manager.connect_to_exchange_with_credentials('okx')
```

### 4. Sử dụng contract type từ config

Tất cả exchanges (authenticated và public) đều sử dụng `DEFAULT_CONTRACT_TYPE` từ config:

```python
# config.py
DEFAULT_CONTRACT_TYPE = 'spot'  # Thay đổi thành spot trading

# Hoặc qua environment variable
export DEFAULT_CONTRACT_TYPE=margin
```

---

## Ví dụ sử dụng

### Ví dụ 1: Multi-exchange portfolio

```python
from modules.ExchangeManager import ExchangeManager

# Khởi tạo
em = ExchangeManager(
    api_key="binance_key",
    api_secret="binance_secret"
)

# Set credentials cho các exchanges khác
em.authenticated.set_exchange_credentials('okx', 'okx_key', 'okx_secret')
em.authenticated.set_exchange_credentials('kucoin', 'kucoin_key', 'kucoin_secret')

# Lấy positions từ nhiều exchanges
binance = em.authenticated.connect_to_exchange_with_credentials('binance')
okx = em.authenticated.connect_to_exchange_with_credentials('okx')
kucoin = em.authenticated.connect_to_exchange_with_credentials('kucoin')

binance_positions = binance.fetch_positions()
okx_positions = okx.fetch_positions()
kucoin_positions = kucoin.fetch_positions()
```

### Ví dụ 2: Spot trading

```python
# config.py
DEFAULT_CONTRACT_TYPE = 'spot'

# Hoặc khi khởi tạo
auth_manager = AuthenticatedExchangeManager(
    api_key="your_key",
    api_secret="your_secret",
    contract_type='spot'  # Override config
)

# Kết nối với spot trading
exchange = auth_manager.connect_to_exchange_with_credentials('binance')
```

### Ví dụ 3: Testnet

```python
auth_manager = AuthenticatedExchangeManager(
    api_key="testnet_key",
    api_secret="testnet_secret",
    testnet=True
)

# Tất cả exchanges sẽ dùng testnet
binance = auth_manager.connect_to_exchange_with_credentials('binance', testnet=True)
okx = auth_manager.connect_to_exchange_with_credentials('okx', testnet=True)
```

---

## Backward Compatibility

✅ Tất cả code cũ vẫn hoạt động:

```python
# Code cũ vẫn hoạt động
em = ExchangeManager(api_key="key", api_secret="secret")
binance = em.get_binance_exchange_instance()  # ✅ Vẫn hoạt động
binance = em.authenticated.connect_to_binance_with_credentials()  # ✅ Vẫn hoạt động
```

---

## Supported Exchanges

Tất cả exchanges được hỗ trợ bởi ccxt đều có thể dùng với `connect_to_exchange_with_credentials()`:

- ✅ Binance
- ✅ OKX (OKEx)
- ✅ KuCoin
- ✅ Bybit
- ✅ Gate.io
- ✅ MEXC
- ✅ Huobi
- ✅ Và nhiều exchanges khác...

---

## Environment Variables

Có thể cấu hình qua environment variables:

```bash
# Contract type
export DEFAULT_CONTRACT_TYPE=spot

# Binance credentials (default)
export BINANCE_API_KEY=your_key
export BINANCE_API_SECRET=your_secret

# Other exchanges
export OKX_API_KEY=your_okx_key
export OKX_API_SECRET=your_okx_secret
export KUCOIN_API_KEY=your_kucoin_key
export KUCOIN_API_SECRET=your_kucoin_secret
```

---

## API Reference

### `AuthenticatedExchangeManager.connect_to_exchange_with_credentials()`

```python
def connect_to_exchange_with_credentials(
    self, 
    exchange_id: str,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    testnet: Optional[bool] = None,
    contract_type: Optional[str] = None
) -> ccxt.Exchange
```

**Parameters:**
- `exchange_id`: Exchange name (e.g., 'binance', 'okx', 'kucoin')
- `api_key`: API key (optional, uses default or per-exchange credentials)
- `api_secret`: API secret (optional, uses default or per-exchange credentials)
- `testnet`: Use testnet (optional, uses instance default)
- `contract_type`: 'spot', 'margin', or 'future' (optional, uses config default)

**Returns:**
- `ccxt.Exchange`: Authenticated exchange instance

### `AuthenticatedExchangeManager.set_exchange_credentials()`

```python
def set_exchange_credentials(
    self,
    exchange_id: str,
    api_key: str,
    api_secret: str
)
```

**Parameters:**
- `exchange_id`: Exchange name
- `api_key`: API key for this exchange
- `api_secret`: API secret for this exchange

---

## Migration Guide

### Từ code cũ sang code mới

**Trước:**
```python
em = ExchangeManager(api_key="key", api_secret="secret")
binance = em.get_binance_exchange_instance()
```

**Sau (tương thích ngược):**
```python
# Vẫn hoạt động như cũ
em = ExchangeManager(api_key="key", api_secret="secret")
binance = em.get_binance_exchange_instance()

# Hoặc dùng method mới
binance = em.authenticated.connect_to_exchange_with_credentials('binance')
```

**Thêm exchanges mới:**
```python
# Set credentials
em.authenticated.set_exchange_credentials('okx', 'okx_key', 'okx_secret')

# Kết nối
okx = em.authenticated.connect_to_exchange_with_credentials('okx')
```

---

## Notes

- ✅ Tất cả exchanges được cache để tránh tạo lại instance
- ✅ Credentials được lưu per-exchange để dễ quản lý
- ✅ Hỗ trợ testnet cho Binance, OKX, KuCoin, Bybit
- ✅ Contract type có thể override per-connection hoặc dùng config default
- ✅ Backward compatible 100% với code cũ

