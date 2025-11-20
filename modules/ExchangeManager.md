# 📚 ExchangeManager Documentation

## Mục lục
1. [Tổng quan](#tổng-quan)
2. [AuthenticatedExchangeManager](#authenticatedexchangemanager)
3. [PublicExchangeManager](#publicexchangemanager)
4. [ExchangeManager (Composite)](#exchangemanager-composite)
5. [Ví dụ sử dụng](#ví-dụ-sử-dụng)
6. [Best Practices](#best-practices)

---

## Tổng quan

`ExchangeManager` là một hệ thống quản lý kết nối đến các sàn giao dịch crypto (exchanges) thông qua thư viện `ccxt`. Hệ thống được thiết kế với 3 lớp:

1. **AuthenticatedExchangeManager**: Quản lý các kết nối cần credentials (API key/secret)
2. **PublicExchangeManager**: Quản lý các kết nối không cần credentials (public data)
3. **ExchangeManager**: Composite manager kết hợp cả 2 managers trên

### Khi nào dùng gì?

| Loại dữ liệu | Cần credentials? | Dùng manager nào? |
|--------------|------------------|-------------------|
| Giá hiện tại (ticker) | ✅ Có | `authenticated.connect_to_binance_with_credentials()` |
| Danh sách symbols (markets) | ✅ Có | `authenticated.connect_to_binance_with_credentials()` |
| Positions từ account | ✅ Có | `authenticated.connect_to_binance_with_credentials()` |
| Dữ liệu OHLCV (lịch sử) | ❌ Không | `public.connect_to_exchange_with_no_credentials()` |
| Dữ liệu public khác | ❌ Không | `public.connect_to_exchange_with_no_credentials()` |

---

## AuthenticatedExchangeManager

### Mục đích
Quản lý các kết nối exchange **cần xác thực** (authentication) thông qua API key và secret. Dùng cho các operations liên quan đến account của bạn.

### Khởi tạo

```python
from modules.ExchangeManager import AuthenticatedExchangeManager

# Cách 1: Truyền credentials trực tiếp
auth_manager = AuthenticatedExchangeManager(
    api_key="your_api_key",
    api_secret="your_api_secret",
    testnet=False  # True nếu dùng testnet
)

# Cách 2: Lấy từ environment variables hoặc config file
auth_manager = AuthenticatedExchangeManager()  # Tự động lấy từ env/config
```

**Thứ tự ưu tiên lấy credentials:**
1. Tham số khi khởi tạo
2. Biến môi trường: `BINANCE_API_KEY`, `BINANCE_API_SECRET`
3. File config: `modules/config_api.py`

### Phương thức

#### `connect_to_exchange_with_credentials(exchange_id, ...) -> ccxt.Exchange`

**Mục đích**: Kết nối đến bất kỳ exchange nào đã được xác thực (authenticated) - YÊU CẦU credentials.

**Hỗ trợ các exchanges**: binance, okx, kucoin, bybit, gate, mexc, huobi, kraken, và tất cả exchanges được hỗ trợ bởi ccxt.

**Khi nào dùng:**
- ✅ Lấy giá hiện tại (`fetch_ticker`)
- ✅ Liệt kê danh sách symbols (`load_markets`)
- ✅ Lấy thông tin positions từ account (`fetch_positions`)
- ✅ Bất kỳ API call nào cần authentication

**Tham số:**
- `exchange_id` (str): Tên exchange (e.g., 'binance', 'okx', 'kucoin', 'bybit')
- `api_key` (Optional[str]): API key cho exchange này (optional)
- `api_secret` (Optional[str]): API secret cho exchange này (optional)
- `testnet` (Optional[bool]): Dùng testnet nếu True (optional)
- `contract_type` (Optional[str]): Loại contract ('spot', 'margin', 'future') (optional)

**Ví dụ:**
```python
# Kết nối đến OKX
okx = auth_manager.connect_to_exchange_with_credentials('okx', 
    api_key='okx_key', 
    api_secret='okx_secret'
)

# Kết nối đến KuCoin với testnet
kucoin = auth_manager.connect_to_exchange_with_credentials('kucoin',
    api_key='kucoin_key',
    api_secret='kucoin_secret',
    testnet=True
)

# Kết nối đến Bybit với spot trading
bybit = auth_manager.connect_to_exchange_with_credentials('bybit',
    api_key='bybit_key',
    api_secret='bybit_secret',
    contract_type='spot'
)
```

**Lưu ý:**
- ⚠️ **Bắt buộc** phải có API key và secret (có thể set qua `set_exchange_credentials()` hoặc truyền trực tiếp)
- ⚠️ Nếu không có credentials, sẽ raise `ValueError`
- ✅ Instance được cache, chỉ tạo một lần (lazy initialization)
- ✅ Tự động enable rate limiting
- ✅ Hỗ trợ testnet cho Binance, OKX, KuCoin, Bybit, Gate

---

#### `set_exchange_credentials(exchange_id, api_key, api_secret)`

**Mục đích**: Set credentials cho một exchange cụ thể để dùng sau này.

**Khi nào dùng:**
- ✅ Khi muốn set credentials một lần và dùng nhiều lần
- ✅ Khi quản lý credentials cho nhiều exchanges

**Ví dụ:**
```python
# Set credentials cho OKX
auth_manager.set_exchange_credentials('okx', 'okx_key', 'okx_secret')

# Set credentials cho KuCoin
auth_manager.set_exchange_credentials('kucoin', 'kucoin_key', 'kucoin_secret')

# Sau đó có thể dùng mà không cần truyền credentials
okx = auth_manager.connect_to_exchange_with_credentials('okx')
kucoin = auth_manager.connect_to_exchange_with_credentials('kucoin')
```

**Lưu ý:**
- ✅ Credentials được lưu per-exchange
- ✅ Khi set credentials mới, cache của exchange đó sẽ bị clear để force reconnection

---

#### `connect_to_binance_with_credentials() -> ccxt.Exchange`

**Mục đích**: Kết nối đến Binance exchange đã được xác thực (authenticated) - YÊU CẦU credentials.

**DEPRECATED**: Nên dùng `connect_to_exchange_with_credentials('binance')` thay thế. Giữ lại để backward compatibility.

**Khi nào dùng:**
- ✅ Lấy giá hiện tại (`fetch_ticker`)
- ✅ Liệt kê danh sách symbols (`load_markets`)
- ✅ Lấy thông tin positions từ account (`fetch_positions`)
- ✅ Bất kỳ API call nào cần authentication

**Ví dụ:**
```python
# Kết nối đến authenticated Binance exchange (cần credentials)
exchange = auth_manager.connect_to_binance_with_credentials()

# Lấy giá hiện tại của BTC/USDT
ticker = exchange.fetch_ticker("BTC/USDT")
print(f"Giá hiện tại: {ticker['last']}")

# Liệt kê tất cả markets
markets = exchange.load_markets()
print(f"Tổng số markets: {len(markets)}")

# Lấy positions từ account
positions = exchange.fetch_positions()
for pos in positions:
    print(f"Symbol: {pos['symbol']}, Size: {pos['size']}")
```

**Lưu ý:**
- ⚠️ **Bắt buộc** phải có API key và secret
- ⚠️ Nếu không có credentials, sẽ raise `ValueError`
- ✅ Instance được cache, chỉ tạo một lần (lazy initialization)
- ✅ Tự động enable rate limiting

**Lỗi có thể gặp:**
```python
# Nếu không có credentials
try:
    exchange = auth_manager.connect_to_binance_with_credentials()
except ValueError as e:
    print(e)  # "API Key và API Secret là bắt buộc..."
```

---

#### Convenience Methods cho các Exchanges

Các phương thức tiện lợi để kết nối đến các exchanges phổ biến:

- `connect_to_kraken_with_credentials(api_key, api_secret, testnet, contract_type)`
- `connect_to_kucoin_with_credentials(api_key, api_secret, testnet, contract_type)`
- `connect_to_gate_with_credentials(api_key, api_secret, testnet, contract_type)`
- `connect_to_okx_with_credentials(api_key, api_secret, testnet, contract_type)`
- `connect_to_bybit_with_credentials(api_key, api_secret, testnet, contract_type)`
- `connect_to_mexc_with_credentials(api_key, api_secret, testnet, contract_type)`
- `connect_to_huobi_with_credentials(api_key, api_secret, testnet, contract_type)`

Tất cả các methods này đều là wrapper của `connect_to_exchange_with_credentials()` với exchange_id tương ứng.

**Ví dụ:**
```python
# Cách 1: Set credentials trước
auth_manager.set_exchange_credentials('okx', 'okx_key', 'okx_secret')
okx = auth_manager.connect_to_okx_with_credentials()

# Cách 2: Truyền credentials trực tiếp
kucoin = auth_manager.connect_to_kucoin_with_credentials(
    api_key='kucoin_key',
    api_secret='kucoin_secret'
)

# Cách 3: Với testnet và contract type
bybit = auth_manager.connect_to_bybit_with_credentials(
    api_key='bybit_key',
    api_secret='bybit_secret',
    testnet=True,
    contract_type='spot'
)
```

---

#### `throttled_call(func, *args, **kwargs)`

**Mục đích**: Gọi một hàm với rate limiting tự động để tránh vượt quá giới hạn API.

**Khi nào dùng:**
- ✅ Bất kỳ API call nào cần đảm bảo không vượt rate limit
- ✅ Khi gọi nhiều API calls liên tiếp
- ✅ Để tránh bị ban IP do quá nhiều requests

**Cách hoạt động:**
- Tự động tính toán thời gian chờ giữa các requests
- Đảm bảo mỗi request cách nhau ít nhất `request_pause` giây (mặc định 0.2s)
- Thread-safe (có thể dùng trong multi-threading)

**Ví dụ:**
```python
exchange = auth_manager.connect_to_binance_with_credentials()

# Gọi API với rate limiting
ticker = auth_manager.throttled_call(
    exchange.fetch_ticker,
    "BTC/USDT"
)

# Gọi nhiều API calls liên tiếp
symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
for symbol in symbols:
    ticker = auth_manager.throttled_call(
        exchange.fetch_ticker,
        symbol
    )
    print(f"{symbol}: {ticker['last']}")
```

**Tham số:**
- `func`: Hàm cần gọi (thường là method của exchange)
- `*args`: Các tham số vị trí cho hàm
- `**kwargs`: Các tham số keyword cho hàm

**Lưu ý:**
- ✅ Tự động sleep nếu cần để đảm bảo rate limit
- ✅ Thread-safe (dùng lock)
- ✅ Có thể điều chỉnh `request_pause` qua environment variable `BINANCE_REQUEST_SLEEP`

---

## PublicExchangeManager

### Mục đích
Quản lý các kết nối exchange **không cần xác thực** (public data). Dùng cho các operations lấy dữ liệu công khai.

### Khởi tạo

```python
from modules.ExchangeManager import PublicExchangeManager

# Khởi tạo (không cần credentials)
public_manager = PublicExchangeManager()
```

### Phương thức

#### `connect_to_exchange_with_no_credentials(exchange_id: str) -> ccxt.Exchange`

**Mục đích**: Kết nối đến một exchange công khai (KHÔNG cần credentials).

**Khi nào dùng:**
- ✅ Lấy dữ liệu OHLCV (lịch sử giá)
- ✅ Lấy dữ liệu public khác
- ✅ Khi cần fallback sang exchange khác nếu Binance không có dữ liệu

**Ví dụ:**
```python
# Kết nối đến Binance public (không cần credentials)
binance = public_manager.connect_to_exchange_with_no_credentials("binance")
ohlcv = binance.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=100)

# Kết nối đến Kraken public
kraken = public_manager.connect_to_exchange_with_no_credentials("kraken")
ohlcv = kraken.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=100)

# Kết nối đến các exchange khác
kucoin = public_manager.connect_to_exchange_with_no_credentials("kucoin")
gate = public_manager.connect_to_exchange_with_no_credentials("gate")
okx = public_manager.connect_to_exchange_with_no_credentials("okx")
```

**Tham số:**
- `exchange_id` (str): Tên exchange (ví dụ: "binance", "kraken", "kucoin", "gate", "okx", "bybit", "mexc", "huobi")

**Lưu ý:**
- ✅ **Không cần** API key/secret
- ✅ Instance được cache, chỉ tạo một lần cho mỗi exchange
- ✅ Tự động enable rate limiting
- ✅ Tự động set `defaultType: 'future'` cho futures trading

**Lỗi có thể gặp:**
```python
# Nếu exchange không được hỗ trợ
try:
    exchange = public_manager.connect_to_exchange_with_no_credentials("unknown_exchange")
except ValueError as e:
    print(e)  # "Exchange 'unknown_exchange' is not supported by ccxt."
```

**Các exchange được hỗ trợ:**
- `binance` - Binance
- `kraken` - Kraken
- `kucoin` - KuCoin
- `gate` - Gate.io
- `okx` - OKX
- `bybit` - Bybit
- `mexc` - MEXC
- `huobi` - Huobi
- Và tất cả exchanges được hỗ trợ bởi ccxt

---

#### `throttled_call(func, *args, **kwargs)`

**Mục đích**: Tương tự như `AuthenticatedExchangeManager.throttled_call()`, nhưng dùng cho public calls.

**Ví dụ:**
```python
exchange = public_manager.connect_to_exchange_with_no_credentials("kraken")

# Gọi API với rate limiting
ohlcv = public_manager.throttled_call(
    exchange.fetch_ohlcv,
    "BTC/USDT",
    timeframe="1h",
    limit=100
)
```

---

#### `exchange_priority_for_fallback` (property)

**Mục đích**: Danh sách các exchange theo thứ tự ưu tiên khi cần fallback.

**Ví dụ:**
```python
# Xem danh sách ưu tiên hiện tại
print(public_manager.exchange_priority_for_fallback)
# Output: ['binance', 'kraken', 'kucoin', 'gate', 'okx', 'bybit', 'mexc', 'huobi']

# Thay đổi thứ tự ưu tiên
public_manager.exchange_priority_for_fallback = ['kraken', 'binance', 'kucoin']

# Hoặc lấy từ environment variable
# Set OHLCV_FALLBACKS="kraken,binance,kucoin"
```

**Cách sử dụng trong fallback:**
```python
# Thử lấy OHLCV từ các exchange theo thứ tự ưu tiên
for exchange_id in public_manager.exchange_priority_for_fallback:
    try:
        exchange = public_manager.connect_to_exchange_with_no_credentials(exchange_id)
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=100)
        if ohlcv:
            print(f"Successfully fetched from {exchange_id}")
            break
    except Exception as e:
        print(f"Failed to fetch from {exchange_id}: {e}")
        continue
```

**Lưu ý:**
- ✅ Có thể set qua environment variable `OHLCV_FALLBACKS`
- ✅ Mặc định: `"binance,kraken,kucoin,gate,okx,bybit,mexc,huobi"`
- ✅ Tương đương với `em.exchange_priority_for_fallback` (trong ExchangeManager)

---

## ExchangeManager (Composite)

### Mục đích
Composite manager kết hợp cả `AuthenticatedExchangeManager` và `PublicExchangeManager`, cung cấp interface thống nhất và giữ backward compatibility.

### Khởi tạo

```python
from modules.ExchangeManager import ExchangeManager

# Khởi tạo với credentials
em = ExchangeManager(
    api_key="your_api_key",
    api_secret="your_api_secret",
    testnet=False
)

# Hoặc không có credentials (chỉ dùng public)
em = ExchangeManager()
```

### Cấu trúc

```python
em = ExchangeManager(api_key, api_secret)

# Truy cập authenticated manager
em.authenticated  # AuthenticatedExchangeManager instance

# Truy cập public manager
em.public  # PublicExchangeManager instance
```

### Phương thức

#### `get_binance_exchange_instance() -> ccxt.Exchange`

**Mục đích**: Lấy authenticated Binance exchange instance (backward compatibility).

**Ví dụ:**
```python
# Cách mới (khuyến nghị)
exchange = em.authenticated.connect_to_binance_with_credentials()

# Cách cũ (vẫn hoạt động)
exchange = em.get_binance_exchange_instance()  # → em.authenticated.connect_to_binance_with_credentials()
```

**Lưu ý:**
- ⚠️ DEPRECATED: Nên dùng `em.authenticated.connect_to_binance_with_credentials()` thay thế
- ✅ Vẫn hoạt động để giữ backward compatibility

---

#### `get_exchange_instance(exchange_id: str) -> ccxt.Exchange`

**Mục đích**: Lấy public exchange instance cho OHLCV data (backward compatibility).

**Ví dụ:**
```python
# Cách mới (khuyến nghị)
exchange = em.public.connect_to_exchange_with_no_credentials("kraken")

# Cách cũ (vẫn hoạt động)
exchange = em.get_exchange_instance("kraken")  # → em.public.connect_to_exchange_with_no_credentials("kraken")
```

**Lưu ý:**
- ⚠️ DEPRECATED: Nên dùng `em.public.connect_to_exchange_with_no_credentials()` thay thế
- ✅ Vẫn hoạt động để giữ backward compatibility

---

#### `throttled_call(func, *args, **kwargs)`

**Mục đích**: Throttled call (backward compatibility).

**Ví dụ:**
```python
# Cách mới (khuyến nghị)
result = em.authenticated.throttled_call(exchange.fetch_ticker, "BTC/USDT")
# hoặc
result = em.public.throttled_call(exchange.fetch_ohlcv, "BTC/USDT", timeframe="1h")

# Cách cũ (vẫn hoạt động)
result = em.throttled_call(exchange.fetch_ticker, "BTC/USDT")  # → authenticated.throttled_call()
```

**Lưu ý:**
- ⚠️ DEPRECATED: Nên dùng `em.authenticated.throttled_call()` hoặc `em.public.throttled_call()` thay thế
- ✅ Mặc định dùng authenticated manager's throttled_call

---

#### `normalize_symbol(market_symbol: str) -> str`

**Mục đích**: Chuẩn hóa symbol từ Binance futures format.

**Ví dụ:**
```python
# Chuẩn hóa symbol
symbol1 = em.normalize_symbol("BTC/USDT:USDT")  # → "BTC/USDT"
symbol2 = em.normalize_symbol("ETHUSDT")        # → "ETH/USDT"
symbol3 = em.normalize_symbol("BNB/USDT")       # → "BNB/USDT"
```

**Khi nào dùng:**
- ✅ Khi nhận symbol từ Binance markets (có format `BTC/USDT:USDT`)
- ✅ Cần chuẩn hóa về format `BASE/QUOTE`

---

#### `exchange_priority_for_fallback` (property)

**Mục đích**: Danh sách exchange ưu tiên cho OHLCV fallback.

**Ví dụ:**
```python
# Xem danh sách
print(em.exchange_priority_for_fallback)

# Thay đổi
em.exchange_priority_for_fallback = ['kraken', 'binance', 'kucoin']
```

**Lưu ý:**
- ✅ Tương đương với `em.public.exchange_priority_for_fallback`
- ✅ Có thể set/get như property
- ✅ Được sử dụng cho OHLCV fallback mechanism

---

## Ví dụ sử dụng

### Ví dụ 1: Lấy giá hiện tại từ Binance (cần credentials)

```python
from modules.ExchangeManager import ExchangeManager

# Khởi tạo
em = ExchangeManager(api_key="...", api_secret="...")

# Kết nối đến authenticated Binance (cần credentials)
exchange = em.authenticated.connect_to_binance_with_credentials()

# Lấy giá với rate limiting
ticker = em.authenticated.throttled_call(
    exchange.fetch_ticker,
    "BTC/USDT"
)

print(f"Giá BTC/USDT: {ticker['last']}")
```

### Ví dụ 1b: Lấy giá từ nhiều exchanges

```python
from modules.ExchangeManager import ExchangeManager

# Khởi tạo
em = ExchangeManager(api_key="binance_key", api_secret="binance_secret")

# Set credentials cho các exchanges khác
em.authenticated.set_exchange_credentials('okx', 'okx_key', 'okx_secret')
em.authenticated.set_exchange_credentials('kucoin', 'kucoin_key', 'kucoin_secret')

# Lấy giá từ Binance
binance = em.authenticated.connect_to_binance_with_credentials()
binance_ticker = em.authenticated.throttled_call(
    binance.fetch_ticker, "BTC/USDT"
)

# Lấy giá từ OKX
okx = em.authenticated.connect_to_okx_with_credentials()
okx_ticker = em.authenticated.throttled_call(
    okx.fetch_ticker, "BTC/USDT"
)

# Lấy giá từ KuCoin
kucoin = em.authenticated.connect_to_kucoin_with_credentials()
kucoin_ticker = em.authenticated.throttled_call(
    kucoin.fetch_ticker, "BTC/USDT"
)

print(f"Binance: {binance_ticker['last']}")
print(f"OKX: {okx_ticker['last']}")
print(f"KuCoin: {kucoin_ticker['last']}")
```

### Ví dụ 2: Lấy dữ liệu OHLCV (không cần credentials)

```python
from modules.ExchangeManager import ExchangeManager

# Khởi tạo (không cần credentials)
em = ExchangeManager()

# Thử lấy từ các exchange theo thứ tự ưu tiên
for exchange_id in em.public.exchange_priority_for_fallback:
    try:
        exchange = em.public.connect_to_exchange_with_no_credentials(exchange_id)
        ohlcv = em.public.throttled_call(
            exchange.fetch_ohlcv,
            "BTC/USDT",
            timeframe="1h",
            limit=100
        )
        if ohlcv:
            print(f"✓ Lấy được {len(ohlcv)} candles từ {exchange_id}")
            break
    except Exception as e:
        print(f"✗ {exchange_id}: {e}")
        continue
```

### Ví dụ 3: Liệt kê symbols từ Binance (cần credentials)

```python
from modules.ExchangeManager import ExchangeManager

em = ExchangeManager(api_key="...", api_secret="...")

# Kết nối đến authenticated Binance (cần credentials)
exchange = em.authenticated.connect_to_binance_with_credentials()

# Load markets
markets = exchange.load_markets()

# Lọc futures USDT pairs
futures_usdt = [
    symbol for symbol, market in markets.items()
    if market.get('contract') and market.get('quote') == 'USDT'
]

print(f"Tổng số futures USDT pairs: {len(futures_usdt)}")
```

### Ví dụ 3b: Liệt kê symbols từ nhiều exchanges

```python
from modules.ExchangeManager import ExchangeManager

em = ExchangeManager(api_key="binance_key", api_secret="binance_secret")

# Set credentials cho OKX
em.authenticated.set_exchange_credentials('okx', 'okx_key', 'okx_secret')

# Lấy markets từ Binance
binance = em.authenticated.connect_to_binance_with_credentials()
binance_markets = binance.load_markets()
print(f"Binance markets: {len(binance_markets)}")

# Lấy markets từ OKX
okx = em.authenticated.connect_to_okx_with_credentials()
okx_markets = okx.load_markets()
print(f"OKX markets: {len(okx_markets)}")
```

### Ví dụ 4: Sử dụng trong DataFetcher

```python
from modules.ExchangeManager import ExchangeManager
from modules.DataFetcher import DataFetcher

# Khởi tạo
em = ExchangeManager(api_key="...", api_secret="...")
data_fetcher = DataFetcher(em)

# Fetch prices (dùng authenticated)
data_fetcher.fetch_prices(["BTC/USDT", "ETH/USDT"])

# Fetch OHLCV (dùng public)
ohlcv = data_fetcher.fetch_ohlcv("BTC/USDT", limit=100, timeframe="1h")
```

### Ví dụ 5: Multi-exchange portfolio management

```python
from modules.ExchangeManager import ExchangeManager

# Khởi tạo
em = ExchangeManager(api_key="binance_key", api_secret="binance_secret")

# Set credentials cho các exchanges khác
em.authenticated.set_exchange_credentials('okx', 'okx_key', 'okx_secret')
em.authenticated.set_exchange_credentials('bybit', 'bybit_key', 'bybit_secret')

# Lấy positions từ nhiều exchanges
binance = em.authenticated.connect_to_binance_with_credentials()
okx = em.authenticated.connect_to_okx_with_credentials()
bybit = em.authenticated.connect_to_bybit_with_credentials()

binance_positions = binance.fetch_positions()
okx_positions = okx.fetch_positions()
bybit_positions = bybit.fetch_positions()

print(f"Binance positions: {len(binance_positions)}")
print(f"OKX positions: {len(okx_positions)}")
print(f"Bybit positions: {len(bybit_positions)}")
```

---

## Best Practices

### 1. Phân biệt rõ authenticated vs public

```python
# ✅ ĐÚNG: Dùng authenticated cho authenticated calls
exchange = em.authenticated.connect_to_binance_with_credentials()
ticker = exchange.fetch_ticker("BTC/USDT")  # Cần credentials

# ❌ SAI: Dùng public cho authenticated calls
exchange = em.public.connect_to_exchange_with_no_credentials("binance")
ticker = exchange.fetch_ticker("BTC/USDT")  # Có thể fail hoặc không chính xác
```

### 2. Luôn dùng throttled_call cho API calls

```python
# ✅ ĐÚNG: Dùng throttled_call
ticker = em.authenticated.throttled_call(exchange.fetch_ticker, "BTC/USDT")

# ❌ SAI: Gọi trực tiếp (có thể vượt rate limit)
ticker = exchange.fetch_ticker("BTC/USDT")
```

### 3. Sử dụng fallback cho OHLCV

```python
# ✅ ĐÚNG: Thử nhiều exchange nếu một exchange fail
for exchange_id in em.public.exchange_priority_for_fallback:
    try:
        exchange = em.public.connect_to_exchange_with_no_credentials(exchange_id)
        ohlcv = em.public.throttled_call(
            exchange.fetch_ohlcv, "BTC/USDT", timeframe="1h", limit=100
        )
        if ohlcv:
            break
    except Exception:
        continue
```

### 4. Cache credentials an toàn

```python
# ✅ ĐÚNG: Lấy từ environment variables
em = ExchangeManager()  # Tự động lấy từ env

# ❌ SAI: Hardcode credentials trong code
em = ExchangeManager(api_key="hardcoded_key", api_secret="hardcoded_secret")
```

### 5. Xử lý lỗi đúng cách

```python
# ✅ ĐÚNG: Xử lý lỗi credentials
try:
    exchange = em.authenticated.connect_to_binance_with_credentials()
except ValueError as e:
    print(f"Lỗi credentials: {e}")
    # Fallback hoặc exit

# ✅ ĐÚNG: Xử lý lỗi exchange không hỗ trợ
try:
    exchange = em.public.connect_to_exchange_with_no_credentials("unknown")
except ValueError as e:
    print(f"Exchange không hỗ trợ: {e}")
    # Thử exchange khác
```

---

## Tóm tắt

| Manager | Khi nào dùng | Cần credentials? | Methods chính |
|---------|--------------|------------------|---------------|
| `AuthenticatedExchangeManager` | Lấy giá, markets, positions | ✅ Có | `connect_to_exchange_with_credentials()`, `connect_to_*_with_credentials()`, `set_exchange_credentials()`, `throttled_call()` |
| `PublicExchangeManager` | Lấy OHLCV, public data | ❌ Không | `connect_to_exchange_with_no_credentials()`, `throttled_call()` |
| `ExchangeManager` | Composite, backward compatibility | Tùy | Tất cả methods trên + `normalize_symbol()` |

### Supported Exchanges (Authenticated)

Các exchanges được hỗ trợ với convenience methods:
- ✅ Binance (`connect_to_binance_with_credentials()`)
- ✅ Kraken (`connect_to_kraken_with_credentials()`)
- ✅ KuCoin (`connect_to_kucoin_with_credentials()`)
- ✅ Gate.io (`connect_to_gate_with_credentials()`)
- ✅ OKX (`connect_to_okx_with_credentials()`)
- ✅ Bybit (`connect_to_bybit_with_credentials()`)
- ✅ MEXC (`connect_to_mexc_with_credentials()`)
- ✅ Huobi (`connect_to_huobi_with_credentials()`)

Hoặc dùng `connect_to_exchange_with_credentials(exchange_id)` cho bất kỳ exchange nào được hỗ trợ bởi ccxt.

---

## Liên kết

- [ccxt Documentation](https://docs.ccxt.com/)
- [Binance API Documentation](https://binance-docs.github.io/apidocs/)

