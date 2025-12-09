# Giải thích cách lấy Signal của Range Oscillator

## Tổng quan

Range Oscillator signal được tạo ra qua 3 bước chính:
1. **Tính toán Range Oscillator indicator** (giá trị oscillator từ -100 đến +100)
2. **Áp dụng các strategies** để tạo signals (LONG/SHORT/NEUTRAL)
3. **Voting mechanism** để kết hợp signals từ nhiều strategies

---

## Bước 1: Tính toán Range Oscillator Indicator

### 1.1. Tính Weighted Moving Average (MA)
```python
# Từ close prices, tính weighted MA dựa trên price deltas
ma = calculate_weighted_ma(close, length=50)
```

**Công thức:**
- Với mỗi bar, tính `delta = |close[i] - close[i+1]|`
- Weight `w = delta / close[i+1]`
- Weighted MA = `Σ(close[i] * w) / Σ(w)`

**Mục đích:** Nhấn mạnh các bar có biến động lớn hơn, tạo MA responsive hơn.

### 1.2. Tính ATR Range
```python
# Tính ATR (Average True Range) và nhân với multiplier
range_atr = calculate_atr_range(high, low, close, mult=2.0)
```

**Công thức:**
- ATR = Average True Range với length 2000 (fallback 200)
- Range ATR = `ATR * mult` (default: 2.0)

**Mục đích:** Xác định độ rộng của range bands, adapt với volatility.

### 1.3. Tính Oscillator Value
```python
# Với mỗi bar:
osc_value = 100 * (close - MA) / RangeATR
```

**Giá trị:**
- **+100**: Price ở upper bound của range (breakout lên trên)
- **0**: Price ở equilibrium (MA)
- **-100**: Price ở lower bound của range (breakout xuống dưới)

**Ví dụ:**
- Nếu `close = 50000`, `MA = 49000`, `RangeATR = 2000`
- `oscillator = 100 * (50000 - 49000) / 2000 = 50` (bullish, ở giữa range)

---

## Bước 2: Áp dụng Strategies để tạo Signals

Mỗi strategy phân tích oscillator và tạo signals dựa trên logic khác nhau:

### Strategy 5: Combined (Sustained + Crossover + Momentum)
```python
signals, signal_strength = generate_signals_combined_all_strategy(
    high=high, low=low, close=close,
    length=50, mult=2.0,
    use_sustained=True,    # Strategy 2
    use_crossover=True,    # Strategy 3
    use_momentum=True,     # Strategy 4
)
```

**Logic:**
- **Strategy 2 (Sustained)**: Oscillator ở trên/dưới 0 trong N bars → LONG/SHORT
- **Strategy 3 (Crossover)**: Oscillator cắt zero line → LONG/SHORT
- **Strategy 4 (Momentum)**: Rate of change của oscillator → LONG/SHORT
- **Voting**: Majority vote từ 3 strategies

**Signal:**
- `1` = LONG (bullish)
- `-1` = SHORT (bearish)
- `0` = NEUTRAL

### Strategy 6: Breakout
```python
signals, signal_strength = generate_signals_strategy6_breakout(
    high=high, low=low, close=close,
    length=50, mult=2.0,
)
```

**Logic:**
- Phát hiện khi oscillator **breakout** khỏi extreme thresholds (±100)
- **LONG**: Oscillator breakout lên trên +100
- **SHORT**: Oscillator breakout xuống dưới -100
- Forward fill signal trong khi oscillator vẫn ở breakout zone

### Strategy 7: Divergence
```python
signals, signal_strength = generate_signals_divergence_strategy(
    high=high, low=low, close=close,
    length=50, mult=2.0,
)
```

**Logic:**
- Phát hiện **divergence** giữa price và oscillator:
  - **Bearish Divergence**: Price tạo higher high, nhưng oscillator tạo lower high → SHORT
  - **Bullish Divergence**: Price tạo lower low, nhưng oscillator tạo higher low → LONG

### Strategy 8: Trend Following
```python
signals, signal_strength = generate_signals_trend_following_strategy(
    high=high, low=low, close=close,
    length=50, mult=2.0,
)
```

**Logic:**
- Theo trend với consistent oscillator position:
  - **LONG**: Oscillator ở trên 0 và trend là bullish
  - **SHORT**: Oscillator ở dưới 0 và trend là bearish

### Strategy 9: Mean Reversion
```python
signals, signal_strength = generate_signals_mean_reversion_strategy(
    high=high, low=low, close=close,
    length=50, mult=2.0,
)
```

**Logic:**
- Phát hiện mean reversion khi oscillator từ extreme về zero:
  - **SHORT**: Oscillator ở extreme positive (>+80), bắt đầu về zero
  - **LONG**: Oscillator ở extreme negative (<-80), bắt đầu về zero

---

## Bước 3: Lấy Latest Signal từ mỗi Strategy

Sau khi mỗi strategy tạo ra một Series signals (với mỗi bar có giá trị 1/-1/0), ta lấy **latest signal** (signal của bar cuối cùng):

```python
# Ví dụ với Strategy 5:
signals, signal_strength = generate_signals_combined_all_strategy(...)

# Lấy latest signal (bỏ qua NaN values)
non_nan_signals = signals.dropna()
if len(non_nan_signals) > 0:
    latest_signal = int(non_nan_signals.iloc[-1])  # Bar cuối cùng
    # latest_signal = 1 (LONG), -1 (SHORT), hoặc 0 (NEUTRAL)
```

**Ví dụ:**
```
Bar 1: NaN
Bar 2: NaN
Bar 3: 0
Bar 4: 1
Bar 5: 1
Bar 6: 1  ← latest_signal = 1 (LONG)
```

---

## Bước 4: Voting Mechanism (Kết hợp nhiều Strategies)

Sau khi có signals từ tất cả strategies, ta sử dụng **voting mechanism**:

```python
# Collect signals từ tất cả strategies
strategy_signals = []  # Ví dụ: [1, 1, -1, 1, 0]

# Đếm votes
long_votes = sum(1 for s in strategy_signals if s == 1)   # 3 votes
short_votes = sum(1 for s in strategy_signals if s == -1)  # 1 vote
total_votes = len(strategy_signals)  # 5 strategies

# Tính consensus
long_consensus = long_votes / total_votes   # 3/5 = 0.6 (60%)
short_consensus = short_votes / total_votes # 1/5 = 0.2 (20%)

# Kiểm tra consensus threshold (default: 0.5 = 50%)
if long_consensus >= 0.5:
    return 1  # LONG (3/5 strategies đồng ý)
elif short_consensus >= 0.5:
    return -1  # SHORT
else:
    return 0  # NEUTRAL (không đủ consensus)
```

**Ví dụ cụ thể:**
- **Strategies**: [5, 6, 7, 8, 9]
- **Signals từ mỗi strategy**: [1, 1, -1, 1, 0]
  - Strategy 5: LONG (1)
  - Strategy 6: LONG (1)
  - Strategy 7: SHORT (-1)
  - Strategy 8: LONG (1)
  - Strategy 9: NEUTRAL (0)
- **Votes**: 3 LONG, 1 SHORT, 1 NEUTRAL
- **Consensus**: 60% LONG → **Final Signal = 1 (LONG)**

---

## Flow hoàn chỉnh trong Code

```python
def get_range_oscillator_signal(...):
    # 1. Fetch OHLCV data
    df = data_fetcher.fetch_ohlcv_with_fallback_exchange(...)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    # 2. Chạy từng strategy
    strategy_signals = []
    for strategy_num in [5, 6, 7, 8, 9]:
        signals, _ = generate_signals_strategyX(...)
        latest_signal = int(signals.dropna().iloc[-1])
        strategy_signals.append(latest_signal)
    
    # 3. Voting mechanism
    long_votes = sum(1 for s in strategy_signals if s == 1)
    short_votes = sum(1 for s in strategy_signals if s == -1)
    total_votes = len(strategy_signals)
    
    long_consensus = long_votes / total_votes
    short_consensus = short_votes / total_votes
    
    # 4. Kiểm tra consensus threshold
    if long_consensus >= 0.5:
        return 1  # LONG
    elif short_consensus >= 0.5:
        return -1  # SHORT
    else:
        return 0  # NEUTRAL
```

---

## Tóm tắt

1. **Tính Oscillator**: `oscillator = 100 * (close - MA) / RangeATR` → giá trị -100 đến +100
2. **Áp dụng Strategies**: Mỗi strategy phân tích oscillator và tạo signals (1/-1/0)
3. **Lấy Latest Signal**: Lấy signal của bar cuối cùng từ mỗi strategy
4. **Voting**: Đếm votes và kiểm tra consensus threshold
5. **Final Signal**: Trả về 1 (LONG), -1 (SHORT), hoặc 0 (NEUTRAL)

**Lợi ích của voting mechanism:**
- ✅ Tăng độ chính xác (nhiều strategies cùng xác nhận)
- ✅ Giảm false signals (cần consensus)
- ✅ Robust (nếu 1 strategy fail, các strategies khác vẫn hoạt động)
- ✅ Linh hoạt (có thể chọn strategies và threshold)

