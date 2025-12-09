# Simplified Percentile Clustering

Một module clustering heuristic nhẹ, thân thiện với streaming, được thiết kế cho phân tích xu hướng. Module này tính toán các "cluster centers" đơn giản cho mỗi feature sử dụng percentiles và running mean, sau đó gán mỗi bar vào center gần nhất.

## Tổng quan

Module này port từ Pine Script indicator "Simplified Percentile Clustering" sang Python. Nó cung cấp:

- **K limited to 2 or 3**: Đảm bảo tính ổn định và dễ giải thích
- **Percentile + Mean Centers**: Sử dụng percentiles (lower/upper) + running mean để tạo centers xác định
- **Feature Fusion**: Cho phép kết hợp nhiều features (RSI, CCI, Fisher, DMI, Z-Score, MAR)
- **Interpolated Values**: Tạo giá trị `real_clust` liên tục giữa các centers (hữu ích để visualize 'proximity-to-flip')

## Cấu trúc Module

```text
simplified_percentile_clustering/
├── __init__.py              # Module exports
├── README.md                 # Tài liệu này
├── core/
│   ├── __init__.py          # Core exports
│   ├── features.py          # Tính toán features (RSI, CCI, Fisher, DMI, Z-Score, MAR)
│   ├── centers.py           # Tính toán cluster centers từ percentiles
│   └── clustering.py        # Logic clustering chính
├── strategies/
│   ├── __init__.py          # Strategy exports
│   ├── cluster_transition.py    # Cluster transition strategy
│   ├── regime_following.py      # Regime following strategy
│   └── mean_reversion.py        # Mean reversion strategy
└── pinescript               # File Pine Script gốc
```

## Sử dụng

### Ví dụ cơ bản

```python
import pandas as pd
from modules.simplified_percentile_clustering import compute_clustering, ClusteringConfig, FeatureConfig

# Chuẩn bị dữ liệu OHLCV
df = pd.DataFrame({
    'high': [...],
    'low': [...],
    'close': [...],
})

# Cấu hình
feature_config = FeatureConfig(
    use_rsi=True,
    rsi_len=14,
    rsi_standardize=True,
    use_cci=True,
    cci_len=20,
    cci_standardize=True,
    # ... các features khác
)

clustering_config = ClusteringConfig(
    k=2,                    # Số clusters (2 hoặc 3)
    lookback=1000,          # Số bars lịch sử
    p_low=5.0,             # Lower percentile
    p_high=95.0,           # Upper percentile
    main_plot="Clusters",   # "Clusters", "RSI", "CCI", "Fisher", "DMI", "Z-Score", "MAR"
    feature_config=feature_config,
)

# Tính toán clustering
result = compute_clustering(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    config=clustering_config,
)

# Kết quả
print(result.cluster_val)      # Cluster index (0, 1, hoặc 2)
print(result.curr_cluster)     # Cluster name ("k0", "k1", "k2")
print(result.real_clust)       # Interpolated cluster value
print(result.plot_val)         # Giá trị để plot
```

### Sử dụng với SimplifiedPercentileClustering class

```python
from modules.simplified_percentile_clustering import SimplifiedPercentileClustering, ClusteringConfig

clustering = SimplifiedPercentileClustering(config=clustering_config)
result = clustering.compute(df['high'], df['low'], df['close'])
```

## Features

Module hỗ trợ các features sau:

1. **RSI** (Relative Strength Index)
2. **CCI** (Commodity Channel Index)
3. **Fisher Transform**
4. **DMI** (Directional Movement Index - difference)
5. **Z-Score** (Z-score của giá)
6. **MAR** (Moving Average Ratio - giá chia cho MA)

Mỗi feature có thể được bật/tắt và có thể được chuẩn hóa (standardize) bằng z-score.

## Cấu hình

### FeatureConfig

- `use_rsi`, `use_cci`, `use_fisher`, `use_dmi`, `use_zscore`, `use_mar`: Bật/tắt features
- `rsi_len`, `cci_len`, `fisher_len`, `dmi_len`, `zscore_len`, `mar_len`: Độ dài cho mỗi indicator
- `rsi_standardize`, `cci_standardize`, ...: Có chuẩn hóa feature hay không
- `mar_type`: "SMA" hoặc "EMA" cho MAR

### ClusteringConfig

- `k`: Số cluster centers (2 hoặc 3)
- `lookback`: Số bars lịch sử để tính percentiles và mean
- `p_low`: Lower percentile (mặc định: 5.0)
- `p_high`: Upper percentile (mặc định: 95.0)
- `main_plot`: Chế độ hiển thị ("Clusters" cho combined mode, hoặc tên feature cho single-feature mode)

## Kết quả

`ClusteringResult` chứa:

- `cluster_val`: Chỉ số cluster rời rạc (0, 1, hoặc 2)
- `curr_cluster`: Tên cluster ("k0", "k1", "k2")
- `real_clust`: Giá trị cluster nội suy (liên tục)
- `min_dist`: Khoảng cách đến center gần nhất
- `second_min_dist`: Khoảng cách đến center gần thứ hai
- `rel_pos`: Vị trí tương đối giữa hai centers gần nhất
- `plot_val`: Giá trị để vẽ biểu đồ
- `plot_k0_center`, `plot_k1_center`, `plot_k2_center`: Các cluster centers
- `features`: Dictionary chứa tất cả các feature values

## Lưu ý

- Đây **KHÔNG phải** k-means clustering. Đây là một heuristic percentile + mean center được thiết kế để ưu tiên tính ổn định và tính toán nhẹ trên live series.
- Phù hợp cho feature engineering và visual regime detection.
- Nếu cần centroid updates dựa trên iterative assignment, hãy xem xét một k-means adaptation (ngoài phạm vi của heuristic đơn giản này).

## Trading Strategies

Module cung cấp 3 trading strategies dựa trên clustering để tạo trading signals từ cluster assignments và transitions.

### 1. Cluster Transition Strategy

**File**: `strategies/cluster_transition.py`

Strategy này tạo signals dựa trên sự chuyển đổi giữa các clusters. Khi thị trường chuyển từ cluster này sang cluster khác, nó có thể báo hiệu một regime change và cơ hội trading.

**Logic**:

- **LONG Signal**: Transition từ k0 (lower cluster) sang k1 hoặc k2 (higher clusters)
- **SHORT Signal**: Transition từ k2 hoặc k1 (higher clusters) sang k0 (lower cluster)
- **NEUTRAL Signal**: Không có transition hoặc transitions mơ hồ

**Cấu hình**:

```python
from modules.simplified_percentile_clustering.strategies import (
    ClusterTransitionConfig,
    generate_signals_cluster_transition,
)

config = ClusterTransitionConfig(
    require_price_confirmation=True,  # Yêu cầu giá di chuyển cùng hướng
    min_rel_pos_change=0.1,           # Thay đổi rel_pos tối thiểu
    use_real_clust_cross=True,        # Sử dụng real_clust crossing boundaries
    min_signal_strength=0.3,          # Độ mạnh signal tối thiểu
)

signals, strength, metadata = generate_signals_cluster_transition(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    config=config,
)
```

### 2. Regime Following Strategy

**File**: `strategies/regime_following.py`

Strategy này follow regime hiện tại và tạo signals khi thị trường đang ở trong một regime mạnh.

**Logic**:

- **LONG Signal**: Market ở k1 hoặc k2 cluster, real_clust cao, regime mạnh (rel_pos thấp)
- **SHORT Signal**: Market ở k0 cluster, real_clust thấp, regime mạnh
- **NEUTRAL Signal**: Regime yếu (rel_pos cao) hoặc đang transition

**Cấu hình**:

```python
from modules.simplified_percentile_clustering.strategies import (
    RegimeFollowingConfig,
    generate_signals_regime_following,
)

config = RegimeFollowingConfig(
    min_regime_strength=0.7,      # Độ mạnh regime tối thiểu (1 - rel_pos)
    min_cluster_duration=2,       # Số bars tối thiểu trong cùng cluster
    require_momentum=True,         # Yêu cầu momentum confirmation
    momentum_period=5,            # Period cho momentum calculation
)

signals, strength, metadata = generate_signals_regime_following(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    config=config,
)
```

### 3. Mean Reversion Strategy

**File**: `strategies/mean_reversion.py`

Strategy này tạo signals khi market ở cluster extremes và kỳ vọng mean reversion về center cluster.

**Logic**:

- **LONG Signal**: Market ở k0 (lower extreme), real_clust gần 0, kỳ vọng reversion lên
- **SHORT Signal**: Market ở k2 hoặc k1 (upper extreme), real_clust gần max, kỳ vọng reversion xuống
- **NEUTRAL Signal**: Market gần center cluster, không có extreme conditions

**Cấu hình**:

```python
from modules.simplified_percentile_clustering.strategies import (
    MeanReversionConfig,
    generate_signals_mean_reversion,
)

config = MeanReversionConfig(
    extreme_threshold=0.2,         # Ngưỡng real_clust cho extreme (0.0-1.0)
    min_extreme_duration=3,        # Số bars tối thiểu ở extreme
    require_reversal_signal=True,  # Yêu cầu reversal confirmation
    reversal_lookback=3,          # Bars để look back cho reversal
    min_signal_strength=0.4,       # Độ mạnh signal tối thiểu
)

signals, strength, metadata = generate_signals_mean_reversion(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    config=config,
)
```

### Ví dụ sử dụng tổng hợp

```python
import pandas as pd
from modules.simplified_percentile_clustering.core.clustering import (
    ClusteringConfig,
    FeatureConfig,
    compute_clustering,
)
from modules.simplified_percentile_clustering.strategies import (
    ClusterTransitionConfig,
    RegimeFollowingConfig,
    MeanReversionConfig,
    generate_signals_cluster_transition,
    generate_signals_regime_following,
    generate_signals_mean_reversion,
)

# Chuẩn bị dữ liệu
df = pd.DataFrame({
    'high': [...],
    'low': [...],
    'close': [...],
})

# Cấu hình clustering
feature_config = FeatureConfig(
    use_rsi=True,
    rsi_len=14,
    use_cci=True,
    cci_len=20,
    # ... các features khác
)

clustering_config = ClusteringConfig(
    k=2,
    lookback=1000,
    p_low=5.0,
    p_high=95.0,
    feature_config=feature_config,
)

# Tính toán clustering một lần (có thể tái sử dụng)
clustering_result = compute_clustering(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    config=clustering_config,
)

# Strategy 1: Cluster Transition
transition_config = ClusterTransitionConfig(
    clustering_config=clustering_config,
    require_price_confirmation=True,
    min_signal_strength=0.3,
)
signals_transition, strength_transition, meta_transition = generate_signals_cluster_transition(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    clustering_result=clustering_result,  # Tái sử dụng kết quả
    config=transition_config,
)

# Strategy 2: Regime Following
regime_config = RegimeFollowingConfig(
    clustering_config=clustering_config,
    min_regime_strength=0.7,
    min_cluster_duration=2,
)
signals_regime, strength_regime, meta_regime = generate_signals_regime_following(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    clustering_result=clustering_result,
    config=regime_config,
)

# Strategy 3: Mean Reversion
reversion_config = MeanReversionConfig(
    clustering_config=clustering_config,
    extreme_threshold=0.2,
    min_extreme_duration=3,
)
signals_reversion, strength_reversion, meta_reversion = generate_signals_mean_reversion(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    clustering_result=clustering_result,
    config=reversion_config,
)

# Kết hợp signals (ví dụ: consensus)
combined_signals = pd.Series(0, index=df.index)
combined_strength = pd.Series(0.0, index=df.index)

for i in range(len(df)):
    signals_list = [
        signals_transition.iloc[i],
        signals_regime.iloc[i],
        signals_reversion.iloc[i],
    ]
    strengths_list = [
        strength_transition.iloc[i],
        strength_regime.iloc[i],
        strength_reversion.iloc[i],
    ]
    
    # Consensus: majority vote với weighted strength
    long_votes = sum(1 for s in signals_list if s == 1)
    short_votes = sum(1 for s in signals_list if s == -1)
    
    if long_votes > short_votes:
        combined_signals.iloc[i] = 1
        combined_strength.iloc[i] = sum(
            s for s, st in zip(signals_list, strengths_list) if s == 1
        ) / max(long_votes, 1)
    elif short_votes > long_votes:
        combined_signals.iloc[i] = -1
        combined_strength.iloc[i] = sum(
            abs(s) * st for s, st in zip(signals_list, strengths_list) if s == -1
        ) / max(short_votes, 1)
```

### Kết quả trả về

Tất cả các strategy functions trả về tuple `(signals, signal_strength, metadata)`:

- **signals**: `pd.Series` với giá trị:
  - `1` = LONG signal
  - `-1` = SHORT signal
  - `0` = NEUTRAL (no signal)

- **signal_strength**: `pd.Series` với giá trị từ `0.0` đến `1.0`, biểu thị độ mạnh của signal

- **metadata**: `pd.DataFrame` chứa thông tin bổ sung:
  - Cluster values
  - Real_clust values
  - Relative positions
  - Price changes
  - Và các metrics khác tùy theo strategy

### Lưu ý về Strategies

1. **Tái sử dụng clustering_result**: Nếu bạn chạy nhiều strategies, nên tính `clustering_result` một lần và truyền vào các strategy functions để tránh tính toán lại.

2. **Kết hợp strategies**: Có thể kết hợp nhiều strategies bằng cách:
   - Consensus voting (majority vote)
   - Weighted voting (dựa trên signal strength)
   - Conditional logic (strategy A chỉ khi điều kiện X, strategy B khi điều kiện Y)

3. **Backtesting**: Luôn backtest strategies trước khi sử dụng live. Các parameters cần được tối ưu cho từng market và timeframe.

4. **Risk Management**: Các strategies này chỉ tạo signals, không bao gồm risk management (stop loss, take profit, position sizing). Cần implement riêng.

## Port từ Pine Script

Module này được port từ Pine Script indicator "Simplified Percentile Clustering" (version 6) của InvestorUnknown.
