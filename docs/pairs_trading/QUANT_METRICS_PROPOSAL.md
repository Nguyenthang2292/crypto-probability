# Quantitative Metrics Proposal for Pairs Trading Analyzer

## Tổng quan

Hiện tại `PairsTradingAnalyzer` chỉ có **correlation** để đánh giá pairs. Để có một pairs trading strategy robust, cần bổ sung các quantitative tests/metrics sau:

---

## 1. Cointegration Tests (QUAN TRỌNG NHẤT)

### 1.1 Augmented Dickey-Fuller (ADF) Test
**Mục đích**: Kiểm tra xem spread có stationary (mean-reverting) không.

**Cách hoạt động**:
- Test null hypothesis: Spread có unit root (non-stationary)
- Nếu reject null (p-value < 0.05): Spread là stationary → Good for pairs trading
- Nếu không reject: Spread có thể drift away → Bad for pairs trading

**Implementation**:
```python
from statsmodels.tsa.stattools import adfuller

def test_cointegration_adf(spread_series):
    """
    Test cointegration using ADF test on spread.
    
    Returns:
        dict with keys: 'adf_statistic', 'pvalue', 'is_cointegrated', 'critical_values'
    """
    result = adfuller(spread_series.dropna())
    return {
        'adf_statistic': result[0],
        'pvalue': result[1],
        'is_cointegrated': result[1] < 0.05,  # Reject null = cointegrated
        'critical_values': result[4]
    }
```

**Threshold**: p-value < 0.05 (hoặc 0.01 cho strict hơn)

---

### 1.2 Johansen Cointegration Test
**Mục đích**: Test cointegration cho multiple time series (advanced).

**Cách hoạt động**:
- Test xem có tồn tại cointegrating vector không
- Trả về trace statistic và max eigenvalue statistic
- Cho phép test nhiều pairs cùng lúc

**Implementation**:
```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def test_cointegration_johansen(price1, price2):
    """
    Test cointegration using Johansen test.
    
    Returns:
        dict with 'trace_stat', 'max_eigen_stat', 'is_cointegrated'
    """
    data = np.column_stack([price1, price2])
    result = coint_johansen(data, det_order=0, k_ar_diff=1)
    
    # Check trace statistic
    trace_critical = result.cvt[:, 1]  # 5% critical value
    trace_stat = result.lr1[0]
    
    return {
        'trace_stat': trace_stat,
        'trace_critical_5pct': trace_critical[0],
        'is_cointegrated': trace_stat > trace_critical[0],
        'max_eigen_stat': result.lr2[0],
        'max_eigen_critical_5pct': result.cvm[0, 1]
    }
```

**Threshold**: Trace statistic > Critical value (5%)

---

## 2. Mean Reversion Metrics

### 2.1 Half-Life of Mean Reversion
**Mục đích**: Đo thời gian spread cần để quay về mean (half-life).

**Công thức**:
```
Half-life = -log(2) / theta
trong đó theta là coefficient từ OLS regression:
    spread(t) - spread(t-1) = theta * spread(t-1) + error
```

**Implementation**:
```python
def calculate_half_life(spread_series):
    """
    Calculate half-life of mean reversion.
    
    Returns:
        Half-life in periods (candles), or None if not mean-reverting
    """
    spread_lag = spread_series.shift(1)
    spread_diff = spread_series - spread_lag
    
    # OLS regression
    spread_lag = spread_lag.dropna()
    spread_diff = spread_diff.dropna()
    
    if len(spread_lag) < 10:
        return None
    
    # Remove NaN alignment
    valid_idx = spread_lag.index.intersection(spread_diff.index)
    spread_lag = spread_lag.loc[valid_idx]
    spread_diff = spread_diff.loc[valid_idx]
    
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(spread_lag.values.reshape(-1, 1), spread_diff.values)
    theta = model.coef_[0]
    
    if theta >= 0:  # Not mean-reverting
        return None
    
    half_life = -np.log(2) / theta
    return max(0, half_life)  # Ensure non-negative
```

**Threshold**: 
- Good: Half-life < 20 periods (cho 1h timeframe = < 20 hours)
- Acceptable: Half-life < 50 periods
- Bad: Half-life > 50 periods hoặc không mean-reverting

---

### 2.2 Hurst Exponent
**Mục đích**: Đo tính mean-reverting (H < 0.5) hay trending (H > 0.5).

**Công thức**:
```
H = 0.5: Random walk
H < 0.5: Mean-reverting (good for pairs trading)
H > 0.5: Trending (bad for pairs trading)
```

**Implementation**:
```python
def calculate_hurst_exponent(series, max_lag=20):
    """
    Calculate Hurst exponent using R/S analysis.
    
    Returns:
        Hurst exponent (0 < H < 1)
    """
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    
    # Linear regression: log(tau) vs log(lag)
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0
```

**Threshold**:
- Excellent: H < 0.4 (strong mean reversion)
- Good: 0.4 <= H < 0.5 (mean-reverting)
- Bad: H >= 0.5 (trending, not suitable)

---

## 3. Spread Statistics

### 3.1 Z-Score Statistics
**Mục đích**: Normalize spread để dễ so sánh và set entry/exit thresholds.

**Công thức**:
```
z-score = (spread - mean(spread)) / std(spread)
```

**Metrics cần tính**:
- Mean z-score (should be ~0)
- Std of z-score (should be ~1)
- Skewness (asymmetry)
- Kurtosis (tail heaviness)
- Min/Max z-score (extreme values)

**Implementation**:
```python
def calculate_zscore_stats(spread_series, lookback=60):
    """
    Calculate z-score statistics for spread.
    
    Returns:
        dict with mean, std, skewness, kurtosis, min, max, current_zscore
    """
    rolling_mean = spread_series.rolling(lookback).mean()
    rolling_std = spread_series.rolling(lookback).std()
    zscore = (spread_series - rolling_mean) / rolling_std
    
    return {
        'mean_zscore': zscore.mean(),
        'std_zscore': zscore.std(),
        'skewness': zscore.skew(),
        'kurtosis': zscore.kurtosis(),
        'min_zscore': zscore.min(),
        'max_zscore': zscore.max(),
        'current_zscore': zscore.iloc[-1] if len(zscore) > 0 else None,
        'zscore_series': zscore
    }
```

**Trading Rules**:
- Entry long spread khi z-score < -2 (spread quá thấp, expect mean reversion up)
- Entry short spread khi z-score > +2 (spread quá cao, expect mean reversion down)
- Exit khi z-score về ~0

---

### 3.2 Spread Sharpe Ratio
**Mục đích**: Đo risk-adjusted return của spread.

**Công thức**:
```
Sharpe = mean(spread_returns) / std(spread_returns) * sqrt(periods_per_year)
```

**Implementation**:
```python
def calculate_spread_sharpe(spread_series, periods_per_year=365*24):  # 1h = 8760 periods/year
    """
    Calculate Sharpe ratio of spread returns.
    """
    returns = spread_series.pct_change().dropna()
    if len(returns) < 10 or returns.std() == 0:
        return None
    
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    return sharpe
```

**Threshold**:
- Excellent: Sharpe > 2.0
- Good: Sharpe > 1.0
- Acceptable: Sharpe > 0.5
- Bad: Sharpe < 0.5

---

### 3.3 Maximum Drawdown
**Mục đích**: Đo worst-case loss từ peak.

**Implementation**:
```python
def calculate_max_drawdown(spread_series):
    """
    Calculate maximum drawdown of spread.
    """
    cumulative = spread_series.cumsum() if spread_series.iloc[0] != 0 else spread_series
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()
```

**Threshold**: 
- Good: Max DD < 20%
- Acceptable: Max DD < 30%
- Bad: Max DD > 30%

---

### 3.4 Calmar Ratio
**Mục đích**: Return / Max Drawdown (risk-adjusted metric).

**Implementation**:
```python
def calculate_calmar_ratio(spread_series, periods_per_year=365*24):
    """
    Calculate Calmar ratio: Annual Return / Max Drawdown
    """
    total_return = (spread_series.iloc[-1] / spread_series.iloc[0] - 1) if len(spread_series) > 1 else 0
    annual_return = total_return * (periods_per_year / len(spread_series))
    max_dd = abs(calculate_max_drawdown(spread_series))
    
    if max_dd == 0:
        return None
    
    return annual_return / max_dd
```

**Threshold**:
- Excellent: Calmar > 3.0
- Good: Calmar > 1.5
- Acceptable: Calmar > 1.0

---

## 4. Classification Metrics (F1-Score, Precision, Recall)

### 4.1 Spread Direction Classification
**Mục đích**: Đánh giá accuracy của việc predict spread direction (up/down).

**Cách hoạt động**:
1. Label spread direction: UP nếu spread tăng, DOWN nếu spread giảm
2. Predict direction dựa trên z-score hoặc technical indicators
3. Tính F1-score, Precision, Recall

**Implementation**:
```python
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

def calculate_direction_metrics(spread_series, prediction_method='zscore', zscore_threshold=0.5):
    """
    Calculate classification metrics for spread direction prediction.
    
    Args:
        spread_series: Historical spread values
        prediction_method: 'zscore' or 'trend'
        zscore_threshold: Threshold for z-score based prediction
    
    Returns:
        dict with f1_score, precision, recall, accuracy
    """
    # Actual direction (1 = up, 0 = down)
    actual = (spread_series.diff() > 0).astype(int).iloc[1:]
    
    # Predicted direction
    if prediction_method == 'zscore':
        rolling_mean = spread_series.rolling(60).mean()
        rolling_std = spread_series.rolling(60).std()
        zscore = (spread_series - rolling_mean) / rolling_std
        predicted = (zscore < -zscore_threshold).astype(int).iloc[1:]  # Expect mean reversion
    else:
        # Simple trend following
        predicted = (spread_series.diff().shift(-1) > 0).astype(int).iloc[1:]
    
    # Align indices
    common_idx = actual.index.intersection(predicted.index)
    actual = actual.loc[common_idx]
    predicted = predicted.loc[common_idx]
    
    if len(actual) < 10:
        return None
    
    return {
        'f1_score': f1_score(actual, predicted, average='weighted'),
        'precision': precision_score(actual, predicted, average='weighted', zero_division=0),
        'recall': recall_score(actual, predicted, average='weighted', zero_division=0),
        'accuracy': (actual == predicted).mean(),
        'classification_report': classification_report(actual, predicted, output_dict=True)
    }
```

**Threshold**:
- Excellent: F1 > 0.7, Precision > 0.7, Recall > 0.7
- Good: F1 > 0.6
- Acceptable: F1 > 0.5
- Bad: F1 < 0.5

---

## 5. Hedge Ratio Calculation

### 5.1 OLS Hedge Ratio (Static)
**Mục đích**: Tính optimal hedge ratio để tạo spread.

**Công thức**:
```
hedge_ratio = cov(price1, price2) / var(price2)
spread = price1 - hedge_ratio * price2
```

**Implementation**:
```python
def calculate_ols_hedge_ratio(price1, price2):
    """
    Calculate OLS hedge ratio.
    
    Returns:
        hedge_ratio, spread_series
    """
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(price2.values.reshape(-1, 1), price1.values)
    hedge_ratio = model.coef_[0]
    
    spread = price1 - hedge_ratio * price2
    return hedge_ratio, spread
```

---

### 5.2 Kalman Filter Hedge Ratio (Dynamic)
**Mục đích**: Hedge ratio thay đổi theo thời gian (adaptive).

**Implementation**:
```python
from pykalman import KalmanFilter

def calculate_kalman_hedge_ratio(price1, price2):
    """
    Calculate dynamic hedge ratio using Kalman filter.
    
    Returns:
        hedge_ratio_series, spread_series
    """
    # State: hedge_ratio
    # Observation: price1 = hedge_ratio * price2 + noise
    
    kf = KalmanFilter(
        transition_matrices=np.array([[1]]),
        observation_matrices=price2.values.reshape(-1, 1),
        initial_state_mean=0,
        n_dim_obs=1
    )
    
    state_means, _ = kf.em(price1.values).smooth()
    hedge_ratio_series = pd.Series(state_means.flatten(), index=price1.index)
    
    spread = price1 - hedge_ratio_series * price2
    return hedge_ratio_series, spread
```

---

## 6. Implementation Priority

### Phase 1 (Essential - Implement First):
1. ✅ **ADF Cointegration Test** - Critical for pairs trading
2. ✅ **Half-Life of Mean Reversion** - Essential metric
3. ✅ **Z-Score Statistics** - Needed for entry/exit signals
4. ✅ **OLS Hedge Ratio** - Required to create proper spread

### Phase 2 (Important):
5. ✅ **Hurst Exponent** - Good indicator of mean reversion
6. ✅ **Spread Sharpe Ratio** - Risk-adjusted return
7. ✅ **Maximum Drawdown** - Risk metric

### Phase 3 (Advanced):
8. ✅ **Johansen Cointegration Test** - For multiple pairs
9. ✅ **Kalman Filter Hedge Ratio** - Dynamic hedging
10. ✅ **F1-Score/Classification Metrics** - Direction prediction accuracy
11. ✅ **Calmar Ratio** - Advanced risk metric

### Phase 4 (Composite Scoring):
12. ✅ **Combined Quantitative Score** - Composite score (0-100) combining all metrics

---

## 7. Suggested DataFrame Columns ✅ IMPLEMENTED

Sau khi implement, DataFrame từ `analyze_pairs_opportunity()` đã có các columns sau:

```python
columns = [
    # Existing
    'long_symbol', 'short_symbol', 'long_score', 'short_score', 
    'spread', 'correlation', 'opportunity_score',
    
    # Quantitative metrics
    'quantitative_score',           # ✅ Combined quantitative score (0-100)
    'adf_pvalue',                   # ✅ ADF test p-value
    'is_cointegrated',              # ✅ Boolean: p-value < 0.05
    'half_life',                    # ✅ Half-life in periods
    'hurst_exponent',               # ✅ Hurst exponent
    'mean_zscore',                  # ✅ Mean z-score
    'std_zscore',                   # ✅ Std of z-score
    'skewness',                     # ✅ Skewness
    'kurtosis',                     # ✅ Kurtosis
    'current_zscore',               # ✅ Current z-score
    'spread_sharpe',                # ✅ Sharpe ratio
    'max_drawdown',                 # ✅ Maximum drawdown
    'calmar_ratio',                 # ✅ Calmar ratio
    'hedge_ratio',                  # ✅ OLS hedge ratio
    'johansen_trace_stat',          # ✅ Johansen trace statistic
    'johansen_critical_value',      # ✅ Johansen critical value
    'is_johansen_cointegrated',     # ✅ Johansen cointegration result
    'kalman_hedge_ratio',           # ✅ Kalman filter hedge ratio
    'classification_f1',            # ✅ F1-score for direction
    'classification_precision',     # ✅ Precision
    'classification_recall',        # ✅ Recall
    'classification_accuracy',      # ✅ Accuracy
]
```

**Note**: Tất cả các columns đã được implement và tự động tính trong `analyze_pairs_opportunity()`.

---

## 8. Combined Quantitative Score ✅ IMPLEMENTED

Tạo một **composite score** kết hợp tất cả metrics:

**Status**: ✅ Đã được triển khai trong `modules/pairs_trading/opportunity_scorer.py`

**Implementation**:
- Method: `OpportunityScorer.calculate_quantitative_score()`
- Returns: Score từ 0-100 (higher is better)
- Location: `modules/pairs_trading/opportunity_scorer.py`
- Usage: Tự động được tính trong `analyze_pairs_opportunity()` và thêm vào DataFrame column `quantitative_score`

**Weights**:
- Cointegration: 30%
- Half-life: 20%
- Hurst: 15%
- Sharpe: 15%
- F1-score: 10%
- Max DD: 10%

**Example Usage**:
```python
from modules.pairs_trading import PairsTradingAnalyzer

analyzer = PairsTradingAnalyzer()
pairs_df = analyzer.analyze_pairs_opportunity(...)

# quantitative_score is automatically included in the DataFrame
print(pairs_df[['long_symbol', 'short_symbol', 'quantitative_score']])
```

**Original Implementation** (để reference):
```python
def calculate_quantitative_score(metrics_dict):
    """
    Calculate combined quantitative score (0-100).
    
    Weights:
    - Cointegration: 30%
    - Half-life: 20%
    - Hurst: 15%
    - Sharpe: 15%
    - F1-score: 10%
    - Max DD: 10%
    """
    score = 0
    
    # Cointegration (30%)
    if metrics_dict.get('is_cointegrated'):
        score += 30
    elif metrics_dict.get('adf_pvalue', 1) < 0.1:
        score += 15
    
    # Half-life (20%)
    half_life = metrics_dict.get('half_life')
    if half_life and half_life < 20:
        score += 20
    elif half_life and half_life < 50:
        score += 10
    
    # Hurst (15%)
    hurst = metrics_dict.get('hurst_exponent')
    if hurst and hurst < 0.4:
        score += 15
    elif hurst and hurst < 0.5:
        score += 8
    
    # Sharpe (15%)
    sharpe = metrics_dict.get('spread_sharpe')
    if sharpe and sharpe > 2.0:
        score += 15
    elif sharpe and sharpe > 1.0:
        score += 8
    
    # F1-score (10%)
    f1 = metrics_dict.get('f1_score')
    if f1 and f1 > 0.7:
        score += 10
    elif f1 and f1 > 0.6:
        score += 5
    
    # Max DD (10%)
    max_dd = metrics_dict.get('max_drawdown')
    if max_dd and abs(max_dd) < 0.2:
        score += 10
    elif max_dd and abs(max_dd) < 0.3:
        score += 5
    
    return min(100, score)
```

---

## 9. Dependencies Required

```python
# Add to requirements.txt
statsmodels>=0.14.0      # For ADF, Johansen tests
scikit-learn>=1.0.0      # For LinearRegression, metrics
pykalman>=0.9.5          # For Kalman filter (optional, Phase 3)
```

---

## 10. Next Steps

1. **Implement Phase 1 metrics** (ADF, Half-life, Z-score, OLS hedge ratio)
2. **Update `analyze_pairs_opportunity()`** để tính các metrics mới
3. **Update `validate_pairs()`** để filter dựa trên quantitative metrics
4. **Add configuration** trong `modules/config.py` cho thresholds
5. **Test với real data** và fine-tune thresholds
6. **Implement Phase 2 & 3** khi Phase 1 stable

---

## References

- **Pairs Trading**: "Pairs Trading: Quantitative Methods and Analysis" by Ganapathy Vidyamurthy
- **Cointegration**: Engle-Granger two-step method, Johansen test
- **Mean Reversion**: "Mean Reversion Strategies" by Ernie Chan
- **Hurst Exponent**: R/S Analysis (Rescaled Range Analysis)

