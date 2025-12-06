"""
Adaptive Trend Classification (ATC) - Main computation module.

Module này cung cấp hàm chính `compute_atc_signals` để tính toán ATC signals
từ price data. ATC sử dụng nhiều loại Moving Averages (EMA, HMA, WMA, DEMA, LSMA, KAMA)
với adaptive weighting dựa trên simulated equity curves.

Cấu trúc tính toán:
1. Layer 1: Tính signals cho từng loại MA dựa trên equity curves
2. Layer 2: Tính trọng số từ Layer 1 signals
3. Final: Kết hợp tất cả để tạo Average_Signal

Các module hỗ trợ:
- utils.py: Core utilities (rate_of_change, diflen, exp_growth)
- moving_averages.py: MA calculations
- signals.py: Signal generation
- equity.py: Equity curve calculations
- layer1.py: Layer 1 processing
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .equity import equity_series
from .layer1 import cut_signal, _layer1_signal_for_ma
from .moving_averages import set_of_moving_averages
from .utils import rate_of_change


def compute_atc_signals(
    prices: pd.Series,
    src: Optional[pd.Series] = None,
    *,
    ema_len: int = 28,
    hull_len: int = 28,
    wma_len: int = 28,
    dema_len: int = 28,
    lsma_len: int = 28,
    kama_len: int = 28,
    ema_w: float = 1.0,
    hma_w: float = 1.0,
    wma_w: float = 1.0,
    dema_w: float = 1.0,
    lsma_w: float = 1.0,
    kama_w: float = 1.0,
    robustness: str = "Medium",
    La: float = 0.02,
    De: float = 0.03,
    cutout: int = 0,
) -> dict[str, pd.Series]:
    """
    Tính toán Adaptive Trend Classification (ATC) signals.

    Hàm này orchestrates toàn bộ quá trình tính toán ATC:
    1. Tính các Moving Averages với nhiều lengths
    2. Tính Layer 1 signals cho từng loại MA
    3. Tính Layer 2 weights từ Layer 1 signals
    4. Kết hợp tất cả để tạo Average_Signal

    Args:
        prices: Series giá (thường là close) để tính rate of change và signals.
        src: Series nguồn để tính MA. Nếu None, dùng prices.
        ema_len, hull_len, wma_len, dema_len, lsma_len, kama_len:
            Độ dài window cho từng loại MA.
        ema_w, hma_w, wma_w, dema_w, lsma_w, kama_w:
            Trọng số khởi tạo cho từng họ MA ở Layer 2.
        robustness: "Narrow" / "Medium" / "Wide" - độ rộng của length offsets.
        La: Lambda (growth rate) cho exponential growth factor.
        De: Decay rate cho equity calculations.
        cutout: Số bar bỏ qua đầu chuỗi.

    Returns:
        Dictionary chứa:
            - EMA_Signal, HMA_Signal, WMA_Signal, DEMA_Signal, LSMA_Signal, KAMA_Signal:
              Layer 1 signals cho từng loại MA
            - EMA_S, HMA_S, WMA_S, DEMA_S, LSMA_S, KAMA_S:
              Layer 2 equity weights
            - Average_Signal: Final combined signal

    Raises:
        ValueError: Nếu prices rỗng hoặc không hợp lệ.
        ZeroDivisionError: Nếu tổng equity weights bằng 0 (rất hiếm).
    """
    # Input validation
    if prices is None or len(prices) == 0:
        raise ValueError("prices không được rỗng hoặc None")
    
    if src is None:
        src = prices
    
    if len(src) == 0:
        raise ValueError("src không được rỗng")
    
    # Validate robustness
    if robustness not in ("Narrow", "Medium", "Wide"):
        robustness = "Medium"  # Default fallback
    
    # Validate cutout
    if cutout < 0:
        cutout = 0
    if cutout >= len(prices):
        raise ValueError(f"cutout ({cutout}) phải nhỏ hơn độ dài prices ({len(prices)})")

    # Định nghĩa cấu hình cho các loại MA
    ma_configs = [
        ("EMA", ema_len, ema_w),
        ("HMA", hull_len, hma_w),
        ("WMA", wma_len, wma_w),
        ("DEMA", dema_len, dema_w),
        ("LSMA", lsma_len, lsma_w),
        ("KAMA", kama_len, kama_w),
    ]

    # DECLARE MOVING AVERAGES (SetOfMovingAverages)
    ma_tuples = {}
    for ma_type, length, _ in ma_configs:
        ma_tuple = set_of_moving_averages(length, src, ma_type, robustness=robustness)
        if ma_tuple is None:
            raise ValueError(f"Không thể tính toán {ma_type} với length={length}")
        ma_tuples[ma_type] = ma_tuple

    # MAIN CALCULATIONS - Adaptability Layer 1
    layer1_signals = {}
    for ma_type, _, _ in ma_configs:
        signal, _, _ = _layer1_signal_for_ma(
            prices, ma_tuples[ma_type], L=La, De=De, cutout=cutout
        )
        layer1_signals[ma_type] = signal

    # Adaptability Layer 2
    R = rate_of_change(prices)
    layer2_equities = {}
    for ma_type, _, weight in ma_configs:
        equity = equity_series(
            weight, layer1_signals[ma_type], R, L=La, De=De, cutout=cutout
        )
        layer2_equities[ma_type] = equity

    # FINAL CALCULATIONS
    nom = pd.Series(0.0, index=prices.index, dtype="float64")
    den = pd.Series(0.0, index=prices.index, dtype="float64")
    
    for ma_type, _, _ in ma_configs:
        signal = layer1_signals[ma_type]
        equity = layer2_equities[ma_type]
        nom += cut_signal(signal) * equity
        den += equity

    # Xử lý division by zero: nếu den = 0 thì Average_Signal = 0
    # (hoặc có thể dùng np.nan tùy vào logic nghiệp vụ)
    Average_Signal = nom / den
    # Thay thế inf và nan bằng 0 khi den = 0
    Average_Signal = Average_Signal.fillna(0.0).replace([float("inf"), float("-inf")], 0.0)

    # Build result dictionary
    result = {}
    for ma_type, _, _ in ma_configs:
        result[f"{ma_type}_Signal"] = layer1_signals[ma_type]
        result[f"{ma_type}_S"] = layer2_equities[ma_type]
    
    result["Average_Signal"] = Average_Signal

    return result


__all__ = [
    "compute_atc_signals",
]
