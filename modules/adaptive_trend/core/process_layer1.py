"""Layer 1 Processing functions for Adaptive Trend Classification (ATC).

This module provides functions for processing signals from multiple Moving
Averages in Layer 1 of the ATC system:
- weighted_signal: Calculate weighted average signal from multiple signals and weights
- cut_signal: Discretize continuous signal into {-1, 0, 1}
- trend_sign: Determine trend direction (+1 for bullish, -1 for bearish, 0 for neutral)
- _layer1_signal_for_ma: Calculate Layer 1 signal for a specific MA type
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from modules.common.utils import log_warn, log_error

from .compute_equity import equity_series
from .signal_detection import generate_signal_from_ma
from modules.adaptive_trend.utils import rate_of_change


def weighted_signal(
    signals: Iterable[pd.Series],
    weights: Iterable[pd.Series],
) -> pd.Series:
    """Calculate weighted average signal from multiple signals and weights.

    Port of Pine Script function:
        Signal(m1, w1, ..., m9, w9) =>
            n = Σ (mi * wi)      # Weighted sum
            d = Σ wi             # Sum of weights
            sig = math.round(n/d, 2)
            sig

    Typically receives 9 signal series and 9 weight series (equity curves).

    Args:
        signals: Iterable of signal series (typically 9 series).
        weights: Iterable of weight series (typically equity curves).

    Returns:
        Weighted signal series rounded to 2 decimal places.

    Raises:
        ValueError: If signals and weights have different lengths or are empty.
        TypeError: If inputs are not pandas Series.
    """
    signals = list(signals)
    weights = list(weights)
    
    if len(signals) != len(weights):
        raise ValueError(
            f"signals and weights must have the same length, "
            f"got {len(signals)} signals and {len(weights)} weights"
        )

    if not signals:
        log_warn("Empty signals/weights provided, returning empty series")
        return pd.Series(dtype="float64")

    # Validate all inputs are Series
    for i, (sig, wgt) in enumerate(zip(signals, weights)):
        if not isinstance(sig, pd.Series):
            raise TypeError(f"signals[{i}] must be a pandas Series, got {type(sig)}")
        if not isinstance(wgt, pd.Series):
            raise TypeError(f"weights[{i}] must be a pandas Series, got {type(wgt)}")

    try:
        # Check index compatibility
        first_sig_index = signals[0].index
        first_wgt_index = weights[0].index
        
        for i, (sig, wgt) in enumerate(zip(signals[1:], weights[1:]), start=1):
            if not sig.index.equals(first_sig_index):
                log_warn(
                    f"signals[{i}] has different index than signals[0]. "
                    f"Attempting to align indices."
                )
            if not wgt.index.equals(first_wgt_index):
                log_warn(
                    f"weights[{i}] has different index than weights[0]. "
                    f"Attempting to align indices."
                )

        # Calculate weighted average
        # Collect all valid indices from all pairs (union, not intersection)
        # This preserves all indices from all pairs instead of shrinking to intersection
        all_indices = pd.Index([])
        valid_pairs = []
        
        for m, w in zip(signals, weights):
            # Align indices if needed
            common_index = m.index.intersection(w.index)
            if len(common_index) == 0:
                log_warn(
                    f"Signal and weight have no common indices. "
                    f"Skipping this pair."
                )
                continue
            
            # Store aligned series and their common indices
            m_aligned = m.loc[common_index]
            w_aligned = w.loc[common_index]
            valid_pairs.append((m_aligned, w_aligned))
            
            # Union all indices to preserve all valid indices from all pairs
            all_indices = all_indices.union(common_index)
        
        if not valid_pairs:
            log_warn("No valid signal/weight pairs found, returning empty series")
            return pd.Series(dtype="float64")
        
        # Initialize numerator and denominator with zeros for all indices
        num = pd.Series(0.0, index=all_indices, dtype="float64")
        den = pd.Series(0.0, index=all_indices, dtype="float64")
        
        # Accumulate weighted signals and weights for each pair independently
        for m_aligned, w_aligned in valid_pairs:
            # Calculate term for this pair
            term = m_aligned * w_aligned
            
            # Add to numerator and denominator only where data exists
            # This preserves all indices from all pairs
            num.loc[m_aligned.index] = num.loc[m_aligned.index] + term
            den.loc[w_aligned.index] = den.loc[w_aligned.index] + w_aligned

        # Avoid division by zero
        sig = num / den.replace(0, np.nan)
        result = sig.round(2)
        
        # Check for excessive NaN values
        nan_count = result.isna().sum()
        if nan_count > 0:
            nan_pct = (nan_count / len(result)) * 100
            if nan_pct > 10:  # Warn if more than 10% NaN
                log_warn(
                    f"Weighted signal contains {nan_count} NaN values ({nan_pct:.1f}%). "
                    f"This may indicate missing data or zero weights."
                )
        
        return result
    
    except Exception as e:
        log_error(f"Error calculating weighted signal: {e}")
        raise


def cut_signal(x: pd.Series, threshold: float = 0.49, cutout: int = 0) -> pd.Series:
    """Discretize continuous signal into {-1, 0, 1} based on threshold.

    Port of Pine Script function:
        Cut(x) =>
            c = x > 0.49 ? 1 : x < -0.49 ? -1 : 0
            c

    Args:
        x: Continuous signal series.
        threshold: Threshold for discretization (default: 0.49).
        cutout: Number of bars to skip at beginning (force to 0).
            Values > threshold → 1, values < -threshold → -1, else → 0.

    Returns:
        Series with discrete values {-1, 0, 1}.

    Raises:
        ValueError: If threshold is invalid.
        TypeError: If x is not a pandas Series.
    """
    if not isinstance(x, pd.Series):
        raise TypeError(f"x must be a pandas Series, got {type(x)}")
    
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    
    if cutout < 0:
        raise ValueError(f"cutout must be >= 0, got {cutout}")
    
    if len(x) == 0:
        log_warn("Empty signal series provided, returning empty series")
        return pd.Series(dtype="int8", index=x.index)
    
    try:
        c = pd.Series(0, index=x.index, dtype="int8")
        
        # Handle NaN values: treat as 0 (no signal)
        valid_mask = ~x.isna()
        
        if valid_mask.any():
            c.loc[valid_mask & (x > threshold)] = 1
            c.loc[valid_mask & (x < -threshold)] = -1
        
        # Enforce cutout: set first 'cutout' bars to 0
        if cutout > 0 and cutout < len(c):
            c.iloc[:cutout] = 0
            # Also ensure NaN handling aligns if needed, though int8 has no NaN
        
        # Check for excessive NaN values
        nan_count = (~valid_mask).sum()
        if nan_count > 0:
            nan_pct = (nan_count / len(x)) * 100
            if nan_pct > 10:  # Warn if more than 10% NaN
                log_warn(
                    f"Input signal contains {nan_count} NaN values ({nan_pct:.1f}%). "
                    f"These will be treated as 0 (no signal)."
                )
        
        return c
    
    except Exception as e:
        log_error(f"Error discretizing signal: {e}")
        raise


def trend_sign(signal: pd.Series, *, strategy: bool = False) -> pd.Series:
    """Determine trend direction from signal series.

    Numeric version (without colors) of Pine Script function:
        trendcol(signal) =>
            c = strategy ? (signal[1] > 0 ? colup : coldw)
                         : (signal > 0) ? colup : coldw

    Args:
        signal: Signal series.
        strategy: If True, uses signal[1] (previous bar) instead of current signal,
            matching Pine Script behavior.

    Returns:
        Series with trend direction values:
        - +1: Bullish trend (signal > 0)
        - -1: Bearish trend (signal < 0)
        - 0: Neutral (signal == 0)

    Raises:
        TypeError: If signal is not a pandas Series.
    """
    if not isinstance(signal, pd.Series):
        raise TypeError(f"signal must be a pandas Series, got {type(signal)}")
    
    if len(signal) == 0:
        log_warn("Empty signal series provided, returning empty series")
        return pd.Series(dtype="int8", index=signal.index)
    
    try:
        base = signal.shift(1) if strategy else signal
        result = pd.Series(0, index=signal.index, dtype="int8")
        
        # Handle NaN values: treat as 0 (neutral)
        valid_mask = ~base.isna()
        
        if valid_mask.any():
            result.loc[valid_mask & (base > 0)] = 1
            result.loc[valid_mask & (base < 0)] = -1
        
        return result
    
    except Exception as e:
        log_error(f"Error determining trend sign: {e}")
        raise


def _layer1_signal_for_ma(
    prices: pd.Series,
    ma_tuple: Tuple[pd.Series, ...],
    *,
    L: float,
    De: float,
    cutout: int = 0,
) -> Tuple[pd.Series, Tuple[pd.Series, ...], Tuple[pd.Series, ...]]:
    """Calculate Layer 1 signal for a specific Moving Average type.

    Port of Pine Script logic block:
        E   = eq(1, signal(MA),   R), sE   = signal(MA)
        E1  = eq(1, signal(MA1),  R), sE1  = signal(MA1)
        ...
        EMA_Signal = Signal(sE, E, sE1, E1, ..., sE_4, E_4)

    For each of the 9 MAs:
    1. Generate signal from price/MA crossover
    2. Calculate equity curve from signal
    3. Weight signals by their equity curves to get final Layer 1 signal

    Args:
        prices: Price series (typically close prices).
        ma_tuple: Tuple of 9 MA Series: (MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4).
        L: Lambda (growth rate) for equity calculations.
        De: Decay factor for equity calculations.
        cutout: Number of bars to skip at beginning.

    Returns:
        Tuple containing:
        - signal_series: Weighted Layer 1 signal for this MA type
        - signals_tuple: Tuple of 9 individual signals (s, s1, s2, s3, s4, s_1, s_2, s_3, s_4)
        - equity_tuple: Tuple of 9 equity curves (E, E1, E2, E3, E4, E_1, E_2, E_3, E_4)

    Raises:
        ValueError: If ma_tuple doesn't have exactly 9 elements or inputs are invalid.
        TypeError: If inputs are not pandas Series.
    """
    # Input validation
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pandas Series, got {type(prices)}")
    
    if len(prices) == 0:
        raise ValueError("prices cannot be empty")
    
    if not isinstance(ma_tuple, tuple):
        raise TypeError(f"ma_tuple must be a tuple, got {type(ma_tuple)}")
    
    EXPECTED_MA_COUNT = 9
    if len(ma_tuple) != EXPECTED_MA_COUNT:
        raise ValueError(
            f"ma_tuple must contain exactly {EXPECTED_MA_COUNT} MA series, "
            f"got {len(ma_tuple)}"
        )
    
    # Validate all MAs are Series
    for i, ma in enumerate(ma_tuple):
        if not isinstance(ma, pd.Series):
            raise TypeError(
                f"ma_tuple[{i}] must be a pandas Series, got {type(ma)}"
            )
        if len(ma) == 0:
            raise ValueError(f"ma_tuple[{i}] cannot be empty")
    
    # Validate parameters
    if not isinstance(L, (int, float)) or np.isnan(L) or np.isinf(L):
        raise ValueError(f"L must be a finite number, got {L}")
    
    if not (0 <= De <= 1):
        raise ValueError(f"De must be between 0 and 1, got {De}")
    
    if cutout < 0:
        raise ValueError(f"cutout must be >= 0, got {cutout}")

    try:
        # Unpack MA tuple
        (
            MA,
            MA1,
            MA2,
            MA3,
            MA4,
            MA_1,
            MA_2,
            MA_3,
            MA_4,
        ) = ma_tuple

        R = rate_of_change(prices)

        # Generate signals for all MAs (optimized with list comprehension)
        ma_list = [MA, MA1, MA2, MA3, MA4, MA_1, MA_2, MA_3, MA_4]
        signals = [generate_signal_from_ma(prices, ma) for ma in ma_list]
        
        # Unpack signals for return tuple (maintaining original variable names)
        s, s1, s2, s3, s4, s_1, s_2, s_3, s_4 = signals

        # Calculate equity curves for all signals (optimized with list comprehension)
        equities = [
            equity_series(1.0, sig, R, L=L, De=De, cutout=cutout)
            for sig in signals
        ]
        
        # Unpack equities for return tuple (maintaining original variable names)
        E, E1, E2, E3, E4, E_1, E_2, E_3, E_4 = equities

        # Calculate weighted signal
        signal_series = weighted_signal(
            signals=signals,
            weights=equities,
        )

        return (
            signal_series,
            (s, s1, s2, s3, s4, s_1, s_2, s_3, s_4),
            (E, E1, E2, E3, E4, E_1, E_2, E_3, E_4),
        )
    
    except Exception as e:
        log_error(f"Error calculating Layer 1 signal for MA: {e}")
        raise


__all__ = [
    "weighted_signal",
    "cut_signal",
    "trend_sign",
    "_layer1_signal_for_ma",
]

