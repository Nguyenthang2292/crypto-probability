"""
Range Oscillator Strategy 7: Divergence Detection.

This module provides the divergence detection signal generation strategy.
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np
import os

from modules.range_oscillator.utils.oscillator_data import get_oscillator_data
from modules.common.utils import log_debug, log_analysis


def generate_signals_divergence_strategy(
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    *,
    oscillator: Optional[pd.Series] = None,
    ma: Optional[pd.Series] = None,
    range_atr: Optional[pd.Series] = None,
    length: int = 50,
    mult: float = 2.0,
    lookback_period: int = 30,
    min_swing_bars: int = 5,
    min_divergence_strength: float = 10.0,
    enable_debug: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate trading signals based on Range Oscillator Strategy 7: Divergence Detection.
    
    Strategy Logic:
    ---------------
    This strategy detects divergences between price and oscillator:
    - Bearish Divergence: Price makes higher high, oscillator makes lower high → SHORT signal
    - Bullish Divergence: Price makes lower low, oscillator makes higher low → LONG signal
    
    Divergences often signal potential trend reversals before price action confirms them.
    
    IMPORTANT - Look-ahead Bias Prevention:
    ----------------------------------------
    To avoid look-ahead bias, peak/trough detection uses only PAST data:
    - Peaks/troughs are detected as candidates at time t
    - Values at time t are stored
    - Peaks/troughs are confirmed at time t + min_swing_bars
    - Signals are emitted at confirmation time (t + min_swing_bars) using values from candidate time (t)
    - This ensures signals are only generated when peaks/troughs are actually confirmed, not predicted
    
    Key Concept:
    "Use oscillator peaks and troughs relative to price action to spot hidden strength 
    or weakness before the next market move."
    
    Args:
        high: High price series (required for divergence detection, also needed if oscillator not provided)
        low: Low price series (required for divergence detection, also needed if oscillator not provided)
        close: Close price series (required for divergence detection, also needed if oscillator not provided)
        oscillator: Pre-calculated oscillator series (optional, if provided with ma and range_atr)
        ma: Pre-calculated moving average series (optional)
        range_atr: Pre-calculated range ATR series (optional)
        length: Minimum range length for oscillator calculation (default: 50, ignored if oscillator provided)
        mult: Range width multiplier (default: 2.0, ignored if oscillator provided)
        lookback_period: Period to look back for finding peaks/troughs (default: 30)
        min_swing_bars: Minimum bars between peaks/troughs (default: 5)
        min_divergence_strength: Minimum difference in oscillator values for valid divergence (default: 10.0)
        enable_debug: If True, enable debug logging (default: False)
    
    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
    """
    # Validate parameters
    if min_swing_bars <= 0:
        raise ValueError(f"min_swing_bars must be > 0, got {min_swing_bars}")
    if lookback_period <= 0:
        raise ValueError(f"lookback_period must be > 0, got {lookback_period}")
    if min_divergence_strength < 0:
        raise ValueError(f"min_divergence_strength must be >= 0, got {min_divergence_strength}")
    
    # Need high, low, close for divergence detection (price peaks/troughs)
    if high is None or low is None or close is None:
        raise ValueError("high, low, close are required for divergence detection")
    
    # Enable debug logging if requested
    debug_enabled = enable_debug or os.environ.get("RANGE_OSCILLATOR_DEBUG", "false").lower() == "true"
    
    if debug_enabled:
        log_analysis(f"[Strategy7] Starting divergence detection signal generation")
        log_debug(f"[Strategy7] Parameters: lookback_period={lookback_period}, "
                 f"min_swing_bars={min_swing_bars}, min_divergence_strength={min_divergence_strength}")
    
    # Get oscillator data (either pre-calculated or calculate now)
    oscillator, ma, range_atr = get_oscillator_data(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )
    
    if debug_enabled:
        log_debug(f"[Strategy7] Data shape: oscillator={len(oscillator)}, price={len(close)}")
    
    # Initialize signal series
    signals = pd.Series(0, index=close.index, dtype="int8")
    signal_strength = pd.Series(0.0, index=close.index, dtype="float64")
    
    # Vectorized peak/trough detection using rolling windows
    # IMPORTANT: To avoid look-ahead bias, we only use PAST data to detect peaks/troughs
    # A peak is confirmed when: value >= max(left_window) AND we've waited min_swing_bars to confirm
    # A trough is confirmed when: value <= min(left_window) AND we've waited min_swing_bars to confirm
    
    # Detect price peaks (local maxima in high prices)
    # Only use left window (past data) - no future data
    left_max_high = high.rolling(window=min_swing_bars, min_periods=1).max().shift(1)
    # A peak candidate is detected when current high >= max of past min_swing_bars
    is_price_peak_candidate = (high >= left_max_high) & ~high.isna()
    
    # Detect price troughs (local minima in low prices)
    left_min_low = low.rolling(window=min_swing_bars, min_periods=1).min().shift(1)
    is_price_trough_candidate = (low <= left_min_low) & ~low.isna()
    
    # Detect oscillator peaks (local maxima)
    left_max_osc = oscillator.rolling(window=min_swing_bars, min_periods=1).max().shift(1)
    is_osc_peak_candidate = (oscillator >= left_max_osc) & ~oscillator.isna()
    
    # Detect oscillator troughs (local minima)
    left_min_osc = oscillator.rolling(window=min_swing_bars, min_periods=1).min().shift(1)
    is_osc_trough_candidate = (oscillator <= left_min_osc) & ~oscillator.isna()
    
    # IMPORTANT: To avoid look-ahead bias:
    # 1. Store peak/trough VALUES at candidate time t
    # 2. Shift the Series forward by min_swing_bars to get confirmation time t+min_swing_bars
    # 3. The values remain from time t, but index is at confirmation time
    # 4. This way, signal is emitted at t+min_swing_bars using values from t
    
    # Create Series with values at candidate times
    price_peak_values_series = pd.Series(high.values, index=close.index).where(is_price_peak_candidate, np.nan)
    price_trough_values_series = pd.Series(low.values, index=close.index).where(is_price_trough_candidate, np.nan)
    osc_peak_values_series = pd.Series(oscillator.values, index=oscillator.index).where(is_osc_peak_candidate, np.nan)
    osc_trough_values_series = pd.Series(oscillator.values, index=oscillator.index).where(is_osc_trough_candidate, np.nan)
    
    # Shift forward by min_swing_bars to get confirmation times
    # Values remain from candidate time, but index shifts to confirmation time
    price_peak_values_confirmed = price_peak_values_series.shift(min_swing_bars)
    price_trough_values_confirmed = price_trough_values_series.shift(min_swing_bars)
    osc_peak_values_confirmed = osc_peak_values_series.shift(min_swing_bars)
    osc_trough_values_confirmed = osc_trough_values_series.shift(min_swing_bars)
    
    # Get confirmed peak/trough indices (where values are not NaN after shift)
    price_peak_indices = price_peak_values_confirmed.dropna().index
    price_trough_indices = price_trough_values_confirmed.dropna().index
    osc_peak_indices = osc_peak_values_confirmed.dropna().index
    osc_trough_indices = osc_trough_values_confirmed.dropna().index
    
    # Create DataFrames with confirmed peak/trough data
    # Index is confirmation time, values are from candidate time (stored before shift)
    price_peaks_df = pd.DataFrame({
        'index': price_peak_indices,
        'value': price_peak_values_confirmed.loc[price_peak_indices].values,
        'position': range(len(price_peak_indices))
    }).set_index('index') if len(price_peak_indices) > 0 else pd.DataFrame(columns=['value', 'position'])
    
    price_troughs_df = pd.DataFrame({
        'index': price_trough_indices,
        'value': price_trough_values_confirmed.loc[price_trough_indices].values,
        'position': range(len(price_trough_indices))
    }).set_index('index') if len(price_trough_indices) > 0 else pd.DataFrame(columns=['value', 'position'])
    
    osc_peaks_df = pd.DataFrame({
        'index': osc_peak_indices,
        'value': osc_peak_values_confirmed.loc[osc_peak_indices].values,
        'position': range(len(osc_peak_indices))
    }).set_index('index') if len(osc_peak_indices) > 0 else pd.DataFrame(columns=['value', 'position'])
    
    osc_troughs_df = pd.DataFrame({
        'index': osc_trough_indices,
        'value': osc_trough_values_confirmed.loc[osc_trough_indices].values,
        'position': range(len(osc_trough_indices))
    }).set_index('index') if len(osc_trough_indices) > 0 else pd.DataFrame(columns=['value', 'position'])
    
    # FULLY VECTORIZED divergence detection
    # Strategy: For each timestamp, find if there are recent peaks/troughs that form divergences
    # Use shift operations on the peak/trough series to compare consecutive extrema
    
    # Convert peak/trough DataFrames to Series for easier manipulation
    price_peaks_series = pd.Series(price_peaks_df['value'].values, index=price_peaks_df.index)
    price_troughs_series = pd.Series(price_troughs_df['value'].values, index=price_troughs_df.index)
    osc_peaks_series = pd.Series(osc_peaks_df['value'].values, index=osc_peaks_df.index)
    osc_troughs_series = pd.Series(osc_troughs_df['value'].values, index=osc_troughs_df.index)
    
    # For peaks: compare each peak with the previous peak
    # Shift by 1 to get previous peak values
    prev_price_peaks = price_peaks_series.shift(1)
    prev_osc_peaks = osc_peaks_series.shift(1)
    
    # Bearish divergence at peak locations: price higher high AND oscillator lower high
    price_higher_at_peaks = price_peaks_series > prev_price_peaks
    osc_lower_at_peaks = osc_peaks_series < prev_osc_peaks
    osc_diff_at_peaks = (osc_peaks_series - prev_osc_peaks).abs()
    strong_osc_diff_peaks = osc_diff_at_peaks >= min_divergence_strength
    
    bearish_div_at_peaks = price_higher_at_peaks & osc_lower_at_peaks & strong_osc_diff_peaks
    
    # Calculate strength at peak locations
    # Avoid division by zero: replace 0 with NaN before division
    prev_price_peaks_safe = prev_price_peaks.copy()
    prev_price_peaks_safe = prev_price_peaks_safe.replace(0.0, np.nan)
    price_pct_change_peaks = ((price_peaks_series - prev_price_peaks) / prev_price_peaks_safe) * 100
    price_pct_change_peaks = price_pct_change_peaks.fillna(0.0)
    osc_pct_change_peaks = osc_diff_at_peaks / 100.0
    strength_at_peaks = np.clip((price_pct_change_peaks + osc_pct_change_peaks) / 2.0, 0.0, 1.0)
    
    # For troughs: compare each trough with the previous trough
    prev_price_troughs = price_troughs_series.shift(1)
    prev_osc_troughs = osc_troughs_series.shift(1)
    
    # Bullish divergence at trough locations: price lower low AND oscillator higher low
    price_lower_at_troughs = price_troughs_series < prev_price_troughs
    osc_higher_at_troughs = osc_troughs_series > prev_osc_troughs
    osc_diff_at_troughs = (osc_troughs_series - prev_osc_troughs).abs()
    strong_osc_diff_troughs = osc_diff_at_troughs >= min_divergence_strength
    
    bullish_div_at_troughs = price_lower_at_troughs & osc_higher_at_troughs & strong_osc_diff_troughs
    
    # Calculate strength at trough locations
    # Avoid division by zero: replace 0 with NaN before division
    prev_price_troughs_safe = prev_price_troughs.copy()
    prev_price_troughs_safe = prev_price_troughs_safe.replace(0.0, np.nan)
    price_pct_change_troughs = ((price_troughs_series - prev_price_troughs).abs() / prev_price_troughs_safe) * 100
    price_pct_change_troughs = price_pct_change_troughs.fillna(0.0)
    osc_pct_change_troughs = osc_diff_at_troughs / 100.0
    strength_at_troughs = np.clip((price_pct_change_troughs + osc_pct_change_troughs) / 2.0, 0.0, 1.0)
    
    # Now broadcast these divergence signals to all timestamps
    # Use reindex to map peak/trough signals to the full time series (fully vectorized)
    
    # Reindex bearish divergences from peak timestamps to full series
    # This creates a series with values only at peak locations, NaN elsewhere
    bearish_div_series = bearish_div_at_peaks.reindex(close.index, fill_value=False)
    bearish_strength_series = strength_at_peaks.reindex(close.index, fill_value=0.0)
    
    # Reindex bullish divergences from trough timestamps to full series
    bullish_div_series = bullish_div_at_troughs.reindex(close.index, fill_value=False)
    bullish_strength_series = strength_at_troughs.reindex(close.index, fill_value=0.0)
    
    if debug_enabled:
        bullish_count = int(bullish_div_series.sum())
        bearish_count = int(bearish_div_series.sum())
        log_debug(f"[Strategy7] Divergences detected: bullish={bullish_count}, bearish={bearish_count}")
    
    # Combine signals: handle conflict explicitly
    # LONG signals: bullish divergence
    long_condition = bullish_div_series
    # SHORT signals: bearish divergence
    short_condition = bearish_div_series
    
    # Handle conflict: if both conditions are met at same time, set to NEUTRAL (0)
    # This can happen in rare edge cases
    conflict_mask = long_condition & short_condition
    long_condition = long_condition & ~conflict_mask
    short_condition = short_condition & ~conflict_mask
    
    if debug_enabled:
        conflict_count = int(conflict_mask.sum())
        if conflict_count > 0:
            log_debug(f"[Strategy7] Conflicts detected: {conflict_count} (set to NEUTRAL)")
    
    # Generate signals (optimized: create once)
    signals_array = np.zeros(len(close.index), dtype=np.int8)
    signals_array = np.where(long_condition, 1, signals_array)
    signals_array = np.where(short_condition, -1, signals_array)
    signals = pd.Series(signals_array, index=close.index, dtype="int8")
    
    # Combine strengths (optimized: create once)
    signal_strength_array = np.zeros(len(close.index), dtype=np.float64)
    signal_strength_array = np.where(long_condition, bullish_strength_series.values, signal_strength_array)
    signal_strength_array = np.where(short_condition, bearish_strength_series.values, signal_strength_array)
    signal_strength = pd.Series(signal_strength_array, index=close.index, dtype="float64")
    
    # Handle NaN values (optimized: combine all NaN checks)
    valid_mask = ~oscillator.isna()
    signals = signals.where(valid_mask, 0)
    signal_strength = signal_strength.where(valid_mask, 0.0)
    
    if debug_enabled:
        final_long = int((signals == 1).sum())
        final_short = int((signals == -1).sum())
        final_neutral = int((signals == 0).sum())
        avg_strength = float(signal_strength.mean())
        log_analysis(f"[Strategy7] Final signals: LONG={final_long}, SHORT={final_short}, "
                    f"NEUTRAL={final_neutral}, avg_strength={avg_strength:.3f}")
    
    return signals, signal_strength


__all__ = [
    "generate_signals_divergence_strategy",
]

