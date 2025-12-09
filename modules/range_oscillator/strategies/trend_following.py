"""
Range Oscillator Strategy 8: Trend Following.

This module provides the trend following signal generation strategy.
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np
import os

from modules.range_oscillator.utils.oscillator_data import get_oscillator_data
from modules.common.utils import log_debug, log_analysis


def generate_signals_trend_following_strategy(
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    *,
    oscillator: Optional[pd.Series] = None,
    ma: Optional[pd.Series] = None,
    range_atr: Optional[pd.Series] = None,
    length: int = 50,
    mult: float = 2.0,
    trend_filter_period: int = 10,
    oscillator_threshold: float = 20.0,
    require_consistency: bool = True,
    enable_debug: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate trading signals based on Range Oscillator Strategy 8: Trend Following.
    
    Strategy Logic:
    ---------------
    This strategy follows the trend by requiring consistent oscillator position:
    - LONG Signal: Oscillator consistently above threshold with upward trend
    - SHORT Signal: Oscillator consistently below -threshold with downward trend
    
    This is a trend-following approach that filters out choppy markets.
    
    Args:
        high: High price series (required if oscillator not provided)
        low: Low price series (required if oscillator not provided)
        close: Close price series (required if oscillator not provided)
        oscillator: Pre-calculated oscillator series (optional, if provided with ma and range_atr)
        ma: Pre-calculated moving average series (optional)
        range_atr: Pre-calculated range ATR series (optional)
        length: Minimum range length for oscillator calculation (default: 50, ignored if oscillator provided)
        mult: Range width multiplier (default: 2.0, ignored if oscillator provided)
        trend_filter_period: Period for trend filter (default: 10)
        oscillator_threshold: Minimum oscillator value for signal (default: 20.0)
        require_consistency: If True, require consistent oscillator position (default: True)
        enable_debug: If True, enable debug logging (default: False)
    
    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
    """
    # Validate parameters
    if trend_filter_period <= 0:
        raise ValueError(f"trend_filter_period must be > 0, got {trend_filter_period}")
    if oscillator_threshold < 0:
        raise ValueError(f"oscillator_threshold must be >= 0, got {oscillator_threshold}")
    
    # Enable debug logging if requested
    debug_enabled = enable_debug or os.environ.get("RANGE_OSCILLATOR_DEBUG", "false").lower() == "true"
    
    if debug_enabled:
        log_analysis(f"[Strategy8] Starting trend following signal generation")
        log_debug(f"[Strategy8] Parameters: trend_filter_period={trend_filter_period}, "
                 f"oscillator_threshold={oscillator_threshold}, require_consistency={require_consistency}")
    
    # Get oscillator data (either pre-calculated or calculate now)
    oscillator, ma, range_atr = get_oscillator_data(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )
    
    if debug_enabled:
        log_debug(f"[Strategy8] Data shape: oscillator={len(oscillator)}")
    
    # Vectorized trend analysis using rolling windows
    # Calculate oscillator trend (slope) - OPTIMIZED: Use vectorized shift instead of rolling().apply()
    # The calculation (x.iloc[-1] - x.iloc[0]) / trend_filter_period in a rolling window
    # is equivalent to: (current_value - value_N_bars_ago) / trend_filter_period
    # This is hundreds of times faster than using apply() with lambda
    min_periods = int(trend_filter_period * 0.7)
    # Calculate trend: difference between current and (trend_filter_period - 1) bars ago
    osc_trend = (oscillator - oscillator.shift(trend_filter_period - 1)) / trend_filter_period
    # Set NaN for values that don't have enough historical data (both current and shifted must be valid)
    # Also ensure we have at least min_periods valid values in the window
    valid_window_mask = oscillator.notna().rolling(window=trend_filter_period, min_periods=min_periods).sum() >= min_periods
    osc_trend = osc_trend.where(valid_window_mask & oscillator.notna() & oscillator.shift(trend_filter_period - 1).notna())
    
    # Count bars above/below threshold using rolling
    above_threshold_count = (oscillator > oscillator_threshold).rolling(
        window=trend_filter_period, min_periods=int(trend_filter_period * 0.7)
    ).sum()
    below_threshold_count = (oscillator < -oscillator_threshold).rolling(
        window=trend_filter_period, min_periods=int(trend_filter_period * 0.7)
    ).sum()
    
    # Calculate consistency ratio (optimized: avoid creating Series for constant)
    consistency_ratio = pd.concat([above_threshold_count, below_threshold_count], axis=1).max(axis=1) / trend_filter_period
    
    # Initialize signals
    signals = pd.Series(0, index=oscillator.index, dtype="int8")
    
    # LONG Signal: Consistent above threshold + upward trend
    long_condition = (oscillator > oscillator_threshold) & (osc_trend > 0)
    if require_consistency:
        long_condition = long_condition & (consistency_ratio >= 0.7)
    
    # SHORT Signal: Consistent below -threshold + downward trend
    short_condition = (oscillator < -oscillator_threshold) & (osc_trend < 0)
    if require_consistency:
        short_condition = short_condition & (consistency_ratio >= 0.7)
    
    # Handle conflict: if both conditions are met, set to NEUTRAL (0)
    # This can happen in edge cases with NaN values or data inconsistencies
    conflict_mask = long_condition & short_condition
    long_condition = long_condition & ~conflict_mask
    short_condition = short_condition & ~conflict_mask
    
    if debug_enabled:
        long_count = int(long_condition.sum())
        short_count = int(short_condition.sum())
        conflict_count = int(conflict_mask.sum())
        log_debug(f"[Strategy8] Initial signals: LONG={long_count}, SHORT={short_count}, "
                 f"conflicts={conflict_count}")
    
    signals = np.where(long_condition, 1, signals)
    signals = np.where(short_condition, -1, signals)
    signals = pd.Series(signals, index=oscillator.index, dtype="int8")
    
    # Calculate signal strength
    value_strength = np.clip(oscillator.abs() / 100.0, 0.0, 1.0)
    trend_strength = np.clip(osc_trend.abs() / 10.0, 0.0, 1.0)
    
    # Combine strengths: average of value strength and trend strength
    signal_mask = signals != 0
    signal_strength = np.where(signal_mask, (value_strength + trend_strength) / 2.0, 0.0)
    signal_strength = pd.Series(signal_strength, index=oscillator.index, dtype="float64")
    
    # Handle NaN values (optimized: combine all NaN checks)
    valid_mask = ~(oscillator.isna() | osc_trend.isna())
    signals = signals.where(valid_mask, 0)
    signal_strength = signal_strength.where(valid_mask, 0.0)
    
    if debug_enabled:
        final_long = int((signals == 1).sum())
        final_short = int((signals == -1).sum())
        final_neutral = int((signals == 0).sum())
        avg_strength = float(signal_strength.mean())
        log_analysis(f"[Strategy8] Final signals: LONG={final_long}, SHORT={final_short}, "
                    f"NEUTRAL={final_neutral}, avg_strength={avg_strength:.3f}")
    
    return signals, signal_strength


__all__ = [
    "generate_signals_trend_following_strategy",
]

