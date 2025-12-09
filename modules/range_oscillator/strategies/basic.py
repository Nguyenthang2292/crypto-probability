"""
Range Oscillator Strategy 1: Basic Oscillator Signals.

This module provides the basic oscillator signal generation strategy.
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np
import os

from modules.range_oscillator.utils.oscillator_data import get_oscillator_data
from modules.common.utils import log_debug, log_analysis


def generate_signals_basic_strategy(
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    *,
    oscillator: Optional[pd.Series] = None,
    ma: Optional[pd.Series] = None,
    range_atr: Optional[pd.Series] = None,
    length: int = 50,
    mult: float = 2.0,
    oscillator_threshold: float = 0.0,
    require_trend_confirmation: bool = True,
    use_breakout_signals: bool = True,
    enable_debug: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate trading signals based on Range Oscillator Strategy 1.
    
    Strategy Logic:
    ---------------
    1. LONG Signal: 
       - Oscillator > threshold (default: 0.0)
       - Trend direction is bullish (close > MA)
       - Optional: Strong bullish breakout (close > MA + RangeATR)
    
    2. SHORT Signal:
       - Oscillator < -threshold (default: 0.0)
       - Trend direction is bearish (close < MA)
       - Optional: Strong bearish breakout (close < MA - RangeATR)
    
    3. NEUTRAL Signal:
       - Oscillator crosses zero line
       - Trend flip detected
       - Oscillator within threshold range
    
    This strategy is based on the concept shown in TradingView charts where:
    - Oscillator staying above 0 with green coloring suggests sustained bullish pressure
    - Oscillator staying below 0 with red coloring suggests bearish pressure
    
    Args:
        high: High price series (required if oscillator not provided)
        low: Low price series (required if oscillator not provided)
        close: Close price series (required if oscillator not provided)
        oscillator: Pre-calculated oscillator series (optional, if provided with ma and range_atr)
        ma: Pre-calculated moving average series (optional)
        range_atr: Pre-calculated range ATR series (optional)
        length: Minimum range length for oscillator calculation (default: 50, ignored if oscillator provided)
        mult: Range width multiplier (default: 2.0, ignored if oscillator provided)
        oscillator_threshold: Minimum oscillator value for signal generation (default: 0.0)
        require_trend_confirmation: If True, require trend direction confirmation (default: True)
        use_breakout_signals: If True, use strong breakout signals (default: True)
        enable_debug: If True, enable debug logging (default: False)
    
    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
    """
    # Enable debug logging if requested (check environment variable or parameter)
    debug_enabled = enable_debug or os.environ.get("RANGE_OSCILLATOR_DEBUG", "false").lower() == "true"
    
    if debug_enabled:
        log_analysis(f"[Strategy1] Starting signal generation")
        log_debug(f"[Strategy1] Parameters: threshold={oscillator_threshold}, "
                 f"trend_confirmation={require_trend_confirmation}, "
                 f"breakout_signals={use_breakout_signals}")
    
    # Get oscillator data (either pre-calculated or calculate now)
    oscillator, ma, range_atr = get_oscillator_data(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )
    
    if debug_enabled:
        log_debug(f"[Strategy1] Data shape: oscillator={len(oscillator)}, "
                 f"ma={len(ma)}, range_atr={len(range_atr)}")
    
    # Get close series for trend calculation
    if close is None:
        if high is not None and low is not None:
            # Approximate close as mid-point if not provided
            close = (high + low) / 2
        elif oscillator is not None and not require_trend_confirmation:
            # If we have oscillator but no close, and trend confirmation is not required,
            # create dummy close for compatibility
            close = pd.Series(oscillator.values, index=oscillator.index)
        else:
            raise ValueError(
                "close is required for trend calculation. "
                "Either provide close prices or set require_trend_confirmation=False"
            )
    
    # Vectorized calculations
    # Determine trend direction
    trend_bullish = close > ma
    trend_bearish = close < ma
    
    # Check for breakouts (optimized: avoid creating Series if not needed)
    if use_breakout_signals:
        strong_bullish_breakout = close > ma + range_atr
        strong_bearish_breakout = close < ma - range_atr
    else:
        # Use numpy array for better performance with large datasets
        strong_bullish_breakout = pd.Series(False, index=oscillator.index, dtype=bool)
        strong_bearish_breakout = pd.Series(False, index=oscillator.index, dtype=bool)
    
    # Calculate signal strength (0.0 to 1.0) - based on oscillator distance from zero
    abs_osc = oscillator.abs()
    signal_strength = np.clip(abs_osc / 100.0, 0.0, 1.0)
    
    # Boost strength for strong breakouts
    signal_strength = np.where(strong_bullish_breakout, np.clip(signal_strength * 1.5, 0.0, 1.0), signal_strength)
    signal_strength = np.where(strong_bearish_breakout, np.clip(signal_strength * 1.5, 0.0, 1.0), signal_strength)
    
    # Initialize signals
    signals = pd.Series(0, index=oscillator.index, dtype="int8")
    
    # LONG Signal Conditions
    long_condition = oscillator > oscillator_threshold
    if require_trend_confirmation:
        long_condition = long_condition & (trend_bullish | strong_bullish_breakout)
    
    # SHORT Signal Conditions
    short_condition = oscillator < -oscillator_threshold
    if require_trend_confirmation:
        short_condition = short_condition & (trend_bearish | strong_bearish_breakout)
    
    # Handle conflict: if both LONG and SHORT conditions are met, set to NEUTRAL (0)
    # This can happen in edge cases with NaN values or data inconsistencies
    conflict_mask = long_condition & short_condition
    long_condition = long_condition & ~conflict_mask
    short_condition = short_condition & ~conflict_mask
    
    signals = np.where(long_condition, 1, signals)
    signals = np.where(short_condition, -1, signals)
    
    if debug_enabled:
        long_count = int(long_condition.sum())
        short_count = int(short_condition.sum())
        conflict_count = int(conflict_mask.sum())
        log_debug(f"[Strategy1] Initial signals: LONG={long_count}, SHORT={short_count}, "
                 f"conflicts={conflict_count}")
    
    # Detect zero line crosses (optimized: use vectorized operations)
    prev_osc = oscillator.shift(1)
    zero_cross_up = (prev_osc <= 0) & (oscillator > 0)
    zero_cross_down = (prev_osc >= 0) & (oscillator < 0)
    zero_cross = zero_cross_up | zero_cross_down
    
    if debug_enabled:
        zero_cross_count = int(zero_cross.sum())
        log_debug(f"[Strategy1] Zero crosses detected: {zero_cross_count}")
    
    # Convert to Series for easier manipulation
    signals = pd.Series(signals, index=oscillator.index, dtype="int8")
    
    # Forward fill to maintain previous signal when oscillator is within threshold
    # Strategy: maintain LONG/SHORT signals until zero cross or new signal appears
    # Forward fill non-zero values, but reset to 0 at zero crosses
    # Optimized: use vectorized operations for large datasets
    # Replace 0 with NaN for forward fill, then fill NaN back to 0
    signals_for_ffill = signals.copy()
    signals_for_ffill = signals_for_ffill.replace(0, np.nan)
    signals_filled = signals_for_ffill.ffill()
    signals_filled = signals_filled.fillna(0.0)
    signals_filled = signals_filled.astype("int8")
    # Reset to 0 at zero cross positions (these are intentional NEUTRAL signals)
    # Use numpy operations directly for better performance
    signals_array = np.where(zero_cross.values, 0, signals_filled.values).astype("int8")
    signals = pd.Series(signals_array, index=oscillator.index, dtype="int8")
    
    # Handle NaN values (optimized: combine all NaN checks in one mask)
    valid_mask = ~(oscillator.isna() | ma.isna() | range_atr.isna())
    signals = signals.where(valid_mask, 0)
    signal_strength = pd.Series(signal_strength, index=oscillator.index).where(valid_mask, 0.0)
    
    if debug_enabled:
        final_long = int((signals == 1).sum())
        final_short = int((signals == -1).sum())
        final_neutral = int((signals == 0).sum())
        avg_strength = float(signal_strength.mean())
        log_analysis(f"[Strategy1] Final signals: LONG={final_long}, SHORT={final_short}, "
                    f"NEUTRAL={final_neutral}, avg_strength={avg_strength:.3f}")
    
    return signals, signal_strength


__all__ = [
    "generate_signals_basic_strategy",
]

