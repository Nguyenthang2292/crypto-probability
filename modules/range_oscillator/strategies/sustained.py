"""
Range Oscillator Strategy 2: Sustained Pressure.

This module provides the sustained pressure signal generation strategy.
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np
import os

from modules.range_oscillator.utils.oscillator_data import get_oscillator_data
from modules.common.utils import log_debug, log_analysis


def generate_signals_sustained_strategy(
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    *,
    oscillator: Optional[pd.Series] = None,
    ma: Optional[pd.Series] = None,
    range_atr: Optional[pd.Series] = None,
    length: int = 50,
    mult: float = 2.0,
    min_bars_above_zero: int = 3,
    min_bars_below_zero: int = 3,
    oscillator_threshold: float = 0.0,
    enable_debug: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate trading signals based on Range Oscillator Strategy 2: Sustained Pressure.
    
    Strategy Logic:
    ---------------
    This strategy focuses on "sustained pressure" as shown in TradingView charts:
    - LONG Signal: Oscillator stays above 0 for at least N bars (sustained bullish pressure)
    - SHORT Signal: Oscillator stays below 0 for at least N bars (sustained bearish pressure)
    
    Key Concept:
    - "Oscillator stays above 0 with green coloring suggests sustained bullish pressure"
    - "Oscillator stays below 0 with red coloring suggests sustained bearish pressure"
    
    Args:
        high: High price series (required if oscillator not provided)
        low: Low price series (required if oscillator not provided)
        close: Close price series (required if oscillator not provided)
        oscillator: Pre-calculated oscillator series (optional, if provided with ma and range_atr)
        ma: Pre-calculated moving average series (optional)
        range_atr: Pre-calculated range ATR series (optional)
        length: Minimum range length for oscillator calculation (default: 50, ignored if oscillator provided)
        mult: Range width multiplier (default: 2.0, ignored if oscillator provided)
        min_bars_above_zero: Minimum consecutive bars above zero for LONG signal (default: 3)
        min_bars_below_zero: Minimum consecutive bars below zero for SHORT signal (default: 3)
        oscillator_threshold: Threshold for zero line (default: 0.0)
        enable_debug: If True, enable debug logging (default: False)
    
    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
    """
    # Validate parameters
    if min_bars_above_zero <= 0:
        raise ValueError(f"min_bars_above_zero must be > 0, got {min_bars_above_zero}")
    if min_bars_below_zero <= 0:
        raise ValueError(f"min_bars_below_zero must be > 0, got {min_bars_below_zero}")
    
    # Enable debug logging if requested
    debug_enabled = enable_debug or os.environ.get("RANGE_OSCILLATOR_DEBUG", "false").lower() == "true"
    
    if debug_enabled:
        log_analysis(f"[Strategy2] Starting sustained pressure signal generation")
        log_debug(f"[Strategy2] Parameters: min_bars_above={min_bars_above_zero}, "
                 f"min_bars_below={min_bars_below_zero}, threshold={oscillator_threshold}")
    
    # Get oscillator data (either pre-calculated or calculate now)
    oscillator, ma, range_atr = get_oscillator_data(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )
    
    if debug_enabled:
        log_debug(f"[Strategy2] Data shape: oscillator={len(oscillator)}")
    
    # Vectorized consecutive bar counting using groupby
    # Optimized for large datasets: groupby operations are efficient for consecutive counting
    above_zero = (oscillator > oscillator_threshold).astype(int)
    below_zero = (oscillator < -oscillator_threshold).astype(int)
    
    # Count consecutive bars above zero using groupby
    # Group changes when value changes (0->1 or 1->0), then cumsum within each group
    above_groups = (above_zero != above_zero.shift()).cumsum()
    bars_above_zero = above_zero.groupby(above_groups, sort=False).cumsum() * above_zero
    
    # Count consecutive bars below zero using groupby
    # sort=False improves performance for large datasets
    below_groups = (below_zero != below_zero.shift()).cumsum()
    bars_below_zero = below_zero.groupby(below_groups, sort=False).cumsum() * below_zero
    
    # Generate signals
    signals = pd.Series(0, index=oscillator.index, dtype="int8")
    
    # LONG signals: oscillator stays above threshold for minimum bars
    long_condition = bars_above_zero >= min_bars_above_zero
    # SHORT signals: oscillator stays below -threshold for minimum bars
    short_condition = bars_below_zero >= min_bars_below_zero
    
    # Handle conflict: if both LONG and SHORT conditions are met, set to NEUTRAL (0)
    # This can happen in edge cases with NaN values or data inconsistencies
    conflict_mask = long_condition & short_condition
    long_condition = long_condition & ~conflict_mask
    short_condition = short_condition & ~conflict_mask
    
    if debug_enabled:
        long_count = int(long_condition.sum())
        short_count = int(short_condition.sum())
        conflict_count = int(conflict_mask.sum())
        log_debug(f"[Strategy2] Initial signals: LONG={long_count}, SHORT={short_count}, "
                 f"conflicts={conflict_count}")
    
    signals = np.where(long_condition, 1, signals)
    signals = np.where(short_condition, -1, signals)
    signals = pd.Series(signals, index=oscillator.index, dtype="int8")
    
    # Calculate signal strength
    # Base strength from oscillator magnitude (0.0 to 1.0, scaled to 0.7 max)
    osc_strength_base = np.clip(oscillator.abs() / 100.0, 0.0, 1.0) * 0.7
    
    # Strength from consecutive bars (0.0 to 1.0)
    long_strength_bars = np.clip(bars_above_zero / (min_bars_above_zero * 2.0), 0.0, 1.0)
    short_strength_bars = np.clip(bars_below_zero / (min_bars_below_zero * 2.0), 0.0, 1.0)
    
    # Combine strengths: take maximum of bar count strength and oscillator magnitude
    long_mask = signals == 1
    short_mask = signals == -1
    
    signal_strength = pd.Series(0.0, index=oscillator.index, dtype="float64")
    signal_strength = np.where(long_mask, np.maximum(long_strength_bars, osc_strength_base), signal_strength)
    signal_strength = np.where(short_mask, np.maximum(short_strength_bars, osc_strength_base), signal_strength)
    signal_strength = pd.Series(signal_strength, index=oscillator.index, dtype="float64")
    
    # Handle NaN values
    valid_mask = ~oscillator.isna()
    signals = signals.where(valid_mask, 0)
    signal_strength = signal_strength.where(valid_mask, 0.0)
    
    if debug_enabled:
        final_long = int((signals == 1).sum())
        final_short = int((signals == -1).sum())
        final_neutral = int((signals == 0).sum())
        avg_strength = float(signal_strength.mean())
        log_analysis(f"[Strategy2] Final signals: LONG={final_long}, SHORT={final_short}, "
                    f"NEUTRAL={final_neutral}, avg_strength={avg_strength:.3f}")
    
    return signals, signal_strength


__all__ = [
    "generate_signals_sustained_strategy",
]

