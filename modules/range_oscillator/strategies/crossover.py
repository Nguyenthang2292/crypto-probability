"""
Range Oscillator Strategy 3: Zero Line Crossover.

This module provides the zero line crossover signal generation strategy.
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np
import os

from modules.range_oscillator.utils.oscillator_data import get_oscillator_data
from modules.common.utils import log_debug, log_analysis


def generate_signals_crossover_strategy(
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    *,
    oscillator: Optional[pd.Series] = None,
    ma: Optional[pd.Series] = None,
    range_atr: Optional[pd.Series] = None,
    length: int = 50,
    mult: float = 2.0,
    confirmation_bars: int = 2,
    oscillator_threshold: float = 0.0,
    enable_debug: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate trading signals based on Range Oscillator Strategy 3: Zero Line Crossover.
    
    Strategy Logic:
    ---------------
    This strategy generates signals when oscillator crosses the zero line:
    - LONG Signal: Oscillator crosses above zero line + confirmation
    - SHORT Signal: Oscillator crosses below zero line + confirmation
    
    Confirmation ensures the crossover is sustained, not just a brief touch.
    
    Args:
        high: High price series (required if oscillator not provided)
        low: Low price series (required if oscillator not provided)
        close: Close price series (required if oscillator not provided)
        oscillator: Pre-calculated oscillator series (optional, if provided with ma and range_atr)
        ma: Pre-calculated moving average series (optional)
        range_atr: Pre-calculated range ATR series (optional)
        length: Minimum range length for oscillator calculation (default: 50, ignored if oscillator provided)
        mult: Range width multiplier (default: 2.0, ignored if oscillator provided)
        confirmation_bars: Number of bars to confirm crossover (default: 2)
        oscillator_threshold: Threshold for zero line (default: 0.0)
        enable_debug: If True, enable debug logging (default: False)
    
    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
    """
    # Validate parameters
    if confirmation_bars <= 0:
        raise ValueError(f"confirmation_bars must be > 0, got {confirmation_bars}")
    
    # Enable debug logging if requested
    debug_enabled = enable_debug or os.environ.get("RANGE_OSCILLATOR_DEBUG", "false").lower() == "true"
    
    if debug_enabled:
        log_analysis(f"[Strategy3] Starting crossover signal generation")
        log_debug(f"[Strategy3] Parameters: confirmation_bars={confirmation_bars}, "
                 f"threshold={oscillator_threshold}")
    
    # Get oscillator data (either pre-calculated or calculate now)
    oscillator, ma, range_atr = get_oscillator_data(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )
    
    if debug_enabled:
        log_debug(f"[Strategy3] Data shape: oscillator={len(oscillator)}")
    
    # Detect crossovers using shift
    prev_osc = oscillator.shift(1)
    cross_above = (prev_osc <= oscillator_threshold) & (oscillator > oscillator_threshold)
    cross_below = (prev_osc >= -oscillator_threshold) & (oscillator < -oscillator_threshold)
    
    # Count consecutive bars above/below threshold
    above_threshold = (oscillator > oscillator_threshold).astype(int)
    below_threshold = (oscillator < -oscillator_threshold).astype(int)
    
    # Use groupby to count consecutive bars (optimized: sort=False for large datasets)
    above_groups = (above_threshold != above_threshold.shift()).cumsum()
    above_counts = above_threshold.groupby(above_groups, sort=False).cumsum()
    
    below_groups = (below_threshold != below_threshold.shift()).cumsum()
    below_counts = below_threshold.groupby(below_groups, sort=False).cumsum()
    
    # FIXED: Avoid look-ahead bias by emitting signals AFTER confirmation period
    # Logic: Signal is only emitted AFTER confirmation_bars bars have passed since crossover
    # 
    # Example with confirmation_bars=2:
    # - Crossover occurs at t=5 (oscillator crosses above threshold)
    # - Signal is emitted at t=7 (after 2 bars of confirmation)
    # - This ensures we only trade after confirmation is complete, not at crossover time
    
    # Shift crossover detection forward by confirmation_bars
    # This means: if crossover happened at t, signal will be emitted at t + confirmation_bars
    # Using shift(confirmation_bars) moves the crossover signal forward in time
    cross_above_shifted = cross_above.shift(confirmation_bars)
    cross_below_shifted = cross_below.shift(confirmation_bars)
    
    # At current time t, check if:
    # 1. Crossover occurred at t - confirmation_bars (via shifted signal)
    # 2. Current consecutive count >= confirmation_bars (oscillator has stayed above/below for enough bars)
    # 3. We're currently above/below threshold
    # 
    # Note: above_counts[t] represents consecutive bars above threshold ending at t
    # If above_counts[t] >= confirmation_bars, it means oscillator has been above threshold
    # for at least confirmation_bars consecutive bars, which confirms the crossover
    
    # For LONG: crossover happened confirmation_bars ago AND oscillator has stayed above threshold
    above_confirmation_mask = (
        (above_counts >= confirmation_bars) & 
        (above_threshold.astype(bool)) &
        cross_above_shifted.astype(bool).fillna(False)  # Crossover occurred confirmation_bars ago (fill NaN for beginning)
    )
    
    # For SHORT: crossover happened confirmation_bars ago AND oscillator has stayed below threshold
    below_confirmation_mask = (
        (below_counts >= confirmation_bars) & 
        (below_threshold.astype(bool)) &
        cross_below_shifted.astype(bool).fillna(False)  # Crossover occurred confirmation_bars ago (fill NaN for beginning)
    )
    
    # Confirmed crossovers: signal emitted AFTER confirmation period
    cross_above_confirmed = above_confirmation_mask
    cross_below_confirmed = below_confirmation_mask
    
    if debug_enabled:
        cross_above_count = int(cross_above.sum())
        cross_below_count = int(cross_below.sum())
        confirmed_above = int(cross_above_confirmed.sum())
        confirmed_below = int(cross_below_confirmed.sum())
        log_debug(f"[Strategy3] Crossovers: above={cross_above_count} (confirmed={confirmed_above}), "
                 f"below={cross_below_count} (confirmed={confirmed_below})")
    
    # Generate signals
    signals = pd.Series(0, index=oscillator.index, dtype="int8")
    
    # LONG signals: confirmed cross above
    long_condition = cross_above_confirmed
    # SHORT signals: confirmed cross below
    short_condition = cross_below_confirmed
    
    # Handle conflict: if both conditions are met, set to NEUTRAL (0)
    conflict_mask = long_condition & short_condition
    long_condition = long_condition & ~conflict_mask
    short_condition = short_condition & ~conflict_mask
    
    if debug_enabled:
        conflict_count = int(conflict_mask.sum())
        if conflict_count > 0:
            log_debug(f"[Strategy3] Conflicts detected: {conflict_count} (set to NEUTRAL)")
    
    signals = np.where(long_condition, 1, signals)
    signals = np.where(short_condition, -1, signals)
    
    # Maintain signal while oscillator stays above/below threshold (vectorized)
    # Forward fill LONG signals while above threshold, SHORT signals while below threshold
    # Reset on opposite crossover
    
    # Create masks for maintaining signals
    long_maintain_mask = above_threshold.astype(bool)
    short_maintain_mask = below_threshold.astype(bool)
    
    # Optimized forward fill using ffill() with groupby
    # Strategy: Create groups based on maintain conditions, then forward fill within each group
    signal_series = pd.Series(signals, index=oscillator.index, dtype="int8")
    
    # For LONG signals: forward fill 1 while above threshold
    # Create groups: each group is a continuous period above threshold
    # Optimized: use sort=False for better performance
    long_groups = (~long_maintain_mask).cumsum()
    # Keep only LONG signals (1) and forward fill within each group
    long_signals = signal_series.where(signal_series == 1, np.nan)
    long_signals = long_signals.groupby(long_groups, sort=False).ffill()
    # Reset to 0 where condition is not met
    long_signals = long_signals.where(long_maintain_mask, 0).fillna(0).astype("int8")
    
    # For SHORT signals: forward fill -1 while below threshold
    # Create groups: each group is a continuous period below threshold
    # Optimized: use sort=False for better performance
    short_groups = (~short_maintain_mask).cumsum()
    # Keep only SHORT signals (-1) and forward fill within each group
    short_signals = signal_series.where(signal_series == -1, np.nan)
    short_signals = short_signals.groupby(short_groups, sort=False).ffill()
    # Reset to 0 where condition is not met
    short_signals = short_signals.where(short_maintain_mask, 0).fillna(0).astype("int8")
    
    # Combine signals: LONG if long_signals == 1, SHORT if short_signals == -1
    # Optimized: use numpy operations directly
    signals_array = np.where(long_signals.values == 1, 1, 
                             np.where(short_signals.values == -1, -1, 0))
    signals = pd.Series(signals_array, index=oscillator.index, dtype="int8")
    
    # Calculate signal strength
    signal_strength = np.clip(oscillator.abs() / 100.0, 0.0, 1.0)
    signal_strength = np.where(signals == 0, 0.0, signal_strength)
    
    # Handle NaN values (optimized: combine all NaN checks)
    valid_mask = ~oscillator.isna()
    signals = signals.where(valid_mask, 0)
    signal_strength = pd.Series(signal_strength, index=oscillator.index, dtype="float64").where(valid_mask, 0.0)
    
    if debug_enabled:
        final_long = int((signals == 1).sum())
        final_short = int((signals == -1).sum())
        final_neutral = int((signals == 0).sum())
        avg_strength = float(signal_strength.mean())
        log_analysis(f"[Strategy3] Final signals: LONG={final_long}, SHORT={final_short}, "
                    f"NEUTRAL={final_neutral}, avg_strength={avg_strength:.3f}")
    
    return signals, signal_strength


__all__ = [
    "generate_signals_crossover_strategy",
]

