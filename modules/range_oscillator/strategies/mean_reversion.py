"""
Range Oscillator Strategy 9: Mean Reversion.

This module provides the mean reversion signal generation strategy.
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np
import os

from modules.range_oscillator.utils.oscillator_data import get_oscillator_data
from modules.common.utils import log_debug, log_analysis


def generate_signals_mean_reversion_strategy(
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    *,
    oscillator: Optional[pd.Series] = None,
    ma: Optional[pd.Series] = None,
    range_atr: Optional[pd.Series] = None,
    length: int = 50,
    mult: float = 2.0,
    extreme_threshold: float = 80.0,
    zero_cross_threshold: float = 10.0,
    min_extreme_bars: int = 3,
    transition_bars: int = 5,
    enable_debug: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate trading signals based on Range Oscillator Strategy 9: Mean Reversion.
    
    Strategy Logic:
    ---------------
    This strategy identifies mean reversion opportunities when the oscillator:
    1. Reaches an extreme (above +threshold or below -threshold)
    2. Crosses back toward zero (transition signal)
    
    Key Concept:
    "Look for the oscillator to cross back toward zero after reaching an extreme. 
    These transitions (often marked by blue tones) can identify early reversals 
    or range resets."
    
    Signals:
    - SHORT Signal: Oscillator was at extreme positive (> +threshold), now crossing back toward zero
    - LONG Signal: Oscillator was at extreme negative (< -threshold), now crossing back toward zero
    
    This strategy captures the transition from extreme momentum back to equilibrium,
    which often precedes price reversals or range resets.
    
    Args:
        high: High price series (required if oscillator not provided)
        low: Low price series (required if oscillator not provided)
        close: Close price series (required if oscillator not provided)
        oscillator: Pre-calculated oscillator series (optional, if provided with ma and range_atr)
        ma: Pre-calculated moving average series (optional)
        range_atr: Pre-calculated range ATR series (optional)
        length: Minimum range length for oscillator calculation (default: 50, ignored if oscillator provided)
        mult: Range width multiplier (default: 2.0, ignored if oscillator provided)
        extreme_threshold: Threshold for extreme oscillator values (default: 80.0)
        zero_cross_threshold: Maximum distance from zero to consider as "crossing back" (default: 10.0)
        min_extreme_bars: Minimum bars oscillator must stay at extreme (default: 3)
        transition_bars: Bars to confirm transition back toward zero (default: 5)
        enable_debug: If True, enable debug logging (default: False)
    
    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
    """
    # Validate parameters
    if extreme_threshold < 0:
        raise ValueError(f"extreme_threshold must be >= 0, got {extreme_threshold}")
    if zero_cross_threshold < 0:
        raise ValueError(f"zero_cross_threshold must be >= 0, got {zero_cross_threshold}")
    if min_extreme_bars <= 0:
        raise ValueError(f"min_extreme_bars must be > 0, got {min_extreme_bars}")
    if transition_bars <= 0:
        raise ValueError(f"transition_bars must be > 0, got {transition_bars}")
    
    # Enable debug logging if requested
    debug_enabled = enable_debug or os.environ.get("RANGE_OSCILLATOR_DEBUG", "false").lower() == "true"
    
    if debug_enabled:
        log_analysis(f"[Strategy9] Starting mean reversion signal generation")
        log_debug(f"[Strategy9] Parameters: extreme_threshold={extreme_threshold}, "
                 f"zero_cross_threshold={zero_cross_threshold}, min_extreme_bars={min_extreme_bars}, "
                 f"transition_bars={transition_bars}")
    
    # Get oscillator data (either pre-calculated or calculate now)
    oscillator, ma, range_atr = get_oscillator_data(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )
    
    if debug_enabled:
        log_debug(f"[Strategy9] Data shape: oscillator={len(oscillator)}")
    
    # Vectorized extreme detection
    in_extreme_positive = oscillator > extreme_threshold
    in_extreme_negative = oscillator < -extreme_threshold
    
    # Count consecutive bars in extreme zones using groupby (optimized: sort=False for large datasets)
    extreme_positive_groups = (in_extreme_positive != in_extreme_positive.shift()).cumsum()
    extreme_positive_counts = in_extreme_positive.astype(int).groupby(extreme_positive_groups, sort=False).cumsum()
    
    extreme_negative_groups = (in_extreme_negative != in_extreme_negative.shift()).cumsum()
    extreme_negative_counts = in_extreme_negative.astype(int).groupby(extreme_negative_groups, sort=False).cumsum()
    
    # Detect transitions: was in extreme, now crossing back toward zero
    # Use explicit boolean conversion to avoid FutureWarning
    prev_extreme_positive_shifted = in_extreme_positive.shift(1)
    prev_extreme_positive = prev_extreme_positive_shifted.fillna(False).astype(bool)
    
    prev_extreme_negative_shifted = in_extreme_negative.shift(1)
    prev_extreme_negative = prev_extreme_negative_shifted.fillna(False).astype(bool)
    
    # Transition from extreme positive to near zero
    transition_from_positive = prev_extreme_positive & ~in_extreme_positive & (oscillator.abs() <= zero_cross_threshold)
    transition_from_positive = transition_from_positive & (extreme_positive_counts.shift(1) >= min_extreme_bars)
    
    # Transition from extreme negative to near zero
    transition_from_negative = prev_extreme_negative & ~in_extreme_negative & (oscillator.abs() <= zero_cross_threshold)
    transition_from_negative = transition_from_negative & (extreme_negative_counts.shift(1) >= min_extreme_bars)
    
    if debug_enabled:
        transition_positive_count = int(transition_from_positive.sum())
        transition_negative_count = int(transition_from_negative.sum())
        log_debug(f"[Strategy9] Transitions detected: from_positive={transition_positive_count}, "
                 f"from_negative={transition_negative_count}")
    
    # Initialize signals
    signals = pd.Series(0, index=oscillator.index, dtype="int8")
    
    # SHORT Signal: Transition from extreme positive (reversal from extreme bullish)
    short_condition = transition_from_positive
    # LONG Signal: Transition from extreme negative (reversal from extreme bearish)
    long_condition = transition_from_negative
    
    # Handle conflict: if both conditions are met, set to NEUTRAL (0)
    # This can happen in rare edge cases
    conflict_mask = long_condition & short_condition
    long_condition = long_condition & ~conflict_mask
    short_condition = short_condition & ~conflict_mask
    
    if debug_enabled:
        conflict_count = int(conflict_mask.sum())
        if conflict_count > 0:
            log_debug(f"[Strategy9] Conflicts detected: {conflict_count} (set to NEUTRAL)")
    
    signals = np.where(long_condition, 1, signals)
    signals = np.where(short_condition, -1, signals)
    signals = pd.Series(signals, index=oscillator.index, dtype="int8")
    
    # Calculate signal strength
    # Get oscillator value at extreme start (use previous value when transitioning)
    prev_osc = oscillator.shift(1)
    extreme_magnitude = np.clip(prev_osc.abs() / 100.0, 0.0, 1.0)
    
    zero_proximity = 1.0 - (oscillator.abs() / zero_cross_threshold)
    zero_proximity = np.clip(zero_proximity, 0.0, 1.0)
    
    # Calculate strength for both transitions (optimized: calculate once)
    transition_strength = (extreme_magnitude + zero_proximity) / 2.0
    transition_strength = np.maximum(transition_strength, 0.4)
    
    # Initialize signal strength
    signal_strength = np.zeros(len(oscillator), dtype=np.float64)
    
    # SHORT strength: transition from extreme positive
    signal_strength = np.where(short_condition, transition_strength, signal_strength)
    
    # LONG strength: transition from extreme negative
    signal_strength = np.where(long_condition, transition_strength, signal_strength)
    
    signal_strength = pd.Series(signal_strength, index=oscillator.index, dtype="float64")
    
    # Handle NaN values (optimized: combine all NaN checks)
    valid_mask = ~oscillator.isna()
    signals = signals.where(valid_mask, 0)
    signal_strength = signal_strength.where(valid_mask, 0.0)
    
    if debug_enabled:
        final_long = int((signals == 1).sum())
        final_short = int((signals == -1).sum())
        final_neutral = int((signals == 0).sum())
        avg_strength = float(signal_strength.mean())
        log_analysis(f"[Strategy9] Final signals: LONG={final_long}, SHORT={final_short}, "
                    f"NEUTRAL={final_neutral}, avg_strength={avg_strength:.3f}")
    
    return signals, signal_strength


__all__ = [
    "generate_signals_mean_reversion_strategy",
]

