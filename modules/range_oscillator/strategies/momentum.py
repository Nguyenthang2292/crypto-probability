"""
Range Oscillator Strategy 4: Momentum.

This module provides the momentum-based signal generation strategy.
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np
import os

from modules.range_oscillator.utils.oscillator_data import get_oscillator_data
from modules.common.utils import log_debug, log_analysis


def generate_signals_momentum_strategy(
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    *,
    oscillator: Optional[pd.Series] = None,
    ma: Optional[pd.Series] = None,
    range_atr: Optional[pd.Series] = None,
    length: int = 50,
    mult: float = 2.0,
    momentum_period: int = 3,
    momentum_threshold: float = 5.0,
    enable_debug: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate trading signals based on Range Oscillator Strategy 4: Momentum.
    
    Strategy Logic:
    ---------------
    This strategy uses oscillator momentum (rate of change) to generate signals:
    - LONG Signal: Strong positive momentum (oscillator rising rapidly)
    - SHORT Signal: Strong negative momentum (oscillator falling rapidly)
    
    Args:
        high: High price series (required if oscillator not provided)
        low: Low price series (required if oscillator not provided)
        close: Close price series (required if oscillator not provided)
        oscillator: Pre-calculated oscillator series (optional, if provided with ma and range_atr)
        ma: Pre-calculated moving average series (optional)
        range_atr: Pre-calculated range ATR series (optional)
        length: Minimum range length for oscillator calculation (default: 50, ignored if oscillator provided)
        mult: Range width multiplier (default: 2.0, ignored if oscillator provided)
        momentum_period: Period for calculating momentum (default: 3)
        momentum_threshold: Minimum momentum value for signal (default: 5.0)
        enable_debug: If True, enable debug logging (default: False)
    
    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
    """
    # Validate parameters
    if momentum_period <= 0:
        raise ValueError(f"momentum_period must be > 0, got {momentum_period}")
    if momentum_threshold < 0:
        raise ValueError(f"momentum_threshold must be >= 0, got {momentum_threshold}")
    
    # Enable debug logging if requested
    debug_enabled = enable_debug or os.environ.get("RANGE_OSCILLATOR_DEBUG", "false").lower() == "true"
    
    if debug_enabled:
        log_analysis(f"[Strategy4] Starting momentum signal generation")
        log_debug(f"[Strategy4] Parameters: momentum_period={momentum_period}, "
                 f"momentum_threshold={momentum_threshold}")
    
    # Get oscillator data (either pre-calculated or calculate now)
    oscillator, ma, range_atr = get_oscillator_data(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )
    
    # Validate momentum_period against data length
    if len(oscillator) > 0 and momentum_period >= len(oscillator):
        raise ValueError(f"momentum_period ({momentum_period}) must be < data length ({len(oscillator)})")
    
    if debug_enabled:
        log_debug(f"[Strategy4] Data shape: oscillator={len(oscillator)}")
    
    # Vectorized momentum calculation using shift
    momentum = oscillator - oscillator.shift(momentum_period)
    
    # Initialize signals
    signals = pd.Series(0, index=oscillator.index, dtype="int8")
    
    # LONG Signal: Strong positive momentum
    long_condition = momentum > momentum_threshold
    # SHORT Signal: Strong negative momentum
    short_condition = momentum < -momentum_threshold
    
    # Handle conflict: if both conditions are met, set to NEUTRAL (0)
    # This can happen in edge cases with NaN values or data inconsistencies
    conflict_mask = long_condition & short_condition
    long_condition = long_condition & ~conflict_mask
    short_condition = short_condition & ~conflict_mask
    
    if debug_enabled:
        long_count = int(long_condition.sum())
        short_count = int(short_condition.sum())
        conflict_count = int(conflict_mask.sum())
        log_debug(f"[Strategy4] Initial signals: LONG={long_count}, SHORT={short_count}, "
                 f"conflicts={conflict_count}")
    
    signals = np.where(long_condition, 1, signals)
    signals = np.where(short_condition, -1, signals)
    signals = pd.Series(signals, index=oscillator.index, dtype="int8")
    
    # Calculate signal strength
    # Strength based on momentum magnitude
    momentum_strength = np.clip(momentum.abs() / (momentum_threshold * 3.0), 0.0, 1.0)
    
    # Also consider current oscillator position
    osc_strength = np.clip(oscillator.abs() / 100.0, 0.0, 1.0) * 0.5
    
    # Combine strengths: take maximum of momentum strength and oscillator position
    signal_mask = long_condition | short_condition
    signal_strength = np.where(signal_mask, np.maximum(momentum_strength, osc_strength), 0.0)
    signal_strength = pd.Series(signal_strength, index=oscillator.index, dtype="float64")
    
    # Handle NaN values and insufficient data (optimized: combine all NaN checks)
    valid_mask = ~(oscillator.isna() | momentum.isna())
    signals = signals.where(valid_mask, 0)
    signal_strength = signal_strength.where(valid_mask, 0.0)
    
    if debug_enabled:
        final_long = int((signals == 1).sum())
        final_short = int((signals == -1).sum())
        final_neutral = int((signals == 0).sum())
        avg_strength = float(signal_strength.mean())
        log_analysis(f"[Strategy4] Final signals: LONG={final_long}, SHORT={final_short}, "
                    f"NEUTRAL={final_neutral}, avg_strength={avg_strength:.3f}")
    
    return signals, signal_strength


__all__ = [
    "generate_signals_momentum_strategy",
]

