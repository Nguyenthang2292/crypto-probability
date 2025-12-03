"""
Range Oscillator Signal Strategy.

This module provides signal generation strategies based on the Range Oscillator indicator.
The strategies generate LONG/SHORT signals based on:
- Oscillator position relative to zero line
- Sustained pressure (oscillator staying above/below 0)
- Trend direction and breakout conditions
- Zero line crossovers
- Momentum and divergence detection
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np

from modules.range_oscillator.range_oscillator import calculate_range_oscillator


def _get_oscillator_data(
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    oscillator: Optional[pd.Series] = None,
    ma: Optional[pd.Series] = None,
    range_atr: Optional[pd.Series] = None,
    length: int = 50,
    mult: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Helper function to get oscillator data, either from pre-calculated values or by calculating.
    
    This function avoids redundant calculations when oscillator data is already available.
    
    Args:
        high: High price series (required if oscillator not provided)
        low: Low price series (required if oscillator not provided)
        close: Close price series (required if oscillator not provided)
        oscillator: Pre-calculated oscillator series (optional)
        ma: Pre-calculated moving average series (optional)
        range_atr: Pre-calculated range ATR series (optional)
        length: Minimum range length for oscillator calculation (default: 50)
        mult: Range width multiplier (default: 2.0)
    
    Returns:
        Tuple containing (oscillator, ma, range_atr)
    """
    if oscillator is not None and ma is not None and range_atr is not None:
        # Use pre-calculated values
        return oscillator, ma, range_atr
    elif high is not None and low is not None and close is not None:
        # Calculate oscillator
        oscillator, _, ma, range_atr = calculate_range_oscillator(
            high=high,
            low=low,
            close=close,
            length=length,
            mult=mult,
        )
        return oscillator, ma, range_atr
    else:
        raise ValueError("Either provide (oscillator, ma, range_atr) or (high, low, close) with length and mult")


def generate_signals_strategy1(
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
    
    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
    """
    # Get oscillator data (either pre-calculated or calculate now)
    oscillator, ma, range_atr = _get_oscillator_data(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )
    
    # Get close series for trend calculation
    if close is None:
        if high is not None and low is not None:
            # Approximate close as mid-point if not provided
            close = (high + low) / 2
        elif oscillator is not None:
            # If we have oscillator but no close, create dummy close for compatibility
            # This allows strategy to work without trend confirmation
            close = pd.Series(oscillator.values, index=oscillator.index)
        else:
            raise ValueError("close is required for trend calculation")
    
    # Vectorized calculations
    # Determine trend direction
    trend_bullish = close > ma
    trend_bearish = close < ma
    
    # Check for breakouts
    strong_bullish_breakout = use_breakout_signals & (close > ma + range_atr)
    strong_bearish_breakout = use_breakout_signals & (close < ma - range_atr)
    
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
    signals = np.where(long_condition, 1, signals)
    
    # SHORT Signal Conditions
    short_condition = oscillator < -oscillator_threshold
    if require_trend_confirmation:
        short_condition = short_condition & (trend_bearish | strong_bearish_breakout)
    signals = np.where(short_condition, -1, signals)
    
    # Detect zero line crosses
    prev_osc = oscillator.shift(1)
    zero_cross_up = (prev_osc <= 0) & (oscillator > 0)
    zero_cross_down = (prev_osc >= 0) & (oscillator < 0)
    zero_cross = zero_cross_up | zero_cross_down
    
    # For zero cross, set signal to 0, otherwise maintain previous signal
    signals = np.where(zero_cross, 0, signals)
    
    # Forward fill to maintain previous signal when oscillator is within threshold
    signals = pd.Series(signals, index=oscillator.index, dtype="int8")
    signals = signals.replace(0, np.nan).ffill().fillna(0).astype("int8")
    
    # Handle NaN values
    valid_mask = ~(oscillator.isna() | ma.isna() | range_atr.isna())
    signals = signals.where(valid_mask, 0)
    signal_strength = pd.Series(signal_strength, index=oscillator.index).where(valid_mask, 0.0)
    
    return signals, signal_strength


def generate_signals_strategy2_sustained(
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
    
    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
    """
    # Get oscillator data (either pre-calculated or calculate now)
    oscillator, ma, range_atr = _get_oscillator_data(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )
    
    # Vectorized consecutive bar counting using groupby
    above_zero = (oscillator > oscillator_threshold).astype(int)
    below_zero = (oscillator < -oscillator_threshold).astype(int)
    
    # Count consecutive bars above zero using groupby
    above_groups = (above_zero != above_zero.shift()).cumsum()
    bars_above_zero = above_zero.groupby(above_groups).cumsum() * above_zero
    
    # Count consecutive bars below zero using groupby
    below_groups = (below_zero != below_zero.shift()).cumsum()
    bars_below_zero = below_zero.groupby(below_groups).cumsum() * below_zero
    
    # Generate signals
    signals = pd.Series(0, index=oscillator.index, dtype="int8")
    signals = pd.Series(np.where(bars_above_zero >= min_bars_above_zero, 1, signals), index=oscillator.index, dtype="int8")
    signals = pd.Series(np.where(bars_below_zero >= min_bars_below_zero, -1, signals), index=oscillator.index, dtype="int8")
    
    # Calculate signal strength
    signal_strength = pd.Series(0.0, index=oscillator.index, dtype="float64")
    
    # LONG strength
    long_mask = signals == 1
    long_strength_bars = np.clip(bars_above_zero / (min_bars_above_zero * 2.0), 0.0, 1.0)
    long_strength_osc = np.clip(oscillator.abs() / 100.0, 0.0, 1.0) * 0.7
    signal_strength = pd.Series(np.where(long_mask, np.maximum(long_strength_bars, long_strength_osc), signal_strength), index=oscillator.index, dtype="float64")
    
    # SHORT strength
    short_mask = signals == -1
    short_strength_bars = np.clip(bars_below_zero / (min_bars_below_zero * 2.0), 0.0, 1.0)
    short_strength_osc = np.clip(oscillator.abs() / 100.0, 0.0, 1.0) * 0.7
    signal_strength = pd.Series(np.where(short_mask, np.maximum(short_strength_bars, short_strength_osc), signal_strength), index=oscillator.index, dtype="float64")
    
    # Handle NaN values
    valid_mask = ~oscillator.isna()
    signals = signals.where(valid_mask, 0)
    signal_strength = signal_strength.where(valid_mask, 0.0)
    
    return signals, signal_strength


def generate_signals_strategy3_crossover(
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
    
    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
    """
    # Get oscillator data (either pre-calculated or calculate now)
    oscillator, ma, range_atr = _get_oscillator_data(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )
    
    # Detect crossovers using shift
    prev_osc = oscillator.shift(1)
    cross_above = (prev_osc <= oscillator_threshold) & (oscillator > oscillator_threshold)
    cross_below = (prev_osc >= -oscillator_threshold) & (oscillator < -oscillator_threshold)
    
    # Count consecutive bars above/below threshold after crossover
    above_threshold = (oscillator > oscillator_threshold).astype(int)
    below_threshold = (oscillator < -oscillator_threshold).astype(int)
    
    # Use groupby to count consecutive bars
    above_groups = (above_threshold != above_threshold.shift()).cumsum()
    above_counts = above_threshold.groupby(above_groups).cumsum()
    
    below_groups = (below_threshold != below_threshold.shift()).cumsum()
    below_counts = below_threshold.groupby(below_groups).cumsum()
    
    # Vectorized confirmation check: For each crossover point, check if we have enough consecutive bars
    # At crossover point idx, we need above_counts[idx + confirmation_bars - 1] >= confirmation_bars
    # This ensures that from idx to idx + confirmation_bars - 1, all bars are above threshold
    # Shift counts forward to check future values at crossover points
    above_counts_shifted = above_counts.shift(-(confirmation_bars - 1))
    below_counts_shifted = below_counts.shift(-(confirmation_bars - 1))
    
    # Check if future count meets confirmation requirement
    # At crossover point idx, above_counts[idx] should be 1 (start of new group)
    # And above_counts[idx + confirmation_bars - 1] should be >= confirmation_bars
    # Also ensure we're still above threshold at the shifted position
    # Only check confirmation at crossover points (where cross_above/cross_below is True)
    above_confirmation_mask = (
        (above_counts_shifted >= confirmation_bars) & 
        (above_threshold == 1) &
        (above_counts == 1)  # Ensure crossover point is start of new group
    )
    below_confirmation_mask = (
        (below_counts_shifted >= confirmation_bars) & 
        (below_threshold == 1) &
        (below_counts == 1)  # Ensure crossover point is start of new group
    )
    
    # Apply confirmation: crossover must occur AND future confirmation must be met
    # Only confirm crossovers that meet the confirmation requirement
    cross_above_confirmed = cross_above & above_confirmation_mask
    cross_below_confirmed = cross_below & below_confirmation_mask
    
    # Generate signals
    signals = pd.Series(0, index=oscillator.index, dtype="int8")
    signals = np.where(cross_above_confirmed, 1, signals)
    signals = np.where(cross_below_confirmed, -1, signals)
    
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
    long_groups = (~long_maintain_mask).cumsum()
    # Keep only LONG signals (1) and forward fill within each group
    long_signals = signal_series.where(signal_series == 1, np.nan)
    long_signals = long_signals.groupby(long_groups).ffill()
    # Reset to 0 where condition is not met
    long_signals = long_signals.where(long_maintain_mask, 0).fillna(0).astype("int8")
    
    # For SHORT signals: forward fill -1 while below threshold
    # Create groups: each group is a continuous period below threshold
    short_groups = (~short_maintain_mask).cumsum()
    # Keep only SHORT signals (-1) and forward fill within each group
    short_signals = signal_series.where(signal_series == -1, np.nan)
    short_signals = short_signals.groupby(short_groups).ffill()
    # Reset to 0 where condition is not met
    short_signals = short_signals.where(short_maintain_mask, 0).fillna(0).astype("int8")
    
    # Combine signals: LONG if long_signals == 1, SHORT if short_signals == -1
    signals = np.where(long_signals == 1, 1, np.where(short_signals == -1, -1, 0))
    signals = pd.Series(signals, index=oscillator.index, dtype="int8")
    
    # Calculate signal strength
    signal_strength = np.clip(oscillator.abs() / 100.0, 0.0, 1.0)
    signal_strength = np.where(signals == 0, 0.0, signal_strength)
    
    # Handle NaN values
    valid_mask = ~oscillator.isna()
    signals = pd.Series(signals, index=oscillator.index, dtype="int8").where(valid_mask, 0)
    signal_strength = pd.Series(signal_strength, index=oscillator.index, dtype="float64").where(valid_mask, 0.0)
    
    return signals, signal_strength


def generate_signals_strategy4_momentum(
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
    
    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
    """
    # Get oscillator data (either pre-calculated or calculate now)
    oscillator, ma, range_atr = _get_oscillator_data(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )
    
    # Vectorized momentum calculation using shift
    momentum = oscillator - oscillator.shift(momentum_period)
    
    # Initialize signals
    signals = pd.Series(0, index=oscillator.index, dtype="int8")
    
    # LONG Signal: Strong positive momentum
    long_mask = momentum > momentum_threshold
    signals = pd.Series(np.where(long_mask, 1, signals), index=oscillator.index, dtype="int8")
    
    # SHORT Signal: Strong negative momentum
    short_mask = momentum < -momentum_threshold
    signals = pd.Series(np.where(short_mask, -1, signals), index=oscillator.index, dtype="int8")
    
    # Calculate signal strength
    signal_strength = pd.Series(0.0, index=oscillator.index, dtype="float64")
    
    # Strength based on momentum magnitude
    momentum_strength = np.clip(momentum.abs() / (momentum_threshold * 3.0), 0.0, 1.0)
    
    # Also consider current oscillator position
    osc_strength = np.clip(oscillator.abs() / 100.0, 0.0, 1.0) * 0.5
    
    # Combine strengths
    signal_strength = pd.Series(np.where(long_mask | short_mask, np.maximum(momentum_strength, osc_strength), 0.0), index=oscillator.index, dtype="float64")
    
    # Handle NaN values and insufficient data
    valid_mask = ~(oscillator.isna() | momentum.isna())
    signals = signals.where(valid_mask, 0)
    signal_strength = signal_strength.where(valid_mask, 0.0)
    
    return signals, signal_strength


def generate_signals_strategy5_combined(
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    *,
    oscillator: Optional[pd.Series] = None,
    ma: Optional[pd.Series] = None,
    range_atr: Optional[pd.Series] = None,
    length: int = 50,
    mult: float = 2.0,
    use_sustained: bool = True,
    use_crossover: bool = True,
    use_momentum: bool = True,
    min_bars_sustained: int = 3,
    confirmation_bars: int = 2,
    momentum_period: int = 3,
    momentum_threshold: float = 5.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate trading signals based on Range Oscillator Strategy 5: Combined.
    
    Strategy Logic:
    ---------------
    This strategy combines multiple signal generation methods:
    1. Sustained pressure (oscillator staying above/below 0)
    2. Zero line crossover with confirmation
    3. Momentum-based signals
    
    Signals are generated when at least one method confirms, with strength
    calculated as the average of all active methods.
    
    Args:
        high: High price series (required if oscillator not provided)
        low: Low price series (required if oscillator not provided)
        close: Close price series (required if oscillator not provided)
        oscillator: Pre-calculated oscillator series (optional, if provided with ma and range_atr)
        ma: Pre-calculated moving average series (optional)
        range_atr: Pre-calculated range ATR series (optional)
        length: Minimum range length for oscillator calculation (default: 50, ignored if oscillator provided)
        mult: Range width multiplier (default: 2.0, ignored if oscillator provided)
        use_sustained: Enable sustained pressure signals (default: True)
        use_crossover: Enable crossover signals (default: True)
        use_momentum: Enable momentum signals (default: True)
        min_bars_sustained: Minimum bars for sustained signal (default: 3)
        confirmation_bars: Bars for crossover confirmation (default: 2)
        momentum_period: Period for momentum calculation (default: 3)
        momentum_threshold: Threshold for momentum signal (default: 5.0)
    
    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
    """
    # Calculate Range Oscillator ONCE (or use pre-calculated values)
    oscillator, ma, range_atr = _get_oscillator_data(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )
    
    index = oscillator.index
    
    # Get signals from individual strategies (pass pre-calculated values to avoid recalculation)
    signal_votes = []
    strength_votes = []
    
    if use_sustained:
        sig_sustained, str_sustained = generate_signals_strategy2_sustained(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            min_bars_above_zero=min_bars_sustained,
            min_bars_below_zero=min_bars_sustained,
        )
        signal_votes.append(sig_sustained)
        strength_votes.append(str_sustained)
    
    if use_crossover:
        sig_cross, str_cross = generate_signals_strategy3_crossover(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            confirmation_bars=confirmation_bars,
        )
        signal_votes.append(sig_cross)
        strength_votes.append(str_cross)
    
    if use_momentum:
        sig_mom, str_mom = generate_signals_strategy4_momentum(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            momentum_period=momentum_period,
            momentum_threshold=momentum_threshold,
        )
        signal_votes.append(sig_mom)
        strength_votes.append(str_mom)
    
    if not signal_votes:
        # Fallback to basic strategy if no methods enabled
        return generate_signals_strategy1(
            high=high, low=low, close=close,
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            length=length, mult=mult
        )
    
    # Combine signals: majority vote (vectorized)
    signals = pd.Series(0, index=index, dtype="int8")
    signal_strength = pd.Series(0.0, index=index, dtype="float64")
    
    # Count votes vectorized
    long_votes = pd.Series(0, index=index, dtype="int32")
    short_votes = pd.Series(0, index=index, dtype="int32")
    
    for sig in signal_votes:
        long_votes += (sig == 1).astype(int)
        short_votes += (sig == -1).astype(int)
    
    # Determine signals based on votes
    signals = np.where((long_votes > short_votes) & (long_votes > 0), 1, signals)
    signals = np.where((short_votes > long_votes) & (short_votes > 0), -1, signals)
    
    # Calculate average strength vectorized
    long_mask = signals == 1
    short_mask = signals == -1
    
    if long_mask.any():
        # Average strength from strategies that voted LONG
        long_strength_sum = pd.Series(0.0, index=index)
        long_strength_count = pd.Series(0, index=index)
        for sig, str_vote in zip(signal_votes, strength_votes):
            mask = (sig == 1) & long_mask
            long_strength_sum += str_vote.where(mask, 0.0)
            long_strength_count += mask.astype(int)
        signal_strength = np.where(long_mask, long_strength_sum / long_strength_count.replace(0, np.nan).fillna(1), signal_strength)
    
    if short_mask.any():
        # Average strength from strategies that voted SHORT
        short_strength_sum = pd.Series(0.0, index=index)
        short_strength_count = pd.Series(0, index=index)
        for sig, str_vote in zip(signal_votes, strength_votes):
            mask = (sig == -1) & short_mask
            short_strength_sum += str_vote.where(mask, 0.0)
            short_strength_count += mask.astype(int)
        signal_strength = np.where(short_mask, short_strength_sum / short_strength_count.replace(0, np.nan).fillna(1), signal_strength)
    
    # Ensure signals and signal_strength are Series
    signals = pd.Series(signals, index=index, dtype="int8")
    signal_strength = pd.Series(signal_strength, index=index, dtype="float64")
    
    return signals, signal_strength


def generate_signals_strategy6_breakout(
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    *,
    oscillator: Optional[pd.Series] = None,
    ma: Optional[pd.Series] = None,
    range_atr: Optional[pd.Series] = None,
    length: int = 50,
    mult: float = 2.0,
    upper_threshold: float = 100.0,
    lower_threshold: float = -100.0,
    require_confirmation: bool = True,
    confirmation_bars: int = 2,
    detect_exhaustion: bool = True,
    exhaustion_threshold: float = 150.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate trading signals based on Range Oscillator Strategy 6: Range Breakouts.
    
    Strategy Logic:
    ---------------
    This strategy focuses on range breakouts as shown in TradingView charts:
    - LONG Signal: Oscillator breaks above +100 (breakout potential)
    - SHORT Signal: Oscillator breaks below -100 (breakout potential)
    - Exhaustion Detection: Oscillator > +150 or < -150 (potential reversal)
    
    Key Concept:
    "When the oscillator line breaks above +100 or below -100, the price is 
    exceeding its normal volatility range, often signaling breakout potential 
    or exhaustion extremes."
    
    Args:
        high: High price series (required if oscillator not provided)
        low: Low price series (required if oscillator not provided)
        close: Close price series (required if oscillator not provided)
        oscillator: Pre-calculated oscillator series (optional, if provided with ma and range_atr)
        ma: Pre-calculated moving average series (optional)
        range_atr: Pre-calculated range ATR series (optional)
        length: Minimum range length for oscillator calculation (default: 50, ignored if oscillator provided)
        mult: Range width multiplier (default: 2.0, ignored if oscillator provided)
        upper_threshold: Upper breakout threshold (default: 100.0)
        lower_threshold: Lower breakout threshold (default: -100.0)
        require_confirmation: If True, require confirmation bars (default: True)
        confirmation_bars: Number of bars to confirm breakout (default: 2)
        detect_exhaustion: If True, detect exhaustion extremes (default: True)
        exhaustion_threshold: Threshold for exhaustion detection (default: 150.0)
    
    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
    """
    # Get oscillator data (either pre-calculated or calculate now)
    oscillator, ma, range_atr = _get_oscillator_data(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )
    
    # Vectorized exhaustion detection
    in_exhaustion_zone = pd.Series(False, index=oscillator.index)
    if detect_exhaustion:
        in_exhaustion_zone = oscillator.abs() >= exhaustion_threshold
    
    # Detect breakouts using shift
    prev_osc = oscillator.shift(1)
    breakout_above = (prev_osc <= upper_threshold) & (oscillator > upper_threshold)
    breakout_below = (prev_osc >= lower_threshold) & (oscillator < lower_threshold)
    
    # Initialize signals
    signals = pd.Series(0, index=oscillator.index, dtype="int8")
    
    if require_confirmation:
        # Count consecutive bars above/below threshold after breakout
        above_threshold = (oscillator > upper_threshold).astype(int)
        below_threshold = (oscillator < lower_threshold).astype(int)
        
        # Use groupby to count consecutive bars
        above_groups = (above_threshold != above_threshold.shift()).cumsum()
        above_counts = above_threshold.groupby(above_groups).cumsum()
        
        below_groups = (below_threshold != below_threshold.shift()).cumsum()
        below_counts = below_threshold.groupby(below_groups).cumsum()
        
        # Vectorized confirmation check: For each breakout point, check if we have enough consecutive bars
        # At breakout point idx, we need above_counts[idx + confirmation_bars - 1] >= confirmation_bars
        # Shift counts forward to check future values at breakout points
        above_counts_shifted = above_counts.shift(-(confirmation_bars - 1))
        below_counts_shifted = below_counts.shift(-(confirmation_bars - 1))
        
        # Check if future count meets confirmation requirement
        # At breakout point, above_counts[idx] should be 1 (start of new group)
        # And above_counts[idx + confirmation_bars - 1] should be >= confirmation_bars
        # Also ensure we're still above threshold at the shifted position
        above_confirmation_mask = (
            (above_counts_shifted >= confirmation_bars) & 
            (above_threshold == 1) &
            (above_counts == 1)  # Ensure breakout point is start of new group
        )
        below_confirmation_mask = (
            (below_counts_shifted >= confirmation_bars) & 
            (below_threshold == 1) &
            (below_counts == 1)  # Ensure breakout point is start of new group
        )
        
        # Apply confirmation: breakout must occur AND future confirmation must be met
        breakout_above_confirmed = breakout_above & above_confirmation_mask
        breakout_below_confirmed = breakout_below & below_confirmation_mask
    else:
        breakout_above_confirmed = breakout_above
        breakout_below_confirmed = breakout_below
    
    # Generate signals
    signals = np.where(breakout_above_confirmed & ~in_exhaustion_zone, 1, signals)
    signals = np.where(breakout_below_confirmed & ~in_exhaustion_zone, -1, signals)
    
    # Maintain signal while oscillator stays in breakout zone (vectorized)
    above_zone = oscillator > upper_threshold
    below_zone = oscillator < lower_threshold
    
    # Optimized forward fill using ffill() with groupby
    signal_series = pd.Series(signals, index=oscillator.index, dtype="int8")
    
    # For LONG signals: forward fill 1 while in above zone
    # Create groups: each group is a continuous period in above zone
    long_groups = (~above_zone).cumsum()
    # Keep only LONG signals (1) and forward fill within each group
    long_signals = signal_series.where(signal_series == 1, np.nan)
    long_signals = long_signals.groupby(long_groups).ffill()
    # Reset to 0 where condition is not met
    long_signals = long_signals.where(above_zone, 0).fillna(0).astype("int8")
    
    # For SHORT signals: forward fill -1 while in below zone
    # Create groups: each group is a continuous period in below zone
    short_groups = (~below_zone).cumsum()
    # Keep only SHORT signals (-1) and forward fill within each group
    short_signals = signal_series.where(signal_series == -1, np.nan)
    short_signals = short_signals.groupby(short_groups).ffill()
    # Reset to 0 where condition is not met
    short_signals = short_signals.where(below_zone, 0).fillna(0).astype("int8")
    
    # Combine signals: LONG if long_signals == 1, SHORT if short_signals == -1
    signals = np.where(long_signals == 1, 1, np.where(short_signals == -1, -1, 0))
    signals = pd.Series(signals, index=oscillator.index, dtype="int8")
    
    # Calculate signal strength
    excess_above = np.clip((oscillator - upper_threshold) / 50.0, 0.0, 1.0)
    excess_below = np.clip((lower_threshold - oscillator) / 50.0, 0.0, 1.0)
    
    signal_strength = pd.Series(0.0, index=oscillator.index, dtype="float64")
    signal_strength = np.where((signals == 1) & breakout_above_confirmed, np.maximum(excess_above, 0.3), signal_strength)
    signal_strength = np.where((signals == -1) & breakout_below_confirmed, np.maximum(excess_below, 0.3), signal_strength)
    signal_strength = np.where((signals == 1) & above_zone & ~breakout_above_confirmed, np.maximum(excess_above, 0.2), signal_strength)
    signal_strength = np.where((signals == -1) & below_zone & ~breakout_below_confirmed, np.maximum(excess_below, 0.2), signal_strength)
    
    # Handle NaN and exhaustion zone
    valid_mask = ~(oscillator.isna() | in_exhaustion_zone)
    signals = pd.Series(signals, index=oscillator.index, dtype="int8").where(valid_mask, 0)
    signal_strength = pd.Series(signal_strength, index=oscillator.index, dtype="float64").where(valid_mask, 0.0)
    
    return signals, signal_strength


def generate_signals_strategy7_divergence(
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
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate trading signals based on Range Oscillator Strategy 7: Divergence Detection.
    
    Strategy Logic:
    ---------------
    This strategy detects divergences between price and oscillator:
    - Bearish Divergence: Price makes higher high, oscillator makes lower high → SHORT signal
    - Bullish Divergence: Price makes lower low, oscillator makes higher low → LONG signal
    
    Divergences often signal potential trend reversals before price action confirms them.
    
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
    
    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
    """
    # Get oscillator data (either pre-calculated or calculate now)
    oscillator, ma, range_atr = _get_oscillator_data(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )
    
    # Need high, low, close for divergence detection (price peaks/troughs)
    if high is None or low is None or close is None:
        raise ValueError("high, low, close are required for divergence detection")
    
    # Initialize signal series
    signals = pd.Series(0, index=close.index, dtype="int8")
    signal_strength = pd.Series(0.0, index=close.index, dtype="float64")
    
    # Vectorized peak/trough detection using rolling windows
    # A peak is a local maximum: value >= max(left_window) and >= max(right_window)
    # A trough is a local minimum: value <= min(left_window) and <= min(right_window)
    
    # Detect price peaks (local maxima in high prices)
    # Compare current value with rolling max of left and right windows
    left_max_high = high.rolling(window=min_swing_bars, min_periods=1).max().shift(1)
    right_max_high = high.rolling(window=min_swing_bars, min_periods=1).max().shift(-min_swing_bars)
    is_price_peak = (high >= left_max_high) & (high >= right_max_high) & ~high.isna()
    
    # Detect price troughs (local minima in low prices)
    left_min_low = low.rolling(window=min_swing_bars, min_periods=1).min().shift(1)
    right_min_low = low.rolling(window=min_swing_bars, min_periods=1).min().shift(-min_swing_bars)
    is_price_trough = (low <= left_min_low) & (low <= right_min_low) & ~low.isna()
    
    # Detect oscillator peaks (local maxima)
    left_max_osc = oscillator.rolling(window=min_swing_bars, min_periods=1).max().shift(1)
    right_max_osc = oscillator.rolling(window=min_swing_bars, min_periods=1).max().shift(-min_swing_bars)
    is_osc_peak = (oscillator >= left_max_osc) & (oscillator >= right_max_osc) & ~oscillator.isna()
    
    # Detect oscillator troughs (local minima)
    left_min_osc = oscillator.rolling(window=min_swing_bars, min_periods=1).min().shift(1)
    right_min_osc = oscillator.rolling(window=min_swing_bars, min_periods=1).min().shift(-min_swing_bars)
    is_osc_trough = (oscillator <= left_min_osc) & (oscillator <= right_min_osc) & ~oscillator.isna()
    
    # Get peak/trough indices and values
    price_peak_indices = close.index[is_price_peak]
    price_trough_indices = close.index[is_price_trough]
    osc_peak_indices = oscillator.index[is_osc_peak]
    osc_trough_indices = oscillator.index[is_osc_trough]
    
    # Create DataFrames with peak/trough data for easier comparison
    price_peaks_df = pd.DataFrame({
        'index': price_peak_indices,
        'value': high.loc[price_peak_indices],
        'position': range(len(price_peak_indices))
    }).set_index('index')
    
    price_troughs_df = pd.DataFrame({
        'index': price_trough_indices,
        'value': low.loc[price_trough_indices],
        'position': range(len(price_trough_indices))
    }).set_index('index')
    
    osc_peaks_df = pd.DataFrame({
        'index': osc_peak_indices,
        'value': oscillator.loc[osc_peak_indices],
        'position': range(len(osc_peak_indices))
    }).set_index('index')
    
    osc_troughs_df = pd.DataFrame({
        'index': osc_trough_indices,
        'value': oscillator.loc[osc_trough_indices],
        'position': range(len(osc_trough_indices))
    }).set_index('index')
    
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
    price_pct_change_peaks = ((price_peaks_series - prev_price_peaks) / prev_price_peaks.replace(0, np.nan)) * 100
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
    price_pct_change_troughs = ((price_troughs_series - prev_price_troughs).abs() / prev_price_troughs.replace(0, np.nan)) * 100
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
    
    # Combine signals: bearish takes precedence if both occur at same time (rare)
    signals = pd.Series(0, index=close.index, dtype="int8")
    signals = pd.Series(np.where(bullish_div_series, 1, signals), index=close.index, dtype="int8")
    signals = pd.Series(np.where(bearish_div_series, -1, signals), index=close.index, dtype="int8")
    
    # Combine strengths
    signal_strength = pd.Series(0.0, index=close.index, dtype="float64")
    signal_strength = pd.Series(np.where(bullish_div_series, bullish_strength_series, signal_strength), index=close.index, dtype="float64")
    signal_strength = pd.Series(np.where(bearish_div_series, bearish_strength_series, signal_strength), index=close.index, dtype="float64")
    
    # Handle NaN values
    valid_mask = ~oscillator.isna()
    signals = signals.where(valid_mask, 0)
    signal_strength = signal_strength.where(valid_mask, 0.0)
    
    return signals, signal_strength


def generate_signals_strategy8_trend_following(
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
    
    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
    """
    # Get oscillator data (either pre-calculated or calculate now)
    oscillator, ma, range_atr = _get_oscillator_data(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )
    
    # Vectorized trend analysis using rolling windows
    # Calculate oscillator trend (slope) using rolling
    osc_trend = oscillator.rolling(window=trend_filter_period, min_periods=int(trend_filter_period * 0.7)).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / trend_filter_period if len(x) >= 2 else np.nan, raw=False
    )
    
    # Count bars above/below threshold using rolling
    above_threshold_count = (oscillator > oscillator_threshold).rolling(
        window=trend_filter_period, min_periods=int(trend_filter_period * 0.7)
    ).sum()
    below_threshold_count = (oscillator < -oscillator_threshold).rolling(
        window=trend_filter_period, min_periods=int(trend_filter_period * 0.7)
    ).sum()
    
    # Calculate consistency ratio
    window_size = pd.Series(trend_filter_period, index=oscillator.index)
    consistency_ratio = pd.concat([above_threshold_count, below_threshold_count], axis=1).max(axis=1) / window_size
    
    # Initialize signals
    signals = pd.Series(0, index=oscillator.index, dtype="int8")
    
    # LONG Signal: Consistent above threshold + upward trend
    long_condition = (oscillator > oscillator_threshold) & (osc_trend > 0)
    if require_consistency:
        long_condition = long_condition & (consistency_ratio >= 0.7)
    signals = pd.Series(np.where(long_condition, 1, signals), index=oscillator.index, dtype="int8")
    
    # SHORT Signal: Consistent below -threshold + downward trend
    short_condition = (oscillator < -oscillator_threshold) & (osc_trend < 0)
    if require_consistency:
        short_condition = short_condition & (consistency_ratio >= 0.7)
    signals = pd.Series(np.where(short_condition, -1, signals), index=oscillator.index, dtype="int8")
    
    # Calculate signal strength
    value_strength = np.clip(oscillator.abs() / 100.0, 0.0, 1.0)
    trend_strength = np.clip(osc_trend.abs() / 10.0, 0.0, 1.0)
    signal_strength = pd.Series(np.where(signals != 0, (value_strength + trend_strength) / 2.0, 0.0), index=oscillator.index, dtype="float64")
    
    # Handle NaN values
    valid_mask = ~(oscillator.isna() | osc_trend.isna())
    signals = signals.where(valid_mask, 0)
    signal_strength = signal_strength.where(valid_mask, 0.0)
    
    return signals, signal_strength


def generate_signals_strategy9_mean_reversion(
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
    
    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
    """
    # Get oscillator data (either pre-calculated or calculate now)
    oscillator, ma, range_atr = _get_oscillator_data(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )
    
    # Vectorized extreme detection
    in_extreme_positive = oscillator > extreme_threshold
    in_extreme_negative = oscillator < -extreme_threshold
    
    # Count consecutive bars in extreme zones using groupby
    extreme_positive_groups = (in_extreme_positive != in_extreme_positive.shift()).cumsum()
    extreme_positive_counts = in_extreme_positive.astype(int).groupby(extreme_positive_groups).cumsum()
    
    extreme_negative_groups = (in_extreme_negative != in_extreme_negative.shift()).cumsum()
    extreme_negative_counts = in_extreme_negative.astype(int).groupby(extreme_negative_groups).cumsum()
    
    # Detect transitions: was in extreme, now crossing back toward zero
    prev_extreme_positive = in_extreme_positive.shift(1).fillna(False)
    prev_extreme_negative = in_extreme_negative.shift(1).fillna(False)
    
    # Transition from extreme positive to near zero
    transition_from_positive = prev_extreme_positive & ~in_extreme_positive & (oscillator.abs() <= zero_cross_threshold)
    transition_from_positive = transition_from_positive & (extreme_positive_counts.shift(1) >= min_extreme_bars)
    
    # Transition from extreme negative to near zero
    transition_from_negative = prev_extreme_negative & ~in_extreme_negative & (oscillator.abs() <= zero_cross_threshold)
    transition_from_negative = transition_from_negative & (extreme_negative_counts.shift(1) >= min_extreme_bars)
    
    # Initialize signals
    signals = pd.Series(0, index=oscillator.index, dtype="int8")
    
    # SHORT Signal: Transition from extreme positive (reversal from extreme bullish)
    signals = pd.Series(np.where(transition_from_positive, -1, signals), index=oscillator.index, dtype="int8")
    
    # LONG Signal: Transition from extreme negative (reversal from extreme bearish)
    signals = pd.Series(np.where(transition_from_negative, 1, signals), index=oscillator.index, dtype="int8")
    
    # Calculate signal strength
    # Get oscillator value at extreme start (use previous value when transitioning)
    prev_osc = oscillator.shift(1)
    extreme_magnitude_positive = np.clip(prev_osc.abs() / 100.0, 0.0, 1.0)
    extreme_magnitude_negative = np.clip(prev_osc.abs() / 100.0, 0.0, 1.0)
    
    zero_proximity = 1.0 - (oscillator.abs() / zero_cross_threshold)
    zero_proximity = np.clip(zero_proximity, 0.0, 1.0)
    
    # Use oscillator.index instead of close.index in case close is None
    signal_strength = pd.Series(0.0, index=oscillator.index, dtype="float64")
    
    # SHORT strength
    short_strength = (extreme_magnitude_positive + zero_proximity) / 2.0
    signal_strength = pd.Series(np.where(transition_from_positive, np.maximum(short_strength, 0.4), signal_strength), index=oscillator.index, dtype="float64")
    
    # LONG strength
    long_strength = (extreme_magnitude_negative + zero_proximity) / 2.0
    signal_strength = pd.Series(np.where(transition_from_negative, np.maximum(long_strength, 0.4), signal_strength), index=oscillator.index, dtype="float64")
    
    # Handle NaN values
    valid_mask = ~oscillator.isna()
    signals = signals.where(valid_mask, 0)
    signal_strength = signal_strength.where(valid_mask, 0.0)
    
    return signals, signal_strength


def get_signal_summary(
    signals: pd.Series,
    signal_strength: pd.Series,
    close: pd.Series,
) -> dict:
    """
    Generate summary statistics for signal strategy.
    
    Args:
        signals: Signal series (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        signal_strength: Signal strength series
        close: Close price series
    
    Returns:
        Dictionary with summary statistics
    """
    if len(signals) == 0:
        return {
            "total_signals": 0,
            "long_signals": 0,
            "short_signals": 0,
            "neutral_signals": 0,
            "avg_signal_strength": 0.0,
            "current_signal": 0,
            "current_strength": 0.0,
        }
    
    long_count = (signals == 1).sum()
    short_count = (signals == -1).sum()
    neutral_count = (signals == 0).sum()
    
    # Get current signal (last non-NaN value) - vectorized
    non_nan_signals = signals.dropna()
    if len(non_nan_signals) > 0:
        current_signal = int(non_nan_signals.iloc[-1])
        last_idx = non_nan_signals.index[-1]
        current_strength = float(signal_strength.loc[last_idx]) if last_idx in signal_strength.index and not pd.isna(signal_strength.loc[last_idx]) else 0.0
    else:
        current_signal = 0
        current_strength = 0.0
    
    # Calculate average strength for non-zero signals
    non_zero_signals = signals[signals != 0]
    avg_strength = 0.0
    if len(non_zero_signals) > 0:
        non_zero_strength = signal_strength[signals != 0]
        avg_strength = non_zero_strength.mean() if len(non_zero_strength) > 0 else 0.0
    
    return {
        "total_signals": len(signals),
        "long_signals": int(long_count),
        "short_signals": int(short_count),
        "neutral_signals": int(neutral_count),
        "avg_signal_strength": float(avg_strength),
        "current_signal": current_signal,
        "current_strength": float(current_strength),
        "long_percentage": float(long_count / len(signals) * 100) if len(signals) > 0 else 0.0,
        "short_percentage": float(short_count / len(signals) * 100) if len(signals) > 0 else 0.0,
    }


__all__ = [
    "generate_signals_strategy1",
    "generate_signals_strategy2_sustained",
    "generate_signals_strategy3_crossover",
    "generate_signals_strategy4_momentum",
    "generate_signals_strategy5_combined",
    "generate_signals_strategy6_breakout",
    "generate_signals_strategy7_divergence",
    "generate_signals_strategy8_trend_following",
    "generate_signals_strategy9_mean_reversion",
    "get_signal_summary",
]

