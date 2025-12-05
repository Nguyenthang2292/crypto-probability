"""
Range Oscillator Strategy 5: Combined (Enhanced).

This module provides an enhanced combined signal generation strategy that combines
multiple methods with advanced features:
- Support for all available strategies (2-9)
- Weighted voting system
- Multiple consensus modes
- Signal strength filtering
- Performance tracking
"""

from typing import Optional, Tuple, Dict, List, Literal
import pandas as pd
import numpy as np
import os

from modules.range_oscillator.core.utils import get_oscillator_data
from modules.range_oscillator.strategies.basic import generate_signals_strategy1
from modules.range_oscillator.strategies.sustained import generate_signals_strategy2_sustained
from modules.range_oscillator.strategies.crossover import generate_signals_strategy3_crossover
from modules.range_oscillator.strategies.momentum import generate_signals_strategy4_momentum
from modules.range_oscillator.strategies.breakout import generate_signals_strategy6_breakout
from modules.range_oscillator.strategies.divergence import generate_signals_strategy7_divergence
from modules.range_oscillator.strategies.trend_following import generate_signals_strategy8_trend_following
from modules.range_oscillator.strategies.mean_reversion import generate_signals_strategy9_mean_reversion
from modules.common.utils import log_debug, log_analysis


# Strategy mapping for easy access
STRATEGY_FUNCTIONS = {
    2: generate_signals_strategy2_sustained,
    3: generate_signals_strategy3_crossover,
    4: generate_signals_strategy4_momentum,
    6: generate_signals_strategy6_breakout,
    7: generate_signals_strategy7_divergence,
    8: generate_signals_strategy8_trend_following,
    9: generate_signals_strategy9_mean_reversion,
}

# Default strategy names for logging
STRATEGY_NAMES = {
    2: "Sustained",
    3: "Crossover",
    4: "Momentum",
    6: "Breakout",
    7: "Divergence",
    8: "Trend Following",
    9: "Mean Reversion",
}


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
    # Strategy selection
    enabled_strategies: Optional[List[int]] = None,
    # Consensus mode
    consensus_mode: Literal["majority", "unanimous", "weighted", "threshold"] = "majority",
    consensus_threshold: float = 0.5,  # For threshold mode: min fraction of strategies that must agree
    # Strategy weights (for weighted mode)
    strategy_weights: Optional[Dict[int, float]] = None,
    # Signal filtering
    min_signal_strength: float = 0.0,  # Minimum signal strength to accept (0.0 to 1.0)
    # Strategy-specific parameters
    # Strategy 2: Sustained
    use_sustained: bool = True,
    min_bars_sustained: int = 3,
    # Strategy 3: Crossover
    use_crossover: bool = True,
    confirmation_bars: int = 2,
    # Strategy 4: Momentum
    use_momentum: bool = True,
    momentum_period: int = 3,
    momentum_threshold: float = 5.0,
    # Strategy 6: Breakout (optional parameters)
    use_breakout: bool = False,
    breakout_upper_threshold: float = 100.0,
    breakout_lower_threshold: float = -100.0,
    breakout_confirmation_bars: int = 2,
    # Strategy 7: Divergence (optional parameters)
    use_divergence: bool = False,
    divergence_lookback_period: int = 30,
    divergence_min_swing_bars: int = 5,
    # Strategy 8: Trend Following (optional parameters)
    use_trend_following: bool = False,
    trend_filter_period: int = 10,
    trend_oscillator_threshold: float = 20.0,
    # Strategy 9: Mean Reversion (optional parameters)
    use_mean_reversion: bool = False,
    mean_reversion_extreme_threshold: float = 80.0,
    mean_reversion_zero_cross_threshold: float = 10.0,
    # Debug
    enable_debug: bool = False,
    return_strategy_stats: bool = False,  # Return statistics about strategy contributions
) -> Tuple[pd.Series, pd.Series, Optional[Dict]]:
    """
    Generate trading signals based on Range Oscillator Strategy 5: Combined (Enhanced).
    
    Strategy Logic:
    ---------------
    This enhanced strategy combines multiple signal generation methods with advanced features:
    
    1. **Strategy Selection**: Choose which strategies to include (2, 3, 4, 6, 7, 8, 9)
    2. **Consensus Modes**:
       - "majority": Signal requires majority vote (default)
       - "unanimous": All enabled strategies must agree
       - "weighted": Signals weighted by strategy_weights
       - "threshold": Requires consensus_threshold fraction of strategies to agree
    
    3. **Signal Filtering**: Only accept signals with strength >= min_signal_strength
    
    4. **Available Strategies**:
       - Strategy 2: Sustained pressure (oscillator staying above/below 0)
       - Strategy 3: Zero line crossover with confirmation
       - Strategy 4: Momentum-based signals
       - Strategy 6: Range breakouts
       - Strategy 7: Divergence detection
       - Strategy 8: Trend following
       - Strategy 9: Mean reversion
    
    Args:
        high: High price series (required if oscillator not provided)
        low: Low price series (required if oscillator not provided)
        close: Close price series (required if oscillator not provided)
        oscillator: Pre-calculated oscillator series (optional)
        ma: Pre-calculated moving average series (optional)
        range_atr: Pre-calculated range ATR series (optional)
        length: Minimum range length for oscillator calculation (default: 50)
        mult: Range width multiplier (default: 2.0)
        
        # Strategy Selection
        enabled_strategies: List of strategy numbers to enable (e.g., [2, 3, 4]).
                          If None, uses use_* flags (default: None)
        
        # Consensus Mode
        consensus_mode: How to combine signals (default: "majority")
        consensus_threshold: For threshold mode, min fraction that must agree (default: 0.5)
        strategy_weights: Dict mapping strategy number to weight (default: None, equal weights)
        
        # Signal Filtering
        min_signal_strength: Minimum signal strength to accept (default: 0.0)
        
        # Strategy 2: Sustained
        use_sustained: Enable sustained pressure signals (default: True)
        min_bars_sustained: Minimum bars for sustained signal (default: 3)
        
        # Strategy 3: Crossover
        use_crossover: Enable crossover signals (default: True)
        confirmation_bars: Bars for crossover confirmation (default: 2)
        
        # Strategy 4: Momentum
        use_momentum: Enable momentum signals (default: True)
        momentum_period: Period for momentum calculation (default: 3)
        momentum_threshold: Threshold for momentum signal (default: 5.0)
        
        # Strategy 6: Breakout
        use_breakout: Enable breakout signals (default: False)
        breakout_upper_threshold: Upper breakout threshold (default: 100.0)
        breakout_lower_threshold: Lower breakout threshold (default: -100.0)
        breakout_confirmation_bars: Bars for breakout confirmation (default: 2)
        
        # Strategy 7: Divergence
        use_divergence: Enable divergence signals (default: False)
        divergence_lookback_period: Period to look back for peaks/troughs (default: 30)
        divergence_min_swing_bars: Minimum bars between peaks/troughs (default: 5)
        
        # Strategy 8: Trend Following
        use_trend_following: Enable trend following signals (default: False)
        trend_filter_period: Period for trend filter (default: 10)
        trend_oscillator_threshold: Minimum oscillator value for signal (default: 20.0)
        
        # Strategy 9: Mean Reversion
        use_mean_reversion: Enable mean reversion signals (default: False)
        mean_reversion_extreme_threshold: Threshold for extreme values (default: 80.0)
        mean_reversion_zero_cross_threshold: Max distance from zero for transition (default: 10.0)
        
        # Debug & Stats
        enable_debug: If True, enable debug logging (default: False)
        return_strategy_stats: If True, return statistics dict (default: False)
    
    Returns:
        Tuple containing:
        - signals: Series with signal values (1 = LONG, -1 = SHORT, 0 = NEUTRAL)
        - signal_strength: Series with signal strength values (0.0 to 1.0)
        - strategy_stats: Dict with strategy contribution statistics, or None if return_strategy_stats=False.
                         Can be safely ignored for backward compatibility.
    """
    # Enable debug logging if requested (check early for validation messages)
    debug_enabled = enable_debug or os.environ.get("RANGE_OSCILLATOR_DEBUG", "false").lower() == "true"
    
    # Determine enabled strategies
    if enabled_strategies is None:
        enabled_strategies = []
        if use_sustained:
            enabled_strategies.append(2)
        if use_crossover:
            enabled_strategies.append(3)
        if use_momentum:
            enabled_strategies.append(4)
        if use_breakout:
            enabled_strategies.append(6)
        if use_divergence:
            enabled_strategies.append(7)
        if use_trend_following:
            enabled_strategies.append(8)
        if use_mean_reversion:
            enabled_strategies.append(9)
    
    # Validate enabled strategies
    if not enabled_strategies:
        # Fallback to basic strategy if no methods enabled
        if debug_enabled:
            log_debug("[Strategy5] No strategies enabled, falling back to basic strategy")
        signals, signal_strength = generate_signals_strategy1(
            high=high, low=low, close=close,
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            length=length, mult=mult
        )
        if return_strategy_stats:
            return (signals, signal_strength, {})
        else:
            return (signals, signal_strength)
    
    # Validate strategy numbers
    invalid_strategies = [s for s in enabled_strategies if s not in STRATEGY_FUNCTIONS]
    if invalid_strategies:
        raise ValueError(f"Invalid strategy numbers: {invalid_strategies}. Valid: {list(STRATEGY_FUNCTIONS.keys())}")
    
    # Validate consensus parameters
    if consensus_mode == "threshold":
        if not (0.0 < consensus_threshold <= 1.0):
            raise ValueError(f"consensus_threshold must be in (0.0, 1.0], got {consensus_threshold}")
    
    if consensus_mode == "weighted" and strategy_weights is None:
        # Default to equal weights
        strategy_weights = {s: 1.0 for s in enabled_strategies}
    
    if strategy_weights is not None:
        # Validate weights
        for strategy_num in enabled_strategies:
            if strategy_num not in strategy_weights:
                strategy_weights[strategy_num] = 1.0  # Default weight
        # Normalize weights
        total_weight = sum(strategy_weights.get(s, 1.0) for s in enabled_strategies)
        if total_weight > 0:
            strategy_weights = {s: strategy_weights.get(s, 1.0) / total_weight for s in enabled_strategies}
    
    # Validate signal strength threshold
    if not (0.0 <= min_signal_strength <= 1.0):
        raise ValueError(f"min_signal_strength must be in [0.0, 1.0], got {min_signal_strength}")
    
    if debug_enabled:
        log_analysis(f"[Strategy5] Starting enhanced combined signal generation")
        log_debug(f"[Strategy5] Enabled strategies: {enabled_strategies}")
        log_debug(f"[Strategy5] Consensus mode: {consensus_mode}")
        if consensus_mode == "threshold":
            log_debug(f"[Strategy5] Consensus threshold: {consensus_threshold}")
        if strategy_weights:
            log_debug(f"[Strategy5] Strategy weights: {strategy_weights}")
        log_debug(f"[Strategy5] Min signal strength: {min_signal_strength}")
    
    # Calculate Range Oscillator ONCE (or use pre-calculated values)
    oscillator, ma, range_atr = get_oscillator_data(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )
    
    if debug_enabled:
        log_debug(f"[Strategy5] Data shape: oscillator={len(oscillator)}")
    
    index = oscillator.index
    
    # Get signals from individual strategies
    signal_votes = []
    strength_votes = []
    strategy_stats = {} if return_strategy_stats else None
    
    # Strategy 2: Sustained
    if 2 in enabled_strategies:
        try:
            sig_sustained, str_sustained = generate_signals_strategy2_sustained(
                oscillator=oscillator, ma=ma, range_atr=range_atr,
                min_bars_above_zero=min_bars_sustained,
                min_bars_below_zero=min_bars_sustained,
                enable_debug=False,
            )
            signal_votes.append(sig_sustained)
            strength_votes.append(str_sustained)
            if return_strategy_stats:
                strategy_stats[2] = {
                    "name": STRATEGY_NAMES[2],
                    "long_count": int((sig_sustained == 1).sum()),
                    "short_count": int((sig_sustained == -1).sum()),
                }
            if debug_enabled:
                log_debug(f"[Strategy5] {STRATEGY_NAMES[2]}: LONG={int((sig_sustained == 1).sum())}, "
                         f"SHORT={int((sig_sustained == -1).sum())}")
        except Exception as e:
            if debug_enabled:
                log_debug(f"[Strategy5] {STRATEGY_NAMES[2]} failed: {e}")
    
    # Strategy 3: Crossover
    if 3 in enabled_strategies:
        try:
            sig_cross, str_cross = generate_signals_strategy3_crossover(
                oscillator=oscillator, ma=ma, range_atr=range_atr,
                confirmation_bars=confirmation_bars,
                enable_debug=False,
            )
            signal_votes.append(sig_cross)
            strength_votes.append(str_cross)
            if return_strategy_stats:
                strategy_stats[3] = {
                    "name": STRATEGY_NAMES[3],
                    "long_count": int((sig_cross == 1).sum()),
                    "short_count": int((sig_cross == -1).sum()),
                }
            if debug_enabled:
                log_debug(f"[Strategy5] {STRATEGY_NAMES[3]}: LONG={int((sig_cross == 1).sum())}, "
                         f"SHORT={int((sig_cross == -1).sum())}")
        except Exception as e:
            if debug_enabled:
                log_debug(f"[Strategy5] {STRATEGY_NAMES[3]} failed: {e}")
    
    # Strategy 4: Momentum
    if 4 in enabled_strategies:
        try:
            # Validate momentum_period
            if len(oscillator) > 0 and momentum_period >= len(oscillator):
                if debug_enabled:
                    log_debug(f"[Strategy5] {STRATEGY_NAMES[4]} skipped: momentum_period too large")
            else:
                sig_mom, str_mom = generate_signals_strategy4_momentum(
                    oscillator=oscillator, ma=ma, range_atr=range_atr,
                    momentum_period=momentum_period,
                    momentum_threshold=momentum_threshold,
                    enable_debug=False,
                )
                signal_votes.append(sig_mom)
                strength_votes.append(str_mom)
                if return_strategy_stats:
                    strategy_stats[4] = {
                        "name": STRATEGY_NAMES[4],
                        "long_count": int((sig_mom == 1).sum()),
                        "short_count": int((sig_mom == -1).sum()),
                    }
                if debug_enabled:
                    log_debug(f"[Strategy5] {STRATEGY_NAMES[4]}: LONG={int((sig_mom == 1).sum())}, "
                             f"SHORT={int((sig_mom == -1).sum())}")
        except Exception as e:
            if debug_enabled:
                log_debug(f"[Strategy5] {STRATEGY_NAMES[4]} failed: {e}")
    
    # Strategy 6: Breakout
    if 6 in enabled_strategies:
        try:
            sig_breakout, str_breakout = generate_signals_strategy6_breakout(
                high=high, low=low, close=close,
                oscillator=oscillator, ma=ma, range_atr=range_atr,
                length=length, mult=mult,
                upper_threshold=breakout_upper_threshold,
                lower_threshold=breakout_lower_threshold,
                confirmation_bars=breakout_confirmation_bars,
                enable_debug=False,
            )
            signal_votes.append(sig_breakout)
            strength_votes.append(str_breakout)
            if return_strategy_stats:
                strategy_stats[6] = {
                    "name": STRATEGY_NAMES[6],
                    "long_count": int((sig_breakout == 1).sum()),
                    "short_count": int((sig_breakout == -1).sum()),
                }
            if debug_enabled:
                log_debug(f"[Strategy5] {STRATEGY_NAMES[6]}: LONG={int((sig_breakout == 1).sum())}, "
                         f"SHORT={int((sig_breakout == -1).sum())}")
        except Exception as e:
            if debug_enabled:
                log_debug(f"[Strategy5] {STRATEGY_NAMES[6]} failed: {e}")
    
    # Strategy 7: Divergence
    if 7 in enabled_strategies:
        try:
            if high is None or low is None or close is None:
                if debug_enabled:
                    log_debug(f"[Strategy5] {STRATEGY_NAMES[7]} skipped: high/low/close required")
            else:
                sig_div, str_div = generate_signals_strategy7_divergence(
                    high=high, low=low, close=close,
                    oscillator=oscillator, ma=ma, range_atr=range_atr,
                    length=length, mult=mult,
                    lookback_period=divergence_lookback_period,
                    min_swing_bars=divergence_min_swing_bars,
                    enable_debug=False,
                )
                signal_votes.append(sig_div)
                strength_votes.append(str_div)
                if return_strategy_stats:
                    strategy_stats[7] = {
                        "name": STRATEGY_NAMES[7],
                        "long_count": int((sig_div == 1).sum()),
                        "short_count": int((sig_div == -1).sum()),
                    }
                if debug_enabled:
                    log_debug(f"[Strategy5] {STRATEGY_NAMES[7]}: LONG={int((sig_div == 1).sum())}, "
                             f"SHORT={int((sig_div == -1).sum())}")
        except Exception as e:
            if debug_enabled:
                log_debug(f"[Strategy5] {STRATEGY_NAMES[7]} failed: {e}")
    
    # Strategy 8: Trend Following
    if 8 in enabled_strategies:
        try:
            sig_trend, str_trend = generate_signals_strategy8_trend_following(
                oscillator=oscillator, ma=ma, range_atr=range_atr,
                length=length, mult=mult,
                trend_filter_period=trend_filter_period,
                oscillator_threshold=trend_oscillator_threshold,
                enable_debug=False,
            )
            signal_votes.append(sig_trend)
            strength_votes.append(str_trend)
            if return_strategy_stats:
                strategy_stats[8] = {
                    "name": STRATEGY_NAMES[8],
                    "long_count": int((sig_trend == 1).sum()),
                    "short_count": int((sig_trend == -1).sum()),
                }
            if debug_enabled:
                log_debug(f"[Strategy5] {STRATEGY_NAMES[8]}: LONG={int((sig_trend == 1).sum())}, "
                         f"SHORT={int((sig_trend == -1).sum())}")
        except Exception as e:
            if debug_enabled:
                log_debug(f"[Strategy5] {STRATEGY_NAMES[8]} failed: {e}")
    
    # Strategy 9: Mean Reversion
    if 9 in enabled_strategies:
        try:
            sig_mean, str_mean = generate_signals_strategy9_mean_reversion(
                oscillator=oscillator, ma=ma, range_atr=range_atr,
                length=length, mult=mult,
                extreme_threshold=mean_reversion_extreme_threshold,
                zero_cross_threshold=mean_reversion_zero_cross_threshold,
                enable_debug=False,
            )
            signal_votes.append(sig_mean)
            strength_votes.append(str_mean)
            if return_strategy_stats:
                strategy_stats[9] = {
                    "name": STRATEGY_NAMES[9],
                    "long_count": int((sig_mean == 1).sum()),
                    "short_count": int((sig_mean == -1).sum()),
                }
            if debug_enabled:
                log_debug(f"[Strategy5] {STRATEGY_NAMES[9]}: LONG={int((sig_mean == 1).sum())}, "
                         f"SHORT={int((sig_mean == -1).sum())}")
        except Exception as e:
            if debug_enabled:
                log_debug(f"[Strategy5] {STRATEGY_NAMES[9]} failed: {e}")
    
    if not signal_votes:
        # Fallback to basic strategy if all strategies failed
        if debug_enabled:
            log_debug("[Strategy5] All strategies failed, falling back to basic strategy")
        signals, signal_strength = generate_signals_strategy1(
            high=high, low=low, close=close,
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            length=length, mult=mult
        )
        return (signals, signal_strength, strategy_stats)
    
    # Combine signals based on consensus mode
    signals_array = np.stack([sig.values for sig in signal_votes], axis=0)  # Shape: (n_strategies, n_bars)
    strengths_array = np.stack([str_vote.values for str_vote in strength_votes], axis=0)  # Shape: (n_strategies, n_bars)
    
    n_strategies = len(signal_votes)
    
    if consensus_mode == "unanimous":
        # All strategies must agree
        long_unanimous = np.all(signals_array == 1, axis=0)
        short_unanimous = np.all(signals_array == -1, axis=0)
        signals = np.where(long_unanimous, 1, np.where(short_unanimous, -1, 0)).astype(np.int8)
        
    elif consensus_mode == "threshold":
        # Require at least consensus_threshold fraction of strategies to agree
        min_agreement = int(np.ceil(n_strategies * consensus_threshold))
        long_votes = np.sum(signals_array == 1, axis=0)
        short_votes = np.sum(signals_array == -1, axis=0)
        signals = np.where(
            long_votes >= min_agreement,
            1,
            np.where(short_votes >= min_agreement, -1, 0)
        ).astype(np.int8)
        
    elif consensus_mode == "weighted":
        # Weighted voting based on strategy_weights
        # Map enabled strategies to their indices in signal_votes
        strategy_to_index = {enabled_strategies[i]: i for i in range(len(enabled_strategies))}
        
        # Calculate weighted votes
        long_weighted = np.zeros(len(index), dtype=np.float64)
        short_weighted = np.zeros(len(index), dtype=np.float64)
        
        for strategy_num, weight in strategy_weights.items():
            if strategy_num in strategy_to_index:
                idx = strategy_to_index[strategy_num]
                long_weighted += (signals_array[idx] == 1).astype(np.float64) * weight
                short_weighted += (signals_array[idx] == -1).astype(np.float64) * weight
        
        signals = np.where(
            long_weighted > short_weighted,
            1,
            np.where(short_weighted > long_weighted, -1, 0)
        ).astype(np.int8)
        
    else:  # "majority" (default)
        # Majority vote
        long_votes = np.sum(signals_array == 1, axis=0).astype(np.int8)
        short_votes = np.sum(signals_array == -1, axis=0).astype(np.int8)
        signals = np.where(
            (long_votes > short_votes) & (long_votes > 0),
            1,
            np.where((short_votes > long_votes) & (short_votes > 0), -1, 0)
        ).astype(np.int8)
    
    if debug_enabled:
        total_long_votes = int(np.sum(signals_array == 1))
        total_short_votes = int(np.sum(signals_array == -1))
        final_long = int((signals == 1).sum())
        final_short = int((signals == -1).sum())
        log_debug(f"[Strategy5] Vote summary: total_long_votes={total_long_votes}, "
                 f"total_short_votes={total_short_votes}")
        log_debug(f"[Strategy5] Final signals (before filtering): LONG={final_long}, SHORT={final_short}")
    
    # Calculate average strength
    long_mask = signals == 1
    short_mask = signals == -1
    
    # For LONG signals: average strength from strategies that voted LONG
    long_signal_mask = (signals_array == 1) & long_mask[np.newaxis, :]
    long_strength_sum = np.sum(strengths_array * long_signal_mask, axis=0)
    long_strength_count = np.sum(long_signal_mask, axis=0).astype(np.float64)
    long_strength_avg = np.divide(
        long_strength_sum, long_strength_count,
        out=np.zeros_like(long_strength_sum),
        where=long_strength_count > 0
    )
    
    # For SHORT signals: average strength from strategies that voted SHORT
    short_signal_mask = (signals_array == -1) & short_mask[np.newaxis, :]
    short_strength_sum = np.sum(strengths_array * short_signal_mask, axis=0)
    short_strength_count = np.sum(short_signal_mask, axis=0).astype(np.float64)
    short_strength_avg = np.divide(
        short_strength_sum, short_strength_count,
        out=np.zeros_like(short_strength_sum),
        where=short_strength_count > 0
    )
    
    # Combine strengths
    signal_strength = np.where(
        long_mask, long_strength_avg,
        np.where(short_mask, short_strength_avg, 0.0)
    )
    
    # Apply signal strength filtering
    if min_signal_strength > 0.0:
        strength_mask = signal_strength >= min_signal_strength
        signals = np.where(strength_mask, signals, 0).astype(np.int8)
        if debug_enabled:
            filtered_out = int((~strength_mask & (signals != 0)).sum())
            log_debug(f"[Strategy5] Filtered out {filtered_out} signals below strength threshold")
    
    # Convert back to Series
    signals = pd.Series(signals, index=index, dtype="int8")
    signal_strength = pd.Series(signal_strength, index=index, dtype="float64")
    
    # Handle NaN values
    valid_mask = ~oscillator.isna()
    signals = signals.where(valid_mask, 0)
    signal_strength = signal_strength.where(valid_mask, 0.0)
    
    if debug_enabled:
        final_long = int((signals == 1).sum())
        final_short = int((signals == -1).sum())
        final_neutral = int((signals == 0).sum())
        avg_strength = float(signal_strength.mean())
        log_analysis(f"[Strategy5] Final combined signals: LONG={final_long}, SHORT={final_short}, "
                    f"NEUTRAL={final_neutral}, avg_strength={avg_strength:.3f}")
    
        # Return based on return_strategy_stats flag for backward compatibility
        if return_strategy_stats:
            return (signals, signal_strength, strategy_stats)
        else:
            return (signals, signal_strength)


__all__ = [
    "generate_signals_strategy5_combined",
    "STRATEGY_FUNCTIONS",
    "STRATEGY_NAMES",
]
