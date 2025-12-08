"""
Range Oscillator Strategy 5: Combined (Enhanced).

This module provides an enhanced combined signal generation strategy that combines
multiple methods with advanced features:
- Support for all available strategies (2-9)
- Weighted voting system
- Multiple consensus modes
- Signal strength filtering
- Performance tracking

Refactored to improve stability, safety, and maintainability.
"""

from typing import Optional, Tuple, Dict, List, Literal, Any, Union
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass, field

from modules.range_oscillator.core.utils import get_oscillator_data
from modules.range_oscillator.strategies.basic import generate_signals_strategy1
from modules.range_oscillator.strategies.sustained import generate_signals_strategy2_sustained
from modules.range_oscillator.strategies.crossover import generate_signals_strategy3_crossover
from modules.range_oscillator.strategies.momentum import generate_signals_strategy4_momentum
from modules.range_oscillator.strategies.breakout import generate_signals_strategy6_breakout
from modules.range_oscillator.strategies.divergence import generate_signals_strategy7_divergence
from modules.range_oscillator.strategies.trend_following import generate_signals_strategy8_trend_following
from modules.range_oscillator.strategies.mean_reversion import generate_signals_strategy9_mean_reversion
from modules.common.utils import log_debug, log_analysis, log_warn


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

# Strategy names for logging
STRATEGY_NAMES = {
    2: "Sustained",
    3: "Crossover",
    4: "Momentum",
    6: "Breakout",
    7: "Divergence",
    8: "Trend Following",
    9: "Mean Reversion",
}

# Strategy categories for dynamic selection
TRENDING_STRATEGIES = [3, 4, 6, 8]
RANGE_BOUND_STRATEGIES = [2, 7, 9]
VOLATILE_STRATEGIES = [6, 7]
STABLE_STRATEGIES = [2, 3, 9]

# Constants for performance scoring weights
AGREEMENT_WEIGHT = 0.6
STRENGTH_WEIGHT = 0.4

# Normalization constant for oscillator extreme calculation
OSCILLATOR_NORMALIZATION = 100.0

# Valid strategy IDs
VALID_STRATEGY_IDS = {2, 3, 4, 6, 7, 8, 9}


# -----------------------------------------------------------------------------
# Configuration Objects (Issue 3 Fix)
# -----------------------------------------------------------------------------

@dataclass
class DynamicSelectionConfig:
    enabled: bool = False
    lookback: int = 20
    volatility_threshold: float = 0.6
    trend_threshold: float = 0.5

@dataclass
class ConsensusConfig:
    mode: Literal["threshold", "weighted"] = "threshold"
    threshold: float = 0.5   # For threshold mode
    
    # Adaptive weights (Weighted mode)
    adaptive_weights: bool = False
    performance_window: int = 10
    
    # Weighted voting rules (Issue 4 Fix)
    weighted_min_diff: float = 0.1    # Difference between Long and Short weights must be > 0.1
    weighted_min_total: float = 0.5   # Total weight of the winning side must be > 0.5

@dataclass
class StrategySpecificConfig:
    # Strategy 2: Sustained
    use_sustained: bool = True
    min_bars_sustained: int = 3
    
    # Strategy 3: Crossover
    use_crossover: bool = True
    confirmation_bars: int = 2
    
    # Strategy 4: Momentum
    use_momentum: bool = True
    momentum_period: int = 3
    momentum_threshold: float = 5.0
    
    # Strategy 6: Breakout
    use_breakout: bool = False
    breakout_upper_threshold: float = 100.0
    breakout_lower_threshold: float = -100.0
    breakout_confirmation_bars: int = 2
    
    # Strategy 7: Divergence
    use_divergence: bool = False
    divergence_lookback_period: int = 30
    divergence_min_swing_bars: int = 5
    
    # Strategy 8: Trend Following
    use_trend_following: bool = False
    trend_filter_period: int = 10
    trend_oscillator_threshold: float = 20.0
    
    # Strategy 9: Mean Reversion
    use_mean_reversion: bool = False
    mean_reversion_extreme_threshold: float = 80.0
    mean_reversion_zero_cross_threshold: float = 10.0

@dataclass
class Strategy5Config:
    enabled_strategies: List[int] = field(default_factory=list)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    dynamic: DynamicSelectionConfig = field(default_factory=DynamicSelectionConfig)
    params: StrategySpecificConfig = field(default_factory=StrategySpecificConfig)
    
    # General
    min_signal_strength: float = 0.0
    strategy_weights: Optional[Dict[int, float]] = None
    
    # Outputs
    return_confidence_score: bool = False
    return_strategy_stats: bool = False
    enable_debug: bool = False


# -----------------------------------------------------------------------------
# Validation Functions
# -----------------------------------------------------------------------------

def _validate_config(config: Strategy5Config) -> None:
    """
    Validate configuration parameters.
    
    Raises:
        ValueError: If any configuration parameter is invalid.
    """
    # Validate consensus mode
    if config.consensus.mode not in ("threshold", "weighted"):
        raise ValueError(
            f"Invalid consensus mode: {config.consensus.mode}. "
            f"Must be 'threshold' or 'weighted'"
        )
    
    # Validate consensus threshold
    if not (0.0 <= config.consensus.threshold <= 1.0):
        raise ValueError(
            f"consensus_threshold must be in [0, 1], got {config.consensus.threshold}"
        )
    
    # Validate enabled strategies
    if config.enabled_strategies:
        invalid = set(config.enabled_strategies) - VALID_STRATEGY_IDS
        if invalid:
            raise ValueError(
                f"Invalid strategy IDs: {invalid}. "
                f"Valid IDs are: {sorted(VALID_STRATEGY_IDS)}"
            )
    
    # Validate weighted voting thresholds
    if config.consensus.mode == "weighted":
        if config.consensus.weighted_min_total < 0:
            raise ValueError(
                f"weighted_min_total must be >= 0, got {config.consensus.weighted_min_total}"
            )
        if config.consensus.weighted_min_diff < 0:
            raise ValueError(
                f"weighted_min_diff must be >= 0, got {config.consensus.weighted_min_diff}"
            )
    
    # Validate min_signal_strength
    if config.min_signal_strength < 0:
        raise ValueError(
            f"min_signal_strength must be >= 0, got {config.min_signal_strength}"
        )
    
    # Validate dynamic selection thresholds
    if config.dynamic.enabled:
        if not (0.0 <= config.dynamic.volatility_threshold <= 1.0):
            raise ValueError(
                f"volatility_threshold must be in [0, 1], got {config.dynamic.volatility_threshold}"
            )
        if not (0.0 <= config.dynamic.trend_threshold <= 1.0):
            raise ValueError(
                f"trend_threshold must be in [0, 1], got {config.dynamic.trend_threshold}"
            )
        if config.dynamic.lookback < 1:
            raise ValueError(
                f"lookback must be >= 1, got {config.dynamic.lookback}"
            )


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _analyze_market_conditions(
    oscillator: pd.Series,
    range_atr: pd.Series,
    close: Optional[pd.Series] = None,
    lookback_period: int = 20,
) -> Dict[str, float]:
    """Analyze market conditions to determine volatility, trend strength, and market regime."""
    if len(oscillator) < lookback_period:
        lookback_period = len(oscillator)
    
    if lookback_period == 0:
        return {
            "volatility": 0.5,
            "trend_strength": 0.5,
            "range_bound_score": 0.5,
            "oscillator_extreme": 0.5,
        }
    
    # Get recent data
    recent_osc = oscillator.iloc[-lookback_period:]
    recent_atr = range_atr.iloc[-lookback_period:]
    
    # Calculate volatility: based on ATR variation
    atr_std = recent_atr.std()
    atr_mean = recent_atr.mean()
    volatility = min(1.0, (atr_std / (atr_mean + 1e-10)) * 2.0) if atr_mean > 0 else 0.5
    
    # Calculate trend strength: based on oscillator trend
    osc_trend = recent_osc.diff().abs().mean()
    osc_range = recent_osc.max() - recent_osc.min()
    trend_strength = min(1.0, (osc_trend / (osc_range + 1e-10)) * 2.0) if osc_range > 0 else 0.5
    
    # Calculate range-bound score: inverse of trend strength
    range_bound_score = 1.0 - trend_strength
    
    # Calculate oscillator extreme: how far from zero
    osc_abs = recent_osc.abs()
    oscillator_extreme = min(1.0, osc_abs.mean() / OSCILLATOR_NORMALIZATION)  # Normalize to 0-1
    
    return {
        "volatility": float(volatility),
        "trend_strength": float(trend_strength),
        "range_bound_score": float(range_bound_score),
        "oscillator_extreme": float(oscillator_extreme),
    }


def _select_strategies_dynamically(
    market_conditions: Dict[str, float],
    available_strategies: List[int],
    config: DynamicSelectionConfig,
) -> List[int]:
    """Dynamically select strategies based on market conditions."""
    selected = []
    volatility = market_conditions["volatility"]
    trend_strength = market_conditions["trend_strength"]
    range_bound = market_conditions["range_bound_score"]
    
    # High volatility: prefer breakout and divergence strategies
    if volatility >= config.volatility_threshold:
        for strategy in VOLATILE_STRATEGIES:
            if strategy in available_strategies:
                selected.append(strategy)
    
    # Trending market: prefer trending strategies
    if trend_strength >= config.trend_threshold:
        for strategy in TRENDING_STRATEGIES:
            if strategy in available_strategies and strategy not in selected:
                selected.append(strategy)
    
    # Range-bound market: prefer range-bound strategies
    if range_bound >= 0.5:
        for strategy in RANGE_BOUND_STRATEGIES:
            if strategy in available_strategies and strategy not in selected:
                selected.append(strategy)
    
    # If no strategies selected, use all available
    if not selected:
        selected = available_strategies.copy()
    
    return sorted(selected)


def _calculate_adaptive_weights(
    signals_df: pd.DataFrame,
    strengths_df: pd.DataFrame,
    enabled_strategies: List[int],
    performance_window: int = 10,
    min_performance_samples: int = 5,
) -> Dict[int, float]:
    """
    Calculate adaptive weights based on strategy performance.
    
    Note: Updated to use DataFrame inputs for better alignment.
    """
    n_bars = len(signals_df)
    
    if n_bars < min_performance_samples:
        return {s: 1.0 for s in enabled_strategies}
    
    # Use recent data
    window = min(performance_window, n_bars)
    recent_signals = signals_df.iloc[-window:]
    recent_strengths = strengths_df.iloc[-window:]
    
    strategy_scores = {}
    
    # Consensus signal (majority vote)
    row_sums = recent_signals.sum(axis=1)
    consensus = np.sign(row_sums)
    
    for strategy_num in enabled_strategies:
        col_name = str(strategy_num)
        if col_name not in recent_signals.columns:
            continue
            
        strategy_signals = recent_signals[col_name]
        strategy_strengths = recent_strengths[col_name]
        
        # Agreement score
        matches = (strategy_signals == consensus)
        agreement = matches.mean() if len(consensus) > 0 else 0.5
        
        # Strength score
        active_mask = np.abs(strategy_signals) > 0
        if active_mask.any():
            avg_strength = strategy_strengths[active_mask].mean()
        else:
            avg_strength = 0.0
        
        performance_score = (agreement * AGREEMENT_WEIGHT) + (avg_strength * STRENGTH_WEIGHT)
        strategy_scores[strategy_num] = max(0.1, performance_score)
    
    # Normalize
    total_score = sum(strategy_scores.values())
    if total_score > 0:
        strategy_scores = {k: v / total_score * len(enabled_strategies) for k, v in strategy_scores.items()}
    
    return strategy_scores


def _calculate_confidence_score(
    signals_array: np.ndarray,  # Shape: (n_strategies, n_bars)
    strengths_array: np.ndarray,  # Shape: (n_strategies, n_bars)
    n_strategies: int,
    consensus_mode: Literal["threshold", "weighted"],
    consensus_threshold: float = 0.5,
) -> np.ndarray:  # Shape: (n_bars,)
    """
    Calculate confidence score based on strategy agreement and signal strength.
    
    Args:
        signals_array: Strategy signals array with shape (n_strategies, n_bars)
        strengths_array: Strategy strengths array with shape (n_strategies, n_bars)
        n_strategies: Number of strategies
        consensus_mode: Consensus mode ("threshold" or "weighted")
        consensus_threshold: Threshold for threshold mode
        
    Returns:
        Confidence scores array with shape (n_bars,)
    """
    
    n_bars = signals_array.shape[1]
    confidence = np.zeros(n_bars, dtype=np.float64)
    
    for i in range(n_bars):
        bar_signals = signals_array[:, i]
        bar_strengths = strengths_array[:, i]
        
        long_votes = np.sum(bar_signals == 1)
        short_votes = np.sum(bar_signals == -1)
        total_votes = long_votes + short_votes
        
        if total_votes == 0:
            confidence[i] = 0.0
            continue
        
        if consensus_mode == "threshold":
            min_agreement = int(np.ceil(n_strategies * consensus_threshold))
            if long_votes >= min_agreement:
                agreement_level = long_votes / n_strategies
            elif short_votes >= min_agreement:
                agreement_level = short_votes / n_strategies
            else:
                agreement_level = max(long_votes, short_votes) / n_strategies
        else:  # weighted
            agreement_level = total_votes / n_strategies
        
        agreeing_mask = (bar_signals != 0)
        if np.any(agreeing_mask):
            avg_strength = np.mean(bar_strengths[agreeing_mask])
        else:
            avg_strength = 0.0
        
        confidence[i] = (agreement_level * AGREEMENT_WEIGHT) + (avg_strength * STRENGTH_WEIGHT)
    
    return confidence


# -----------------------------------------------------------------------------
# Main Strategy Function
# -----------------------------------------------------------------------------

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
    
    # Configuration Object (Preferred)
    config: Optional[Strategy5Config] = None,
    
    # Legacy arguments (for backward compatibility, will override config if provided)
    enabled_strategies: Optional[List[int]] = None,
    enable_dynamic_selection: Optional[bool] = None,
    dynamic_selection_lookback: int = 20,
    dynamic_volatility_threshold: float = 0.6,
    dynamic_trend_threshold: float = 0.5,
    consensus_mode: str = "threshold",
    consensus_threshold: float = 0.5,
    strategy_weights: Optional[Dict[int, float]] = None,
    enable_adaptive_weights: bool = False,
    adaptive_performance_window: int = 10,
    min_signal_strength: float = 0.0,
    return_confidence_score: bool = False,
    return_strategy_stats: bool = False,
    enable_debug: bool = False,
    
    # Strategy specific params (Legacy)
    use_sustained: bool = True,
    min_bars_sustained: int = 3,
    use_crossover: bool = True,
    confirmation_bars: int = 2,
    use_momentum: bool = True,
    momentum_period: int = 3,
    momentum_threshold: float = 5.0,
    use_breakout: bool = False,
    breakout_upper_threshold: float = 100.0,
    breakout_lower_threshold: float = -100.0,
    breakout_confirmation_bars: int = 2,
    use_divergence: bool = False,
    divergence_lookback_period: int = 30,
    divergence_min_swing_bars: int = 5,
    use_trend_following: bool = False,
    trend_filter_period: int = 10,
    trend_oscillator_threshold: float = 20.0,
    use_mean_reversion: bool = False,
    mean_reversion_extreme_threshold: float = 80.0,
    mean_reversion_zero_cross_threshold: float = 10.0,
    
    # Issue 4 specific args (if passed manually)
    weighted_min_diff: float = 0.1,
    weighted_min_total: float = 0.5,
) -> Tuple[pd.Series, pd.Series, Optional[Dict], Optional[pd.Series]]:
    """
    Generate trading signals based on Range Oscillator Strategy 5: Combined (Enhanced).
    
    Now supports `Strategy5Config` object to group parameters.
    
    Returns:
        Tuple containing:
        - signal_series (pd.Series): Trading signals (-1, 0, 1)
        - strength_series (pd.Series): Signal strengths
        - strategy_stats (Dict, optional): Strategy statistics if return_strategy_stats=True
        - confidence_series (pd.Series, optional): Confidence scores if return_confidence_score=True
    """
    
    # -------------------------------------------------------------------------
    # 1. Config Object Conversion (Sửa lỗi số 3)
    # -------------------------------------------------------------------------
    if config is None:
        # Construct config from arguments if not provided
        config = Strategy5Config()
        
        # Populate ConsensusConfig
        config.consensus.mode = consensus_mode
        config.consensus.threshold = consensus_threshold
        config.consensus.adaptive_weights = enable_adaptive_weights
        config.consensus.performance_window = adaptive_performance_window
        config.consensus.weighted_min_diff = weighted_min_diff
        config.consensus.weighted_min_total = weighted_min_total
        
        # Populate DynamicSelectionConfig
        config.dynamic.enabled = (enable_dynamic_selection if enable_dynamic_selection is not None else False)
        config.dynamic.lookback = dynamic_selection_lookback
        config.dynamic.volatility_threshold = dynamic_volatility_threshold
        config.dynamic.trend_threshold = dynamic_trend_threshold
        
        # Populate StrategySpecificConfig
        config.params.use_sustained = use_sustained
        config.params.min_bars_sustained = min_bars_sustained
        config.params.use_crossover = use_crossover
        config.params.confirmation_bars = confirmation_bars
        config.params.use_momentum = use_momentum
        config.params.momentum_period = momentum_period
        config.params.momentum_threshold = momentum_threshold
        config.params.use_breakout = use_breakout
        config.params.breakout_upper_threshold = breakout_upper_threshold
        config.params.breakout_lower_threshold = breakout_lower_threshold
        config.params.breakout_confirmation_bars = breakout_confirmation_bars
        config.params.use_divergence = use_divergence
        config.params.divergence_lookback_period = divergence_lookback_period
        config.params.divergence_min_swing_bars = divergence_min_swing_bars
        config.params.use_trend_following = use_trend_following
        config.params.trend_filter_period = trend_filter_period
        config.params.trend_oscillator_threshold = trend_oscillator_threshold
        config.params.use_mean_reversion = use_mean_reversion
        config.params.mean_reversion_extreme_threshold = mean_reversion_extreme_threshold
        config.params.mean_reversion_zero_cross_threshold = mean_reversion_zero_cross_threshold
        
        # Populate General
        config.min_signal_strength = min_signal_strength
        config.return_confidence_score = return_confidence_score
        config.return_strategy_stats = return_strategy_stats
        config.enable_debug = enable_debug
        config.strategy_weights = strategy_weights

        # Handle Enabled Strategies
        if enabled_strategies is None:
            es = []
            if config.params.use_sustained: es.append(2)
            if config.params.use_crossover: es.append(3)
            if config.params.use_momentum: es.append(4)
            if config.params.use_breakout: es.append(6)
            if config.params.use_divergence: es.append(7)
            if config.params.use_trend_following: es.append(8)
            if config.params.use_mean_reversion: es.append(9)
            config.enabled_strategies = es
        else:
            config.enabled_strategies = enabled_strategies

    # Validate configuration
    _validate_config(config)

    debug_enabled = config.enable_debug or os.environ.get("RANGE_OSCILLATOR_DEBUG", "false").lower() == "true"

    if debug_enabled:
        log_analysis(f"[Strategy5] Starting combined strategy with Config: {config.enabled_strategies}")

    # -------------------------------------------------------------------------
    # 2. Strategy Execution
    # -------------------------------------------------------------------------
    
    # Calculate Oscillator
    oscillator, ma, range_atr = get_oscillator_data(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )
    
    index = oscillator.index
    
    # Dynamic Selection
    current_enabled_strategies = config.enabled_strategies.copy()
    if config.dynamic.enabled:
        market_conditions = _analyze_market_conditions(
            oscillator=oscillator,
            range_atr=range_atr,
            close=close,
            lookback_period=config.dynamic.lookback,
        )
        current_enabled_strategies = _select_strategies_dynamically(
            market_conditions=market_conditions,
            available_strategies=config.enabled_strategies,
            config=config.dynamic,
        )
        if debug_enabled:
            log_debug(f"[Strategy5] Dynamic selection: {config.enabled_strategies} -> {current_enabled_strategies}")

    if not current_enabled_strategies:
        if debug_enabled: log_debug("[Strategy5] No strategies enabled, fallback to Strategy 1")
        signals, strength = generate_signals_strategy1(high, low, close, oscillator, ma, range_atr, length, mult)
        
        # Consistent 4-element return
        stats_out = {} if config.return_strategy_stats else None
        conf_out = pd.Series(0.0, index=index, dtype="float64") if config.return_confidence_score else None
        
        return signals, strength, stats_out, conf_out

    # Run strategies
    # Sửa lỗi số 2: Remove broad try-except, let it fail for syntax errors, catch only operational if needed.
    # We will still use try-except but log explicit errors and ensure we don't swallow logic bugs.
    # Actually, for reliability, we will remove try-except inside the loop to expose bugs during dev.
    
    signals_dict = {}
    strengths_dict = {}
    strategy_stats = {} if config.return_strategy_stats else None
    
    # Define runners
    # We use a helper to run to avoid repetitive code
    def run_strategy(sid, name, func, **kwargs):
        if sid not in current_enabled_strategies:
            return
        
        # Filter valid kwargs for the specific function if needed, 
        # or rely on functions to accept **kwargs if we standardized them.
        # Since we didn't standardize sub-strategies, we must map manually.
        try:
            sig, strength = func(**kwargs)
            signals_dict[str(sid)] = sig
            strengths_dict[str(sid)] = strength
            
            if config.return_strategy_stats:
                strategy_stats[sid] = {
                    "name": name,
                    "long_count": int((sig == 1).sum()),
                    "short_count": int((sig == -1).sum()),
                }
        except Exception as e:
            # We catch specific errors if we must, but generally better to let it raise if it's a code error.
            # For data errors (e.g. not enough data), we log and continue.
            if debug_enabled:
                log_warn(f"[Strategy5] Strategy {sid} ({name}) failed: {str(e)}")
            # Raise if it seems to be a code error (TypeError, NameError)
            if isinstance(e, (TypeError, NameError, AttributeError)):
                raise e

    # Strategy 2
    run_strategy(2, STRATEGY_NAMES[2], generate_signals_strategy2_sustained,
                 oscillator=oscillator, ma=ma, range_atr=range_atr,
                 min_bars_above_zero=config.params.min_bars_sustained,
                 min_bars_below_zero=config.params.min_bars_sustained, enable_debug=False)

    # Strategy 3
    run_strategy(3, STRATEGY_NAMES[3], generate_signals_strategy3_crossover,
                 oscillator=oscillator, ma=ma, range_atr=range_atr,
                 confirmation_bars=config.params.confirmation_bars, enable_debug=False)
                 
    # Strategy 4
    if len(oscillator) > config.params.momentum_period:
        run_strategy(4, STRATEGY_NAMES[4], generate_signals_strategy4_momentum,
                     oscillator=oscillator, ma=ma, range_atr=range_atr,
                     momentum_period=config.params.momentum_period,
                     momentum_threshold=config.params.momentum_threshold, enable_debug=False)

    # Strategy 6
    run_strategy(6, STRATEGY_NAMES[6], generate_signals_strategy6_breakout,
                 high=high, low=low, close=close, oscillator=oscillator, ma=ma, range_atr=range_atr,
                 length=length, mult=mult,
                 upper_threshold=config.params.breakout_upper_threshold,
                 lower_threshold=config.params.breakout_lower_threshold,
                 confirmation_bars=config.params.breakout_confirmation_bars, enable_debug=False)

    # Strategy 7
    if high is not None:
        run_strategy(7, STRATEGY_NAMES[7], generate_signals_strategy7_divergence,
                     high=high, low=low, close=close, oscillator=oscillator, ma=ma, range_atr=range_atr,
                     length=length, mult=mult,
                     lookback_period=config.params.divergence_lookback_period,
                     min_swing_bars=config.params.divergence_min_swing_bars, enable_debug=False)

    # Strategy 8
    run_strategy(8, STRATEGY_NAMES[8], generate_signals_strategy8_trend_following,
                 oscillator=oscillator, ma=ma, range_atr=range_atr, length=length, mult=mult,
                 trend_filter_period=config.params.trend_filter_period,
                 oscillator_threshold=config.params.trend_oscillator_threshold, enable_debug=False)

    # Strategy 9
    run_strategy(9, STRATEGY_NAMES[9], generate_signals_strategy9_mean_reversion,
                 oscillator=oscillator, ma=ma, range_atr=range_atr, length=length, mult=mult,
                 extreme_threshold=config.params.mean_reversion_extreme_threshold,
                 zero_cross_threshold=config.params.mean_reversion_zero_cross_threshold, enable_debug=False)

    if not signals_dict:
        if debug_enabled: log_debug("[Strategy5] All strategies returned no signals.")
        signals, strength = generate_signals_strategy1(high, low, close, oscillator, ma, range_atr, length, mult)
        
        # Consistent 4-element return
        stats_out = {} if config.return_strategy_stats else None
        conf_out = pd.Series(0.0, index=index, dtype="float64") if config.return_confidence_score else None
        
        return signals, strength, stats_out, conf_out

    # -------------------------------------------------------------------------
    # 3. Aggregation (Sửa lỗi số 1: Index Safety)
    # -------------------------------------------------------------------------
    
    # Use pd.concat to align indexes safely compared to np.stack
    signals_df = pd.concat(signals_dict, axis=1).fillna(0).astype(int)
    strengths_df = pd.concat(strengths_dict, axis=1).fillna(0.0)
    
    # Ensure they have the original index (reindex to handle potential missing rows if strategies dropped index)
    signals_df = signals_df.reindex(index, fill_value=0)
    strengths_df = strengths_df.reindex(index, fill_value=0.0)
    
    # Convert to numpy for fast processing
    signals_array = signals_df.values.T  # (n_strategies, n_bars)
    strengths_array = strengths_df.values.T
    
    successful_strategies = [int(k) for k in signals_dict.keys()]
    
    # -------------------------------------------------------------------------
    # 4. Consensus & Voting
    # -------------------------------------------------------------------------
    
    final_signals = np.zeros(len(index), dtype=np.int8)

    # Calculate Weights
    weights_map = {}
    if config.strategy_weights:
         weights_map = config.strategy_weights.copy()
    else:
         weights_map = {s: 1.0 for s in successful_strategies}

    if config.consensus.adaptive_weights and config.consensus.mode == "weighted":
        adaptive_weights = _calculate_adaptive_weights(
            signals_df, strengths_df, successful_strategies,
            performance_window=config.consensus.performance_window
        )
        weights_map = adaptive_weights
        if debug_enabled:
            log_analysis(f"[Strategy5] Adaptive weights: {weights_map}")

    if config.consensus.mode == "weighted":
        # Sửa lỗi số 4: Improved Weighted Logic
        long_weighted = np.zeros(len(index), dtype=np.float64)
        short_weighted = np.zeros(len(index), dtype=np.float64)
        
        for strategy_id_str in signals_df.columns:
            sid = int(strategy_id_str)
            w = weights_map.get(sid, 0.0)
            
            s_sig = signals_df[strategy_id_str].values
            
            long_weighted += (s_sig == 1) * w
            short_weighted += (s_sig == -1) * w
        
        # Apply Thresholds (Issue 4)
        # 1. Total weight of winning side > weighted_min_total
        # 2. Difference > weighted_min_diff
        
        min_total = config.consensus.weighted_min_total
        min_diff = config.consensus.weighted_min_diff
        
        is_long = (long_weighted > short_weighted) & \
                  (long_weighted > min_total) & \
                  ((long_weighted - short_weighted) > min_diff)
                  
        is_short = (short_weighted > long_weighted) & \
                   (short_weighted > min_total) & \
                   ((short_weighted - long_weighted) > min_diff)
        
        final_signals = np.where(is_long, 1, np.where(is_short, -1, 0))
        
    else:
        # Threshold Mode
        n_strategies = len(successful_strategies)
        min_agreement = int(np.ceil(n_strategies * config.consensus.threshold))
        
        # Sum votes
        long_votes = (signals_df == 1).sum(axis=1).values
        short_votes = (signals_df == -1).sum(axis=1).values
        
        final_signals = np.where(
            (long_votes >= min_agreement) & (long_votes > short_votes), 1,
            np.where(
                (short_votes >= min_agreement) & (short_votes > long_votes), -1,
                0
            )
        )

    # -------------------------------------------------------------------------
    # 5. Signal Strength & Filtering
    # -------------------------------------------------------------------------
    
    # Calculate average strength of agreeing strategies
    final_strengths = np.zeros(len(index), dtype=np.float64)
    
    # Iterate quickly usually, but vectorization is better
    # Vectorized approach:
    # Mask of strategies that agree with final signal
    # This is complex in 2D. Let's do a simple approximation or loop if fast enough. 
    # With numpy arrays aligned:
    
    # Expand final_signals to match shape (n_bars, 1) -> (n_strategies, n_bars) or reverse
    # signals_array is (n_strategies, n_bars)
    # final_signals is (n_bars,)
    
    final_sig_broadcast = final_signals[np.newaxis, :] # (1, n_bars)
    
    # Where individual strategy agrees with final signal
    agree_mask = (signals_array == final_sig_broadcast) & (final_sig_broadcast != 0)
    
    # Sum strengths of agreeing strategies
    # strengths_array is (n_strategies, n_bars)
    strength_sum = np.sum(strengths_array * agree_mask, axis=0)
    count_sum = np.sum(agree_mask, axis=0)
    
    final_strengths = np.divide(strength_sum, count_sum, out=np.zeros_like(strength_sum), where=count_sum > 0)
    
    # Filter by min_signal_strength
    if config.min_signal_strength > 0:
        mask = final_strengths >= config.min_signal_strength
        final_signals = np.where(mask, final_signals, 0)
    
    # -------------------------------------------------------------------------
    # 6. Returns
    # -------------------------------------------------------------------------
    
    signal_series = pd.Series(final_signals, index=index, name="signal").astype("int8")
    strength_series = pd.Series(final_strengths, index=index, name="strength").astype("float64")
    
    # Handle NaNs from source
    valid_mask = ~oscillator.isna()
    signal_series = signal_series.where(valid_mask, 0)
    strength_series = strength_series.where(valid_mask, 0.0)
    
    # Consistent 4-element return
    stats_out = strategy_stats if config.return_strategy_stats else None
    
    conf_out = None
    if config.return_confidence_score:
        confidence = _calculate_confidence_score(
            signals_array, strengths_array,
            len(successful_strategies),
            config.consensus.mode,
            config.consensus.threshold
        )
        conf_out = pd.Series(confidence, index=index)
        
    return signal_series, strength_series, stats_out, conf_out

__all__ = [
    "generate_signals_strategy5_combined",
    "Strategy5Config",
    "ConsensusConfig",
    "DynamicSelectionConfig",
    "StrategySpecificConfig",
    "STRATEGY_FUNCTIONS"
]
