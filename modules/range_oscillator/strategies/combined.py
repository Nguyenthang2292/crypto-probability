"""
Range Oscillator Strategy 5: Combined (Enhanced).

This module provides an enhanced combined signal generation strategy that combines
multiple methods with advanced features:
- Support for all available strategies (2-9)
- Weighted voting system
- Multiple consensus modes
- Signal strength filtering
- Performance tracking

Refactored to use a class-based architecture (`CombinedStrategy`) for better maintainability.
"""

from typing import Optional, Tuple, Dict, List, Literal, Any, Union
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass, field

from modules.range_oscillator.core.utils import get_oscillator_data
from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
from modules.range_oscillator.strategies.sustained import generate_signals_strategy2_sustained
from modules.range_oscillator.strategies.crossover import generate_signals_strategy3_crossover
from modules.range_oscillator.strategies.momentum import generate_signals_strategy4_momentum
from modules.range_oscillator.strategies.breakout import generate_signals_breakout_strategy
from modules.range_oscillator.strategies.divergence import generate_signals_strategy7_divergence
from modules.range_oscillator.strategies.trend_following import generate_signals_strategy8_trend_following
from modules.range_oscillator.strategies.mean_reversion import generate_signals_strategy9_mean_reversion
from modules.common.utils import log_debug, log_analysis, log_warn


# Strategy mapping for easy access
STRATEGY_FUNCTIONS = {
    2: generate_signals_strategy2_sustained,
    3: generate_signals_strategy3_crossover,
    4: generate_signals_strategy4_momentum,
    6: generate_signals_breakout_strategy,
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
# Configuration Objects
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
    
    # Weighted voting rules
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
# Combined Strategy Class
# -----------------------------------------------------------------------------

class CombinedStrategy:
    """
    Class-based implementation of Range Oscillator Strategy 5: Combined (Enhanced).
    """

    def __init__(self, config: Optional[Strategy5Config] = None, **kwargs):
        """
        Initialize the CombinedStrategy.

        Args:
            config: Configuration object. If None, constructed from kwargs.
            **kwargs: Legacy arguments to override config or create it.
        """
        self.config = self._build_config(config, **kwargs)
        self.debug_enabled = self.config.enable_debug or os.environ.get("RANGE_OSCILLATOR_DEBUG", "false").lower() == "true"
        self._validate_config()

    def _build_config(self, config: Optional[Strategy5Config], **kwargs) -> Strategy5Config:
        """Construct configuration from arguments."""
        if config is None:
            config = Strategy5Config()
            
        # Helper to get arg from kwargs or default
        def get_arg(key, default):
            return kwargs.get(key, default)

        # Apply overrides if kwargs provided (legacy support)
        if kwargs:
            # Consensus
            if "consensus_mode" in kwargs: config.consensus.mode = kwargs["consensus_mode"]
            if "consensus_threshold" in kwargs: config.consensus.threshold = kwargs["consensus_threshold"]
            if "enable_adaptive_weights" in kwargs: config.consensus.adaptive_weights = kwargs["enable_adaptive_weights"]
            if "adaptive_performance_window" in kwargs: config.consensus.performance_window = kwargs["adaptive_performance_window"]
            if "weighted_min_diff" in kwargs: config.consensus.weighted_min_diff = kwargs["weighted_min_diff"]
            if "weighted_min_total" in kwargs: config.consensus.weighted_min_total = kwargs["weighted_min_total"]

            # Dynamic
            if "enable_dynamic_selection" in kwargs: 
                 val = kwargs["enable_dynamic_selection"]
                 config.dynamic.enabled = (val if val is not None else False)
            if "dynamic_selection_lookback" in kwargs: config.dynamic.lookback = kwargs["dynamic_selection_lookback"]
            if "dynamic_volatility_threshold" in kwargs: config.dynamic.volatility_threshold = kwargs["dynamic_volatility_threshold"]
            if "dynamic_trend_threshold" in kwargs: config.dynamic.trend_threshold = kwargs["dynamic_trend_threshold"]

            # Params
            p = config.params
            p.use_sustained = get_arg("use_sustained", p.use_sustained)
            p.min_bars_sustained = get_arg("min_bars_sustained", p.min_bars_sustained)
            p.use_crossover = get_arg("use_crossover", p.use_crossover)
            p.confirmation_bars = get_arg("confirmation_bars", p.confirmation_bars)
            p.use_momentum = get_arg("use_momentum", p.use_momentum)
            p.momentum_period = get_arg("momentum_period", p.momentum_period)
            p.momentum_threshold = get_arg("momentum_threshold", p.momentum_threshold)
            p.use_breakout = get_arg("use_breakout", p.use_breakout)
            p.breakout_upper_threshold = get_arg("breakout_upper_threshold", p.breakout_upper_threshold)
            p.breakout_lower_threshold = get_arg("breakout_lower_threshold", p.breakout_lower_threshold)
            p.breakout_confirmation_bars = get_arg("breakout_confirmation_bars", p.breakout_confirmation_bars)
            p.use_divergence = get_arg("use_divergence", p.use_divergence)
            p.divergence_lookback_period = get_arg("divergence_lookback_period", p.divergence_lookback_period)
            p.divergence_min_swing_bars = get_arg("divergence_min_swing_bars", p.divergence_min_swing_bars)
            p.use_trend_following = get_arg("use_trend_following", p.use_trend_following)
            p.trend_filter_period = get_arg("trend_filter_period", p.trend_filter_period)
            p.trend_oscillator_threshold = get_arg("trend_oscillator_threshold", p.trend_oscillator_threshold)
            p.use_mean_reversion = get_arg("use_mean_reversion", p.use_mean_reversion)
            p.mean_reversion_extreme_threshold = get_arg("mean_reversion_extreme_threshold", p.mean_reversion_extreme_threshold)
            p.mean_reversion_zero_cross_threshold = get_arg("mean_reversion_zero_cross_threshold", p.mean_reversion_zero_cross_threshold)

            # General
            config.min_signal_strength = get_arg("min_signal_strength", config.min_signal_strength)
            config.return_confidence_score = get_arg("return_confidence_score", config.return_confidence_score)
            config.return_strategy_stats = get_arg("return_strategy_stats", config.return_strategy_stats)
            config.enable_debug = get_arg("enable_debug", config.enable_debug)
            if "strategy_weights" in kwargs: config.strategy_weights = kwargs["strategy_weights"]

            # Enabled Strategies
            if "enabled_strategies" in kwargs and kwargs["enabled_strategies"] is not None:
                config.enabled_strategies = kwargs["enabled_strategies"]

        # If enabled_strategies is empty/None AND not explicitly passed, populate based on flags (legacy behavior)
        # Only rebuild from flags if enabled_strategies was not in kwargs at all
        if not config.enabled_strategies and "enabled_strategies" not in kwargs:
            es = []
            if config.params.use_sustained: es.append(2)
            if config.params.use_crossover: es.append(3)
            if config.params.use_momentum: es.append(4)
            if config.params.use_breakout: es.append(6)
            if config.params.use_divergence: es.append(7)
            if config.params.use_trend_following: es.append(8)
            if config.params.use_mean_reversion: es.append(9)
            config.enabled_strategies = es

        return config

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        c = self.config
        if c.consensus.mode not in ("threshold", "weighted"):
            raise ValueError(f"Invalid consensus mode: {c.consensus.mode}")
        if not (0.0 <= c.consensus.threshold <= 1.0):
            raise ValueError(f"consensus_threshold must be in [0, 1], got {c.consensus.threshold}")
        if c.enabled_strategies:
            invalid = set(c.enabled_strategies) - VALID_STRATEGY_IDS
            if invalid:
                raise ValueError(f"Invalid strategy IDs: {invalid}")
        if c.consensus.mode == "weighted":
            if c.consensus.weighted_min_total < 0:
                raise ValueError(f"weighted_min_total must be >= 0")
            if c.consensus.weighted_min_diff < 0:
                raise ValueError(f"weighted_min_diff must be >= 0")
        if c.min_signal_strength < 0:
            raise ValueError(f"min_signal_strength must be >= 0")
        if c.dynamic.enabled:
            if not (0.0 <= c.dynamic.volatility_threshold <= 1.0):
                raise ValueError(f"volatility_threshold must be in [0, 1]")
            if not (0.0 <= c.dynamic.trend_threshold <= 1.0):
                raise ValueError(f"trend_threshold must be in [0, 1]")
            if c.dynamic.lookback < 1:
                raise ValueError(f"lookback must be >= 1")
        
        # Validate strategy specific params
        p = c.params
        if p.use_sustained and p.min_bars_sustained <= 0:
            raise ValueError(f"min_bars_sustained must be > 0")
        if p.use_crossover and p.confirmation_bars <= 0:
            raise ValueError(f"confirmation_bars must be > 0")
        if p.use_momentum:
            if p.momentum_period <= 0:
                raise ValueError(f"momentum_period must be > 0")
            if p.momentum_threshold < 0:
                raise ValueError(f"momentum_threshold must be >= 0")
        if p.use_trend_following and p.trend_filter_period <= 0:
            raise ValueError(f"trend_filter_period must be > 0")

    def _analyze_market_conditions(
        self, oscillator: pd.Series, range_atr: pd.Series, lookback_period: int
    ) -> Dict[str, float]:
        """Analyze market conditions."""
        if len(oscillator) < lookback_period:
            lookback_period = len(oscillator)
        
        if lookback_period == 0:
            return {"volatility": 0.5, "trend_strength": 0.5, "range_bound_score": 0.5, "oscillator_extreme": 0.5}
        
        recent_osc = oscillator.iloc[-lookback_period:]
        recent_atr = range_atr.iloc[-lookback_period:]
        
        atr_std = recent_atr.std()
        atr_mean = recent_atr.mean()
        volatility = min(1.0, (atr_std / (atr_mean + 1e-10)) * 2.0) if atr_mean > 0 else 0.5
        
        osc_trend = recent_osc.diff().abs().mean()
        osc_range = recent_osc.max() - recent_osc.min()
        trend_strength = min(1.0, (osc_trend / (osc_range + 1e-10)) * 2.0) if osc_range > 0 else 0.5
        
        range_bound_score = 1.0 - trend_strength
        osc_abs = recent_osc.abs()
        oscillator_extreme = min(1.0, osc_abs.mean() / OSCILLATOR_NORMALIZATION)
        
        return {
            "volatility": float(volatility),
            "trend_strength": float(trend_strength),
            "range_bound_score": float(range_bound_score),
            "oscillator_extreme": float(oscillator_extreme),
        }

    def _select_strategies_dynamically(
        self, market_conditions: Dict[str, float], available_strategies: List[int]
    ) -> List[int]:
        """Dynamically select strategies."""
        selected = []
        volatility = market_conditions["volatility"]
        trend_strength = market_conditions["trend_strength"]
        range_bound = market_conditions["range_bound_score"]
        conf = self.config.dynamic
        
        if volatility >= conf.volatility_threshold:
            for s in VOLATILE_STRATEGIES:
                if s in available_strategies: selected.append(s)
        
        if trend_strength >= conf.trend_threshold:
            for s in TRENDING_STRATEGIES:
                if s in available_strategies and s not in selected: selected.append(s)
        
        if range_bound >= 0.5:
            for s in RANGE_BOUND_STRATEGIES:
                if s in available_strategies and s not in selected: selected.append(s)
        
        if not selected:
            selected = available_strategies.copy()
        
        return sorted(selected)

    def _calculate_adaptive_weights(
        self, signals_df: pd.DataFrame, strengths_df: pd.DataFrame, enabled_strategies: List[int]
    ) -> Dict[int, float]:
        """Calculate adaptive weights."""
        n_bars = len(signals_df)
        window = self.config.consensus.performance_window
        min_samples = 5
        
        if n_bars < min_samples:
            return {s: 1.0 for s in enabled_strategies}
        
        window = min(window, n_bars)
        recent_signals = signals_df.iloc[-window:]
        recent_strengths = strengths_df.iloc[-window:]
        
        strategy_scores = {}
        row_sums = recent_signals.sum(axis=1)
        consensus = np.sign(row_sums)
        
        for sid in enabled_strategies:
            col = str(sid)
            if col not in recent_signals.columns: continue
            
            s_sig = recent_signals[col]
            s_str = recent_strengths[col]
            
            matches = (s_sig == consensus)
            agreement = matches.mean() if len(consensus) > 0 else 0.5
            
            active_mask = np.abs(s_sig) > 0
            avg_strength = s_str[active_mask].mean() if active_mask.any() else 0.0
            
            score = (agreement * AGREEMENT_WEIGHT) + (avg_strength * STRENGTH_WEIGHT)
            strategy_scores[sid] = max(0.1, score)
        
        total = sum(strategy_scores.values())
        if total > 0:
            strategy_scores = {k: v / total * len(enabled_strategies) for k, v in strategy_scores.items()}
        
        return strategy_scores

    def _calculate_confidence_score(
        self, signals_array: np.ndarray, strengths_array: np.ndarray, n_strategies: int
    ) -> np.ndarray:
        """Calculate confidence scores."""
        n_bars = signals_array.shape[1]
        confidence = np.zeros(n_bars, dtype=np.float64)
        c_conf = self.config.consensus
        
        for i in range(n_bars):
            bar_sig = signals_array[:, i]
            bar_str = strengths_array[:, i]
            
            long_votes = np.sum(bar_sig == 1)
            short_votes = np.sum(bar_sig == -1)
            total = long_votes + short_votes
            
            if total == 0:
                confidence[i] = 0.0
                continue
            
            if c_conf.mode == "threshold":
                min_agree = int(np.ceil(n_strategies * c_conf.threshold))
                if long_votes >= min_agree:
                    agree_level = long_votes / n_strategies
                elif short_votes >= min_agree:
                    agree_level = short_votes / n_strategies
                else:
                    agree_level = max(long_votes, short_votes) / n_strategies
            else: # weighted (simplified for confidence score calculation)
                agree_level = total / n_strategies
            
            agree_mask = (bar_sig != 0)
            avg_str = np.mean(bar_str[agree_mask]) if np.any(agree_mask) else 0.0
            
            confidence[i] = (agree_level * AGREEMENT_WEIGHT) + (avg_str * STRENGTH_WEIGHT)
        
        return confidence

    def generate_signals(
        self,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        close: Optional[pd.Series] = None,
        *,
        oscillator: Optional[pd.Series] = None,
        ma: Optional[pd.Series] = None,
        range_atr: Optional[pd.Series] = None,
        length: int = 50,
        mult: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, Optional[Dict], Optional[pd.Series]]:
        """
        Generate trading signals using the combined strategy logic.
        """
        if self.debug_enabled:
            log_analysis(f"[Strategy5] Starting combined strategy with Config: {self.config.enabled_strategies}")

        # 1. Prepare Data
        oscillator, ma, range_atr = get_oscillator_data(
            high=high, low=low, close=close,
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            length=length, mult=mult
        )
        index = oscillator.index

        # 2. Dynamic Selection
        current_enabled_strategies = self.config.enabled_strategies.copy()
        if self.config.dynamic.enabled:
            conditions = self._analyze_market_conditions(
                oscillator, range_atr, self.config.dynamic.lookback
            )
            current_enabled_strategies = self._select_strategies_dynamically(
                conditions, self.config.enabled_strategies
            )
            if self.debug_enabled:
                log_debug(f"[Strategy5] Dynamic selection: {self.config.enabled_strategies} -> {current_enabled_strategies}")

        # Fallback if no strategies
        if not current_enabled_strategies:
            if self.debug_enabled: log_debug("[Strategy5] No strategies enabled, fallback to Strategy 1")
            return self._fallback_strategy1(high, low, close, oscillator, ma, range_atr, length, mult, index)

        # 3. Run Strategies
        signals_dict = {}
        strengths_dict = {}
        strategy_stats = {} if self.config.return_strategy_stats else None
        
        # Helper to run individual strategy
        def run_st(sid, name, func, **k):
            if sid not in current_enabled_strategies: return
            try:
                s, str_ = func(**k)
                signals_dict[str(sid)] = s
                strengths_dict[str(sid)] = str_
                if self.config.return_strategy_stats:
                    strategy_stats[sid] = {
                        "name": name, 
                        "long_count": int((s == 1).sum()), 
                        "short_count": int((s == -1).sum())
                    }
            except Exception as e:
                if self.debug_enabled:
                    log_warn(f"[Strategy5] Strategy {sid} ({name}) failed: {str(e)}")
                if isinstance(e, (TypeError, NameError, AttributeError)):
                    raise e

        p = self.config.params
        
        run_st(2, STRATEGY_NAMES[2], generate_signals_strategy2_sustained,
               oscillator=oscillator, ma=ma, range_atr=range_atr,
               min_bars_above_zero=p.min_bars_sustained, min_bars_below_zero=p.min_bars_sustained, enable_debug=False)
               
        run_st(3, STRATEGY_NAMES[3], generate_signals_strategy3_crossover,
               oscillator=oscillator, ma=ma, range_atr=range_atr,
               confirmation_bars=p.confirmation_bars, enable_debug=False)

        if len(oscillator) > p.momentum_period:
            run_st(4, STRATEGY_NAMES[4], generate_signals_strategy4_momentum,
                   oscillator=oscillator, ma=ma, range_atr=range_atr,
                   momentum_period=p.momentum_period, momentum_threshold=p.momentum_threshold, enable_debug=False)

        run_st(6, STRATEGY_NAMES[6], generate_signals_breakout_strategy,
               high=high, low=low, close=close, oscillator=oscillator, ma=ma, range_atr=range_atr,
               length=length, mult=mult, upper_threshold=p.breakout_upper_threshold,
               lower_threshold=p.breakout_lower_threshold, confirmation_bars=p.breakout_confirmation_bars, enable_debug=False)

        if high is not None:
            run_st(7, STRATEGY_NAMES[7], generate_signals_strategy7_divergence,
                   high=high, low=low, close=close, oscillator=oscillator, ma=ma, range_atr=range_atr,
                   length=length, mult=mult, lookback_period=p.divergence_lookback_period,
                   min_swing_bars=p.divergence_min_swing_bars, enable_debug=False)

        run_st(8, STRATEGY_NAMES[8], generate_signals_strategy8_trend_following,
               oscillator=oscillator, ma=ma, range_atr=range_atr, length=length, mult=mult,
               trend_filter_period=p.trend_filter_period, oscillator_threshold=p.trend_oscillator_threshold, enable_debug=False)

        run_st(9, STRATEGY_NAMES[9], generate_signals_strategy9_mean_reversion,
               oscillator=oscillator, ma=ma, range_atr=range_atr, length=length, mult=mult,
               extreme_threshold=p.mean_reversion_extreme_threshold, zero_cross_threshold=p.mean_reversion_zero_cross_threshold, enable_debug=False)

        if not signals_dict:
             if self.debug_enabled: log_debug("[Strategy5] All strategies returned no signals.")
             return self._fallback_strategy1(high, low, close, oscillator, ma, range_atr, length, mult, index)

        # 4. Aggregation
        signals_df = pd.concat(signals_dict, axis=1).fillna(0).astype(int)
        strengths_df = pd.concat(strengths_dict, axis=1).fillna(0.0)
        
        signals_df = signals_df.reindex(index, fill_value=0)
        strengths_df = strengths_df.reindex(index, fill_value=0.0)
        
        signals_array = signals_df.values.T
        strengths_array = strengths_df.values.T
        successful_strategies = [int(k) for k in signals_dict.keys()]

        # 5. Consensus & Voting
        final_signals = np.zeros(len(index), dtype=np.int8)
        
        # Calculate Weights
        weights_map = self.config.strategy_weights.copy() if self.config.strategy_weights else {s: 1.0 for s in successful_strategies}
        
        if self.config.consensus.adaptive_weights and self.config.consensus.mode == "weighted":
            weights_map = self._calculate_adaptive_weights(signals_df, strengths_df, successful_strategies)
            if self.debug_enabled: log_analysis(f"[Strategy5] Adaptive weights: {weights_map}")

        # Voting Logic
        if self.config.consensus.mode == "weighted":
            long_w = np.zeros(len(index), dtype=np.float64)
            short_w = np.zeros(len(index), dtype=np.float64)
            
            for col in signals_df.columns:
                sid = int(col)
                w = weights_map.get(sid, 0.0)
                sig = signals_df[col].values
                long_w += (sig == 1) * w
                short_w += (sig == -1) * w
            
            min_tot = self.config.consensus.weighted_min_total
            min_diff = self.config.consensus.weighted_min_diff
            
            is_long = (long_w > short_w) & (long_w > min_tot) & ((long_w - short_w) > min_diff)
            is_short = (short_w > long_w) & (short_w > min_tot) & ((short_w - long_w) > min_diff)
            
            final_signals = np.where(is_long, 1, np.where(is_short, -1, 0))
        else:
            # Threshold mode
            n_strat = len(successful_strategies)
            min_agree = int(np.ceil(n_strat * self.config.consensus.threshold))
            
            long_votes = (signals_df == 1).sum(axis=1).values
            short_votes = (signals_df == -1).sum(axis=1).values
            
            final_signals = np.where(
                (long_votes >= min_agree) & (long_votes > short_votes), 1,
                np.where(
                    (short_votes >= min_agree) & (short_votes > long_votes), -1, 0
                )
            )

        # 6. Signal Strength
        # Calculate final strengths based on agreeing strategies
        final_sig_broadcast = final_signals[np.newaxis, :]
        agree_mask = (signals_array == final_sig_broadcast) & (final_sig_broadcast != 0)
        
        str_sum = np.sum(strengths_array * agree_mask, axis=0)
        cnt_sum = np.sum(agree_mask, axis=0)
        final_strengths = np.divide(str_sum, cnt_sum, out=np.zeros_like(str_sum), where=cnt_sum > 0)
        
        if self.config.min_signal_strength > 0:
            mask = final_strengths >= self.config.min_signal_strength
            final_signals = np.where(mask, final_signals, 0)
            
        # 7. Outputs
        signal_series = pd.Series(final_signals, index=index, name="signal").astype("int8")
        strength_series = pd.Series(final_strengths, index=index, name="strength").astype("float64")
        
        valid_mask = ~oscillator.isna()
        signal_series = signal_series.where(valid_mask, 0)
        strength_series = strength_series.where(valid_mask, 0.0)
        
        conf_out = None
        if self.config.return_confidence_score:
            conf = self._calculate_confidence_score(signals_array, strengths_array, len(successful_strategies))
            conf_out = pd.Series(conf, index=index)
            

        if not self.config.return_strategy_stats and not self.config.return_confidence_score:
            return signal_series, strength_series

        return signal_series, strength_series, strategy_stats, conf_out

    def _fallback_strategy1(self, high, low, close, oscillator, ma, range_atr, length, mult, index):
        """Run basic strategy as fallback."""
        s, str_ = generate_signals_basic_strategy(
            high=high, low=low, close=close,
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            length=length, mult=mult
        )
        if not self.config.return_strategy_stats and not self.config.return_confidence_score:
            return s, str_
            
        stats = {} if self.config.return_strategy_stats else None
        conf = pd.Series(0.0, index=index, dtype="float64") if self.config.return_confidence_score else None
        return s, str_, stats, conf


# -----------------------------------------------------------------------------
# Wrapper for Backward Compatibility
# -----------------------------------------------------------------------------

def generate_signals_strategy5_combined(
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    *,
    # Core params
    oscillator: Optional[pd.Series] = None,
    ma: Optional[pd.Series] = None,
    range_atr: Optional[pd.Series] = None,
    length: int = 50,
    mult: float = 2.0,
    
    # Config object
    config: Optional[Strategy5Config] = None,
    
    # Legacy args
    **kwargs
) -> Union[Tuple[pd.Series, pd.Series], Tuple[pd.Series, pd.Series, Optional[Dict], Optional[pd.Series]]]:
    """
    Generate trading signals based on Range Oscillator Strategy 5: Combined (Enhanced).
    Wrapper around CombinedStrategy class for backward compatibility.
    """
    strategy = CombinedStrategy(config=config, **kwargs)
    return strategy.generate_signals(
        high=high, low=low, close=close,
        oscillator=oscillator, ma=ma, range_atr=range_atr,
        length=length, mult=mult
    )

__all__ = [
    "generate_signals_strategy5_combined",
    "CombinedStrategy",
    "Strategy5Config",
    "ConsensusConfig",
    "DynamicSelectionConfig",
    "StrategySpecificConfig",
    "STRATEGY_FUNCTIONS"
]
