"""Configuration for strategy-specific parameters."""

from dataclasses import dataclass


@dataclass
class StrategySpecificConfig:
    """Configuration for strategy-specific parameters.
    
    This class contains parameters specific to each individual strategy
    used in the combined strategy approach.
    """
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

