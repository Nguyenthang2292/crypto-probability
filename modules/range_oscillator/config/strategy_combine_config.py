"""Main configuration for Combined Strategy."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

from .dynamic_selection_config import DynamicSelectionConfig
from .consensus_config import ConsensusConfig
from .strategy_specific_config import StrategySpecificConfig


@dataclass
class CombinedStrategyConfig:
    """Main configuration for Range Oscillator Combined Strategy.
    
    This configuration class combines all settings for the combined strategy,
    including which strategies to use, consensus settings, dynamic selection,
    and strategy-specific parameters.
    
    Attributes:
        enabled_strategies: List of strategy IDs to enable (e.g., [2, 3, 4]).
        consensus: Consensus configuration for combining signals.
        dynamic: Dynamic selection configuration.
        params: Strategy-specific parameter configuration.
        min_signal_strength: Minimum signal strength to include (0-1).
        strategy_weights: Optional custom weights for each strategy.
        return_confidence_score: Whether to return confidence scores.
        return_strategy_stats: Whether to return strategy statistics.
        enable_debug: Whether to enable debug logging.
    """
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

