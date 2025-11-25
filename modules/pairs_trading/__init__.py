"""Pairs trading analysis component."""

from modules.pairs_trading.pairs_analyzer import PairsTradingAnalyzer
from modules.pairs_trading.pair_metrics_computer import PairMetricsComputer
from modules.pairs_trading.opportunity_scorer import OpportunityScorer

# Statistical tests
from modules.pairs_trading.statistical_tests import (
    calculate_adf_test,
    calculate_half_life,
    calculate_johansen_test,
)

# Risk metrics
from modules.pairs_trading.risk_metrics import (
    calculate_spread_sharpe,
    calculate_max_drawdown,
    calculate_calmar_ratio,
)

# Hedge ratio calculations
from modules.pairs_trading.hedge_ratio import (
    calculate_ols_hedge_ratio,
    calculate_kalman_hedge_ratio,
)

# Z-score metrics
from modules.pairs_trading.zscore_metrics import (
    calculate_zscore_stats,
    calculate_hurst_exponent,
    calculate_direction_metrics,
)

__all__ = [
    # Main classes
    'PairsTradingAnalyzer',
    'PairMetricsComputer',
    'OpportunityScorer',
    # Statistical tests
    'calculate_adf_test',
    'calculate_half_life',
    'calculate_johansen_test',
    # Risk metrics
    'calculate_spread_sharpe',
    'calculate_max_drawdown',
    'calculate_calmar_ratio',
    # Hedge ratio
    'calculate_ols_hedge_ratio',
    'calculate_kalman_hedge_ratio',
    # Z-score metrics
    'calculate_zscore_stats',
    'calculate_hurst_exponent',
    'calculate_direction_metrics',
]
