"""Metrics calculations for pairs trading."""

# Statistical tests
from modules.pairs_trading.metrics.statistical_tests import (
    calculate_adf_test,
    calculate_johansen_test,
    calculate_correlation,
)

# Mean reversion metrics
from modules.pairs_trading.metrics.mean_reversion import (
    calculate_half_life,
    calculate_hurst_exponent,
    calculate_zscore_stats,
)

# Risk metrics
from modules.pairs_trading.metrics.risk import (
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_spread_sharpe,
)

# Hedge ratio calculations
from modules.pairs_trading.metrics.hedge_ratios import (
    calculate_kalman_hedge_ratio,
    calculate_ols_hedge_ratio,
)

# Classification metrics
from modules.pairs_trading.metrics.classification import (
    calculate_direction_metrics,
)

__all__ = [
    # Statistical tests
    'calculate_adf_test',
    'calculate_johansen_test',
    'calculate_correlation',
    # Mean reversion metrics
    'calculate_half_life',
    'calculate_hurst_exponent',
    'calculate_zscore_stats',
    # Risk metrics
    'calculate_spread_sharpe',
    'calculate_max_drawdown',
    'calculate_calmar_ratio',
    # Hedge ratio
    'calculate_ols_hedge_ratio',
    'calculate_kalman_hedge_ratio',
    # Classification metrics
    'calculate_direction_metrics',
]
