"""Hedge ratio calculations for pairs trading."""

from modules.pairs_trading.metrics.hedge_ratios.kalman_hedge_ratio import (
    calculate_kalman_hedge_ratio,
)
from modules.pairs_trading.metrics.hedge_ratios.ols_hedge_ratio import (
    calculate_ols_hedge_ratio,
)

__all__ = [
    'calculate_kalman_hedge_ratio',
    'calculate_ols_hedge_ratio',
]

