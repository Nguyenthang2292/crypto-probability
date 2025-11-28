"""Risk metrics for pairs trading analysis."""

from modules.pairs_trading.metrics.risk.calmar_ratio import (
    calculate_calmar_ratio,
)
from modules.pairs_trading.metrics.risk.max_drawdown import (
    calculate_max_drawdown,
)
from modules.pairs_trading.metrics.risk.sharpe_ratio import (
    calculate_spread_sharpe,
)

__all__ = [
    'calculate_calmar_ratio',
    'calculate_max_drawdown',
    'calculate_spread_sharpe',
]

