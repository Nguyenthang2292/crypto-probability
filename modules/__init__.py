"""
Modules package for crypto prediction system.
Provides compatibility aliases for legacy import paths used in tests.
"""

from . import config

__all__ = ["config"]

# Backwards-compatible import aliases
import importlib
import sys

_ALIASES = {
    # common subpackage
    "modules.DataFetcher": "modules.common.DataFetcher",
    "modules.ExchangeManager": "modules.common.ExchangeManager",
    "modules.ProgressBar": "modules.common.ProgressBar",
    "modules.Position": "modules.common.Position",
    "modules.utils": "modules.common.utils",
    # deeplearning subpackage
    "modules.deeplearning_data_pipeline": "modules.deeplearning.data_pipeline",
    "modules.deeplearning_dataset": "modules.deeplearning.dataset",
    "modules.deeplearning_environment_setup": "modules.deeplearning.environment_setup",
    "modules.deeplearning_feature_selection": "modules.deeplearning.feature_selection",
    "modules.deeplearning_model": "modules.deeplearning.model",
    "modules.feature_selection": "modules.deeplearning.feature_selection",
    # pairs trading subpackage
    "modules.pairs_trading_cli": "modules.pairs_trading.cli",
    "modules.pairs_trading_hedge_ratio": "modules.pairs_trading.metrics.ols_hedge_ratio",
    "modules.pairs_trading_opportunity_scorer": "modules.pairs_trading.opportunity_scorer",
    "modules.pairs_trading_pair_metrics_computer": "modules.pairs_trading.pair_metrics_computer",
    "modules.pairs_trading_pairs_analyzer": "modules.pairs_trading.pairs_analyzer",
    "modules.pairs_trading_performance_analyzer": "modules.pairs_trading.performance_analyzer",
    "modules.pairs_trading_risk_metrics": "modules.pairs_trading.metrics.max_drawdown",
    "modules.pairs_trading_zscore_metrics": "modules.pairs_trading.metrics.direction_metrics",
    # portfolio subpackage
    "modules.portfolio_correlation_analyzer": "modules.portfolio.correlation_analyzer",
    "modules.portfolio_hedge_finder": "modules.portfolio.hedge_finder",
    "modules.portfolio_risk_calculator": "modules.portfolio.risk_calculator",
    "modules.PortfolioCorrelationAnalyzer": "modules.portfolio.correlation_analyzer",
    "modules.HedgeFinder": "modules.portfolio.hedge_finder",
    "modules.PortfolioRiskCalculator": "modules.portfolio.risk_calculator",
    # xgboost subpackage
    "modules.xgboost_prediction_cli": "modules.xgboost.cli",
    "modules.xgboost_prediction_display": "modules.xgboost.display",
    "modules.xgboost_prediction_labeling": "modules.xgboost.labeling",
    "modules.xgboost_prediction_model": "modules.xgboost.model",
    "modules.xgboost_prediction_utils": "modules.xgboost.utils",
}

_this_package = sys.modules[__name__]

for alias, target in _ALIASES.items():
    try:
        module = importlib.import_module(target)
    except Exception:
        continue
    sys.modules.setdefault(alias, module)
    attr_name = alias.split(".", 1)[1] if "." in alias else None
    if attr_name and not hasattr(_this_package, attr_name):
        setattr(_this_package, attr_name, module)
