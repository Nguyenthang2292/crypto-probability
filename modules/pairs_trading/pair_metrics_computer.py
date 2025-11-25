"""
Pair metrics computer that orchestrates all quantitative metrics calculation.
"""

import pandas as pd
from typing import Dict, Optional

from modules.pairs_trading.statistical_tests import (
    calculate_adf_test,
    calculate_half_life,
    calculate_johansen_test,
)
from modules.pairs_trading.risk_metrics import (
    calculate_spread_sharpe,
    calculate_max_drawdown,
    calculate_calmar_ratio,
)
from modules.pairs_trading.hedge_ratio import (
    calculate_ols_hedge_ratio,
    calculate_kalman_hedge_ratio,
)
from modules.pairs_trading.zscore_metrics import (
    calculate_zscore_stats,
    calculate_hurst_exponent,
    calculate_direction_metrics,
)

try:
    from modules.config import (
        PAIRS_TRADING_TIMEFRAME,
        PAIRS_TRADING_ADF_PVALUE_THRESHOLD,
        PAIRS_TRADING_PERIODS_PER_YEAR,
        PAIRS_TRADING_ZSCORE_LOOKBACK,
        PAIRS_TRADING_CLASSIFICATION_ZSCORE,
        PAIRS_TRADING_JOHANSEN_CONFIDENCE,
        PAIRS_TRADING_CORRELATION_MIN_POINTS,
    )
except ImportError:
    PAIRS_TRADING_TIMEFRAME = "1h"
    PAIRS_TRADING_ADF_PVALUE_THRESHOLD = 0.05
    PAIRS_TRADING_PERIODS_PER_YEAR = 365 * 24
    PAIRS_TRADING_ZSCORE_LOOKBACK = 60
    PAIRS_TRADING_CLASSIFICATION_ZSCORE = 0.5
    PAIRS_TRADING_JOHANSEN_CONFIDENCE = 0.95
    PAIRS_TRADING_CORRELATION_MIN_POINTS = 50


class PairMetricsComputer:
    """Computes comprehensive quantitative metrics for trading pairs."""

    def __init__(
        self,
        adf_pvalue_threshold: float = PAIRS_TRADING_ADF_PVALUE_THRESHOLD,
        periods_per_year: int = PAIRS_TRADING_PERIODS_PER_YEAR,
        zscore_lookback: int = PAIRS_TRADING_ZSCORE_LOOKBACK,
        classification_zscore: float = PAIRS_TRADING_CLASSIFICATION_ZSCORE,
        johansen_confidence: float = PAIRS_TRADING_JOHANSEN_CONFIDENCE,
        correlation_min_points: int = PAIRS_TRADING_CORRELATION_MIN_POINTS,
    ):
        """
        Initialize PairMetricsComputer.
        
        Args:
            adf_pvalue_threshold: P-value threshold for ADF test
            periods_per_year: Number of periods per year
            zscore_lookback: Lookback period for z-score calculation
            classification_zscore: Z-score threshold for classification
            johansen_confidence: Confidence level for Johansen test
            correlation_min_points: Minimum data points required
        """
        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.periods_per_year = periods_per_year
        self.zscore_lookback = zscore_lookback
        self.classification_zscore = classification_zscore
        self.johansen_confidence = johansen_confidence
        self.correlation_min_points = correlation_min_points

    def compute_pair_metrics(
        self,
        price1: pd.Series,
        price2: pd.Series,
    ) -> Dict[str, Optional[float]]:
        """
        Compute comprehensive quantitative metrics for a pair.
        
        Args:
            price1: First price series
            price2: Second price series
            
        Returns:
            Dictionary with all computed metrics
        """
        metrics: Dict[str, Optional[float]] = {
            "hedge_ratio": None,
            "adf_pvalue": None,
            "is_cointegrated": None,
            "half_life": None,
            "mean_zscore": None,
            "std_zscore": None,
            "skewness": None,
            "kurtosis": None,
            "current_zscore": None,
            "hurst_exponent": None,
            "spread_sharpe": None,
            "max_drawdown": None,
            "calmar_ratio": None,
            "johansen_trace_stat": None,
            "johansen_critical_value": None,
            "is_johansen_cointegrated": None,
            "kalman_hedge_ratio": None,
            "classification_f1": None,
            "classification_precision": None,
            "classification_recall": None,
            "classification_accuracy": None,
        }

        # Calculate hedge ratio
        hedge_ratio = calculate_ols_hedge_ratio(price1, price2)
        if hedge_ratio is None:
            return metrics

        # Calculate spread
        spread_series = price1 - hedge_ratio * price2
        metrics["hedge_ratio"] = hedge_ratio

        # ADF test
        adf_result = calculate_adf_test(spread_series, self.correlation_min_points)
        if adf_result:
            metrics["adf_pvalue"] = adf_result.get("adf_pvalue")
            metrics["is_cointegrated"] = (
                adf_result.get("adf_pvalue") is not None
                and adf_result["adf_pvalue"] < self.adf_pvalue_threshold
            )

        # Half-life
        half_life = calculate_half_life(spread_series)
        if half_life is not None:
            metrics["half_life"] = half_life

        # Z-score stats
        zscore_stats = calculate_zscore_stats(spread_series, self.zscore_lookback)
        metrics.update(zscore_stats)

        # Hurst exponent
        metrics["hurst_exponent"] = calculate_hurst_exponent(
            spread_series, self.zscore_lookback
        )

        # Risk metrics
        metrics["spread_sharpe"] = calculate_spread_sharpe(
            spread_series, self.periods_per_year
        )
        metrics["max_drawdown"] = calculate_max_drawdown(spread_series)
        metrics["calmar_ratio"] = calculate_calmar_ratio(
            spread_series, self.periods_per_year
        )

        # Johansen test
        johansen = calculate_johansen_test(
            price1,
            price2,
            self.correlation_min_points,
            self.johansen_confidence,
        )
        if johansen:
            metrics.update(johansen)

        # Kalman hedge ratio
        kalman_beta = calculate_kalman_hedge_ratio(price1, price2)
        if kalman_beta is not None:
            metrics["kalman_hedge_ratio"] = kalman_beta

        # Direction metrics
        direction_metrics = calculate_direction_metrics(
            spread_series, self.zscore_lookback, self.classification_zscore
        )
        metrics.update(direction_metrics)

        return metrics

