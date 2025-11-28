"""
Pair metrics computer that orchestrates all quantitative metrics calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union

from modules.pairs_trading.metrics import (
    calculate_adf_test,
    calculate_half_life,
    calculate_johansen_test,
    calculate_spread_sharpe,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_ols_hedge_ratio,
    calculate_kalman_hedge_ratio,
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
        PAIRS_TRADING_OLS_FIT_INTERCEPT,
        PAIRS_TRADING_KALMAN_DELTA,
        PAIRS_TRADING_KALMAN_OBS_COV,
    )
except ImportError:
    PAIRS_TRADING_TIMEFRAME = "1h"
    PAIRS_TRADING_ADF_PVALUE_THRESHOLD = 0.05
    PAIRS_TRADING_PERIODS_PER_YEAR = 365 * 24
    PAIRS_TRADING_ZSCORE_LOOKBACK = 60
    PAIRS_TRADING_CLASSIFICATION_ZSCORE = 0.5
    PAIRS_TRADING_JOHANSEN_CONFIDENCE = 0.95
    PAIRS_TRADING_CORRELATION_MIN_POINTS = 50
    PAIRS_TRADING_OLS_FIT_INTERCEPT = True
    PAIRS_TRADING_KALMAN_DELTA = 1e-5
    PAIRS_TRADING_KALMAN_OBS_COV = 1.0


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
        ols_fit_intercept: bool = PAIRS_TRADING_OLS_FIT_INTERCEPT,
        kalman_delta: float = PAIRS_TRADING_KALMAN_DELTA,
        kalman_obs_cov: float = PAIRS_TRADING_KALMAN_OBS_COV,
    ):
        """
        Initialize PairMetricsComputer.
        
        Args:
            adf_pvalue_threshold: P-value threshold for ADF test (must be in (0, 1])
            periods_per_year: Number of periods per year (must be > 0)
            zscore_lookback: Lookback period for z-score calculation (must be > 0)
            classification_zscore: Z-score threshold for classification (must be > 0)
            johansen_confidence: Confidence level for Johansen test (must be in (0, 1))
            correlation_min_points: Minimum data points required (must be >= 2)
            ols_fit_intercept: Whether to fit intercept in OLS regression
            kalman_delta: Kalman filter delta parameter (must be in (0, 1))
            kalman_obs_cov: Kalman filter observation covariance (must be > 0)
            
        Raises:
            ValueError: If parameter values are invalid
        """
        # Validate parameters
        if not (0 < adf_pvalue_threshold <= 1):
            raise ValueError(
                f"adf_pvalue_threshold must be in (0, 1], got {adf_pvalue_threshold}"
            )
        if periods_per_year <= 0:
            raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
        if zscore_lookback <= 0:
            raise ValueError(f"zscore_lookback must be positive, got {zscore_lookback}")
        if classification_zscore <= 0:
            raise ValueError(
                f"classification_zscore must be positive, got {classification_zscore}"
            )
        if not (0 < johansen_confidence < 1):
            raise ValueError(
                f"johansen_confidence must be in (0, 1), got {johansen_confidence}"
            )
        if correlation_min_points < 2:
            raise ValueError(
                f"correlation_min_points must be >= 2, got {correlation_min_points}"
            )
        if kalman_delta <= 0 or kalman_delta >= 1:
            raise ValueError(f"kalman_delta must be in (0, 1), got {kalman_delta}")
        if kalman_obs_cov <= 0:
            raise ValueError(f"kalman_obs_cov must be positive, got {kalman_obs_cov}")
        
        # Check for NaN/Inf
        if np.isnan(adf_pvalue_threshold) or np.isinf(adf_pvalue_threshold):
            raise ValueError(f"adf_pvalue_threshold must be finite, got {adf_pvalue_threshold}")
        if np.isnan(classification_zscore) or np.isinf(classification_zscore):
            raise ValueError(f"classification_zscore must be finite, got {classification_zscore}")
        if np.isnan(kalman_delta) or np.isinf(kalman_delta):
            raise ValueError(f"kalman_delta must be finite, got {kalman_delta}")
        if np.isnan(kalman_obs_cov) or np.isinf(kalman_obs_cov):
            raise ValueError(f"kalman_obs_cov must be finite, got {kalman_obs_cov}")
        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.periods_per_year = periods_per_year
        self.zscore_lookback = zscore_lookback
        self.classification_zscore = classification_zscore
        self.johansen_confidence = johansen_confidence
        self.correlation_min_points = correlation_min_points
        self.ols_fit_intercept = ols_fit_intercept
        self.kalman_delta = kalman_delta
        self.kalman_obs_cov = kalman_obs_cov

    def compute_pair_metrics(
        self,
        price1: pd.Series,
        price2: pd.Series,
    ) -> Dict[str, Optional[Union[float, bool]]]:
        """
        Compute comprehensive quantitative metrics for a pair.
        
        Calculates metrics for both OLS and Kalman hedge ratio methods:
        - OLS-based metrics: half_life, zscore stats, hurst, sharpe, etc. (based on static hedge ratio)
        - Kalman-based metrics: kalman_half_life, kalman_* metrics (based on dynamic hedge ratio)
        
        Note: ADF test and Johansen test are calculated once as they test cointegration
        between price1 and price2, independent of the hedge ratio method.
        
        Args:
            price1: First price series (pd.Series, must not be None/empty, will be validated)
            price2: Second price series (pd.Series, must not be None/empty, will be validated)
            
        Returns:
            Dictionary with all computed metrics (both OLS and Kalman-based).
            Returns metrics dict with all None values if inputs are invalid or calculation fails.
            
        Raises:
            ValueError: If price1 or price2 are None, empty, or invalid type
        """
        # Validate inputs
        if price1 is None or price2 is None:
            raise ValueError("price1 and price2 must not be None")
        
        if not isinstance(price1, pd.Series) or not isinstance(price2, pd.Series):
            raise ValueError(
                f"price1 and price2 must be pd.Series, got {type(price1)} and {type(price2)}"
            )
        
        if price1.empty or price2.empty:
            raise ValueError("price1 and price2 must not be empty")
        
        # Check for infinite values in inputs
        if np.isinf(price1).any() or np.isinf(price2).any():
            raise ValueError("price1 and price2 must not contain infinite values")
        metrics: Dict[str, Optional[Union[float, bool]]] = {
            # OLS-based metrics
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
            "classification_f1": None,
            "classification_precision": None,
            "classification_recall": None,
            "classification_accuracy": None,
            # Johansen test (independent of hedge ratio method)
            "johansen_trace_stat": None,
            "johansen_critical_value": None,
            "is_johansen_cointegrated": None,
            # Kalman hedge ratio
            "kalman_hedge_ratio": None,
            # Kalman-based metrics
            "kalman_half_life": None,
            "kalman_mean_zscore": None,
            "kalman_std_zscore": None,
            "kalman_skewness": None,
            "kalman_kurtosis": None,
            "kalman_current_zscore": None,
            "kalman_hurst_exponent": None,
            "kalman_spread_sharpe": None,
            "kalman_max_drawdown": None,
            "kalman_calmar_ratio": None,
            "kalman_classification_f1": None,
            "kalman_classification_precision": None,
            "kalman_classification_recall": None,
            "kalman_classification_accuracy": None,
        }

        # Calculate hedge ratio
        try:
            hedge_ratio = calculate_ols_hedge_ratio(
                price1,
                price2,
                fit_intercept=self.ols_fit_intercept,
            )
        except Exception:
            # If OLS calculation fails, return empty metrics
            return metrics
        
        if hedge_ratio is None or np.isnan(hedge_ratio) or np.isinf(hedge_ratio):
            return metrics

        # Calculate spread
        try:
            spread_series = price1 - hedge_ratio * price2
            # Validate spread_series
            if spread_series.empty or np.isinf(spread_series).all():
                return metrics
        except Exception:
            return metrics
        
        metrics["hedge_ratio"] = float(hedge_ratio)

        # ADF test
        try:
            adf_result = calculate_adf_test(spread_series, self.correlation_min_points)
            if adf_result:
                adf_pvalue = adf_result.get("adf_pvalue")
                if adf_pvalue is not None and not (np.isnan(adf_pvalue) or np.isinf(adf_pvalue)):
                    metrics["adf_pvalue"] = float(adf_pvalue)
                    metrics["is_cointegrated"] = adf_pvalue < self.adf_pvalue_threshold
        except Exception:
            # Continue if ADF test fails
            pass

        # Half-life
        try:
            half_life = calculate_half_life(spread_series)
            if half_life is not None and not (np.isnan(half_life) or np.isinf(half_life)):
                metrics["half_life"] = float(half_life)
        except Exception:
            pass

        # Z-score stats
        try:
            zscore_stats = calculate_zscore_stats(spread_series, self.zscore_lookback)
            if zscore_stats:
                # Filter out NaN/Inf values
                for key, value in zscore_stats.items():
                    if value is not None and not (np.isnan(value) or np.isinf(value)):
                        metrics[key] = float(value)
        except Exception:
            pass

        # Hurst exponent
        try:
            hurst = calculate_hurst_exponent(spread_series, self.zscore_lookback)
            if hurst is not None and not (np.isnan(hurst) or np.isinf(hurst)):
                metrics["hurst_exponent"] = float(hurst)
        except Exception:
            pass

        # Risk metrics
        # Calculate PnL and Equity for Risk Metrics (assuming 1 unit of spread traded)
        try:
            pnl_series = spread_series.diff()
            equity_curve = pnl_series.cumsum()
            
            # Validate series
            if not pnl_series.empty and not equity_curve.empty:
                sharpe = calculate_spread_sharpe(pnl_series, self.periods_per_year)
                if sharpe is not None and not (np.isnan(sharpe) or np.isinf(sharpe)):
                    metrics["spread_sharpe"] = float(sharpe)
                
                max_dd = calculate_max_drawdown(equity_curve)
                if max_dd is not None and not (np.isnan(max_dd) or np.isinf(max_dd)):
                    metrics["max_drawdown"] = float(max_dd)
                
                calmar = calculate_calmar_ratio(equity_curve, self.periods_per_year)
                if calmar is not None and not (np.isnan(calmar) or np.isinf(calmar)):
                    metrics["calmar_ratio"] = float(calmar)
        except Exception:
            pass

        # Johansen test
        try:
            johansen = calculate_johansen_test(
                price1,
                price2,
                self.correlation_min_points,
                self.johansen_confidence,
            )
            if johansen:
                # Filter out NaN/Inf values
                for key, value in johansen.items():
                    if value is not None:
                        if isinstance(value, bool):
                            metrics[key] = value
                        elif not (np.isnan(value) or np.isinf(value)):
                            metrics[key] = float(value) if isinstance(value, (int, float)) else value

                # Combine ADF and Johansen cointegration decisions.
                # Johansen is generally stronger, so we treat cointegration as True
                # if either test signals cointegration.
                adf_cointegrated = metrics.get("is_cointegrated")
                johansen_cointegrated = johansen.get("is_johansen_cointegrated")

                if johansen_cointegrated is not None:
                    if adf_cointegrated is None:
                        metrics["is_cointegrated"] = bool(johansen_cointegrated)
                    else:
                        metrics["is_cointegrated"] = bool(
                            adf_cointegrated or johansen_cointegrated
                        )
        except Exception:
            pass
        
        # Kalman hedge ratio and Kalman-based metrics
        try:
            kalman_beta = calculate_kalman_hedge_ratio(
                price1,
                price2,
                delta=self.kalman_delta,
                observation_covariance=self.kalman_obs_cov,
            )
        except Exception:
            kalman_beta = None
        
        if kalman_beta is not None and not (np.isnan(kalman_beta) or np.isinf(kalman_beta)):
            metrics["kalman_hedge_ratio"] = float(kalman_beta)
            
            # Calculate Kalman spread
            try:
                kalman_spread_series = price1 - kalman_beta * price2
                # Validate kalman_spread_series
                if kalman_spread_series.empty or np.isinf(kalman_spread_series).all():
                    return metrics
            except Exception:
                return metrics
            
            # Kalman half-life
            try:
                kalman_half_life = calculate_half_life(kalman_spread_series)
                if kalman_half_life is not None and not (np.isnan(kalman_half_life) or np.isinf(kalman_half_life)):
                    metrics["kalman_half_life"] = float(kalman_half_life)
            except Exception:
                pass
            
            # Kalman z-score stats
            try:
                kalman_zscore_stats = calculate_zscore_stats(
                    kalman_spread_series, self.zscore_lookback
                )
                if kalman_zscore_stats:
                    # Filter out NaN/Inf values
                    for key in ["mean_zscore", "std_zscore", "skewness", "kurtosis", "current_zscore"]:
                        value = kalman_zscore_stats.get(key)
                        if value is not None and not (np.isnan(value) or np.isinf(value)):
                            metrics[f"kalman_{key}"] = float(value)
            except Exception:
                pass
            
            # Kalman Hurst exponent
            try:
                kalman_hurst = calculate_hurst_exponent(
                    kalman_spread_series, self.zscore_lookback
                )
                if kalman_hurst is not None and not (np.isnan(kalman_hurst) or np.isinf(kalman_hurst)):
                    metrics["kalman_hurst_exponent"] = float(kalman_hurst)
            except Exception:
                pass
            
            # Kalman risk metrics
            # Calculate Kalman PnL and Equity
            try:
                kalman_pnl_series = kalman_spread_series.diff()
                kalman_equity_curve = kalman_pnl_series.cumsum()
                
                # Validate series
                if not kalman_pnl_series.empty and not kalman_equity_curve.empty:
                    kalman_sharpe = calculate_spread_sharpe(
                        kalman_pnl_series, self.periods_per_year
                    )
                    if kalman_sharpe is not None and not (np.isnan(kalman_sharpe) or np.isinf(kalman_sharpe)):
                        metrics["kalman_spread_sharpe"] = float(kalman_sharpe)
                    
                    kalman_max_dd = calculate_max_drawdown(kalman_equity_curve)
                    if kalman_max_dd is not None and not (np.isnan(kalman_max_dd) or np.isinf(kalman_max_dd)):
                        metrics["kalman_max_drawdown"] = float(kalman_max_dd)
                    
                    kalman_calmar = calculate_calmar_ratio(
                        kalman_equity_curve, self.periods_per_year
                    )
                    if kalman_calmar is not None and not (np.isnan(kalman_calmar) or np.isinf(kalman_calmar)):
                        metrics["kalman_calmar_ratio"] = float(kalman_calmar)
            except Exception:
                pass
            
            # Kalman direction metrics
            try:
                kalman_direction_metrics = calculate_direction_metrics(
                    kalman_spread_series, self.zscore_lookback, self.classification_zscore
                )
                if kalman_direction_metrics:
                    # Filter out NaN/Inf values and validate ranges
                    for key in ["classification_f1", "classification_precision", "classification_recall", "classification_accuracy"]:
                        value = kalman_direction_metrics.get(key)
                        if value is not None:
                            # Classification metrics should be in [0, 1]
                            if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                                if 0 <= value <= 1:
                                    metrics[f"kalman_{key}"] = float(value)
            except Exception:
                pass

        # Direction metrics (OLS-based)
        try:
            direction_metrics = calculate_direction_metrics(
                spread_series, self.zscore_lookback, self.classification_zscore
            )
            if direction_metrics:
                # Filter out NaN/Inf values and validate ranges
                for key, value in direction_metrics.items():
                    if value is not None:
                        # Classification metrics should be in [0, 1]
                        if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                            if key in ["classification_f1", "classification_precision", "classification_recall", "classification_accuracy"]:
                                if 0 <= value <= 1:
                                    metrics[key] = float(value)
                            else:
                                metrics[key] = float(value)
        except Exception:
            pass

        return metrics

