"""
Opportunity scoring logic for pairs trading.
"""

import numpy as np
from typing import Dict, Optional

try:
    from modules.config import (
        PAIRS_TRADING_ADF_PVALUE_THRESHOLD,
        PAIRS_TRADING_MAX_HALF_LIFE,
        PAIRS_TRADING_HURST_THRESHOLD,
        PAIRS_TRADING_MIN_SPREAD_SHARPE,
        PAIRS_TRADING_MAX_DRAWDOWN,
        PAIRS_TRADING_MIN_CALMAR,
        PAIRS_TRADING_OPPORTUNITY_PRESETS,
        PAIRS_TRADING_QUANTITATIVE_SCORE_WEIGHTS,
        PAIRS_TRADING_MOMENTUM_FILTERS,
    )
except ImportError:
    PAIRS_TRADING_ADF_PVALUE_THRESHOLD = 0.05
    PAIRS_TRADING_MAX_HALF_LIFE = 50
    PAIRS_TRADING_HURST_THRESHOLD = 0.5
    PAIRS_TRADING_MIN_SPREAD_SHARPE = 1.0
    PAIRS_TRADING_MAX_DRAWDOWN = 0.3
    PAIRS_TRADING_MIN_CALMAR = 1.0
    PAIRS_TRADING_OPPORTUNITY_PRESETS = {
        "balanced": {
            "corr_good_bonus": 1.20,
            "corr_low_penalty": 0.80,
            "corr_high_penalty": 0.90,
            "cointegration_bonus": 1.15,
            "weak_cointegration_bonus": 1.05,
            "half_life_bonus": 1.10,
            "zscore_divisor": 5.0,
            "zscore_cap": 0.20,
            "hurst_good_bonus": 1.08,
            "hurst_ok_bonus": 1.02,
            "hurst_ok_threshold": 0.60,
            "sharpe_good_bonus": 1.08,
            "sharpe_ok_bonus": 1.03,
            "maxdd_bonus": 1.05,
            "calmar_bonus": 1.05,
            "johansen_bonus": 1.08,
            "f1_high_bonus": 1.05,
            "f1_mid_bonus": 1.02,
        }
    }
    PAIRS_TRADING_QUANTITATIVE_SCORE_WEIGHTS = {
        "cointegration_full_weight": 30.0,
        "cointegration_weak_weight": 15.0,
        "cointegration_weak_pvalue_threshold": 0.1,
        "half_life_excellent_weight": 20.0,
        "half_life_good_weight": 10.0,
        "half_life_excellent_threshold": 20.0,
        "half_life_good_threshold": 50.0,
        "hurst_excellent_weight": 15.0,
        "hurst_good_weight": 8.0,
        "hurst_excellent_threshold": 0.4,
        "hurst_good_threshold": 0.5,
        "sharpe_excellent_weight": 15.0,
        "sharpe_good_weight": 8.0,
        "sharpe_excellent_threshold": 2.0,
        "sharpe_good_threshold": 1.0,
        "f1_excellent_weight": 10.0,
        "f1_good_weight": 5.0,
        "f1_excellent_threshold": 0.7,
        "f1_good_threshold": 0.6,
        "maxdd_excellent_weight": 10.0,
        "maxdd_good_weight": 5.0,
        "maxdd_excellent_threshold": 0.2,
        "maxdd_good_threshold": 0.3,
        "calmar_excellent_weight": 5.0,
        "calmar_good_weight": 2.5,
        "calmar_excellent_threshold": 1.0,
        "calmar_good_threshold": 0.5,
        "max_score": 100.0,
    }
    PAIRS_TRADING_MOMENTUM_FILTERS = {
        "min_adx": 18.0,
        "strong_adx": 25.0,
        "adx_base_bonus": 1.03,
        "adx_strong_bonus": 1.08,
        "low_corr_threshold": 0.30,
        "high_corr_threshold": 0.75,
        "low_corr_bonus": 1.05,
        "negative_corr_bonus": 1.10,
        "high_corr_penalty": 0.90,
    }


class OpportunityScorer:
    """Calculates opportunity scores for pairs trading opportunities."""

    def __init__(
        self,
        min_correlation: float = 0.3,
        max_correlation: float = 0.9,
        adf_pvalue_threshold: float = PAIRS_TRADING_ADF_PVALUE_THRESHOLD,
        max_half_life: float = PAIRS_TRADING_MAX_HALF_LIFE,
        hurst_threshold: float = PAIRS_TRADING_HURST_THRESHOLD,
        min_spread_sharpe: float = PAIRS_TRADING_MIN_SPREAD_SHARPE,
        max_drawdown_threshold: float = PAIRS_TRADING_MAX_DRAWDOWN,
        min_calmar: float = PAIRS_TRADING_MIN_CALMAR,
        scoring_multipliers: Optional[Dict[str, float]] = None,
        strategy: str = "reversion",
    ):
        """
        Initialize OpportunityScorer.
        
        Args:
            min_correlation: Minimum correlation threshold
            max_correlation: Maximum correlation threshold
            adf_pvalue_threshold: P-value threshold for ADF test
            max_half_life: Maximum acceptable half-life
            hurst_threshold: Hurst exponent threshold
            min_spread_sharpe: Minimum Sharpe ratio threshold
            max_drawdown_threshold: Maximum drawdown threshold
            min_calmar: Minimum Calmar ratio threshold
            scoring_multipliers: Optional dict to override default scoring multipliers
            strategy: Trading strategy ('reversion' or 'momentum')
            
        Raises:
            ValueError: If parameter values are invalid
        """
        # Validate parameters
        if not (-1 <= min_correlation <= 1):
            raise ValueError(f"min_correlation must be in [-1, 1], got {min_correlation}")
        if not (-1 <= max_correlation <= 1):
            raise ValueError(f"max_correlation must be in [-1, 1], got {max_correlation}")
        if min_correlation > max_correlation:
            raise ValueError(
                f"min_correlation ({min_correlation}) must be <= max_correlation ({max_correlation})"
            )
        if not (0 < adf_pvalue_threshold <= 1):
            raise ValueError(
                f"adf_pvalue_threshold must be in (0, 1], got {adf_pvalue_threshold}"
            )
        if max_half_life <= 0:
            raise ValueError(f"max_half_life must be positive, got {max_half_life}")
        if not (0 < hurst_threshold <= 1):
            raise ValueError(f"hurst_threshold must be in (0, 1], got {hurst_threshold}")
        if np.isnan(min_spread_sharpe) or np.isinf(min_spread_sharpe):
            raise ValueError(f"min_spread_sharpe must be finite, got {min_spread_sharpe}")
        if max_drawdown_threshold <= 0 or max_drawdown_threshold > 1:
            raise ValueError(
                f"max_drawdown_threshold must be in (0, 1], got {max_drawdown_threshold}"
            )
        if np.isnan(min_calmar) or np.isinf(min_calmar) or min_calmar < 0:
            raise ValueError(f"min_calmar must be finite and non-negative, got {min_calmar}")
        if strategy not in ["reversion", "momentum"]:
            raise ValueError(f"strategy must be 'reversion' or 'momentum', got {strategy}")
        
        # Validate scoring_multipliers if provided
        # Note: Some keys like 'description' and 'hedge_ratio_strategy' are metadata, not multipliers
        if scoring_multipliers is not None:
            # Keys that are metadata, not numeric multipliers
            metadata_keys = {'description', 'hedge_ratio_strategy'}
            
            for key, value in scoring_multipliers.items():
                # Skip metadata keys
                if key in metadata_keys:
                    continue
                
                # Validate numeric multipliers
                if not isinstance(value, (int, float)):
                    raise ValueError(f"scoring_multipliers[{key}] must be numeric, got {type(value)}")
                if np.isnan(value) or np.isinf(value):
                    raise ValueError(f"scoring_multipliers[{key}] must be finite, got {value}")
        self.min_correlation = min_correlation
        self.max_correlation = max_correlation
        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.max_half_life = max_half_life
        self.hurst_threshold = hurst_threshold
        self.min_spread_sharpe = min_spread_sharpe
        self.max_drawdown_threshold = max_drawdown_threshold
        self.min_calmar = min_calmar
        self.strategy = strategy
        default_profile = PAIRS_TRADING_OPPORTUNITY_PRESETS.get("balanced", {})
        self.scoring = {**default_profile, **(scoring_multipliers or {})}
        self.quant_weights = PAIRS_TRADING_QUANTITATIVE_SCORE_WEIGHTS.copy()
        # Strategy: 'ols', 'kalman', 'best' (use best of both), 'avg' (average of both)
        self.hedge_ratio_strategy = self.scoring.get("hedge_ratio_strategy", "best")
        self.momentum_filters = PAIRS_TRADING_MOMENTUM_FILTERS.copy()

    def _get_metric(
        self, 
        quant_metrics: Dict[str, Optional[float]], 
        ols_key: str, 
        kalman_key: str, 
        strategy: Optional[str] = None
    ) -> Optional[float]:
        """
        Get metric value using specified strategy (OLS, Kalman, best, or average).
        
        Args:
            quant_metrics: Quantitative metrics dictionary
            ols_key: OLS metric key (e.g., 'half_life')
            kalman_key: Kalman metric key (e.g., 'kalman_half_life')
            strategy: Strategy to use ('ols', 'kalman', 'best', 'avg'). Defaults to instance strategy.
            
        Returns:
            Metric value or None if not available or invalid (NaN/Inf)
        """
        if strategy is None:
            strategy = self.hedge_ratio_strategy
        
        ols_val = quant_metrics.get(ols_key)
        kalman_val = quant_metrics.get(kalman_key)
        
        # Filter out NaN and Inf values
        if ols_val is not None and (np.isnan(ols_val) or np.isinf(ols_val)):
            ols_val = None
        if kalman_val is not None and (np.isnan(kalman_val) or np.isinf(kalman_val)):
            kalman_val = None
        
        if strategy == "ols":
            return ols_val
        elif strategy == "kalman":
            return kalman_val if kalman_val is not None else ols_val
        elif strategy == "best":
            # Use Kalman if available and better, otherwise OLS
            if kalman_val is not None and ols_val is not None:
                # For metrics where lower is better (half_life, max_drawdown), use min
                # For metrics where higher is better (sharpe, calmar), use max
                if "half_life" in ols_key or "drawdown" in ols_key:
                    return min(ols_val, kalman_val)
                else:
                    return max(ols_val, kalman_val)
            return kalman_val if kalman_val is not None else ols_val
        elif strategy == "avg":
            if ols_val is not None and kalman_val is not None:
                avg_val = (ols_val + kalman_val) / 2.0
                # Validate result
                if np.isnan(avg_val) or np.isinf(avg_val):
                    return None
                return avg_val
            return kalman_val if kalman_val is not None else ols_val
        else:
            # Default: prefer Kalman, fallback to OLS
            return kalman_val if kalman_val is not None else ols_val

    def calculate_opportunity_score(
        self,
        spread: float,
        correlation: Optional[float] = None,
        quant_metrics: Optional[Dict[str, Optional[float]]] = None,
    ) -> float:
        """
        Calculate opportunity score for a trading pair.
        
        Uses both OLS and Kalman metrics when available. The strategy for selecting
        between OLS and Kalman metrics is controlled by `hedge_ratio_strategy`:
        - 'best': Uses the better value (min for half_life/drawdown, max for sharpe/calmar)
        - 'kalman': Prefers Kalman, falls back to OLS
        - 'ols': Uses only OLS metrics
        - 'avg': Uses average of both when available
        
        Args:
            spread: Spread between long and short symbols (must be >= 0)
            correlation: Correlation coefficient (optional, must be in [-1, 1] if provided)
            quant_metrics: Quantitative metrics dictionary (optional, includes both OLS and Kalman metrics)
            
        Returns:
            Opportunity score (higher is better, always >= 0)
            
        Raises:
            ValueError: If spread is negative, NaN, or Inf, or if correlation is out of range
        """
        # Validate inputs
        if np.isnan(spread) or np.isinf(spread):
            raise ValueError(f"spread must be finite, got {spread}")
        if spread < 0:
            raise ValueError(f"spread must be non-negative, got {spread}")
        
        if correlation is not None:
            if np.isnan(correlation) or np.isinf(correlation):
                raise ValueError(f"correlation must be finite, got {correlation}")
            if not (-1 <= correlation <= 1):
                raise ValueError(f"correlation must be in [-1, 1], got {correlation}")
        
        if quant_metrics is None:
            quant_metrics = {}

        # Start with base spread
        # Spread is the performance gap between the short and long legs:
        # spread = short_score - long_score (absolute). Large spread implies
        # a stronger mean-reversion opportunity, so it becomes the base score.
        opportunity_score = float(spread)

        # Adjust based on correlation
        sc = self.scoring

        if correlation is not None:
            if self.strategy == "momentum":
                opportunity_score = self._apply_momentum_correlation(
                    opportunity_score, correlation
                )
            else:
                abs_corr = abs(correlation)
                if self.min_correlation <= abs_corr <= self.max_correlation:
                    opportunity_score *= sc.get("corr_good_bonus", 1.2)
                elif abs_corr < self.min_correlation:
                    opportunity_score *= sc.get("corr_low_penalty", 0.8)
                elif abs_corr > self.max_correlation:
                    opportunity_score *= sc.get("corr_high_penalty", 0.9)

        if self.strategy == "momentum":
            # --- MOMENTUM SCORING LOGIC ---
            
            # 1. Hurst Exponent: Reward trending behavior (Hurst > 0.5)
            hurst = self._get_metric(quant_metrics, "hurst_exponent", "kalman_hurst_exponent")
            if hurst is not None:
                if hurst > 0.5:
                    opportunity_score *= sc.get("hurst_good_bonus", 1.08)
            
            # 2. Z-Score: Reward divergence (High absolute Z-score) with capped bonus
            current_z = self._get_metric(quant_metrics, "current_zscore", "kalman_current_zscore")
            if current_z is not None and not np.isnan(current_z):
                abs_z = abs(current_z)
                if abs_z > 2.0:
                    opportunity_score *= 1.15
                elif abs_z > 1.0:
                    opportunity_score *= 1.08

            # 3. ADX filter: require both legs to trend strongly
            long_adx = quant_metrics.get("long_adx")
            short_adx = quant_metrics.get("short_adx")
            opportunity_score = self._apply_momentum_adx_filter(
                opportunity_score, long_adx, short_adx
            )
            if opportunity_score == 0:
                return 0.0

            # 4. Cointegration penalty (momentum prefers divergence)
            if quant_metrics.get("is_cointegrated") or quant_metrics.get("is_johansen_cointegrated"):
                opportunity_score *= sc.get("momentum_cointegration_penalty", 0.95)
            
        else:
            # --- MEAN REVERSION SCORING LOGIC (Original) ---

            # Boost score if cointegrated and half-life within acceptable range
            if quant_metrics.get("is_cointegrated"):
                opportunity_score *= sc.get("cointegration_bonus", 1.15)
            elif quant_metrics.get("adf_pvalue") is not None and quant_metrics["adf_pvalue"] < (
                self.adf_pvalue_threshold * 1.5
            ):
                opportunity_score *= sc.get("weak_cointegration_bonus", 1.05)

            # Use Kalman metrics when available (integrated with OLS)
            half_life = self._get_metric(quant_metrics, "half_life", "kalman_half_life")
            if half_life is not None and half_life <= self.max_half_life:
                opportunity_score *= sc.get("half_life_bonus", 1.1)

            current_z = self._get_metric(quant_metrics, "current_zscore", "kalman_current_zscore")
            if current_z is not None and not np.isnan(current_z):
                z_div = sc.get("zscore_divisor", 5.0)
                z_cap = sc.get("zscore_cap", 0.2)
                opportunity_score *= 1 + min(abs(current_z) / max(z_div, 1e-6), z_cap)

            hurst = self._get_metric(quant_metrics, "hurst_exponent", "kalman_hurst_exponent")
            if hurst is not None:
                if hurst <= self.hurst_threshold:
                    opportunity_score *= sc.get("hurst_good_bonus", 1.08)
                elif hurst < sc.get("hurst_ok_threshold", 0.6):
                    opportunity_score *= sc.get("hurst_ok_bonus", 1.02)

        if self.strategy == "momentum":
            opportunity_score = self._apply_momentum_risk_adjustments(
                opportunity_score, quant_metrics
            )
        else:
            sharpe = self._get_metric(quant_metrics, "spread_sharpe", "kalman_spread_sharpe")
            if sharpe is not None:
                if sharpe >= self.min_spread_sharpe:
                    opportunity_score *= sc.get("sharpe_good_bonus", 1.08)
                elif sharpe >= self.min_spread_sharpe / 2:
                    opportunity_score *= sc.get("sharpe_ok_bonus", 1.03)

            max_dd = self._get_metric(quant_metrics, "max_drawdown", "kalman_max_drawdown")
            if max_dd is not None and abs(max_dd) <= self.max_drawdown_threshold:
                opportunity_score *= sc.get("maxdd_bonus", 1.05)

            calmar = self._get_metric(quant_metrics, "calmar_ratio", "kalman_calmar_ratio")
            if calmar is not None and calmar >= self.min_calmar:
                opportunity_score *= sc.get("calmar_bonus", 1.05)

            if quant_metrics.get("is_johansen_cointegrated"):
                opportunity_score *= sc.get("johansen_bonus", 1.08)

            f1_metric = self._get_metric(quant_metrics, "classification_f1", "kalman_classification_f1")
            if f1_metric is not None:
                # Validate F1 is in valid range [0, 1]
                if not (0 <= f1_metric <= 1):
                    # Skip invalid F1 values
                    pass
                elif f1_metric >= 0.7:
                    opportunity_score *= sc.get("f1_high_bonus", 1.05)
                elif f1_metric >= 0.6:
                    opportunity_score *= sc.get("f1_mid_bonus", 1.02)

        # Validate final result
        if np.isnan(opportunity_score) or np.isinf(opportunity_score):
            return 0.0
        
        # Ensure non-negative
        return max(0.0, float(opportunity_score))

    def _apply_momentum_correlation(self, score: float, correlation: float) -> float:
        """
        Adjust score for momentum strategy based on correlation characteristics.
        
        Args:
            score: Current opportunity score
            correlation: Correlation coefficient (should be in [-1, 1])
            
        Returns:
            Adjusted score
        """
        # Validate inputs
        if np.isnan(score) or np.isinf(score):
            return 0.0
        if np.isnan(correlation) or np.isinf(correlation):
            return score
        
        mf = self.momentum_filters
        abs_corr = abs(correlation)

        if correlation < 0:
            score *= mf.get("negative_corr_bonus", 1.0)
        elif abs_corr < mf.get("low_corr_threshold", 0.3):
            score *= mf.get("low_corr_bonus", 1.0)
        elif abs_corr > mf.get("high_corr_threshold", 0.75):
            score *= mf.get("high_corr_penalty", 1.0)

        # Validate result
        if np.isnan(score) or np.isinf(score):
            return 0.0
        return max(0.0, score)

    def _apply_momentum_adx_filter(
        self,
        score: float,
        long_adx: Optional[float],
        short_adx: Optional[float],
    ) -> float:
        """
        Require both legs to have sufficient ADX strength for momentum setups.
        
        Args:
            score: Current opportunity score
            long_adx: ADX value for long symbol (optional)
            short_adx: ADX value for short symbol (optional)
            
        Returns:
            Adjusted score (0.0 if ADX requirements not met)
        """
        # Validate score
        if np.isnan(score) or np.isinf(score):
            return 0.0
        
        if long_adx is None or short_adx is None:
            return score

        # Filter out NaN/Inf ADX values
        if np.isnan(long_adx) or np.isinf(long_adx) or np.isnan(short_adx) or np.isinf(short_adx):
            return 0.0

        mf = self.momentum_filters
        min_adx = mf.get("min_adx", 18.0)
        strong_adx = mf.get("strong_adx", 25.0)

        if long_adx < min_adx or short_adx < min_adx:
            return 0.0

        score *= mf.get("adx_base_bonus", 1.0)

        if long_adx >= strong_adx and short_adx >= strong_adx:
            score *= mf.get("adx_strong_bonus", 1.0)

        # Validate result
        if np.isnan(score) or np.isinf(score):
            return 0.0
        return max(0.0, score)

    def _apply_momentum_risk_adjustments(
        self,
        score: float,
        quant_metrics: Dict[str, Optional[float]],
    ) -> float:
        """
        Apply lightweight risk checks for momentum strategy.
        
        Args:
            score: Current opportunity score
            quant_metrics: Quantitative metrics dictionary
            
        Returns:
            Adjusted score
        """
        # Validate score
        if np.isnan(score) or np.isinf(score):
            return 0.0
        
        sharpe = self._get_metric(quant_metrics, "spread_sharpe", "kalman_spread_sharpe")
        if sharpe is not None and sharpe < (self.min_spread_sharpe / 2):
            score *= self.scoring.get("momentum_low_sharpe_penalty", 0.97)

        max_dd = self._get_metric(quant_metrics, "max_drawdown", "kalman_max_drawdown")
        if max_dd is not None and abs(max_dd) > self.max_drawdown_threshold * 1.5:
            score *= self.scoring.get("momentum_maxdd_penalty", 0.97)

        calmar = self._get_metric(quant_metrics, "calmar_ratio", "kalman_calmar_ratio")
        if calmar is not None and calmar < max(self.min_calmar / 2, 0.1):
            score *= self.scoring.get("momentum_calmar_penalty", 0.98)

        # Validate result
        if np.isnan(score) or np.isinf(score):
            return 0.0
        return max(0.0, score)

    def calculate_quantitative_score(
        self, quant_metrics: Optional[Dict[str, Optional[float]]] = None
    ) -> float:
        """
        Calculate combined quantitative score (0-100) based on all metrics.
        
        Uses both OLS and Kalman metrics when available, following the same strategy
        as calculate_opportunity_score (controlled by hedge_ratio_strategy).
        
        Default weights (configurable via PAIRS_TRADING_QUANTITATIVE_SCORE_WEIGHTS):
        - Cointegration: 30% (full), 15% (weak if pvalue < 0.1)
        - Half-life: 20% (< 20 periods), 10% (< 50 periods)
        - Hurst: 15% (< 0.4), 8% (< 0.5)
        - Sharpe: 15% (> 2.0), 8% (> 1.0)
        - F1-score: 10% (> 0.7), 5% (> 0.6)
        - Max DD: 10% (< 0.2), 5% (< 0.3)
        - Calmar ratio: 5% (>= 1.0), 2.5% (>= 0.5)
        - Overall score capped at 100
        
        All weights and thresholds can be customized via config without code changes.
        
        Args:
            quant_metrics: Quantitative metrics dictionary (includes both OLS and Kalman metrics)
            
        Returns:
            Quantitative score from 0-100 (higher is better)
        """
        if quant_metrics is None:
            quant_metrics = {}
        
        score = 0.0
        w = self.quant_weights
        
        # Cointegration
        if quant_metrics.get("is_cointegrated"):
            score += w.get("cointegration_full_weight", 30.0)
        elif quant_metrics.get("is_johansen_cointegrated"):
            score += w.get("cointegration_full_weight", 30.0)
        else:
            adf_pvalue = quant_metrics.get("adf_pvalue")
            weak_threshold = w.get("cointegration_weak_pvalue_threshold", 0.1)
            if adf_pvalue is not None and adf_pvalue < weak_threshold:
                score += w.get("cointegration_weak_weight", 15.0)
        
        # Half-life (uses Kalman if available per strategy)
        half_life = self._get_metric(quant_metrics, "half_life", "kalman_half_life")
        if half_life is not None:
            excellent_threshold = w.get("half_life_excellent_threshold", 20.0)
            good_threshold = w.get("half_life_good_threshold", 50.0)
            if half_life < excellent_threshold:
                score += w.get("half_life_excellent_weight", 20.0)
            elif half_life < good_threshold:
                score += w.get("half_life_good_weight", 10.0)
        
        # Hurst (uses Kalman if available per strategy)
        hurst = self._get_metric(quant_metrics, "hurst_exponent", "kalman_hurst_exponent")
        if hurst is not None:
            excellent_threshold = w.get("hurst_excellent_threshold", 0.4)
            good_threshold = w.get("hurst_good_threshold", 0.5)
            if hurst < excellent_threshold:
                score += w.get("hurst_excellent_weight", 15.0)
            elif hurst < good_threshold:
                score += w.get("hurst_good_weight", 8.0)
        
        # Sharpe (uses Kalman if available per strategy)
        sharpe = self._get_metric(quant_metrics, "spread_sharpe", "kalman_spread_sharpe")
        if sharpe is not None:
            excellent_threshold = w.get("sharpe_excellent_threshold", 2.0)
            good_threshold = w.get("sharpe_good_threshold", 1.0)
            if sharpe > excellent_threshold:
                score += w.get("sharpe_excellent_weight", 15.0)
            elif sharpe > good_threshold:
                score += w.get("sharpe_good_weight", 8.0)
        
        # F1-score (uses Kalman if available per strategy)
        f1 = self._get_metric(quant_metrics, "classification_f1", "kalman_classification_f1")
        if f1 is not None:
            # Validate F1 is in valid range [0, 1]
            if 0 <= f1 <= 1:
                excellent_threshold = w.get("f1_excellent_threshold", 0.7)
                good_threshold = w.get("f1_good_threshold", 0.6)
                if f1 > excellent_threshold:
                    score += w.get("f1_excellent_weight", 10.0)
                elif f1 > good_threshold:
                    score += w.get("f1_good_weight", 5.0)
        
        # Max DD (uses Kalman if available per strategy)
        max_dd = self._get_metric(quant_metrics, "max_drawdown", "kalman_max_drawdown")
        if max_dd is not None:
            abs_max_dd = abs(max_dd)
            excellent_threshold = w.get("maxdd_excellent_threshold", 0.2)
            good_threshold = w.get("maxdd_good_threshold", 0.3)
            if abs_max_dd < excellent_threshold:
                score += w.get("maxdd_excellent_weight", 10.0)
            elif abs_max_dd < good_threshold:
                score += w.get("maxdd_good_weight", 5.0)
        
        # Calmar ratio (uses Kalman if available per strategy)
        calmar_ratio = self._get_metric(quant_metrics, "calmar_ratio", "kalman_calmar_ratio")
        if calmar_ratio is not None:
            excellent_threshold = w.get("calmar_excellent_threshold", 1.0)
            good_threshold = w.get("calmar_good_threshold", 0.5)
            if calmar_ratio >= excellent_threshold:
                score += w.get("calmar_excellent_weight", 5.0)
            elif calmar_ratio >= good_threshold:
                score += w.get("calmar_good_weight", 2.5)

        # Momentum-specific ADX contribution
        if self.strategy == "momentum":
            long_adx = quant_metrics.get("long_adx")
            short_adx = quant_metrics.get("short_adx")
            if long_adx is not None and short_adx is not None:
                # Validate ADX values
                if not (np.isnan(long_adx) or np.isinf(long_adx) or np.isnan(short_adx) or np.isinf(short_adx)):
                    avg_adx = (long_adx + short_adx) / 2.0
                    if not (np.isnan(avg_adx) or np.isinf(avg_adx)):
                        if avg_adx >= self.momentum_filters.get("strong_adx", 25.0):
                            score += w.get("momentum_adx_strong_weight", 10.0)
                        elif avg_adx >= self.momentum_filters.get("min_adx", 18.0):
                            score += w.get("momentum_adx_moderate_weight", 5.0)

        max_score = w.get("max_score", 100.0)
        # Validate final score
        if np.isnan(score) or np.isinf(score):
            return 0.0
        return min(max_score, max(0.0, score))