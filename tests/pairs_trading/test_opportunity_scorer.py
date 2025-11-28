import math
import pytest
import numpy as np

from modules.pairs_trading.core.opportunity_scorer import OpportunityScorer


def test_calculate_opportunity_score_applies_all_adjustments():
    scorer = OpportunityScorer(
        min_correlation=0.3,
        max_correlation=0.9,
        adf_pvalue_threshold=0.05,
        max_half_life=50,
        hurst_threshold=0.5,
        min_spread_sharpe=1.0,
        max_drawdown_threshold=0.3,
        min_calmar=1.0,
    )

    quant_metrics = {
        "is_cointegrated": True,
        "half_life": 25,
        "current_zscore": 1.5,
        "hurst_exponent": 0.45,
        "spread_sharpe": 1.2,
        "max_drawdown": -0.2,
        "calmar_ratio": 1.3,
        "is_johansen_cointegrated": True,
        "classification_f1": 0.75,
    }

    score = scorer.calculate_opportunity_score(0.1, correlation=0.6, quant_metrics=quant_metrics)

    assert score > 0.1 * 1.2  # correlation boost
    assert score > 0.1  # definitely boosted overall


def test_calculate_opportunity_score_penalizes_low_correlation():
    scorer = OpportunityScorer(min_correlation=0.4, max_correlation=0.9)
    score = scorer.calculate_opportunity_score(0.2, correlation=0.1, quant_metrics={})
    assert math.isclose(score, 0.2 * 0.8)


def test_calculate_quantitative_score_combines_metrics():
    scorer = OpportunityScorer()

    metrics = {
        "is_cointegrated": True,
        "half_life": 10,
        "hurst_exponent": 0.35,
        "spread_sharpe": 2.5,
        "classification_f1": 0.8,
        "max_drawdown": -0.1,
    }

    result = scorer.calculate_quantitative_score(metrics)

    # Expect full credit for each bucket
    assert math.isclose(result, 30 + 20 + 15 + 15 + 10 + 10)


def test_momentum_correlation_and_adx_filters():
    scorer = OpportunityScorer(strategy="momentum")

    strong_metrics = {"long_adx": 28.0, "short_adx": 30.0}
    weak_metrics = {"long_adx": 10.0, "short_adx": 12.0}

    strong_score = scorer.calculate_opportunity_score(
        spread=0.2,
        correlation=-0.2,
        quant_metrics=strong_metrics,
    )
    weak_score = scorer.calculate_opportunity_score(
        spread=0.2,
        correlation=0.8,
        quant_metrics=weak_metrics,
    )

    assert strong_score > 0.2  # negative correlation + ADX bonuses
    assert weak_score == 0.0  # fails ADX filter


def test_momentum_high_correlation_penalty():
    scorer = OpportunityScorer(strategy="momentum")
    metrics = {"long_adx": 26.0, "short_adx": 27.0}

    low_corr_score = scorer.calculate_opportunity_score(
        spread=0.2,
        correlation=0.1,
        quant_metrics=metrics,
    )
    high_corr_score = scorer.calculate_opportunity_score(
        spread=0.2,
        correlation=0.95,
        quant_metrics=metrics,
    )

    assert high_corr_score < low_corr_score


def test_momentum_cointegration_penalty():
    scorer = OpportunityScorer(strategy="momentum")
    base_metrics = {"long_adx": 26.0, "short_adx": 27.0}
    coint_metrics = {**base_metrics, "is_cointegrated": True}

    base_score = scorer.calculate_opportunity_score(0.3, 0.2, base_metrics)
    penalized_score = scorer.calculate_opportunity_score(0.3, 0.2, coint_metrics)

    assert penalized_score < base_score


def test_momentum_risk_penalties_only_for_bad_cases():
    scorer = OpportunityScorer(strategy="momentum")
    good_metrics = {"long_adx": 26.0, "short_adx": 27.0, "spread_sharpe": 1.5}
    bad_metrics = {**good_metrics, "spread_sharpe": 0.1, "kalman_spread_sharpe": 0.1}

    good_score = scorer.calculate_opportunity_score(0.25, 0.2, good_metrics)
    bad_score = scorer.calculate_opportunity_score(0.25, 0.2, bad_metrics)

    assert bad_score < good_score


def test_quantitative_score_adds_momentum_adx_bonus():
    scorer = OpportunityScorer(strategy="momentum")
    strong_trend = {"long_adx": 30.0, "short_adx": 28.0}
    moderate_trend = {"long_adx": 20.0, "short_adx": 19.0}

    strong_score = scorer.calculate_quantitative_score(strong_trend)
    moderate_score = scorer.calculate_quantitative_score(moderate_trend)

    assert strong_score >= 10.0
    assert moderate_score >= 5.0
    assert strong_score > moderate_score


# ============================================================================
# __init__ Validation Tests
# ============================================================================

def test_init_invalid_min_correlation_out_of_range():
    with pytest.raises(ValueError, match="min_correlation must be in"):
        OpportunityScorer(min_correlation=-1.5)
    
    with pytest.raises(ValueError, match="min_correlation must be in"):
        OpportunityScorer(min_correlation=1.5)


def test_init_invalid_max_correlation_out_of_range():
    with pytest.raises(ValueError, match="max_correlation must be in"):
        OpportunityScorer(max_correlation=-1.5)
    
    with pytest.raises(ValueError, match="max_correlation must be in"):
        OpportunityScorer(max_correlation=1.5)


def test_init_min_correlation_greater_than_max():
    with pytest.raises(ValueError, match="min_correlation.*must be <= max_correlation"):
        OpportunityScorer(min_correlation=0.8, max_correlation=0.5)


def test_init_invalid_adf_pvalue_threshold():
    with pytest.raises(ValueError, match="adf_pvalue_threshold must be in"):
        OpportunityScorer(adf_pvalue_threshold=0.0)
    
    with pytest.raises(ValueError, match="adf_pvalue_threshold must be in"):
        OpportunityScorer(adf_pvalue_threshold=1.5)


def test_init_invalid_max_half_life():
    with pytest.raises(ValueError, match="max_half_life must be positive"):
        OpportunityScorer(max_half_life=0)
    
    with pytest.raises(ValueError, match="max_half_life must be positive"):
        OpportunityScorer(max_half_life=-10)


def test_init_invalid_hurst_threshold():
    with pytest.raises(ValueError, match="hurst_threshold must be in"):
        OpportunityScorer(hurst_threshold=0.0)
    
    with pytest.raises(ValueError, match="hurst_threshold must be in"):
        OpportunityScorer(hurst_threshold=1.5)


def test_init_invalid_min_spread_sharpe():
    with pytest.raises(ValueError, match="min_spread_sharpe must be finite"):
        OpportunityScorer(min_spread_sharpe=np.nan)
    
    with pytest.raises(ValueError, match="min_spread_sharpe must be finite"):
        OpportunityScorer(min_spread_sharpe=np.inf)


def test_init_invalid_max_drawdown_threshold():
    with pytest.raises(ValueError, match="max_drawdown_threshold must be in"):
        OpportunityScorer(max_drawdown_threshold=0.0)
    
    with pytest.raises(ValueError, match="max_drawdown_threshold must be in"):
        OpportunityScorer(max_drawdown_threshold=1.5)


def test_init_invalid_min_calmar():
    with pytest.raises(ValueError, match="min_calmar must be finite"):
        OpportunityScorer(min_calmar=np.nan)
    
    with pytest.raises(ValueError, match="min_calmar must be finite"):
        OpportunityScorer(min_calmar=np.inf)
    
    with pytest.raises(ValueError, match="min_calmar must be finite"):
        OpportunityScorer(min_calmar=-1.0)


def test_init_invalid_strategy():
    with pytest.raises(ValueError, match="strategy must be 'reversion' or 'momentum'"):
        OpportunityScorer(strategy="invalid")


def test_init_invalid_scoring_multipliers_non_numeric():
    with pytest.raises(ValueError, match="scoring_multipliers.*must be numeric"):
        OpportunityScorer(scoring_multipliers={"corr_good_bonus": "invalid"})


def test_init_invalid_scoring_multipliers_nan():
    with pytest.raises(ValueError, match="scoring_multipliers.*must be finite"):
        OpportunityScorer(scoring_multipliers={"corr_good_bonus": np.nan})


def test_init_invalid_scoring_multipliers_inf():
    with pytest.raises(ValueError, match="scoring_multipliers.*must be finite"):
        OpportunityScorer(scoring_multipliers={"corr_good_bonus": np.inf})


def test_init_valid_scoring_multipliers_with_metadata():
    # Should not raise error - metadata keys are skipped
    scorer = OpportunityScorer(
        scoring_multipliers={
            "description": "Test preset",
            "hedge_ratio_strategy": "best",
            "corr_good_bonus": 1.25,
        }
    )
    assert scorer.scoring["corr_good_bonus"] == 1.25
    assert scorer.scoring["description"] == "Test preset"
    assert scorer.scoring["hedge_ratio_strategy"] == "best"


# ============================================================================
# calculate_opportunity_score Validation Tests
# ============================================================================

def test_calculate_opportunity_score_invalid_spread_negative():
    scorer = OpportunityScorer()
    with pytest.raises(ValueError, match="spread must be non-negative"):
        scorer.calculate_opportunity_score(spread=-0.1)


def test_calculate_opportunity_score_invalid_spread_nan():
    scorer = OpportunityScorer()
    with pytest.raises(ValueError, match="spread must be finite"):
        scorer.calculate_opportunity_score(spread=np.nan)


def test_calculate_opportunity_score_invalid_spread_inf():
    scorer = OpportunityScorer()
    with pytest.raises(ValueError, match="spread must be finite"):
        scorer.calculate_opportunity_score(spread=np.inf)


def test_calculate_opportunity_score_invalid_correlation_out_of_range():
    scorer = OpportunityScorer()
    with pytest.raises(ValueError, match="correlation must be in"):
        scorer.calculate_opportunity_score(spread=0.1, correlation=-1.5)
    
    with pytest.raises(ValueError, match="correlation must be in"):
        scorer.calculate_opportunity_score(spread=0.1, correlation=1.5)


def test_calculate_opportunity_score_invalid_correlation_nan():
    scorer = OpportunityScorer()
    with pytest.raises(ValueError, match="correlation must be finite"):
        scorer.calculate_opportunity_score(spread=0.1, correlation=np.nan)


def test_calculate_opportunity_score_invalid_correlation_inf():
    scorer = OpportunityScorer()
    with pytest.raises(ValueError, match="correlation must be finite"):
        scorer.calculate_opportunity_score(spread=0.1, correlation=np.inf)


def test_calculate_opportunity_score_zero_spread():
    scorer = OpportunityScorer()
    score = scorer.calculate_opportunity_score(spread=0.0)
    assert score >= 0.0
    assert not (np.isnan(score) or np.isinf(score))


def test_calculate_opportunity_score_none_correlation():
    scorer = OpportunityScorer()
    score = scorer.calculate_opportunity_score(spread=0.1, correlation=None)
    assert score >= 0.0
    assert not (np.isnan(score) or np.isinf(score))


def test_calculate_opportunity_score_handles_nan_in_quant_metrics():
    scorer = OpportunityScorer()
    quant_metrics = {
        "spread_sharpe": np.nan,
        "half_life": np.inf,
        "hurst_exponent": 0.4,
    }
    score = scorer.calculate_opportunity_score(spread=0.1, quant_metrics=quant_metrics)
    assert score >= 0.0
    assert not (np.isnan(score) or np.isinf(score))


# ============================================================================
# calculate_quantitative_score Validation Tests
# ============================================================================

def test_calculate_quantitative_score_empty_metrics():
    scorer = OpportunityScorer()
    score = scorer.calculate_quantitative_score({})
    assert score >= 0.0
    assert score <= 100.0
    assert not (np.isnan(score) or np.isinf(score))


def test_calculate_quantitative_score_none_metrics():
    scorer = OpportunityScorer()
    score = scorer.calculate_quantitative_score(None)
    assert score >= 0.0
    assert score <= 100.0
    assert not (np.isnan(score) or np.isinf(score))


def test_calculate_quantitative_score_handles_nan_in_metrics():
    scorer = OpportunityScorer()
    quant_metrics = {
        "half_life": np.nan,
        "hurst_exponent": np.inf,
        "spread_sharpe": 1.5,
        "classification_f1": 0.7,
    }
    score = scorer.calculate_quantitative_score(quant_metrics)
    assert score >= 0.0
    assert score <= 100.0
    assert not (np.isnan(score) or np.isinf(score))


def test_calculate_quantitative_score_f1_out_of_range():
    scorer = OpportunityScorer()
    # F1 > 1 should be ignored
    quant_metrics = {"classification_f1": 1.5}
    score1 = scorer.calculate_quantitative_score(quant_metrics)
    
    # F1 < 0 should be ignored
    quant_metrics = {"classification_f1": -0.5}
    score2 = scorer.calculate_quantitative_score(quant_metrics)
    
    # Valid F1 should contribute
    quant_metrics = {"classification_f1": 0.75}
    score3 = scorer.calculate_quantitative_score(quant_metrics)
    
    assert score3 > score1
    assert score3 > score2
    assert score1 == score2  # Both invalid, should be treated the same


def test_calculate_quantitative_score_adx_nan_handling():
    scorer = OpportunityScorer(strategy="momentum")
    # NaN ADX should not contribute
    quant_metrics = {"long_adx": np.nan, "short_adx": 25.0}
    score1 = scorer.calculate_quantitative_score(quant_metrics)
    
    # Valid ADX should contribute
    quant_metrics = {"long_adx": 28.0, "short_adx": 30.0}
    score2 = scorer.calculate_quantitative_score(quant_metrics)
    
    assert score2 >= score1


def test_calculate_quantitative_score_final_score_capped_at_100():
    scorer = OpportunityScorer()
    # Create metrics that would exceed 100 if not capped
    quant_metrics = {
        "is_cointegrated": True,
        "half_life": 10,
        "hurst_exponent": 0.35,
        "spread_sharpe": 3.0,
        "classification_f1": 0.9,
        "max_drawdown": -0.05,
        "calmar_ratio": 2.0,
    }
    score = scorer.calculate_quantitative_score(quant_metrics)
    assert score <= 100.0
    assert not (np.isnan(score) or np.isinf(score))


# ============================================================================
# _apply_momentum_adx_filter Validation Tests
# ============================================================================

def test_apply_momentum_adx_filter_invalid_adx_nan():
    scorer = OpportunityScorer(strategy="momentum")
    score = scorer._apply_momentum_adx_filter(0.5, np.nan, 25.0)
    assert score == 0.0


def test_apply_momentum_adx_filter_invalid_adx_inf():
    scorer = OpportunityScorer(strategy="momentum")
    score = scorer._apply_momentum_adx_filter(0.5, np.inf, 25.0)
    assert score == 0.0


def test_apply_momentum_adx_filter_none_adx():
    scorer = OpportunityScorer(strategy="momentum")
    # Should return original score if ADX is None
    score = scorer._apply_momentum_adx_filter(0.5, None, None)
    assert score == 0.5


# ============================================================================
# _get_metric Validation Tests
# ============================================================================

def test_get_metric_filters_nan_inf():
    scorer = OpportunityScorer()
    quant_metrics = {
        "half_life": np.nan,
        "kalman_half_life": 20.0,
    }
    result = scorer._get_metric(quant_metrics, "half_life", "kalman_half_life")
    assert result == 20.0


def test_get_metric_avg_strategy_handles_nan():
    scorer = OpportunityScorer()
    scorer.hedge_ratio_strategy = "avg"
    quant_metrics = {
        "half_life": 15.0,
        "kalman_half_life": np.inf,
    }
    result = scorer._get_metric(quant_metrics, "half_life", "kalman_half_life")
    assert result == 15.0  # Should fallback to OLS