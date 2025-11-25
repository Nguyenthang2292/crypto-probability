import math

from modules.pairs_trading.opportunity_scorer import OpportunityScorer


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

