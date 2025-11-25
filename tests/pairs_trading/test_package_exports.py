from modules.pairs_trading import __all__ as exported_names


def test_package_exports_expected_symbols():
    expected = {
        "PairsTradingAnalyzer",
        "PairMetricsComputer",
        "OpportunityScorer",
        "calculate_adf_test",
        "calculate_half_life",
        "calculate_johansen_test",
        "calculate_spread_sharpe",
        "calculate_max_drawdown",
        "calculate_calmar_ratio",
        "calculate_ols_hedge_ratio",
        "calculate_kalman_hedge_ratio",
        "calculate_zscore_stats",
        "calculate_hurst_exponent",
        "calculate_direction_metrics",
    }

    assert set(exported_names) == expected

