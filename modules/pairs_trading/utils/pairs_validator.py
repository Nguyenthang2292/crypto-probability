"""
Pairs validation utilities for pairs trading analysis.

This module provides functions for validating pairs trading opportunities
based on various quantitative and qualitative criteria.
"""

import numpy as np
import pandas as pd
from typing import Optional, Any, List

try:
    from modules.common.utils import (
        log_warn,
        log_info,
        log_success,
        log_progress,
    )
    from modules.common.ProgressBar import ProgressBar
except ImportError:
    # Fallback logging functions if modules.common.utils is not available
    def log_warn(msg: str) -> None:
        print(f"[WARN] {msg}")
    
    def log_info(msg: str) -> None:
        print(f"[INFO] {msg}")
    
    def log_success(msg: str) -> None:
        print(f"[SUCCESS] {msg}")
    
    def log_progress(msg: str) -> None:
        print(f"[PROGRESS] {msg}")
    
    ProgressBar = None

try:
    from modules.config import PAIRS_TRADING_PAIR_COLUMNS
except ImportError:
    PAIRS_TRADING_PAIR_COLUMNS = [
        'long_symbol',
        'short_symbol',
        'long_score',
        'short_score',
        'spread',
        'correlation',
        'opportunity_score',
        'quantitative_score',
        'hedge_ratio',
        'adf_pvalue',
        'is_cointegrated',
        'half_life',
        'mean_zscore',
        'std_zscore',
        'skewness',
        'kurtosis',
        'current_zscore',
        'hurst_exponent',
        'spread_sharpe',
        'max_drawdown',
        'calmar_ratio',
        'classification_f1',
        'classification_precision',
        'classification_recall',
        'classification_accuracy',
        'johansen_trace_stat',
        'johansen_critical_value',
        'is_johansen_cointegrated',
        'kalman_hedge_ratio',
        'kalman_half_life',
        'kalman_mean_zscore',
        'kalman_std_zscore',
        'kalman_skewness',
        'kalman_kurtosis',
        'kalman_current_zscore',
        'kalman_hurst_exponent',
        'kalman_spread_sharpe',
        'kalman_max_drawdown',
        'kalman_calmar_ratio',
        'kalman_classification_f1',
        'kalman_classification_precision',
        'kalman_classification_recall',
        'kalman_classification_accuracy',
        'long_adx',
        'short_adx',
    ]


def _get_all_pair_columns() -> List[str]:
    """
    Get all column names for pair DataFrames.
    
    Returns column list from PAIRS_TRADING_PAIR_COLUMNS constant.
    This includes core pair information (long_symbol, short_symbol, spread, etc.)
    and all quantitative metrics from PairMetricsComputer.
    """
    try:
        return PAIRS_TRADING_PAIR_COLUMNS.copy()
    except NameError:
        # Fallback if constant not imported
        return [
            'long_symbol',
            'short_symbol',
            'long_score',
            'short_score',
            'spread',
            'correlation',
            'opportunity_score',
            'quantitative_score',
            'hedge_ratio',
            'adf_pvalue',
            'is_cointegrated',
            'half_life',
            'mean_zscore',
            'std_zscore',
            'skewness',
            'kurtosis',
            'current_zscore',
            'hurst_exponent',
            'spread_sharpe',
            'max_drawdown',
            'calmar_ratio',
            'classification_f1',
            'classification_precision',
            'classification_recall',
            'classification_accuracy',
            'johansen_trace_stat',
            'johansen_critical_value',
            'is_johansen_cointegrated',
            'kalman_hedge_ratio',
            'kalman_half_life',
            'kalman_mean_zscore',
            'kalman_std_zscore',
            'kalman_skewness',
            'kalman_kurtosis',
            'kalman_current_zscore',
            'kalman_hurst_exponent',
            'kalman_spread_sharpe',
            'kalman_max_drawdown',
            'kalman_calmar_ratio',
            'kalman_classification_f1',
            'kalman_classification_precision',
            'kalman_classification_recall',
            'kalman_classification_accuracy',
            'long_adx',
            'short_adx',
        ]


def validate_pairs(
    pairs_df: pd.DataFrame,
    min_spread: float,
    max_spread: float,
    min_correlation: float,
    max_correlation: float,
    require_cointegration: bool = False,
    max_half_life: Optional[float] = None,
    hurst_threshold: Optional[float] = None,
    min_spread_sharpe: Optional[float] = None,
    max_drawdown_threshold: Optional[float] = None,
    min_quantitative_score: Optional[float] = None,
    data_fetcher: Optional[Any] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Validate pairs trading opportunities.

    Checks:
    - Spread is within acceptable range (min_spread to max_spread)
    - Correlation is within acceptable range (if available)
    - Cointegration requirement (if enabled)
    - Half-life, Hurst exponent, Sharpe ratio, max drawdown, quantitative score

    Args:
        pairs_df: DataFrame from analyze_pairs_opportunity()
        min_spread: Minimum spread (%) between long and short symbols
        max_spread: Maximum spread (%) between long and short symbols
        min_correlation: Minimum correlation to consider a pair
        max_correlation: Maximum correlation (avoid over-correlated pairs)
        require_cointegration: If True, only accept cointegrated pairs
        max_half_life: Maximum acceptable half-life for mean reversion (None to disable)
        hurst_threshold: Maximum Hurst exponent (should be < 0.5 for mean reversion, None to disable)
        min_spread_sharpe: Minimum Sharpe ratio (None to disable)
        max_drawdown_threshold: Maximum drawdown threshold (None to disable)
        min_quantitative_score: Minimum quantitative score (0-100, None to disable)
        data_fetcher: DataFetcher instance for validation (currently unused, reserved for future use)
        verbose: If True, print validation messages

    Returns:
        DataFrame with validated pairs only, sorted by opportunity_score descending
    """
    empty_df = pd.DataFrame(columns=_get_all_pair_columns())
    
    if pairs_df is None or pairs_df.empty:
        return empty_df

    if verbose:
        log_progress(f"Validating {len(pairs_df)} pairs...")

    validated_pairs = []

    progress = None
    if verbose and ProgressBar:
        progress = ProgressBar(len(pairs_df), "Validation")

    for _, row in pairs_df.iterrows():
        long_symbol = row.get('long_symbol')
        short_symbol = row.get('short_symbol')
        spread = row.get('spread')
        correlation = row.get('correlation')

        # Validate required fields
        if pd.isna(long_symbol) or pd.isna(short_symbol):
            if progress:
                progress.update()
            continue
        
        if pd.isna(spread) or np.isinf(spread):
            if verbose:
                log_warn(f"Invalid spread for {long_symbol}/{short_symbol}: {spread}")
            if progress:
                progress.update()
            continue

        is_valid = True
        validation_errors = []

        # Check spread
        if spread < min_spread:
            is_valid = False
            validation_errors.append(f"Spread too small ({spread*100:.2f}% < {min_spread*100:.2f}%)")
        elif spread > max_spread:
            is_valid = False
            validation_errors.append(f"Spread too large ({spread*100:.2f}% > {max_spread*100:.2f}%)")

        # Check correlation if available
        if correlation is not None and not pd.isna(correlation) and not np.isinf(correlation):
            # Validate correlation is in valid range
            if not (-1 <= correlation <= 1):
                if verbose:
                    log_warn(f"Invalid correlation value for {long_symbol}/{short_symbol}: {correlation}")
                if progress:
                    progress.update()
                continue
            
            abs_corr = abs(correlation)
            if abs_corr < min_correlation:
                is_valid = False
                validation_errors.append(
                    f"Correlation too low ({abs_corr:.3f} < {min_correlation:.3f})"
                )
            elif abs_corr > max_correlation:
                is_valid = False
                validation_errors.append(
                    f"Correlation too high ({abs_corr:.3f} > {max_correlation:.3f})"
                )

        # Check quantitative metrics if available
        # Cointegration requirement
        if require_cointegration:
            is_cointegrated = row.get('is_cointegrated')
            # Fall back to Johansen cointegration flag if ADF-based flag is missing
            if (is_cointegrated is None or pd.isna(is_cointegrated)) and 'is_johansen_cointegrated' in row:
                alt_coint = row.get('is_johansen_cointegrated')
                if alt_coint is not None and not pd.isna(alt_coint):
                    is_cointegrated = bool(alt_coint)
            if is_cointegrated is None or pd.isna(is_cointegrated) or not is_cointegrated:
                is_valid = False
                validation_errors.append("Not cointegrated (required)")

        # Half-life check
        if max_half_life is not None:
            half_life = row.get('half_life')
            if half_life is not None and not pd.isna(half_life) and not np.isinf(half_life):
                if half_life > max_half_life:
                    is_valid = False
                    validation_errors.append(
                        f"Half-life too high ({half_life:.1f} > {max_half_life})"
                    )

        # Hurst exponent check
        if hurst_threshold is not None:
            hurst = row.get('hurst_exponent')
            if hurst is not None and not pd.isna(hurst) and not np.isinf(hurst):
                if hurst >= hurst_threshold:
                    is_valid = False
                    validation_errors.append(
                        f"Hurst exponent too high ({hurst:.3f} >= {hurst_threshold}, not mean-reverting)"
                    )

        # Sharpe ratio check
        if min_spread_sharpe is not None:
            spread_sharpe = row.get('spread_sharpe')
            if spread_sharpe is not None and not pd.isna(spread_sharpe) and not np.isinf(spread_sharpe):
                if spread_sharpe < min_spread_sharpe:
                    is_valid = False
                    validation_errors.append(
                        f"Sharpe ratio too low ({spread_sharpe:.2f} < {min_spread_sharpe})"
                    )

        # Max drawdown check
        if max_drawdown_threshold is not None:
            max_dd = row.get('max_drawdown')
            if max_dd is not None and not pd.isna(max_dd) and not np.isinf(max_dd):
                if abs(max_dd) > max_drawdown_threshold:
                    is_valid = False
                    validation_errors.append(
                        f"Max drawdown too high ({abs(max_dd)*100:.2f}% > {max_drawdown_threshold*100:.2f}%)"
                    )

        # Quantitative score check
        if min_quantitative_score is not None:
            quant_score = row.get('quantitative_score')
            if quant_score is not None and not pd.isna(quant_score) and not np.isinf(quant_score):
                if quant_score < min_quantitative_score:
                    is_valid = False
                    validation_errors.append(
                        f"Quantitative score too low ({quant_score:.1f} < {min_quantitative_score})"
                    )

        if is_valid:
            validated_pairs.append(row.to_dict())
        elif verbose:
            log_warn(f"Rejected {long_symbol} / {short_symbol}: {', '.join(validation_errors)}")

        if progress:
            progress.update()

    if progress:
        progress.finish()

    if not validated_pairs:
        if verbose:
            log_warn("No pairs passed validation.")
        return empty_df

    df_validated = pd.DataFrame(validated_pairs)
    # Maintain sort order by opportunity_score
    df_validated = df_validated.sort_values('opportunity_score', ascending=False).reset_index(
        drop=True
    )

    if verbose:
        log_success(f"{len(df_validated)}/{len(pairs_df)} pairs passed validation.")

    return df_validated

