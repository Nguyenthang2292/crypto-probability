"""
Pairs trading analyzer for identifying and validating pairs trading opportunities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.common.DataFetcher import DataFetcher

try:
    from modules.config import (
        PAIRS_TRADING_MIN_SPREAD,
        PAIRS_TRADING_MAX_SPREAD,
        PAIRS_TRADING_MIN_CORRELATION,
        PAIRS_TRADING_MAX_CORRELATION,
        PAIRS_TRADING_CORRELATION_MIN_POINTS,
        PAIRS_TRADING_TIMEFRAME,
        PAIRS_TRADING_LIMIT,
        PAIRS_TRADING_ADF_PVALUE_THRESHOLD,
        PAIRS_TRADING_MAX_HALF_LIFE,
        PAIRS_TRADING_ZSCORE_LOOKBACK,
        PAIRS_TRADING_HURST_THRESHOLD,
        PAIRS_TRADING_MIN_SPREAD_SHARPE,
        PAIRS_TRADING_MAX_DRAWDOWN,
        PAIRS_TRADING_MIN_CALMAR,
        PAIRS_TRADING_JOHANSEN_CONFIDENCE,
        PAIRS_TRADING_PERIODS_PER_YEAR,
        PAIRS_TRADING_CLASSIFICATION_ZSCORE,
        PAIRS_TRADING_OLS_FIT_INTERCEPT,
        PAIRS_TRADING_KALMAN_DELTA,
        PAIRS_TRADING_KALMAN_OBS_COV,
        PAIRS_TRADING_PAIR_COLUMNS,
        PAIRS_TRADING_ADX_PERIOD,
    )
    from modules.common.utils import (
        log_warn,
        log_info,
        log_success,
        log_progress,
    )
    from modules.common.ProgressBar import ProgressBar
    from modules.common.indicators import calculate_adx
except ImportError:
    PAIRS_TRADING_MIN_SPREAD = 0.01
    PAIRS_TRADING_MAX_SPREAD = 0.50
    PAIRS_TRADING_MIN_CORRELATION = 0.3
    PAIRS_TRADING_MAX_CORRELATION = 0.9
    PAIRS_TRADING_CORRELATION_MIN_POINTS = 50
    PAIRS_TRADING_TIMEFRAME = "1h"
    PAIRS_TRADING_LIMIT = 200
    PAIRS_TRADING_ADF_PVALUE_THRESHOLD = 0.05
    PAIRS_TRADING_MAX_HALF_LIFE = 50
    PAIRS_TRADING_ZSCORE_LOOKBACK = 60
    PAIRS_TRADING_HURST_THRESHOLD = 0.5
    PAIRS_TRADING_MIN_SPREAD_SHARPE = 1.0
    PAIRS_TRADING_MAX_DRAWDOWN = 0.3
    PAIRS_TRADING_MIN_CALMAR = 1.0
    PAIRS_TRADING_JOHANSEN_CONFIDENCE = 0.95
    PAIRS_TRADING_PERIODS_PER_YEAR = 365 * 24
    PAIRS_TRADING_CLASSIFICATION_ZSCORE = 0.5
    PAIRS_TRADING_OLS_FIT_INTERCEPT = True
    PAIRS_TRADING_KALMAN_DELTA = 1e-5
    PAIRS_TRADING_KALMAN_OBS_COV = 1.0
    PAIRS_TRADING_PAIR_COLUMNS = [
        # Core pair information
        'long_symbol',
        'short_symbol',
        'long_score',
        'short_score',
        'spread',
        'correlation',
        'opportunity_score',
        'quantitative_score',
        # OLS-based metrics
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
        # Johansen test (independent of hedge ratio method)
        'johansen_trace_stat',
        'johansen_critical_value',
        'is_johansen_cointegrated',
        # Kalman hedge ratio
        'kalman_hedge_ratio',
        # Kalman-based metrics
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
    PAIRS_TRADING_ADX_PERIOD = 14
    
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

    def calculate_adx(ohlcv: pd.DataFrame, period: int = 14) -> Optional[float]:
        if ohlcv is None or len(ohlcv) < period * 2:
            return None

        required = {"high", "low", "close"}
        if not required.issubset(ohlcv.columns):
            return None

        data = ohlcv[list(required)].astype(float)
        high = data["high"]
        low = data["low"]
        close = data["close"]

        up_move = high.diff()
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_components = pd.concat(
            [
                (high - low),
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1)

        atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = (
            pd.Series(plus_dm, index=data.index)
            .ewm(alpha=1 / period, adjust=False)
            .mean()
            * 100
            / atr.replace(0, pd.NA)
        )
        minus_di = (
            pd.Series(minus_dm, index=data.index)
            .ewm(alpha=1 / period, adjust=False)
            .mean()
            * 100
            / atr.replace(0, pd.NA)
        )

        denom = (plus_di + minus_di).replace(0, pd.NA)
        dx = ((plus_di - minus_di).abs() / denom) * 100
        adx = dx.ewm(alpha=1 / period, adjust=False).mean().dropna()

        if adx.empty:
            return None

        last_value = adx.iloc[-1]
        if pd.isna(last_value) or np.isinf(last_value):
            return None

        return float(last_value)

from modules.pairs_trading.core.pair_metrics_computer import PairMetricsComputer
from modules.pairs_trading.core.opportunity_scorer import OpportunityScorer
from modules.pairs_trading.utils.pairs_validator import validate_pairs as validate_pairs_util
from modules.pairs_trading.metrics import calculate_correlation as calculate_correlation_metric


def _get_all_pair_columns() -> list:
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


class PairsTradingAnalyzer:
    """
    Analyzes pairs trading opportunities from best and worst performing symbols.
    
    Pairs trading strategy:
    - Long worst performers (expect mean reversion upward)
    - Short best performers (expect mean reversion downward)
    """

    def __init__(
        self,
        min_spread: float = PAIRS_TRADING_MIN_SPREAD,
        max_spread: float = PAIRS_TRADING_MAX_SPREAD,
        min_correlation: float = PAIRS_TRADING_MIN_CORRELATION,
        max_correlation: float = PAIRS_TRADING_MAX_CORRELATION,
        correlation_min_points: int = PAIRS_TRADING_CORRELATION_MIN_POINTS,
        require_cointegration: bool = False,
        max_half_life: float = PAIRS_TRADING_MAX_HALF_LIFE,
        hurst_threshold: float = PAIRS_TRADING_HURST_THRESHOLD,
        min_spread_sharpe: Optional[float] = None,
        max_drawdown_threshold: Optional[float] = None,
        min_quantitative_score: Optional[float] = None,
        ols_fit_intercept: bool = PAIRS_TRADING_OLS_FIT_INTERCEPT,
        kalman_delta: float = PAIRS_TRADING_KALMAN_DELTA,
        kalman_obs_cov: float = PAIRS_TRADING_KALMAN_OBS_COV,
        scoring_multipliers: Optional[Dict[str, float]] = None,
        strategy: str = "reversion",
    ):
        """
        Initialize PairsTradingAnalyzer.

        Args:
            min_spread: Minimum spread (%) between long and short symbols
            max_spread: Maximum spread (%) between long and short symbols
            min_correlation: Minimum correlation to consider a pair
            max_correlation: Maximum correlation (avoid over-correlated pairs)
            correlation_min_points: Minimum data points for correlation calculation
            require_cointegration: If True, only accept cointegrated pairs
            max_half_life: Maximum acceptable half-life for mean reversion
            hurst_threshold: Maximum Hurst exponent (should be < 0.5 for mean reversion)
            min_spread_sharpe: Minimum Sharpe ratio (None to disable)
            max_drawdown_threshold: Maximum drawdown threshold (None to disable)
            min_quantitative_score: Minimum quantitative score (0-100, None to disable)
            strategy: Trading strategy ('reversion' or 'momentum')
            
        Raises:
            ValueError: If parameter values are invalid (e.g., min_spread > max_spread)
        """
        # Validate parameters
        if min_spread < 0:
            raise ValueError(f"min_spread must be non-negative, got {min_spread}")
        if max_spread <= 0:
            raise ValueError(f"max_spread must be positive, got {max_spread}")
        if min_spread > max_spread:
            raise ValueError(
                f"min_spread ({min_spread}) must be <= max_spread ({max_spread})"
            )
        if not (-1 <= min_correlation <= 1):
            raise ValueError(
                f"min_correlation must be in [-1, 1], got {min_correlation}"
            )
        if not (-1 <= max_correlation <= 1):
            raise ValueError(
                f"max_correlation must be in [-1, 1], got {max_correlation}"
            )
        if min_correlation > max_correlation:
            raise ValueError(
                f"min_correlation ({min_correlation}) must be <= max_correlation ({max_correlation})"
            )
        if correlation_min_points < 2:
            raise ValueError(
                f"correlation_min_points must be >= 2, got {correlation_min_points}"
            )
        if max_half_life <= 0:
            raise ValueError(f"max_half_life must be positive, got {max_half_life}")
        if not (0 < hurst_threshold <= 1):
            raise ValueError(
                f"hurst_threshold must be in (0, 1], got {hurst_threshold}"
            )
        if min_spread_sharpe is not None and (np.isnan(min_spread_sharpe) or np.isinf(min_spread_sharpe)):
            raise ValueError(f"min_spread_sharpe must be finite, got {min_spread_sharpe}")
        if max_drawdown_threshold is not None:
            if max_drawdown_threshold <= 0 or max_drawdown_threshold > 1:
                raise ValueError(
                    f"max_drawdown_threshold must be in (0, 1], got {max_drawdown_threshold}"
                )
        if min_quantitative_score is not None:
            if not (0 <= min_quantitative_score <= 100):
                raise ValueError(
                    f"min_quantitative_score must be in [0, 100], got {min_quantitative_score}"
                )
        if kalman_delta <= 0 or kalman_delta >= 1:
            raise ValueError(f"kalman_delta must be in (0, 1), got {kalman_delta}")
        if kalman_obs_cov <= 0:
            raise ValueError(f"kalman_obs_cov must be positive, got {kalman_obs_cov}")
        if strategy not in ["reversion", "momentum"]:
            raise ValueError(f"strategy must be 'reversion' or 'momentum', got {strategy}")
        
        self.min_spread = min_spread
        self.max_spread = max_spread
        self.min_correlation = min_correlation
        self.max_correlation = max_correlation
        self.correlation_min_points = correlation_min_points
        self.require_cointegration = require_cointegration
        self.max_half_life = max_half_life
        self.hurst_threshold = hurst_threshold
        self.min_spread_sharpe = min_spread_sharpe if min_spread_sharpe is not None else PAIRS_TRADING_MIN_SPREAD_SHARPE
        self.max_drawdown_threshold = max_drawdown_threshold if max_drawdown_threshold is not None else PAIRS_TRADING_MAX_DRAWDOWN
        self.min_quantitative_score = min_quantitative_score
        self.ols_fit_intercept = ols_fit_intercept
        self.kalman_delta = kalman_delta
        self.kalman_obs_cov = kalman_obs_cov
        self.scoring_multipliers = scoring_multipliers
        self.adf_pvalue_threshold = PAIRS_TRADING_ADF_PVALUE_THRESHOLD
        self.johansen_confidence = PAIRS_TRADING_JOHANSEN_CONFIDENCE
        self.min_calmar = PAIRS_TRADING_MIN_CALMAR
        self._correlation_cache: Dict[Tuple[str, str], float] = {}
        self._price_cache: Dict[Tuple[str, str], Optional[pd.DataFrame]] = {}
        self._adx_cache: Dict[str, Optional[float]] = {}
        self.adx_period = PAIRS_TRADING_ADX_PERIOD
        
        # Initialize metrics computer and opportunity scorer
        self.metrics_computer = PairMetricsComputer(
            adf_pvalue_threshold=self.adf_pvalue_threshold,
            periods_per_year=PAIRS_TRADING_PERIODS_PER_YEAR,
            zscore_lookback=PAIRS_TRADING_ZSCORE_LOOKBACK,
            classification_zscore=PAIRS_TRADING_CLASSIFICATION_ZSCORE,
            johansen_confidence=self.johansen_confidence,
            correlation_min_points=correlation_min_points,
            ols_fit_intercept=self.ols_fit_intercept,
            kalman_delta=self.kalman_delta,
            kalman_obs_cov=self.kalman_obs_cov,
        )
        
        self.opportunity_scorer = OpportunityScorer(
            min_correlation=self.min_correlation,
            max_correlation=self.max_correlation,
            adf_pvalue_threshold=self.adf_pvalue_threshold,
            max_half_life=self.max_half_life,
            hurst_threshold=self.hurst_threshold,
            min_spread_sharpe=self.min_spread_sharpe,
            max_drawdown_threshold=self.max_drawdown_threshold,
            min_calmar=self.min_calmar,
            scoring_multipliers=self.scoring_multipliers,
            strategy=strategy,
        )

    def _fetch_aligned_prices(
        self,
        symbol1: str,
        symbol2: str,
        data_fetcher: Optional[Any],
        timeframe: str = PAIRS_TRADING_TIMEFRAME,
        limit: int = PAIRS_TRADING_LIMIT,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch and align price series for two symbols.
        Returns DataFrame with columns ['close1', 'close2'] aligned by timestamp/index.
        """
        cache_key = tuple(sorted([symbol1, symbol2]))
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        if data_fetcher is None:
            return None

        try:
            df1, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol1, limit=limit, timeframe=timeframe, check_freshness=False
            )
            df2, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol2, limit=limit, timeframe=timeframe, check_freshness=False
            )
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            # Specific exceptions for data fetching errors
            log_warn(f"Failed to fetch data for {symbol1} or {symbol2}: {e}")
            self._price_cache[cache_key] = None
            return None
        except Exception as e:
            # Catch-all for unexpected errors
            log_warn(f"Unexpected error fetching data for {symbol1} or {symbol2}: {e}")
            self._price_cache[cache_key] = None
            return None

        if (
            df1 is None
            or df2 is None
            or df1.empty
            or df2.empty
            or "close" not in df1.columns
            or "close" not in df2.columns
        ):
            self._price_cache[cache_key] = None
            return None

        if "timestamp" not in df1.columns or "timestamp" not in df2.columns:
            log_warn(
                f"Missing timestamp column for {symbol1} or {symbol2}. "
                "Cannot align series reliably."
            )
            self._price_cache[cache_key] = None
            return None

        # Check for duplicate timestamps before setting index
        if df1["timestamp"].duplicated().any():
            log_warn(f"Duplicate timestamps found for {symbol1}, using first occurrence")
            df1 = df1.drop_duplicates(subset=["timestamp"], keep="first")
        if df2["timestamp"].duplicated().any():
            log_warn(f"Duplicate timestamps found for {symbol2}, using first occurrence")
            df2 = df2.drop_duplicates(subset=["timestamp"], keep="first")
        
        df1 = df1.set_index("timestamp")
        df2 = df2.set_index("timestamp")
        
        # Validate close columns contain valid numeric data
        if df1["close"].isna().all() or df2["close"].isna().all():
            self._price_cache[cache_key] = None
            return None
        
        # Check for infinite values
        if np.isinf(df1["close"]).any() or np.isinf(df2["close"]).any():
            log_warn(f"Infinite values found in price data for {symbol1} or {symbol2}")
            self._price_cache[cache_key] = None
            return None
        
        df_combined = pd.concat(
            [df1[["close"]], df2[["close"]]], axis=1, join="inner"
        )
        df_combined.columns = ["close1", "close2"]
        
        # Validate combined DataFrame
        if df_combined.empty:
            self._price_cache[cache_key] = None
            return None

        if len(df_combined) < self.correlation_min_points:
            self._price_cache[cache_key] = None
            return None

        self._price_cache[cache_key] = df_combined
        return df_combined

    def _get_symbol_adx(self, symbol: str, data_fetcher: Optional[Any]) -> Optional[float]:
        """
        Retrieve cached ADX for symbol, computing if necessary.
        """
        if symbol in self._adx_cache:
            return self._adx_cache[symbol]

        if data_fetcher is None:
            self._adx_cache[symbol] = None
            return None
        try:
            ohlcv, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
                symbol,
                limit=PAIRS_TRADING_LIMIT,
                timeframe=PAIRS_TRADING_TIMEFRAME,
                check_freshness=False,
            )
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            log_warn(f"Failed to fetch ADX data for {symbol}: {e}")
            self._adx_cache[symbol] = None
            return None
        except Exception as e:
            log_warn(f"Unexpected error fetching ADX data for {symbol}: {e}")
            self._adx_cache[symbol] = None
            return None

        if (
            ohlcv is None
            or ohlcv.empty
            or not {"high", "low", "close"}.issubset(ohlcv.columns)
        ):
            self._adx_cache[symbol] = None
            return None

        adx_value = calculate_adx(ohlcv, self.adx_period)
        self._adx_cache[symbol] = adx_value
        return adx_value

    def calculate_correlation(
        self,
        symbol1: str,
        symbol2: str,
        data_fetcher: Optional[Any],
        timeframe: str = PAIRS_TRADING_TIMEFRAME,
        limit: int = PAIRS_TRADING_LIMIT,
    ) -> Optional[float]:
        """
        Calculate correlation between two symbols based on returns.

        This method wraps the metric function with caching and data fetching logic.

        Args:
            symbol1: First trading symbol
            symbol2: Second trading symbol
            data_fetcher: DataFetcher instance
            timeframe: Timeframe for OHLCV data
            limit: Number of candles to fetch

        Returns:
            Correlation coefficient (float between -1 and 1), or None if calculation fails
        """
        # Check cache
        cache_key = tuple(sorted([symbol1, symbol2]))
        if cache_key in self._correlation_cache:
            return self._correlation_cache[cache_key]

        # Fetch and align prices
        df_combined = self._fetch_aligned_prices(
            symbol1, symbol2, data_fetcher, timeframe=timeframe, limit=limit
        )

        if df_combined is None:
            return None

        # Extract price series
        price1 = df_combined["close1"]
        price2 = df_combined["close2"]

        # Calculate correlation using metric function
        correlation = calculate_correlation_metric(
            price1=price1,
            price2=price2,
            min_points=self.correlation_min_points,
        )

        # Cache result if valid
        if correlation is not None:
            self._correlation_cache[cache_key] = correlation

        return correlation

    def _compute_pair_metrics(
        self,
        symbol1: str,
        symbol2: str,
        data_fetcher: Optional[Any],
    ) -> Dict[str, Optional[float]]:
        """Compute Phase 1 quantitative metrics for a pair."""
        aligned_prices = self._fetch_aligned_prices(
            symbol1, symbol2, data_fetcher, timeframe=PAIRS_TRADING_TIMEFRAME
        )
        if aligned_prices is None:
            return {}

        price1 = aligned_prices["close1"]
        price2 = aligned_prices["close2"]

        metrics = self.metrics_computer.compute_pair_metrics(price1, price2)
        metrics["long_adx"] = self._get_symbol_adx(symbol1, data_fetcher)
        metrics["short_adx"] = self._get_symbol_adx(symbol2, data_fetcher)
        return metrics

    def calculate_spread(
        self, long_symbol: str, short_symbol: str, long_score: float, short_score: float
    ) -> float:
        """
        Calculate spread between long and short symbols based on performance scores.

        Args:
            long_symbol: Symbol to long (worst performer)
            short_symbol: Symbol to short (best performer)
            long_score: Performance score of long symbol (typically negative)
            short_score: Performance score of short symbol (typically positive)

        Returns:
            Spread as percentage (positive value)
            
        Raises:
            ValueError: If scores are NaN or Inf
        """
        # Validate inputs
        if pd.isna(long_score) or pd.isna(short_score):
            raise ValueError(
                f"Invalid scores: long_score={long_score}, short_score={short_score}"
            )
        if np.isinf(long_score) or np.isinf(short_score):
            raise ValueError(
                f"Infinite scores: long_score={long_score}, short_score={short_score}"
            )
        
        # Spread = difference between short score and long score
        # Since long_score is negative and short_score is positive,
        # spread = short_score - long_score (which is positive)
        spread = short_score - long_score
        
        # Validate result
        if np.isnan(spread) or np.isinf(spread):
            raise ValueError(f"Invalid spread calculation result: {spread}")
        
        return abs(spread)  # Ensure positive

    def analyze_pairs_opportunity(
        self,
        best_performers: pd.DataFrame,
        worst_performers: pd.DataFrame,
        data_fetcher: Optional[Any] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Analyze pairs trading opportunities from best and worst performers.

        Args:
            best_performers: DataFrame with top performers (columns: symbol, score, etc.)
            worst_performers: DataFrame with worst performers (columns: symbol, score, etc.)
            data_fetcher: Optional DataFetcher for correlation calculation
            verbose: If True, print progress messages

        Returns:
            DataFrame with columns: ['long_symbol', 'short_symbol', 'long_score', 
                                   'short_score', 'spread', 'correlation', 'opportunity_score']
        """
        empty_df = pd.DataFrame(columns=_get_all_pair_columns())
        
        # Validate inputs
        if best_performers is None or worst_performers is None:
            if verbose:
                log_warn("None DataFrame provided for pairs analysis.")
            return empty_df
        
        if best_performers.empty:
            if verbose:
                log_warn("No best performers provided for pairs analysis.")
            return empty_df

        if worst_performers.empty:
            if verbose:
                log_warn("No worst performers provided for pairs analysis.")
            return empty_df
        
        # Validate required columns
        required_columns = {"symbol", "score"}
        if not required_columns.issubset(best_performers.columns):
            missing = required_columns - set(best_performers.columns)
            raise ValueError(f"best_performers missing required columns: {missing}")
        if not required_columns.issubset(worst_performers.columns):
            missing = required_columns - set(worst_performers.columns)
            raise ValueError(f"worst_performers missing required columns: {missing}")

        pairs = []

        if verbose:
            log_info(
                f"Analyzing pairs: {len(worst_performers)} worst × {len(best_performers)} best = "
                f"{len(worst_performers) * len(best_performers)} combinations..."
            )

        progress = None
        total_combinations = len(worst_performers) * len(best_performers)
        if verbose and ProgressBar:
            progress = ProgressBar(total_combinations, "Pairs Analysis")

        for _, worst_row in worst_performers.iterrows():
            long_symbol = worst_row.get('symbol')
            long_score = worst_row.get('score')

            # Validate row data
            if pd.isna(long_symbol) or long_symbol is None:
                if progress:
                    progress.update()
                continue
            if pd.isna(long_score) or np.isinf(long_score):
                if verbose:
                    log_warn(f"Invalid long_score for {long_symbol}: {long_score}")
                if progress:
                    progress.update()
                continue

            for _, best_row in best_performers.iterrows():
                short_symbol = best_row.get('symbol')
                short_score = best_row.get('score')

                # Validate row data
                if pd.isna(short_symbol) or short_symbol is None:
                    if progress:
                        progress.update()
                    continue
                if pd.isna(short_score) or np.isinf(short_score):
                    if verbose:
                        log_warn(f"Invalid short_score for {short_symbol}: {short_score}")
                    if progress:
                        progress.update()
                    continue

                # Skip if same symbol
                if long_symbol == short_symbol:
                    if progress:
                        progress.update()
                    continue

                # Calculate spread
                try:
                    spread = self.calculate_spread(long_symbol, short_symbol, long_score, short_score)
                except ValueError as e:
                    if verbose:
                        log_warn(f"Failed to calculate spread for {long_symbol}/{short_symbol}: {e}")
                    if progress:
                        progress.update()
                    continue

                # Calculate correlation if data_fetcher provided
                correlation = None
                quant_metrics = {}
                if data_fetcher is not None:
                    correlation = self.calculate_correlation(
                        long_symbol, short_symbol, data_fetcher
                    )
                    # Không loại bỏ pair ngay cả khi tương quan ngoài vùng lý tưởng;
                    # OpportunityScorer sẽ tự áp dụng hình phạt thông qua hệ số.
                    quant_metrics = self._compute_pair_metrics(
                        long_symbol, short_symbol, data_fetcher
                    )

                # Calculate opportunity score
                try:
                    opportunity_score = self.opportunity_scorer.calculate_opportunity_score(
                        spread, correlation, quant_metrics
                    )
                    if pd.isna(opportunity_score) or np.isinf(opportunity_score):
                        opportunity_score = 0.0
                except Exception as e:
                    if verbose:
                        log_warn(f"Error calculating opportunity_score for {long_symbol}/{short_symbol}: {e}")
                    opportunity_score = 0.0
                
                # Calculate quantitative score
                try:
                    quantitative_score = self.opportunity_scorer.calculate_quantitative_score(
                        quant_metrics
                    )
                    if pd.isna(quantitative_score) or np.isinf(quantitative_score):
                        quantitative_score = 0.0
                except Exception as e:
                    if verbose:
                        log_warn(f"Error calculating quantitative_score for {long_symbol}/{short_symbol}: {e}")
                    quantitative_score = 0.0

                # Build pair record
                pair_record = {
                    'long_symbol': str(long_symbol),
                    'short_symbol': str(short_symbol),
                    'long_score': float(long_score),
                    'short_score': float(short_score),
                    'spread': float(spread),
                    'correlation': float(correlation) if correlation is not None and not (pd.isna(correlation) or np.isinf(correlation)) else None,
                    'opportunity_score': float(opportunity_score),
                    'quantitative_score': float(quantitative_score),
                }
                
                # Add all quant metrics
                for key in _get_all_pair_columns():
                    if key not in pair_record:
                        pair_record[key] = quant_metrics.get(key)

                pairs.append(pair_record)

                if progress:
                    progress.update()

        if progress:
            progress.finish()

        if not pairs:
            if verbose:
                log_warn("No pairs opportunities found.")
            return empty_df

        df_pairs = pd.DataFrame(pairs)
        # Sort by opportunity_score descending
        df_pairs = df_pairs.sort_values('opportunity_score', ascending=False).reset_index(
            drop=True
        )

        if verbose:
            log_success(f"Found {len(df_pairs)} pairs opportunities.")

        return df_pairs

    def validate_pairs(
        self,
        pairs_df: pd.DataFrame,
        data_fetcher: Optional[Any],
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Validate pairs trading opportunities.

        Checks:
        - Spread is within acceptable range (min_spread to max_spread)
        - Correlation is within acceptable range (if available)
        - Symbols are available and have sufficient volume

        Args:
            pairs_df: DataFrame from analyze_pairs_opportunity()
            data_fetcher: DataFetcher instance for validation (currently unused, reserved for future use)
            verbose: If True, print validation messages

        Returns:
            DataFrame with validated pairs only
        """
        return validate_pairs_util(
            pairs_df=pairs_df,
            min_spread=self.min_spread,
            max_spread=self.max_spread,
            min_correlation=self.min_correlation,
            max_correlation=self.max_correlation,
            require_cointegration=self.require_cointegration,
            max_half_life=self.max_half_life,
            hurst_threshold=self.hurst_threshold,
            min_spread_sharpe=self.min_spread_sharpe,
            max_drawdown_threshold=self.max_drawdown_threshold,
            min_quantitative_score=self.min_quantitative_score,
            data_fetcher=data_fetcher,
            verbose=verbose,
        )
