"""
ATC + Range Oscillator + SPC Pure Voting System (Phương án 2).

This program combines signals from:
1. Adaptive Trend Classification (ATC)
2. Range Oscillator
3. Simplified Percentile Clustering (SPC)

Workflow (Pure Voting System):
1. Calculate signals from all 3 indicators in parallel
2. Each indicator votes (0 or 1)
3. Calculate weighted vote based on accuracy, importance, signal strength
4. Cumulative vote
5. Final prediction

Phương án 2: Thay thế hoàn toàn sequential filtering bằng voting system
"""

import warnings
import sys
import threading
from typing import Optional, Dict, Any, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from modules.common.utils import configure_windows_stdio

configure_windows_stdio()

from colorama import Fore, Style, init as colorama_init

from modules.config import (
    DEFAULT_TIMEFRAME,
    DECISION_MATRIX_SPC_STRATEGY_ACCURACIES,
    DECISION_MATRIX_INDICATOR_ACCURACIES,
    SPC_STRATEGY_PARAMETERS,
    RANGE_OSCILLATOR_LENGTH,
    RANGE_OSCILLATOR_MULTIPLIER,
    SPC_AGGREGATION_MODE,
    SPC_AGGREGATION_THRESHOLD,
    SPC_AGGREGATION_WEIGHTED_MIN_TOTAL,
    SPC_AGGREGATION_WEIGHTED_MIN_DIFF,
    SPC_AGGREGATION_ENABLE_ADAPTIVE_WEIGHTS,
    SPC_AGGREGATION_ADAPTIVE_PERFORMANCE_WINDOW,
    SPC_AGGREGATION_MIN_SIGNAL_STRENGTH,
    SPC_AGGREGATION_STRATEGY_WEIGHTS,
)
from modules.common.utils import (
    color_text,
    log_error,
    log_progress,
    log_success,
    log_warn,
    log_data,
    prompt_user_input,
)
from modules.common.ExchangeManager import ExchangeManager
from modules.common.DataFetcher import DataFetcher
from modules.adaptive_trend.cli import prompt_timeframe
from main_atc import ATCAnalyzer
from modules.range_oscillator.cli import (
    display_configuration,
    display_final_results,
)
import argparse
from modules.range_oscillator.analysis.combined import (
    generate_signals_combined_all_strategy,
)
from modules.range_oscillator.config import (
    CombinedStrategyConfig,
)
from modules.simplified_percentile_clustering.core.clustering import (
    SimplifiedPercentileClustering,
    ClusteringConfig,
)
from modules.simplified_percentile_clustering.core.features import FeatureConfig
from modules.simplified_percentile_clustering.strategies import (
    generate_signals_cluster_transition,
    generate_signals_regime_following,
    generate_signals_mean_reversion,
)
from modules.simplified_percentile_clustering.config import (
    ClusterTransitionConfig,
    RegimeFollowingConfig,
    MeanReversionConfig,
)
from modules.simplified_percentile_clustering.aggregation import (
    SPCVoteAggregator,
)
from modules.simplified_percentile_clustering.config import (
    SPCAggregationConfig,
)
from modules.decision_matrix.classifier import DecisionMatrixClassifier

warnings.filterwarnings("ignore")
colorama_init(autoreset=True)


def get_range_oscillator_signal(
    data_fetcher: DataFetcher,
    symbol: str,
    timeframe: str,
    limit: int,
    osc_length: int = 50,
    osc_mult: float = 2.0,
    strategies: Optional[list] = None,
) -> Optional[tuple]:
    """Calculate Range Oscillator signal for a symbol."""
    try:
        df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol,
            limit=limit,
            timeframe=timeframe,
            check_freshness=True,
        )

        if df is None or df.empty:
            return None

        if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns:
            return None

        high = df["high"]
        low = df["low"]
        close = df["close"]

        if strategies is None:
            enabled_strategies = [2, 3, 4, 6, 7, 8, 9]
        else:
            if 5 in strategies:
                enabled_strategies = [2, 3, 4, 6, 7, 8, 9]
            else:
                enabled_strategies = strategies

        config = CombinedStrategyConfig()
        config.enabled_strategies = enabled_strategies
        config.return_confidence_score = True
        config.dynamic.enabled = True
        config.dynamic.lookback = 20
        config.dynamic.volatility_threshold = 0.6
        config.dynamic.trend_threshold = 0.5
        config.consensus.mode = "weighted"
        config.consensus.adaptive_weights = True
        config.consensus.performance_window = 10
        
        result = generate_signals_combined_all_strategy(
            high=high,
            low=low,
            close=close,
            length=osc_length,
            mult=osc_mult,
            config=config,
        )

        signals = result[0]
        confidence = result[3]

        if signals is None or signals.empty:
            return None

        non_nan_mask = ~signals.isna()
        if not non_nan_mask.any():
            return None

        latest_idx = signals[non_nan_mask].index[-1]
        latest_signal = int(signals.loc[latest_idx])
        latest_confidence = float(confidence.loc[latest_idx]) if confidence is not None and not confidence.empty else 0.0

        return (latest_signal, latest_confidence)

    except Exception as e:
        return None


def get_spc_signal(
    data_fetcher: DataFetcher,
    symbol: str,
    timeframe: str,
    limit: int,
    strategy: str = "cluster_transition",
    strategy_params: Optional[dict] = None,
    clustering_config: Optional[ClusteringConfig] = None,
) -> Optional[tuple]:
    """Calculate SPC signal for a symbol."""
    try:
        df, _ = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol,
            limit=limit,
            timeframe=timeframe,
            check_freshness=True,
        )

        if df is None or df.empty:
            return None

        if "high" not in df.columns or "low" not in df.columns or "close" not in df.columns:
            return None

        high = df["high"]
        low = df["low"]
        close = df["close"]

        if clustering_config is None:
            feature_config = FeatureConfig()
            clustering_config = ClusteringConfig(
                k=2,
                lookback=limit,
                p_low=5.0,
                p_high=95.0,
                main_plot="Clusters",
                feature_config=feature_config,
            )

        clustering = SimplifiedPercentileClustering(clustering_config)
        clustering_result = clustering.compute(high, low, close)

        strategy_params = strategy_params or {}
        
        if strategy == "cluster_transition":
            strategy_config = ClusterTransitionConfig(
                min_signal_strength=strategy_params.get("min_signal_strength", 0.3),
                min_rel_pos_change=strategy_params.get("min_rel_pos_change", 0.1),
                clustering_config=clustering_config,
            )
            signals, signal_strength, metadata = generate_signals_cluster_transition(
                high=high,
                low=low,
                close=close,
                clustering_result=clustering_result,
                config=strategy_config,
            )
        elif strategy == "regime_following":
            strategy_config = RegimeFollowingConfig(
                min_regime_strength=strategy_params.get("min_regime_strength", 0.7),
                min_cluster_duration=strategy_params.get("min_cluster_duration", 2),
                clustering_config=clustering_config,
            )
            signals, signal_strength, metadata = generate_signals_regime_following(
                high=high,
                low=low,
                close=close,
                clustering_result=clustering_result,
                config=strategy_config,
            )
        elif strategy == "mean_reversion":
            strategy_config = MeanReversionConfig(
                extreme_threshold=strategy_params.get("extreme_threshold", 0.2),
                min_extreme_duration=strategy_params.get("min_extreme_duration", 3),
                clustering_config=clustering_config,
            )
            signals, signal_strength, metadata = generate_signals_mean_reversion(
                high=high,
                low=low,
                close=close,
                clustering_result=clustering_result,
                config=strategy_config,
            )
        else:
            return None

        if signals is None or signals.empty:
            return None

        non_nan_mask = ~signals.isna()
        if not non_nan_mask.any():
            return None

        latest_idx = signals[non_nan_mask].index[-1]
        latest_signal = int(signals.loc[latest_idx])
        latest_strength = float(signal_strength.loc[latest_idx]) if not signal_strength.empty else 0.0

        return (latest_signal, latest_strength)

    except Exception as e:
        return None


def initialize_components() -> Tuple[ExchangeManager, DataFetcher]:
    """Initialize ExchangeManager and DataFetcher components."""
    log_progress("Initializing components...")
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)
    return exchange_manager, data_fetcher


def interactive_config_menu():
    """
    Interactive menu for configuring ATC + Range Oscillator + SPC Pure Voting.
    
    Returns:
        argparse.Namespace object with all configuration values
    """
    from modules.config import DEFAULT_TIMEFRAME
    
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("ATC + Range Oscillator + SPC Pure Voting - Configuration Menu", Fore.CYAN, Style.BRIGHT))
    print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    
    # Create namespace object
    class Config:
        pass
    
    config = Config()
    
    # 1. Timeframe selection
    print("\n" + color_text("1. TIMEFRAME SELECTION", Fore.YELLOW, Style.BRIGHT))
    config.timeframe = prompt_timeframe(default_timeframe=DEFAULT_TIMEFRAME)
    config.no_menu = True  # Already selected, skip menu
    
    # 2. Set default values (not shown in menu, can be changed in modules/config.py)
    config.limit = 500  # Default: 500 candles
    config.max_workers = 10  # Default: 10 parallel workers
    
    # 3. Range Oscillator parameters (loaded from config, not shown in menu)
    # Adjust these values in modules/config.py if needed
    from modules.config import RANGE_OSCILLATOR_LENGTH, RANGE_OSCILLATOR_MULTIPLIER
    config.osc_length = RANGE_OSCILLATOR_LENGTH
    config.osc_mult = RANGE_OSCILLATOR_MULTIPLIER
    
    # 4. SPC configuration (always enabled, all 3 strategies will be used)
    print("\n" + color_text("4. SPC (Simplified Percentile Clustering) CONFIGURATION", Fore.YELLOW, Style.BRIGHT))
    print("Note: All 3 SPC strategies will be calculated and aggregated into 1 vote")
    
    config.enable_spc = True  # Always enabled
    
    k_input = prompt_user_input("Number of clusters (2 or 3) [2]: ", default="2")
    config.spc_k = int(k_input) if k_input in ['2', '3'] else 2
    
    lookback_input = prompt_user_input("SPC lookback (historical bars) [1000]: ", default="1000")
    config.spc_lookback = int(lookback_input) if lookback_input.isdigit() else 1000
    
    p_low_input = prompt_user_input("Lower percentile [5.0]: ", default="5.0")
    config.spc_p_low = float(p_low_input) if p_low_input.replace('.', '').isdigit() else 5.0
    
    p_high_input = prompt_user_input("Upper percentile [95.0]: ", default="95.0")
    config.spc_p_high = float(p_high_input) if p_high_input.replace('.', '').isdigit() else 95.0
    
    # Strategy-specific parameters (loaded from config, not shown in menu)
    # Adjust these values in modules/config.py if needed
    from modules.config import SPC_STRATEGY_PARAMETERS
    config.spc_min_signal_strength = SPC_STRATEGY_PARAMETERS['cluster_transition']['min_signal_strength']
    config.spc_min_rel_pos_change = SPC_STRATEGY_PARAMETERS['cluster_transition']['min_rel_pos_change']
    config.spc_min_regime_strength = SPC_STRATEGY_PARAMETERS['regime_following']['min_regime_strength']
    config.spc_min_cluster_duration = SPC_STRATEGY_PARAMETERS['regime_following']['min_cluster_duration']
    config.spc_extreme_threshold = SPC_STRATEGY_PARAMETERS['mean_reversion']['extreme_threshold']
    config.spc_min_extreme_duration = SPC_STRATEGY_PARAMETERS['mean_reversion']['min_extreme_duration']
    config.spc_strategy = "all"  # Indicates all 3 strategies will be used
    
    # 5. Decision Matrix configuration (required for pure voting)
    print("\n" + color_text("5. DECISION MATRIX CONFIGURATION (Required)", Fore.YELLOW, Style.BRIGHT))
    threshold_input = prompt_user_input("Voting threshold (0.0-1.0) [0.5]: ", default="0.5")
    config.voting_threshold = float(threshold_input) if threshold_input.replace('.', '').isdigit() else 0.5
    
    min_votes_input = prompt_user_input("Minimum votes required [2]: ", default="2")
    config.min_votes = int(min_votes_input) if min_votes_input.isdigit() else 2
    
    # Set default values for other parameters
    config.ema_len = 28
    config.hma_len = 28
    config.wma_len = 28
    config.dema_len = 28
    config.lsma_len = 28
    config.kama_len = 28
    config.robustness = "Medium"
    config.lambda_param = 0.5
    config.decay = 0.1
    config.cutout = 5
    config.min_signal = 0.01
    config.max_symbols = None
    config.osc_strategies = None
    
    return config


def parse_args():
    """
    Parse command-line arguments or use interactive menu.
    
    If no arguments provided, shows interactive menu.
    Otherwise, parses command-line arguments.
    """
    from modules.config import DEFAULT_TIMEFRAME
    
    # Check if any arguments were provided
    if len(sys.argv) == 1:
        # No arguments, use interactive menu
        return interactive_config_menu()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="ATC + Range Oscillator + SPC Pure Voting Signal Filter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Copy all arguments from range_oscillator/cli/argument_parser.py
    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME,
        help=f"Timeframe for analysis (default: {DEFAULT_TIMEFRAME})")
    parser.add_argument("--no-menu", action="store_true",
        help="Disable interactive timeframe menu")
    parser.add_argument("--limit", type=int, default=500,
        help="Number of candles to fetch (default: 500)")
    parser.add_argument("--ema-len", type=int, default=28, help="EMA length (default: 28)")
    parser.add_argument("--hma-len", type=int, default=28, help="HMA length (default: 28)")
    parser.add_argument("--wma-len", type=int, default=28, help="WMA length (default: 28)")
    parser.add_argument("--dema-len", type=int, default=28, help="DEMA length (default: 28)")
    parser.add_argument("--lsma-len", type=int, default=28, help="LSMA length (default: 28)")
    parser.add_argument("--kama-len", type=int, default=28, help="KAMA length (default: 28)")
    parser.add_argument("--robustness", type=str, choices=["Narrow", "Medium", "Wide"],
        default="Medium", help="Robustness setting (default: Medium)")
    parser.add_argument("--lambda", type=float, default=0.5, dest="lambda_param",
        help="Lambda parameter (default: 0.5)")
    parser.add_argument("--decay", type=float, default=0.1, help="Decay rate (default: 0.1)")
    parser.add_argument("--cutout", type=int, default=5,
        help="Number of bars to skip at start (default: 5)")
    parser.add_argument("--min-signal", type=float, default=0.01,
        help="Minimum signal strength to display (default: 0.01)")
    parser.add_argument("--max-symbols", type=int, default=None,
        help="Maximum number of symbols to scan (default: None = all)")
    parser.add_argument("--osc-length", type=int, default=50,
        help="Range Oscillator length parameter (default: 50)")
    parser.add_argument("--osc-mult", type=float, default=2.0,
        help="Range Oscillator multiplier (default: 2.0)")
    parser.add_argument("--max-workers", type=int, default=10,
        help="Maximum number of parallel workers for Range Oscillator filtering (default: 10)")
    parser.add_argument("--osc-strategies", type=int, nargs="+", default=None,
        help="Range Oscillator strategies to use (e.g., --osc-strategies 5 6 7 8 9). Default: all [5, 6, 7, 8, 9]")
    
    # SPC configuration (same as hybrid)
    parser.add_argument("--enable-spc", action="store_true", help="Enable SPC (default: False)")
    parser.add_argument("--spc-strategy", type=str, default="cluster_transition",
        choices=["cluster_transition", "regime_following", "mean_reversion"],
        help="SPC strategy (default: cluster_transition)")
    parser.add_argument("--spc-k", type=int, default=2, choices=[2, 3], help="SPC clusters (default: 2)")
    parser.add_argument("--spc-lookback", type=int, default=1000, help="SPC lookback (default: 1000)")
    parser.add_argument("--spc-p-low", type=float, default=5.0, help="SPC lower percentile (default: 5.0)")
    parser.add_argument("--spc-p-high", type=float, default=95.0, help="SPC upper percentile (default: 95.0)")
    parser.add_argument("--spc-min-signal-strength", type=float, default=0.3, help="SPC min signal strength (default: 0.3)")
    parser.add_argument("--spc-min-rel-pos-change", type=float, default=0.1, help="SPC min rel pos change (default: 0.1)")
    parser.add_argument("--spc-min-regime-strength", type=float, default=0.7, help="SPC min regime strength (default: 0.7)")
    parser.add_argument("--spc-min-cluster-duration", type=int, default=2, help="SPC min cluster duration (default: 2)")
    parser.add_argument("--spc-extreme-threshold", type=float, default=0.2, help="SPC extreme threshold (default: 0.2)")
    parser.add_argument("--spc-min-extreme-duration", type=int, default=3, help="SPC min extreme duration (default: 3)")
    
    # Decision Matrix options (required for pure voting)
    parser.add_argument("--voting-threshold", type=float, default=0.5,
        help="Minimum weighted score for positive vote (default: 0.5)")
    parser.add_argument("--min-votes", type=int, default=2,
        help="Minimum number of indicators that must agree (default: 2)")
    
    return parser.parse_args()


class ATCOscillatorSPCVotingAnalyzer:
    """
    ATC + Range Oscillator + SPC Pure Voting Analyzer.
    
    Phương án 2: Thay thế hoàn toàn sequential filtering bằng voting system.
    """
    
    def __init__(self, args, data_fetcher: DataFetcher):
        """Initialize analyzer."""
        self.args = args
        self.data_fetcher = data_fetcher
        self.atc_analyzer = ATCAnalyzer(args, data_fetcher)
        self.selected_timeframe = args.timeframe
        self.atc_analyzer.selected_timeframe = args.timeframe
        
        # Initialize SPC Vote Aggregator
        aggregation_config = SPCAggregationConfig(
            mode=SPC_AGGREGATION_MODE,
            threshold=SPC_AGGREGATION_THRESHOLD,
            weighted_min_total=SPC_AGGREGATION_WEIGHTED_MIN_TOTAL,
            weighted_min_diff=SPC_AGGREGATION_WEIGHTED_MIN_DIFF,
            enable_adaptive_weights=SPC_AGGREGATION_ENABLE_ADAPTIVE_WEIGHTS,
            adaptive_performance_window=SPC_AGGREGATION_ADAPTIVE_PERFORMANCE_WINDOW,
            min_signal_strength=SPC_AGGREGATION_MIN_SIGNAL_STRENGTH,
            strategy_weights=SPC_AGGREGATION_STRATEGY_WEIGHTS,
        )
        self.spc_aggregator = SPCVoteAggregator(aggregation_config)
        
        # Results storage
        self.long_signals_atc = pd.DataFrame()
        self.short_signals_atc = pd.DataFrame()
        self.long_signals_final = pd.DataFrame()
        self.short_signals_final = pd.DataFrame()
    
    def determine_timeframe(self) -> str:
        """Determine timeframe from arguments and interactive menu."""
        self.selected_timeframe = self.args.timeframe
        
        if not self.args.no_menu:
            print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
            print(color_text("TIMEFRAME SELECTION", Fore.CYAN, Style.BRIGHT))
            print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
            self.selected_timeframe = prompt_timeframe(default_timeframe=self.selected_timeframe)
            print(color_text(f"\nSelected timeframe: {self.selected_timeframe}", Fore.GREEN))
        
        self.atc_analyzer.selected_timeframe = self.selected_timeframe
        return self.selected_timeframe
    
    def get_oscillator_params(self) -> dict:
        """Extract Range Oscillator parameters."""
        return {
            "osc_length": self.args.osc_length,
            "osc_mult": self.args.osc_mult,
            "max_workers": self.args.max_workers,
            "strategies": self.args.osc_strategies,
        }
    
    def get_spc_params(self) -> dict:
        """Extract SPC parameters for all 3 strategies."""
        # Use values from config if not provided in args
        cluster_transition_params = SPC_STRATEGY_PARAMETERS['cluster_transition'].copy()
        regime_following_params = SPC_STRATEGY_PARAMETERS['regime_following'].copy()
        mean_reversion_params = SPC_STRATEGY_PARAMETERS['mean_reversion'].copy()
        
        # Override with args if provided (for command-line usage)
        if hasattr(self.args, 'spc_min_signal_strength'):
            cluster_transition_params['min_signal_strength'] = self.args.spc_min_signal_strength
        if hasattr(self.args, 'spc_min_rel_pos_change'):
            cluster_transition_params['min_rel_pos_change'] = self.args.spc_min_rel_pos_change
        if hasattr(self.args, 'spc_min_regime_strength'):
            regime_following_params['min_regime_strength'] = self.args.spc_min_regime_strength
        if hasattr(self.args, 'spc_min_cluster_duration'):
            regime_following_params['min_cluster_duration'] = self.args.spc_min_cluster_duration
        if hasattr(self.args, 'spc_extreme_threshold'):
            mean_reversion_params['extreme_threshold'] = self.args.spc_extreme_threshold
        if hasattr(self.args, 'spc_min_extreme_duration'):
            mean_reversion_params['min_extreme_duration'] = self.args.spc_min_extreme_duration
        
        return {
            "k": self.args.spc_k,
            "lookback": self.args.spc_lookback,
            "p_low": self.args.spc_p_low,
            "p_high": self.args.spc_p_high,
            "cluster_transition_params": cluster_transition_params,
            "regime_following_params": regime_following_params,
            "mean_reversion_params": mean_reversion_params,
        }
    
    def display_config(self) -> None:
        """Display configuration information."""
        osc_params = self.get_oscillator_params()
        display_configuration(
            timeframe=self.selected_timeframe,
            limit=self.args.limit,
            min_signal=self.args.min_signal,
            max_workers=osc_params["max_workers"],
            strategies=osc_params["strategies"],
            max_symbols=self.args.max_symbols,
        )
        
        if self.args.enable_spc:
            spc_params = self.get_spc_params()
            log_progress("\nSPC Configuration (All 3 strategies enabled):")
            log_data(f"  K: {spc_params['k']}")
            log_data(f"  Lookback: {spc_params['lookback']}")
            log_data(f"  Percentiles: {spc_params['p_low']}% - {spc_params['p_high']}%")
            log_data(f"  Strategies: Cluster Transition, Regime Following, Mean Reversion")
        
        log_progress("\nDecision Matrix Configuration (Pure Voting):")
        log_data(f"  Voting Threshold: {self.args.voting_threshold}")
        log_data(f"  Min Votes: {self.args.min_votes}")
    
    def run_atc_scan(self) -> None:
        """Run ATC auto scan to get LONG/SHORT signals."""
        log_progress("\nStep 1: Running ATC auto scan...")
        log_progress("=" * 80)
        
        self.long_signals_atc, self.short_signals_atc = self.atc_analyzer.run_auto_scan()
        
        original_long_count = len(self.long_signals_atc)
        original_short_count = len(self.short_signals_atc)
        
        log_success(f"\nATC Scan Complete: Found {original_long_count} LONG + {original_short_count} SHORT signals")
        
        if self.long_signals_atc.empty and self.short_signals_atc.empty:
            log_warn("No ATC signals found. Exiting.")
            raise ValueError("No ATC signals found")
    
    def _process_symbol_for_all_indicators(
        self,
        symbol_data: Dict[str, Any],
        exchange_manager: ExchangeManager,
        timeframe: str,
        limit: int,
        signal_type: str,
        osc_params: dict,
        spc_params: Optional[dict],
    ) -> Optional[Dict[str, Any]]:
        """
        Worker function to calculate signals from all indicators in parallel.
        
        This is the key difference from hybrid approach - we calculate all signals
        at once instead of filtering sequentially.
        """
        try:
            data_fetcher = DataFetcher(exchange_manager)
            symbol = symbol_data["symbol"]
            expected_signal = 1 if signal_type == "LONG" else -1
            
            # Calculate all signals in parallel
            results = {
                "symbol": symbol,
                "signal": symbol_data["signal"],
                "trend": symbol_data["trend"],
                "price": symbol_data["price"],
                "exchange": symbol_data["exchange"],
            }
            
            # ATC signal (already have it from scan)
            atc_signal = symbol_data["signal"]
            atc_vote = 1 if atc_signal == expected_signal else 0
            atc_strength = abs(atc_signal) / 100.0 if atc_signal != 0 else 0.0
            results['atc_signal'] = atc_signal
            results['atc_vote'] = atc_vote
            results['atc_strength'] = min(atc_strength, 1.0)
            
            # Range Oscillator signal
            osc_result = get_range_oscillator_signal(
                data_fetcher=data_fetcher,
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                osc_length=osc_params["osc_length"],
                osc_mult=osc_params["osc_mult"],
                strategies=osc_params["strategies"],
            )
            
            if osc_result is not None:
                osc_signal, osc_confidence = osc_result
                osc_vote = 1 if osc_signal == expected_signal else 0
                results['osc_signal'] = osc_signal
                results['osc_vote'] = osc_vote
                results['osc_confidence'] = osc_confidence
            else:
                results['osc_signal'] = 0
                results['osc_vote'] = 0
                results['osc_confidence'] = 0.0
            
            # SPC signals from all 3 strategies (if enabled)
            if self.args.enable_spc and spc_params:
                feature_config = FeatureConfig()
                clustering_config = ClusteringConfig(
                    k=spc_params["k"],
                    lookback=spc_params["lookback"],
                    p_low=spc_params["p_low"],
                    p_high=spc_params["p_high"],
                    main_plot="Clusters",
                    feature_config=feature_config,
                )
                
                # Calculate signals from all 3 strategies
                # Cluster Transition
                ct_result = get_spc_signal(
                    data_fetcher=data_fetcher,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    strategy="cluster_transition",
                    strategy_params=spc_params["cluster_transition_params"],
                    clustering_config=clustering_config,
                )
                if ct_result is not None:
                    results['spc_cluster_transition_signal'] = ct_result[0]
                    results['spc_cluster_transition_strength'] = ct_result[1]
                else:
                    results['spc_cluster_transition_signal'] = 0
                    results['spc_cluster_transition_strength'] = 0.0
                
                # Regime Following
                rf_result = get_spc_signal(
                    data_fetcher=data_fetcher,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    strategy="regime_following",
                    strategy_params=spc_params["regime_following_params"],
                    clustering_config=clustering_config,
                )
                if rf_result is not None:
                    results['spc_regime_following_signal'] = rf_result[0]
                    results['spc_regime_following_strength'] = rf_result[1]
                else:
                    results['spc_regime_following_signal'] = 0
                    results['spc_regime_following_strength'] = 0.0
                
                # Mean Reversion
                mr_result = get_spc_signal(
                    data_fetcher=data_fetcher,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    strategy="mean_reversion",
                    strategy_params=spc_params["mean_reversion_params"],
                    clustering_config=clustering_config,
                )
                if mr_result is not None:
                    results['spc_mean_reversion_signal'] = mr_result[0]
                    results['spc_mean_reversion_strength'] = mr_result[1]
                else:
                    results['spc_mean_reversion_signal'] = 0
                    results['spc_mean_reversion_strength'] = 0.0
            
            return results
            
        except Exception as e:
            return None
    
    def calculate_signals_for_all_indicators(
        self,
        atc_signals_df: pd.DataFrame,
        signal_type: str,
    ) -> pd.DataFrame:
        """
        Calculate signals from all indicators in parallel.
        
        This replaces sequential filtering with parallel calculation.
        """
        if atc_signals_df.empty:
            return pd.DataFrame()

        osc_params = self.get_oscillator_params()
        spc_params = self.get_spc_params() if self.args.enable_spc else None
        total = len(atc_signals_df)
        
        log_progress(
            f"Calculating signals from all indicators for {total} {signal_type} symbols "
            f"(workers: {osc_params['max_workers']})..."
        )

        exchange_manager = self.data_fetcher.exchange_manager
        
        symbol_data_list = [
            {
                "symbol": row["symbol"],
                "signal": row["signal"],
                "trend": row["trend"],
                "price": row["price"],
                "exchange": row["exchange"],
            }
            for _, row in atc_signals_df.iterrows()
        ]

        progress_lock = threading.Lock()
        checked_count = [0]
        processed_count = [0]

        results = []
        
        with ThreadPoolExecutor(max_workers=osc_params["max_workers"]) as executor:
            future_to_symbol = {
                executor.submit(
                    self._process_symbol_for_all_indicators,
                    symbol_data,
                    exchange_manager,
                    self.selected_timeframe,
                    self.args.limit,
                    signal_type,
                    osc_params,
                    spc_params,
                ): symbol_data["symbol"]
                for symbol_data in symbol_data_list
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result is not None:
                        with progress_lock:
                            processed_count[0] += 1
                            results.append(result)
                except Exception as e:
                    pass
                finally:
                    with progress_lock:
                        checked_count[0] += 1
                        current_checked = checked_count[0]
                        current_processed = processed_count[0]
                        
                        if current_checked % 10 == 0 or current_checked == total:
                            log_progress(
                                f"Processed {current_checked}/{total} symbols... "
                                f"Got {current_processed} with all indicator signals"
                            )

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)
    
    def _aggregate_spc_votes(
        self,
        symbol_data: Dict[str, Any],
        signal_type: str,
        use_threshold_fallback: bool = False,
    ) -> Tuple[int, float]:
        """
        Aggregate 3 SPC strategy votes into a single vote.
        
        Uses SPCVoteAggregator with improved voting logic similar to Range Oscillator:
        - Separate LONG/SHORT weight calculation
        - Configurable consensus modes (threshold/weighted)
        - Optional adaptive weights based on performance
        - Signal strength filtering
        - Fallback to threshold mode if weighted mode gives no vote
        
        Args:
            symbol_data: Symbol data with SPC signals
            signal_type: "LONG" or "SHORT"
            use_threshold_fallback: If True, force use threshold mode
        
        Returns:
            (vote, strength) where vote is 1 if matches expected signal_type, 0 otherwise
        """
        expected_signal = 1 if signal_type == "LONG" else -1
        
        # Use threshold mode if fallback requested
        if use_threshold_fallback or self.spc_aggregator.config.mode == "threshold":
            # Temporarily switch to threshold mode
            original_mode = self.spc_aggregator.config.mode
            self.spc_aggregator.config.mode = "threshold"
            try:
                vote, strength, _ = self.spc_aggregator.aggregate(symbol_data, signal_type)
            finally:
                self.spc_aggregator.config.mode = original_mode
        else:
            # Try weighted mode first
            vote, strength, _ = self.spc_aggregator.aggregate(symbol_data, signal_type)
            
            # If weighted mode gives no vote (vote = 0), fallback to threshold mode
            if vote == 0:
                original_mode = self.spc_aggregator.config.mode
                self.spc_aggregator.config.mode = "threshold"
                try:
                    vote, strength, _ = self.spc_aggregator.aggregate(symbol_data, signal_type)
                finally:
                    self.spc_aggregator.config.mode = original_mode
        
        # Convert vote to 1/0 format for Decision Matrix compatibility
        # Only accept vote if it matches the expected signal direction
        final_vote = 1 if vote == expected_signal else 0
        return (final_vote, strength)
    
    def _get_indicator_accuracy(self, indicator: str, signal_type: str) -> float:
        """Get historical accuracy for an indicator from config."""
        return DECISION_MATRIX_INDICATOR_ACCURACIES.get(indicator, 0.5)
    
    def apply_voting_system(
        self,
        signals_df: pd.DataFrame,
        signal_type: str,
    ) -> pd.DataFrame:
        """
        Apply pure voting system to all signals.
        
        This is the core of Phương án 2 - no sequential filtering,
        just calculate all signals and vote.
        """
        if signals_df.empty:
            return pd.DataFrame()
        
        indicators = ['atc', 'oscillator']
        if self.args.enable_spc:
            indicators.append('spc')
        
        results = []
        
        for _, row in signals_df.iterrows():
            classifier = DecisionMatrixClassifier(indicators=indicators)
            
            # Get votes from all indicators
            atc_vote = row.get('atc_vote', 0)
            atc_strength = row.get('atc_strength', 0.0)
            classifier.add_node_vote('atc', atc_vote, atc_strength, 
                self._get_indicator_accuracy('atc', signal_type))
            
            osc_vote = row.get('osc_vote', 0)
            osc_strength = row.get('osc_confidence', 0.0)
            classifier.add_node_vote('oscillator', osc_vote, osc_strength,
                self._get_indicator_accuracy('oscillator', signal_type))
            
            if self.args.enable_spc:
                # Aggregate 3 SPC votes into 1
                spc_vote, spc_strength = self._aggregate_spc_votes(row.to_dict(), signal_type)
                classifier.add_node_vote('spc', spc_vote, spc_strength,
                    self._get_indicator_accuracy('spc', signal_type))
            
            classifier.calculate_weighted_impact()
            
            cumulative_vote, weighted_score, voting_breakdown = classifier.calculate_cumulative_vote(
                threshold=self.args.voting_threshold,
                min_votes=self.args.min_votes,
            )
            
            # Fallback logic: If SPC contribution = 0.00%, retry with threshold mode
            if self.args.enable_spc and 'spc' in voting_breakdown:
                spc_contribution = voting_breakdown['spc'].get('contribution', 0.0)
                if abs(spc_contribution) < 0.0001:  # Contribution ≈ 0.00%
                    # Re-aggregate SPC votes with threshold mode
                    spc_vote_fallback, spc_strength_fallback = self._aggregate_spc_votes(
                        row.to_dict(), 
                        signal_type, 
                        use_threshold_fallback=True
                    )
                    
                    # Update classifier with new SPC vote using add_node_vote
                    spc_accuracy = self._get_indicator_accuracy('spc', signal_type)
                    classifier.add_node_vote('spc', spc_vote_fallback, spc_strength_fallback, spc_accuracy)
                    
                    # Recalculate
                    classifier.calculate_weighted_impact()
                    cumulative_vote, weighted_score, voting_breakdown = classifier.calculate_cumulative_vote(
                        threshold=self.args.voting_threshold,
                        min_votes=self.args.min_votes,
                    )
            
            # Only keep if cumulative vote is positive
            if cumulative_vote == 1:
                result = row.to_dict()
                result['cumulative_vote'] = cumulative_vote
                result['weighted_score'] = weighted_score
                result['voting_breakdown'] = voting_breakdown
                
                metadata = classifier.get_metadata()
                result['feature_importance'] = metadata['feature_importance']
                result['weighted_impact'] = metadata['weighted_impact']
                result['independent_accuracy'] = metadata['independent_accuracy']
                
                votes_count = sum(v for v in classifier.node_votes.values())
                if votes_count == len(indicators):
                    result['source'] = 'ALL_INDICATORS'
                elif votes_count >= self.args.min_votes:
                    result['source'] = 'MAJORITY_VOTE'
                else:
                    result['source'] = 'WEIGHTED_VOTE'
                
                results.append(result)
        
        if not results:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values('weighted_score', ascending=False).reset_index(drop=True)
        
        return result_df
    
    def calculate_and_vote(self) -> None:
        """
        Calculate signals from all indicators and apply voting system.
        
        This is the main step for Phương án 2.
        """
        log_progress("\nStep 2: Calculating signals from all indicators...")
        log_progress("=" * 80)
        
        # Calculate all signals in parallel
        if not self.long_signals_atc.empty:
            long_with_signals = self.calculate_signals_for_all_indicators(
                atc_signals_df=self.long_signals_atc,
                signal_type="LONG",
            )
            
            # Apply voting system
            log_progress("\nStep 3: Applying voting system to LONG signals...")
            self.long_signals_final = self.apply_voting_system(long_with_signals, "LONG")
            log_progress(f"LONG signals: {len(self.long_signals_atc)} → {len(self.long_signals_final)} after voting")
        else:
            self.long_signals_final = pd.DataFrame()
        
        if not self.short_signals_atc.empty:
            short_with_signals = self.calculate_signals_for_all_indicators(
                atc_signals_df=self.short_signals_atc,
                signal_type="SHORT",
            )
            
            # Apply voting system
            log_progress("\nStep 3: Applying voting system to SHORT signals...")
            self.short_signals_final = self.apply_voting_system(short_with_signals, "SHORT")
            log_progress(f"SHORT signals: {len(self.short_signals_atc)} → {len(self.short_signals_final)} after voting")
        else:
            self.short_signals_final = pd.DataFrame()
    
    def display_results(self) -> None:
        """Display final results with voting metadata."""
        log_progress("\nStep 4: Displaying final results...")
        display_final_results(
            long_signals=self.long_signals_final,
            short_signals=self.short_signals_final,
            original_long_count=len(self.long_signals_atc),
            original_short_count=len(self.short_signals_atc),
            long_uses_fallback=False,
            short_uses_fallback=False,
        )
        
        # Display voting metadata
        if not self.long_signals_final.empty:
            self._display_voting_metadata(self.long_signals_final, "LONG")
        
        if not self.short_signals_final.empty:
            self._display_voting_metadata(self.short_signals_final, "SHORT")
    
    def _display_voting_metadata(self, signals_df: pd.DataFrame, signal_type: str) -> None:
        """Display voting metadata for signals."""
        if signals_df.empty:
            return
        
        log_progress(f"\n{signal_type} Signals - Voting Breakdown:")
        log_progress("-" * 80)
        
        for idx, row in signals_df.head(10).iterrows():
            symbol = row['symbol']
            weighted_score = row.get('weighted_score', 0.0)
            voting_breakdown = row.get('voting_breakdown', {})
            feature_importance = row.get('feature_importance', {})
            weighted_impact = row.get('weighted_impact', {})
            
            log_data(f"\nSymbol: {symbol}")
            log_data(f"  Weighted Score: {weighted_score:.2%}")
            log_data(f"  Voting Breakdown:")
            
            for indicator, vote_info in voting_breakdown.items():
                vote = vote_info['vote']
                weight = vote_info['weight']
                contribution = vote_info['contribution']
                importance = feature_importance.get(indicator, 0.0)
                impact = weighted_impact.get(indicator, 0.0)
                
                vote_str = "✓" if vote == 1 else "✗"
                log_data(
                    f"    {indicator.upper()}: {vote_str} "
                    f"(Weight: {weight:.1%}, Impact: {impact:.1%}, "
                    f"Importance: {importance:.1%}, Contribution: {contribution:.2%})"
                )
    
    def run(self) -> None:
        """
        Run the complete Pure Voting System workflow.
        
        Workflow:
        1. Determine timeframe
        2. Display configuration
        3. Run ATC auto scan
        4. Calculate signals from all indicators in parallel
        5. Apply voting system
        6. Display final results
        """
        self.determine_timeframe()
        self.display_config()
        log_progress("Initializing components...")
        
        self.run_atc_scan()
        self.calculate_and_vote()
        self.display_results()
        
        log_success("\nAnalysis complete!")


def main() -> None:
    """Main function for Pure Voting System workflow."""
    args = parse_args()
    exchange_manager, data_fetcher = initialize_components()
    analyzer = ATCOscillatorSPCVotingAnalyzer(args, data_fetcher)
    analyzer.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(color_text("\nExiting program by user request.", Fore.YELLOW))
        sys.exit(0)
    except Exception as e:
        log_error(f"Error: {type(e).__name__}: {e}")
        import traceback
        log_error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

