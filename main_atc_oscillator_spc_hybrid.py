"""
ATC + Range Oscillator + SPC Hybrid Approach (Phương án 1).

This program combines signals from:
1. Adaptive Trend Classification (ATC)
2. Range Oscillator
3. Simplified Percentile Clustering (SPC)

Workflow (Hybrid Approach):
1. Runs ATC auto scan to find LONG/SHORT signals
2. Filters symbols by checking if Range Oscillator signals match ATC signals
3. Filters symbols by checking if SPC signals match (optional)
4. Applies Decision Matrix voting system (optional)
5. Returns final list of symbols with confirmed signals

Phương án 1: Kết hợp sequential filtering và voting system
"""

import warnings
import sys
import threading
from typing import Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from modules.common.utils import configure_windows_stdio

# Fix encoding issues on Windows for interactive CLI runs only
configure_windows_stdio()

from colorama import Fore, Style, init as colorama_init

from modules.config import (
    DEFAULT_TIMEFRAME,
    DECISION_MATRIX_SPC_STRATEGY_ACCURACIES,
    DECISION_MATRIX_INDICATOR_ACCURACIES,
    SPC_STRATEGY_PARAMETERS,
    RANGE_OSCILLATOR_LENGTH,
    RANGE_OSCILLATOR_MULTIPLIER,
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
from modules.decision_matrix.classifier import DecisionMatrixClassifier

# Suppress warnings for cleaner output
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


def interactive_config_menu():
    """
    Interactive menu for configuring ATC + Range Oscillator + SPC Hybrid.
    
    Returns:
        argparse.Namespace object with all configuration values
    """
    from modules.config import DEFAULT_TIMEFRAME
    
    print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
    print(color_text("ATC + Range Oscillator + SPC Hybrid - Configuration Menu", Fore.CYAN, Style.BRIGHT))
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
    print("Note: All 3 SPC strategies will be calculated and used in Decision Matrix")
    
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
    
    # 5. Decision Matrix configuration (always enabled when SPC is enabled)
    print("\n" + color_text("5. DECISION MATRIX CONFIGURATION", Fore.YELLOW, Style.BRIGHT))
    print("Note: Decision Matrix is required when using all 3 SPC strategies")
    config.use_decision_matrix = True  # Always enabled
    
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
    config.spc_strategy = "all"  # Indicates all 3 strategies will be used
    
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
        description="ATC + Range Oscillator + SPC Hybrid Signal Filter",
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
    
    # SPC configuration (always enabled, all 3 strategies used)
    parser.add_argument(
        "--enable-spc",
        action="store_true",
        default=True,
        help="Enable SPC (always enabled, all 3 strategies used) (default: True)",
    )
    parser.add_argument(
        "--spc-k",
        type=int,
        default=2,
        choices=[2, 3],
        help="Number of clusters for SPC (default: 2)",
    )
    parser.add_argument(
        "--spc-lookback",
        type=int,
        default=1000,
        help="Historical bars for SPC (default: 1000)",
    )
    parser.add_argument(
        "--spc-p-low",
        type=float,
        default=5.0,
        help="Lower percentile for SPC (default: 5.0)",
    )
    parser.add_argument(
        "--spc-p-high",
        type=float,
        default=95.0,
        help="Upper percentile for SPC (default: 95.0)",
    )
    parser.add_argument(
        "--spc-min-signal-strength",
        type=float,
        default=0.3,
        help="Minimum signal strength for SPC cluster transition (default: 0.3)",
    )
    parser.add_argument(
        "--spc-min-rel-pos-change",
        type=float,
        default=0.1,
        help="Minimum relative position change for SPC cluster transition (default: 0.1)",
    )
    parser.add_argument(
        "--spc-min-regime-strength",
        type=float,
        default=0.7,
        help="Minimum regime strength for SPC regime following (default: 0.7)",
    )
    parser.add_argument(
        "--spc-min-cluster-duration",
        type=int,
        default=2,
        help="Minimum bars in same cluster for SPC regime following (default: 2)",
    )
    parser.add_argument(
        "--spc-extreme-threshold",
        type=float,
        default=0.2,
        help="Real_clust threshold for extreme in SPC mean reversion (default: 0.2)",
    )
    parser.add_argument(
        "--spc-min-extreme-duration",
        type=int,
        default=3,
        help="Minimum bars in extreme for SPC mean reversion (default: 3)",
    )
    
    # Decision Matrix options (always enabled when SPC is enabled)
    parser.add_argument(
        "--use-decision-matrix",
        action="store_true",
        default=True,
        help="Use decision matrix voting system (always enabled with SPC) (default: True)",
    )
    parser.add_argument(
        "--voting-threshold",
        type=float,
        default=0.5,
        help="Minimum weighted score for positive vote (default: 0.5)",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=2,
        help="Minimum number of indicators that must agree (default: 2)",
    )
    
    args = parser.parse_args()
    
    # Ensure SPC and Decision Matrix are always enabled
    args.enable_spc = True
    args.use_decision_matrix = True
    
    return args


def initialize_components() -> Tuple[ExchangeManager, DataFetcher]:
    """Initialize ExchangeManager and DataFetcher components."""
    log_progress("Initializing components...")
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)
    return exchange_manager, data_fetcher


class ATCOscillatorSPCHybridAnalyzer:
    """
    ATC + Range Oscillator + SPC Hybrid Analyzer.
    
    Phương án 1: Kết hợp sequential filtering và voting system.
    """
    
    def __init__(self, args, data_fetcher: DataFetcher):
        """Initialize analyzer."""
        self.args = args
        self.data_fetcher = data_fetcher
        self.atc_analyzer = ATCAnalyzer(args, data_fetcher)
        self.selected_timeframe = args.timeframe
        self.atc_analyzer.selected_timeframe = args.timeframe
        
        # Results storage
        self.long_signals_atc = pd.DataFrame()
        self.short_signals_atc = pd.DataFrame()
        self.long_signals_confirmed = pd.DataFrame()
        self.short_signals_confirmed = pd.DataFrame()
        self.long_uses_fallback = False
        self.short_uses_fallback = False
    
    def determine_timeframe(self) -> str:
        """Determine timeframe from arguments and interactive menu."""
        self.selected_timeframe = self.args.timeframe
        
        if not self.args.no_menu:
            print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
            print(color_text("ATC PHASE - TIMEFRAME SELECTION", Fore.CYAN, Style.BRIGHT))
            print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
            self.selected_timeframe = prompt_timeframe(default_timeframe=self.selected_timeframe)
            print(color_text(f"\nSelected timeframe for ATC analysis: {self.selected_timeframe}", Fore.GREEN))
        
        self.atc_analyzer.selected_timeframe = self.selected_timeframe
        return self.selected_timeframe
    
    def get_oscillator_params(self) -> dict:
        """Extract Range Oscillator parameters from arguments."""
        return {
            "osc_length": self.args.osc_length,
            "osc_mult": self.args.osc_mult,
            "max_workers": self.args.max_workers,
            "strategies": self.args.osc_strategies,
        }
    
    def get_spc_params(self) -> dict:
        """Extract SPC parameters from arguments for all 3 strategies."""
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
        
        if self.args.use_decision_matrix:
            log_progress("\nDecision Matrix Configuration:")
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
    
    def _process_symbol_for_oscillator(
        self,
        symbol_data: Dict[str, Any],
        exchange_manager: ExchangeManager,
        timeframe: str,
        limit: int,
        expected_osc_signal: int,
        osc_length: int,
        osc_mult: float,
        strategies: Optional[list] = None,
    ) -> Optional[Dict[str, Any]]:
        """Worker function to process a single symbol for Range Oscillator confirmation."""
        try:
            data_fetcher = DataFetcher(exchange_manager)
            symbol = symbol_data["symbol"]
            
            osc_result = get_range_oscillator_signal(
                data_fetcher=data_fetcher,
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                osc_length=osc_length,
                osc_mult=osc_mult,
                strategies=strategies,
            )

            if osc_result is None:
                return None
            
            osc_signal, osc_confidence = osc_result

            if osc_signal == expected_osc_signal:
                return {
                    "symbol": symbol,
                    "signal": symbol_data["signal"],
                    "trend": symbol_data["trend"],
                    "price": symbol_data["price"],
                    "exchange": symbol_data["exchange"],
                    "osc_signal": osc_signal,
                    "osc_confidence": osc_confidence,
                }
            
            return None
            
        except Exception as e:
            return None
    
    def _process_symbol_for_spc(
        self,
        symbol_data: Dict[str, Any],
        exchange_manager: ExchangeManager,
        timeframe: str,
        limit: int,
        spc_params: dict,
    ) -> Optional[Dict[str, Any]]:
        """Worker function to calculate SPC signals from all 3 strategies for a symbol."""
        try:
            data_fetcher = DataFetcher(exchange_manager)
            symbol = symbol_data["symbol"]
            
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
            result = {
                "symbol": symbol,
                "signal": symbol_data["signal"],
                "trend": symbol_data["trend"],
                "price": symbol_data["price"],
                "exchange": symbol_data["exchange"],
            }
            
            # Copy existing fields
            if "osc_signal" in symbol_data:
                result["osc_signal"] = symbol_data["osc_signal"]
                result["osc_confidence"] = symbol_data.get("osc_confidence", 0.0)
            if "source" in symbol_data:
                result["source"] = symbol_data["source"]
            
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
                result["spc_cluster_transition_signal"] = ct_result[0]
                result["spc_cluster_transition_strength"] = ct_result[1]
            else:
                result["spc_cluster_transition_signal"] = 0
                result["spc_cluster_transition_strength"] = 0.0
            
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
                result["spc_regime_following_signal"] = rf_result[0]
                result["spc_regime_following_strength"] = rf_result[1]
            else:
                result["spc_regime_following_signal"] = 0
                result["spc_regime_following_strength"] = 0.0
            
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
                result["spc_mean_reversion_signal"] = mr_result[0]
                result["spc_mean_reversion_strength"] = mr_result[1]
            else:
                result["spc_mean_reversion_signal"] = 0
                result["spc_mean_reversion_strength"] = 0.0
            
            return result
            
        except Exception as e:
            return None
    
    def filter_signals_by_range_oscillator(
        self,
        atc_signals_df: pd.DataFrame,
        signal_type: str,
    ) -> pd.DataFrame:
        """Filter ATC signals by checking Range Oscillator confirmation."""
        if atc_signals_df.empty:
            return pd.DataFrame()

        osc_params = self.get_oscillator_params()
        expected_osc_signal = 1 if signal_type == "LONG" else -1
        total = len(atc_signals_df)
        
        log_progress(
            f"Checking Range Oscillator signals for {total} {signal_type} symbols "
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
        confirmed_count = [0]

        filtered_results = []
        
        with ThreadPoolExecutor(max_workers=osc_params["max_workers"]) as executor:
            future_to_symbol = {
                executor.submit(
                    self._process_symbol_for_oscillator,
                    symbol_data,
                    exchange_manager,
                    self.selected_timeframe,
                    self.args.limit,
                    expected_osc_signal,
                    osc_params["osc_length"],
                    osc_params["osc_mult"],
                    osc_params["strategies"],
                ): symbol_data["symbol"]
                for symbol_data in symbol_data_list
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result is not None:
                        with progress_lock:
                            confirmed_count[0] += 1
                            filtered_results.append(result)
                except Exception as e:
                    pass
                finally:
                    with progress_lock:
                        checked_count[0] += 1
                        current_checked = checked_count[0]
                        current_confirmed = confirmed_count[0]
                        
                        if current_checked % 10 == 0 or current_checked == total:
                            log_progress(
                                f"Checked {current_checked}/{total} symbols... "
                                f"Found {current_confirmed} confirmed {signal_type} signals"
                            )

        if not filtered_results:
            return pd.DataFrame()

        filtered_df = pd.DataFrame(filtered_results)
        
        if "osc_confidence" in filtered_df.columns:
            if signal_type == "LONG":
                filtered_df = filtered_df.sort_values(
                    ["osc_confidence", "signal"], 
                    ascending=[False, False]
                ).reset_index(drop=True)
            else:
                filtered_df = filtered_df.sort_values(
                    ["osc_confidence", "signal"], 
                    ascending=[False, True]
                ).reset_index(drop=True)
        else:
            if signal_type == "LONG":
                filtered_df = filtered_df.sort_values("signal", ascending=False).reset_index(drop=True)
            else:
                filtered_df = filtered_df.sort_values("signal", ascending=True).reset_index(drop=True)

        return filtered_df
    
    def calculate_spc_signals(
        self,
        signals_df: pd.DataFrame,
        signal_type: str,
    ) -> pd.DataFrame:
        """Calculate SPC signals from all 3 strategies for all symbols."""
        if signals_df.empty:
            return pd.DataFrame()

        spc_params = self.get_spc_params()
        total = len(signals_df)
        
        log_progress(
            f"Calculating SPC signals (all 3 strategies) for {total} {signal_type} symbols "
            f"(workers: {self.args.max_workers})..."
        )

        exchange_manager = self.data_fetcher.exchange_manager
        
        symbol_data_list = [row.to_dict() for _, row in signals_df.iterrows()]

        progress_lock = threading.Lock()
        checked_count = [0]

        results = []
        
        with ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self._process_symbol_for_spc,
                    symbol_data,
                    exchange_manager,
                    self.selected_timeframe,
                    self.args.limit,
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
                            results.append(result)
                except Exception as e:
                    pass
                finally:
                    with progress_lock:
                        checked_count[0] += 1
                        current_checked = checked_count[0]
                        
                        if current_checked % 10 == 0 or current_checked == total:
                            log_progress(
                                f"Calculated SPC signals for {current_checked}/{total} symbols..."
                            )

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        
        # Sort by signal strength (use average of all 3 strategies)
        if signal_type == "LONG":
            result_df = result_df.sort_values("signal", ascending=False).reset_index(drop=True)
        else:
            result_df = result_df.sort_values("signal", ascending=True).reset_index(drop=True)

        return result_df
    
    def _aggregate_spc_votes(
        self,
        symbol_data: Dict[str, Any],
        signal_type: str,
    ) -> Tuple[int, float]:
        """
        Aggregate 3 SPC strategy votes into a single vote.
        
        Uses weighted average based on strategy accuracies:
        - Cluster Transition: 0.68
        - Regime Following: 0.66
        - Mean Reversion: 0.64
        
        Returns:
            (vote, strength) where vote is 1 if weighted average > 0.5, else 0
        """
        expected_signal = 1 if signal_type == "LONG" else -1
        
        # Strategy accuracies (used as weights) - from config
        accuracies = DECISION_MATRIX_SPC_STRATEGY_ACCURACIES
        
        # Get votes and strengths from all 3 strategies
        ct_signal = symbol_data.get('spc_cluster_transition_signal', 0)
        ct_vote = 1 if ct_signal == expected_signal else 0
        ct_strength = symbol_data.get('spc_cluster_transition_strength', 0.0)
        
        rf_signal = symbol_data.get('spc_regime_following_signal', 0)
        rf_vote = 1 if rf_signal == expected_signal else 0
        rf_strength = symbol_data.get('spc_regime_following_strength', 0.0)
        
        mr_signal = symbol_data.get('spc_mean_reversion_signal', 0)
        mr_vote = 1 if mr_signal == expected_signal else 0
        mr_strength = symbol_data.get('spc_mean_reversion_strength', 0.0)
        
        # Calculate weighted average
        total_weight = sum(accuracies.values())
        weighted_vote = (
            ct_vote * accuracies['cluster_transition'] +
            rf_vote * accuracies['regime_following'] +
            mr_vote * accuracies['mean_reversion']
        ) / total_weight
        
        weighted_strength = (
            ct_strength * accuracies['cluster_transition'] +
            rf_strength * accuracies['regime_following'] +
            mr_strength * accuracies['mean_reversion']
        ) / total_weight
        
        # Final vote: 1 if weighted average > 0.5, else 0
        final_vote = 1 if weighted_vote > 0.5 else 0
        
        return (final_vote, min(weighted_strength, 1.0))
    
    def calculate_indicator_votes(
        self,
        symbol_data: Dict[str, Any],
        signal_type: str,
    ) -> Dict[str, Tuple[int, float]]:
        """Calculate votes from all indicators for a symbol (SPC votes aggregated into 1)."""
        expected_signal = 1 if signal_type == "LONG" else -1
        votes = {}
        
        # ATC vote (always 1 if symbol passed ATC scan)
        atc_signal = symbol_data.get('signal', 0)
        atc_vote = 1 if atc_signal == expected_signal else 0
        atc_strength = abs(atc_signal) / 100.0 if atc_signal != 0 else 0.0
        votes['atc'] = (atc_vote, min(atc_strength, 1.0))
        
        # Range Oscillator vote
        osc_signal = symbol_data.get('osc_signal', 0)
        osc_vote = 1 if osc_signal == expected_signal else 0
        osc_strength = symbol_data.get('osc_confidence', 0.0)
        votes['oscillator'] = (osc_vote, osc_strength)
        
        # SPC vote (aggregated from all 3 strategies)
        if self.args.enable_spc:
            spc_vote, spc_strength = self._aggregate_spc_votes(symbol_data, signal_type)
            votes['spc'] = (spc_vote, spc_strength)
        
        return votes
    
    def _get_indicator_accuracy(self, indicator: str, signal_type: str) -> float:
        """Get historical accuracy for an indicator from config."""
        return DECISION_MATRIX_INDICATOR_ACCURACIES.get(indicator, 0.5)
    
    def apply_decision_matrix(
        self,
        signals_df: pd.DataFrame,
        signal_type: str,
    ) -> pd.DataFrame:
        """Apply decision matrix voting system to filter signals."""
        if signals_df.empty:
            return pd.DataFrame()
        
        # Build indicators list: ATC, Oscillator, and aggregated SPC
        indicators = ['atc', 'oscillator']
        if self.args.enable_spc:
            indicators.append('spc')
        
        results = []
        
        for _, row in signals_df.iterrows():
            classifier = DecisionMatrixClassifier(indicators=indicators)
            
            votes = self.calculate_indicator_votes(row.to_dict(), signal_type)
            
            for indicator, (vote, strength) in votes.items():
                accuracy = self._get_indicator_accuracy(indicator, signal_type)
                classifier.add_node_vote(indicator, vote, strength, accuracy)
            
            classifier.calculate_weighted_impact()
            
            cumulative_vote, weighted_score, voting_breakdown = classifier.calculate_cumulative_vote(
                threshold=self.args.voting_threshold,
                min_votes=self.args.min_votes,
            )
            
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
                elif votes_count >= 2:  # At least 2 out of 3 indicators agree
                    result['source'] = 'MAJORITY_VOTE'
                else:
                    result['source'] = 'WEIGHTED_VOTE'
                
                results.append(result)
        
        if not results:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values('weighted_score', ascending=False).reset_index(drop=True)
        
        return result_df
    
    def filter_by_oscillator(self) -> None:
        """Filter ATC signals by Range Oscillator confirmation."""
        log_progress("\nStep 2: Filtering by Range Oscillator confirmation...")
        log_progress("=" * 80)

        if not self.long_signals_atc.empty:
            self.long_signals_confirmed = self.filter_signals_by_range_oscillator(
                atc_signals_df=self.long_signals_atc,
                signal_type="LONG",
            )
            
            if self.long_signals_confirmed.empty:
                log_warn("No LONG signals confirmed by Range Oscillator. Falling back to ATC signals only.")
                self.long_signals_confirmed = self.long_signals_atc.copy()
                self.long_signals_confirmed['source'] = 'ATC_ONLY'
                self.long_uses_fallback = True
            else:
                self.long_signals_confirmed['source'] = 'ATC_OSCILLATOR'
                self.long_uses_fallback = False
        else:
            self.long_signals_confirmed = pd.DataFrame()
            self.long_uses_fallback = False

        if not self.short_signals_atc.empty:
            self.short_signals_confirmed = self.filter_signals_by_range_oscillator(
                atc_signals_df=self.short_signals_atc,
                signal_type="SHORT",
            )
            
            if self.short_signals_confirmed.empty:
                log_warn("No SHORT signals confirmed by Range Oscillator. Falling back to ATC signals only.")
                self.short_signals_confirmed = self.short_signals_atc.copy()
                self.short_signals_confirmed['source'] = 'ATC_ONLY'
                self.short_uses_fallback = True
            else:
                self.short_signals_confirmed['source'] = 'ATC_OSCILLATOR'
                self.short_uses_fallback = False
        else:
            self.short_signals_confirmed = pd.DataFrame()
            self.short_uses_fallback = False
    
    def calculate_spc_signals_for_all(self) -> None:
        """Calculate SPC signals from all 3 strategies for all confirmed signals."""
        log_progress("\nStep 3: Calculating SPC signals (all 3 strategies)...")
        log_progress("=" * 80)

        if not self.long_signals_confirmed.empty:
            self.long_signals_confirmed = self.calculate_spc_signals(
                signals_df=self.long_signals_confirmed,
                signal_type="LONG",
            )
            log_progress(f"Calculated SPC signals for {len(self.long_signals_confirmed)} LONG symbols")

        if not self.short_signals_confirmed.empty:
            self.short_signals_confirmed = self.calculate_spc_signals(
                signals_df=self.short_signals_confirmed,
                signal_type="SHORT",
            )
            log_progress(f"Calculated SPC signals for {len(self.short_signals_confirmed)} SHORT symbols")
    
    def filter_by_decision_matrix(self) -> None:
        """Filter signals using decision matrix voting system."""
        log_progress("\nStep 4: Applying Decision Matrix voting system...")
        log_progress("=" * 80)

        if not self.long_signals_confirmed.empty:
            long_before = len(self.long_signals_confirmed)
            self.long_signals_confirmed = self.apply_decision_matrix(
                self.long_signals_confirmed,
                "LONG",
            )
            long_after = len(self.long_signals_confirmed)
            log_progress(f"LONG signals: {long_before} → {long_after} after voting")
        else:
            self.long_signals_confirmed = pd.DataFrame()

        if not self.short_signals_confirmed.empty:
            short_before = len(self.short_signals_confirmed)
            self.short_signals_confirmed = self.apply_decision_matrix(
                self.short_signals_confirmed,
                "SHORT",
            )
            short_after = len(self.short_signals_confirmed)
            log_progress(f"SHORT signals: {short_before} → {short_after} after voting")
        else:
            self.short_signals_confirmed = pd.DataFrame()
    
    def display_results(self) -> None:
        """Display final filtered results."""
        log_progress("\nStep 5: Displaying final results...")
        display_final_results(
            long_signals=self.long_signals_confirmed,
            short_signals=self.short_signals_confirmed,
            original_long_count=len(self.long_signals_atc),
            original_short_count=len(self.short_signals_atc),
            long_uses_fallback=self.long_uses_fallback,
            short_uses_fallback=self.short_uses_fallback,
        )
        
        # Display voting metadata if decision matrix was used
        if self.args.use_decision_matrix and not self.long_signals_confirmed.empty:
            self._display_voting_metadata(self.long_signals_confirmed, "LONG")
        
        if self.args.use_decision_matrix and not self.short_signals_confirmed.empty:
            self._display_voting_metadata(self.short_signals_confirmed, "SHORT")
    
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
        Run the complete ATC + Range Oscillator + SPC Hybrid workflow.
        
        Workflow:
        1. Determine timeframe
        2. Display configuration
        3. Run ATC auto scan
        4. Filter by Range Oscillator confirmation
        5. Filter by SPC confirmation (if enabled)
        6. Apply Decision Matrix voting (if enabled)
        7. Display final results
        """
        self.determine_timeframe()
        self.display_config()
        log_progress("Initializing components...")
        
        self.run_atc_scan()
        self.filter_by_oscillator()
        
        if self.args.enable_spc:
            self.calculate_spc_signals_for_all()
        
        if self.args.use_decision_matrix:
            self.filter_by_decision_matrix()
        
        self.display_results()
        log_success("\nAnalysis complete!")


def main() -> None:
    """Main function for ATC + Range Oscillator + SPC Hybrid workflow."""
    args = parse_args()
    exchange_manager, data_fetcher = initialize_components()
    analyzer = ATCOscillatorSPCHybridAnalyzer(args, data_fetcher)
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

