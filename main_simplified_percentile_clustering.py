"""
Simplified Percentile Clustering (SPC) Main Program

Analyzes futures pairs on Binance using Simplified Percentile Clustering:
- Fetches OHLCV data from Binance futures
- Calculates SPC signals using cluster analysis
- Displays cluster signals and analysis
"""

import warnings
import sys
import argparse
from typing import Optional, Tuple
import pandas as pd

from modules.common.utils import configure_windows_stdio

# Fix encoding issues on Windows for interactive CLI runs only
configure_windows_stdio()

from colorama import Fore, Style, init as colorama_init

from modules.config import (
    DEFAULT_SYMBOL,
    DEFAULT_QUOTE,
    DEFAULT_TIMEFRAME,
)
from modules.common.utils import (
    color_text,
    normalize_symbol,
    log_error,
    log_analysis,
    log_data,
    log_progress,
    prompt_user_input,
)
from modules.common.ExchangeManager import ExchangeManager
from modules.common.DataFetcher import DataFetcher
from modules.simplified_percentile_clustering.core.clustering import (
    SimplifiedPercentileClustering,
    ClusteringConfig,
)
from modules.simplified_percentile_clustering.core.features import FeatureConfig
from modules.simplified_percentile_clustering.strategies import (
    generate_signals_cluster_transition,
    generate_signals_regime_following,
    generate_signals_mean_reversion,
    ClusterTransitionConfig,
    RegimeFollowingConfig,
    MeanReversionConfig,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
colorama_init(autoreset=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for SPC analysis."""
    parser = argparse.ArgumentParser(
        description="Simplified Percentile Clustering (SPC) Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Symbol and data options
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help=f"Trading pair symbol (default: {DEFAULT_SYMBOL})",
    )
    parser.add_argument(
        "--quote",
        type=str,
        default=DEFAULT_QUOTE,
        help=f"Quote currency (default: {DEFAULT_QUOTE})",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=DEFAULT_TIMEFRAME,
        help=f"Timeframe (default: {DEFAULT_TIMEFRAME})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of candles to fetch (default: 1000)",
    )

    # Clustering parameters
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        choices=[2, 3],
        help="Number of clusters (default: 2)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=1000,
        help="Historical bars for percentile/mean calculations (default: 1000)",
    )
    parser.add_argument(
        "--p-low",
        type=float,
        default=5.0,
        dest="p_low",
        help="Lower percentile (default: 5.0)",
    )
    parser.add_argument(
        "--p-high",
        type=float,
        default=95.0,
        dest="p_high",
        help="Upper percentile (default: 95.0)",
    )
    parser.add_argument(
        "--main-plot",
        type=str,
        default="Clusters",
        choices=["Clusters", "RSI", "CCI", "Fisher", "DMI", "Z-Score", "MAR"],
        dest="main_plot",
        help="Main plot mode (default: Clusters)",
    )

    # Feature configuration
    parser.add_argument(
        "--rsi-len",
        type=int,
        default=14,
        dest="rsi_len",
        help="RSI length (default: 14)",
    )
    parser.add_argument(
        "--cci-len",
        type=int,
        default=20,
        dest="cci_len",
        help="CCI length (default: 20)",
    )
    parser.add_argument(
        "--fisher-len",
        type=int,
        default=9,
        dest="fisher_len",
        help="Fisher Transform length (default: 9)",
    )
    parser.add_argument(
        "--dmi-len",
        type=int,
        default=9,
        dest="dmi_len",
        help="DMI length (default: 9)",
    )
    parser.add_argument(
        "--zscore-len",
        type=int,
        default=20,
        dest="zscore_len",
        help="Z-Score length (default: 20)",
    )
    parser.add_argument(
        "--mar-len",
        type=int,
        default=14,
        dest="mar_len",
        help="MAR length (default: 14)",
    )

    # Strategy selection
    parser.add_argument(
        "--strategy",
        type=str,
        default="cluster_transition",
        choices=["cluster_transition", "regime_following", "mean_reversion"],
        help="Trading strategy to use (default: cluster_transition)",
    )

    # Strategy-specific parameters for Cluster Transition
    parser.add_argument(
        "--min-signal-strength",
        type=float,
        default=0.3,
        dest="min_signal_strength",
        help="Minimum signal strength for cluster transition (default: 0.3)",
    )
    parser.add_argument(
        "--min-rel-pos-change",
        type=float,
        default=0.1,
        dest="min_rel_pos_change",
        help="Minimum relative position change for cluster transition (default: 0.1)",
    )
    
    # Strategy-specific parameters for Regime Following
    parser.add_argument(
        "--min-regime-strength",
        type=float,
        default=0.7,
        dest="min_regime_strength",
        help="Minimum regime strength for regime following (default: 0.7)",
    )
    parser.add_argument(
        "--min-cluster-duration",
        type=int,
        default=2,
        dest="min_cluster_duration",
        help="Minimum bars in same cluster for regime following (default: 2)",
    )
    
    # Strategy-specific parameters for Mean Reversion
    parser.add_argument(
        "--extreme-threshold",
        type=float,
        default=0.2,
        dest="extreme_threshold",
        help="Real_clust threshold for extreme in mean reversion (default: 0.2)",
    )
    parser.add_argument(
        "--min-extreme-duration",
        type=int,
        default=3,
        dest="min_extreme_duration",
        help="Minimum bars in extreme for mean reversion (default: 3)",
    )

    # Display options
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disable interactive prompts",
    )
    parser.add_argument(
        "--list-symbols",
        action="store_true",
        help="List available futures symbols and exit",
    )

    return parser.parse_args()


def display_spc_signals(
    symbol: str,
    df: pd.DataFrame,
    signals: pd.Series,
    clustering_result,
    current_price: float,
    exchange_label: str,
) -> None:
    """
    Display SPC analysis results and signals.
    
    Args:
        symbol: Trading pair symbol
        df: DataFrame with OHLCV data
        signals: Series with trading signals
        clustering_result: ClusteringResult object
        current_price: Current price
        exchange_label: Exchange label
    """
    if log_analysis:
        log_analysis("=" * 80)
        log_analysis("SIMPLIFIED PERCENTILE CLUSTERING (SPC) ANALYSIS")
        log_analysis("=" * 80)
    
    if log_data:
        log_data(f"Symbol: {symbol}")
        log_data(f"Exchange: {exchange_label}")
        log_data(f"Current Price: {current_price:.8f}")
        log_data(f"Data Points: {len(df)}")
    
    # Get last signal
    if len(signals) > 0:
        last_signal = signals.iloc[-1]
        last_cluster = clustering_result.curr_cluster.iloc[-1]
        last_real_clust = clustering_result.real_clust.iloc[-1]
        last_cluster_val = clustering_result.cluster_val.iloc[-1]
        
        if log_analysis:
            log_analysis("\n" + "-" * 80)
            log_analysis("LATEST SIGNAL")
            log_analysis("-" * 80)
        
        # Display signal
        if pd.notna(last_signal):
            if last_signal > 0:
                signal_text = color_text("LONG", Fore.GREEN)
            elif last_signal < 0:
                signal_text = color_text("SHORT", Fore.RED)
            else:
                signal_text = color_text("NEUTRAL", Fore.YELLOW)
        else:
            signal_text = color_text("NO SIGNAL", Fore.YELLOW)
        
        if log_data:
            log_data(f"Signal: {signal_text}")
            log_data(f"Cluster: {last_cluster} (value: {last_cluster_val:.2f})")
            log_data(f"Real Cluster: {last_real_clust:.4f}")
            log_data(f"Min Distance: {clustering_result.min_dist.iloc[-1]:.4f}")
            log_data(f"Relative Position: {clustering_result.rel_pos.iloc[-1]:.4f}")
        
        # Display last few signals
        if log_analysis:
            log_analysis("\n" + "-" * 80)
            log_analysis("RECENT SIGNALS (Last 10)")
            log_analysis("-" * 80)
        
        recent_df = df.tail(10).copy()
        recent_df["Signal"] = signals.tail(10)
        recent_df["Cluster"] = clustering_result.curr_cluster.tail(10)
        recent_df["Real_Clust"] = clustering_result.real_clust.tail(10)
        
        if log_data:
            for idx, row in recent_df.iterrows():
                signal_val = row["Signal"]
                if pd.notna(signal_val):
                    if signal_val > 0:
                        sig = color_text("LONG", Fore.GREEN)
                    elif signal_val < 0:
                        sig = color_text("SHORT", Fore.RED)
                    else:
                        sig = color_text("NEUTRAL", Fore.YELLOW)
                else:
                    sig = "NO SIGNAL"
                
                log_data(
                    f"{idx} | Price: {row['close']:.8f} | "
                    f"Signal: {sig} | Cluster: {row['Cluster']} | "
                    f"Real_Clust: {row['Real_Clust']:.4f}"
                )


def analyze_symbol(
    symbol: str,
    data_fetcher: DataFetcher,
    config: ClusteringConfig,
    strategy: str = "cluster_transition",
    strategy_params: Optional[dict] = None,
    timeframe: str = "1h",
) -> Optional[dict]:
    """
    Analyze a symbol using Simplified Percentile Clustering.
    
    Args:
        symbol: Trading pair symbol
        data_fetcher: DataFetcher instance
        config: ClusteringConfig
        strategy: Strategy name
        strategy_params: Strategy-specific parameters
        timeframe: Timeframe for data fetching
        
    Returns:
        Dictionary with analysis results or None if failed
    """
    try:
        # Fetch data
        log_progress(f"Fetching data for {symbol}...")
        df, exchange_id = data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol=symbol,
            timeframe=timeframe,
            limit=config.lookback,
            check_freshness=False,
        )
        
        if df is None or len(df) == 0:
            log_error(f"Failed to fetch data for {symbol}")
            return None
        
        # Extract OHLCV
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # Compute clustering
        log_progress("Computing clustering...")
        clustering = SimplifiedPercentileClustering(config)
        clustering_result = clustering.compute(high, low, close)
        
        # Generate signals based on strategy
        log_progress(f"Generating signals using {strategy} strategy...")
        strategy_params = strategy_params or {}
        
        if strategy == "cluster_transition":
            strategy_config = ClusterTransitionConfig(
                min_signal_strength=strategy_params.get("min_signal_strength", 0.3),
                min_rel_pos_change=strategy_params.get("min_rel_pos_change", 0.1),
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
            )
            signals, signal_strength, metadata = generate_signals_mean_reversion(
                high=high,
                low=low,
                close=close,
                clustering_result=clustering_result,
                config=strategy_config,
            )
        else:
            log_error(f"Unknown strategy: {strategy}")
            return None
        
        # Get current price
        current_price = float(close.iloc[-1])
        
        # Get exchange label
        exchange_label = exchange_id.upper() if exchange_id else "Unknown Exchange"
        
        return {
            "symbol": symbol,
            "df": df,
            "signals": signals,
            "signal_strength": signal_strength,
            "metadata": metadata,
            "clustering_result": clustering_result,
            "current_price": current_price,
            "exchange_label": exchange_label,
        }
    except Exception as e:
        log_error(f"Error analyzing {symbol}: {e}")
        import traceback
        log_error(f"Traceback: {traceback.format_exc()}")
        return None


def list_futures_symbols(data_fetcher: DataFetcher) -> None:
    """List available futures symbols."""
    try:
        exchange_manager = data_fetcher.exchange_manager
        markets = exchange_manager.exchange.load_markets()
        
        futures_symbols = [
            symbol
            for symbol, market in markets.items()
            if market.get("type") == "swap" or market.get("future")
        ]
        
        if log_analysis:
            log_analysis("=" * 80)
            log_analysis("AVAILABLE FUTURES SYMBOLS")
            log_analysis("=" * 80)
        
        if log_data:
            for symbol in sorted(futures_symbols)[:50]:  # Show first 50
                log_data(f"  {symbol}")
            
            if len(futures_symbols) > 50:
                log_data(f"\n  ... and {len(futures_symbols) - 50} more symbols")
    except Exception as e:
        log_error(f"Error listing symbols: {e}")


class SPCAnalyzer:
    """
    SPC Analysis Orchestrator.
    
    Manages the complete SPC analysis workflow including configuration
    and execution of analysis.
    """
    
    def __init__(self, args, data_fetcher: DataFetcher):
        """
        Initialize SPC Analyzer.
        
        Args:
            args: Parsed command-line arguments
            data_fetcher: DataFetcher instance
        """
        self.args = args
        self.data_fetcher = data_fetcher
    
    def get_clustering_config(self) -> ClusteringConfig:
        """Create ClusteringConfig from arguments."""
        feature_config = FeatureConfig(
            rsi_len=self.args.rsi_len,
            cci_len=self.args.cci_len,
            fisher_len=self.args.fisher_len,
            dmi_len=self.args.dmi_len,
            zscore_len=self.args.zscore_len,
            mar_len=self.args.mar_len,
        )
        
        return ClusteringConfig(
            k=self.args.k,
            lookback=self.args.lookback,
            p_low=self.args.p_low,
            p_high=self.args.p_high,
            main_plot=self.args.main_plot,
            feature_config=feature_config,
        )
    
    def get_strategy_params(self) -> dict:
        """Get strategy-specific parameters from arguments."""
        return {
            # Cluster Transition parameters
            "min_signal_strength": self.args.min_signal_strength,
            "min_rel_pos_change": self.args.min_rel_pos_change,
            # Regime Following parameters
            "min_regime_strength": self.args.min_regime_strength,
            "min_cluster_duration": self.args.min_cluster_duration,
            # Mean Reversion parameters
            "extreme_threshold": self.args.extreme_threshold,
            "min_extreme_duration": self.args.min_extreme_duration,
        }
    
    def display_config(self, symbol: str) -> None:
        """Display configuration."""
        if log_analysis:
            log_analysis("=" * 80)
            log_analysis("SIMPLIFIED PERCENTILE CLUSTERING (SPC) ANALYSIS")
            log_analysis("=" * 80)
            log_analysis("Configuration:")
        if log_data:
            log_data(f"  Symbol: {symbol}")
            log_data(f"  Timeframe: {self.args.timeframe}")
            log_data(f"  Limit: {self.args.limit} candles")
            log_data(f"  K: {self.args.k}")
            log_data(f"  Lookback: {self.args.lookback}")
            log_data(f"  Percentiles: {self.args.p_low}% - {self.args.p_high}%")
            log_data(f"  Main Plot: {self.args.main_plot}")
            log_data(f"  Strategy: {self.args.strategy}")
    
    def get_symbol_input(self) -> str:
        """
        Get symbol input from arguments or user prompt.
        
        Returns:
            str: Normalized symbol
        """
        quote = self.args.quote.upper() if self.args.quote else DEFAULT_QUOTE
        symbol_input = self.args.symbol

        if not symbol_input and not self.args.no_prompt:
            symbol_input = prompt_user_input(
                f"Enter symbol pair (default: {DEFAULT_SYMBOL}): ",
                default=DEFAULT_SYMBOL,
            )

        if not symbol_input:
            symbol_input = DEFAULT_SYMBOL

        return normalize_symbol(symbol_input, quote)
    
    def run_analysis(self) -> None:
        """Run SPC analysis for a symbol."""
        symbol = self.get_symbol_input()
        self.display_config(symbol)

        # Get configurations
        clustering_config = self.get_clustering_config()
        strategy_params = self.get_strategy_params()

        # Analyze symbol
        result = analyze_symbol(
            symbol=symbol,
            data_fetcher=self.data_fetcher,
            config=clustering_config,
            strategy=self.args.strategy,
            strategy_params=strategy_params,
            timeframe=self.args.timeframe,
        )

        if result is None:
            log_error("Analysis failed")
            return

        # Display results
        display_spc_signals(
            symbol=result["symbol"],
            df=result["df"],
            signals=result["signals"],
            clustering_result=result["clustering_result"],
            current_price=result["current_price"],
            exchange_label=result["exchange_label"],
        )

        # Interactive loop if prompts enabled
        if not self.args.no_prompt:
            self.run_interactive_loop(
                symbol=symbol,
                quote=self.args.quote.upper() if self.args.quote else DEFAULT_QUOTE,
                clustering_config=clustering_config,
                strategy_params=strategy_params,
            )
    
    def run_interactive_loop(
        self,
        symbol: str,
        quote: str,
        clustering_config: ClusteringConfig,
        strategy_params: dict,
    ) -> None:
        """
        Run interactive loop for analyzing multiple symbols.
        
        Args:
            symbol: Initial symbol
            quote: Quote currency
            clustering_config: ClusteringConfig instance
            strategy_params: Strategy parameters dictionary
        """
        try:
            while True:
                print(
                    color_text(
                        "\nPress Ctrl+C to exit. Provide a new symbol to continue.",
                        Fore.YELLOW,
                    )
                )
                symbol_input = prompt_user_input(
                    f"Enter symbol pair (default: {symbol}): ",
                    default=symbol,
                )

                symbol = normalize_symbol(symbol_input, quote)

                result = analyze_symbol(
                    symbol=symbol,
                    data_fetcher=self.data_fetcher,
                    config=clustering_config,
                    strategy=self.args.strategy,
                    strategy_params=strategy_params,
                    timeframe=self.args.timeframe,
                )

                if result is None:
                    log_error("Analysis failed")
                    continue

                # Display results
                display_spc_signals(
                    symbol=result["symbol"],
                    df=result["df"],
                    signals=result["signals"],
                    clustering_result=result["clustering_result"],
                    current_price=result["current_price"],
                    exchange_label=result["exchange_label"],
                )
        except KeyboardInterrupt:
            print(color_text("\nExiting program by user request.", Fore.YELLOW))


def initialize_components() -> Tuple[ExchangeManager, DataFetcher]:
    """
    Initialize ExchangeManager and DataFetcher components.
    
    Returns:
        Tuple of (ExchangeManager, DataFetcher) instances
    """
    log_progress("Initializing components...")
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)
    return exchange_manager, data_fetcher


def main() -> None:
    """
    Main function for SPC analysis.

    Orchestrates the complete SPC analysis workflow:
    1. Parse command-line arguments
    2. Initialize components (ExchangeManager, DataFetcher)
    3. Create SPC Analyzer instance
    4. Run analysis
    """
    args = parse_args()

    # List symbols if requested
    if args.list_symbols:
        exchange_manager, data_fetcher = initialize_components()
        list_futures_symbols(data_fetcher)
        return

    # Initialize components
    exchange_manager, data_fetcher = initialize_components()

    # Create analyzer instance
    analyzer = SPCAnalyzer(args, data_fetcher)

    # Run analysis
    analyzer.run_analysis()


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


