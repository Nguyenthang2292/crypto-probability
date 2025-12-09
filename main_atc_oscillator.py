"""
ATC + Range Oscillator Combined Signal Filter.

This program combines signals from Adaptive Trend Classification (ATC) and Range Oscillator:
1. Runs ATC auto scan to find LONG/SHORT signals
2. Filters symbols by checking if Range Oscillator signals match ATC signals
3. Returns final list of symbols with confirmed signals from both indicators
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

from modules.config import DEFAULT_TIMEFRAME
from modules.common.utils import (
    color_text,
    log_error,
    log_progress,
    log_success,
    log_warn,
)
from modules.common.ExchangeManager import ExchangeManager
from modules.common.DataFetcher import DataFetcher
from modules.adaptive_trend.cli import prompt_timeframe
from main_atc import ATCAnalyzer
from modules.range_oscillator.cli import (
    parse_args,
    display_configuration,
    display_final_results,
)
from modules.range_oscillator.analysis.combined import (
    generate_signals_combined_all_strategy,
)
from modules.range_oscillator.config import (
    CombinedStrategyConfig,
    ConsensusConfig,
    DynamicSelectionConfig,
)

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
    """
    Calculate Range Oscillator signal for a symbol using Strategy 5 Combined with Dynamic Selection and Adaptive Weights.
    
    Args:
        data_fetcher: DataFetcher instance
        symbol: Symbol to analyze
        timeframe: Timeframe for data
        limit: Number of candles
        osc_length: Range Oscillator length parameter
        osc_mult: Range Oscillator multiplier
        strategies: List of strategy numbers to enable (e.g., [2, 3, 4, 6, 7, 8, 9]). 
                    If None, uses all available strategies [2, 3, 4, 6, 7, 8, 9]
        
    Returns:
        Tuple of (signal, confidence_score):
        - signal: 1 (LONG), -1 (SHORT), 0 (NEUTRAL), or None if error
        - confidence_score: Confidence score (0.0 to 1.0) or None if error
    """
    try:
        # Fetch OHLCV data
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

        # Default to all available strategies if not specified
        if strategies is None:
            enabled_strategies = [2, 3, 4, 6, 7, 8, 9]  # All available strategies
        else:
            # Convert strategy numbers if needed (e.g., [5] -> [2, 3, 4, 6, 7, 8, 9])
            # Strategy 5 is the combined strategy, so we enable all sub-strategies
            if 5 in strategies:
                enabled_strategies = [2, 3, 4, 6, 7, 8, 9]
            else:
                enabled_strategies = strategies

        # Use Strategy 5 Combined with Dynamic Selection and Adaptive Weights
        # Create config object for cleaner code
        config = CombinedStrategyConfig()
        config.enabled_strategies = enabled_strategies
        config.return_confidence_score = True
        
        # Configure dynamic selection
        config.dynamic.enabled = True
        config.dynamic.lookback = 20
        config.dynamic.volatility_threshold = 0.6
        config.dynamic.trend_threshold = 0.5
        
        # Configure consensus with adaptive weights
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

        # Unpack tuple: (signal_series, strength_series, strategy_stats, confidence_series)
        # Return type is always 4 elements: Tuple[pd.Series, pd.Series, Optional[Dict], Optional[pd.Series]]
        signals = result[0]  # signal_series
        confidence = result[3]  # confidence_series (index 3, None if return_confidence_score=False)

        if signals is None or signals.empty:
            return None

        # Get latest signal and confidence (last non-NaN value)
        non_nan_mask = ~signals.isna()
        if not non_nan_mask.any():
            return None

        latest_idx = signals[non_nan_mask].index[-1]
        latest_signal = int(signals.loc[latest_idx])
        latest_confidence = float(confidence.loc[latest_idx]) if confidence is not None and not confidence.empty else 0.0

        return (latest_signal, latest_confidence)

    except Exception as e:
        # Skip symbols with errors
            return None


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


class ATCOscillatorAnalyzer:
    """
    ATC + Range Oscillator Combined Signal Filter Orchestrator.
    
    Manages the complete workflow of combining ATC and Range Oscillator signals:
    1. Runs ATC auto scan to find LONG/SHORT signals
    2. Filters symbols by checking if Range Oscillator signals match ATC signals
    3. Returns final list of symbols with confirmed signals from both indicators
    
    This class uses ATCAnalyzer through composition to reuse ATC analysis logic,
    making it easy to extend with other strategies in the future (e.g., ATC + OtherStrategy).
    """
    
    def __init__(self, args, data_fetcher: DataFetcher):
        """
        Initialize ATC + Range Oscillator Analyzer.
        
        Args:
            args: Parsed command-line arguments
            data_fetcher: DataFetcher instance
        """
        self.args = args
        self.data_fetcher = data_fetcher
        
        # Use ATCAnalyzer for ATC-related operations (composition pattern)
        # This allows reuse of ATC logic and makes it easy to extend with other strategies
        self.atc_analyzer = ATCAnalyzer(args, data_fetcher)
        
        # Update timeframe in ATCAnalyzer to match our selected timeframe
        self.selected_timeframe = args.timeframe
        self.atc_analyzer.selected_timeframe = args.timeframe
        
        # Results storage
        self.long_signals_atc = pd.DataFrame()
        self.short_signals_atc = pd.DataFrame()
        self.long_signals_confirmed = pd.DataFrame()
        self.short_signals_confirmed = pd.DataFrame()
    
    def determine_timeframe(self) -> str:
        """
        Determine timeframe from arguments and interactive menu.
        
        Returns:
            str: Selected timeframe
        """
        self.selected_timeframe = self.args.timeframe
        
        # Prompt for timeframe selection if menu is enabled
        if not self.args.no_menu:
            print("\n" + color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
            print(color_text("ATC PHASE - TIMEFRAME SELECTION", Fore.CYAN, Style.BRIGHT))
            print(color_text("=" * 80, Fore.CYAN, Style.BRIGHT))
            self.selected_timeframe = prompt_timeframe(default_timeframe=self.selected_timeframe)
            print(color_text(f"\nSelected timeframe for ATC analysis: {self.selected_timeframe}", Fore.GREEN))
        
        # Sync timeframe with ATCAnalyzer
        self.atc_analyzer.selected_timeframe = self.selected_timeframe
        
        return self.selected_timeframe
    
    def get_atc_params(self) -> dict:
        """Extract ATC parameters from arguments using ATCAnalyzer."""
        return self.atc_analyzer.get_atc_params()
    
    def get_oscillator_params(self) -> dict:
        """Extract Range Oscillator parameters from arguments."""
        return {
            "osc_length": self.args.osc_length,
            "osc_mult": self.args.osc_mult,
            "max_workers": self.args.max_workers,
            "strategies": self.args.osc_strategies,
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
    
    def run_atc_scan(self) -> None:
        """Run ATC auto scan to get LONG/SHORT signals using ATCAnalyzer."""
        log_progress("\nStep 1: Running ATC auto scan...")
        log_progress("=" * 80)
        
        # Use ATCAnalyzer's run_auto_scan() method to get scan results
        # This reuses the ATC logic and makes the code more maintainable
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
        """
        Worker function to process a single symbol for Range Oscillator confirmation.
        
        This function is designed to be thread-safe by creating its own DataFetcher instance.
        
        Args:
            symbol_data: Dictionary with symbol information (symbol, signal, trend, price, exchange)
            exchange_manager: ExchangeManager instance (shared, thread-safe)
            timeframe: Timeframe for data
            limit: Number of candles
            expected_osc_signal: Expected signal value (1 for LONG, -1 for SHORT)
            osc_length: Range Oscillator length parameter
            osc_mult: Range Oscillator multiplier
            strategies: List of strategy numbers to use
            
        Returns:
            Dictionary with confirmed signal data if signals match, None otherwise
        """
        try:
            # Create a new DataFetcher instance for this thread (thread-safe)
            data_fetcher = DataFetcher(exchange_manager)
            
            symbol = symbol_data["symbol"]
            
            # Calculate Range Oscillator signal (returns tuple of (signal, confidence_score))
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

            # Check if signals match
            if osc_signal == expected_osc_signal:
                # Signals match - return confirmed signal data
                return {
                    "symbol": symbol,
                    "signal": symbol_data["signal"],
                    "trend": symbol_data["trend"],
                    "price": symbol_data["price"],
                    "exchange": symbol_data["exchange"],
                    "osc_signal": osc_signal,
                    "osc_confidence": osc_confidence,  # Confidence score (0.0 to 1.0)
                }
            
            return None
            
        except Exception as e:
            # Skip symbols with errors
            return None
    
    def filter_signals_by_range_oscillator(
        self,
        atc_signals_df: pd.DataFrame,
        signal_type: str,  # "LONG" or "SHORT"
    ) -> pd.DataFrame:
        """
        Filter ATC signals by checking Range Oscillator confirmation using parallel processing.
        Uses "Any Strategy Mode" - accepts signal from any single strategy.
        
        Args:
            atc_signals_df: DataFrame with ATC signals (columns: symbol, signal, trend, price, exchange)
            signal_type: "LONG" or "SHORT"
            
        Returns:
            DataFrame with filtered signals that match Range Oscillator
        """
        if atc_signals_df.empty:
            return pd.DataFrame()

        osc_params = self.get_oscillator_params()
        expected_osc_signal = 1 if signal_type == "LONG" else -1
        total = len(atc_signals_df)
        
        strategies_str = "Strategy 5 Combined (Dynamic Selection + Adaptive Weights)"
        log_progress(
            f"Checking Range Oscillator signals for {total} {signal_type} symbols "
            f"({strategies_str}, workers: {osc_params['max_workers']})..."
        )

        # Get ExchangeManager from DataFetcher (shared, thread-safe)
        exchange_manager = self.data_fetcher.exchange_manager
        
        # Convert DataFrame rows to list of dictionaries for parallel processing
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

        # Thread-safe progress tracking
        progress_lock = threading.Lock()
        checked_count = [0]  # Use list to allow modification in nested function
        confirmed_count = [0]

        # Process symbols in parallel using ThreadPoolExecutor
        filtered_results = []
        
        with ThreadPoolExecutor(max_workers=osc_params["max_workers"]) as executor:
            # Submit all tasks
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

            # Process completed tasks as they finish
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result is not None:
                        with progress_lock:
                            confirmed_count[0] += 1
                            filtered_results.append(result)
                except Exception as e:
                    # Skip symbols with errors
                    pass
                finally:
                    # Update progress (thread-safe)
                    with progress_lock:
                        checked_count[0] += 1
                        current_checked = checked_count[0]
                        current_confirmed = confirmed_count[0]
                        
                        # Update progress every 10 symbols or at completion
                        if current_checked % 10 == 0 or current_checked == total:
                            log_progress(
                                f"Checked {current_checked}/{total} symbols... "
                                f"Found {current_confirmed} confirmed {signal_type} signals"
                            )

        if not filtered_results:
            return pd.DataFrame()

        filtered_df = pd.DataFrame(filtered_results)
        
        # Sort by confidence score (descending) first, then by signal strength
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
            # Fallback: sort by signal strength (absolute value)
            if signal_type == "LONG":
                filtered_df = filtered_df.sort_values("signal", ascending=False).reset_index(drop=True)
            else:
                filtered_df = filtered_df.sort_values("signal", ascending=True).reset_index(drop=True)

        return filtered_df
    
    def filter_by_oscillator(self) -> None:
        """Filter ATC signals by Range Oscillator confirmation."""
        log_progress("\nStep 2: Filtering by Range Oscillator confirmation...")
        log_progress("=" * 80)

        # Filter LONG signals (parallel processing)
        if not self.long_signals_atc.empty:
            self.long_signals_confirmed = self.filter_signals_by_range_oscillator(
                atc_signals_df=self.long_signals_atc,
                signal_type="LONG",
            )
        else:
            self.long_signals_confirmed = pd.DataFrame()

        # Filter SHORT signals (parallel processing)
        if not self.short_signals_atc.empty:
            self.short_signals_confirmed = self.filter_signals_by_range_oscillator(
                atc_signals_df=self.short_signals_atc,
                signal_type="SHORT",
            )
        else:
            self.short_signals_confirmed = pd.DataFrame()
    
    def display_results(self) -> None:
        """Display final filtered results."""
        log_progress("\nStep 3: Displaying final results...")
        display_final_results(
            long_signals=self.long_signals_confirmed,
            short_signals=self.short_signals_confirmed,
            original_long_count=len(self.long_signals_atc),
            original_short_count=len(self.short_signals_atc),
        )
    
    def run(self) -> None:
        """
        Run the complete ATC + Range Oscillator analysis workflow.
        
        Workflow:
        1. Determine timeframe
        2. Display configuration
        3. Run ATC auto scan
        4. Filter by Range Oscillator confirmation
        5. Display final results
        """
        # Step 1: Determine timeframe
        self.determine_timeframe()
        
        # Step 2: Display configuration
        self.display_config()
        
        # Step 3: Initialize components (already done in __init__)
        log_progress("Initializing components...")
        
        # Step 4: Run ATC auto scan
        self.run_atc_scan()
        
        # Step 5: Filter by Range Oscillator confirmation
        self.filter_by_oscillator()
        
        # Step 6: Display final results
        self.display_results()
        
        log_success("\nAnalysis complete!")


def main() -> None:
    """
    Main function for ATC + Range Oscillator combined signal filtering.
    
    Orchestrates the complete workflow:
    1. Parse command-line arguments
    2. Initialize components (ExchangeManager, DataFetcher)
    3. Create ATCOscillatorAnalyzer instance
    4. Run analysis workflow
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Initialize components
    exchange_manager, data_fetcher = initialize_components()
    
    # Create analyzer instance
    analyzer = ATCOscillatorAnalyzer(args, data_fetcher)
    
    # Run analysis workflow
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
