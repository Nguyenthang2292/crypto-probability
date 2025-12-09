"""
Adaptive Trend Classification (ATC) Main Program

Analyzes futures pairs on Binance using Adaptive Trend Classification:
- Fetches OHLCV data from Binance futures
- Calculates ATC signals using multiple moving averages
- Displays trend signals and analysis
"""

import warnings
import sys
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
    extract_dict_from_namespace,
)
from modules.common.ExchangeManager import ExchangeManager
from modules.common.DataFetcher import DataFetcher
from modules.adaptive_trend.core.analyzer import analyze_symbol
from modules.adaptive_trend.utils.config import create_atc_config_from_dict
from modules.adaptive_trend.core.scanner import scan_all_symbols
from modules.adaptive_trend.cli import (
    parse_args,
    prompt_interactive_mode,
    display_scan_results,
    list_futures_symbols,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
colorama_init(autoreset=True)


class ATCAnalyzer:
    """
    ATC Analysis Orchestrator.
    
    Manages the complete ATC analysis workflow including mode selection,
    configuration, and execution of auto/manual analysis modes.
    """
    
    def __init__(self, args, data_fetcher: DataFetcher):
        """
        Initialize ATC Analyzer.
        
        Args:
            args: Parsed command-line arguments
            data_fetcher: DataFetcher instance
        """
        self.args = args
        self.data_fetcher = data_fetcher
        self.selected_timeframe = args.timeframe
        self.mode = "manual"
        self._atc_params = None
    
    def determine_mode_and_timeframe(self) -> Tuple[str, str]:
        """
        Determine analysis mode and timeframe from arguments and interactive menu.
        
        Returns:
            tuple: (mode, selected_timeframe)
        """
        self.mode = "manual"
        self.selected_timeframe = self.args.timeframe
        
        if self.args.auto:
            self.mode = "auto"
        elif not self.args.no_menu:
            menu_result = prompt_interactive_mode(default_timeframe=self.args.timeframe)
            self.mode = menu_result.get("mode", "manual")
            # Use timeframe from menu if selected
            if "timeframe" in menu_result:
                self.selected_timeframe = menu_result["timeframe"]
            
            # If user only selected timeframe, show menu again
            if self.mode is None:
                menu_result = prompt_interactive_mode(default_timeframe=self.selected_timeframe)
                self.mode = menu_result.get("mode", "manual")
                if "timeframe" in menu_result:
                    self.selected_timeframe = menu_result["timeframe"]
        
        return self.mode, self.selected_timeframe
    
    def get_atc_params(self) -> dict:
        """Extract and cache ATC parameters from arguments."""
        if self._atc_params is None:
            atc_param_keys = [
                "limit",
                "ema_len",
                "hma_len",
                "wma_len",
                "dema_len",
                "lsma_len",
                "kama_len",
                "robustness",
                "lambda_param",
                "decay",
                "cutout",
            ]
            self._atc_params = extract_dict_from_namespace(self.args, atc_param_keys)
        return self._atc_params
    
    def display_auto_mode_config(self) -> None:
        """Display configuration for auto mode."""
        if log_analysis:
            log_analysis("=" * 80)
            log_analysis("ADAPTIVE TREND CLASSIFICATION (ATC) - AUTO SCAN MODE")
            log_analysis("=" * 80)
            log_analysis("Configuration:")
        if log_data:
            log_data(f"  Mode: AUTO (scan all symbols)")
            log_data(f"  Timeframe: {self.selected_timeframe}")
            log_data(f"  Limit: {self.args.limit} candles")
            log_data(f"  Robustness: {self.args.robustness}")
            log_data(f"  MA Lengths: EMA={self.args.ema_len}, HMA={self.args.hma_len}, WMA={self.args.wma_len}, DEMA={self.args.dema_len}, LSMA={self.args.lsma_len}, KAMA={self.args.kama_len}")
            log_data(f"  Lambda: {self.args.lambda_param}, Decay: {self.args.decay}, Cutout: {self.args.cutout}")
            log_data(f"  Min Signal: {self.args.min_signal}")
            if self.args.max_symbols:
                log_data(f"  Max Symbols: {self.args.max_symbols}")
    
    def run_auto_scan(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run ATC auto scan and return results without displaying.
        
        This method is designed to be reusable by other analyzers that combine
        ATC with other strategies (e.g., ATC + Range Oscillator).
        
        Returns:
            Tuple of (long_signals_df, short_signals_df):
            - long_signals_df: DataFrame with LONG signals
            - short_signals_df: DataFrame with SHORT signals
        """
        # Get ATC parameters and create config
        atc_params = self.get_atc_params()
        atc_config = create_atc_config_from_dict(atc_params, timeframe=self.selected_timeframe)

        # Scan all symbols
        long_signals, short_signals = scan_all_symbols(
            data_fetcher=self.data_fetcher,
            atc_config=atc_config,
            max_symbols=self.args.max_symbols,
            min_signal=self.args.min_signal,
        )

        return long_signals, short_signals
    
    def run_auto_mode(self) -> None:
        """Run auto mode: scan all symbols for LONG/SHORT signals."""
        self.display_auto_mode_config()

        # Use run_auto_scan() for the actual scanning
        long_signals, short_signals = self.run_auto_scan()

        # Display results
        display_scan_results(long_signals, short_signals, self.args.min_signal)
    
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
    
    def display_manual_mode_config(self, symbol: str) -> None:
        """Display configuration for manual mode."""
        if log_analysis:
            log_analysis("=" * 80)
            log_analysis("ADAPTIVE TREND CLASSIFICATION (ATC) ANALYSIS")
            log_analysis("=" * 80)
            log_analysis("Configuration:")
        if log_data:
            log_data(f"  Symbol: {symbol}")
            log_data(f"  Timeframe: {self.selected_timeframe}")
            log_data(f"  Limit: {self.args.limit} candles")
            log_data(f"  Robustness: {self.args.robustness}")
            log_data(f"  MA Lengths: EMA={self.args.ema_len}, HMA={self.args.hma_len}, WMA={self.args.wma_len}, DEMA={self.args.dema_len}, LSMA={self.args.lsma_len}, KAMA={self.args.kama_len}")
            log_data(f"  Lambda: {self.args.lambda_param}, Decay: {self.args.decay}, Cutout: {self.args.cutout}")
    
    def run_manual_mode(self) -> None:
        """Run manual mode: analyze specific symbol."""
        symbol = self.get_symbol_input()
        self.display_manual_mode_config(symbol)

        # Get ATC parameters and create config
        atc_params = self.get_atc_params()
        atc_config = create_atc_config_from_dict(atc_params, timeframe=self.selected_timeframe)

        # Analyze symbol
        result = analyze_symbol(
            symbol=symbol,
            data_fetcher=self.data_fetcher,
            config=atc_config,
        )

        if result is None:
            log_error("Analysis failed")
            return

        # Display results
        from modules.adaptive_trend.cli.display import display_atc_signals
        display_atc_signals(
            symbol=result["symbol"],
            df=result["df"],
            atc_results=result["atc_results"],
            current_price=result["current_price"],
            exchange_label=result["exchange_label"],
        )

        # Interactive loop if prompts enabled
        if not self.args.no_prompt:
            self.run_interactive_loop(
                symbol=symbol,
                quote=self.args.quote.upper() if self.args.quote else DEFAULT_QUOTE,
                atc_params=atc_params,
            )
    
    def run_interactive_loop(self, symbol: str, quote: str, atc_params: dict) -> None:
        """
        Run interactive loop for analyzing multiple symbols.
        
        Args:
            symbol: Initial symbol
            quote: Quote currency
            atc_params: ATC parameters dictionary
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
                
                # Create ATCConfig
                atc_config = create_atc_config_from_dict(atc_params, timeframe=self.selected_timeframe)

                result = analyze_symbol(
                    symbol=symbol,
                    data_fetcher=self.data_fetcher,
                    config=atc_config,
                )

                if result is None:
                    log_error("Analysis failed")
                    continue

                # Display results
                from modules.adaptive_trend.cli.display import display_atc_signals
                display_atc_signals(
                    symbol=result["symbol"],
                    df=result["df"],
                    atc_results=result["atc_results"],
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
    Main function for ATC analysis.

    Orchestrates the complete ATC analysis workflow:
    1. Parse command-line arguments
    2. Initialize components (ExchangeManager, DataFetcher)
    3. Create ATC Analyzer instance
    4. Determine mode and timeframe
    5. Run appropriate analysis mode
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
    analyzer = ATCAnalyzer(args, data_fetcher)

    # Determine mode and timeframe
    mode, _ = analyzer.determine_mode_and_timeframe()

    # Run appropriate mode
    if mode == "auto":
        analyzer.run_auto_mode()
    else:
        analyzer.run_manual_mode()


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

