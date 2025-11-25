"""
Portfolio Manager - Refactored version using modular components.
"""

import signal
import sys
import threading
from typing import List, Optional
from colorama import Fore, Style, init as colorama_init

try:
    from modules.common.Position import Position
    from modules.common.utils import color_text
    from modules.common.ExchangeManager import ExchangeManager
    from modules.common.DataFetcher import DataFetcher
    from modules.portfolio.risk_calculator import PortfolioRiskCalculator
    from modules.portfolio.correlation_analyzer import PortfolioCorrelationAnalyzer
    from modules.portfolio.hedge_finder import HedgeFinder
    from modules.config import (
        BENCHMARK_SYMBOL,
        DEFAULT_VAR_CONFIDENCE,
        DEFAULT_VAR_LOOKBACK_DAYS,
    )
except ImportError:
    Position = None
    color_text = None
    ExchangeManager = None
    DataFetcher = None
    PortfolioRiskCalculator = None
    PortfolioCorrelationAnalyzer = None
    HedgeFinder = None
    BENCHMARK_SYMBOL = "BTC/USDT"
    DEFAULT_VAR_CONFIDENCE = 0.95
    DEFAULT_VAR_LOOKBACK_DAYS = 90

colorama_init(autoreset=True)


class PortfolioManager:
    """Main portfolio manager orchestrating all components."""

    def __init__(
        self,
        api_key=None,
        api_secret=None,
        testnet=False,
        install_signal_handlers: bool = False,
    ):
        self.positions: List[Position] = []
        self.benchmark_symbol = BENCHMARK_SYMBOL
        self.shutdown_event = threading.Event()
        self._signal_handlers_registered = False
        if install_signal_handlers:
            self.install_signal_handlers()

        # Initialize components
        self.exchange_manager = ExchangeManager(api_key, api_secret, testnet)
        self.data_fetcher = DataFetcher(self.exchange_manager, self.shutdown_event)
        self.risk_calculator = PortfolioRiskCalculator(self.data_fetcher, self.benchmark_symbol)

    def add_position(
        self, symbol: str, direction: str, entry_price: float, size_usdt: float
    ):
        """Add a position to the portfolio."""
        self.positions.append(
            Position(symbol.upper(), direction.upper(), entry_price, size_usdt)
        )

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal."""
        if not self.shutdown_event.is_set():
            print(
                color_text(
                    "\nInterrupt received. Cancelling ongoing tasks...", Fore.YELLOW
                )
            )
            self.shutdown_event.set()
        sys.exit(0)

    def install_signal_handlers(self):
        """Register OS signal handlers for graceful shutdown when running from CLI."""
        if self._signal_handlers_registered:
            return
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "Signal handlers can only be installed from the main thread."
            )
        signal.signal(signal.SIGINT, self._handle_shutdown)
        try:
            signal.signal(signal.SIGTERM, self._handle_shutdown)
        except AttributeError:
            # SIGTERM may not be available on some platforms (e.g., Windows)
            pass
        self._signal_handlers_registered = True

    def _should_stop(self) -> bool:
        """Check if shutdown was requested."""
        return self.shutdown_event.is_set()

    def load_from_binance(
        self, api_key=None, api_secret=None, testnet=None, debug=False
    ):
        """Load positions directly from Binance Futures USDT-M."""
        try:
            binance_positions = self.data_fetcher.fetch_binance_futures_positions(
                api_key=api_key or self.exchange_manager.api_key,
                api_secret=api_secret or self.exchange_manager.api_secret,
                testnet=self.exchange_manager.testnet if testnet is None else testnet,
                debug=debug,
            )
        except Exception as exc:
            raise ValueError(f"Error loading positions from Binance: {exc}")

        if not binance_positions:
            print(color_text("No open positions found on Binance.", Fore.YELLOW))
            self.positions = []
            return

        self.positions = [
            Position(
                symbol=pos["symbol"].upper(),
                direction=pos["direction"].upper(),
                entry_price=pos["entry_price"],
                size_usdt=pos["size_usdt"],
            )
            for pos in binance_positions
        ]

        # Update exchange manager credentials if provided
        credentials_updated = False
        if api_key is not None:
            self.exchange_manager.api_key = api_key
            credentials_updated = True
        if api_secret is not None:
            self.exchange_manager.api_secret = api_secret
            credentials_updated = True
        if testnet is not None:
            self.exchange_manager.testnet = testnet

        if credentials_updated:
            self.exchange_manager.authenticated.update_default_credentials(
                api_key=self.exchange_manager.api_key,
                api_secret=self.exchange_manager.api_secret,
            )

    def fetch_prices(self):
        """Fetches current prices for all symbols from Binance."""
        symbols = list(set([p.symbol for p in self.positions]))
        if symbols:
            self.data_fetcher.fetch_current_prices_from_binance(symbols)

    @property
    def market_prices(self):
        """Get market prices from data fetcher."""
        return self.data_fetcher.market_prices

    def calculate_stats(self):
        """Calculates PnL, simple delta, and beta-weighted delta for the portfolio."""
        return self.risk_calculator.calculate_stats(self.positions, self.market_prices)

    def calculate_beta(
        self, symbol: str, benchmark_symbol: Optional[str] = None, **kwargs
    ):
        """Calculates beta of a symbol versus a benchmark."""
        return self.risk_calculator.calculate_beta(symbol, benchmark_symbol, **kwargs)

    def calculate_portfolio_var(
        self,
        confidence: float = DEFAULT_VAR_CONFIDENCE,
        lookback_days: int = DEFAULT_VAR_LOOKBACK_DAYS,
    ):
        """Calculates Historical Simulation VaR for the current portfolio."""
        return self.risk_calculator.calculate_portfolio_var(
            self.positions, confidence, lookback_days
        )

    @property
    def last_var_value(self):
        """Get last VaR value from risk calculator."""
        return self.risk_calculator.last_var_value

    @property
    def last_var_confidence(self):
        """Get last VaR confidence from risk calculator."""
        return self.risk_calculator.last_var_confidence

    def fetch_ohlcv(self, symbol, limit=1500, timeframe="1h"):
        """Fetches OHLCV data using ccxt with fallback exchanges."""
        df, _ = self.data_fetcher.fetch_ohlcv_with_fallback_exchange(
            symbol, limit, timeframe
        )
        return df

    def calculate_weighted_correlation(self, new_symbol: str, verbose: bool = True):
        """Calculates weighted correlation with entire portfolio."""
        analyzer = PortfolioCorrelationAnalyzer(self.data_fetcher, self.positions)
        return analyzer.calculate_weighted_correlation_with_new_symbol(new_symbol, verbose)

    def calculate_portfolio_return_correlation(self, new_symbol: str, **kwargs):
        """Calculates correlation between portfolio return and new symbol."""
        analyzer = PortfolioCorrelationAnalyzer(self.data_fetcher, self.positions)
        return analyzer.calculate_portfolio_return_correlation(new_symbol, **kwargs)

    def find_best_hedge_candidate(
        self, total_delta: float, total_beta_delta: float, **kwargs
    ):
        """Automatically scans Binance futures symbols to find the best hedge candidate."""
        analyzer = PortfolioCorrelationAnalyzer(self.data_fetcher, self.positions)
        hedge_finder = HedgeFinder(
            self.exchange_manager,
            analyzer,
            self.risk_calculator,
            self.positions,
            self.benchmark_symbol,
            self.shutdown_event,
            self.data_fetcher,
        )
        return hedge_finder.find_best_hedge_candidate(
            total_delta, total_beta_delta, **kwargs
        )

    def analyze_new_trade(
        self, new_symbol: str, total_delta: float, total_beta_delta: float, **kwargs
    ):
        """Analyzes a potential new trade and automatically recommends direction for beta-weighted hedging."""
        analyzer = PortfolioCorrelationAnalyzer(self.data_fetcher, self.positions)
        hedge_finder = HedgeFinder(
            self.exchange_manager,
            analyzer,
            self.risk_calculator,
            self.positions,
            self.benchmark_symbol,
            self.shutdown_event,
            self.data_fetcher,
        )
        return hedge_finder.analyze_new_trade(
            new_symbol,
            total_delta,
            total_beta_delta,
            self.last_var_value,
            self.last_var_confidence,
            **kwargs,
        )


def display_portfolio_analysis(pm: PortfolioManager):
    """
    Tính năng 1: Hiển thị Portfolio Correlation và VaR hiện có.
    """
    print("\n" + color_text("=== PORTFOLIO ANALYSIS ===", Fore.CYAN, Style.BRIGHT))

    # Fetch prices and calculate stats
    pm.fetch_prices()
    df, total_pnl, total_delta, total_beta_delta = pm.calculate_stats()

    # Display portfolio status
    print("\n" + color_text("=== PORTFOLIO STATUS ===", Fore.WHITE, Style.BRIGHT))
    print(df.to_string(index=False))
    print("-" * 50)
    print(
        f"Total PnL: {color_text(f'{total_pnl:.2f} USDT', Fore.GREEN if total_pnl >= 0 else Fore.RED)}"
    )
    print(f"Total Delta: {color_text(f'{total_delta:.2f} USDT', Fore.YELLOW)}")
    print(
        f"Total Beta Delta (vs {pm.benchmark_symbol}): {color_text(f'{total_beta_delta:.2f} USDT', Fore.YELLOW)}"
    )

    # Calculate and display VaR
    print("\n" + color_text("=== VALUE AT RISK (VaR) ===", Fore.CYAN, Style.BRIGHT))
    var_value = pm.calculate_portfolio_var(
        confidence=DEFAULT_VAR_CONFIDENCE, lookback_days=DEFAULT_VAR_LOOKBACK_DAYS
    )
    if var_value is not None:
        conf_pct = int((pm.last_var_confidence or 0) * 100)
        print(
            color_text(
                f"With {conf_pct}% confidence, daily loss should stay within {var_value:.2f} USDT.",
                Fore.WHITE,
            )
        )
    else:
        print(
            color_text(
                "Not enough history for a reliable VaR estimate.",
                Fore.YELLOW,
            )
        )

    # Calculate and display Portfolio Internal Correlation
    print("\n" + color_text("=== PORTFOLIO CORRELATION ===", Fore.CYAN, Style.BRIGHT))
    if len(pm.positions) >= 2:
        analyzer = PortfolioCorrelationAnalyzer(pm.data_fetcher, pm.positions)
        internal_corr, pairs = analyzer.calculate_weighted_correlation(verbose=True)

        if internal_corr is not None:
            if abs(internal_corr) > 0.7:
                status = color_text("HIGH - Consider diversification", Fore.RED)
            elif abs(internal_corr) > 0.4:
                status = color_text("MODERATE", Fore.YELLOW)
            else:
                status = color_text("LOW - Good diversification", Fore.GREEN)
            print(f"\nPortfolio Correlation Status: {status}")
    else:
        print(
            color_text(
                "Need at least 2 positions to calculate portfolio correlation.",
                Fore.YELLOW,
            )
        )


def display_portfolio_with_hedge_analysis(pm: PortfolioManager):
    """
    Tính năng 2: Hiển thị Portfolio + tự động tìm hedge candidate.
    """
    # Hiển thị tất cả từ tính năng 1
    display_portfolio_analysis(pm)

    # Thêm phần auto hedge
    print("\n" + color_text("=== AUTO HEDGE ANALYSIS ===", Fore.MAGENTA, Style.BRIGHT))

    pm.fetch_prices()
    _, total_pnl, total_delta, total_beta_delta = pm.calculate_stats()

    best_candidate = pm.find_best_hedge_candidate(total_delta, total_beta_delta)
    if best_candidate:
        symbol = best_candidate["symbol"]
        recommended_direction, recommended_size, correlation = pm.analyze_new_trade(
            symbol, total_delta, total_beta_delta
        )
        if recommended_direction and recommended_size is not None:
            print(
                color_text(
                    f"\n✓ Auto-selected hedge: {symbol} | {recommended_direction} {recommended_size:.2f} USDT",
                    Fore.GREEN,
                    Style.BRIGHT,
                )
            )
        else:
            print(
                color_text(
                    f"\n{symbol}: Portfolio already neutral, no trade required.",
                    Fore.WHITE,
                )
            )
    else:
        print(
            color_text(
                "\nCould not determine a suitable hedge candidate automatically.",
                Fore.YELLOW,
            )
        )


def main():
    print(
        color_text(
            "=== Crypto Portfolio Manager (Binance Integration) ===",
            Fore.MAGENTA,
            Style.BRIGHT,
        )
    )

    try:
        pm = PortfolioManager()
        pm.install_signal_handlers()
    except Exception as e:
        print(color_text(f"Error initializing PortfolioManager: {e}", Fore.RED))
        return

    print("\n" + color_text("Loading positions from Binance...", Fore.CYAN))
    try:
        pm.load_from_binance()
    except Exception as e:
        print(color_text(f"Error loading from Binance: {e}", Fore.RED))
        print(
            color_text("Please check your API credentials and try again.", Fore.YELLOW)
        )
        return

    if not pm.positions:
        print(color_text("No positions available. Exiting.", Fore.YELLOW))
        return

    # Interactive menu
    print("\n" + "=" * 60)
    print(color_text("Select Analysis Mode:", Fore.CYAN, Style.BRIGHT))
    print("=" * 60)
    print("1. Portfolio Analysis (Correlation + VaR)")
    print("2. Portfolio Analysis + Auto Hedge")
    print("3. Exit")
    print("=" * 60)

    while True:
        try:
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == "1":
                display_portfolio_analysis(pm)
                break
            elif choice == "2":
                display_portfolio_with_hedge_analysis(pm)
                break
            elif choice == "3":
                print(color_text("\nExiting...", Fore.YELLOW))
                break
            else:
                print(color_text("Invalid choice. Please enter 1, 2, or 3.", Fore.RED))
        except KeyboardInterrupt:
            print(color_text("\n\nInterrupted by user. Exiting...", Fore.YELLOW))
            break
        except EOFError:
            print(color_text("\n\nExiting...", Fore.YELLOW))
            break


if __name__ == "__main__":
    main()
