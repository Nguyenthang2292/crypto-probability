"""
Portfolio Manager - Refactored version using modular components.
"""
import signal
import sys
import threading
from typing import List, Optional, Dict
from colorama import Fore, Style, init as colorama_init

try:
    from modules.Position import Position
    from modules.utils import color_text
    from modules.ExchangeManager import ExchangeManager
    from modules.DataFetcher import DataFetcher
    from modules.RiskCalculator import RiskCalculator
    from modules.CorrelationAnalyzer import CorrelationAnalyzer
    from modules.HedgeFinder import HedgeFinder
    from modules.PositionLoader import PositionLoader
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
    RiskCalculator = None
    CorrelationAnalyzer = None
    HedgeFinder = None
    PositionLoader = None
    BENCHMARK_SYMBOL = "BTC/USDT"
    DEFAULT_VAR_CONFIDENCE = 0.95
    DEFAULT_VAR_LOOKBACK_DAYS = 90

colorama_init(autoreset=True)


class PortfolioManager:
    """Main portfolio manager orchestrating all components."""
    
    def __init__(self, api_key=None, api_secret=None, testnet=False):
        self.positions: List[Position] = []
        self.benchmark_symbol = BENCHMARK_SYMBOL
        self.shutdown_event = threading.Event()
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
        # Initialize components
        self.exchange_manager = ExchangeManager(api_key, api_secret, testnet)
        self.data_fetcher = DataFetcher(self.exchange_manager, self.shutdown_event)
        self.risk_calculator = RiskCalculator(self.data_fetcher, self.benchmark_symbol)
        self.position_loader = PositionLoader(api_key, api_secret, testnet)
    
    def add_position(self, symbol: str, direction: str, entry_price: float, size_usdt: float):
        """Add a position to the portfolio."""
        self.positions.append(Position(symbol.upper(), direction.upper(), entry_price, size_usdt))
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal."""
        if not self.shutdown_event.is_set():
            print(color_text("\nInterrupt received. Cancelling ongoing tasks...", Fore.YELLOW))
            self.shutdown_event.set()
        sys.exit(0)
    
    def _should_stop(self) -> bool:
        """Check if shutdown was requested."""
        return self.shutdown_event.is_set()
    
    def load_from_binance(self, api_key=None, api_secret=None, testnet=None, debug=False):
        """Load positions directly from Binance Futures USDT-M."""
        positions = self.position_loader.load_from_binance(api_key, api_secret, testnet, debug)
        if positions:
            self.positions = positions
            # Update exchange manager credentials if provided
            if api_key is not None:
                self.exchange_manager.api_key = api_key
            if api_secret is not None:
                self.exchange_manager.api_secret = api_secret
            if testnet is not None:
                self.exchange_manager.testnet = testnet
    
    def fetch_prices(self):
        """Fetches current prices for all symbols from Binance."""
        symbols = list(set([p.symbol for p in self.positions]))
        if symbols:
            self.data_fetcher.fetch_prices(symbols)
    
    @property
    def market_prices(self):
        """Get market prices from data fetcher."""
        return self.data_fetcher.market_prices
    
    def calculate_stats(self):
        """Calculates PnL, simple delta, and beta-weighted delta for the portfolio."""
        return self.risk_calculator.calculate_stats(self.positions, self.market_prices)
    
    def calculate_beta(self, symbol: str, benchmark_symbol: Optional[str] = None, **kwargs):
        """Calculates beta of a symbol versus a benchmark."""
        return self.risk_calculator.calculate_beta(symbol, benchmark_symbol, **kwargs)
    
    def calculate_portfolio_var(self, confidence: float = DEFAULT_VAR_CONFIDENCE, 
                                lookback_days: int = DEFAULT_VAR_LOOKBACK_DAYS):
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
    
    def fetch_ohlcv(self, symbol, limit=1500, timeframe='1h'):
        """Fetches OHLCV data using ccxt with fallback exchanges."""
        return self.data_fetcher.fetch_ohlcv(symbol, limit, timeframe)
    
    def calculate_weighted_correlation(self, new_symbol: str, verbose: bool = True):
        """Calculates weighted correlation with entire portfolio."""
        analyzer = CorrelationAnalyzer(self.data_fetcher, self.positions)
        return analyzer.calculate_weighted_correlation(new_symbol, verbose)
    
    def calculate_portfolio_return_correlation(self, new_symbol: str, **kwargs):
        """Calculates correlation between portfolio return and new symbol."""
        analyzer = CorrelationAnalyzer(self.data_fetcher, self.positions)
        return analyzer.calculate_portfolio_return_correlation(new_symbol, **kwargs)
    
    def find_best_hedge_candidate(self, total_delta: float, total_beta_delta: float, **kwargs):
        """Automatically scans Binance futures symbols to find the best hedge candidate."""
        analyzer = CorrelationAnalyzer(self.data_fetcher, self.positions)
        hedge_finder = HedgeFinder(
            self.exchange_manager,
            analyzer,
            self.risk_calculator,
            self.positions,
            self.benchmark_symbol,
            self.shutdown_event
        )
        return hedge_finder.find_best_hedge_candidate(total_delta, total_beta_delta, **kwargs)
    
    def analyze_new_trade(self, new_symbol: str, total_delta: float, total_beta_delta: float, **kwargs):
        """Analyzes a potential new trade and automatically recommends direction for beta-weighted hedging."""
        analyzer = CorrelationAnalyzer(self.data_fetcher, self.positions)
        hedge_finder = HedgeFinder(
            self.exchange_manager,
            analyzer,
            self.risk_calculator,
            self.positions,
            self.benchmark_symbol,
            self.shutdown_event
        )
        return hedge_finder.analyze_new_trade(
            new_symbol, 
            total_delta, 
            total_beta_delta,
            self.last_var_value,
            self.last_var_confidence,
            **kwargs
        )


def main():
    print(color_text("=== Crypto Portfolio Manager (Binance Integration) ===", Fore.MAGENTA, Style.BRIGHT))
    
    try:
        pm = PortfolioManager()
    except Exception as e:
        print(color_text(f"Error initializing PortfolioManager: {e}", Fore.RED))
        return
    
    print("\n" + color_text("Loading positions from Binance...", Fore.CYAN))
    try:
        pm.load_from_binance()
    except Exception as e:
        print(color_text(f"Error loading from Binance: {e}", Fore.RED))
        print(color_text("Please check your API credentials and try again.", Fore.YELLOW))
        return
            
    if not pm.positions:
        print(color_text("No positions available. Exiting.", Fore.YELLOW))
        return

    pm.fetch_prices()
    df, total_pnl, total_delta, total_beta_delta = pm.calculate_stats()
    
    print("\n" + color_text("=== PORTFOLIO STATUS ===", Fore.WHITE, Style.BRIGHT))
    print(df.to_string(index=False))
    print("-" * 50)
    print(f"Total PnL: {color_text(f'{total_pnl:.2f} USDT', Fore.GREEN if total_pnl >= 0 else Fore.RED)}")
    print(f"Total Delta: {color_text(f'{total_delta:.2f} USDT', Fore.YELLOW)}")
    print(f"Total Beta Delta (vs {pm.benchmark_symbol}): {color_text(f'{total_beta_delta:.2f} USDT', Fore.YELLOW)}")
    
    var_value = pm.calculate_portfolio_var(confidence=DEFAULT_VAR_CONFIDENCE, lookback_days=DEFAULT_VAR_LOOKBACK_DAYS)
    if var_value is not None:
        conf_pct = int((pm.last_var_confidence or 0) * 100)
        print(color_text(f"Interpretation: With {conf_pct}% confidence, daily loss should stay within {var_value:.2f} USDT.", Fore.WHITE))
    else:
        print(color_text("VaR Interpretation: Not enough history for a reliable estimate.", Fore.YELLOW))
    
    best_candidate = pm.find_best_hedge_candidate(total_delta, total_beta_delta)
    if best_candidate:
        symbol = best_candidate['symbol']
        print("\n" + color_text("=== AUTO HEDGE ANALYSIS ===", Fore.MAGENTA, Style.BRIGHT))
        recommended_direction, recommended_size, correlation = pm.analyze_new_trade(symbol, total_delta, total_beta_delta)
        if recommended_direction and recommended_size is not None:
            print(color_text(f"\n✓ Auto-selected hedge: {symbol} | {recommended_direction} {recommended_size:.2f} USDT", Fore.GREEN, Style.BRIGHT))
        else:
            print(color_text(f"\n{symbol}: Portfolio already neutral, no trade required.", Fore.WHITE))
    else:
        print(color_text("\nCould not determine a suitable hedge candidate automatically.", Fore.YELLOW))


if __name__ == "__main__":
    main()
