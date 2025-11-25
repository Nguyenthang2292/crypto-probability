"""
Test script for PortfolioCorrelationAnalyzer
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
from colorama import Fore, Style, init as colorama_init

from modules.ExchangeManager import ExchangeManager
from modules.DataFetcher import DataFetcher
from modules.Position import Position
from modules.PortfolioCorrelationAnalyzer import PortfolioCorrelationAnalyzer
from modules.utils import color_text

# Suppress warnings
warnings.filterwarnings("ignore")
colorama_init(autoreset=True)


def main():
    print(color_text("\n=== Testing PortfolioCorrelationAnalyzer ===\n", Fore.CYAN, Style.BRIGHT))

    # Initialize ExchangeManager and DataFetcher
    print(color_text("Initializing ExchangeManager and DataFetcher...", Fore.YELLOW))
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)

    # Create sample positions
    print(color_text("\nCreating sample positions...", Fore.YELLOW))
    positions = [
        Position("BTC/USDT", "LONG", entry_price=50000.0, size_usdt=1000.0),
        Position("ETH/USDT", "LONG", entry_price=3000.0, size_usdt=500.0),
        # Position("SOL/USDT", "SHORT", entry_price=100.0, size_usdt=300.0),  # Uncomment to test SHORT
    ]

    print(color_text(f"Positions created: {len(positions)}", Fore.GREEN))
    for pos in positions:
        print(f"  - {pos.symbol}: {pos.direction}, {pos.size_usdt} USDT")

    # Initialize analyzer
    print(color_text("\nInitializing PortfolioCorrelationAnalyzer...", Fore.YELLOW))
    analyzer = PortfolioCorrelationAnalyzer(data_fetcher, positions)

    try:
        # Test 1: Calculate portfolio internal correlation
        print(color_text("\n" + "="*60, Fore.CYAN))
        print(color_text("Test 1: Portfolio Internal Correlation", Fore.CYAN, Style.BRIGHT))
        print(color_text("="*60, Fore.CYAN))
        
        internal_corr, pairs = analyzer.calculate_weighted_correlation(verbose=True)
        
        if internal_corr is not None:
            print(color_text(f"\n[OK] Internal correlation calculated: {internal_corr:.4f}", Fore.GREEN))
        else:
            print(color_text("\n[WARNING] Could not calculate internal correlation", Fore.YELLOW))

        # Test 2: Calculate weighted correlation with new symbol
        print(color_text("\n" + "="*60, Fore.CYAN))
        print(color_text("Test 2: Weighted Correlation with New Symbol", Fore.CYAN, Style.BRIGHT))
        print(color_text("="*60, Fore.CYAN))
        
        new_symbol = "BNB/USDT"
        weighted_corr, details = analyzer.calculate_weighted_correlation_with_new_symbol(new_symbol, verbose=True)
        
        if weighted_corr is not None:
            print(color_text(f"\n[OK] Weighted correlation calculated: {weighted_corr:.4f}", Fore.GREEN))
        else:
            print(color_text("\n[WARNING] Could not calculate weighted correlation", Fore.YELLOW))

        # Test 3: Calculate portfolio return correlation
        print(color_text("\n" + "="*60, Fore.CYAN))
        print(color_text("Test 3: Portfolio Return Correlation", Fore.CYAN, Style.BRIGHT))
        print(color_text("="*60, Fore.CYAN))
        
        portfolio_return_corr, metadata = analyzer.calculate_portfolio_return_correlation(
            new_symbol, verbose=True
        )
        
        if portfolio_return_corr is not None:
            print(color_text(f"\n[OK] Portfolio return correlation calculated: {portfolio_return_corr:.4f}", Fore.GREEN))
            if "samples" in metadata:
                print(color_text(f"Samples used: {metadata['samples']}", Fore.CYAN))
        else:
            print(color_text("\n[WARNING] Could not calculate portfolio return correlation", Fore.YELLOW))

        # Test 4: Analyze correlation impact of adding new symbol
        print(color_text("\n" + "="*60, Fore.CYAN))
        print(color_text("Test 4: Correlation Impact Analysis", Fore.CYAN, Style.BRIGHT))
        print(color_text("="*60, Fore.CYAN))
        
        impact = analyzer.analyze_correlation_with_new_symbol(
            new_symbol=new_symbol,
            new_position_size=800.0,
            new_direction="LONG",
            verbose=True
        )
        
        print(color_text("\n[OK] Impact analysis completed!", Fore.GREEN))
        print(color_text("\nImpact Summary:", Fore.CYAN))
        print(f"  Before internal correlation: {impact['before'].get('internal_correlation', 'N/A')}")
        print(f"  New symbol correlation: {impact['after'].get('new_symbol_correlation', 'N/A')}")
        print(f"  Portfolio return correlation: {impact['after'].get('portfolio_return_correlation', 'N/A')}")
        if 'internal_correlation' in impact['after']:
            print(f"  After internal correlation: {impact['after']['internal_correlation']}")
        if 'correlation_change' in impact['impact']:
            print(f"  Correlation change: {impact['impact']['correlation_change']:.4f}")
            print(f"  Diversification improvement: {impact['impact']['diversification_improvement']}")

        print(color_text("\n[OK] All tests completed successfully!\n", Fore.GREEN, Style.BRIGHT))

    except Exception as e:
        print(color_text(f"\n[ERROR] Error: {e}", Fore.RED, Style.BRIGHT))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

