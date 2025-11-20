"""
Script to run portfolio_manager with positions extracted from the image.
"""
import sys
from portfolio_manager import PortfolioManager
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

def color_text(text: str, color: str = Fore.WHITE, style: str = Style.NORMAL) -> str:
    return f"{style}{color}{text}{Style.RESET_ALL}"

def main():
    """Run portfolio analysis with positions from image."""
    pm = PortfolioManager()
    
    print(color_text("=== Loading Portfolio from Image ===", Fore.MAGENTA, Style.BRIGHT))
    
    # Positions extracted from image
    # Format: (symbol, direction, entry_price, size_usdt)
    positions = [
        ("DASH/USDT", "SHORT", 75.70, 250.34),
        ("LSK/USDT", "LONG", 0.214454, 481.634400),
        ("LISTA/USDT", "SHORT", 0.2125000, 255.8666000),
        ("DEGO/USDT", "LONG", 0.5942176, 274.8626400),
        ("SKATE/USDT", "LONG", 0.0265334, 273.4063600),
        ("WLFI/USDT", "SHORT", 0.1316000, 483.5600000),
        ("ENSO/USDT", "SHORT", 0.847000, 279.042900),
    ]
    
    print(color_text(f"\nAdding {len(positions)} positions...", Fore.CYAN))
    for symbol, direction, entry, size in positions:
        pm.add_position(symbol, direction, entry, size)
        print(f"  {symbol:15} {direction:5} Entry: {entry:>12.8f} Size: {size:>12.2f} USDT")
    
    print(color_text("\n=== Fetching Current Prices ===", Fore.CYAN, Style.BRIGHT))
    pm.fetch_prices()
    
    print(color_text("\n=== Calculating Portfolio Statistics ===", Fore.CYAN, Style.BRIGHT))
    df, total_pnl, total_delta = pm.calculate_stats()
    
    print("\n" + color_text("=== PORTFOLIO STATUS ===", Fore.WHITE, Style.BRIGHT))
    if not df.empty:
        print(df.to_string(index=False))
    else:
        print(color_text("Warning: Could not fetch prices for some symbols. Portfolio stats may be incomplete.", Fore.YELLOW))
        # Calculate delta manually from positions
        total_delta = sum(p.size_usdt if p.direction == 'LONG' else -p.size_usdt for p in pm.positions)
        print(f"\nCalculated Total Delta from positions: {color_text(f'{total_delta:.2f} USDT', Fore.YELLOW, Style.BRIGHT)}")
    
    print("-" * 80)
    if not df.empty:
        print(f"Total PnL: {color_text(f'{total_pnl:.2f} USDT', Fore.GREEN if total_pnl >= 0 else Fore.RED, Style.BRIGHT)}")
    print(f"Total Delta: {color_text(f'{total_delta:.2f} USDT', Fore.YELLOW, Style.BRIGHT)}")
    
    # Analyze potential new trades
    print("\n" + color_text("=== PORTFOLIO ANALYSIS ===", Fore.MAGENTA, Style.BRIGHT))
    print(color_text(f"\nCurrent Portfolio Delta: {total_delta:.2f} USDT", Fore.YELLOW))
    
    if total_delta > 0:
        print(color_text("Portfolio has LONG exposure. Consider SHORT positions to hedge.", Fore.CYAN))
    elif total_delta < 0:
        print(color_text("Portfolio has SHORT exposure. Consider LONG positions to hedge.", Fore.CYAN))
    else:
        print(color_text("Portfolio is Delta Neutral.", Fore.GREEN))
    
    # Auto-analyze common symbols
    print("\n" + color_text("=== AUTOMATIC ANALYSIS ===", Fore.MAGENTA, Style.BRIGHT))
    analysis_symbols = ["BTC/USDT", "ETH/USDT"]
    
    for symbol in analysis_symbols:
        print("\n" + "="*80)
        recommended_direction, recommended_size, correlation = pm.analyze_new_trade(symbol, total_delta)
        
        if recommended_direction:
            print(color_text(f"\n[OK] Final Recommendation for {symbol}:", Fore.GREEN, Style.BRIGHT))
            print(color_text(f"   Direction: {recommended_direction}", Fore.WHITE))
            print(color_text(f"   Size: {recommended_size:.2f} USDT", Fore.GREEN))
            if correlation is not None:
                corr_status = "High" if abs(correlation) > 0.7 else ("Moderate" if abs(correlation) > 0.4 else "Low")
                print(color_text(f"   Correlation: {correlation:.4f} ({corr_status})", Fore.CYAN))
        else:
            print(color_text(f"\n{symbol}: Portfolio is already delta neutral.", Fore.WHITE))
    
    print("\n" + "="*80)
    print(color_text("\n=== ANALYSIS COMPLETE ===", Fore.GREEN, Style.BRIGHT))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(color_text("\n\nProgram interrupted by user.", Fore.YELLOW))
    except Exception as e:
        print(color_text(f"\nError: {e}", Fore.RED, Style.BRIGHT))
        import traceback
        traceback.print_exc()

