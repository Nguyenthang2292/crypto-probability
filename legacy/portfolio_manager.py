import ccxt
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict
from colorama import Fore, Style, init as colorama_init

# Try to import from modules.config, fallback to default
try:
    from modules.config import DEFAULT_EXCHANGES
except ImportError:
    DEFAULT_EXCHANGES = [
        "binance",
        "kraken",
        "kucoin",
        "gate",
        "okx",
        "bybit",
        "mexc",
        "huobi",
    ]

# Initialize colorama
colorama_init(autoreset=True)

def color_text(text: str, color: str = Fore.WHITE, style: str = Style.NORMAL) -> str:
    return f"{style}{color}{text}{Style.RESET_ALL}"

@dataclass
class Position:
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    size_usdt: float  # Notional Value (Margin * Leverage)

class PortfolioManager:
    def __init__(self, exchanges=None):
        self.positions: List[Position] = []
        self.exchanges = exchanges or DEFAULT_EXCHANGES
        self.exchange_cache: Dict[str, ccxt.Exchange] = {}  # Cache exchanges by symbol
        self.market_prices = {}

    def add_position(self, symbol: str, direction: str, entry_price: float, size_usdt: float):
        self.positions.append(Position(symbol.upper(), direction.upper(), entry_price, size_usdt))

    def _get_exchange_for_symbol(self, symbol: str):
        """Get working exchange for a symbol, trying multiple exchanges."""
        # Check cache first
        if symbol in self.exchange_cache:
            return self.exchange_cache[symbol]
        
        # Try each exchange
        for exchange_id in self.exchanges:
            try:
                exchange_cls = getattr(ccxt, exchange_id)
                exchange = exchange_cls()
                # Test if symbol exists by trying to fetch ticker
                ticker = exchange.fetch_ticker(symbol)
                if ticker and 'last' in ticker:
                    self.exchange_cache[symbol] = exchange
                    return exchange
            except Exception:
                continue
        
        # No exchange found
        return None

    def fetch_prices(self):
        """Fetches current prices for all symbols in the portfolio with exchange fallback."""
        symbols = list(set([p.symbol for p in self.positions]))
        if not symbols:
            return
        
        print(color_text("Fetching current prices...", Fore.CYAN))
        
        fetched_count = 0
        failed_symbols = []
        
        for symbol in symbols:
            exchange = self._get_exchange_for_symbol(symbol)
            if exchange:
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    if ticker and 'last' in ticker:
                        self.market_prices[symbol] = ticker['last']
                        fetched_count += 1
                        exchange_name = exchange.id if hasattr(exchange, 'id') else 'exchange'
                        print(color_text(f"  [{exchange_name.upper()}] {symbol}: {ticker['last']:.8f}", Fore.GREEN))
                    else:
                        failed_symbols.append(symbol)
                except Exception as e:
                    failed_symbols.append(symbol)
                    print(color_text(f"  Error fetching {symbol}: {e}", Fore.YELLOW))
            else:
                failed_symbols.append(symbol)
                print(color_text(f"  {symbol}: Not found on any exchange", Fore.RED))
        
        if failed_symbols:
            print(color_text(f"\nWarning: Could not fetch prices for {len(failed_symbols)} symbol(s): {', '.join(failed_symbols)}", Fore.YELLOW))
        
        if fetched_count > 0:
            print(color_text(f"\nSuccessfully fetched prices for {fetched_count}/{len(symbols)} symbols", Fore.GREEN))

    def calculate_stats(self):
        """Calculates PnL and Delta for the portfolio."""
        total_pnl = 0
        total_delta = 0
        
        results = []
        
        for p in self.positions:
            current_price = self.market_prices.get(p.symbol)
            if current_price is None:
                continue
                
            # Calculate PnL
            # Long: (Current - Entry) / Entry * Size
            # Short: (Entry - Current) / Entry * Size
            if p.direction == 'LONG':
                pnl_pct = (current_price - p.entry_price) / p.entry_price
                delta = p.size_usdt # Long delta is positive
            else:
                pnl_pct = (p.entry_price - current_price) / p.entry_price
                delta = -p.size_usdt # Short delta is negative
                
            pnl_usdt = pnl_pct * p.size_usdt
            
            total_pnl += pnl_usdt
            total_delta += delta
            
            results.append({
                'Symbol': p.symbol,
                'Direction': p.direction,
                'Entry': p.entry_price,
                'Current': current_price,
                'Size': p.size_usdt,
                'PnL': pnl_usdt,
                'Delta': delta
            })
            
        return pd.DataFrame(results), total_pnl, total_delta

    def fetch_ohlcv(self, symbol, limit=100, timeframe='1h'):
        """Fetches OHLCV data with exchange fallback mechanism."""
        # Try cached exchange first
        exchange = self._get_exchange_for_symbol(symbol)
        if exchange:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if ohlcv and len(ohlcv) > 0:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    return df['close']
            except Exception:
                pass  # Will try other exchanges below
        
        # Try all exchanges if cached one failed
        for exchange_id in self.exchanges:
            try:
                exchange_cls = getattr(ccxt, exchange_id)
                exchange = exchange_cls()
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if ohlcv and len(ohlcv) > 0:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    # Cache successful exchange
                    self.exchange_cache[symbol] = exchange
                    return df['close']
            except Exception:
                continue
        
        # All exchanges failed
        print(color_text(f"Error fetching history for {symbol}: Not found on any exchange", Fore.RED))
        return None

    def calculate_weighted_correlation(self, new_symbol: str):
        """Calculates weighted correlation with entire portfolio based on position sizes."""
        correlations = []
        weights = []
        position_details = []
        
        print(color_text(f"\nCorrelation Analysis (Weighted by Position Size):", Fore.CYAN))
        
        new_series = self.fetch_ohlcv(new_symbol)
        if new_series is None:
            print(color_text(f"Could not fetch price history for {new_symbol}", Fore.RED))
            return None, []
        
        for pos in self.positions:
            pos_series = self.fetch_ohlcv(pos.symbol)
            
            if pos_series is not None:
                # Align timestamps
                df = pd.concat([pos_series, new_series], axis=1, join='inner')
                if len(df) < 10:  # Need at least 10 data points
                    continue
                
                corr = df.iloc[:, 0].corr(df.iloc[:, 1])
                weight = pos.size_usdt
                
                correlations.append(corr)
                weights.append(weight)
                
                position_details.append({
                    'symbol': pos.symbol,
                    'direction': pos.direction,
                    'size': pos.size_usdt,
                    'correlation': corr,
                    'weight': weight
                })
        
        if not correlations:
            print(color_text("Insufficient data for correlation analysis.", Fore.YELLOW))
            return None, []
        
        # Calculate weighted average correlation
        total_weight = sum(weights)
        weighted_corr = sum(c * w for c, w in zip(correlations, weights)) / total_weight
        
        # Display individual correlations
        print(f"\nIndividual Correlations:")
        for detail in position_details:
            corr_color = Fore.GREEN if abs(detail['correlation']) > 0.7 else (Fore.YELLOW if abs(detail['correlation']) > 0.4 else Fore.RED)
            weight_pct = (detail['weight'] / total_weight) * 100
            print(f"  {detail['symbol']:12} ({detail['direction']:5}, {detail['size']:>8.2f} USDT, {weight_pct:>5.1f}%): "
                  + color_text(f"{detail['correlation']:>6.4f}", corr_color))
        
        # Display weighted correlation
        print(f"\n{color_text('Weighted Portfolio Correlation:', Fore.CYAN, Style.BRIGHT)}")
        weighted_corr_color = Fore.GREEN if abs(weighted_corr) > 0.7 else (Fore.YELLOW if abs(weighted_corr) > 0.4 else Fore.RED)
        print(f"  {new_symbol} vs Portfolio: {color_text(f'{weighted_corr:>6.4f}', weighted_corr_color, Style.BRIGHT)}")
        
        return weighted_corr, position_details

    def calculate_portfolio_return_correlation(self, new_symbol: str, min_points: int = 10):
        """Calculates correlation between the portfolio's aggregated return and the new symbol."""
        print(color_text(f"\nPortfolio Return Correlation Analysis:", Fore.CYAN))

        if not self.positions:
            print(color_text("No positions in portfolio to compare against.", Fore.YELLOW))
            return None, {}

        new_series = self.fetch_ohlcv(new_symbol)
        if new_series is None:
            print(color_text(f"Could not fetch price history for {new_symbol}", Fore.RED))
            return None, {}

        symbol_series = {}
        for pos in self.positions:
            if pos.symbol not in symbol_series:
                series = self.fetch_ohlcv(pos.symbol)
                if series is not None:
                    symbol_series[pos.symbol] = series

        if not symbol_series:
            print(color_text("Unable to fetch history for existing positions.", Fore.YELLOW))
            return None, {}

        price_df = pd.DataFrame(symbol_series).dropna(how="all")
        if price_df.empty:
            print(color_text("Insufficient overlapping data among current positions.", Fore.YELLOW))
            return None, {}

        portfolio_returns_df = price_df.pct_change().dropna(how="all")
        new_returns = new_series.pct_change().dropna()

        if portfolio_returns_df.empty or new_returns.empty:
            print(color_text("Insufficient price history to compute returns.", Fore.YELLOW))
            return None, {}

        common_index = portfolio_returns_df.index.intersection(new_returns.index)
        if len(common_index) < min_points:
            print(color_text(f"Need at least {min_points} overlapping points, found {len(common_index)}.", Fore.YELLOW))
            return None, {}

        portfolio_returns = []
        aligned_new_returns = []

        for idx in common_index:
            total_weight = 0.0
            weighted_return = 0.0
            for pos in self.positions:
                if pos.symbol not in portfolio_returns_df.columns:
                    continue
                ret = portfolio_returns_df.at[idx, pos.symbol]
                if pd.isna(ret):
                    continue
                if pos.direction == "SHORT":
                    ret = -ret
                weight = abs(pos.size_usdt)
                if weight <= 0:
                    continue
                weighted_return += ret * weight
                total_weight += weight

            new_ret = new_returns.loc[idx]
            if total_weight > 0 and not pd.isna(new_ret):
                portfolio_returns.append(weighted_return / total_weight)
                aligned_new_returns.append(new_ret)

        if len(portfolio_returns) < min_points:
            print(color_text("Not enough aligned return samples for correlation.", Fore.YELLOW))
            return None, {"samples": len(portfolio_returns)}

        portfolio_return_series = pd.Series(portfolio_returns)
        new_return_series = pd.Series(aligned_new_returns)
        correlation = portfolio_return_series.corr(new_return_series)

        if pd.isna(correlation):
            print(color_text("Unable to compute correlation (insufficient variance).", Fore.YELLOW))
            return None, {"samples": len(portfolio_returns)}

        corr_color = Fore.GREEN if abs(correlation) > 0.7 else (Fore.YELLOW if abs(correlation) > 0.4 else Fore.RED)
        print(f"  Portfolio Return vs {new_symbol}: {color_text(f'{correlation:>6.4f}', corr_color, Style.BRIGHT)}")
        print(color_text(f"  Samples used: {len(portfolio_returns)}", Fore.WHITE))

        return correlation, {"samples": len(portfolio_returns)}

    def analyze_new_trade(self, new_symbol: str, total_delta: float, correlation_mode: str = "weighted"):
        """Analyzes a potential new trade and automatically recommends direction for delta balancing."""
        print(color_text(f"\nAnalyzing potential trade on {new_symbol}...", Fore.CYAN, Style.BRIGHT))
        
        # 1. Auto-recommend direction based on total_delta
        target_delta = -total_delta
        
        if abs(total_delta) < 0.01:  # Already delta neutral
            print(color_text("Portfolio is already Delta Neutral (Delta ≈ 0).", Fore.GREEN))
            recommended_direction = None
        elif total_delta > 0:  # Too much LONG exposure
            recommended_direction = 'SHORT'
            recommended_size = -target_delta  # SHORT: Size = -Delta
            print(color_text(f"Current Total Delta: +{total_delta:.2f} USDT (LONG exposure)", Fore.YELLOW))
            print(color_text(f"RECOMMENDED DIRECTION: SHORT (to reduce LONG exposure)", Fore.GREEN, Style.BRIGHT))
        else:  # total_delta < 0, too much SHORT exposure
            recommended_direction = 'LONG'
            recommended_size = target_delta  # LONG: Size = Delta
            print(color_text(f"Current Total Delta: {total_delta:.2f} USDT (SHORT exposure)", Fore.YELLOW))
            print(color_text(f"RECOMMENDED DIRECTION: LONG (to reduce SHORT exposure)", Fore.GREEN, Style.BRIGHT))
        
        # 2. Calculate Required Size for Delta Neutrality
        if recommended_direction:
            print(color_text(f"\nTo achieve Delta Neutrality:", Fore.CYAN))
            print(color_text(f"  Direction: {recommended_direction}", Fore.WHITE))
            print(color_text(f"  Recommended Size: {recommended_size:.2f} USDT", Fore.GREEN, Style.BRIGHT))
            
            # Also show alternative direction (for comparison)
            alternative_direction = 'LONG' if recommended_direction == 'SHORT' else 'SHORT'
            alternative_size = abs(target_delta) if alternative_direction == 'SHORT' else abs(target_delta)
            print(color_text(f"\nAlternative (NOT recommended):", Fore.YELLOW))
            print(color_text(f"  Direction: {alternative_direction} (will INCREASE delta exposure)", Fore.RED))
            print(color_text(f"  Size needed: {alternative_size:.2f} USDT", Fore.WHITE))

        # 3. Correlation Analysis (Weighted by Portfolio)
        if not self.positions:
            print(color_text("\nNo existing positions for correlation analysis.", Fore.WHITE))
            return recommended_direction, recommended_size if recommended_direction else None, None

        # Calculate weighted correlation with entire portfolio
        if correlation_mode == "portfolio_return":
            corr_value, corr_details = self.calculate_portfolio_return_correlation(new_symbol)
        else:
            corr_value, corr_details = self.calculate_weighted_correlation(new_symbol)
        
        if corr_value is not None:
            print()  # Empty line for spacing
            if abs(corr_value) > 0.7:
                print(color_text("[OK] High correlation detected. This pair is suitable for statistical hedging.", Fore.GREEN, Style.BRIGHT))
            elif abs(corr_value) > 0.4:
                print(color_text("[!] Moderate correlation. Hedge may be partially effective.", Fore.YELLOW))
            else:
                print(color_text("[X] Low correlation. This hedge might be less effective systematically.", Fore.RED))
        
        return recommended_direction, recommended_size if recommended_direction else None, corr_value

def main():
    pm = PortfolioManager()
    
    print(color_text("=== Crypto Portfolio Manager ===", Fore.MAGENTA, Style.BRIGHT))
    
    # Input Loop
    while True:
        print("\nAdd a position (or type 'done' to finish):")
        symbol = input("Symbol (e.g., BTC/USDT): ").strip()
        if symbol.lower() == 'done':
            break
        if not symbol: continue
        
        direction = input("Direction (LONG/SHORT): ").strip().upper()
        if direction not in ['LONG', 'SHORT']:
            print("Invalid direction.")
            continue
            
        try:
            entry = float(input("Entry Price: "))
            size = float(input("Size (USDT Notional): "))
            pm.add_position(symbol, direction, entry, size)
        except ValueError:
            print("Invalid numbers.")
            
    if not pm.positions:
        print("No positions added. Exiting.")
        return

    # Analysis
    pm.fetch_prices()
    df, total_pnl, total_delta = pm.calculate_stats()
    
    print("\n" + color_text("=== PORTFOLIO STATUS ===", Fore.WHITE, Style.BRIGHT))
    print(df.to_string(index=False))
    print("-" * 50)
    print(f"Total PnL: {color_text(f'{total_pnl:.2f} USDT', Fore.GREEN if total_pnl >= 0 else Fore.RED)}")
    print(f"Total Delta: {color_text(f'{total_delta:.2f} USDT', Fore.YELLOW)}")
    
    # Recommendation Loop
    while True:
        print("\n" + color_text("=== NEW TRADE ANALYSIS ===", Fore.MAGENTA, Style.BRIGHT))
        new_symbol = input("Enter candidate symbol (e.g., ETH/USDT) or 'exit': ").strip()
        if new_symbol.lower() == 'exit':
            break
        if not new_symbol:
            continue
            
        # Auto-analyze and recommend direction
        recommended_direction, recommended_size, correlation = pm.analyze_new_trade(new_symbol, total_delta)
        
        # Optional: Allow user to override with manual direction
        print("\n" + color_text("Options:", Fore.CYAN))
        print("1. Accept recommendation (press Enter)")
        print("2. Enter custom direction (type LONG or SHORT)")
        print("3. Skip (type 'skip')")
        
        user_choice = input("Your choice: ").strip().upper()
        
        if user_choice == 'SKIP' or user_choice == '3':
            continue
        elif user_choice in ['LONG', 'SHORT']:
            # User wants custom direction
            custom_direction = user_choice
            print(color_text(f"\nAnalyzing custom {custom_direction} position...", Fore.CYAN))
            
            target_delta = -total_delta
            if custom_direction == 'LONG':
                custom_size = target_delta
            else:
                custom_size = -target_delta
            
            if custom_size < 0:
                print(color_text(f"WARNING: A {custom_direction} position will INCREASE your delta exposure.", Fore.RED))
                print(color_text(f"Recommended direction was: {recommended_direction}", Fore.YELLOW))
            else:
                print(color_text(f"Custom {custom_direction} position size for Delta Neutrality: {custom_size:.2f} USDT", Fore.GREEN))
        elif recommended_direction:
            # User accepts recommendation
            print(color_text(f"\n✓ Accepted recommendation: {recommended_direction} {recommended_size:.2f} USDT", Fore.GREEN, Style.BRIGHT))
        else:
            print(color_text("\nPortfolio is already delta neutral. No recommendation needed.", Fore.WHITE))

if __name__ == "__main__":
    main()
