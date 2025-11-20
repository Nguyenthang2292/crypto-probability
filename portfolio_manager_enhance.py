import ccxt
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
from typing import List, Optional, Dict
from colorama import Fore, Style, init as colorama_init

# Import Binance positions fetcher
try:
    from modules.binance_positions import get_binance_futures_positions
except ImportError:
    print("Warning: Could not import get_binance_futures_positions from modules.binance_positions")
    get_binance_futures_positions = None

# Try to import API keys from config
try:
    from modules.config_api import BINANCE_API_KEY, BINANCE_API_SECRET
except ImportError:
    BINANCE_API_KEY = None
    BINANCE_API_SECRET = None

# Try to import normalize_symbol from modules
try:
    from modules.data_fetcher import normalize_symbol
except ImportError:
    # Fallback normalize_symbol function
    DEFAULT_QUOTE = "USDT"
    def normalize_symbol(user_input: str, quote: str = DEFAULT_QUOTE) -> str:
        """Converts user input like 'eth' into 'ETH/USDT'. Keeps existing slash pairs."""
        if not user_input:
            return f"BTC/{quote}"
        
        norm = user_input.strip().upper()
        if "/" in norm:
            return norm
        
        if norm.endswith(quote):
            return f"{norm[:-len(quote)]}/{quote}"
        
        return f"{norm}/{quote}"

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
    def __init__(self, api_key=None, api_secret=None, testnet=False):
        self.positions: List[Position] = []
        self.market_prices = {}
        self.api_key = api_key or os.getenv('BINANCE_API_KEY') or BINANCE_API_KEY
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET') or BINANCE_API_SECRET
        self.testnet = testnet
        self._binance_exchange = None  # Lazy-loaded Binance exchange
        self.benchmark_symbol = "BTC/USDT"
        self._beta_cache: Dict[str, float] = {}
        self.last_var_value: Optional[float] = None
        self.last_var_confidence: Optional[float] = None

    def add_position(self, symbol: str, direction: str, entry_price: float, size_usdt: float):
        self.positions.append(Position(symbol.upper(), direction.upper(), entry_price, size_usdt))

    def _get_binance_exchange(self):
        """Get Binance exchange instance, creating if needed."""
        if self._binance_exchange is None:
            if not self.api_key or not self.api_secret:
                raise ValueError(
                    "API Key và API Secret là bắt buộc!\n"
                    "Cung cấp qua một trong các cách sau:\n"
                    "  1. Tham số khi khởi tạo PortfolioManager\n"
                    "  2. Biến môi trường: BINANCE_API_KEY và BINANCE_API_SECRET\n"
                    "  3. File config: modules/config_api.py (BINANCE_API_KEY và BINANCE_API_SECRET)"
                )
            
            options = {
                'defaultType': 'future',
                'options': {
                    'defaultType': 'future',
                }
            }
            
            if self.testnet:
                self._binance_exchange = ccxt.binance({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'options': options,
                    'sandbox': True,
                })
            else:
                self._binance_exchange = ccxt.binance({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'options': options,
                })
        
        return self._binance_exchange

    def load_from_binance(self, api_key=None, api_secret=None, testnet=None, debug=False):
        """Load positions directly from Binance Futures USDT-M."""
        if get_binance_futures_positions is None:
            raise ImportError("Cannot import get_binance_futures_positions from modules.binance_positions")
        
        # Use provided credentials or instance credentials
        if api_key is not None:
            self.api_key = api_key
        if api_secret is not None:
            self.api_secret = api_secret
        if testnet is not None:
            self.testnet = testnet
        
        print(color_text("Loading positions from Binance Futures USDT-M...", Fore.CYAN, Style.BRIGHT))
        
        try:
            binance_positions = get_binance_futures_positions(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
                debug=debug
            )
            
            if not binance_positions:
                print(color_text("No open positions found on Binance.", Fore.YELLOW))
                return
            
            # Clear existing positions (or append if needed)
            self.positions.clear()
            
            # Add positions from Binance
            for pos in binance_positions:
                self.add_position(
                    symbol=pos['symbol'],
                    direction=pos['direction'],
                    entry_price=pos['entry_price'],
                    size_usdt=pos['size_usdt']
                )
            
            print(color_text(f"✓ Loaded {len(binance_positions)} position(s) from Binance", Fore.GREEN))
            
            # Display loaded positions
            print("\n" + color_text("Loaded Positions:", Fore.CYAN))
            for pos in binance_positions:
                print(f"  {pos['symbol']:<15} {pos['direction']:<5} Entry: {pos['entry_price']:>12.8f} Size: {pos['size_usdt']:>12.2f} USDT")
            
        except Exception as e:
            raise ValueError(f"Error loading positions from Binance: {e}")

    def fetch_prices(self):
        """Fetches current prices for all symbols from Binance."""
        symbols = list(set([p.symbol for p in self.positions]))
        if not symbols:
            return
        
        print(color_text("Fetching current prices from Binance...", Fore.CYAN))
        
        try:
            exchange = self._get_binance_exchange()
        except ValueError as e:
            print(color_text(f"Error: {e}", Fore.RED))
            return
        
        fetched_count = 0
        failed_symbols = []
        
        for symbol in symbols:
            # Ensure symbol is normalized
            normalized_symbol = normalize_symbol(symbol)
            try:
                ticker = exchange.fetch_ticker(normalized_symbol)
                if ticker and 'last' in ticker:
                    # Store price with original symbol key
                    self.market_prices[symbol] = ticker['last']
                    fetched_count += 1
                    print(color_text(f"  [BINANCE] {normalized_symbol}: {ticker['last']:.8f}", Fore.GREEN))
                else:
                    failed_symbols.append(symbol)
                    print(color_text(f"  {normalized_symbol}: No price data available", Fore.YELLOW))
            except Exception as e:
                failed_symbols.append(symbol)
                print(color_text(f"  Error fetching {normalized_symbol}: {e}", Fore.YELLOW))
        
        if failed_symbols:
            print(color_text(f"\nWarning: Could not fetch prices for {len(failed_symbols)} symbol(s): {', '.join(failed_symbols)}", Fore.YELLOW))
        
        if fetched_count > 0:
            print(color_text(f"\nSuccessfully fetched prices for {fetched_count}/{len(symbols)} symbols", Fore.GREEN))

    def calculate_stats(self):
        """Calculates PnL, simple delta, and beta-weighted delta for the portfolio."""
        total_pnl = 0
        total_delta = 0
        total_beta_delta = 0
        
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
                delta = p.size_usdt  # Long delta is positive
            else:
                pnl_pct = (p.entry_price - current_price) / p.entry_price
                delta = -p.size_usdt  # Short delta is negative
                
            pnl_usdt = pnl_pct * p.size_usdt
            
            total_pnl += pnl_usdt
            total_delta += delta
            
            beta = self.calculate_beta(p.symbol)
            beta_delta = None
            if beta is not None:
                beta_delta = delta * beta
                total_beta_delta += beta_delta
            
            results.append({
                'Symbol': p.symbol,
                'Direction': p.direction,
                'Entry': p.entry_price,
                'Current': current_price,
                'Size': p.size_usdt,
                'PnL': pnl_usdt,
                'Delta': delta,
                'Beta': beta,
                'Beta Delta': beta_delta
            })
            
        return pd.DataFrame(results), total_pnl, total_delta, total_beta_delta

    def fetch_ohlcv(self, symbol, limit=1000, timeframe='1h'):
        """Fetches OHLCV data from Binance with a longer lookback window for correlation stability."""
        # Normalize symbol format
        normalized_symbol = normalize_symbol(symbol)
        
        try:
            exchange = self._get_binance_exchange()
        except ValueError as e:
            print(color_text(f"Error: {e}", Fore.RED))
            return None
        
        try:
            ohlcv = exchange.fetch_ohlcv(normalized_symbol, timeframe=timeframe, limit=limit)
            if ohlcv and len(ohlcv) > 0:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df['close']
            else:
                print(color_text(f"Error fetching history for {normalized_symbol}: No data returned from Binance", Fore.RED))
                return None
        except Exception as e:
            print(color_text(f"Error fetching history for {normalized_symbol}: {e}", Fore.RED))
            return None

    def calculate_beta(self, symbol: str, benchmark_symbol: Optional[str] = None, min_points: int = 50,
                       limit: int = 1000, timeframe: str = '1h') -> Optional[float]:
        """Calculates beta of a symbol versus a benchmark (default BTC/USDT)."""
        benchmark_symbol = benchmark_symbol or self.benchmark_symbol
        normalized_symbol = normalize_symbol(symbol)
        normalized_benchmark = normalize_symbol(benchmark_symbol)
        cache_key = f"{normalized_symbol}|{normalized_benchmark}|{timeframe}|{limit}"
        
        if normalized_symbol == normalized_benchmark:
            return 1.0
        
        if cache_key in self._beta_cache:
            return self._beta_cache[cache_key]
        
        asset_series = self.fetch_ohlcv(normalized_symbol, limit=limit, timeframe=timeframe)
        benchmark_series = self.fetch_ohlcv(normalized_benchmark, limit=limit, timeframe=timeframe)
        
        if asset_series is None or benchmark_series is None:
            return None
        
        df = pd.concat([asset_series, benchmark_series], axis=1, join='inner').dropna()
        if len(df) < min_points:
            return None
        
        returns = df.pct_change().dropna()
        if returns.empty:
            return None
        
        benchmark_var = returns.iloc[:, 1].var()
        if benchmark_var is None or benchmark_var <= 0:
            return None
        
        covariance = returns.iloc[:, 0].cov(returns.iloc[:, 1])
        if covariance is None:
            return None
        
        beta = covariance / benchmark_var
        if pd.isna(beta):
            return None
        
        self._beta_cache[cache_key] = beta
        return beta

    def calculate_portfolio_var(self, confidence: float = 0.95, lookback_days: int = 90) -> Optional[float]:
        """Calculates Historical Simulation VaR for the current portfolio."""
        self.last_var_value = None
        self.last_var_confidence = None
        if not self.positions:
            print(color_text("No positions available for VaR calculation.", Fore.YELLOW))
            return None
        
        confidence_pct = int(confidence * 100)
        print(color_text(f"\nCalculating Historical VaR ({confidence_pct}% confidence, {lookback_days}d lookback)...", Fore.CYAN))
        
        price_history = {}
        fetch_limit = max(lookback_days * 2, lookback_days + 50)
        for pos in self.positions:
            series = self.fetch_ohlcv(pos.symbol, limit=fetch_limit, timeframe='1d')
            if series is not None:
                price_history[pos.symbol] = series
        
        if not price_history:
            print(color_text("Unable to fetch historical data for VaR.", Fore.YELLOW))
            return None
        
        price_df = pd.DataFrame(price_history).dropna(how="all")
        if price_df.empty:
            print(color_text("No overlapping history found for VaR.", Fore.YELLOW))
            return None
        
        if len(price_df) < lookback_days:
            print(color_text(f"Only {len(price_df)} daily points available (requested {lookback_days}). Using available history.", Fore.YELLOW))
        price_df = price_df.tail(lookback_days)
        
        if len(price_df) < 20:
            print(color_text("Insufficient history (<20 days) for reliable VaR.", Fore.YELLOW))
            return None
        
        returns_df = price_df.pct_change().dropna(how="all")
        if returns_df.empty:
            print(color_text("Unable to compute returns for VaR.", Fore.YELLOW))
            return None
        
        daily_pnls = []
        for idx in returns_df.index:
            daily_pnl = 0.0
            has_data = False
            for pos in self.positions:
                if pos.symbol not in returns_df.columns:
                    continue
                ret = returns_df.at[idx, pos.symbol]
                if pd.isna(ret):
                    continue
                exposure = pos.size_usdt if pos.direction == 'LONG' else -pos.size_usdt
                daily_pnl += exposure * ret
                has_data = True
            if has_data:
                daily_pnls.append(daily_pnl)
        
        if len(daily_pnls) < 10:
            print(color_text("Not enough historical PnL samples for VaR.", Fore.YELLOW))
            return None
        
        percentile = max(0, min(100, (1 - confidence) * 100))
        loss_percentile = np.percentile(daily_pnls, percentile)
        var_amount = max(0.0, -loss_percentile)
        
        print(color_text(f"Historical VaR ({confidence_pct}%): {var_amount:.2f} USDT", Fore.MAGENTA, Style.BRIGHT))
        self.last_var_value = var_amount
        self.last_var_confidence = confidence
        return var_amount

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

    def analyze_new_trade(self, new_symbol: str, total_delta: float, total_beta_delta: float,
                          correlation_mode: str = "weighted"):
        """Analyzes a potential new trade and automatically recommends direction for beta-weighted hedging."""
        # Normalize symbol: "eth" -> "ETH/USDT"
        normalized_symbol = normalize_symbol(new_symbol)
        if normalized_symbol != new_symbol:
            print(color_text(f"Symbol normalized: '{new_symbol}' -> '{normalized_symbol}'", Fore.CYAN))
        
        new_symbol = normalized_symbol
        print(color_text(f"\nAnalyzing potential trade on {new_symbol}...", Fore.CYAN, Style.BRIGHT))
        print(color_text(f"Current Total Delta: {total_delta:+.2f} USDT", Fore.WHITE))
        print(color_text(f"Current Total Beta Delta (vs {self.benchmark_symbol}): {total_beta_delta:+.2f} USDT", Fore.WHITE))
        
        new_symbol_beta = self.calculate_beta(new_symbol)
        beta_available = new_symbol_beta is not None and abs(new_symbol_beta) > 1e-6
        if beta_available:
            print(color_text(f"{new_symbol} beta vs {self.benchmark_symbol}: {new_symbol_beta:.4f}", Fore.CYAN))
        else:
            print(color_text(f"Could not compute beta for {new_symbol}. Falling back to simple delta hedging.", Fore.YELLOW))
        
        hedge_mode = "beta" if beta_available else "delta"
        metric_label = "Beta Delta" if beta_available else "Delta"
        current_metric = total_beta_delta if beta_available else total_delta
        target_metric = -current_metric
        
        recommended_direction = None
        recommended_size = None
        
        if abs(current_metric) < 0.01:
            print(color_text(f"Portfolio is already {metric_label} Neutral ({metric_label} ≈ 0).", Fore.GREEN))
        else:
            if beta_available:
                beta_sign = np.sign(new_symbol_beta)
                if beta_sign == 0:
                    beta_available = False
                    hedge_mode = "delta"
                    metric_label = "Delta"
                    current_metric = total_delta
                    target_metric = -current_metric
                else:
                    direction_multiplier = -np.sign(current_metric) * beta_sign
                    recommended_direction = 'LONG' if direction_multiplier >= 0 else 'SHORT'
                    recommended_size = abs(current_metric) / max(abs(new_symbol_beta), 1e-6)
                    print(color_text(f"Targeting Beta Neutrality using {metric_label}.", Fore.CYAN))
            if not beta_available:
                if current_metric > 0:
                    recommended_direction = 'SHORT'
                    recommended_size = abs(target_metric)
                    print(color_text("Portfolio has excess LONG delta exposure.", Fore.YELLOW))
                else:
                    recommended_direction = 'LONG'
                    recommended_size = abs(target_metric)
                    print(color_text("Portfolio has excess SHORT delta exposure.", Fore.YELLOW))
                print(color_text("Targeting simple Delta Neutrality.", Fore.CYAN))
        
        if recommended_direction and recommended_size is not None:
            print(color_text(f"\nRecommended {hedge_mode.upper()} hedge:", Fore.CYAN, Style.BRIGHT))
            print(color_text(f"  Direction: {recommended_direction}", Fore.WHITE))
            print(color_text(f"  Size: {recommended_size:.2f} USDT", Fore.GREEN, Style.BRIGHT))

        # 3. Correlation Analysis - Hiển thị cả 2 phương pháp
        if not self.positions:
            print(color_text("\nNo existing positions for correlation analysis.", Fore.WHITE))
            return recommended_direction, recommended_size if recommended_direction else None, None

        print(color_text("\n" + "="*70, Fore.CYAN))
        print(color_text("CORRELATION ANALYSIS - COMPARING BOTH METHODS", Fore.CYAN, Style.BRIGHT))
        print(color_text("="*70, Fore.CYAN))
        
        # Method 1: Weighted Correlation
        weighted_corr, weighted_details = self.calculate_weighted_correlation(new_symbol)
        
        # Method 2: Portfolio Return Correlation
        portfolio_return_corr, portfolio_return_details = self.calculate_portfolio_return_correlation(new_symbol)
        
        # Tổng hợp kết quả
        print(color_text("\n" + "="*70, Fore.CYAN))
        print(color_text("CORRELATION SUMMARY", Fore.MAGENTA, Style.BRIGHT))
        print(color_text("="*70, Fore.CYAN))
        
        if weighted_corr is not None:
            weighted_color = Fore.GREEN if abs(weighted_corr) > 0.7 else (Fore.YELLOW if abs(weighted_corr) > 0.4 else Fore.RED)
            print(f"1. Weighted Correlation (by Position Size):")
            print(f"   {new_symbol} vs Portfolio: {color_text(f'{weighted_corr:>6.4f}', weighted_color, Style.BRIGHT)}")
            
            if abs(weighted_corr) > 0.7:
                print(color_text("   → High correlation. Good for hedging.", Fore.GREEN))
            elif abs(weighted_corr) > 0.4:
                print(color_text("   → Moderate correlation. Partial hedging effect.", Fore.YELLOW))
            else:
                print(color_text("   → Low correlation. Limited hedging effectiveness.", Fore.RED))
        else:
            print(f"1. Weighted Correlation: {color_text('N/A (insufficient data)', Fore.YELLOW)}")
        
        if portfolio_return_corr is not None:
            portfolio_color = Fore.GREEN if abs(portfolio_return_corr) > 0.7 else (Fore.YELLOW if abs(portfolio_return_corr) > 0.4 else Fore.RED)
            samples_info = portfolio_return_details.get('samples', 'N/A') if isinstance(portfolio_return_details, dict) else 'N/A'
            print(f"\n2. Portfolio Return Correlation (includes direction):")
            print(f"   {new_symbol} vs Portfolio Return: {color_text(f'{portfolio_return_corr:>6.4f}', portfolio_color, Style.BRIGHT)}")
            print(f"   Samples used: {samples_info}")
            
            if abs(portfolio_return_corr) > 0.7:
                print(color_text("   → High correlation. Excellent for hedging.", Fore.GREEN))
            elif abs(portfolio_return_corr) > 0.4:
                print(color_text("   → Moderate correlation. Acceptable hedging effect.", Fore.YELLOW))
            else:
                print(color_text("   → Low correlation. Poor hedging effectiveness.", Fore.RED))
        else:
            print(f"\n2. Portfolio Return Correlation: {color_text('N/A (insufficient data)', Fore.YELLOW)}")
        
        # So sánh và đánh giá chung
        print(color_text("\n" + "-"*70, Fore.WHITE))
        print(color_text("OVERALL ASSESSMENT:", Fore.CYAN, Style.BRIGHT))
        
        if weighted_corr is not None and portfolio_return_corr is not None:
            # So sánh 2 phương pháp
            diff = abs(weighted_corr - portfolio_return_corr)
            if diff < 0.1:
                print(color_text("   ✓ Both methods show similar correlation → Consistent result", Fore.GREEN))
            else:
                print(color_text(f"   ⚠ Methods differ by {diff:.4f} → Check if portfolio has SHORT positions", Fore.YELLOW))
            
            # Đánh giá chung
            avg_corr = (abs(weighted_corr) + abs(portfolio_return_corr)) / 2
            if avg_corr > 0.7:
                print(color_text("   [OK] High correlation detected. This pair is suitable for statistical hedging.", Fore.GREEN, Style.BRIGHT))
            elif avg_corr > 0.4:
                print(color_text("   [!] Moderate correlation. Hedge may be partially effective.", Fore.YELLOW))
            else:
                print(color_text("   [X] Low correlation. This hedge might be less effective systematically.", Fore.RED))
        
        elif weighted_corr is not None:
            # Chỉ có weighted correlation
            if abs(weighted_corr) > 0.7:
                print(color_text("   [OK] High correlation detected. This pair is suitable for statistical hedging.", Fore.GREEN, Style.BRIGHT))
            elif abs(weighted_corr) > 0.4:
                print(color_text("   [!] Moderate correlation. Hedge may be partially effective.", Fore.YELLOW))
            else:
                print(color_text("   [X] Low correlation. This hedge might be less effective systematically.", Fore.RED))
        
        elif portfolio_return_corr is not None:
            # Chỉ có portfolio return correlation
            if abs(portfolio_return_corr) > 0.7:
                print(color_text("   [OK] High correlation detected. This pair is suitable for statistical hedging.", Fore.GREEN, Style.BRIGHT))
            elif abs(portfolio_return_corr) > 0.4:
                print(color_text("   [!] Moderate correlation. Hedge may be partially effective.", Fore.YELLOW))
            else:
                print(color_text("   [X] Low correlation. This hedge might be less effective systematically.", Fore.RED))

        if self.last_var_value is not None and self.last_var_confidence is not None:
            conf_pct = int(self.last_var_confidence * 100)
            print(color_text("\nVaR INSIGHT:", Fore.MAGENTA, Style.BRIGHT))
            print(color_text(f"  With {conf_pct}% confidence, daily loss is unlikely to exceed {self.last_var_value:.2f} USDT.", Fore.WHITE))
            print(color_text("  Use this ceiling to judge whether the proposed hedge keeps risk tolerable.", Fore.WHITE))
        else:
            print(color_text("\nVaR INSIGHT: N/A (insufficient historical data for VaR).", Fore.YELLOW))
        
        print(color_text("="*70 + "\n", Fore.CYAN))
        
        # Trả về correlation chính (weighted làm mặc định vì đơn giản hơn)
        final_corr = weighted_corr if weighted_corr is not None else portfolio_return_corr
        
        return recommended_direction, recommended_size if recommended_direction else None, final_corr

def main():
    print(color_text("=== Crypto Portfolio Manager (Binance Integration) ===", Fore.MAGENTA, Style.BRIGHT))
    
    # Initialize PortfolioManager
    try:
        pm = PortfolioManager()
    except Exception as e:
        print(color_text(f"Error initializing PortfolioManager: {e}", Fore.RED))
        return
    
    # Auto-load from Binance
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

    # Analysis
    pm.fetch_prices()
    df, total_pnl, total_delta, total_beta_delta = pm.calculate_stats()
    
    print("\n" + color_text("=== PORTFOLIO STATUS ===", Fore.WHITE, Style.BRIGHT))
    print(df.to_string(index=False))
    print("-" * 50)
    print(f"Total PnL: {color_text(f'{total_pnl:.2f} USDT', Fore.GREEN if total_pnl >= 0 else Fore.RED)}")
    print(f"Total Delta: {color_text(f'{total_delta:.2f} USDT', Fore.YELLOW)}")
    print(f"Total Beta Delta (vs {pm.benchmark_symbol}): {color_text(f'{total_beta_delta:.2f} USDT', Fore.YELLOW)}")
    
    var_value = pm.calculate_portfolio_var(confidence=0.95, lookback_days=90)
    if var_value is not None:
        conf_pct = int((pm.last_var_confidence or 0) * 100)
        print(color_text(f"Interpretation: With {conf_pct}% confidence, daily loss should stay within {var_value:.2f} USDT.", Fore.WHITE))
    else:
        print(color_text("VaR Interpretation: Not enough history for a reliable estimate.", Fore.YELLOW))
    
    # Recommendation Loop
    while True:
        print("\n" + color_text("=== NEW TRADE ANALYSIS ===", Fore.MAGENTA, Style.BRIGHT))
        new_symbol_input = input("Enter candidate symbol (e.g., ETH/USDT or eth) or 'exit': ").strip()
        if new_symbol_input.lower() == 'exit':
            break
        if not new_symbol_input:
            continue
        
        # Normalize symbol: "eth" -> "ETH/USDT"
        new_symbol = normalize_symbol(new_symbol_input)
        if new_symbol != new_symbol_input:
            print(color_text(f"✓ Symbol normalized: '{new_symbol_input}' -> '{new_symbol}'", Fore.GREEN))
            
        # Auto-analyze and recommend direction
        recommended_direction, recommended_size, correlation = pm.analyze_new_trade(new_symbol, total_delta, total_beta_delta)
        
        # Automatically accept recommendation if available
        if recommended_direction and recommended_size is not None:
            print(color_text(f"\n✓ Automatically accepted recommendation: {recommended_direction} {recommended_size:.2f} USDT", Fore.GREEN, Style.BRIGHT))
        else:
            print(color_text("\nPortfolio is already delta neutral. No recommendation needed.", Fore.WHITE))

if __name__ == "__main__":
    main()
