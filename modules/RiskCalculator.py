"""
Risk calculator for portfolio risk metrics (PnL, Delta, Beta, VaR).
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from colorama import Fore, Style

try:
    from modules.Position import Position
    from modules.utils import color_text
    from modules.config import (
        DEFAULT_BETA_MIN_POINTS,
        DEFAULT_BETA_LIMIT,
        DEFAULT_BETA_TIMEFRAME,
        DEFAULT_VAR_CONFIDENCE,
        DEFAULT_VAR_LOOKBACK_DAYS,
        DEFAULT_VAR_MIN_HISTORY_DAYS,
        DEFAULT_VAR_MIN_PNL_SAMPLES,
        BENCHMARK_SYMBOL,
    )
except ImportError:
    Position = None
    color_text = None
    DEFAULT_BETA_MIN_POINTS = 50
    DEFAULT_BETA_LIMIT = 1000
    DEFAULT_BETA_TIMEFRAME = "1h"
    DEFAULT_VAR_CONFIDENCE = 0.95
    DEFAULT_VAR_LOOKBACK_DAYS = 90
    DEFAULT_VAR_MIN_HISTORY_DAYS = 20
    DEFAULT_VAR_MIN_PNL_SAMPLES = 10
    BENCHMARK_SYMBOL = "BTC/USDT"


class RiskCalculator:
    """Calculates portfolio risk metrics."""
    
    def __init__(self, data_fetcher, benchmark_symbol: str = BENCHMARK_SYMBOL):
        self.data_fetcher = data_fetcher
        self.benchmark_symbol = benchmark_symbol
        self._beta_cache: Dict[str, float] = {}
        self.last_var_value: Optional[float] = None
        self.last_var_confidence: Optional[float] = None
    
    def calculate_stats(self, positions: List[Position], market_prices: Dict[str, float]):
        """Calculates PnL, simple delta, and beta-weighted delta for the portfolio."""
        total_pnl = 0
        total_delta = 0
        total_beta_delta = 0
        
        results = []
        
        for p in positions:
            current_price = market_prices.get(p.symbol)
            if current_price is None:
                continue
                
            if p.direction == 'LONG':
                pnl_pct = (current_price - p.entry_price) / p.entry_price
                delta = p.size_usdt
            else:
                pnl_pct = (p.entry_price - current_price) / p.entry_price
                delta = -p.size_usdt
                
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
    
    def calculate_beta(self, symbol: str, benchmark_symbol: Optional[str] = None, 
                      min_points: int = DEFAULT_BETA_MIN_POINTS,
                      limit: int = DEFAULT_BETA_LIMIT, 
                      timeframe: str = DEFAULT_BETA_TIMEFRAME) -> Optional[float]:
        """Calculates beta of a symbol versus a benchmark (default BTC/USDT)."""
        benchmark_symbol = benchmark_symbol or self.benchmark_symbol
        try:
            from modules.utils import normalize_symbol
        except ImportError:
            def normalize_symbol(user_input: str, quote: str = "USDT") -> str:
                if not user_input:
                    return f"BTC/{quote}"
                norm = user_input.strip().upper()
                if "/" in norm:
                    return norm
                if norm.endswith(quote):
                    return f"{norm[:-len(quote)]}/{quote}"
                return f"{norm}/{quote}"
        
        normalized_symbol = normalize_symbol(symbol)
        normalized_benchmark = normalize_symbol(benchmark_symbol)
        cache_key = f"{normalized_symbol}|{normalized_benchmark}|{timeframe}|{limit}"
        
        if normalized_symbol == normalized_benchmark:
            return 1.0
        
        if cache_key in self._beta_cache:
            return self._beta_cache[cache_key]
        
        asset_series = self.data_fetcher.fetch_ohlcv(normalized_symbol, limit=limit, timeframe=timeframe)
        benchmark_series = self.data_fetcher.fetch_ohlcv(normalized_benchmark, limit=limit, timeframe=timeframe)
        
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
    
    def calculate_portfolio_var(self, positions: List[Position], 
                                confidence: float = DEFAULT_VAR_CONFIDENCE, 
                                lookback_days: int = DEFAULT_VAR_LOOKBACK_DAYS) -> Optional[float]:
        """Calculates Historical Simulation VaR for the current portfolio."""
        self.last_var_value = None
        self.last_var_confidence = None
        if not positions:
            print(color_text("No positions available for VaR calculation.", Fore.YELLOW))
            return None
        
        confidence_pct = int(confidence * 100)
        print(color_text(f"\nCalculating Historical VaR ({confidence_pct}% confidence, {lookback_days}d lookback)...", Fore.CYAN))
        
        price_history = {}
        fetch_limit = max(lookback_days * 2, lookback_days + 50)
        for pos in positions:
            series = self.data_fetcher.fetch_ohlcv(pos.symbol, limit=fetch_limit, timeframe='1d')
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
        
        if len(price_df) < DEFAULT_VAR_MIN_HISTORY_DAYS:
            print(color_text(f"Insufficient history (<{DEFAULT_VAR_MIN_HISTORY_DAYS} days) for reliable VaR.", Fore.YELLOW))
            return None
        
        returns_df = price_df.pct_change().dropna(how="all")
        if returns_df.empty:
            print(color_text("Unable to compute returns for VaR.", Fore.YELLOW))
            return None
        
        daily_pnls = []
        for idx in returns_df.index:
            daily_pnl = 0.0
            has_data = False
            for pos in positions:
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
        
        if len(daily_pnls) < DEFAULT_VAR_MIN_PNL_SAMPLES:
            print(color_text(f"Not enough historical PnL samples for VaR (need at least {DEFAULT_VAR_MIN_PNL_SAMPLES}).", Fore.YELLOW))
            return None
        
        percentile = max(0, min(100, (1 - confidence) * 100))
        loss_percentile = np.percentile(daily_pnls, percentile)
        var_amount = max(0.0, -loss_percentile)
        
        print(color_text(f"Historical VaR ({confidence_pct}%): {var_amount:.2f} USDT", Fore.MAGENTA, Style.BRIGHT))
        self.last_var_value = var_amount
        self.last_var_confidence = confidence
        return var_amount

