"""
Correlation analyzer for portfolio correlation calculations.
"""
import pandas as pd
from typing import List, Optional, Dict
from colorama import Fore, Style

try:
    from modules.Position import Position
    from modules.utils import color_text, normalize_symbol
    from modules.config import (
        DEFAULT_CORRELATION_MIN_POINTS,
        DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS,
    )
except ImportError:
    Position = None
    color_text = None
    normalize_symbol = None
    DEFAULT_CORRELATION_MIN_POINTS = 10
    DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS = 10


class CorrelationAnalyzer:
    """Analyzes correlation between portfolio and new symbols."""
    
    def __init__(self, data_fetcher, positions: List[Position]):
        self.data_fetcher = data_fetcher
        self.positions = positions
    
    def calculate_weighted_correlation(self, new_symbol: str, verbose: bool = True):
        """Calculates weighted correlation with entire portfolio based on position sizes."""
        correlations = []
        weights = []
        position_details = []
        
        if verbose:
            print(color_text(f"\nCorrelation Analysis (Weighted by Position Size):", Fore.CYAN))
        
        new_series = self.data_fetcher.fetch_ohlcv(new_symbol)
        if new_series is None:
            if verbose:
                print(color_text(f"Could not fetch price history for {new_symbol}", Fore.RED))
            return None, []
        
        for pos in self.positions:
            pos_series = self.data_fetcher.fetch_ohlcv(pos.symbol)
            
            if pos_series is not None:
                df = pd.concat([pos_series, new_series], axis=1, join='inner')
                if len(df) < DEFAULT_WEIGHTED_CORRELATION_MIN_POINTS:
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
            if verbose:
                print(color_text("Insufficient data for correlation analysis.", Fore.YELLOW))
            return None, []
        
        total_weight = sum(weights)
        weighted_corr = sum(c * w for c, w in zip(correlations, weights)) / total_weight
        
        if verbose:
            print(f"\nIndividual Correlations:")
            for detail in position_details:
                corr_color = Fore.GREEN if abs(detail['correlation']) > 0.7 else (Fore.YELLOW if abs(detail['correlation']) > 0.4 else Fore.RED)
                weight_pct = (detail['weight'] / total_weight) * 100
                print(f"  {detail['symbol']:12} ({detail['direction']:5}, {detail['size']:>8.2f} USDT, {weight_pct:>5.1f}%): "
                      + color_text(f"{detail['correlation']:>6.4f}", corr_color))
            
            print(f"\n{color_text('Weighted Portfolio Correlation:', Fore.CYAN, Style.BRIGHT)}")
            weighted_corr_color = Fore.GREEN if abs(weighted_corr) > 0.7 else (Fore.YELLOW if abs(weighted_corr) > 0.4 else Fore.RED)
            print(f"  {new_symbol} vs Portfolio: {color_text(f'{weighted_corr:>6.4f}', weighted_corr_color, Style.BRIGHT)}")
        
        return weighted_corr, position_details
    
    def calculate_portfolio_return_correlation(self, new_symbol: str, 
                                               min_points: int = DEFAULT_CORRELATION_MIN_POINTS, 
                                               verbose: bool = True):
        """Calculates correlation between the portfolio's aggregated return and the new symbol."""
        if verbose:
            print(color_text(f"\nPortfolio Return Correlation Analysis:", Fore.CYAN))

        if not self.positions:
            if verbose:
                print(color_text("No positions in portfolio to compare against.", Fore.YELLOW))
            return None, {}

        new_series = self.data_fetcher.fetch_ohlcv(new_symbol)
        if new_series is None:
            if verbose:
                print(color_text(f"Could not fetch price history for {new_symbol}", Fore.RED))
            return None, {}

        symbol_series = {}
        for pos in self.positions:
            if pos.symbol not in symbol_series:
                series = self.data_fetcher.fetch_ohlcv(pos.symbol)
                if series is not None:
                    symbol_series[pos.symbol] = series

        if not symbol_series:
            if verbose:
                print(color_text("Unable to fetch history for existing positions.", Fore.YELLOW))
            return None, {}

        price_df = pd.DataFrame(symbol_series).dropna(how="all")
        if price_df.empty:
            if verbose:
                print(color_text("Insufficient overlapping data among current positions.", Fore.YELLOW))
            return None, {}

        portfolio_returns_df = price_df.pct_change().dropna(how="all")
        new_returns = new_series.pct_change().dropna()

        if portfolio_returns_df.empty or new_returns.empty:
            if verbose:
                print(color_text("Insufficient price history to compute returns.", Fore.YELLOW))
            return None, {}

        common_index = portfolio_returns_df.index.intersection(new_returns.index)
        if len(common_index) < min_points:
            if verbose:
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
            if verbose:
                print(color_text("Not enough aligned return samples for correlation.", Fore.YELLOW))
            return None, {"samples": len(portfolio_returns)}

        portfolio_return_series = pd.Series(portfolio_returns)
        new_return_series = pd.Series(aligned_new_returns)
        correlation = portfolio_return_series.corr(new_return_series)

        if pd.isna(correlation):
            if verbose:
                print(color_text("Unable to compute correlation (insufficient variance).", Fore.YELLOW))
            return None, {"samples": len(portfolio_returns)}

        if verbose:
            corr_color = Fore.GREEN if abs(correlation) > 0.7 else (Fore.YELLOW if abs(correlation) > 0.4 else Fore.RED)
            print(f"  Portfolio Return vs {new_symbol}: {color_text(f'{correlation:>6.4f}', corr_color, Style.BRIGHT)}")
            print(color_text(f"  Samples used: {len(portfolio_returns)}", Fore.WHITE))

        return correlation, {"samples": len(portfolio_returns)}

