import argparse
import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from colorama import Fore, Style, init as colorama_init
from datetime import datetime

# Initialize colorama
colorama_init(autoreset=True)

def color_text(text: str, color: str = Fore.WHITE, style: str = Style.NORMAL) -> str:
    return f"{style}{color}{text}{Style.RESET_ALL}"

def fetch_data(symbol, timeframe="1h", limit=1000):
    """Fetch OHLCV data for a single symbol."""
    print(color_text(f"Fetching {limit} candles for {symbol}...", Fore.CYAN))
    try:
        exchange = ccxt.binance() # Default to Binance for consistency
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        print(color_text(f"Error fetching {symbol}: {e}", Fore.RED))
        return None

def align_data(df1, df2):
    """Align two DataFrames on their index (timestamp)."""
    # Inner join to keep only overlapping timestamps
    aligned = pd.concat([df1["close"], df2["close"]], axis=1, join="inner")
    aligned.columns = ["asset1", "asset2"]
    return aligned

def calculate_spread_and_zscore(df, window=20):
    """
    Calculates the spread, hedge ratio, and Z-Score.
    Uses a rolling linear regression to adapt to changing correlations.
    """
    # Log prices are often better for ratios/spreads
    y = np.log(df["asset1"])
    x = np.log(df["asset2"])
    
    # Calculate rolling Hedge Ratio (Beta)
    # This is a simplified rolling regression
    # Beta = Cov(x,y) / Var(x)
    rolling_cov = x.rolling(window=window).cov(y)
    rolling_var = x.rolling(window=window).var()
    df["beta"] = rolling_cov / rolling_var
    
    # Spread = log(Asset1) - Beta * log(Asset2)
    # Note: In a real static model, Beta is constant. In dynamic, it changes.
    # We use the current beta to estimate the current spread relation.
    df["spread"] = y - (df["beta"] * x)
    
    # Calculate Z-Score of the spread
    # Z = (Spread - Mean(Spread)) / Std(Spread)
    df["spread_mean"] = df["spread"].rolling(window=window).mean()
    df["spread_std"] = df["spread"].rolling(window=window).std()
    df["z_score"] = (df["spread"] - df["spread_mean"]) / df["spread_std"]
    
    return df

def generate_signals(df, entry_threshold=2.0, exit_threshold=0.0):
    """
    Generates Long/Short signals for the spread.
    
    Long Spread (Buy Asset1, Sell Asset2) when Z-Score < -Threshold
    Short Spread (Sell Asset1, Buy Asset2) when Z-Score > +Threshold
    Exit when Z-Score crosses zero (or exit threshold).
    """
    df["signal"] = 0 # 0: None, 1: Long Spread, -1: Short Spread, 9: Exit
    df["position"] = 0 # Current position state
    
    current_pos = 0 # 0: Flat, 1: Long, -1: Short
    
    positions = []
    
    for i in range(len(df)):
        z = df["z_score"].iloc[i]
        
        if pd.isna(z):
            positions.append(0)
            continue
            
        # Entry Logic
        if current_pos == 0:
            if z < -entry_threshold:
                current_pos = 1 # Long Spread
            elif z > entry_threshold:
                current_pos = -1 # Short Spread
        
        # Exit Logic
        elif current_pos == 1: # Currently Long
            if z >= -exit_threshold: # Crossed back up to mean
                current_pos = 0
        elif current_pos == -1: # Currently Short
            if z <= exit_threshold: # Crossed back down to mean
                current_pos = 0
                
        positions.append(current_pos)
        
    df["position"] = positions
    return df

def plot_results(df, symbol1, symbol2):
    """Visualize the prices, spread, and Z-Score using Plotly."""
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05,
                        subplot_titles=(f"Price Comparison: {symbol1} vs {symbol2}", "Spread (Log Divergence)", "Z-Score (Trading Signals)"),
                        row_heights=[0.4, 0.3, 0.3])

    # 1. Prices (Normalized to start at 100 for comparison)
    norm_asset1 = df["asset1"] / df["asset1"].iloc[0] * 100
    norm_asset2 = df["asset2"] / df["asset2"].iloc[0] * 100
    
    fig.add_trace(go.Scatter(x=df.index, y=norm_asset1, name=symbol1, line=dict(color='cyan')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=norm_asset2, name=symbol2, line=dict(color='orange')), row=1, col=1)

    # 2. Spread
    fig.add_trace(go.Scatter(x=df.index, y=df["spread"], name="Spread", line=dict(color='white', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["spread_mean"], name="Spread Mean", line=dict(color='gray', dash='dot')), row=2, col=1)

    # 3. Z-Score
    fig.add_trace(go.Scatter(x=df.index, y=df["z_score"], name="Z-Score", line=dict(color='yellow')), row=3, col=1)
    
    # Add Threshold Lines
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=3, col=1, annotation_text="Short Entry")
    fig.add_hline(y=-2.0, line_dash="dash", line_color="green", row=3, col=1, annotation_text="Long Entry")
    fig.add_hline(y=0, line_color="gray", row=3, col=1)

    # Highlight Active Positions on Z-Score
    # Green zones for Long Spread, Red zones for Short Spread
    # This is a bit complex to shade efficiently in plotly without many shapes, 
    # so we'll just add markers for entries
    
    # Identify entry points
    entries = df["position"].diff()
    long_entries = df[entries == 1].index
    short_entries = df[entries == -1].index
    exits = df[(entries != 0) & (df["position"] == 0)].index
    
    fig.add_trace(go.Scatter(x=long_entries, y=df.loc[long_entries, "z_score"], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name="Long Entry"), row=3, col=1)
    fig.add_trace(go.Scatter(x=short_entries, y=df.loc[short_entries, "z_score"], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name="Short Entry"), row=3, col=1)
    fig.add_trace(go.Scatter(x=exits, y=df.loc[exits, "z_score"], mode='markers', marker=dict(color='white', size=8, symbol='x'), name="Exit"), row=3, col=1)

    fig.update_layout(template="plotly_dark", title_text="Statistical Arbitrage: Pairs Trading Analysis", height=900)
    fig.show()

def main():
    parser = argparse.ArgumentParser(description="Crypto Pairs Trading (Statistical Arbitrage) Tool")
    parser.add_argument("--s1", default="BTC/USDT", help="First Symbol (e.g., BTC/USDT)")
    parser.add_argument("--s2", default="ETH/USDT", help="Second Symbol (e.g., ETH/USDT)")
    parser.add_argument("--timeframe", default="1h", help="Timeframe (e.g., 1h, 15m)")
    parser.add_argument("--limit", type=int, default=1000, help="Data limit")
    parser.add_argument("--window", type=int, default=20, help="Rolling window for Z-Score")
    
    args = parser.parse_args()
    
    print(color_text("="*60, Fore.MAGENTA, Style.BRIGHT))
    print(color_text("   CRYPTO PAIRS TRADING - STATISTICAL ARBITRAGE", Fore.MAGENTA, Style.BRIGHT))
    print(color_text("="*60, Fore.MAGENTA, Style.BRIGHT))
    
    # 1. Fetch Data
    df1 = fetch_data(args.s1, args.timeframe, args.limit)
    df2 = fetch_data(args.s2, args.timeframe, args.limit)
    
    if df1 is None or df2 is None:
        return

    # 2. Align Data
    print(color_text("Aligning data...", Fore.CYAN))
    df = align_data(df1, df2)
    print(color_text(f"Aligned Data Points: {len(df)}", Fore.GREEN))
    
    # 3. Calculate Correlation
    corr = df["asset1"].corr(df["asset2"])
    print(color_text(f"Correlation ({args.s1} vs {args.s2}): {corr:.4f}", Fore.YELLOW if corr < 0.8 else Fore.GREEN, Style.BRIGHT))
    
    if corr < 0.5:
        print(color_text("WARNING: Correlation is low. Pairs trading may not be effective.", Fore.RED))
        
    # 4. Calculate Spread & Z-Score
    print(color_text("Calculating Spread and Z-Scores...", Fore.CYAN))
    df = calculate_spread_and_zscore(df, window=args.window)
    
    # 5. Generate Signals
    df = generate_signals(df)
    
    # 6. Show recent status
    last_row = df.iloc[-1]
    print("\n" + color_text("CURRENT STATUS:", Fore.WHITE, Style.BRIGHT))
    print(f"Spread: {last_row['spread']:.6f}")
    print(f"Z-Score: {last_row['z_score']:.4f}")
    
    status_color = Fore.WHITE
    status_msg = "NEUTRAL (No Position)"
    if last_row['position'] == 1:
        status_msg = f"LONG SPREAD (Buy {args.s1}, Sell {args.s2})"
        status_color = Fore.GREEN
    elif last_row['position'] == -1:
        status_msg = f"SHORT SPREAD (Sell {args.s1}, Buy {args.s2})"
        status_color = Fore.RED
        
    print(f"Signal: {color_text(status_msg, status_color, Style.BRIGHT)}")
    
    # 7. Plot
    print(color_text("\nOpening interactive chart...", Fore.CYAN))
    plot_results(df, args.s1, args.s2)

if __name__ == "__main__":
    main()
