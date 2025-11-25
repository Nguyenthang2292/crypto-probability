"""
Test script for DeepLearningDataPipeline
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
from colorama import Fore, Style, init as colorama_init

from modules.ExchangeManager import ExchangeManager
from modules.DataFetcher import DataFetcher
from modules.deeplearning_data_pipeline import DeepLearningDataPipeline
from modules.utils import color_text

# Suppress warnings
warnings.filterwarnings("ignore")
colorama_init(autoreset=True)


def main():
    print(color_text("\n=== Testing DeepLearningDataPipeline ===\n", Fore.CYAN, Style.BRIGHT))

    # Initialize ExchangeManager and DataFetcher
    print(color_text("Initializing ExchangeManager and DataFetcher...", Fore.YELLOW))
    exchange_manager = ExchangeManager()
    data_fetcher = DataFetcher(exchange_manager)

    # Initialize pipeline
    print(color_text("Initializing DeepLearningDataPipeline...", Fore.YELLOW))
    pipeline = DeepLearningDataPipeline(
        data_fetcher=data_fetcher,
        use_fractional_diff=True,
        use_triple_barrier=False,  # Set to True if you want triple barrier labeling
    )

    # Fetch and prepare data for BTC/USDT
    symbol = "BTC/USDT"
    print(color_text(f"\nFetching and preparing data for {symbol}...", Fore.CYAN, Style.BRIGHT))
    
    try:
        df = pipeline.fetch_and_prepare(
            symbols=[symbol],
            timeframe="1h",
            limit=500,  # Use smaller limit for testing
            check_freshness=False,
        )

        print(color_text(f"\n[OK] Data prepared successfully!", Fore.GREEN))
        print(color_text(f"Shape: {df.shape}", Fore.GREEN))
        print(color_text(f"Columns: {len(df.columns)}", Fore.GREEN))
        print(color_text(f"\nFirst few columns: {list(df.columns[:10])}", Fore.CYAN))
        print(color_text(f"\nData info:", Fore.CYAN))
        print(df.info())

        # Show sample data
        print(color_text(f"\n=== Sample Data (first 5 rows) ===", Fore.CYAN, Style.BRIGHT))
        print(df.head())

        # Show statistics
        print(color_text(f"\n=== Data Statistics ===", Fore.CYAN, Style.BRIGHT))
        print(df.describe())

        # Split data
        print(color_text(f"\n=== Splitting Data ===", Fore.CYAN, Style.BRIGHT))
        train_df, val_df, test_df = pipeline.split_chronological(df)

        print(color_text(f"\n[OK] Split completed!", Fore.GREEN))
        print(color_text(f"Train: {len(train_df)} rows", Fore.GREEN))
        print(color_text(f"Validation: {len(val_df)} rows", Fore.GREEN))
        print(color_text(f"Test: {len(test_df)} rows", Fore.GREEN))

        # Check for NaN values
        print(color_text(f"\n=== Checking for NaN values ===", Fore.CYAN, Style.BRIGHT))
        nan_counts = df.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            print(color_text(f"Columns with NaN values:", Fore.YELLOW))
            print(nan_cols)
        else:
            print(color_text("[OK] No NaN values found!", Fore.GREEN))

        print(color_text("\n[OK] Test completed successfully!\n", Fore.GREEN, Style.BRIGHT))

    except Exception as e:
        print(color_text(f"\n[ERROR] Error: {e}", Fore.RED, Style.BRIGHT))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

