"""
Candidate pool management utilities for pairs trading analysis.

This module provides functions for managing candidate pools of best and worst
performing symbols, ensuring target symbols are included in appropriate pools.
"""

import pandas as pd
from typing import List, Tuple


def ensure_symbols_in_candidate_pools(
    performance_df: pd.DataFrame,
    best_df: pd.DataFrame,
    worst_df: pd.DataFrame,
    target_symbols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure target symbols are present in candidate pools based on their score direction.
    
    Adds target symbols to appropriate pools: positive scores -> best_df, negative scores -> worst_df.
    Symbols already in pools are not duplicated. Missing symbols are silently skipped.
    
    Args:
        performance_df: DataFrame with 'symbol' and 'score' columns (all performance data)
        best_df: DataFrame of best performers (high scores)
        worst_df: DataFrame of worst performers (low scores)
        target_symbols: List of symbols to ensure are in pools
        
    Returns:
        Tuple of (updated_best_df, updated_worst_df), both sorted by score (best descending,
        worst ascending) with reset indices.
    """
    if not target_symbols:
        return best_df, worst_df

    # Track which symbols are already in each pool
    best_symbols = set(best_df["symbol"].tolist())
    worst_symbols = set(worst_df["symbol"].tolist())

    # Add each target symbol to appropriate pool
    for symbol in target_symbols:
        # Find symbol in performance data
        row = performance_df[performance_df["symbol"] == symbol]
        if row.empty:
            # Symbol not found in performance data, skip
            continue
            
        score = row.iloc[0]["score"]
        
        # Add to best pool if score >= 0
        if score >= 0:
            if symbol not in best_symbols:
                # Avoid FutureWarning: check if best_df is empty before concat
                if best_df.empty:
                    best_df = row.copy()
                else:
                    best_df = pd.concat([best_df, row], ignore_index=True)
                best_symbols.add(symbol)
        # Add to worst pool if score < 0
        else:
            if symbol not in worst_symbols:
                # Avoid FutureWarning: check if worst_df is empty before concat
                if worst_df.empty:
                    worst_df = row.copy()
                else:
                    worst_df = pd.concat([worst_df, row], ignore_index=True)
                worst_symbols.add(symbol)

    # Re-sort pools and reset indices
    best_df = (
        best_df.sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    worst_df = (
        worst_df.sort_values("score", ascending=True)
        .reset_index(drop=True)
    )
    
    return best_df, worst_df

