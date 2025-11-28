"""
Pair selection utilities for pairs trading analysis.

This module provides functions for selecting and filtering trading pairs
based on various criteria such as uniqueness, target symbols, and scores.
"""

import pandas as pd
from typing import Optional, List


def select_top_unique_pairs(pairs_df: pd.DataFrame, target_pairs: int) -> pd.DataFrame:
    """
    Select up to target_pairs trading pairs, prioritizing unique symbols for diversification.
    
    Uses a two-pass strategy: (1) Select pairs with completely unique symbols (non-overlapping),
    (2) Fill remaining slots with any available pairs. Assumes pairs_df is pre-sorted by score.
    
    Args:
        pairs_df: DataFrame with 'long_symbol' and 'short_symbol' columns. Should be sorted
            by desirability (e.g., opportunity_score) as pairs are selected in order.
        target_pairs: Maximum number of pairs to select. May return fewer if insufficient
            unique pairs are available.
        
    Returns:
        DataFrame with selected pairs (all original columns preserved, index reset).
        Returns original pairs_df if empty/None, or top N pairs if no unique selections found.
    """
    if pairs_df is None or pairs_df.empty:
        return pairs_df

    selected_indices = []
    used_symbols = set()

    # First pass: Select pairs with completely unique symbols
    for idx, row in pairs_df.iterrows():
        long_symbol = row["long_symbol"]
        short_symbol = row["short_symbol"]
        if long_symbol in used_symbols or short_symbol in used_symbols:
            continue
        selected_indices.append(idx)
        used_symbols.update([long_symbol, short_symbol])
        if len(selected_indices) == target_pairs:
            break

    # Second pass: Fill remaining slots if needed
    if len(selected_indices) < target_pairs:
        for idx in pairs_df.index:
            if idx in selected_indices:
                continue
            selected_indices.append(idx)
            if len(selected_indices) == target_pairs:
                break

    # Fallback: If still no selections, just take top N
    if not selected_indices:
        return pairs_df.head(target_pairs).reset_index(drop=True)

    return pairs_df.loc[selected_indices].reset_index(drop=True)


def select_pairs_for_symbols(
    pairs_df: pd.DataFrame, 
    target_symbols: List[str], 
    max_pairs: Optional[int] = None
) -> pd.DataFrame:
    """
    Select the best pair (highest score) for each requested symbol.
    
    This function finds the best pair opportunity for each symbol in target_symbols,
    where the symbol appears as either the long or short side of the pair.
    
    Args:
        pairs_df: DataFrame containing pairs data with 'long_symbol' and 'short_symbol' columns
        target_symbols: List of symbols to find pairs for
        max_pairs: Maximum number of pairs to return (None = no limit)
        
    Returns:
        DataFrame with selected pairs for the requested symbols
        Returns empty DataFrame if no matches found
        
    Example:
        >>> pairs = pd.DataFrame({
        ...     'long_symbol': ['BTC/USDT', 'ETH/USDT'],
        ...     'short_symbol': ['ETH/USDT', 'BNB/USDT'],
        ...     'score': [0.9, 0.8]
        ... })
        >>> selected = select_pairs_for_symbols(pairs, ['BTC/USDT'])
        >>> len(selected)
        1
    """
    if pairs_df is None or pairs_df.empty or not target_symbols:
        return pd.DataFrame(columns=pairs_df.columns if pairs_df is not None else [])

    selected_rows = []
    for symbol in target_symbols:
        # Find pairs where symbol appears on either side
        matches = pairs_df[
            (pairs_df["long_symbol"] == symbol) | (pairs_df["short_symbol"] == symbol)
        ]
        if matches.empty:
            continue
        # Take the best match (first row, assuming sorted by score)
        selected_rows.append(matches.iloc[0])
        if max_pairs is not None and len(selected_rows) >= max_pairs:
            break

    if not selected_rows:
        return pd.DataFrame(columns=pairs_df.columns)

    return pd.DataFrame(selected_rows).reset_index(drop=True)

