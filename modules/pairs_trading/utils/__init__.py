"""
Utility functions for pairs trading analysis.

This package provides business logic utilities for managing and manipulating
trading pairs data, including selection, transformation, and candidate pool management.
"""

from modules.pairs_trading.utils.pairs_selector import (
    select_top_unique_pairs,
    select_pairs_for_symbols,
)
from modules.pairs_trading.utils.ensure_symbols_in_pools import (
    ensure_symbols_in_candidate_pools,
)
from modules.pairs_trading.utils.pairs_validator import (
    validate_pairs,
)

__all__ = [
    # Pair selection
    'select_top_unique_pairs',
    'select_pairs_for_symbols',
    # Candidate pool management
    'ensure_symbols_in_candidate_pools',
    # Pair validation
    'validate_pairs',
]
