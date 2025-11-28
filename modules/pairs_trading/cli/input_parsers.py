"""
Input parsing utilities for pairs trading CLI.

This module provides functions for parsing and validating user input including
weights, symbols, and other configuration parameters.
"""

from typing import Dict, Optional, Tuple

try:
    from modules.config import (
        PAIRS_TRADING_WEIGHTS,
        PAIRS_TRADING_WEIGHT_PRESETS,
    )
except ImportError:
    PAIRS_TRADING_WEIGHTS = {"1d": 0.5, "3d": 0.3, "1w": 0.2}
    PAIRS_TRADING_WEIGHT_PRESETS = {
        "momentum": {"1d": 0.5, "3d": 0.3, "1w": 0.2},
        "balanced": {"1d": 0.3, "3d": 0.4, "1w": 0.3},
    }

try:
    from modules.common.utils import (
        color_text,
        log_warn,
        log_error,
    )
except ImportError:
    def color_text(text, color=None, style=None):
        return text
    
    def log_warn(message: str) -> None:
        print(f"[WARN] {message}")
    
    def log_error(message: str) -> None:
        print(f"[ERROR] {message}")

from colorama import Fore


def standardize_symbol_input(symbol: str) -> str:
    """Convert raw user input into f'{base}/USDT' style if needed."""
    if not symbol:
        return ""
    cleaned = symbol.strip().upper()
    if "/" in cleaned:
        base, quote = cleaned.split("/", 1)
        base = base.strip()
        quote = quote.strip() or "USDT"
        return f"{base}/{quote}"
    if cleaned.endswith("USDT"):
        base = cleaned[:-4]
        base = base.strip()
        return f"{base}/USDT"
    return f"{cleaned}/USDT"


def parse_weights(weights_str: Optional[str], preset_key: Optional[str] = None) -> Dict[str, float]:
    """Parse weights string into dictionary.
    
    Args:
        weights_str: Weights in format '1d:0.5,3d:0.3,1w:0.2'
        preset_key: Named preset (momentum/balanced)
        
    Returns:
        Dictionary with weights, normalized to sum to 1.0
    """
    # Highest precedence: manual weights string
    if not weights_str and preset_key:
        preset = PAIRS_TRADING_WEIGHT_PRESETS.get(preset_key)
        if preset:
            return preset.copy()

    weights = PAIRS_TRADING_WEIGHTS.copy()
    if not weights_str:
        return weights
    
    try:
        weight_parts = weights_str.split(",")
        weights = {}
        for part in weight_parts:
            key, value = part.split(":")
            weights[key.strip()] = float(value.strip())
        # Validate weights sum to 1.0
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            log_warn(f"Weights sum to {total:.3f}, not 1.0. Normalizing...")
            weights = {k: v / total for k, v in weights.items()}
    except Exception as e:
        log_error(f"Error parsing weights: {e}. Using default weights.")
        weights = PAIRS_TRADING_WEIGHTS.copy()
    
    return weights


def parse_symbols(symbols_str: Optional[str]) -> Tuple[list, list]:
    """Parse symbols string into display and parsed lists.
    
    Args:
        symbols_str: Comma/space separated symbols
        
    Returns:
        Tuple of (target_symbol_inputs, parsed_target_symbols)
    """
    target_symbol_inputs = []
    parsed_target_symbols = []
    
    if not symbols_str:
        return target_symbol_inputs, parsed_target_symbols
    
    raw_parts = (
        symbols_str.replace(",", " ")
        .replace(";", " ")
        .replace("|", " ")
        .split()
    )
    seen_display = set()
    seen_parsed = set()
    for part in raw_parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        display_value = cleaned.upper()
        parsed_value = standardize_symbol_input(cleaned)
        if display_value not in seen_display:
            seen_display.add(display_value)
            target_symbol_inputs.append(display_value)
        parsed_key = parsed_value.upper()
        if parsed_key not in seen_parsed:
            seen_parsed.add(parsed_key)
            parsed_target_symbols.append(parsed_value)
    
    return target_symbol_inputs, parsed_target_symbols

