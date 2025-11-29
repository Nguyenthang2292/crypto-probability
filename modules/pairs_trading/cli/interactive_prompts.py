"""
Interactive prompts for pairs trading CLI.

This module provides interactive user input prompts for selecting presets,
configuring parameters, and choosing analysis modes in the pairs trading CLI.
"""

import sys
from typing import Dict, Optional, Tuple

from colorama import Fore, Style

try:
    from modules.common.utils import (
        color_text,
        log_info,
        log_success,
        log_error,
        log_warn,
        log_data,
    )
except ImportError:
    def color_text(text, color=None, style=None):
        return text
    
    def log_info(message: str) -> None:
        print(f"[INFO] {message}")
    
    def log_success(message: str) -> None:
        print(f"[SUCCESS] {message}")
    
    def log_error(message: str) -> None:
        print(f"[ERROR] {message}")
    
    def log_warn(message: str) -> None:
        print(f"[WARN] {message}")
    
    def log_data(message: str) -> None:
        print(f"[DATA] {message}")

try:
    from modules.config import (
        PAIRS_TRADING_WEIGHT_PRESETS,
        PAIRS_TRADING_KALMAN_PRESETS,
        PAIRS_TRADING_OPPORTUNITY_PRESETS,
    )
except ImportError:
    PAIRS_TRADING_WEIGHT_PRESETS = {
        "momentum": {"1d": 0.5, "3d": 0.3, "1w": 0.2},
        "balanced": {"1d": 0.3, "3d": 0.4, "1w": 0.3},
    }
    PAIRS_TRADING_KALMAN_PRESETS = {
        "balanced": {"delta": 1e-5, "obs_cov": 1.0, "description": "Default balanced profile"},
    }
    PAIRS_TRADING_OPPORTUNITY_PRESETS = {
        "balanced": {
            "description": "Default balanced scoring",
        }
    }


def prompt_interactive_mode() -> Dict[str, Optional[str]]:
    """Interactive launcher for selecting analysis mode and symbol source."""
    log_data("=" * 60)
    log_info("Pairs Trading Analysis - Interactive Launcher")
    log_data("=" * 60)
    print(
        color_text(
            "1) Auto mode  - analyze entire market to surface opportunities",
            Fore.MAGENTA,
            Style.BRIGHT,
        )
    )
    print("2) Manual mode - focus on specific symbols you provide")
    print("3) Exit")

    while True:
        choice = input(color_text("\nSelect option [1-3] (default 1): ", Fore.YELLOW)).strip() or "1"
        if choice in {"1", "2", "3"}:
            break
        log_error("Invalid selection. Please enter 1, 2, or 3.")

    if choice == "3":
        log_warn("Exiting...")
        sys.exit(0)

    manual_symbols = None
    if choice == "2":
        manual_symbols = input(
            color_text(
                "Enter symbols separated by comma/space (e.g., BTC/USDT, ETH/USDT): ",
                Fore.YELLOW,
            )
        ).strip()

    return {
        "mode": "manual" if choice == "2" else "auto",
        "symbols_raw": manual_symbols or None,
    }


def prompt_weight_preset_selection(current_preset: Optional[str]) -> str:
    """Interactive selection menu for weight presets."""
    presets = list(PAIRS_TRADING_WEIGHT_PRESETS.items())
    if not presets:
        return current_preset or "momentum"

    default_choice = None
    for idx, (key, _) in enumerate(presets, start=1):
        if key == current_preset:
            default_choice = str(idx)
            break
    if default_choice is None:
        default_choice = "1"

    log_info("Select weight preset for calculating performance score for pairs trading:")
    for idx, (key, weights) in enumerate(presets, start=1):
        weights_desc = f"1d={weights['1d']:.2f}, 3d={weights['3d']:.2f}, 1w={weights['1w']:.2f}"
        highlight = Style.BRIGHT if key == current_preset else Style.NORMAL
        print(
            color_text(
                f"{idx}) {key.capitalize()} ({weights_desc})",
                Fore.MAGENTA if key == current_preset else Fore.WHITE,
                highlight,
            )
        )

    choice_map = {str(idx + 1): key for idx, (key, _) in enumerate(presets)}

    while True:
        user_choice = input(
            color_text(
                f"\nEnter preset [1-{len(presets)}] (default {default_choice}): ",
                Fore.YELLOW,
            )
        ).strip() or default_choice
        if user_choice in choice_map:
            selected = choice_map[user_choice]
            log_success(f"Using {selected.capitalize()} preset")
            return selected
        log_error("Invalid selection. Please try again.")


def prompt_kalman_preset_selection(
    current_delta: float,
    current_obs_cov: float,
) -> Tuple[float, float, Optional[str]]:
    """Interactive selection menu for Kalman parameter presets."""
    presets = list(PAIRS_TRADING_KALMAN_PRESETS.items())
    if not presets:
        return current_delta, current_obs_cov, None

    default_choice = "2"
    log_info("Select Kalman filter profile for hedge ratio:")
    for idx, (key, data) in enumerate(presets, start=1):
        desc = data.get("description", "")
        delta = data.get("delta")
        obs_cov = data.get("obs_cov")
        is_default = str(idx) == default_choice
        print(
            color_text(
                f"{idx}) {key} (delta={delta:.2e}, obs_cov={obs_cov:.2f}) - {desc}",
                Fore.MAGENTA if is_default else Fore.WHITE,
                Style.BRIGHT if is_default else Style.NORMAL,
            )
        )
    choice_map = {str(idx): (key, data) for idx, (key, data) in enumerate(presets, start=1)}
    while True:
        user_choice = input(
            color_text(
                f"\nEnter preset [1-{len(presets)}] (default {default_choice}): ",
                Fore.YELLOW,
            )
        ).strip() or default_choice
        if user_choice in choice_map:
            key, data = choice_map[user_choice]
            delta = float(data.get("delta", current_delta))
            obs_cov = float(data.get("obs_cov", current_obs_cov))
            log_success(f"Using {key} profile (delta={delta:.2e}, obs_cov={obs_cov:.2f})")
            return delta, obs_cov, key
        log_error("Invalid selection. Please try again.")


def prompt_opportunity_preset_selection(
    current_key: Optional[str],
) -> str:
    """Interactive selection for opportunity scoring profiles."""
    presets = list(PAIRS_TRADING_OPPORTUNITY_PRESETS.items())
    if not presets:
        return current_key or "balanced"

    default_choice = None
    for idx, (key, _) in enumerate(presets, start=1):
        if key == current_key:
            default_choice = str(idx)
            break
    if default_choice is None:
        default_choice = "1"

    log_info("Select opportunity scoring profile:")
    for idx, (key, data) in enumerate(presets, start=1):
        desc = data.get("description", "")
        print(
            color_text(
                f"{idx}) {key} - {desc}",
                Fore.WHITE if key != current_key else Fore.MAGENTA,
                Style.BRIGHT if key == current_key else Style.NORMAL,
            )
        )

    choice_map = {str(idx): key for idx, (key, _) in enumerate(presets, start=1)}
    while True:
        selection = input(
            color_text(
                f"\nEnter preset [1-{len(presets)}] (default {default_choice}): ",
                Fore.YELLOW,
            )
        ).strip() or default_choice
        if selection in choice_map:
            chosen = choice_map[selection]
            log_success(f"Using {chosen} scoring profile")
            return chosen
        log_error("Invalid selection. Please try again.")


def prompt_target_pairs(default_count: int) -> int:
    """Interactive prompt for number of target pairs to return.
    
    Args:
        default_count: Default number of pairs
        
    Returns:
        User-selected number of pairs
    """
    while True:
        user_input = input(
            color_text(
                f"\nEnter number of pairs to return (default: {default_count}): ",
                Fore.YELLOW,
            )
        ).strip()
        
        if not user_input:
            return default_count
        
        try:
            count = int(user_input)
            if count > 0:
                return count
            log_error("Please enter a positive number.")
        except ValueError:
            log_error("Invalid input. Please enter a number.")


def prompt_candidate_depth(default_depth: int) -> int:
    """Interactive prompt for candidate depth (number of top/bottom symbols to consider).
    
    Args:
        default_depth: Default candidate depth
        
    Returns:
        User-selected candidate depth
    """
    while True:
        user_input = input(
            color_text(
                f"\nEnter candidate depth - number of top/bottom symbols to consider (default: {default_depth}): ",
                Fore.YELLOW,
            )
        ).strip()
        
        if not user_input:
            return default_depth
        
        try:
            depth = int(user_input)
            if depth > 0:
                return depth
            log_error("Please enter a positive number.")
        except ValueError:
            log_error("Invalid input. Please enter a number.")

