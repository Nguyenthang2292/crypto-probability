"""Configuration for Adaptive Trend Classification (ATC) analysis."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ATCConfig:
    """Configuration for Adaptive Trend Classification (ATC) analysis."""
    # Moving Average lengths
    ema_len: int = 28
    hma_len: int = 28
    wma_len: int = 28
    dema_len: int = 28
    lsma_len: int = 28
    kama_len: int = 28
    
    # ATC parameters
    robustness: str = "Medium"  # "Narrow", "Medium", or "Wide"
    lambda_param: float = 0.02
    decay: float = 0.03
    cutout: int = 0
    
    # Data parameters
    limit: int = 1500
    timeframe: str = "15m"


def create_atc_config_from_dict(
    params: Dict[str, Any],
    timeframe: str = "15m",
) -> ATCConfig:
    """
    Create ATCConfig from a dictionary of parameters.
    
    Args:
        params: Dictionary containing ATC parameters
        timeframe: Timeframe for data (default: "15m")
    
    Returns:
        ATCConfig instance with parameters from dict
    """
    return ATCConfig(
        timeframe=timeframe,
        limit=params.get("limit", 1500),
        ema_len=params.get("ema_len", 28),
        hma_len=params.get("hma_len", 28),
        wma_len=params.get("wma_len", 28),
        dema_len=params.get("dema_len", 28),
        lsma_len=params.get("lsma_len", 28),
        kama_len=params.get("kama_len", 28),
        robustness=params.get("robustness", "Medium"),
        lambda_param=params.get("lambda_param", 0.02),
        decay=params.get("decay", 0.03),
        cutout=params.get("cutout", 0),
    )

