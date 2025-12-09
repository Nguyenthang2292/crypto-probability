"""
Tests for utils/config module.
"""
import pytest

from modules.adaptive_trend.utils.config import ATCConfig, create_atc_config_from_dict


def test_atc_config_defaults():
    """Test that ATCConfig has correct default values."""
    config = ATCConfig()
    
    assert config.ema_len == 28
    assert config.hma_len == 28
    assert config.wma_len == 28
    assert config.dema_len == 28
    assert config.lsma_len == 28
    assert config.kama_len == 28
    assert config.robustness == "Medium"
    assert config.lambda_param == 0.02
    assert config.decay == 0.03
    assert config.cutout == 0
    assert config.limit == 1500
    assert config.timeframe == "15m"


def test_atc_config_custom_values():
    """Test that ATCConfig accepts custom values."""
    config = ATCConfig(
        ema_len=14,
        hma_len=21,
        wma_len=28,
        dema_len=35,
        lsma_len=42,
        kama_len=50,
        robustness="Narrow",
        lambda_param=0.01,
        decay=0.05,
        cutout=10,
        limit=2000,
        timeframe="1h",
    )
    
    assert config.ema_len == 14
    assert config.hma_len == 21
    assert config.wma_len == 28
    assert config.dema_len == 35
    assert config.lsma_len == 42
    assert config.kama_len == 50
    assert config.robustness == "Narrow"
    assert config.lambda_param == 0.01
    assert config.decay == 0.05
    assert config.cutout == 10
    assert config.limit == 2000
    assert config.timeframe == "1h"


def test_atc_config_robustness_values():
    """Test that ATCConfig accepts different robustness values."""
    config_narrow = ATCConfig(robustness="Narrow")
    config_medium = ATCConfig(robustness="Medium")
    config_wide = ATCConfig(robustness="Wide")
    
    assert config_narrow.robustness == "Narrow"
    assert config_medium.robustness == "Medium"
    assert config_wide.robustness == "Wide"


def test_create_atc_config_from_dict_full():
    """Test create_atc_config_from_dict with all parameters."""
    params = {
        "limit": 2000,
        "ema_len": 14,
        "hma_len": 21,
        "wma_len": 28,
        "dema_len": 35,
        "lsma_len": 42,
        "kama_len": 50,
        "robustness": "Wide",
        "lambda_param": 0.01,
        "decay": 0.05,
        "cutout": 10,
    }
    timeframe = "4h"
    
    config = create_atc_config_from_dict(params, timeframe=timeframe)
    
    assert isinstance(config, ATCConfig)
    assert config.timeframe == "4h"
    assert config.limit == 2000
    assert config.ema_len == 14
    assert config.hma_len == 21
    assert config.wma_len == 28
    assert config.dema_len == 35
    assert config.lsma_len == 42
    assert config.kama_len == 50
    assert config.robustness == "Wide"
    assert config.lambda_param == 0.01
    assert config.decay == 0.05
    assert config.cutout == 10


def test_create_atc_config_from_dict_partial():
    """Test create_atc_config_from_dict with partial parameters."""
    params = {
        "ema_len": 14,
        "robustness": "Narrow",
    }
    
    config = create_atc_config_from_dict(params)
    
    assert isinstance(config, ATCConfig)
    assert config.ema_len == 14
    assert config.robustness == "Narrow"
    # Should use defaults for missing parameters
    assert config.hma_len == 28
    assert config.limit == 1500
    assert config.timeframe == "15m"  # Default timeframe
    assert config.lambda_param == 0.02
    assert config.decay == 0.03


def test_create_atc_config_from_dict_empty():
    """Test create_atc_config_from_dict with empty dict."""
    params = {}
    
    config = create_atc_config_from_dict(params)
    
    assert isinstance(config, ATCConfig)
    # Should use all defaults
    assert config.ema_len == 28
    assert config.robustness == "Medium"
    assert config.limit == 1500
    assert config.timeframe == "15m"


def test_create_atc_config_from_dict_custom_timeframe():
    """Test create_atc_config_from_dict with custom timeframe."""
    params = {"ema_len": 14}
    
    config = create_atc_config_from_dict(params, timeframe="1d")
    
    assert config.timeframe == "1d"
    assert config.ema_len == 14


def test_create_atc_config_from_dict_default_timeframe():
    """Test create_atc_config_from_dict uses default timeframe when not specified."""
    params = {"ema_len": 14}
    
    config = create_atc_config_from_dict(params)
    
    assert config.timeframe == "15m"  # Default


def test_atc_config_immutability():
    """Test that ATCConfig is a dataclass (can be modified but structure is fixed)."""
    config = ATCConfig()
    
    # Dataclass allows attribute modification
    config.ema_len = 14
    assert config.ema_len == 14
    
    # But structure is defined
    assert hasattr(config, "ema_len")
    assert hasattr(config, "robustness")
    assert hasattr(config, "lambda_param")


def test_atc_config_all_parameters():
    """Test that ATCConfig has all expected parameters."""
    config = ATCConfig()
    
    expected_params = [
        "ema_len", "hma_len", "wma_len", "dema_len", "lsma_len", "kama_len",
        "robustness", "lambda_param", "decay", "cutout",
        "limit", "timeframe",
    ]
    
    for param in expected_params:
        assert hasattr(config, param), f"Missing parameter: {param}"

