from modules.utils import format_price, normalize_symbol, timeframe_to_minutes


def test_normalize_symbol_defaults_to_btc_when_empty():
    assert normalize_symbol("") == "BTC/USDT"
    assert normalize_symbol(None) == "BTC/USDT"


def test_normalize_symbol_preserves_existing_pairs():
    assert normalize_symbol("eth/btc") == "ETH/BTC"
    assert normalize_symbol("ada/usdt") == "ADA/USDT"


def test_normalize_symbol_adds_quote_suffix():
    assert normalize_symbol("adausdt") == "ADA/USDT"
    assert normalize_symbol("sol", quote="BUSD") == "SOL/BUSD"


def test_timeframe_to_minutes_handles_units():
    assert timeframe_to_minutes("15m") == 15
    assert timeframe_to_minutes("2h") == 120
    assert timeframe_to_minutes("1d") == 1440
    assert timeframe_to_minutes("2w") == 20160
    assert timeframe_to_minutes("invalid") == 60


def test_format_price_adapts_precision():
    assert format_price(123.456) == "123.46"
    assert format_price(0.1234) == "0.1234"
    assert format_price(0.00001234) == "0.00001234"
    assert format_price(None) == "N/A"
