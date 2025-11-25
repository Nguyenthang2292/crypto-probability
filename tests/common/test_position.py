"""
Test script for modules.Position - Position dataclass.
"""

import pytest
from modules.Position import Position


def test_position_creation():
    """Test Position dataclass can be instantiated."""
    position = Position(
        symbol="BTC/USDT",
        direction="LONG",
        entry_price=50000.0,
        size_usdt=1000.0,
    )
    
    assert position.symbol == "BTC/USDT"
    assert position.direction == "LONG"
    assert position.entry_price == 50000.0
    assert position.size_usdt == 1000.0


def test_position_short():
    """Test Position with SHORT direction."""
    position = Position(
        symbol="ETH/USDT",
        direction="SHORT",
        entry_price=3000.0,
        size_usdt=500.0,
    )
    
    assert position.direction == "SHORT"
    assert position.entry_price == 3000.0


def test_position_equality():
    """Test Position equality comparison."""
    pos1 = Position("BTC/USDT", "LONG", 50000.0, 1000.0)
    pos2 = Position("BTC/USDT", "LONG", 50000.0, 1000.0)
    pos3 = Position("ETH/USDT", "LONG", 50000.0, 1000.0)
    
    assert pos1 == pos2
    assert pos1 != pos3


def test_position_repr():
    """Test Position string representation."""
    position = Position("BTC/USDT", "LONG", 50000.0, 1000.0)
    repr_str = repr(position)
    
    assert "BTC/USDT" in repr_str
    assert "LONG" in repr_str
    assert "50000.0" in repr_str


def test_position_str_representation():
    """Test Position string representation in different formats."""
    pos1 = Position("BTC/USDT", "LONG", 50000.0, 1000.0)
    pos2 = Position("ETH/USDT", "SHORT", 3000.0, 500.0)
    
    str1 = str(pos1)
    str2 = str(pos2)
    
    assert "BTC/USDT" in str1 or "LONG" in str1
    assert "ETH/USDT" in str2 or "SHORT" in str2


def test_position_zero_size():
    """Test Position with zero size."""
    position = Position("BTC/USDT", "LONG", 50000.0, 0.0)
    assert position.size_usdt == 0.0


def test_position_negative_price():
    """Test Position with negative entry price (edge case)."""
    # This might not be realistic but tests the dataclass
    position = Position("BTC/USDT", "SHORT", -100.0, 1000.0)
    assert position.entry_price == -100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

