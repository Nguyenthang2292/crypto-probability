"""
Tests for range_oscillator strategy module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.range_oscillator.strategy import (
    generate_signals_strategy1,
    generate_signals_strategy2_sustained,
    generate_signals_strategy3_crossover,
    generate_signals_strategy4_momentum,
    generate_signals_strategy5_combined,
    generate_signals_strategy6_breakout,
    generate_signals_strategy7_divergence,
    generate_signals_strategy8_trend_following,
    generate_signals_strategy9_mean_reversion,
    get_signal_summary,
)


@pytest.fixture
def sample_ohlc_data():
    """Create sample OHLC data for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2024-01-01', periods=n, freq='1H')
    
    # Generate realistic price data
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    
    return pd.Series(high, index=dates), pd.Series(low, index=dates), pd.Series(close, index=dates)


@pytest.fixture
def sample_oscillator_data(sample_ohlc_data):
    """Create sample oscillator data for testing."""
    high, low, close = sample_ohlc_data
    
    # Simple mock oscillator values
    oscillator = pd.Series(np.sin(np.linspace(0, 4*np.pi, len(close))) * 50, index=close.index)
    ma = close.rolling(50).mean()
    range_atr = pd.Series(np.ones(len(close)) * 1000, index=close.index)
    
    return oscillator, ma, range_atr


class TestStrategy1:
    """Tests for Strategy 1: Basic oscillator signals."""
    
    def test_strategy1_basic(self, sample_ohlc_data):
        """Test basic Strategy 1 functionality."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_strategy1(
            high=high, low=low, close=close,
            length=50, mult=2.0
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)
        assert len(strength) == len(close)
        assert signals.dtype == 'int8'
        assert all(signals.isin([-1, 0, 1]))
        assert all((strength >= 0) & (strength <= 1))
    
    def test_strategy1_with_precalculated(self, sample_ohlc_data, sample_oscillator_data):
        """Test Strategy 1 with pre-calculated oscillator values."""
        high, low, close = sample_ohlc_data
        oscillator, ma, range_atr = sample_oscillator_data
        
        signals, strength = generate_signals_strategy1(
            oscillator=oscillator, ma=ma, range_atr=range_atr
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(oscillator)
    
    def test_strategy1_parameters(self, sample_ohlc_data):
        """Test Strategy 1 with different parameters."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_strategy1(
            high=high, low=low, close=close,
            oscillator_threshold=10.0,
            require_trend_confirmation=False,
            use_breakout_signals=False
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)


class TestStrategy2:
    """Tests for Strategy 2: Sustained pressure."""
    
    def test_strategy2_basic(self, sample_ohlc_data):
        """Test basic Strategy 2 functionality."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_strategy2_sustained(
            high=high, low=low, close=close,
            min_bars_above_zero=3,
            min_bars_below_zero=3
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy2_with_precalculated(self, sample_ohlc_data, sample_oscillator_data):
        """Test Strategy 2 with pre-calculated values."""
        high, low, close = sample_ohlc_data
        oscillator, ma, range_atr = sample_oscillator_data
        
        signals, strength = generate_signals_strategy2_sustained(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            min_bars_above_zero=5
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)


class TestStrategy3:
    """Tests for Strategy 3: Zero line crossover."""
    
    def test_strategy3_basic(self, sample_ohlc_data):
        """Test basic Strategy 3 functionality."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_strategy3_crossover(
            high=high, low=low, close=close,
            confirmation_bars=2
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy3_with_precalculated(self, sample_ohlc_data, sample_oscillator_data):
        """Test Strategy 3 with pre-calculated values."""
        high, low, close = sample_ohlc_data
        oscillator, ma, range_atr = sample_oscillator_data
        
        signals, strength = generate_signals_strategy3_crossover(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            confirmation_bars=3
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)


class TestStrategy4:
    """Tests for Strategy 4: Momentum."""
    
    def test_strategy4_basic(self, sample_ohlc_data):
        """Test basic Strategy 4 functionality."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_strategy4_momentum(
            high=high, low=low, close=close,
            momentum_period=3,
            momentum_threshold=5.0
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy4_with_precalculated(self, sample_ohlc_data, sample_oscillator_data):
        """Test Strategy 4 with pre-calculated values."""
        high, low, close = sample_ohlc_data
        oscillator, ma, range_atr = sample_oscillator_data
        
        signals, strength = generate_signals_strategy4_momentum(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            momentum_period=5
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)


class TestStrategy5:
    """Tests for Strategy 5: Combined."""
    
    def test_strategy5_basic(self, sample_ohlc_data):
        """Test basic Strategy 5 functionality."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_strategy5_combined(
            high=high, low=low, close=close,
            use_sustained=True,
            use_crossover=True,
            use_momentum=True
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy5_with_precalculated(self, sample_ohlc_data, sample_oscillator_data):
        """Test Strategy 5 with pre-calculated values."""
        high, low, close = sample_ohlc_data
        oscillator, ma, range_atr = sample_oscillator_data
        
        signals, strength = generate_signals_strategy5_combined(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            use_sustained=True,
            use_crossover=False,
            use_momentum=False
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
    
    def test_strategy5_no_methods_enabled(self, sample_ohlc_data):
        """Test Strategy 5 fallback when no methods enabled."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_strategy5_combined(
            high=high, low=low, close=close,
            use_sustained=False,
            use_crossover=False,
            use_momentum=False
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)


class TestStrategy6:
    """Tests for Strategy 6: Range breakouts."""
    
    def test_strategy6_basic(self, sample_ohlc_data):
        """Test basic Strategy 6 functionality."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_strategy6_breakout(
            high=high, low=low, close=close,
            upper_threshold=100.0,
            lower_threshold=-100.0,
            require_confirmation=True
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy6_with_precalculated(self, sample_ohlc_data, sample_oscillator_data):
        """Test Strategy 6 with pre-calculated values."""
        high, low, close = sample_ohlc_data
        oscillator, ma, range_atr = sample_oscillator_data
        
        signals, strength = generate_signals_strategy6_breakout(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            upper_threshold=80.0,
            detect_exhaustion=False
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)


class TestStrategy7:
    """Tests for Strategy 7: Divergence detection."""
    
    def test_strategy7_basic(self, sample_ohlc_data):
        """Test basic Strategy 7 functionality."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_strategy7_divergence(
            high=high, low=low, close=close,
            lookback_period=30,
            min_swing_bars=5
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy7_requires_price_data(self, sample_oscillator_data):
        """Test that Strategy 7 requires price data."""
        oscillator, ma, range_atr = sample_oscillator_data
        
        with pytest.raises(ValueError, match="high, low, close are required"):
            generate_signals_strategy7_divergence(
                oscillator=oscillator, ma=ma, range_atr=range_atr
            )


class TestStrategy8:
    """Tests for Strategy 8: Trend following."""
    
    def test_strategy8_basic(self, sample_ohlc_data):
        """Test basic Strategy 8 functionality."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_strategy8_trend_following(
            high=high, low=low, close=close,
            trend_filter_period=10,
            oscillator_threshold=20.0
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy8_with_precalculated(self, sample_ohlc_data, sample_oscillator_data):
        """Test Strategy 8 with pre-calculated values."""
        high, low, close = sample_ohlc_data
        oscillator, ma, range_atr = sample_oscillator_data
        
        signals, strength = generate_signals_strategy8_trend_following(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            require_consistency=False
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)


class TestStrategy9:
    """Tests for Strategy 9: Mean reversion."""
    
    def test_strategy9_basic(self, sample_ohlc_data):
        """Test basic Strategy 9 functionality."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_strategy9_mean_reversion(
            high=high, low=low, close=close,
            extreme_threshold=80.0,
            zero_cross_threshold=10.0
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy9_with_precalculated(self, sample_ohlc_data, sample_oscillator_data):
        """Test Strategy 9 with pre-calculated values."""
        high, low, close = sample_ohlc_data
        oscillator, ma, range_atr = sample_oscillator_data
        
        signals, strength = generate_signals_strategy9_mean_reversion(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            min_extreme_bars=3
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)


class TestSignalSummary:
    """Tests for get_signal_summary function."""
    
    def test_get_signal_summary_basic(self, sample_ohlc_data):
        """Test basic signal summary functionality."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_strategy1(
            high=high, low=low, close=close
        )
        
        summary = get_signal_summary(signals, strength, close)
        
        assert isinstance(summary, dict)
        assert "total_signals" in summary
        assert "long_signals" in summary
        assert "short_signals" in summary
        assert "neutral_signals" in summary
        assert "avg_signal_strength" in summary
        assert "current_signal" in summary
        assert "current_strength" in summary
        assert summary["total_signals"] == len(signals)
        assert summary["long_signals"] + summary["short_signals"] + summary["neutral_signals"] == len(signals)
    
    def test_get_signal_summary_empty(self):
        """Test signal summary with empty signals."""
        empty_signals = pd.Series([], dtype='int8')
        empty_strength = pd.Series([], dtype='float64')
        empty_close = pd.Series([], dtype='float64')
        
        summary = get_signal_summary(empty_signals, empty_strength, empty_close)
        
        assert summary["total_signals"] == 0
        assert summary["long_signals"] == 0
        assert summary["short_signals"] == 0
        assert summary["current_signal"] == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_all_strategies_with_short_data(self):
        """Test all strategies with very short data."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1H')
        high = pd.Series([50000] * 10, index=dates)
        low = pd.Series([49000] * 10, index=dates)
        close = pd.Series([49500] * 10, index=dates)
        
        # Should not raise errors, but may return mostly zeros
        signals, strength = generate_signals_strategy1(
            high=high, low=low, close=close, length=5
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
    
    def test_strategies_with_nan_values(self, sample_ohlc_data):
        """Test strategies handle NaN values gracefully."""
        high, low, close = sample_ohlc_data
        
        # Add some NaN values
        close_with_nan = close.copy()
        close_with_nan.iloc[10:15] = np.nan
        
        signals, strength = generate_signals_strategy1(
            high=high, low=low, close=close_with_nan
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        # Signals should be 0 where data is NaN
        assert all(signals.iloc[10:15] == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

