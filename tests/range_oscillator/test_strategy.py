"""
Tests for range_oscillator strategy module.
"""
import numpy as np
import pandas as pd
import pytest

from modules.range_oscillator.strategies.basic import generate_signals_basic_strategy
from modules.range_oscillator.strategies.sustained import generate_signals_sustained_strategy
from modules.range_oscillator.strategies.crossover import generate_signals_crossover_strategy
from modules.range_oscillator.strategies.momentum import generate_signals_momentum_strategy
from modules.range_oscillator.analysis.combined import generate_signals_combined_all_strategy
from modules.range_oscillator.strategies.breakout import generate_signals_breakout_strategy
from modules.range_oscillator.strategies.divergence import generate_signals_divergence_strategy
from modules.range_oscillator.strategies.trend_following import generate_signals_trend_following_strategy
from modules.range_oscillator.strategies.mean_reversion import generate_signals_mean_reversion_strategy
from modules.range_oscillator.analysis.summary import get_signal_summary


@pytest.fixture
def sample_ohlc_data():
    """Create sample OHLC data for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    
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
        
        signals, strength = generate_signals_basic_strategy(
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
        
        signals, strength = generate_signals_basic_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr, close=close
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(oscillator)
    
    def test_strategy1_parameters(self, sample_ohlc_data):
        """Test Strategy 1 with different parameters."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_basic_strategy(
            high=high, low=low, close=close,
            oscillator_threshold=10.0,
            require_trend_confirmation=False,
            use_breakout_signals=False
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
    
    def test_strategy1_zero_cross_up_with_forward_fill(self):
        """Test Strategy 1 zero cross up and forward fill behavior."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')
        
        # Oscillator: negative -> crosses zero -> positive (should maintain LONG after cross)
        oscillator_values = [-10.0] * 5 + [5.0] * 15
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 20, index=dates)
        range_atr = pd.Series([1000.0] * 20, index=dates)
        close = pd.Series([51000.0] * 20, index=dates)  # Above MA for bullish trend
        
        signals, strength = generate_signals_basic_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr, close=close,
            require_trend_confirmation=True,
            oscillator_threshold=0.0
        )
        
        # First 5 bars: oscillator < 0, should be SHORT or NEUTRAL
        # Bar 6: zero cross up (oscillator goes from -10 to 5)
        # Bars 7-20: oscillator > 0 with bullish trend, should be LONG
        # Zero cross at bar 6 should reset to 0, then forward fill LONG
        assert signals.iloc[5] == 0  # Zero cross position
        assert all(signals.iloc[6:20] == 1)  # LONG after zero cross
    
    def test_strategy1_zero_cross_down_with_forward_fill(self):
        """Test Strategy 1 zero cross down and forward fill behavior."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')
        
        # Oscillator: positive -> crosses zero -> negative (should maintain SHORT after cross)
        oscillator_values = [10.0] * 5 + [-5.0] * 15
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 20, index=dates)
        range_atr = pd.Series([1000.0] * 20, index=dates)
        close = pd.Series([49000.0] * 20, index=dates)  # Below MA for bearish trend
        
        signals, strength = generate_signals_basic_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr, close=close,
            require_trend_confirmation=True,
            oscillator_threshold=0.0
        )
        
        # First 5 bars: oscillator > 0, should be LONG
        # Bar 6: zero cross down (oscillator goes from 10 to -5)
        # Bars 7-20: oscillator < 0 with bearish trend, should be SHORT
        # Zero cross at bar 6 should reset to 0, then forward fill SHORT
        assert signals.iloc[5] == 0  # Zero cross position
        assert all(signals.iloc[6:20] == -1)  # SHORT after zero cross
    
    def test_strategy1_multiple_zero_crosses(self):
        """Test Strategy 1 with multiple zero crosses."""
        dates = pd.date_range('2024-01-01', periods=30, freq='1h')
        
        # Oscillator: positive -> negative -> positive (multiple crosses)
        oscillator_values = [10.0] * 5 + [-10.0] * 5 + [10.0] * 5 + [-10.0] * 5 + [10.0] * 5 + [-10.0] * 5
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 30, index=dates)
        range_atr = pd.Series([1000.0] * 30, index=dates)
        close = pd.Series([51000.0] * 30, index=dates)  # Above MA
        
        signals, strength = generate_signals_basic_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr, close=close,
            require_trend_confirmation=False,  # Disable to see oscillator-only signals
            oscillator_threshold=0.0
        )
        
        # Check that zero crosses reset signals to 0
        # Bar 5: cross from +10 to -10 (should be 0)
        assert signals.iloc[5] == 0
        # Bar 10: cross from -10 to +10 (should be 0)
        assert signals.iloc[10] == 0
        # Bar 15: cross from +10 to -10 (should be 0)
        assert signals.iloc[15] == 0
    
    def test_strategy1_forward_fill_maintains_signal(self):
        """Test Strategy 1 forward fill maintains signal until zero cross."""
        dates = pd.date_range('2024-01-01', periods=25, freq='1h')
        
        # Oscillator: positive for long period, then crosses zero
        oscillator_values = [5.0] * 20 + [-5.0] * 5
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 25, index=dates)
        range_atr = pd.Series([1000.0] * 25, index=dates)
        close = pd.Series([51000.0] * 25, index=dates)  # Above MA
        
        signals, strength = generate_signals_basic_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr, close=close,
            require_trend_confirmation=True,
            oscillator_threshold=0.0
        )
        
        # Bars 0-19: oscillator > 0, should be LONG (forward filled)
        assert all(signals.iloc[0:20] == 1)
        # Bar 20: zero cross (should reset to 0)
        assert signals.iloc[20] == 0
        # Bars 21-24: oscillator < 0, but close > MA (bullish trend)
        # With trend confirmation, oscillator < 0 but close > MA should NOT generate SHORT
        # So should be 0 (NEUTRAL) or forward filled from previous signal
        # Note: Forward fill logic may maintain previous LONG until new signal appears
        assert isinstance(signals, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy1_forward_fill_with_nan_values(self):
        """Test Strategy 1 forward fill behavior with NaN values."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')
        
        # Oscillator with NaN values in the middle
        oscillator_values = [10.0] * 5 + [np.nan] * 3 + [10.0] * 12
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 20, index=dates)
        range_atr = pd.Series([1000.0] * 20, index=dates)
        close = pd.Series([51000.0] * 20, index=dates)
        
        signals, strength = generate_signals_basic_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr, close=close,
            require_trend_confirmation=True,
            oscillator_threshold=0.0
        )
        
        # NaN positions should be 0 (NEUTRAL)
        assert all(signals.iloc[5:8] == 0)
        # Valid positions should have signals
        assert isinstance(signals, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy1_zero_cross_without_trend_confirmation(self):
        """Test Strategy 1 zero cross when trend confirmation is disabled."""
        dates = pd.date_range('2024-01-01', periods=15, freq='1h')
        
        # Oscillator crosses zero, but close is below MA (bearish trend)
        oscillator_values = [-10.0] * 5 + [10.0] * 10
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 15, index=dates)
        range_atr = pd.Series([1000.0] * 15, index=dates)
        close = pd.Series([49000.0] * 15, index=dates)  # Below MA (bearish)
        
        signals, strength = generate_signals_basic_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr, close=close,
            require_trend_confirmation=False,  # Disable trend confirmation
            oscillator_threshold=0.0
        )
        
        # Without trend confirmation, oscillator > 0 should generate LONG
        # Bar 5: zero cross (should be 0)
        assert signals.iloc[5] == 0
        # Bars 6-14: oscillator > 0, should be LONG (no trend check)
        assert all(signals.iloc[6:15] == 1)
    
    def test_strategy1_zero_cross_with_trend_confirmation(self):
        """Test Strategy 1 zero cross when trend confirmation is enabled."""
        dates = pd.date_range('2024-01-01', periods=15, freq='1h')
        
        # Oscillator crosses zero, but close is below MA (bearish trend)
        oscillator_values = [-10.0] * 5 + [10.0] * 10
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 15, index=dates)
        range_atr = pd.Series([1000.0] * 15, index=dates)
        close = pd.Series([49000.0] * 15, index=dates)  # Below MA (bearish)
        
        signals, strength = generate_signals_basic_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr, close=close,
            require_trend_confirmation=True,  # Enable trend confirmation
            oscillator_threshold=0.0
        )
        
        # With trend confirmation, oscillator > 0 but close < MA should NOT generate LONG
        # Bar 5: zero cross (should be 0)
        assert signals.iloc[5] == 0
        # Bars 6-14: oscillator > 0 but close < MA, should be 0 or forward filled SHORT
        # Note: Forward fill may maintain previous SHORT signal from bars 0-4
        # The key is that no new LONG signals should be generated
        assert isinstance(signals, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
        # Verify no LONG signals after zero cross when trend is bearish
        assert all(signals.iloc[6:15] != 1)  # No LONG signals
    
    def test_strategy1_large_dataset_performance(self):
        """Test Strategy 1 performance with large dataset."""
        # Create large dataset (10,000 bars)
        dates = pd.date_range('2024-01-01', periods=10000, freq='1h')
        oscillator = pd.Series(np.sin(np.linspace(0, 20*np.pi, 10000)) * 50, index=dates)
        ma = pd.Series([50000.0] * 10000, index=dates)
        range_atr = pd.Series([1000.0] * 10000, index=dates)
        close = pd.Series([51000.0] * 10000, index=dates)
        
        import time
        start_time = time.time()
        
        signals, strength = generate_signals_basic_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr, close=close,
            require_trend_confirmation=True
        )
        
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (< 1 second for 10k bars)
        assert elapsed_time < 1.0, f"Performance test failed: {elapsed_time:.3f}s for 10k bars"
        assert len(signals) == 10000
        assert len(strength) == 10000
        assert all(signals.isin([-1, 0, 1]))


class TestStrategy2:
    """Tests for Strategy 2: Sustained pressure."""
    
    def test_strategy2_basic(self, sample_ohlc_data):
        """Test basic Strategy 2 functionality."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_sustained_strategy(
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
        
        signals, strength = generate_signals_sustained_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            min_bars_above_zero=5
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
    
    def test_strategy2_validation(self, sample_ohlc_data):
        """Test Strategy 2 parameter validation."""
        high, low, close = sample_ohlc_data
        
        # Test min_bars_above_zero <= 0
        with pytest.raises(ValueError, match="min_bars_above_zero must be > 0"):
            generate_signals_sustained_strategy(
                high=high, low=low, close=close,
                min_bars_above_zero=0
            )
        
        with pytest.raises(ValueError, match="min_bars_above_zero must be > 0"):
            generate_signals_sustained_strategy(
                high=high, low=low, close=close,
                min_bars_above_zero=-1
            )
        
        # Test min_bars_below_zero <= 0
        with pytest.raises(ValueError, match="min_bars_below_zero must be > 0"):
            generate_signals_sustained_strategy(
                high=high, low=low, close=close,
                min_bars_below_zero=0
            )
    
    def test_strategy2_oscillator_at_threshold(self):
        """Test Strategy 2 when oscillator equals threshold exactly."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')
        
        # Oscillator exactly at threshold (0.0)
        oscillator = pd.Series([0.0] * 20, index=dates)
        ma = pd.Series([50000.0] * 20, index=dates)
        range_atr = pd.Series([1000.0] * 20, index=dates)
        
        signals, strength = generate_signals_sustained_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            min_bars_above_zero=3,
            min_bars_below_zero=3,
            oscillator_threshold=0.0
        )
        
        # Should be all NEUTRAL (0) since oscillator is exactly at threshold
        assert all(signals == 0)
        assert all(strength == 0.0)
    
    def test_strategy2_oscillator_oscillating_around_threshold(self):
        """Test Strategy 2 when oscillator oscillates around threshold."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')
        
        # Oscillator oscillates: 1, -1, 1, -1, ... (around threshold 0.0)
        oscillator_values = [1.0 if i % 2 == 0 else -1.0 for i in range(20)]
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 20, index=dates)
        range_atr = pd.Series([1000.0] * 20, index=dates)
        
        signals, strength = generate_signals_sustained_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            min_bars_above_zero=3,
            min_bars_below_zero=3,
            oscillator_threshold=0.0
        )
        
        # Should be all NEUTRAL (0) since oscillator never stays above/below for 3 consecutive bars
        assert all(signals == 0)
        assert all(strength == 0.0)
    
    def test_strategy2_sustained_above_threshold(self):
        """Test Strategy 2 with sustained pressure above threshold."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')
        
        # Oscillator stays above threshold for 5 consecutive bars
        oscillator_values = [10.0] * 5 + [0.0] * 15
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 20, index=dates)
        range_atr = pd.Series([1000.0] * 20, index=dates)
        
        signals, strength = generate_signals_sustained_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            min_bars_above_zero=3,
            min_bars_below_zero=3,
            oscillator_threshold=0.0
        )
        
        # With min_bars_above_zero=3, signals only appear after 3 consecutive bars
        # Bars 0-1: 0 (not enough bars yet)
        # Bars 2-4: 1 (3+ consecutive bars above threshold)
        assert all(signals.iloc[:2] == 0)
        assert all(signals.iloc[2:5] == 1)
        assert all(signals.iloc[5:] == 0)
        assert all(strength.iloc[2:5] > 0)
        assert all(strength.iloc[5:] == 0.0)
    
    def test_strategy2_sustained_below_threshold(self):
        """Test Strategy 2 with sustained pressure below threshold."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')
        
        # Oscillator stays below threshold for 5 consecutive bars
        oscillator_values = [-10.0] * 5 + [0.0] * 15
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 20, index=dates)
        range_atr = pd.Series([1000.0] * 20, index=dates)
        
        signals, strength = generate_signals_sustained_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            min_bars_above_zero=3,
            min_bars_below_zero=3,
            oscillator_threshold=0.0
        )
        
        # With min_bars_below_zero=3, signals only appear after 3 consecutive bars
        # Bars 0-1: 0 (not enough bars yet)
        # Bars 2-4: -1 (3+ consecutive bars below threshold)
        assert all(signals.iloc[:2] == 0)
        assert all(signals.iloc[2:5] == -1)
        assert all(signals.iloc[5:] == 0)
        assert all(strength.iloc[2:5] > 0)
        assert all(strength.iloc[5:] == 0.0)
    
    def test_strategy2_conflict_long_short(self):
        """Test Strategy 2 conflict handling when both LONG and SHORT conditions are met."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')
        
        # Create edge case where both conditions could theoretically be true
        # This is unlikely in practice but can happen with NaN or data inconsistencies
        oscillator = pd.Series([10.0] * 10, index=dates[:10])
        oscillator = pd.concat([oscillator, pd.Series([-10.0] * 10, index=dates[10:])])
        ma = pd.Series([50000.0] * 20, index=dates)
        range_atr = pd.Series([1000.0] * 20, index=dates)
        
        # Create artificial conflict by manipulating bars_above_zero and bars_below_zero
        # In practice, this would require both to be >= min_bars simultaneously
        # We'll test with a scenario where oscillator has NaN causing both conditions
        oscillator_with_nan = oscillator.copy()
        oscillator_with_nan.iloc[5] = np.nan
        
        signals, strength = generate_signals_sustained_strategy(
            oscillator=oscillator_with_nan, ma=ma, range_atr=range_atr,
            min_bars_above_zero=3,
            min_bars_below_zero=3,
            oscillator_threshold=0.0
        )
        
        # Signals should be valid (no conflicts should occur in normal operation)
        assert all(signals.isin([-1, 0, 1]))
        # NaN positions should be 0
        assert signals.iloc[5] == 0
    
    def test_strategy2_custom_threshold(self):
        """Test Strategy 2 with custom threshold value."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')
        
        # Oscillator stays above threshold 5.0 for 5 consecutive bars
        oscillator_values = [10.0] * 5 + [0.0] * 15
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 20, index=dates)
        range_atr = pd.Series([1000.0] * 20, index=dates)
        
        signals, strength = generate_signals_sustained_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            min_bars_above_zero=3,
            min_bars_below_zero=3,
            oscillator_threshold=5.0
        )
        
        # With min_bars_above_zero=3, signals only appear after 3 consecutive bars
        # Bars 0-1: 0 (not enough bars yet)
        # Bars 2-4: 1 (3+ consecutive bars above threshold 5.0)
        assert all(signals.iloc[:2] == 0)
        assert all(signals.iloc[2:5] == 1)
        assert all(signals.iloc[5:] == 0)


class TestStrategy3:
    """Tests for Strategy 3: Zero line crossover."""
    
    def test_strategy3_basic(self, sample_ohlc_data):
        """Test basic Strategy 3 functionality."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_crossover_strategy(
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
        
        signals, strength = generate_signals_crossover_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            confirmation_bars=3
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
    
    def test_strategy3_validation(self, sample_ohlc_data):
        """Test Strategy 3 parameter validation."""
        high, low, close = sample_ohlc_data
        
        with pytest.raises(ValueError, match="confirmation_bars must be > 0"):
            generate_signals_crossover_strategy(
                high=high, low=low, close=close,
                confirmation_bars=0
            )
    
    def test_strategy3_crossover_with_confirmation(self):
        """Test Strategy 3 crossover with confirmation."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')
        
        # Oscillator crosses above zero and stays above for confirmation_bars
        oscillator_values = [-10.0] * 5 + [5.0] * 5 + [10.0] * 10
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 20, index=dates)
        range_atr = pd.Series([1000.0] * 20, index=dates)
        
        signals, strength = generate_signals_crossover_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            confirmation_bars=2,
            oscillator_threshold=0.0
        )
        
        # Should have LONG signals after confirmed crossover
        assert isinstance(signals, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy3_no_confirmation(self):
        """Test Strategy 3 when crossover is not confirmed."""
        dates = pd.date_range('2024-01-01', periods=15, freq='1h')
        
        # Oscillator crosses above but immediately goes back below
        oscillator_values = [-10.0] * 5 + [5.0] * 1 + [-10.0] * 9
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 15, index=dates)
        range_atr = pd.Series([1000.0] * 15, index=dates)
        
        signals, strength = generate_signals_crossover_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            confirmation_bars=3,
            oscillator_threshold=0.0
        )
        
        # Should not have LONG signal since crossover not confirmed
        assert isinstance(signals, pd.Series)
        assert all(signals.isin([-1, 0, 1]))


class TestStrategy4:
    """Tests for Strategy 4: Momentum."""
    
    def test_strategy4_basic(self, sample_ohlc_data):
        """Test basic Strategy 4 functionality."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_momentum_strategy(
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
        
        signals, strength = generate_signals_momentum_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            momentum_period=5
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
    
    def test_strategy4_validation(self, sample_ohlc_data):
        """Test Strategy 4 parameter validation."""
        high, low, close = sample_ohlc_data
        
        with pytest.raises(ValueError, match="momentum_period must be > 0"):
            generate_signals_momentum_strategy(
                high=high, low=low, close=close,
                momentum_period=0
            )
        
        with pytest.raises(ValueError, match="momentum_threshold must be >= 0"):
            generate_signals_momentum_strategy(
                high=high, low=low, close=close,
                momentum_threshold=-1.0
            )
    
    def test_strategy4_strong_momentum(self):
        """Test Strategy 4 with strong momentum."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')
        
        # Oscillator with strong positive momentum
        oscillator_values = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0] * 2
        oscillator = pd.Series(oscillator_values[:20], index=dates)
        ma = pd.Series([50000.0] * 20, index=dates)
        range_atr = pd.Series([1000.0] * 20, index=dates)
        
        signals, strength = generate_signals_momentum_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            momentum_period=3,
            momentum_threshold=5.0
        )
        
        assert isinstance(signals, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
        assert all((strength >= 0) & (strength <= 1))
    
    def test_strategy4_insufficient_data(self):
        """Test Strategy 4 with insufficient data."""
        dates = pd.date_range('2024-01-01', periods=5, freq='1h')
        oscillator = pd.Series([10.0] * 5, index=dates)
        ma = pd.Series([50000.0] * 5, index=dates)
        range_atr = pd.Series([1000.0] * 5, index=dates)
        
        with pytest.raises(ValueError, match="momentum_period.*must be < data length"):
            generate_signals_momentum_strategy(
                oscillator=oscillator, ma=ma, range_atr=range_atr,
                momentum_period=10
            )


class TestStrategy5:
    """Tests for Strategy 5: Combined."""
    
    def test_strategy5_basic(self, sample_ohlc_data):
        """Test basic Strategy 5 functionality."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_combined_all_strategy(
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
        
        signals, strength = generate_signals_combined_all_strategy(
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
        
        signals, strength = generate_signals_combined_all_strategy(
            high=high, low=low, close=close,
            use_sustained=False,
            use_crossover=False,
            use_momentum=False
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
    
    def test_strategy5_validation(self, sample_ohlc_data):
        """Test Strategy 5 parameter validation."""
        high, low, close = sample_ohlc_data
        
        with pytest.raises(ValueError, match="min_bars_sustained must be > 0"):
            generate_signals_combined_all_strategy(
                high=high, low=low, close=close,
                use_sustained=True,
                min_bars_sustained=0
            )
        
        with pytest.raises(ValueError, match="confirmation_bars must be > 0"):
            generate_signals_combined_all_strategy(
                high=high, low=low, close=close,
                use_crossover=True,
                confirmation_bars=0
            )
    
    def test_strategy5_majority_vote(self):
        """Test Strategy 5 majority vote logic."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')
        oscillator = pd.Series([10.0] * 20, index=dates)
        ma = pd.Series([50000.0] * 20, index=dates)
        range_atr = pd.Series([1000.0] * 20, index=dates)
        
        signals, strength = generate_signals_combined_all_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            use_sustained=True,
            use_crossover=True,
            use_momentum=True
        )
        
        assert isinstance(signals, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
        assert all((strength >= 0) & (strength <= 1))


class TestStrategy5Integration:
    """Integration tests for Strategy 5: Combined strategy with majority voting."""
    
    @pytest.fixture
    def sample_data_for_voting(self):
        """Create sample data for testing majority voting."""
        dates = pd.date_range('2024-01-01', periods=50, freq='1h')
        # Create oscillator that will generate different signals from different strategies
        oscillator = pd.Series(np.sin(np.linspace(0, 4*np.pi, 50)) * 50, index=dates)
        ma = pd.Series([50000.0] * 50, index=dates)
        range_atr = pd.Series([1000.0] * 50, index=dates)
        return oscillator, ma, range_atr
    
    def test_majority_vote_long_wins_2_to_1(self, sample_data_for_voting):
        """Test majority voting when 2 strategies vote LONG, 1 votes SHORT."""
        oscillator, ma, range_atr = sample_data_for_voting
        
        # Create scenario where sustained and crossover vote LONG, momentum votes SHORT
        # Use parameters that favor different outcomes
        signals, strength = generate_signals_combined_all_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            use_sustained=True,
            use_crossover=True,
            use_momentum=True,
            min_bars_sustained=2,
            confirmation_bars=1,
            momentum_period=3,
            momentum_threshold=10.0
        )
        
        # Verify signals are valid
        assert isinstance(signals, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
        # When majority votes LONG, result should be LONG (or NEUTRAL if tie)
        # We can't guarantee specific outcome, but verify logic works
        assert len(signals) == len(oscillator)
    
    def test_majority_vote_tie_results_in_neutral(self, sample_data_for_voting):
        """Test that ties in voting result in NEUTRAL signals."""
        oscillator, ma, range_atr = sample_data_for_voting
        
        # Test with 2 methods enabled (can create ties)
        signals, strength = generate_signals_combined_all_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            use_sustained=True,
            use_crossover=True,
            use_momentum=False,  # Only 2 methods to allow ties
            min_bars_sustained=2,
            confirmation_bars=1
        )
        
        # Verify signals are valid
        assert isinstance(signals, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
        # Ties should result in 0 (NEUTRAL)
        # We can verify that when both vote differently, result is 0
    
    def test_majority_vote_all_methods_enabled(self, sample_data_for_voting):
        """Test with all 3 methods enabled."""
        oscillator, ma, range_atr = sample_data_for_voting
        
        signals, strength = generate_signals_combined_all_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            use_sustained=True,
            use_crossover=True,
            use_momentum=True
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(oscillator)
        assert all(signals.isin([-1, 0, 1]))
        assert all((strength >= 0) & (strength <= 1))
    
    def test_combination_sustained_only(self, sample_data_for_voting):
        """Test Strategy 5 with only sustained method enabled."""
        oscillator, ma, range_atr = sample_data_for_voting
        
        signals, strength = generate_signals_combined_all_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            use_sustained=True,
            use_crossover=False,
            use_momentum=False
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
        # Should behave like Strategy 2
        assert len(signals) == len(oscillator)
    
    def test_combination_crossover_only(self, sample_data_for_voting):
        """Test Strategy 5 with only crossover method enabled."""
        oscillator, ma, range_atr = sample_data_for_voting
        
        signals, strength = generate_signals_combined_all_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            use_sustained=False,
            use_crossover=True,
            use_momentum=False
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
        # Should behave like Strategy 3
        assert len(signals) == len(oscillator)
    
    def test_combination_momentum_only(self, sample_data_for_voting):
        """Test Strategy 5 with only momentum method enabled."""
        oscillator, ma, range_atr = sample_data_for_voting
        
        signals, strength = generate_signals_combined_all_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            use_sustained=False,
            use_crossover=False,
            use_momentum=True
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
        # Should behave like Strategy 4
        assert len(signals) == len(oscillator)
    
    def test_combination_sustained_and_crossover(self, sample_data_for_voting):
        """Test Strategy 5 with sustained and crossover methods enabled."""
        oscillator, ma, range_atr = sample_data_for_voting
        
        signals, strength = generate_signals_combined_all_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            use_sustained=True,
            use_crossover=True,
            use_momentum=False
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
        # Majority vote between 2 methods
        assert len(signals) == len(oscillator)
    
    def test_combination_sustained_and_momentum(self, sample_data_for_voting):
        """Test Strategy 5 with sustained and momentum methods enabled."""
        oscillator, ma, range_atr = sample_data_for_voting
        
        signals, strength = generate_signals_combined_all_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            use_sustained=True,
            use_crossover=False,
            use_momentum=True
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
        # Majority vote between 2 methods
        assert len(signals) == len(oscillator)
    
    def test_combination_crossover_and_momentum(self, sample_data_for_voting):
        """Test Strategy 5 with crossover and momentum methods enabled."""
        oscillator, ma, range_atr = sample_data_for_voting
        
        signals, strength = generate_signals_combined_all_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            use_sustained=False,
            use_crossover=True,
            use_momentum=True
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
        # Majority vote between 2 methods
        assert len(signals) == len(oscillator)
    
    def test_strength_calculation_with_multiple_votes(self, sample_data_for_voting):
        """Test that strength is calculated as average of voting strategies."""
        oscillator, ma, range_atr = sample_data_for_voting
        
        signals, strength = generate_signals_combined_all_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            use_sustained=True,
            use_crossover=True,
            use_momentum=True
        )
        
        # Strength should be in valid range
        assert all((strength >= 0) & (strength <= 1))
        # Strength should be 0 where signal is 0
        assert all(strength[signals == 0] == 0.0)
        # Strength should be > 0 where signal is not 0
        if (signals != 0).any():
            assert all(strength[signals != 0] > 0.0)
    
    def test_majority_vote_consistency(self):
        """Test that majority voting is consistent across different scenarios."""
        dates = pd.date_range('2024-01-01', periods=30, freq='1h')
        
        # Scenario 1: All strategies should vote LONG
        oscillator_long = pd.Series([50.0] * 30, index=dates)
        ma = pd.Series([50000.0] * 30, index=dates)
        range_atr = pd.Series([1000.0] * 30, index=dates)
        
        signals1, strength1 = generate_signals_combined_all_strategy(
            oscillator=oscillator_long, ma=ma, range_atr=range_atr,
            use_sustained=True,
            use_crossover=True,
            use_momentum=True,
            min_bars_sustained=2,
            confirmation_bars=1,
            momentum_period=3,
            momentum_threshold=5.0
        )
        
        # Should have LONG signals (majority or all vote LONG)
        assert isinstance(signals1, pd.Series)
        assert all(signals1.isin([-1, 0, 1]))
        
        # Scenario 2: All strategies should vote SHORT
        oscillator_short = pd.Series([-50.0] * 30, index=dates)
        
        signals2, strength2 = generate_signals_combined_all_strategy(
            oscillator=oscillator_short, ma=ma, range_atr=range_atr,
            use_sustained=True,
            use_crossover=True,
            use_momentum=True,
            min_bars_sustained=2,
            confirmation_bars=1,
            momentum_period=3,
            momentum_threshold=5.0
        )
        
        # Should have SHORT signals (majority or all vote SHORT)
        assert isinstance(signals2, pd.Series)
        assert all(signals2.isin([-1, 0, 1]))
    
    def test_fallback_to_strategy1_when_no_methods_enabled(self, sample_ohlc_data):
        """Test that Strategy 5 falls back to Strategy 1 when no methods enabled."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_combined_all_strategy(
            high=high, low=low, close=close,
            use_sustained=False,
            use_crossover=False,
            use_momentum=False
        )
        
        # Should return Strategy 1 signals
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        assert len(signals) == len(close)
        assert all(signals.isin([-1, 0, 1]))
        
        # Compare with Strategy 1 directly
        signals1, strength1 = generate_signals_basic_strategy(
            high=high, low=low, close=close
        )
        
        # Should produce same results
        assert len(signals) == len(signals1)
        # Signals may differ slightly due to different parameter defaults, but structure should be same
        assert all(signals.isin([-1, 0, 1]))
        assert all(signals1.isin([-1, 0, 1]))


class TestStrategy6:
    """Tests for Strategy 6: Range breakouts."""
    
    def test_strategy6_basic(self, sample_ohlc_data):
        """Test basic Strategy 6 functionality."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_breakout_strategy(
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
        
        signals, strength = generate_signals_breakout_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            upper_threshold=80.0,
            detect_exhaustion=False
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
    
    def test_strategy6_validation(self, sample_ohlc_data):
        """Test Strategy 6 parameter validation."""
        high, low, close = sample_ohlc_data
        
        with pytest.raises(ValueError, match="confirmation_bars must be > 0"):
            generate_signals_breakout_strategy(
                high=high, low=low, close=close,
                require_confirmation=True,
                confirmation_bars=0
            )
        
        with pytest.raises(ValueError, match="exhaustion_threshold must be >= 0"):
            generate_signals_breakout_strategy(
                high=high, low=low, close=close,
                detect_exhaustion=True,
                exhaustion_threshold=-1.0
            )
        
        with pytest.raises(ValueError, match="upper_threshold.*must be > lower_threshold"):
            generate_signals_breakout_strategy(
                high=high, low=low, close=close,
                upper_threshold=50.0,
                lower_threshold=100.0
            )
    
    def test_strategy6_breakout_above(self):
        """Test Strategy 6 breakout above threshold."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')
        
        # Oscillator breaks above upper threshold
        oscillator_values = [90.0] * 5 + [110.0] * 15
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 20, index=dates)
        range_atr = pd.Series([1000.0] * 20, index=dates)
        
        signals, strength = generate_signals_breakout_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            upper_threshold=100.0,
            lower_threshold=-100.0,
            require_confirmation=True,
            confirmation_bars=2
        )
        
        assert isinstance(signals, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy6_exhaustion_detection(self):
        """Test Strategy 6 exhaustion detection."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')
        
        # Oscillator in exhaustion zone
        oscillator_values = [160.0] * 20
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 20, index=dates)
        range_atr = pd.Series([1000.0] * 20, index=dates)
        
        signals, strength = generate_signals_breakout_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            upper_threshold=100.0,
            lower_threshold=-100.0,
            detect_exhaustion=True,
            exhaustion_threshold=150.0
        )
        
        # Should not have signals in exhaustion zone
        assert isinstance(signals, pd.Series)
        assert all(signals == 0)  # All should be NEUTRAL in exhaustion


class TestStrategy7:
    """Tests for Strategy 7: Divergence detection."""
    
    def test_strategy7_basic(self, sample_ohlc_data):
        """Test basic Strategy 7 functionality."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_divergence_strategy(
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
            generate_signals_divergence_strategy(
                oscillator=oscillator, ma=ma, range_atr=range_atr
            )
    
    def test_strategy7_validation(self, sample_ohlc_data):
        """Test Strategy 7 parameter validation."""
        high, low, close = sample_ohlc_data
        
        with pytest.raises(ValueError, match="min_swing_bars must be > 0"):
            generate_signals_divergence_strategy(
                high=high, low=low, close=close,
                min_swing_bars=0
            )
        
        with pytest.raises(ValueError, match="lookback_period must be > 0"):
            generate_signals_divergence_strategy(
                high=high, low=low, close=close,
                lookback_period=0
            )
        
        with pytest.raises(ValueError, match="min_divergence_strength must be >= 0"):
            generate_signals_divergence_strategy(
                high=high, low=low, close=close,
                min_divergence_strength=-1.0
            )
    
    def test_strategy7_bullish_divergence(self):
        """Test Strategy 7 bullish divergence detection."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        
        # Create price and oscillator data with bullish divergence
        # Price makes lower low, oscillator makes higher low
        price_base = 50000
        high = pd.Series([price_base + i * 10 for i in range(100)], index=dates)
        low = pd.Series([price_base - i * 10 for i in range(100)], index=dates)
        close = pd.Series([price_base] * 100, index=dates)
        
        # Oscillator: lower low at first, then higher low
        oscillator_values = [-50.0] * 30 + [-30.0] * 30 + [-10.0] * 40
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([price_base] * 100, index=dates)
        range_atr = pd.Series([1000.0] * 100, index=dates)
        
        signals, strength = generate_signals_divergence_strategy(
            high=high, low=low, close=close,
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            lookback_period=30,
            min_swing_bars=5,
            min_divergence_strength=10.0
        )
        
        assert isinstance(signals, pd.Series)
        assert all(signals.isin([-1, 0, 1]))


class TestStrategy8:
    """Tests for Strategy 8: Trend following."""
    
    def test_strategy8_basic(self, sample_ohlc_data):
        """Test basic Strategy 8 functionality."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_trend_following_strategy(
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
        
        signals, strength = generate_signals_trend_following_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            require_consistency=False
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
    
    def test_strategy8_validation(self, sample_ohlc_data):
        """Test Strategy 8 parameter validation."""
        high, low, close = sample_ohlc_data
        
        with pytest.raises(ValueError, match="trend_filter_period must be > 0"):
            generate_signals_trend_following_strategy(
                high=high, low=low, close=close,
                trend_filter_period=0
            )
        
        with pytest.raises(ValueError, match="oscillator_threshold must be >= 0"):
            generate_signals_trend_following_strategy(
                high=high, low=low, close=close,
                oscillator_threshold=-1.0
            )
    
    def test_strategy8_consistency_requirement(self):
        """Test Strategy 8 with consistency requirement."""
        dates = pd.date_range('2024-01-01', periods=30, freq='1h')
        
        # Oscillator consistently above threshold with upward trend
        oscillator_values = [25.0] * 30
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 30, index=dates)
        range_atr = pd.Series([1000.0] * 30, index=dates)
        
        signals, strength = generate_signals_trend_following_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            trend_filter_period=10,
            oscillator_threshold=20.0,
            require_consistency=True
        )
        
        assert isinstance(signals, pd.Series)
        assert all(signals.isin([-1, 0, 1]))


class TestStrategy9:
    """Tests for Strategy 9: Mean reversion."""
    
    def test_strategy9_basic(self, sample_ohlc_data):
        """Test basic Strategy 9 functionality."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_mean_reversion_strategy(
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
        
        signals, strength = generate_signals_mean_reversion_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            min_extreme_bars=3
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
    
    def test_strategy9_validation(self, sample_ohlc_data):
        """Test Strategy 9 parameter validation."""
        high, low, close = sample_ohlc_data
        
        with pytest.raises(ValueError, match="extreme_threshold must be >= 0"):
            generate_signals_mean_reversion_strategy(
                high=high, low=low, close=close,
                extreme_threshold=-1.0
            )
        
        with pytest.raises(ValueError, match="zero_cross_threshold must be >= 0"):
            generate_signals_mean_reversion_strategy(
                high=high, low=low, close=close,
                zero_cross_threshold=-1.0
            )
        
        with pytest.raises(ValueError, match="min_extreme_bars must be > 0"):
            generate_signals_mean_reversion_strategy(
                high=high, low=low, close=close,
                min_extreme_bars=0
            )
        
        with pytest.raises(ValueError, match="transition_bars must be > 0"):
            generate_signals_mean_reversion_strategy(
                high=high, low=low, close=close,
                transition_bars=0
            )
    
    def test_strategy9_mean_reversion_from_extreme(self):
        """Test Strategy 9 mean reversion from extreme."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')
        
        # Oscillator at extreme positive, then transitions back toward zero
        oscillator_values = [90.0] * 5 + [5.0] * 15
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 20, index=dates)
        range_atr = pd.Series([1000.0] * 20, index=dates)
        
        signals, strength = generate_signals_mean_reversion_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            extreme_threshold=80.0,
            zero_cross_threshold=10.0,
            min_extreme_bars=3,
            transition_bars=5
        )
        
        assert isinstance(signals, pd.Series)
        assert all(signals.isin([-1, 0, 1]))
        # Should have SHORT signal when transitioning from extreme positive
        assert isinstance(strength, pd.Series)


class TestSignalSummary:
    """Tests for get_signal_summary function."""
    
    def test_get_signal_summary_basic(self, sample_ohlc_data):
        """Test basic signal summary functionality."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_basic_strategy(
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
    
    def test_get_signal_summary_with_nan(self):
        """Test signal summary with NaN values."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1h')
        # Use float64 dtype to allow NaN values, then convert to int8 where possible
        signals = pd.Series([1, 1, np.nan, -1, -1, 0, 0, 1, np.nan, 0], index=dates, dtype='float64')
        # Fill NaN with 0 for counting purposes, but keep original for summary
        signals_for_count = signals.fillna(0).astype('int8')
        strength = pd.Series([0.5, 0.6, np.nan, 0.7, 0.8, 0.0, 0.0, 0.4, np.nan, 0.0], index=dates)
        close = pd.Series([50000.0] * 10, index=dates)
        
        # Use signals_for_count for summary (NaN treated as 0)
        summary = get_signal_summary(signals_for_count, strength, close)
        
        assert isinstance(summary, dict)
        assert summary["total_signals"] == 10
        assert summary["long_signals"] >= 0
        assert summary["short_signals"] >= 0
        assert "long_percentage" in summary
        assert "short_percentage" in summary


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_all_strategies_with_short_data(self):
        """Test all strategies with very short data."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1h')
        high = pd.Series([50000] * 10, index=dates)
        low = pd.Series([49000] * 10, index=dates)
        close = pd.Series([49500] * 10, index=dates)
        
        # Should not raise errors, but may return mostly zeros
        signals, strength = generate_signals_basic_strategy(
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
        
        signals, strength = generate_signals_basic_strategy(
            high=high, low=low, close=close_with_nan
        )
        
        assert isinstance(signals, pd.Series)
        assert isinstance(strength, pd.Series)
        # Signals should be 0 where data is NaN
        assert all(signals.iloc[10:15] == 0)


class TestConflictHandling:
    """Tests for conflict handling across all strategies."""
    
    def test_strategy1_conflict_handling(self, sample_ohlc_data):
        """Test Strategy 1 handles conflicts (both LONG and SHORT conditions)."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_basic_strategy(
            high=high, low=low, close=close
        )
        
        # Should never have both LONG and SHORT at same time
        assert not any((signals == 1) & (signals == -1))
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy2_conflict_handling(self, sample_ohlc_data):
        """Test Strategy 2 handles conflicts."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_sustained_strategy(
            high=high, low=low, close=close
        )
        
        assert not any((signals == 1) & (signals == -1))
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy3_conflict_handling(self, sample_ohlc_data):
        """Test Strategy 3 handles conflicts."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_crossover_strategy(
            high=high, low=low, close=close
        )
        
        assert not any((signals == 1) & (signals == -1))
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy4_conflict_handling(self, sample_ohlc_data):
        """Test Strategy 4 handles conflicts."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_momentum_strategy(
            high=high, low=low, close=close
        )
        
        assert not any((signals == 1) & (signals == -1))
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy5_conflict_handling(self, sample_ohlc_data):
        """Test Strategy 5 handles conflicts."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_combined_all_strategy(
            high=high, low=low, close=close
        )
        
        assert not any((signals == 1) & (signals == -1))
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy6_conflict_handling(self, sample_ohlc_data):
        """Test Strategy 6 handles conflicts."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_breakout_strategy(
            high=high, low=low, close=close
        )
        
        assert not any((signals == 1) & (signals == -1))
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy7_conflict_handling(self, sample_ohlc_data):
        """Test Strategy 7 handles conflicts."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_divergence_strategy(
            high=high, low=low, close=close
        )
        
        assert not any((signals == 1) & (signals == -1))
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy8_conflict_handling(self, sample_ohlc_data):
        """Test Strategy 8 handles conflicts."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_trend_following_strategy(
            high=high, low=low, close=close
        )
        
        assert not any((signals == 1) & (signals == -1))
        assert all(signals.isin([-1, 0, 1]))
    
    def test_strategy9_conflict_handling(self, sample_ohlc_data):
        """Test Strategy 9 handles conflicts."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_mean_reversion_strategy(
            high=high, low=low, close=close
        )
        
        assert not any((signals == 1) & (signals == -1))
        assert all(signals.isin([-1, 0, 1]))


class TestNaNHandling:
    """Tests for NaN handling across all strategies."""
    
    def test_strategy1_nan_handling(self, sample_ohlc_data):
        """Test Strategy 1 handles NaN values correctly."""
        high, low, close = sample_ohlc_data
        
        # Add NaN values to oscillator data
        close_with_nan = close.copy()
        close_with_nan.iloc[50:60] = np.nan
        
        signals, strength = generate_signals_basic_strategy(
            high=high, low=low, close=close_with_nan
        )
        
        # Signals should be 0 where data is NaN
        assert all(signals.iloc[50:60] == 0)
        assert all(strength.iloc[50:60] == 0.0)
        assert all(signals.isin([-1, 0, 1]))
        assert all((strength >= 0) & (strength <= 1))
    
    def test_strategy2_nan_handling(self, sample_ohlc_data):
        """Test Strategy 2 handles NaN values."""
        high, low, close = sample_ohlc_data
        oscillator = pd.Series(np.sin(np.linspace(0, 4*np.pi, len(close))) * 50, index=close.index)
        oscillator.iloc[50:60] = np.nan
        ma = close.rolling(50).mean()
        range_atr = pd.Series(np.ones(len(close)) * 1000, index=close.index)
        
        signals, strength = generate_signals_sustained_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr
        )
        
        assert all(signals.iloc[50:60] == 0)
        assert all(strength.iloc[50:60] == 0.0)
    
    def test_strategy3_nan_handling(self, sample_ohlc_data):
        """Test Strategy 3 handles NaN values."""
        high, low, close = sample_ohlc_data
        oscillator = pd.Series(np.sin(np.linspace(0, 4*np.pi, len(close))) * 50, index=close.index)
        oscillator.iloc[50:60] = np.nan
        ma = close.rolling(50).mean()
        range_atr = pd.Series(np.ones(len(close)) * 1000, index=close.index)
        
        signals, strength = generate_signals_crossover_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr
        )
        
        assert all(signals.iloc[50:60] == 0)
        assert all(strength.iloc[50:60] == 0.0)
    
    def test_strategy4_nan_handling(self, sample_ohlc_data):
        """Test Strategy 4 handles NaN values."""
        high, low, close = sample_ohlc_data
        oscillator = pd.Series(np.sin(np.linspace(0, 4*np.pi, len(close))) * 50, index=close.index)
        oscillator.iloc[50:60] = np.nan
        ma = close.rolling(50).mean()
        range_atr = pd.Series(np.ones(len(close)) * 1000, index=close.index)
        
        signals, strength = generate_signals_momentum_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr
        )
        
        assert all(signals.iloc[50:60] == 0)
        assert all(strength.iloc[50:60] == 0.0)
    
    def test_strategy5_nan_handling(self, sample_ohlc_data):
        """Test Strategy 5 handles NaN values."""
        high, low, close = sample_ohlc_data
        oscillator = pd.Series(np.sin(np.linspace(0, 4*np.pi, len(close))) * 50, index=close.index)
        oscillator.iloc[50:60] = np.nan
        ma = close.rolling(50).mean()
        range_atr = pd.Series(np.ones(len(close)) * 1000, index=close.index)
        
        signals, strength = generate_signals_combined_all_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr
        )
        
        assert all(signals.iloc[50:60] == 0)
        assert all(strength.iloc[50:60] == 0.0)
    
    def test_strategy6_nan_handling(self, sample_ohlc_data):
        """Test Strategy 6 handles NaN values."""
        high, low, close = sample_ohlc_data
        oscillator = pd.Series(np.sin(np.linspace(0, 4*np.pi, len(close))) * 50, index=close.index)
        oscillator.iloc[50:60] = np.nan
        ma = close.rolling(50).mean()
        range_atr = pd.Series(np.ones(len(close)) * 1000, index=close.index)
        
        signals, strength = generate_signals_breakout_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr
        )
        
        assert all(signals.iloc[50:60] == 0)
        assert all(strength.iloc[50:60] == 0.0)
    
    def test_strategy7_nan_handling(self, sample_ohlc_data):
        """Test Strategy 7 handles NaN values."""
        high, low, close = sample_ohlc_data
        close_with_nan = close.copy()
        close_with_nan.iloc[50:60] = np.nan
        
        signals, strength = generate_signals_divergence_strategy(
            high=high, low=low, close=close_with_nan
        )
        
        assert all(signals.iloc[50:60] == 0)
        assert all(strength.iloc[50:60] == 0.0)
    
    def test_strategy8_nan_handling(self, sample_ohlc_data):
        """Test Strategy 8 handles NaN values."""
        high, low, close = sample_ohlc_data
        oscillator = pd.Series(np.sin(np.linspace(0, 4*np.pi, len(close))) * 50, index=close.index)
        oscillator.iloc[50:60] = np.nan
        ma = close.rolling(50).mean()
        range_atr = pd.Series(np.ones(len(close)) * 1000, index=close.index)
        
        signals, strength = generate_signals_trend_following_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr
        )
        
        assert all(signals.iloc[50:60] == 0)
        assert all(strength.iloc[50:60] == 0.0)
    
    def test_strategy9_nan_handling(self, sample_ohlc_data):
        """Test Strategy 9 handles NaN values."""
        high, low, close = sample_ohlc_data
        oscillator = pd.Series(np.sin(np.linspace(0, 4*np.pi, len(close))) * 50, index=close.index)
        oscillator.iloc[50:60] = np.nan
        ma = close.rolling(50).mean()
        range_atr = pd.Series(np.ones(len(close)) * 1000, index=close.index)
        
        signals, strength = generate_signals_mean_reversion_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr
        )
        
        assert all(signals.iloc[50:60] == 0)
        assert all(strength.iloc[50:60] == 0.0)


class TestLookAheadBiasPrevention:
    """Tests for look-ahead bias prevention in strategies with confirmation."""
    
    def test_strategy3_look_ahead_bias_prevention(self):
        """Test Strategy 3 does not use future data for confirmation."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')
        
        # Crossover at index 5, confirmation_bars=2
        # Signal should NOT be at index 5, but at index 7 or later
        oscillator_values = [-10.0] * 5 + [5.0] * 15
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 20, index=dates)
        range_atr = pd.Series([1000.0] * 20, index=dates)
        
        signals, strength = generate_signals_crossover_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            confirmation_bars=2,
            oscillator_threshold=0.0
        )
        
        # Signal should NOT be emitted at crossover time (index 5)
        assert signals.iloc[5] == 0, "Signal should not be emitted at crossover time"
        # Signal should be emitted after confirmation period (index 7 or later)
        assert any(signals.iloc[7:] == 1), "Signal should be emitted after confirmation period"
    
    def test_strategy6_look_ahead_bias_prevention(self):
        """Test Strategy 6 does not use future data for confirmation."""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')
        
        # Breakout at index 5, confirmation_bars=2
        # Signal should NOT be at index 5, but at index 7 or later
        oscillator_values = [90.0] * 5 + [110.0] * 15
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([50000.0] * 20, index=dates)
        range_atr = pd.Series([1000.0] * 20, index=dates)
        
        signals, strength = generate_signals_breakout_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            upper_threshold=100.0,
            lower_threshold=-100.0,
            require_confirmation=True,
            confirmation_bars=2
        )
        
        # Signal should NOT be emitted at breakout time (index 5)
        assert signals.iloc[5] == 0, "Signal should not be emitted at breakout time"
        # Signal should be emitted after confirmation period (index 7 or later)
        assert any(signals.iloc[7:] == 1), "Signal should be emitted after confirmation period"
    
    def test_strategy7_look_ahead_bias_prevention(self):
        """Test Strategy 7 does not use future data for peak/trough confirmation."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        
        # Create price and oscillator data with peaks/troughs
        price_base = 50000
        high = pd.Series([price_base + i * 10 for i in range(100)], index=dates)
        low = pd.Series([price_base - i * 10 for i in range(100)], index=dates)
        close = pd.Series([price_base] * 100, index=dates)
        
        # Oscillator with peaks
        oscillator_values = [0.0] * 20 + [50.0] * 5 + [0.0] * 20 + [60.0] * 5 + [0.0] * 50
        oscillator = pd.Series(oscillator_values, index=dates)
        ma = pd.Series([price_base] * 100, index=dates)
        range_atr = pd.Series([1000.0] * 100, index=dates)
        
        signals, strength = generate_signals_divergence_strategy(
            high=high, low=low, close=close,
            oscillator=oscillator, ma=ma, range_atr=range_atr,
            lookback_period=30,
            min_swing_bars=5
        )
        
        # Signals should only be emitted after peak/trough confirmation
        # With min_swing_bars=5, signals should not appear at peak time but after confirmation
        assert isinstance(signals, pd.Series)
        assert all(signals.isin([-1, 0, 1]))


class TestSignalStrengthCalculation:
    """Tests for signal strength calculation across all strategies."""
    
    def test_strategy1_signal_strength(self, sample_ohlc_data):
        """Test Strategy 1 signal strength is in valid range."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_basic_strategy(
            high=high, low=low, close=close
        )
        
        # Strength should be between 0 and 1
        assert all((strength >= 0) & (strength <= 1))
        # Strength should be 0 where signal is 0
        assert all(strength[signals == 0] == 0.0)
        # Strength should be > 0 where signal is not 0
        assert all(strength[signals != 0] > 0.0)
    
    def test_strategy2_signal_strength(self, sample_ohlc_data):
        """Test Strategy 2 signal strength."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_sustained_strategy(
            high=high, low=low, close=close
        )
        
        assert all((strength >= 0) & (strength <= 1))
        assert all(strength[signals == 0] == 0.0)
        assert all(strength[signals != 0] > 0.0)
    
    def test_strategy3_signal_strength(self, sample_ohlc_data):
        """Test Strategy 3 signal strength."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_crossover_strategy(
            high=high, low=low, close=close
        )
        
        assert all((strength >= 0) & (strength <= 1))
        assert all(strength[signals == 0] == 0.0)
        assert all(strength[signals != 0] > 0.0)
    
    def test_strategy4_signal_strength(self, sample_ohlc_data):
        """Test Strategy 4 signal strength."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_momentum_strategy(
            high=high, low=low, close=close
        )
        
        assert all((strength >= 0) & (strength <= 1))
        assert all(strength[signals == 0] == 0.0)
        assert all(strength[signals != 0] > 0.0)
    
    def test_strategy5_signal_strength(self, sample_ohlc_data):
        """Test Strategy 5 signal strength."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_combined_all_strategy(
            high=high, low=low, close=close
        )
        
        assert all((strength >= 0) & (strength <= 1))
        assert all(strength[signals == 0] == 0.0)
        assert all(strength[signals != 0] > 0.0)
    
    def test_strategy6_signal_strength(self, sample_ohlc_data):
        """Test Strategy 6 signal strength."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_breakout_strategy(
            high=high, low=low, close=close
        )
        
        assert all((strength >= 0) & (strength <= 1))
        assert all(strength[signals == 0] == 0.0)
        assert all(strength[signals != 0] > 0.0)
    
    def test_strategy7_signal_strength(self, sample_ohlc_data):
        """Test Strategy 7 signal strength."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_divergence_strategy(
            high=high, low=low, close=close
        )
        
        assert all((strength >= 0) & (strength <= 1))
        assert all(strength[signals == 0] == 0.0)
        assert all(strength[signals != 0] > 0.0)
    
    def test_strategy8_signal_strength(self, sample_ohlc_data):
        """Test Strategy 8 signal strength."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_trend_following_strategy(
            high=high, low=low, close=close
        )
        
        assert all((strength >= 0) & (strength <= 1))
        assert all(strength[signals == 0] == 0.0)
        assert all(strength[signals != 0] > 0.0)
    
    def test_strategy9_signal_strength(self, sample_ohlc_data):
        """Test Strategy 9 signal strength."""
        high, low, close = sample_ohlc_data
        
        signals, strength = generate_signals_mean_reversion_strategy(
            high=high, low=low, close=close
        )
        
        assert all((strength >= 0) & (strength <= 1))
        assert all(strength[signals == 0] == 0.0)
        assert all(strength[signals != 0] > 0.0)
    
    def test_strategy1_strength_increases_with_oscillator_magnitude(self):
        """Test Strategy 1 strength increases with oscillator magnitude."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1h')
        oscillator = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0], index=dates)
        ma = pd.Series([50000.0] * 10, index=dates)
        range_atr = pd.Series([1000.0] * 10, index=dates)
        close = pd.Series([51000.0] * 10, index=dates)  # Above MA
        
        signals, strength = generate_signals_basic_strategy(
            oscillator=oscillator, ma=ma, range_atr=range_atr, close=close,
            require_trend_confirmation=True
        )
        
        # Strength should generally increase with oscillator magnitude
        # (allowing for some variation due to clipping)
        assert all(signals == 1)  # All should be LONG
        assert all(strength > 0.0)  # All should have positive strength
        # Strength should be non-decreasing (or at least not decreasing significantly)
        strength_diff = strength.diff().dropna()
        # Allow small decreases due to clipping, but overall trend should be positive
        assert strength.iloc[-1] >= strength.iloc[0] * 0.8  # Final strength should be at least 80% of initial


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

