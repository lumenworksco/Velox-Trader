"""Tests for microstructure/dark_pool.py (T5-006).

Covers DarkPoolDetector:
  - Trade ingestion with dark pool condition codes
  - Signal computation (ratio, net buy, alpha)
  - Confidence multiplier
  - Edge cases: disabled detector, no trades, no dark trades
  - Pruning old trades
"""

import time
from unittest.mock import patch, MagicMock

import pytest

# We must mock config before importing dark_pool
import sys
_mock_config = MagicMock()
_mock_config.DARK_POOL_ROLLING_MINUTES = 30
_mock_config.DARK_POOL_RATIO_THRESHOLD = 0.35
_mock_config.DARK_POOL_ALPHA_WEIGHT = 0.15
_mock_config.DARK_POOL_ENABLED = True
sys.modules.setdefault("config", _mock_config)

from microstructure.dark_pool import DarkPoolDetector, DarkPoolSignal, DarkPoolTrade


class TestDarkPoolTradeIngestion:
    """Test add_trade and condition code detection."""

    def test_add_regular_trade(self):
        d = DarkPoolDetector()
        d.add_trade("AAPL", price=185.50, volume=500, condition="", side="buy")
        assert "AAPL" in d.tracked_symbols

    def test_add_dark_pool_trade(self):
        d = DarkPoolDetector()
        d.add_trade("AAPL", price=185.50, volume=500, condition="D", side="buy")
        signal = d.get_signal("AAPL")
        assert signal.total_dark_volume == 500

    def test_multiple_condition_codes(self):
        d = DarkPoolDetector()
        d.add_trade("AAPL", price=185.50, volume=500, condition="D,X", side="buy")
        signal = d.get_signal("AAPL")
        assert signal.total_dark_volume == 500

    def test_zero_volume_ignored(self):
        d = DarkPoolDetector()
        d.add_trade("AAPL", price=185.50, volume=0, condition="D", side="buy")
        assert "AAPL" not in d.tracked_symbols

    def test_tick_rule_classification(self):
        """Side inferred via tick rule when not provided."""
        d = DarkPoolDetector()
        d.add_trade("AAPL", price=100.0, volume=100, condition="D")   # buy (no prev)
        d.add_trade("AAPL", price=99.0, volume=100, condition="D")    # sell (downtick)
        d.add_trade("AAPL", price=100.0, volume=100, condition="D")   # buy (uptick)
        signal = d.get_signal("AAPL")
        assert signal.total_dark_volume == 300

    def test_custom_timestamp(self):
        d = DarkPoolDetector()
        ts = time.time()
        d.add_trade("AAPL", price=185.50, volume=500, condition="D",
                     side="buy", timestamp=ts)
        assert "AAPL" in d.tracked_symbols


class TestDarkPoolSignal:
    """Test get_signal computation."""

    def test_signal_no_trades(self):
        d = DarkPoolDetector()
        sig = d.get_signal("AAPL")
        assert sig.dark_pool_ratio == 0.0
        assert sig.is_significant is False
        assert sig.confidence_mult == 1.0

    def test_signal_all_dark(self):
        d = DarkPoolDetector()
        ts = time.time()
        for i in range(10):
            d.add_trade("AAPL", price=185.50, volume=100, condition="D",
                        side="buy", timestamp=ts + i)
        sig = d.get_signal("AAPL")
        assert sig.dark_pool_ratio == pytest.approx(1.0)
        assert sig.total_dark_volume == 1000
        assert sig.total_volume == 1000

    def test_signal_no_dark(self):
        d = DarkPoolDetector()
        ts = time.time()
        for i in range(10):
            d.add_trade("AAPL", price=185.50, volume=100, condition="",
                        side="buy", timestamp=ts + i)
        sig = d.get_signal("AAPL")
        assert sig.dark_pool_ratio == 0.0
        assert sig.is_significant is False

    def test_signal_mixed_ratio(self):
        d = DarkPoolDetector()
        ts = time.time()
        # 4 dark out of 10 -> ratio = 0.4 > 0.35 threshold
        for i in range(10):
            cond = "D" if i < 4 else ""
            d.add_trade("AAPL", price=185.50, volume=100, condition=cond,
                        side="buy", timestamp=ts + i)
        sig = d.get_signal("AAPL")
        assert sig.dark_pool_ratio == pytest.approx(0.4, abs=0.01)

    def test_significance_requires_direction(self):
        """Need both ratio > threshold and |net_buy_ratio| > 0.1."""
        d = DarkPoolDetector()
        ts = time.time()
        # 5 dark buys, 5 dark sells -> net_buy_ratio = 0, not significant
        for i in range(10):
            side = "buy" if i < 5 else "sell"
            d.add_trade("AAPL", price=185.50, volume=100, condition="D",
                        side=side, timestamp=ts + i)
        sig = d.get_signal("AAPL")
        assert sig.is_significant is False

    def test_alpha_signal_positive(self):
        d = DarkPoolDetector()
        ts = time.time()
        # All dark, all buys -> high ratio, high net buy
        for i in range(10):
            d.add_trade("AAPL", price=185.50, volume=100, condition="D",
                        side="buy", timestamp=ts + i)
        sig = d.get_signal("AAPL")
        assert sig.alpha_signal > 0

    def test_alpha_signal_negative(self):
        d = DarkPoolDetector()
        ts = time.time()
        for i in range(10):
            d.add_trade("AAPL", price=185.50, volume=100, condition="D",
                        side="sell", timestamp=ts + i)
        sig = d.get_signal("AAPL")
        assert sig.alpha_signal < 0

    def test_signal_output_format(self):
        d = DarkPoolDetector()
        sig = d.get_signal("AAPL")
        assert isinstance(sig, DarkPoolSignal)
        assert isinstance(sig.symbol, str)
        assert isinstance(sig.dark_pool_ratio, float)
        assert isinstance(sig.net_buy_ratio, float)
        assert isinstance(sig.confidence_mult, float)
        assert isinstance(sig.is_significant, bool)


class TestDarkPoolConfidenceMultiplier:
    """Test confidence multiplier clamping."""

    def test_confidence_mult_clamped_range(self):
        d = DarkPoolDetector()
        ts = time.time()
        for i in range(10):
            d.add_trade("AAPL", price=185.50, volume=100, condition="D",
                        side="buy", timestamp=ts + i)
        mult = d.get_confidence_multiplier("AAPL")
        assert 0.6 <= mult <= 1.4

    def test_confidence_mult_no_data(self):
        d = DarkPoolDetector()
        assert d.get_confidence_multiplier("AAPL") == 1.0


class TestDarkPoolDisabled:
    """Test behavior when detector is disabled."""

    def test_disabled_ignores_trades(self):
        d = DarkPoolDetector()
        d._enabled = False
        d.add_trade("AAPL", price=185.50, volume=500, condition="D", side="buy")
        assert "AAPL" not in d.tracked_symbols

    def test_disabled_returns_neutral_signal(self):
        d = DarkPoolDetector()
        d._enabled = False
        sig = d.get_signal("AAPL")
        assert sig.dark_pool_ratio == 0.0
        assert sig.confidence_mult == 1.0
        assert sig.is_significant is False


class TestDarkPoolPruning:
    """Test old trade pruning."""

    def test_prune_removes_old_trades(self):
        d = DarkPoolDetector(rolling_minutes=1)
        old_ts = time.time() - 120  # 2 minutes ago
        d.add_trade("AAPL", price=185.50, volume=100, condition="D",
                     side="buy", timestamp=old_ts)
        pruned = d.prune_old_trades()
        assert pruned == 1
        assert "AAPL" not in d.tracked_symbols

    def test_prune_keeps_recent(self):
        d = DarkPoolDetector(rolling_minutes=30)
        ts = time.time()
        d.add_trade("AAPL", price=185.50, volume=100, condition="D",
                     side="buy", timestamp=ts)
        pruned = d.prune_old_trades()
        assert pruned == 0
        assert "AAPL" in d.tracked_symbols


class TestDarkPoolHousekeeping:
    """Test reset and tracked symbols."""

    def test_reset_symbol(self):
        d = DarkPoolDetector()
        d.add_trade("AAPL", price=100.0, volume=100, condition="D", side="buy",
                     timestamp=time.time())
        d.reset(symbol="AAPL")
        assert "AAPL" not in d.tracked_symbols

    def test_reset_all(self):
        d = DarkPoolDetector()
        ts = time.time()
        d.add_trade("AAPL", price=100.0, volume=100, condition="D",
                     side="buy", timestamp=ts)
        d.add_trade("GOOG", price=100.0, volume=100, condition="D",
                     side="buy", timestamp=ts)
        d.reset()
        assert d.tracked_symbols == []
