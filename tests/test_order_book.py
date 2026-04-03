"""Tests for microstructure/order_book.py (MICRO-002).

Covers OrderBookAnalyzer:
  - Quote ingestion and imbalance computation
  - Rolling imbalance smoothing
  - Imbalance trend (slope)
  - Spread computation
  - Edge cases: invalid quotes, zero sizes, empty history
  - Per-symbol tracking and reset
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from microstructure.order_book import OrderBookAnalyzer


class TestOrderBookAnalyzerInit:
    """Test constructor."""

    def test_default_init(self):
        a = OrderBookAnalyzer()
        assert a.quote_count == 0

    def test_invalid_rolling_window(self):
        with pytest.raises(ValueError, match="rolling_window must be >= 1"):
            OrderBookAnalyzer(rolling_window=0)


class TestQuoteIngestion:
    """Test update_quote and raw imbalance."""

    def test_basic_imbalance_positive(self):
        a = OrderBookAnalyzer()
        imb = a.update_quote(bid=150.10, ask=150.20, bid_size=500, ask_size=200)
        expected = (500 - 200) / (500 + 200)
        assert imb == pytest.approx(expected)

    def test_basic_imbalance_negative(self):
        a = OrderBookAnalyzer()
        imb = a.update_quote(bid=150.10, ask=150.20, bid_size=200, ask_size=500)
        expected = (200 - 500) / (200 + 500)
        assert imb == pytest.approx(expected)

    def test_balanced_imbalance(self):
        a = OrderBookAnalyzer()
        imb = a.update_quote(bid=150.10, ask=150.20, bid_size=300, ask_size=300)
        assert imb == pytest.approx(0.0)

    def test_zero_sizes_imbalance(self):
        a = OrderBookAnalyzer()
        imb = a.update_quote(bid=150.10, ask=150.20, bid_size=0, ask_size=0)
        assert imb == 0.0

    def test_invalid_bid_returns_previous(self):
        a = OrderBookAnalyzer()
        a.update_quote(bid=150.10, ask=150.20, bid_size=500, ask_size=200)
        imb = a.update_quote(bid=-1, ask=150.20, bid_size=500, ask_size=200)
        # Should return previous imbalance, not crash
        assert isinstance(imb, float)

    def test_invalid_ask_returns_previous(self):
        a = OrderBookAnalyzer()
        imb = a.update_quote(bid=150.10, ask=0, bid_size=500, ask_size=200)
        assert isinstance(imb, float)

    def test_negative_sizes_returns_previous(self):
        a = OrderBookAnalyzer()
        imb = a.update_quote(bid=150.10, ask=150.20, bid_size=-100, ask_size=200)
        assert isinstance(imb, float)

    def test_custom_timestamp(self):
        a = OrderBookAnalyzer()
        ts = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        a.update_quote(bid=150.10, ask=150.20, bid_size=500, ask_size=200,
                       timestamp=ts)
        assert a.quote_count == 1

    def test_per_symbol_tracking(self):
        a = OrderBookAnalyzer()
        a.update_quote(bid=150.10, ask=150.20, bid_size=500, ask_size=200, symbol="AAPL")
        a.update_quote(bid=100.10, ask=100.20, bid_size=200, ask_size=500, symbol="GOOG")
        assert a.get_imbalance(symbol="AAPL") > 0
        assert a.get_imbalance(symbol="GOOG") < 0


class TestImbalanceQueries:
    """Test imbalance query methods."""

    def test_get_imbalance_empty(self):
        a = OrderBookAnalyzer()
        assert a.get_imbalance() == 0.0

    def test_get_imbalance_returns_latest(self):
        a = OrderBookAnalyzer()
        a.update_quote(bid=150.10, ask=150.20, bid_size=500, ask_size=200)
        a.update_quote(bid=150.10, ask=150.20, bid_size=200, ask_size=500)
        imb = a.get_imbalance()
        expected = (200 - 500) / (200 + 500)
        assert imb == pytest.approx(expected)

    def test_get_imbalance_unknown_symbol(self):
        a = OrderBookAnalyzer()
        # Falls back to global history
        assert a.get_imbalance(symbol="UNKNOWN") == 0.0

    def test_rolling_imbalance_single(self):
        a = OrderBookAnalyzer(rolling_window=5)
        a.update_quote(bid=150.10, ask=150.20, bid_size=500, ask_size=200)
        rolling = a.get_rolling_imbalance()
        assert rolling == pytest.approx(a.get_imbalance())

    def test_rolling_imbalance_smoothed(self):
        a = OrderBookAnalyzer(rolling_window=3)
        a.update_quote(bid=150.10, ask=150.20, bid_size=500, ask_size=200)
        a.update_quote(bid=150.10, ask=150.20, bid_size=200, ask_size=500)
        a.update_quote(bid=150.10, ask=150.20, bid_size=300, ask_size=300)
        rolling = a.get_rolling_imbalance()
        # Average of 3 imbalances
        imb1 = (500 - 200) / 700
        imb2 = (200 - 500) / 700
        imb3 = 0.0
        assert rolling == pytest.approx(np.mean([imb1, imb2, imb3]), abs=1e-9)

    def test_rolling_imbalance_empty(self):
        a = OrderBookAnalyzer()
        assert a.get_rolling_imbalance() == 0.0


class TestImbalanceTrend:
    """Test trend slope computation."""

    def test_trend_insufficient_data(self):
        a = OrderBookAnalyzer()
        assert a.get_imbalance_trend() == 0.0
        a.update_quote(bid=150.10, ask=150.20, bid_size=500, ask_size=200)
        assert a.get_imbalance_trend() == 0.0

    def test_trend_positive(self):
        a = OrderBookAnalyzer(rolling_window=10)
        for i in range(10):
            a.update_quote(bid=150.10, ask=150.20,
                           bid_size=100 + i * 50, ask_size=100)
        trend = a.get_imbalance_trend()
        assert trend > 0

    def test_trend_negative(self):
        a = OrderBookAnalyzer(rolling_window=10)
        for i in range(10):
            a.update_quote(bid=150.10, ask=150.20,
                           bid_size=100, ask_size=100 + i * 50)
        trend = a.get_imbalance_trend()
        assert trend < 0


class TestSpread:
    """Test spread computation."""

    def test_spread_empty(self):
        a = OrderBookAnalyzer()
        assert a.get_spread() == 0.0

    def test_spread_normal(self):
        a = OrderBookAnalyzer()
        a.update_quote(bid=150.00, ask=150.10, bid_size=100, ask_size=100)
        spread = a.get_spread()
        midpoint = (150.00 + 150.10) / 2
        expected = (150.10 - 150.00) / midpoint
        assert spread == pytest.approx(expected)

    def test_spread_zero(self):
        """Bid == Ask -> spread = 0."""
        a = OrderBookAnalyzer()
        a.update_quote(bid=150.0, ask=150.0, bid_size=100, ask_size=100)
        assert a.get_spread() == pytest.approx(0.0)


class TestOrderBookAnalyzerHousekeeping:
    """Test reset and repr."""

    def test_reset_all(self):
        a = OrderBookAnalyzer()
        a.update_quote(bid=150.10, ask=150.20, bid_size=500, ask_size=200)
        a.reset()
        assert a.quote_count == 0
        assert a.get_imbalance() == 0.0

    def test_reset_symbol(self):
        a = OrderBookAnalyzer()
        a.update_quote(bid=150.10, ask=150.20, bid_size=500, ask_size=200, symbol="AAPL")
        a.reset(symbol="AAPL")
        # Global history still has data, but per-symbol cleared
        assert a.get_imbalance(symbol="AAPL") == a.get_imbalance()

    def test_repr(self):
        a = OrderBookAnalyzer()
        r = repr(a)
        assert "OrderBookAnalyzer" in r
