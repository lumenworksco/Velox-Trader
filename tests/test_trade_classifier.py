"""Tests for microstructure/trade_classifier.py (MICRO-003).

Covers TradeClassifier:
  - Trade size classification thresholds
  - Institutional and retail flow tracking
  - Flow divergence detection
  - Volume breakdown
  - Edge cases: zero size, negative price, empty state
"""

from datetime import datetime, timezone

import pytest

from microstructure.trade_classifier import TradeClassifier, TradeType


class TestTradeClassification:
    """Test classify_trade thresholds."""

    def test_small_trade(self):
        tc = TradeClassifier()
        assert tc.classify_trade(size=50, price=100.0) == TradeType.SMALL

    def test_medium_trade(self):
        tc = TradeClassifier()
        assert tc.classify_trade(size=500, price=100.0) == TradeType.MEDIUM

    def test_large_trade_by_shares(self):
        tc = TradeClassifier()
        assert tc.classify_trade(size=1500, price=10.0) == TradeType.LARGE

    def test_large_trade_by_notional(self):
        """Even 200 shares at $300 = $60k > $50k threshold."""
        tc = TradeClassifier()
        assert tc.classify_trade(size=200, price=300.0) == TradeType.LARGE

    def test_exact_small_boundary(self):
        tc = TradeClassifier()
        assert tc.classify_trade(size=99, price=100.0) == TradeType.SMALL
        assert tc.classify_trade(size=100, price=10.0) == TradeType.MEDIUM

    def test_exact_large_boundary(self):
        tc = TradeClassifier()
        assert tc.classify_trade(size=999, price=10.0) == TradeType.MEDIUM
        assert tc.classify_trade(size=1000, price=10.0) == TradeType.LARGE

    def test_zero_size(self):
        tc = TradeClassifier()
        assert tc.classify_trade(size=0, price=100.0) == TradeType.SMALL

    def test_negative_size(self):
        tc = TradeClassifier()
        assert tc.classify_trade(size=-10, price=100.0) == TradeType.SMALL

    def test_zero_price(self):
        tc = TradeClassifier()
        assert tc.classify_trade(size=500, price=0.0) == TradeType.SMALL

    def test_custom_thresholds(self):
        tc = TradeClassifier(small_threshold=50, large_threshold=500,
                             large_notional_threshold=25000.0)
        assert tc.classify_trade(size=30, price=100.0) == TradeType.SMALL
        assert tc.classify_trade(size=100, price=100.0) == TradeType.MEDIUM
        assert tc.classify_trade(size=600, price=10.0) == TradeType.LARGE


class TestTradeRecording:
    """Test record_trade and flow tracking."""

    def test_record_trade_returns_type(self):
        tc = TradeClassifier()
        result = tc.record_trade("AAPL", size=5000, price=150.0, side="buy")
        assert result == TradeType.LARGE

    def test_record_trade_zero_size(self):
        tc = TradeClassifier()
        result = tc.record_trade("AAPL", size=0, price=150.0, side="buy")
        assert result == TradeType.SMALL

    def test_record_trade_tracks_symbol(self):
        tc = TradeClassifier()
        tc.record_trade("AAPL", size=500, price=150.0, side="buy")
        assert "AAPL" in tc.tracked_symbols


class TestInstitutionalFlow:
    """Test institutional flow metrics."""

    def test_institutional_flow_all_buys(self):
        tc = TradeClassifier()
        tc.record_trade("AAPL", size=5000, price=150.0, side="buy")
        tc.record_trade("AAPL", size=3000, price=150.0, side="buy")
        flow = tc.get_institutional_flow("AAPL")
        assert flow == pytest.approx(1.0)

    def test_institutional_flow_all_sells(self):
        tc = TradeClassifier()
        tc.record_trade("AAPL", size=5000, price=150.0, side="sell")
        flow = tc.get_institutional_flow("AAPL")
        assert flow == pytest.approx(-1.0)

    def test_institutional_flow_balanced(self):
        tc = TradeClassifier()
        tc.record_trade("AAPL", size=5000, price=150.0, side="buy")
        tc.record_trade("AAPL", size=5000, price=150.0, side="sell")
        flow = tc.get_institutional_flow("AAPL")
        assert flow == pytest.approx(0.0)

    def test_institutional_flow_no_large_trades(self):
        tc = TradeClassifier()
        tc.record_trade("AAPL", size=50, price=10.0, side="buy")
        flow = tc.get_institutional_flow("AAPL")
        assert flow == 0.0

    def test_institutional_flow_unknown_symbol(self):
        tc = TradeClassifier()
        assert tc.get_institutional_flow("UNKNOWN") == 0.0

    def test_institutional_flow_range(self):
        tc = TradeClassifier()
        tc.record_trade("AAPL", size=5000, price=150.0, side="buy")
        tc.record_trade("AAPL", size=2000, price=150.0, side="sell")
        flow = tc.get_institutional_flow("AAPL")
        assert -1.0 <= flow <= 1.0


class TestRetailFlow:
    """Test retail flow metrics."""

    def test_retail_flow_all_buys(self):
        tc = TradeClassifier()
        tc.record_trade("AAPL", size=50, price=10.0, side="buy")
        tc.record_trade("AAPL", size=30, price=10.0, side="buy")
        flow = tc.get_retail_flow("AAPL")
        assert flow == pytest.approx(1.0)

    def test_retail_flow_no_data(self):
        tc = TradeClassifier()
        assert tc.get_retail_flow("AAPL") == 0.0


class TestFlowDivergence:
    """Test divergence between institutional and retail flow."""

    def test_bullish_divergence(self):
        tc = TradeClassifier()
        # Institutions buying, retail selling
        tc.record_trade("AAPL", size=5000, price=150.0, side="buy")
        tc.record_trade("AAPL", size=50, price=10.0, side="sell")
        div = tc.get_flow_divergence("AAPL")
        assert div > 0  # bullish

    def test_bearish_divergence(self):
        tc = TradeClassifier()
        tc.record_trade("AAPL", size=5000, price=150.0, side="sell")
        tc.record_trade("AAPL", size=50, price=10.0, side="buy")
        div = tc.get_flow_divergence("AAPL")
        assert div < 0  # bearish

    def test_divergence_range(self):
        tc = TradeClassifier()
        tc.record_trade("AAPL", size=5000, price=150.0, side="buy")
        tc.record_trade("AAPL", size=50, price=10.0, side="sell")
        div = tc.get_flow_divergence("AAPL")
        assert -2.0 <= div <= 2.0


class TestVolumeBreakdown:
    """Test volume breakdown reporting."""

    def test_volume_breakdown_empty(self):
        tc = TradeClassifier()
        bd = tc.get_volume_breakdown("AAPL")
        assert bd["small_volume"] == 0
        assert bd["medium_volume"] == 0
        assert bd["large_volume"] == 0
        assert bd["total_volume"] == 0

    def test_volume_breakdown_mixed(self):
        tc = TradeClassifier()
        tc.record_trade("AAPL", size=50, price=10.0, side="buy")     # small
        tc.record_trade("AAPL", size=500, price=10.0, side="buy")    # medium
        tc.record_trade("AAPL", size=5000, price=10.0, side="buy")   # large
        bd = tc.get_volume_breakdown("AAPL")
        assert bd["small_volume"] == 50
        assert bd["medium_volume"] == 500
        assert bd["large_volume"] == 5000
        assert bd["total_volume"] == 5550

    def test_institutional_participation_rate(self):
        tc = TradeClassifier()
        tc.record_trade("AAPL", size=5000, price=10.0, side="buy")
        tc.record_trade("AAPL", size=5000, price=10.0, side="sell")
        rate = tc.get_institutional_participation_rate("AAPL")
        assert rate == pytest.approx(1.0)

    def test_institutional_participation_rate_zero(self):
        tc = TradeClassifier()
        tc.record_trade("AAPL", size=50, price=10.0, side="buy")
        rate = tc.get_institutional_participation_rate("AAPL")
        assert rate == pytest.approx(0.0)

    def test_institutional_participation_empty(self):
        tc = TradeClassifier()
        assert tc.get_institutional_participation_rate("AAPL") == 0.0


class TestTradeClassifierHousekeeping:
    """Test reset and repr."""

    def test_reset_symbol(self):
        tc = TradeClassifier()
        tc.record_trade("AAPL", size=500, price=100.0, side="buy")
        tc.reset(symbol="AAPL")
        assert "AAPL" not in tc.tracked_symbols

    def test_reset_all(self):
        tc = TradeClassifier()
        tc.record_trade("AAPL", size=500, price=100.0, side="buy")
        tc.record_trade("GOOG", size=500, price=100.0, side="buy")
        tc.reset()
        assert tc.tracked_symbols == []

    def test_repr(self):
        tc = TradeClassifier()
        r = repr(tc)
        assert "TradeClassifier" in r
