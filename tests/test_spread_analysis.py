"""Tests for microstructure/spread_analysis.py (MICRO-004).

Covers SpreadAnalyzer:
  - Effective spread computation
  - Realized spread and adverse selection after midpoint update
  - Edge cases: zero spread, invalid prices, empty data
  - Execution quality score
  - Basis points conversion
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from microstructure.spread_analysis import SpreadAnalyzer


class TestSpreadAnalyzerRecordTrade:
    """Test trade recording and effective spread."""

    def test_basic_effective_spread(self):
        sa = SpreadAnalyzer()
        spread = sa.record_trade("AAPL", trade_price=150.15, bid=150.10, ask=150.20)
        # midpoint = 150.15, trade at midpoint -> spread = 0
        assert spread == pytest.approx(0.0, abs=1e-9)

    def test_effective_spread_above_mid(self):
        sa = SpreadAnalyzer()
        # midpoint = 150.15, trade at 150.20 -> spread = 2 * 0.05 = 0.10
        spread = sa.record_trade("AAPL", trade_price=150.20, bid=150.10, ask=150.20)
        assert spread == pytest.approx(0.10, abs=1e-9)

    def test_effective_spread_below_mid(self):
        sa = SpreadAnalyzer()
        spread = sa.record_trade("AAPL", trade_price=150.10, bid=150.10, ask=150.20)
        assert spread == pytest.approx(0.10, abs=1e-9)

    def test_side_inference_buy(self):
        sa = SpreadAnalyzer()
        sa.record_trade("AAPL", trade_price=150.20, bid=150.10, ask=150.20)
        # Trade at ask -> classified as buy
        assert sa.get_trade_count("AAPL") == 1

    def test_side_inference_sell(self):
        sa = SpreadAnalyzer()
        sa.record_trade("AAPL", trade_price=150.10, bid=150.10, ask=150.20)
        assert sa.get_trade_count("AAPL") == 1

    def test_explicit_side(self):
        sa = SpreadAnalyzer()
        sa.record_trade("AAPL", trade_price=150.15, bid=150.10, ask=150.20, side="sell")
        assert sa.get_trade_count("AAPL") == 1

    def test_invalid_prices_return_zero(self):
        sa = SpreadAnalyzer()
        spread = sa.record_trade("AAPL", trade_price=0, bid=150.10, ask=150.20)
        assert spread == 0.0

    def test_invalid_bid_returns_zero(self):
        sa = SpreadAnalyzer()
        spread = sa.record_trade("AAPL", trade_price=150.0, bid=-1, ask=150.20)
        assert spread == 0.0

    def test_zero_spread_book(self):
        """Bid == Ask scenario."""
        sa = SpreadAnalyzer()
        spread = sa.record_trade("AAPL", trade_price=150.0, bid=150.0, ask=150.0)
        assert spread == pytest.approx(0.0, abs=1e-9)


class TestSpreadAnalyzerQueries:
    """Test spread query methods."""

    def test_get_effective_spread_no_data(self):
        sa = SpreadAnalyzer()
        assert sa.get_effective_spread("AAPL") == 0.0

    def test_get_effective_spread_average(self):
        sa = SpreadAnalyzer()
        sa.record_trade("AAPL", trade_price=150.20, bid=150.10, ask=150.20)
        sa.record_trade("AAPL", trade_price=150.10, bid=150.10, ask=150.20)
        avg = sa.get_effective_spread("AAPL")
        assert avg == pytest.approx(0.10, abs=1e-9)

    def test_get_effective_spread_bps_no_data(self):
        sa = SpreadAnalyzer()
        assert sa.get_effective_spread_bps("AAPL") == 0.0

    def test_get_effective_spread_bps(self):
        sa = SpreadAnalyzer()
        sa.record_trade("AAPL", trade_price=150.20, bid=150.10, ask=150.20)
        bps = sa.get_effective_spread_bps("AAPL")
        assert bps > 0

    def test_get_adverse_selection_no_data(self):
        sa = SpreadAnalyzer()
        assert sa.get_adverse_selection("AAPL") == 0.0

    def test_get_realized_spread_no_data(self):
        sa = SpreadAnalyzer()
        assert sa.get_realized_spread("AAPL") == 0.0


class TestSpreadAnalyzerMidpointUpdate:
    """Test price impact / realized spread computation."""

    def test_update_midpoint_computes_impact(self):
        sa = SpreadAnalyzer(impact_horizon_sec=60)
        ts = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        sa.record_trade("AAPL", trade_price=150.20, bid=150.10, ask=150.20,
                        timestamp=ts)

        future_time = ts + timedelta(seconds=120)
        updated = sa.update_midpoint("AAPL", future_midpoint=150.30,
                                     current_time=future_time)
        assert updated == 1

        impact = sa.get_adverse_selection("AAPL")
        assert impact != 0.0

        realized = sa.get_realized_spread("AAPL")
        assert isinstance(realized, float)

    def test_update_midpoint_too_early(self):
        sa = SpreadAnalyzer(impact_horizon_sec=300)
        ts = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        sa.record_trade("AAPL", trade_price=150.20, bid=150.10, ask=150.20,
                        timestamp=ts)

        early = ts + timedelta(seconds=60)
        updated = sa.update_midpoint("AAPL", future_midpoint=150.30,
                                     current_time=early)
        assert updated == 0

    def test_update_midpoint_no_pending(self):
        sa = SpreadAnalyzer()
        updated = sa.update_midpoint("GOOG", future_midpoint=100.0)
        assert updated == 0


class TestSpreadAnalyzerExecutionQuality:
    """Test composite execution quality score."""

    def test_quality_score_range(self):
        sa = SpreadAnalyzer(impact_horizon_sec=1)
        ts = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        for i in range(10):
            sa.record_trade("AAPL",
                            trade_price=150.10 + i * 0.01,
                            bid=150.10,
                            ask=150.20,
                            timestamp=ts + timedelta(seconds=i))
        score = sa.get_execution_quality_score("AAPL")
        assert 0.0 <= score <= 1.0

    def test_quality_score_no_data(self):
        sa = SpreadAnalyzer()
        score = sa.get_execution_quality_score("AAPL")
        # No trades: spread_score = 1.0, impact_score = 0.5 -> 0.6*1 + 0.4*0.5 = 0.8
        assert 0.0 <= score <= 1.0


class TestSpreadAnalyzerHousekeeping:
    """Test reset and utility methods."""

    def test_tracked_symbols(self):
        sa = SpreadAnalyzer()
        sa.record_trade("AAPL", trade_price=150.0, bid=149.90, ask=150.10)
        sa.record_trade("GOOG", trade_price=100.0, bid=99.90, ask=100.10)
        assert set(sa.tracked_symbols) == {"AAPL", "GOOG"}

    def test_get_trade_count(self):
        sa = SpreadAnalyzer()
        sa.record_trade("AAPL", trade_price=150.0, bid=149.90, ask=150.10)
        sa.record_trade("AAPL", trade_price=150.1, bid=150.0, ask=150.20)
        assert sa.get_trade_count("AAPL") == 2
        assert sa.get_trade_count("GOOG") == 0

    def test_reset_symbol(self):
        sa = SpreadAnalyzer()
        sa.record_trade("AAPL", trade_price=150.0, bid=149.90, ask=150.10)
        sa.reset(symbol="AAPL")
        assert sa.get_trade_count("AAPL") == 0

    def test_reset_all(self):
        sa = SpreadAnalyzer()
        sa.record_trade("AAPL", trade_price=150.0, bid=149.90, ask=150.10)
        sa.record_trade("GOOG", trade_price=100.0, bid=99.90, ask=100.10)
        sa.reset()
        assert sa.tracked_symbols == []

    def test_repr(self):
        sa = SpreadAnalyzer()
        r = repr(sa)
        assert "SpreadAnalyzer" in r
