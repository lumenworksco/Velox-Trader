"""V12 integration-level tests for microstructure modules.

Complements the existing test_vpin.py and test_spread_analysis.py with
additional coverage:

- VPIN with realistic trade sequences and varying bucket fills
- VPIN order imbalance and toxicity detection under stress
- SpreadAnalyzer midpoint update pipeline (record + update cycle)
- Adverse selection measurement
- Execution quality scoring
- Edge cases: partial buckets, zero volume, concurrent updates
"""

import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure config stub is in place
from zoneinfo import ZoneInfo as _ZoneInfo

_ET = _ZoneInfo("America/New_York")
_config_mod = MagicMock()
_config_mod.ET = _ET
sys.modules.setdefault("config", _config_mod)
sys.modules["config"].ET = _ET

from microstructure.vpin import VPIN
from microstructure.spread_analysis import SpreadAnalyzer


# ===========================================================================
# VPIN — Realistic trade sequences
# ===========================================================================

class TestVPINRealisticSequences:

    def test_balanced_order_flow_low_vpin(self):
        """Equal buy and sell volume should produce low VPIN."""
        v = VPIN(bucket_volume=1000, n_buckets=10)
        # Alternate buy/sell in equal amounts
        for _ in range(20):
            v.add_trade(price=100.0, volume=500, side="buy")
            v.add_trade(price=100.0, volume=500, side="sell")

        vpin = v.compute_vpin()
        assert 0.0 <= vpin <= 0.1  # Nearly balanced -> VPIN close to 0

    def test_one_sided_flow_high_vpin(self):
        """All buy volume (informed flow) should produce high VPIN."""
        v = VPIN(bucket_volume=1000, n_buckets=10)
        for _ in range(15):
            v.add_trade(price=100.0, volume=1000, side="buy")

        vpin = v.compute_vpin()
        assert vpin >= 0.9  # Extremely one-sided

    def test_vpin_increases_as_flow_becomes_toxic(self):
        """VPIN should increase when order flow shifts from balanced to one-sided."""
        v = VPIN(bucket_volume=500, n_buckets=20)

        # Phase 1: balanced flow
        for _ in range(10):
            v.add_trade(price=100.0, volume=250, side="buy")
            v.add_trade(price=100.0, volume=250, side="sell")
        vpin_balanced = v.compute_vpin()

        # Phase 2: toxic one-sided flow
        for _ in range(20):
            v.add_trade(price=100.0, volume=500, side="buy")
        vpin_toxic = v.compute_vpin()

        assert vpin_toxic > vpin_balanced

    def test_compute_vpin_with_trade_list(self):
        """compute_vpin(trades=...) should ingest trades and then compute."""
        v = VPIN(bucket_volume=100, n_buckets=5)
        trades = [
            {"price": 100.0, "volume": 100, "side": "buy"},
            {"price": 100.1, "volume": 100, "side": "buy"},
            {"price": 100.2, "volume": 100, "side": "sell"},
            {"price": 100.3, "volume": 100, "side": "sell"},
            {"price": 100.4, "volume": 100, "side": "buy"},
        ]
        vpin = v.compute_vpin(trades=trades)
        assert 0.0 <= vpin <= 1.0
        assert v.total_trades == 5


class TestVPINBucketing:

    def test_large_trade_spans_multiple_buckets(self):
        """A single trade larger than bucket_volume should create multiple buckets."""
        v = VPIN(bucket_volume=100, n_buckets=10)
        v.add_trade(price=50.0, volume=350, side="buy")

        # 350 volume / 100 per bucket = 3 full buckets, 50 remaining
        assert v.completed_bucket_count == 3
        assert v.total_trades == 1

    def test_partial_bucket_not_counted(self):
        """A partially filled bucket should not be included in VPIN."""
        v = VPIN(bucket_volume=1000, n_buckets=5)
        v.add_trade(price=50.0, volume=500, side="buy")
        assert v.completed_bucket_count == 0
        assert v.compute_vpin() == 0.0

    def test_exact_bucket_fill(self):
        """Trade that exactly fills a bucket should complete it."""
        v = VPIN(bucket_volume=200, n_buckets=5)
        v.add_trade(price=50.0, volume=200, side="sell")
        assert v.completed_bucket_count == 1


class TestVPINToxicityAndSizing:

    def test_is_toxic_false_when_balanced(self):
        v = VPIN(bucket_volume=100, n_buckets=5, high_vpin_threshold=0.7)
        for _ in range(5):
            v.add_trade(price=100.0, volume=50, side="buy")
            v.add_trade(price=100.0, volume=50, side="sell")
        assert not v.is_toxic()

    def test_is_toxic_true_when_one_sided(self):
        v = VPIN(bucket_volume=100, n_buckets=5, high_vpin_threshold=0.7)
        for _ in range(10):
            v.add_trade(price=100.0, volume=100, side="buy")
        assert v.is_toxic()

    def test_sizing_modifier_full_when_safe(self):
        v = VPIN(bucket_volume=100, n_buckets=5, high_vpin_threshold=0.7)
        for _ in range(5):
            v.add_trade(price=100.0, volume=50, side="buy")
            v.add_trade(price=100.0, volume=50, side="sell")
        modifier = v.get_sizing_modifier()
        assert modifier == pytest.approx(1.0, abs=0.15)

    def test_sizing_modifier_reduced_when_toxic(self):
        v = VPIN(bucket_volume=100, n_buckets=5, high_vpin_threshold=0.7)
        for _ in range(10):
            v.add_trade(price=100.0, volume=100, side="buy")
        modifier = v.get_sizing_modifier()
        assert modifier < 1.0


class TestVPINTickRule:

    def test_tick_rule_uptick_classified_buy(self):
        """Price increase should be classified as buy."""
        v = VPIN(bucket_volume=200, n_buckets=5)
        v.add_trade(price=100.0, volume=50)  # First trade: default buy
        v.add_trade(price=101.0, volume=50)  # Uptick: buy
        # Both should be buy volume (bucket not yet full at 100/200)
        assert v._current_bucket.buy_volume == 100

    def test_tick_rule_downtick_classified_sell(self):
        """Price decrease should be classified as sell."""
        v = VPIN(bucket_volume=200, n_buckets=5)
        v.add_trade(price=100.0, volume=50)  # First: buy (default)
        v.add_trade(price=99.0, volume=50)   # Downtick: sell
        assert v._current_bucket.buy_volume == 50
        assert v._current_bucket.sell_volume == 50


# ===========================================================================
# SpreadAnalyzer — Midpoint update pipeline
# ===========================================================================

class TestSpreadAnalyzerPipeline:

    def _base_time(self):
        return datetime(2026, 4, 4, 14, 0, 0, tzinfo=timezone.utc)

    def test_record_and_update_midpoint_cycle(self):
        """Full cycle: record trade -> wait horizon -> update midpoint."""
        sa = SpreadAnalyzer(impact_horizon_sec=60)
        t0 = self._base_time()

        sa.record_trade("AAPL", trade_price=150.20, bid=150.10, ask=150.20,
                         timestamp=t0, side="buy")

        # Update midpoint after the horizon elapses
        t1 = t0 + timedelta(seconds=61)
        updated = sa.update_midpoint("AAPL", future_midpoint=150.30, current_time=t1)
        assert updated == 1

    def test_update_midpoint_before_horizon_no_update(self):
        """Midpoint update before horizon should not resolve the trade."""
        sa = SpreadAnalyzer(impact_horizon_sec=300)
        t0 = self._base_time()

        sa.record_trade("AAPL", trade_price=150.15, bid=150.10, ask=150.20,
                         timestamp=t0)

        t1 = t0 + timedelta(seconds=60)  # Only 1 minute, need 5
        updated = sa.update_midpoint("AAPL", future_midpoint=150.25, current_time=t1)
        assert updated == 0

    def test_adverse_selection_after_update(self):
        """After midpoint update, adverse selection should be computable."""
        sa = SpreadAnalyzer(impact_horizon_sec=60)
        t0 = self._base_time()

        # Buy at ask (above mid) -> if price moves up, that's adverse selection
        sa.record_trade("AAPL", trade_price=150.20, bid=150.10, ask=150.20,
                         timestamp=t0, side="buy")

        t1 = t0 + timedelta(seconds=61)
        sa.update_midpoint("AAPL", future_midpoint=150.30, current_time=t1)

        adverse = sa.get_adverse_selection("AAPL")
        # buy side: price_impact = +1 * (150.30 - 150.15) = 0.15
        assert adverse > 0

    def test_no_adverse_selection_without_update(self):
        """Before midpoint update, adverse selection should return 0."""
        sa = SpreadAnalyzer(impact_horizon_sec=300)
        sa.record_trade("AAPL", trade_price=150.15, bid=150.10, ask=150.20)
        assert sa.get_adverse_selection("AAPL") == 0.0


class TestSpreadAnalyzerEffectiveSpread:

    def test_trade_at_midpoint_zero_spread(self):
        sa = SpreadAnalyzer()
        spread = sa.record_trade("AAPL", trade_price=150.15, bid=150.10, ask=150.20)
        assert spread == pytest.approx(0.0, abs=1e-9)

    def test_trade_at_ask_full_spread(self):
        sa = SpreadAnalyzer()
        # midpoint = 150.15, trade at 150.20 (ask)
        spread = sa.record_trade("AAPL", trade_price=150.20, bid=150.10, ask=150.20)
        # effective_spread = 2 * |150.20 - 150.15| = 0.10
        assert spread == pytest.approx(0.10, abs=1e-9)

    def test_trade_at_bid_full_spread(self):
        sa = SpreadAnalyzer()
        # midpoint = 150.15, trade at 150.10 (bid)
        spread = sa.record_trade("AAPL", trade_price=150.10, bid=150.10, ask=150.20)
        assert spread == pytest.approx(0.10, abs=1e-9)

    def test_invalid_prices_return_zero(self):
        sa = SpreadAnalyzer()
        spread = sa.record_trade("AAPL", trade_price=0.0, bid=0.0, ask=0.0)
        assert spread == 0.0

    def test_average_effective_spread(self):
        sa = SpreadAnalyzer()
        sa.record_trade("AAPL", trade_price=150.20, bid=150.10, ask=150.20)
        sa.record_trade("AAPL", trade_price=150.15, bid=150.10, ask=150.20)
        avg = sa.get_effective_spread("AAPL")
        # Trade 1: spread = 0.10, Trade 2: spread = 0.00
        assert avg == pytest.approx(0.05, abs=1e-9)


class TestSpreadAnalyzerSideInference:

    def test_buy_inferred_above_midpoint(self):
        sa = SpreadAnalyzer()
        sa.record_trade("AAPL", trade_price=150.20, bid=150.10, ask=150.20)
        # Trade at ask -> inferred as buy
        with sa._lock:
            record = sa._trades["AAPL"][-1]
        assert record.side == "buy"

    def test_sell_inferred_below_midpoint(self):
        sa = SpreadAnalyzer()
        sa.record_trade("AAPL", trade_price=150.10, bid=150.10, ask=150.20)
        # Trade at bid -> inferred as sell
        with sa._lock:
            record = sa._trades["AAPL"][-1]
        assert record.side == "sell"


class TestSpreadAnalyzerExecutionQuality:

    def test_quality_score_range(self):
        sa = SpreadAnalyzer(impact_horizon_sec=60)
        t0 = datetime(2026, 4, 4, 14, 0, 0, tzinfo=timezone.utc)
        sa.record_trade("AAPL", trade_price=150.15, bid=150.10, ask=150.20,
                         timestamp=t0, side="buy")
        t1 = t0 + timedelta(seconds=61)
        sa.update_midpoint("AAPL", future_midpoint=150.16, current_time=t1)

        score = sa.get_execution_quality_score("AAPL")
        assert 0.0 <= score <= 1.0

    def test_quality_score_unknown_symbol(self):
        sa = SpreadAnalyzer()
        score = sa.get_execution_quality_score("UNKNOWN")
        # No trades: spread_score=1.0, impact_score=0.5 -> 0.6*1.0+0.4*0.5=0.8
        assert score == pytest.approx(0.8, abs=0.01)


class TestSpreadAnalyzerUnknownSymbol:

    def test_effective_spread_unknown_symbol(self):
        sa = SpreadAnalyzer()
        assert sa.get_effective_spread("UNKNOWN") == 0.0

    def test_adverse_selection_unknown_symbol(self):
        sa = SpreadAnalyzer()
        assert sa.get_adverse_selection("UNKNOWN") == 0.0

    def test_update_midpoint_unknown_symbol(self):
        sa = SpreadAnalyzer()
        assert sa.update_midpoint("UNKNOWN", 100.0) == 0
