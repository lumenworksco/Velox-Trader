"""Tests for microstructure/vpin.py (MICRO-001).

Covers VPIN:
  - Trade ingestion and bucketing
  - VPIN computation and range [0, 1]
  - Tick rule classification
  - Toxicity detection
  - Sizing modifier
  - Edge cases: zero volume, single trade, insufficient buckets
"""

import pytest

from microstructure.vpin import VPIN


class TestVPINInit:
    """Test constructor validation."""

    def test_default_init(self):
        v = VPIN()
        assert v.completed_bucket_count == 0
        assert v.total_trades == 0

    def test_invalid_bucket_volume(self):
        with pytest.raises(ValueError, match="bucket_volume must be positive"):
            VPIN(bucket_volume=0)

    def test_invalid_n_buckets(self):
        with pytest.raises(ValueError, match="n_buckets must be positive"):
            VPIN(n_buckets=0)


class TestVPINTradeIngestion:
    """Test add_trade and bucket filling."""

    def test_add_single_trade(self):
        v = VPIN(bucket_volume=1000)
        v.add_trade(price=100.0, volume=500, side="buy")
        assert v.total_trades == 1
        assert v.completed_bucket_count == 0

    def test_fill_one_bucket(self):
        v = VPIN(bucket_volume=1000)
        v.add_trade(price=100.0, volume=1000, side="buy")
        assert v.completed_bucket_count == 1

    def test_fill_multiple_buckets(self):
        v = VPIN(bucket_volume=100)
        v.add_trade(price=100.0, volume=500, side="buy")
        assert v.completed_bucket_count == 5

    def test_zero_volume_ignored(self):
        v = VPIN(bucket_volume=1000)
        v.add_trade(price=100.0, volume=0, side="buy")
        assert v.total_trades == 0

    def test_negative_volume_ignored(self):
        v = VPIN(bucket_volume=1000)
        v.add_trade(price=100.0, volume=-100, side="buy")
        assert v.total_trades == 0

    def test_tick_rule_uptick_is_buy(self):
        v = VPIN(bucket_volume=100)
        v.add_trade(price=100.0, volume=100)  # first trade -> buy (no prev)
        v.add_trade(price=101.0, volume=100)  # uptick -> buy
        assert v.completed_bucket_count == 2

    def test_tick_rule_downtick_is_sell(self):
        v = VPIN(bucket_volume=100)
        v.add_trade(price=100.0, volume=100)
        v.add_trade(price=99.0, volume=100)   # downtick -> sell
        assert v.completed_bucket_count == 2

    def test_trade_splits_across_buckets(self):
        """A large trade fills current bucket and spills into next."""
        v = VPIN(bucket_volume=100)
        v.add_trade(price=100.0, volume=50, side="buy")
        v.add_trade(price=100.0, volume=80, side="buy")
        # 50+80=130 -> bucket 1 full at 100, 30 spill to bucket 2
        assert v.completed_bucket_count == 1


class TestVPINComputation:
    """Test compute_vpin output."""

    def test_vpin_no_buckets(self):
        v = VPIN(bucket_volume=1000)
        assert v.compute_vpin() == 0.0

    def test_vpin_all_buys(self):
        """All buys -> imbalance = 1.0 per bucket -> VPIN = 1.0."""
        v = VPIN(bucket_volume=100, n_buckets=5)
        for _ in range(5):
            v.add_trade(price=100.0, volume=100, side="buy")
        vpin = v.compute_vpin()
        assert vpin == pytest.approx(1.0)

    def test_vpin_all_sells(self):
        v = VPIN(bucket_volume=100, n_buckets=5)
        for _ in range(5):
            v.add_trade(price=100.0, volume=100, side="sell")
        vpin = v.compute_vpin()
        assert vpin == pytest.approx(1.0)

    def test_vpin_balanced_flow(self):
        """Equal buy and sell per bucket -> imbalance = 0.0."""
        v = VPIN(bucket_volume=100, n_buckets=5)
        for _ in range(5):
            v.add_trade(price=100.0, volume=50, side="buy")
            v.add_trade(price=100.0, volume=50, side="sell")
        vpin = v.compute_vpin()
        assert vpin == pytest.approx(0.0)

    def test_vpin_range(self):
        v = VPIN(bucket_volume=100, n_buckets=10)
        for i in range(20):
            side = "buy" if i % 3 == 0 else "sell"
            v.add_trade(price=100.0 + i * 0.1, volume=100, side=side)
        vpin = v.compute_vpin()
        assert 0.0 <= vpin <= 1.0

    def test_vpin_with_trades_argument(self):
        v = VPIN(bucket_volume=100, n_buckets=5)
        trades = [
            {"price": 100.0, "volume": 100, "side": "buy"},
            {"price": 100.0, "volume": 100, "side": "sell"},
            {"price": 100.0, "volume": 100, "side": "buy"},
        ]
        vpin = v.compute_vpin(trades=trades)
        assert 0.0 <= vpin <= 1.0

    def test_vpin_deterministic(self):
        """Same inputs produce same output."""
        def make_vpin():
            v = VPIN(bucket_volume=100, n_buckets=5)
            for i in range(10):
                v.add_trade(price=100.0 + i, volume=100, side="buy" if i % 2 == 0 else "sell")
            return v.compute_vpin()
        assert make_vpin() == make_vpin()


class TestVPINToxicity:
    """Test toxicity detection."""

    def test_is_toxic_below_threshold(self):
        v = VPIN(bucket_volume=100, n_buckets=5, high_vpin_threshold=0.7)
        for _ in range(5):
            v.add_trade(price=100.0, volume=50, side="buy")
            v.add_trade(price=100.0, volume=50, side="sell")
        assert v.is_toxic() is False

    def test_is_toxic_above_threshold(self):
        v = VPIN(bucket_volume=100, n_buckets=5, high_vpin_threshold=0.7)
        for _ in range(5):
            v.add_trade(price=100.0, volume=100, side="buy")
        assert v.is_toxic() is True

    def test_estimate_toxicity_output_format(self):
        v = VPIN(bucket_volume=100, n_buckets=5)
        for _ in range(5):
            v.add_trade(price=100.0, volume=100, side="buy")
        result = v.estimate_toxicity()
        assert "vpin" in result
        assert "is_toxic" in result
        assert "label" in result
        assert "buckets_filled" in result
        assert "sizing_modifier" in result
        assert isinstance(result["vpin"], float)
        assert isinstance(result["is_toxic"], bool)
        assert result["label"] in ("TOXIC_FLOW", "NORMAL")

    def test_estimate_toxicity_with_trades(self):
        v = VPIN(bucket_volume=100, n_buckets=5)
        trades = [{"price": 100.0, "volume": 100, "side": "buy"}] * 5
        result = v.estimate_toxicity(trades=trades)
        assert result["buckets_filled"] == 5


class TestVPINSizingModifier:
    """Test position sizing based on VPIN."""

    def test_low_vpin_modifier_is_one(self):
        v = VPIN(bucket_volume=100, n_buckets=5)
        for _ in range(5):
            v.add_trade(price=100.0, volume=50, side="buy")
            v.add_trade(price=100.0, volume=50, side="sell")
        mod = v.get_sizing_modifier()
        assert mod == pytest.approx(1.0)

    def test_high_vpin_modifier_is_low(self):
        v = VPIN(bucket_volume=100, n_buckets=5, high_vpin_threshold=0.7)
        for _ in range(5):
            v.add_trade(price=100.0, volume=100, side="buy")
        mod = v.get_sizing_modifier()
        assert mod == pytest.approx(0.3)

    def test_modifier_range(self):
        v = VPIN(bucket_volume=100, n_buckets=10)
        for i in range(20):
            v.add_trade(price=100.0, volume=100, side="buy" if i % 3 == 0 else "sell")
        mod = v.get_sizing_modifier()
        assert 0.3 <= mod <= 1.0


class TestVPINFactoryAndReset:
    """Test from_adv factory and reset."""

    def test_from_adv(self):
        v = VPIN.from_adv(adv=500_000, n_buckets=50, toxic_threshold=0.35)
        assert v._bucket_volume == 10_000
        assert v._n_buckets == 50
        assert v._high_vpin_threshold == 0.35

    def test_from_adv_small(self):
        v = VPIN.from_adv(adv=10)
        assert v._bucket_volume >= 1

    def test_reset(self):
        v = VPIN(bucket_volume=100, n_buckets=5)
        v.add_trade(price=100.0, volume=500, side="buy")
        v.reset()
        assert v.completed_bucket_count == 0
        assert v.total_trades == 0
        assert v.compute_vpin() == 0.0

    def test_repr(self):
        v = VPIN(bucket_volume=100, n_buckets=5)
        r = repr(v)
        assert "VPIN" in r
