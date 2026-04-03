"""Tests for microstructure/order_book_signal.py (T5-014).

Covers OrderBookFeatureExtractor and OrderImbalanceModel:
  - Normal operation with realistic order book data
  - Edge cases: empty book, zero sizes, single update
  - Output format and value ranges
  - Per-symbol tracking and reset
  - Model fit, predict, and sizing multiplier
"""

import threading
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from microstructure.order_book_signal import (
    OrderBookFeatureExtractor,
    OrderBookFeatures,
    OrderBookSnapshot,
    OrderImbalanceModel,
    get_order_book_model,
    BUY_MULTIPLIER,
    SELL_MULTIPLIER,
    DEPTH_LEVELS,
    HISTORY_SIZE,
    IMBALANCE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# OrderBookFeatureExtractor
# ---------------------------------------------------------------------------

class TestOrderBookFeatureExtractor:
    """Tests for OrderBookFeatureExtractor."""

    def test_basic_update_returns_snapshot(self):
        ext = OrderBookFeatureExtractor()
        snap = ext.update(
            bid_sizes=[100, 200, 150, 80, 50],
            ask_sizes=[80, 150, 200, 120, 90],
        )
        assert isinstance(snap, OrderBookSnapshot)
        assert snap.timestamp > 0

    def test_depth_imbalance_balanced(self):
        ext = OrderBookFeatureExtractor()
        snap = ext.update(bid_sizes=[100, 100], ask_sizes=[100, 100])
        assert snap.depth_imbalance == pytest.approx(0.0, abs=1e-9)

    def test_depth_imbalance_all_bids(self):
        ext = OrderBookFeatureExtractor()
        snap = ext.update(bid_sizes=[100, 200, 300, 400, 500], ask_sizes=[0, 0, 0, 0, 0])
        assert snap.depth_imbalance == pytest.approx(1.0, abs=1e-9)

    def test_depth_imbalance_all_asks(self):
        ext = OrderBookFeatureExtractor()
        snap = ext.update(bid_sizes=[0, 0, 0, 0, 0], ask_sizes=[100, 200, 300, 400, 500])
        assert snap.depth_imbalance == pytest.approx(-1.0, abs=1e-9)

    def test_depth_imbalance_empty_book(self):
        """Empty order book (all zeros) should produce 0 imbalance."""
        ext = OrderBookFeatureExtractor()
        snap = ext.update(bid_sizes=[], ask_sizes=[])
        assert snap.depth_imbalance == 0.0

    def test_sweep_indicator_triggered(self):
        ext = OrderBookFeatureExtractor()
        snap = ext.update(
            bid_sizes=[100], ask_sizes=[100],
            last_trade_size=500, top_bid_size=100,
        )
        assert snap.sweep_indicator == 1

    def test_sweep_indicator_not_triggered(self):
        ext = OrderBookFeatureExtractor()
        snap = ext.update(
            bid_sizes=[100], ask_sizes=[100],
            last_trade_size=50, top_bid_size=100,
        )
        assert snap.sweep_indicator == 0

    def test_sweep_indicator_zero_sizes(self):
        ext = OrderBookFeatureExtractor()
        snap = ext.update(
            bid_sizes=[100], ask_sizes=[100],
            last_trade_size=0, top_bid_size=0,
        )
        assert snap.sweep_indicator == 0

    def test_queue_depletion_rate_first_update(self):
        """First update should have zero depletion (no prior reference)."""
        ext = OrderBookFeatureExtractor()
        snap = ext.update(bid_sizes=[100, 200], ask_sizes=[100, 200])
        assert snap.queue_depletion_rate == 0.0

    def test_queue_depletion_rate_decrease(self):
        ext = OrderBookFeatureExtractor()
        ext.update(bid_sizes=[200, 200], ask_sizes=[100, 100])
        snap2 = ext.update(bid_sizes=[100, 100], ask_sizes=[100, 100])
        # Bid total went from 400 to 200 -> (200-400)/400 = -0.5
        assert snap2.queue_depletion_rate == pytest.approx(-0.5, abs=1e-9)

    def test_padding_short_bid_ask_lists(self):
        """Lists shorter than DEPTH_LEVELS are padded with zeros."""
        ext = OrderBookFeatureExtractor()
        snap = ext.update(bid_sizes=[100], ask_sizes=[50, 50])
        assert len(snap.bid_sizes) == DEPTH_LEVELS
        assert len(snap.ask_sizes) == DEPTH_LEVELS

    def test_truncation_long_bid_ask_lists(self):
        """Lists longer than DEPTH_LEVELS are truncated."""
        ext = OrderBookFeatureExtractor()
        snap = ext.update(
            bid_sizes=[10] * 10,
            ask_sizes=[20] * 10,
        )
        assert len(snap.bid_sizes) == DEPTH_LEVELS
        assert len(snap.ask_sizes) == DEPTH_LEVELS

    def test_custom_timestamp(self):
        ext = OrderBookFeatureExtractor()
        snap = ext.update(bid_sizes=[100], ask_sizes=[100], timestamp=1000.0)
        assert snap.timestamp == 1000.0

    def test_per_symbol_tracking(self):
        ext = OrderBookFeatureExtractor()
        ext.update(bid_sizes=[200, 100], ask_sizes=[100, 100], symbol="AAPL")
        ext.update(bid_sizes=[100, 100], ask_sizes=[200, 100], symbol="GOOG")

        feat_aapl = ext.get_features(symbol="AAPL")
        feat_goog = ext.get_features(symbol="GOOG")
        assert feat_aapl.depth_imbalance > 0  # more bids
        assert feat_goog.depth_imbalance < 0  # more asks

    def test_get_features_empty_history(self):
        ext = OrderBookFeatureExtractor()
        feat = ext.get_features()
        assert isinstance(feat, OrderBookFeatures)
        assert feat.depth_imbalance == 0.0
        assert feat.effective_signal == 0.0
        assert feat.bid_ask_size_ratio == 1.0

    def test_get_features_single_snapshot(self):
        ext = OrderBookFeatureExtractor()
        ext.update(bid_sizes=[300, 200], ask_sizes=[100, 100])
        feat = ext.get_features()
        assert isinstance(feat, OrderBookFeatures)
        assert feat.depth_imbalance > 0

    def test_effective_signal_bounded(self):
        """Effective signal must be in [-1, 1]."""
        ext = OrderBookFeatureExtractor()
        for _ in range(100):
            ext.update(bid_sizes=[1000, 500, 300, 200, 100], ask_sizes=[1, 1, 1, 1, 1])
        feat = ext.get_features()
        assert -1.0 <= feat.effective_signal <= 1.0

    def test_imbalance_trend_with_enough_data(self):
        ext = OrderBookFeatureExtractor()
        # Insert 60+ snapshots with increasing bid dominance
        for i in range(70):
            ext.update(bid_sizes=[100 + i * 5], ask_sizes=[100], timestamp=float(i))
        feat = ext.get_features()
        assert feat.imbalance_trend > 0  # Trend should be positive

    def test_sweep_count_in_features(self):
        ext = OrderBookFeatureExtractor()
        base_ts = 1000.0
        for i in range(10):
            ext.update(
                bid_sizes=[100], ask_sizes=[100],
                last_trade_size=500, top_bid_size=100,
                timestamp=base_ts + i,
            )
        feat = ext.get_features()
        assert feat.sweep_indicator >= 1

    def test_reset_global(self):
        ext = OrderBookFeatureExtractor()
        ext.update(bid_sizes=[100], ask_sizes=[100])
        ext.reset()
        feat = ext.get_features()
        assert feat.depth_imbalance == 0.0

    def test_reset_per_symbol(self):
        ext = OrderBookFeatureExtractor()
        ext.update(bid_sizes=[200], ask_sizes=[100], symbol="AAPL")
        ext.reset(symbol="AAPL")
        feat = ext.get_features(symbol="AAPL")
        assert feat.depth_imbalance == 0.0

    def test_ema_updates_across_snapshots(self):
        ext = OrderBookFeatureExtractor(ema_alpha=0.5)
        ext.update(bid_sizes=[200, 100], ask_sizes=[100, 100])
        feat1 = ext.get_features()
        ext.update(bid_sizes=[200, 100], ask_sizes=[100, 100])
        feat2 = ext.get_features()
        # EMA should be closer to current imbalance after second update
        assert abs(feat2.depth_imbalance_ema) >= abs(feat1.depth_imbalance_ema) * 0.5

    def test_thread_safety(self):
        """Multiple threads updating concurrently should not raise."""
        ext = OrderBookFeatureExtractor()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    ext.update(bid_sizes=[100, 50], ask_sizes=[80, 60])
                    ext.get_features()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# OrderImbalanceModel
# ---------------------------------------------------------------------------

class TestOrderImbalanceModel:
    """Tests for OrderImbalanceModel."""

    def test_predict_signal_unfitted_returns_effective_signal(self):
        model = OrderImbalanceModel()
        # No data, unfitted -> should return 0.0 (effective_signal default)
        signal = model.predict_signal()
        assert isinstance(signal, float)
        assert -1.0 <= signal <= 1.0

    def test_predict_signal_range(self):
        model = OrderImbalanceModel()
        for _ in range(50):
            model.extractor.update(bid_sizes=[300, 200], ask_sizes=[50, 50])
        signal = model.predict_signal()
        assert -1.0 <= signal <= 1.0

    def test_fit_returns_metrics(self):
        model = OrderImbalanceModel()
        np.random.seed(42)
        X = np.random.randn(100, 7)
        y = (X[:, 0] > 0).astype(float)
        metrics = model.fit(X, y, lr=0.01, epochs=50)
        assert "accuracy" in metrics
        assert "epochs" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["epochs"] == 50

    def test_predict_after_fit(self):
        model = OrderImbalanceModel()
        np.random.seed(42)
        X = np.random.randn(100, 7)
        y = (X[:, 0] > 0).astype(float)
        model.fit(X, y, lr=0.01, epochs=50)
        # Feed some data into extractor
        model.extractor.update(bid_sizes=[300, 200, 100, 50, 20], ask_sizes=[50, 50, 50, 50, 50])
        signal = model.predict_signal()
        assert isinstance(signal, float)
        assert -1.0 <= signal <= 1.0

    def test_get_sizing_multiplier_disabled(self):
        """When config flag is disabled (default for tests), returns 1.0."""
        model = OrderImbalanceModel()
        mult = model.get_sizing_multiplier("AAPL")
        assert mult == 1.0

    def test_get_sizing_multiplier_enabled_buy(self):
        model = OrderImbalanceModel()
        with patch("microstructure.order_book_signal.OrderImbalanceModel.predict_signal",
                    return_value=0.5):
            with patch.dict("sys.modules", {"config": MagicMock(ORDER_BOOK_SIGNAL_ENABLED=True)}):
                mult = model.get_sizing_multiplier("AAPL")
                assert mult == BUY_MULTIPLIER

    def test_get_sizing_multiplier_enabled_sell(self):
        model = OrderImbalanceModel()
        with patch("microstructure.order_book_signal.OrderImbalanceModel.predict_signal",
                    return_value=-0.5):
            with patch.dict("sys.modules", {"config": MagicMock(ORDER_BOOK_SIGNAL_ENABLED=True)}):
                mult = model.get_sizing_multiplier("AAPL")
                assert mult == SELL_MULTIPLIER

    def test_get_sizing_multiplier_enabled_neutral(self):
        model = OrderImbalanceModel()
        with patch("microstructure.order_book_signal.OrderImbalanceModel.predict_signal",
                    return_value=0.1):
            with patch.dict("sys.modules", {"config": MagicMock(ORDER_BOOK_SIGNAL_ENABLED=True)}):
                mult = model.get_sizing_multiplier("AAPL")
                assert mult == 1.0

    def test_extractor_property(self):
        model = OrderImbalanceModel()
        assert isinstance(model.extractor, OrderBookFeatureExtractor)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class TestGetOrderBookModel:
    def test_returns_model_instance(self):
        model = get_order_book_model()
        assert isinstance(model, OrderImbalanceModel)

    def test_singleton_returns_same_instance(self):
        m1 = get_order_book_model()
        m2 = get_order_book_model()
        assert m1 is m2
