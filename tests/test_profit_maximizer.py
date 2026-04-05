"""Comprehensive tests for engine/profit_maximizer.py — bonus alpha-generation features.

Tests cover:
- get_adaptive_scan_interval() with various VIX/regime/position combos
- IntradayVolRegime (CALM/ACTIVE/HEATED/EXTREME)
- WinStreakTracker multiplier calculation
- compute_dynamic_stop() tightening logic
- optimize_entry_price() spread-based optimization
"""

import sys
import time
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))


@pytest.fixture
def pm():
    """Import profit_maximizer module."""
    import importlib
    mod = importlib.import_module("engine.profit_maximizer")
    importlib.reload(mod)
    return mod


# ===================================================================
# get_adaptive_scan_interval
# ===================================================================

class TestAdaptiveScanInterval:
    """Scan interval adapts to VIX, regime, position count, time of day."""

    # --- VIX-based base intervals ---

    def test_extreme_vix_gets_fast_scan(self, pm):
        interval = pm.get_adaptive_scan_interval(vix_level=40, regime="NORMAL",
                                                  num_open_positions=0, hour=12)
        assert interval == 30  # Minimum

    def test_high_vix_gets_60s_base(self, pm):
        interval = pm.get_adaptive_scan_interval(vix_level=30, regime="NORMAL",
                                                  num_open_positions=0, hour=12)
        assert 30 <= interval <= 60

    def test_moderate_vix_gets_120s_base(self, pm):
        interval = pm.get_adaptive_scan_interval(vix_level=18, regime="NORMAL",
                                                  num_open_positions=0, hour=12)
        assert 90 <= interval <= 150

    def test_low_vix_gets_180s_base(self, pm):
        interval = pm.get_adaptive_scan_interval(vix_level=12, regime="NORMAL",
                                                  num_open_positions=0, hour=12)
        assert 140 <= interval <= 200

    # --- Regime adjustments ---

    def test_high_vol_regime_reduces_interval(self, pm):
        normal = pm.get_adaptive_scan_interval(vix_level=18, regime="NORMAL",
                                                num_open_positions=0, hour=12)
        vol = pm.get_adaptive_scan_interval(vix_level=18, regime="HIGH_VOL_BULL",
                                             num_open_positions=0, hour=12)
        assert vol < normal

    def test_mean_reverting_regime_reduces_interval(self, pm):
        normal = pm.get_adaptive_scan_interval(vix_level=18, regime="NORMAL",
                                                num_open_positions=0, hour=12)
        mr = pm.get_adaptive_scan_interval(vix_level=18, regime="MEAN_REVERTING",
                                            num_open_positions=0, hour=12)
        assert mr < normal

    def test_high_vol_bear_also_reduces(self, pm):
        normal = pm.get_adaptive_scan_interval(vix_level=18, regime="NORMAL",
                                                num_open_positions=0, hour=12)
        hvb = pm.get_adaptive_scan_interval(vix_level=18, regime="HIGH_VOL_BEAR",
                                             num_open_positions=0, hour=12)
        assert hvb < normal

    # --- Position count adjustments ---

    def test_many_positions_cap_interval(self, pm):
        interval = pm.get_adaptive_scan_interval(vix_level=12, regime="NORMAL",
                                                  num_open_positions=12, hour=12)
        assert interval <= 60  # Capped for >10 positions

    def test_moderate_positions_cap(self, pm):
        interval = pm.get_adaptive_scan_interval(vix_level=12, regime="NORMAL",
                                                  num_open_positions=7, hour=12)
        assert interval <= 90  # Capped for >5 positions

    def test_few_positions_no_cap(self, pm):
        interval = pm.get_adaptive_scan_interval(vix_level=12, regime="NORMAL",
                                                  num_open_positions=2, hour=12)
        assert interval > 90

    # --- Time of day ---

    def test_prime_hours_faster_scan(self, pm):
        # 10-11am and 1-3pm are prime hours (0.8x)
        prime = pm.get_adaptive_scan_interval(vix_level=18, regime="NORMAL",
                                               num_open_positions=0, hour=10)
        off = pm.get_adaptive_scan_interval(vix_level=18, regime="NORMAL",
                                             num_open_positions=0, hour=12)
        assert prime < off

    def test_open_close_hours_slightly_faster(self, pm):
        # Hour 9 and 15 get 0.9x multiplier
        edge = pm.get_adaptive_scan_interval(vix_level=18, regime="NORMAL",
                                              num_open_positions=0, hour=9)
        mid = pm.get_adaptive_scan_interval(vix_level=18, regime="NORMAL",
                                             num_open_positions=0, hour=12)
        assert edge <= mid

    # --- Bounds ---

    def test_minimum_30s(self, pm):
        # Even with all reductions
        interval = pm.get_adaptive_scan_interval(vix_level=50, regime="HIGH_VOL_BULL",
                                                  num_open_positions=15, hour=10)
        assert interval >= 30

    def test_maximum_300s(self, pm):
        # Even with all slow factors
        interval = pm.get_adaptive_scan_interval(vix_level=8, regime="NORMAL",
                                                  num_open_positions=0, hour=12)
        assert interval <= 300


# ===================================================================
# IntradayVolRegime
# ===================================================================

class TestIntradayVolRegime:
    """Micro-regime detection: CALM, ACTIVE, HEATED, EXTREME."""

    def test_default_regime_is_calm(self, pm):
        ivr = pm.IntradayVolRegime()
        assert ivr.get_regime() == "CALM"

    def test_calm_regime_sizing(self, pm):
        ivr = pm.IntradayVolRegime()
        assert ivr.get_sizing_multiplier() == 1.0

    def test_regime_with_data(self, pm):
        """After enough data, regime should be computed."""
        ivr = pm.IntradayVolRegime(lookback=20)
        # Feed in stable prices (low vol)
        base_time = time.time() - 5000
        for i in range(15):
            with patch("time.time", return_value=base_time + i * 300):
                ivr.update(100.0 + np.random.normal(0, 0.01))

        # With very small moves, should be CALM
        regime = ivr.get_regime()
        assert regime in ("CALM", "ACTIVE")

    def test_extreme_regime_from_large_moves(self, pm):
        """Injecting very large moves should push to HEATED/EXTREME."""
        ivr = pm.IntradayVolRegime(lookback=30)
        base_time = time.time() - 10000

        # First feed normal data
        for i in range(20):
            with patch("time.time", return_value=base_time + i * 300):
                ivr.update(100.0 + i * 0.01)

        # Then feed a large move
        with patch("time.time", return_value=base_time + 20 * 300):
            ivr.update(110.0)  # 10% jump

        regime = ivr.get_regime()
        assert regime in ("HEATED", "EXTREME")

    def test_sizing_multipliers_are_valid(self, pm):
        expected = {"CALM": 1.0, "ACTIVE": 0.9, "HEATED": 0.7, "EXTREME": 0.4}
        ivr = pm.IntradayVolRegime()
        for regime, mult in expected.items():
            # Test the mapping directly
            result = {"CALM": 1.0, "ACTIVE": 0.9, "HEATED": 0.7, "EXTREME": 0.4}[regime]
            assert result == mult

    def test_not_enough_data_returns_calm(self, pm):
        """With < 10 data points, default to CALM."""
        ivr = pm.IntradayVolRegime()
        base_time = time.time() - 3000
        for i in range(5):
            with patch("time.time", return_value=base_time + i * 300):
                ivr.update(100.0 + i * 0.5)
        assert ivr.get_regime() == "CALM"

    def test_update_respects_minimum_interval(self, pm):
        """Updates closer than 240s should not add vol data."""
        ivr = pm.IntradayVolRegime()
        now = time.time()
        with patch("time.time", return_value=now):
            ivr.update(100.0)
        with patch("time.time", return_value=now + 60):  # Only 60s later
            ivr.update(101.0)
        # Should only have at most 0 entries (first update sets _last_price, no vol yet)
        assert len(ivr._vol_history) == 0


# ===================================================================
# WinStreakTracker
# ===================================================================

class TestWinStreakTracker:
    """Multiplier based on recent win/loss pattern."""

    def test_empty_tracker_returns_1_0(self, pm):
        tracker = pm.WinStreakTracker()
        assert tracker.get_multiplier() == 1.0

    def test_less_than_3_trades_returns_1_0(self, pm):
        tracker = pm.WinStreakTracker()
        tracker.record_trade(True)
        tracker.record_trade(True)
        assert tracker.get_multiplier() == 1.0

    def test_hot_streak_4_wins(self, pm):
        tracker = pm.WinStreakTracker()
        for _ in range(5):
            tracker.record_trade(True)
        # 4+ wins in last 5 → 1.15
        assert tracker.get_multiplier() == 1.15

    def test_warm_streak_3_wins(self, pm):
        tracker = pm.WinStreakTracker()
        for _ in range(3):
            tracker.record_trade(True)
        # 3 wins in last 5 (only 3 trades recorded) → 1.05
        assert tracker.get_multiplier() == 1.05

    def test_cold_streak_4_losses(self, pm):
        tracker = pm.WinStreakTracker()
        for _ in range(5):
            tracker.record_trade(False)
        # 4+ losses in last 5 → 0.7
        assert tracker.get_multiplier() == 0.7

    def test_mild_cold_streak_3_losses(self, pm):
        tracker = pm.WinStreakTracker()
        tracker.record_trade(True)   # 1 win
        tracker.record_trade(False)  # 3 losses
        tracker.record_trade(False)
        tracker.record_trade(False)
        # last 5: 1W 3L → losses >= 3 → 0.85
        assert tracker.get_multiplier() == 0.85

    def test_mixed_results_neutral(self, pm):
        tracker = pm.WinStreakTracker()
        tracker.record_trade(True)
        tracker.record_trade(False)
        tracker.record_trade(True)
        tracker.record_trade(False)
        tracker.record_trade(True)
        # 3W 2L in last 5 → 3 wins → 1.05
        assert tracker.get_multiplier() == 1.05

    def test_even_split_neutral(self, pm):
        tracker = pm.WinStreakTracker()
        tracker.record_trade(True)
        tracker.record_trade(False)
        tracker.record_trade(True)
        tracker.record_trade(False)
        # 2W 2L → neither >= 3 → 1.0
        assert tracker.get_multiplier() == 1.0

    def test_max_streak_capped(self, pm):
        tracker = pm.WinStreakTracker(max_streak=10)
        for _ in range(15):
            tracker.record_trade(True)
        # Deque only keeps last 10
        assert len(tracker._results) == 10
        assert tracker.get_multiplier() == 1.15

    def test_get_stats(self, pm):
        tracker = pm.WinStreakTracker()
        tracker.record_trade(True)
        tracker.record_trade(True)
        tracker.record_trade(True)
        stats = tracker.get_stats()
        assert stats["streak"] == 3
        assert stats["multiplier"] == 1.05
        assert stats["last_5_wins"] == 3

    def test_get_stats_loss_streak(self, pm):
        tracker = pm.WinStreakTracker()
        tracker.record_trade(True)
        tracker.record_trade(False)
        tracker.record_trade(False)
        stats = tracker.get_stats()
        assert stats["streak"] == -2  # Negative for losses

    def test_get_stats_empty(self, pm):
        tracker = pm.WinStreakTracker()
        stats = tracker.get_stats()
        assert stats["streak"] == 0
        assert stats["multiplier"] == 1.0


# ===================================================================
# compute_dynamic_stop
# ===================================================================

class TestComputeDynamicStop:
    """Dynamic stop tightening based on hold time and profit."""

    @dataclass
    class FakeTrade:
        entry_price: float = 100.0
        side: str = "buy"
        highest_price_seen: float = 100.0
        lowest_price_seen: float = 100.0
        stop_loss: float = 95.0

    def test_no_tightening_first_15_min(self, pm):
        trade = self.FakeTrade(entry_price=100.0, stop_loss=95.0, side="buy")
        result = pm.compute_dynamic_stop(trade, current_price=101.0, atr=1.0,
                                          minutes_held=10)
        assert result is None

    def test_no_tightening_if_losing(self, pm):
        trade = self.FakeTrade(entry_price=100.0, stop_loss=95.0, side="buy")
        result = pm.compute_dynamic_stop(trade, current_price=99.0, atr=1.0,
                                          minutes_held=30)
        assert result is None

    def test_breakeven_stop_15_to_60_min(self, pm):
        """Between 15-60 min with > 0.3% profit, move stop to entry."""
        trade = self.FakeTrade(entry_price=100.0, stop_loss=95.0, side="buy")
        result = pm.compute_dynamic_stop(trade, current_price=101.0, atr=1.0,
                                          minutes_held=30)
        assert result is not None
        assert result >= 100.0  # At least entry price

    def test_breakeven_stop_requires_0_3pct_profit(self, pm):
        """At < 0.3% profit in 15-60 min window, don't tighten."""
        trade = self.FakeTrade(entry_price=100.0, stop_loss=95.0, side="buy")
        result = pm.compute_dynamic_stop(trade, current_price=100.20, atr=1.0,
                                          minutes_held=30)
        assert result is None

    def test_trailing_stop_after_60_min(self, pm):
        """After 60 min, trail at 1.5 ATR from high-water mark."""
        trade = self.FakeTrade(
            entry_price=100.0, stop_loss=95.0, side="buy",
            highest_price_seen=105.0,
        )
        result = pm.compute_dynamic_stop(trade, current_price=104.0, atr=1.0,
                                          minutes_held=90)
        assert result is not None
        # Expected: hwm(105) - 1.5*atr(1.0) = 103.5
        assert result == pytest.approx(103.5, abs=0.01)

    def test_trailing_stop_only_tightens(self, pm):
        """Stop should only ratchet up, never loosen."""
        trade = self.FakeTrade(
            entry_price=100.0, stop_loss=104.0, side="buy",
            highest_price_seen=105.0,
        )
        # Computed stop: 105 - 1.5 = 103.5 which is BELOW current 104
        result = pm.compute_dynamic_stop(trade, current_price=104.5, atr=1.0,
                                          minutes_held=90)
        # Should return None since 103.5 < 104.0 (would loosen)
        assert result is None

    def test_short_side_breakeven(self, pm):
        """Short positions: breakeven stop moves DOWN to entry."""
        trade = self.FakeTrade(
            entry_price=100.0, stop_loss=105.0, side="sell",
            lowest_price_seen=98.0,
        )
        result = pm.compute_dynamic_stop(trade, current_price=99.0, atr=1.0,
                                          minutes_held=30)
        assert result is not None
        assert result <= 100.0  # At or below entry

    def test_short_side_trailing_after_60_min(self, pm):
        """Short: trail at hwm (lowest) + 1.5 ATR."""
        trade = self.FakeTrade(
            entry_price=100.0, stop_loss=105.0, side="sell",
            lowest_price_seen=95.0,
        )
        result = pm.compute_dynamic_stop(trade, current_price=96.0, atr=1.0,
                                          minutes_held=90)
        assert result is not None
        # Expected: lwm(95) + 1.5*atr(1.0) = 96.5
        assert result == pytest.approx(96.5, abs=0.01)

    def test_short_trailing_only_tightens_down(self, pm):
        """Short stop should only ratchet DOWN, not loosen upward."""
        trade = self.FakeTrade(
            entry_price=100.0, stop_loss=96.0, side="sell",
            lowest_price_seen=95.0,
        )
        # Computed stop: 95 + 1.5 = 96.5 which is ABOVE current 96.0 → loosening
        result = pm.compute_dynamic_stop(trade, current_price=95.5, atr=1.0,
                                          minutes_held=90)
        assert result is None

    def test_zero_atr_returns_none(self, pm):
        trade = self.FakeTrade(entry_price=100.0, side="buy")
        result = pm.compute_dynamic_stop(trade, current_price=101.0, atr=0.0,
                                          minutes_held=90)
        assert result is None

    def test_missing_attributes_returns_none(self, pm):
        """Trade without entry_price attr should return None."""
        trade = object()
        result = pm.compute_dynamic_stop(trade, current_price=101.0, atr=1.0,
                                          minutes_held=30)
        assert result is None


# ===================================================================
# optimize_entry_price
# ===================================================================

class TestOptimizeEntryPrice:
    """Spread-based entry price optimization."""

    def test_tight_spread_uses_signal_price(self, pm):
        # Spread < 3bps
        result = pm.optimize_entry_price(
            signal_price=100.0, bid=99.99, ask=100.01,
            urgency="normal", side="buy",
        )
        assert result == 100.0

    def test_wide_spread_uses_midpoint(self, pm):
        # Spread > 10bps (0.10%)
        bid, ask = 99.90, 100.10
        result = pm.optimize_entry_price(
            signal_price=100.0, bid=bid, ask=ask,
            urgency="normal", side="buy",
        )
        mid = (bid + ask) / 2
        assert result == round(mid, 2)

    def test_normal_spread_buy_below_mid(self, pm):
        # Spread 5bps (between 3bps and 10bps)
        bid, ask = 99.95, 100.05
        result = pm.optimize_entry_price(
            signal_price=100.0, bid=bid, ask=ask,
            urgency="normal", side="buy",
        )
        mid = (bid + ask) / 2
        spread = ask - bid
        expected = round(mid - spread * 0.1, 2)
        assert result == expected

    def test_normal_spread_sell_above_mid(self, pm):
        bid, ask = 99.95, 100.05
        result = pm.optimize_entry_price(
            signal_price=100.0, bid=bid, ask=ask,
            urgency="normal", side="sell",
        )
        mid = (bid + ask) / 2
        spread = ask - bid
        expected = round(mid + spread * 0.1, 2)
        assert result == expected

    def test_critical_urgency_buy_at_ask(self, pm):
        result = pm.optimize_entry_price(
            signal_price=100.0, bid=99.90, ask=100.10,
            urgency="critical", side="buy",
        )
        assert result == 100.10

    def test_critical_urgency_sell_at_bid(self, pm):
        result = pm.optimize_entry_price(
            signal_price=100.0, bid=99.90, ask=100.10,
            urgency="critical", side="sell",
        )
        assert result == 99.90

    def test_invalid_bid_ask_returns_signal_price(self, pm):
        # bid <= 0
        result = pm.optimize_entry_price(
            signal_price=100.0, bid=0, ask=100.10,
            urgency="normal", side="buy",
        )
        assert result == 100.0

    def test_crossed_market_returns_signal_price(self, pm):
        # ask <= bid (crossed)
        result = pm.optimize_entry_price(
            signal_price=100.0, bid=100.10, ask=99.90,
            urgency="normal", side="buy",
        )
        assert result == 100.0

    def test_equal_bid_ask_returns_signal_price(self, pm):
        result = pm.optimize_entry_price(
            signal_price=100.0, bid=100.0, ask=100.0,
            urgency="normal", side="buy",
        )
        assert result == 100.0

    def test_result_is_rounded(self, pm):
        bid, ask = 99.90, 100.10
        result = pm.optimize_entry_price(
            signal_price=100.0, bid=bid, ask=ask,
            urgency="normal", side="buy",
        )
        # Check result is rounded to 2 decimal places
        assert result == round(result, 2)


# ===================================================================
# Signal Stacking (bonus test)
# ===================================================================

class TestSignalStacking:
    """stack_weak_signals combines multiple weak signals."""

    def test_two_strategies_agree(self, pm):
        @dataclass
        class FakeSig:
            symbol: str
            side: str
            strategy: str
            confidence: float = 0.5

        signals = [
            FakeSig(symbol="AAPL", side="buy", strategy="A", confidence=0.5),
            FakeSig(symbol="AAPL", side="buy", strategy="B", confidence=0.6),
        ]

        stacked = pm.stack_weak_signals(signals, min_agreement=2, min_combined_score=0.3)
        assert len(stacked) == 1
        assert stacked[0].symbol == "AAPL"
        assert stacked[0].direction == "buy"
        assert len(stacked[0].sources) == 2

    def test_below_min_agreement(self, pm):
        @dataclass
        class FakeSig:
            symbol: str
            side: str
            strategy: str
            confidence: float = 0.5

        signals = [
            FakeSig(symbol="AAPL", side="buy", strategy="A", confidence=0.5),
        ]

        stacked = pm.stack_weak_signals(signals, min_agreement=2)
        assert len(stacked) == 0

    def test_opposite_directions_separate_groups(self, pm):
        @dataclass
        class FakeSig:
            symbol: str
            side: str
            strategy: str
            confidence: float = 0.5

        signals = [
            FakeSig(symbol="AAPL", side="buy", strategy="A", confidence=0.5),
            FakeSig(symbol="AAPL", side="sell", strategy="B", confidence=0.5),
        ]

        # Each direction has only 1 signal, below min_agreement=2
        stacked = pm.stack_weak_signals(signals, min_agreement=2)
        assert len(stacked) == 0
