"""Tests for ORBStrategyV2."""

import types
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(high=102.0, low=98.0, open_=99.0, close=101.0, volume=100_000,
               num_bars=6, base_time=None):
    """Build a DataFrame mimicking intraday OHLCV bars."""
    if base_time is None:
        base_time = datetime(2026, 3, 13, 9, 30, tzinfo=ET)
    idx = pd.date_range(start=base_time, periods=num_bars, freq="5min")
    return pd.DataFrame({
        "open": [open_] * num_bars,
        "high": [high] * num_bars,
        "low": [low] * num_bars,
        "close": [close] * num_bars,
        "volume": [volume] * num_bars,
    }, index=idx)


def _make_snapshot(prev_close=100.0):
    """Build a mock snapshot with prev_daily_bar."""
    snap = types.SimpleNamespace(
        prev_daily_bar=types.SimpleNamespace(close=prev_close),
        latest_trade=types.SimpleNamespace(price=101.0),
    )
    return snap


# ---------------------------------------------------------------------------
# record_opening_range
# ---------------------------------------------------------------------------

class TestRecordOpeningRange:

    @patch("strategies.orb_v2.get_snapshot")
    def test_record_opening_range_valid(self, mock_snap):
        """A normal ORB with small gap and tight range is marked valid."""
        from strategies.orb_v2 import ORBStrategyV2

        mock_snap.return_value = _make_snapshot(prev_close=100.0)
        # range_pct = (101.5 - 99.0) / 100.25 ≈ 2.5% — within 3.5% limit
        bars = _make_bars(high=101.5, low=99.0, open_=100.0, close=101.0)

        strat = ORBStrategyV2()
        strat.record_opening_range("AAPL", bars)

        orb = strat.opening_ranges["AAPL"]
        assert orb["high"] == 101.5
        assert orb["low"] == 99.0
        assert orb["valid"] == True
        assert orb["established"] == True
        # V10 BUG-033: range_pct now uses orb_low as divisor (standard)
        assert orb["range_pct"] == pytest.approx((101.5 - 99.0) / 99.0, rel=0.01)

    @patch("strategies.orb_v2.get_snapshot")
    def test_record_opening_range_gap_filter(self, mock_snap):
        """ORB with gap > ORB_MAX_GAP_PCT is marked invalid."""
        from strategies.orb_v2 import ORBStrategyV2
        import config

        # Gap = |110 - 100| / 100 = 10%, well above 4% default
        mock_snap.return_value = _make_snapshot(prev_close=100.0)
        bars = _make_bars(high=112.0, low=108.0, open_=110.0, close=111.0)

        strat = ORBStrategyV2()
        strat.record_opening_range("AAPL", bars)

        orb = strat.opening_ranges["AAPL"]
        assert orb["valid"] == False
        assert orb["gap_pct"] > config.ORB_MAX_GAP_PCT

    @patch("strategies.orb_v2.get_snapshot")
    def test_record_opening_range_wide_range_filter(self, mock_snap):
        """ORB with range_pct > ORB_MAX_RANGE_PCT is marked invalid."""
        from strategies.orb_v2 import ORBStrategyV2
        import config

        # No gap, but range = (120 - 80) / 100 = 40% — way above 3.5%
        mock_snap.return_value = _make_snapshot(prev_close=100.0)
        bars = _make_bars(high=120.0, low=80.0, open_=100.0, close=100.0)

        strat = ORBStrategyV2()
        strat.record_opening_range("AAPL", bars)

        orb = strat.opening_ranges["AAPL"]
        assert orb["valid"] == False
        assert orb["range_pct"] > config.ORB_MAX_RANGE_PCT


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

class TestScan:

    def _setup_strategy_with_valid_range(self):
        """Return an ORBStrategyV2 with a pre-loaded valid range."""
        from strategies.orb_v2 import ORBStrategyV2
        strat = ORBStrategyV2()
        strat.opening_ranges["AAPL"] = {
            "high": 102.0,
            "low": 98.0,
            "range_pct": 0.04 / 100,  # ~0.04% — well within limits
            "gap_pct": 0.01,
            "valid": True,
            "established": True,
        }
        return strat

    @patch("strategies.orb_v2.get_intraday_bars")
    def test_scan_long_breakout(self, mock_bars):
        """Price above ORB high with volume ratio triggers a long signal."""
        strat = self._setup_strategy_with_valid_range()

        # Latest bar: close > 102 * 1.001, high volume
        now = datetime(2026, 3, 13, 10, 30, tzinfo=ET)
        bars = _make_bars(
            high=103.5, low=102.2, open_=102.3, close=103.0,
            volume=200_000, num_bars=12,
            base_time=now - timedelta(hours=1),
        )
        # Make last bar have much higher volume for vol_ratio > 1.3
        bars.iloc[-1, bars.columns.get_loc("volume")] = 500_000
        mock_bars.return_value = bars

        signals = strat.scan(now, regime="BULLISH")

        assert len(signals) == 1
        sig = signals[0]
        assert sig.symbol == "AAPL"
        assert sig.strategy == "ORB"
        assert sig.side == "buy"
        assert sig.hold_type == "day"
        assert sig.entry_price > 102.0
        assert sig.take_profit > sig.entry_price
        assert sig.stop_loss < sig.entry_price

    @patch("strategies.orb_v2.get_intraday_bars")
    def test_scan_no_signal_before_10am(self, mock_bars):
        """Scan returns nothing before 10:00 AM."""
        strat = self._setup_strategy_with_valid_range()

        now = datetime(2026, 3, 13, 9, 45, tzinfo=ET)
        signals = strat.scan(now, regime="BULLISH")

        assert signals == []
        mock_bars.assert_not_called()

    @patch("strategies.orb_v2.get_intraday_bars")
    def test_scan_no_signal_bearish_regime(self, mock_bars):
        """Scan returns nothing in BEARISH regime."""
        strat = self._setup_strategy_with_valid_range()

        now = datetime(2026, 3, 13, 10, 30, tzinfo=ET)
        signals = strat.scan(now, regime="BEARISH")

        assert signals == []
        mock_bars.assert_not_called()


# ---------------------------------------------------------------------------
# check_exits
# ---------------------------------------------------------------------------

class TestCheckExits:

    def test_check_exits_time_stop(self):
        """ORB trades older than 2 hours get a time stop exit."""
        from strategies.orb_v2 import ORBStrategyV2

        strat = ORBStrategyV2()
        now = datetime(2026, 3, 13, 12, 30, tzinfo=ET)

        trade = types.SimpleNamespace(
            symbol="AAPL",
            strategy="ORB",
            entry_time=datetime(2026, 3, 13, 10, 15, tzinfo=ET),
            side="buy",
        )

        exits = strat.check_exits({"AAPL": trade}, now)

        assert len(exits) == 1
        assert exits[0]["symbol"] == "AAPL"
        assert exits[0]["action"] == "full"
        assert exits[0]["reason"] == "orb_time_stop"

    def test_check_exits_no_exit_within_window(self):
        """ORB trades within the 2-hour window are NOT exited."""
        from strategies.orb_v2 import ORBStrategyV2

        strat = ORBStrategyV2()
        now = datetime(2026, 3, 13, 11, 0, tzinfo=ET)

        trade = types.SimpleNamespace(
            symbol="AAPL",
            strategy="ORB",
            entry_time=datetime(2026, 3, 13, 10, 15, tzinfo=ET),
            side="buy",
        )

        exits = strat.check_exits({"AAPL": trade}, now)
        assert exits == []

    def test_check_exits_ignores_non_orb(self):
        """Non-ORB trades are ignored by check_exits."""
        from strategies.orb_v2 import ORBStrategyV2

        strat = ORBStrategyV2()
        now = datetime(2026, 3, 13, 14, 0, tzinfo=ET)

        trade = types.SimpleNamespace(
            symbol="MSFT",
            strategy="VWAP",
            entry_time=datetime(2026, 3, 13, 10, 0, tzinfo=ET),
            side="buy",
        )

        exits = strat.check_exits({"MSFT": trade}, now)
        assert exits == []


# ---------------------------------------------------------------------------
# reset_daily
# ---------------------------------------------------------------------------

class TestResetDaily:

    def test_reset_daily(self):
        """reset_daily clears all internal state."""
        from strategies.orb_v2 import ORBStrategyV2

        strat = ORBStrategyV2()
        strat.opening_ranges["AAPL"] = {"high": 100, "low": 98, "valid": True}
        strat._universe = ["AAPL", "MSFT"]
        strat._trades_today = 5

        strat.reset_daily()

        assert strat.opening_ranges == {}
        assert strat._universe == []
        assert strat._trades_today == 0
