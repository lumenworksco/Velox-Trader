"""Tests for Post-Earnings Announcement Drift (PEAD) strategy."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from strategies.pead import PEADStrategy
from strategies.base import Signal
from tests.conftest import _make_trade

ET = ZoneInfo("America/New_York")

# A standard scan time: 9:05 AM ET on a weekday
SCAN_TIME = datetime(2026, 3, 16, 9, 5, tzinfo=ET)


def _mock_surprise(symbol="AAPL", surprise_pct=8.0, volume_ratio=3.0,
                   gap_pct=2.5, current_price=150.0):
    """Build a mock earnings surprise dict."""
    return {
        "symbol": symbol,
        "surprise_pct": surprise_pct,
        "volume_ratio": volume_ratio,
        "gap_pct": gap_pct,
        "current_price": current_price,
    }


# ===================================================================
# scan() tests
# ===================================================================

class TestPEADScan:
    """Tests for PEADStrategy.scan()."""

    def test_scan_returns_empty_when_no_earnings(self):
        """scan() returns empty list when no earnings surprises found."""
        strat = PEADStrategy()
        with patch.object(strat, "_get_earnings_surprises", return_value=[]):
            signals = strat.scan(SCAN_TIME, "NORMAL")
        assert signals == []

    def test_scan_returns_signal_for_qualifying_surprise(self):
        """scan() returns a Signal for a qualifying positive surprise."""
        strat = PEADStrategy()
        surprise = _mock_surprise("AAPL", surprise_pct=8.0, volume_ratio=3.0,
                                  gap_pct=2.5, current_price=150.0)
        with patch.object(strat, "_get_earnings_surprises", return_value=[surprise]):
            signals = strat.scan(SCAN_TIME, "NORMAL")
        assert len(signals) == 1
        sig = signals[0]
        assert sig.symbol == "AAPL"
        assert sig.strategy == "PEAD"
        assert sig.side == "buy"
        assert sig.hold_type == "swing"
        assert sig.entry_price == 150.0

    def test_scan_long_take_profit_and_stop_loss(self):
        """Long signal has correct TP (+5%) and SL (-3%)."""
        strat = PEADStrategy()
        surprise = _mock_surprise("AAPL", surprise_pct=10.0, volume_ratio=2.5,
                                  gap_pct=3.0, current_price=100.0)
        with patch.object(strat, "_get_earnings_surprises", return_value=[surprise]):
            signals = strat.scan(SCAN_TIME, "NORMAL")
        sig = signals[0]
        assert sig.take_profit == 105.0   # 100 * 1.05
        assert sig.stop_loss == 97.0      # 100 * 0.97

    def test_scan_short_signal_on_negative_surprise(self):
        """scan() returns short Signal for negative surprise with gap down."""
        strat = PEADStrategy()
        surprise = _mock_surprise("NVDA", surprise_pct=-7.0, volume_ratio=2.5,
                                  gap_pct=-3.0, current_price=200.0)
        with patch.object(strat, "_get_earnings_surprises", return_value=[surprise]), \
             patch("config.ALLOW_SHORT", True):
            signals = strat.scan(SCAN_TIME, "NORMAL")
        assert len(signals) == 1
        sig = signals[0]
        assert sig.side == "sell"
        assert sig.take_profit == 190.0   # 200 * 0.95
        assert sig.stop_loss == 206.0     # 200 * 1.03

    def test_scan_filters_small_surprise(self):
        """scan() filters out surprises below PEAD_MIN_SURPRISE_PCT (5%)."""
        strat = PEADStrategy()
        surprise = _mock_surprise("AAPL", surprise_pct=3.0, volume_ratio=3.0,
                                  gap_pct=1.0, current_price=150.0)
        with patch.object(strat, "_get_earnings_surprises", return_value=[surprise]):
            signals = strat.scan(SCAN_TIME, "NORMAL")
        assert signals == []

    def test_scan_filters_low_volume(self):
        """scan() filters out low volume ratio (< 2x)."""
        strat = PEADStrategy()
        surprise = _mock_surprise("AAPL", surprise_pct=8.0, volume_ratio=1.5,
                                  gap_pct=2.0, current_price=150.0)
        with patch.object(strat, "_get_earnings_surprises", return_value=[surprise]):
            signals = strat.scan(SCAN_TIME, "NORMAL")
        assert signals == []

    def test_scan_filters_mismatched_gap(self):
        """scan() filters out when surprise and gap directions mismatch."""
        strat = PEADStrategy()
        # Positive surprise but negative gap
        surprise = _mock_surprise("AAPL", surprise_pct=8.0, volume_ratio=3.0,
                                  gap_pct=-2.0, current_price=150.0)
        with patch.object(strat, "_get_earnings_surprises", return_value=[surprise]):
            signals = strat.scan(SCAN_TIME, "NORMAL")
        assert signals == []

    def test_scan_respects_high_vol_bear_regime(self):
        """scan() returns empty list in HIGH_VOL_BEAR regime."""
        strat = PEADStrategy()
        surprise = _mock_surprise("AAPL", surprise_pct=10.0, volume_ratio=3.0,
                                  gap_pct=3.0, current_price=150.0)
        with patch.object(strat, "_get_earnings_surprises", return_value=[surprise]):
            signals = strat.scan(SCAN_TIME, "HIGH_VOL_BEAR")
        assert signals == []

    def test_scan_only_runs_once_per_day(self):
        """scan() only executes once per day; subsequent calls return empty."""
        strat = PEADStrategy()
        surprise = _mock_surprise("AAPL", surprise_pct=8.0, volume_ratio=3.0,
                                  gap_pct=2.0, current_price=150.0)
        with patch.object(strat, "_get_earnings_surprises", return_value=[surprise]):
            signals1 = strat.scan(SCAN_TIME, "NORMAL")
            signals2 = strat.scan(SCAN_TIME, "NORMAL")
        assert len(signals1) == 1
        assert signals2 == []

    def test_scan_before_9am_returns_empty(self):
        """scan() returns empty before 9:00 AM."""
        strat = PEADStrategy()
        early = datetime(2026, 3, 16, 8, 30, tzinfo=ET)
        surprise = _mock_surprise("AAPL", surprise_pct=8.0, volume_ratio=3.0,
                                  gap_pct=2.0, current_price=150.0)
        with patch.object(strat, "_get_earnings_surprises", return_value=[surprise]):
            signals = strat.scan(early, "NORMAL")
        assert signals == []

    def test_scan_disabled_returns_empty(self):
        """scan() returns empty when PEAD_ENABLED is False."""
        strat = PEADStrategy()
        surprise = _mock_surprise("AAPL", surprise_pct=8.0, volume_ratio=3.0,
                                  gap_pct=2.0, current_price=150.0)
        with patch.object(strat, "_get_earnings_surprises", return_value=[surprise]), \
             patch("config.PEAD_ENABLED", False):
            signals = strat.scan(SCAN_TIME, "NORMAL")
        assert signals == []

    def test_scan_max_positions_limit(self):
        """scan() respects PEAD_MAX_POSITIONS limit."""
        strat = PEADStrategy()
        # Pre-fill triggered to max
        for i in range(5):
            strat.triggered[f"SYM{i}"] = SCAN_TIME
        surprise = _mock_surprise("AAPL", surprise_pct=8.0, volume_ratio=3.0,
                                  gap_pct=2.0, current_price=150.0)
        with patch.object(strat, "_get_earnings_surprises", return_value=[surprise]):
            signals = strat.scan(SCAN_TIME, "NORMAL")
        assert signals == []

    def test_scan_limits_new_signals_to_remaining_slots(self):
        """scan() only fills remaining slots up to PEAD_MAX_POSITIONS."""
        strat = PEADStrategy()
        # 4 existing positions, max 5 -> only 1 slot
        for i in range(4):
            strat.triggered[f"SYM{i}"] = SCAN_TIME
        surprises = [
            _mock_surprise("AAPL", surprise_pct=8.0, volume_ratio=3.0,
                           gap_pct=2.0, current_price=150.0),
            _mock_surprise("MSFT", surprise_pct=10.0, volume_ratio=4.0,
                           gap_pct=3.0, current_price=300.0),
        ]
        with patch.object(strat, "_get_earnings_surprises", return_value=surprises):
            signals = strat.scan(SCAN_TIME, "NORMAL")
        assert len(signals) == 1

    def test_scan_short_blocked_when_allow_short_false(self):
        """Short signals are blocked when ALLOW_SHORT is False."""
        strat = PEADStrategy()
        surprise = _mock_surprise("NVDA", surprise_pct=-7.0, volume_ratio=3.0,
                                  gap_pct=-2.0, current_price=200.0)
        with patch.object(strat, "_get_earnings_surprises", return_value=[surprise]), \
             patch("config.ALLOW_SHORT", False):
            signals = strat.scan(SCAN_TIME, "NORMAL")
        assert signals == []

    def test_scan_hold_type_is_swing(self):
        """All PEAD signals have hold_type='swing'."""
        strat = PEADStrategy()
        surprise = _mock_surprise("AAPL", surprise_pct=8.0, volume_ratio=3.0,
                                  gap_pct=2.0, current_price=150.0)
        with patch.object(strat, "_get_earnings_surprises", return_value=[surprise]):
            signals = strat.scan(SCAN_TIME, "NORMAL")
        assert signals[0].hold_type == "swing"

    def test_scan_records_trigger_time(self):
        """scan() records symbol in triggered dict to prevent re-entry."""
        strat = PEADStrategy()
        surprise = _mock_surprise("AAPL", surprise_pct=8.0, volume_ratio=3.0,
                                  gap_pct=2.0, current_price=150.0)
        with patch.object(strat, "_get_earnings_surprises", return_value=[surprise]):
            strat.scan(SCAN_TIME, "NORMAL")
        assert "AAPL" in strat.triggered
        assert strat.triggered["AAPL"] == SCAN_TIME


# ===================================================================
# check_exits() tests
# ===================================================================

class TestPEADExits:
    """Tests for PEADStrategy.check_exits()."""

    def test_check_exits_time_stop(self):
        """check_exits() triggers time stop after PEAD_HOLD_DAYS_MAX days."""
        strat = PEADStrategy()
        entry_time = SCAN_TIME - timedelta(days=21)
        trade = _make_trade(
            symbol="AAPL", strategy="PEAD", side="buy",
            entry_price=150.0, entry_time=entry_time,
            hold_type="swing",
        )
        exits = strat.check_exits({"AAPL": trade}, SCAN_TIME)
        assert len(exits) == 1
        assert exits[0]["symbol"] == "AAPL"
        assert exits[0]["action"] == "full"
        assert "time stop" in exits[0]["reason"]

    def test_check_exits_take_profit(self):
        """check_exits() triggers take profit at 5% gain."""
        strat = PEADStrategy()
        entry_time = SCAN_TIME - timedelta(days=5)
        trade = _make_trade(
            symbol="AAPL", strategy="PEAD", side="buy",
            entry_price=100.0, entry_time=entry_time,
            hold_type="swing", highest_price_seen=105.5,
        )
        mock_snap = MagicMock()
        mock_snap.latest_trade.price = 105.5
        with patch("data.get_snapshot", return_value=mock_snap):
            exits = strat.check_exits({"AAPL": trade}, SCAN_TIME)
        assert len(exits) == 1
        assert "take profit" in exits[0]["reason"]

    def test_check_exits_stop_loss(self):
        """check_exits() triggers stop loss at 3% loss."""
        strat = PEADStrategy()
        entry_time = SCAN_TIME - timedelta(days=2)
        trade = _make_trade(
            symbol="AAPL", strategy="PEAD", side="buy",
            entry_price=100.0, entry_time=entry_time,
            hold_type="swing", highest_price_seen=96.5,
        )
        mock_snap = MagicMock()
        mock_snap.latest_trade.price = 96.5
        with patch("data.get_snapshot", return_value=mock_snap):
            exits = strat.check_exits({"AAPL": trade}, SCAN_TIME)
        assert len(exits) == 1
        assert "stop loss" in exits[0]["reason"]

    def test_check_exits_ignores_non_pead_trades(self):
        """check_exits() skips trades with strategy != PEAD."""
        strat = PEADStrategy()
        trade = _make_trade(
            symbol="AAPL", strategy="ORB", side="buy",
            entry_price=150.0, hold_type="day",
        )
        exits = strat.check_exits({"AAPL": trade}, SCAN_TIME)
        assert exits == []

    def test_check_exits_no_exit_within_bounds(self):
        """check_exits() returns nothing when trade is within bounds."""
        strat = PEADStrategy()
        entry_time = SCAN_TIME - timedelta(days=3)
        trade = _make_trade(
            symbol="AAPL", strategy="PEAD", side="buy",
            entry_price=100.0, entry_time=entry_time,
            hold_type="swing", highest_price_seen=102.0,
        )
        exits = strat.check_exits({"AAPL": trade}, SCAN_TIME)
        assert exits == []


# ===================================================================
# reset_daily() tests
# ===================================================================

class TestPEADReset:
    """Tests for PEADStrategy.reset_daily()."""

    def test_reset_daily_clears_scan_flag(self):
        """reset_daily() allows scan to run again."""
        strat = PEADStrategy()
        strat._scanned_today = True
        strat._candidates = [{"symbol": "AAPL"}]
        strat.reset_daily()
        assert strat._scanned_today is False
        assert strat._candidates == []

    def test_reset_daily_preserves_triggered(self):
        """reset_daily() preserves triggered dict (multi-day tracking)."""
        strat = PEADStrategy()
        strat.triggered["AAPL"] = SCAN_TIME
        strat.reset_daily()
        assert "AAPL" in strat.triggered


# ===================================================================
# _get_earnings_surprises() tests
# ===================================================================

class TestPEADEarningsFetch:
    """Tests for _get_earnings_surprises (mocked yfinance)."""

    def test_get_earnings_surprises_returns_empty_on_import_error(self):
        """Returns empty list if yfinance is not importable."""
        strat = PEADStrategy()
        with patch.dict("sys.modules", {"yfinance": None}):
            # Force ImportError by removing yfinance from import path
            import builtins
            real_import = builtins.__import__
            def mock_import(name, *args, **kwargs):
                if name == "yfinance":
                    raise ImportError("No module named 'yfinance'")
                return real_import(name, *args, **kwargs)
            with patch("builtins.__import__", side_effect=mock_import):
                result = strat._get_earnings_surprises(["AAPL"])
        assert result == []

    def test_get_earnings_surprises_fail_open(self):
        """Returns empty list on any exception (fail-open)."""
        strat = PEADStrategy()
        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = Exception("API error")
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = strat._get_earnings_surprises(["AAPL"])
        assert result == []
