"""Comprehensive tests for compliance/pdt.py — Pattern Day Trader compliance.

Tests cover:
- can_day_trade() with sufficient equity (>$25K)
- can_day_trade() blocked at PDT limit (3 trades in 5 days)
- Day trade recording and window tracking
- PDT restriction triggering and status reporting
- Thread safety
"""

import sys
import os
import tempfile
import threading
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_config():
    """Provide a minimal mock config module."""
    import types
    cfg = types.ModuleType("config")
    cfg.ET = ET
    with patch.dict(sys.modules, {"config": cfg}):
        yield cfg


@pytest.fixture
def pdt_module(_mock_config):
    """Import and return a fresh pdt module."""
    import importlib
    mod = importlib.import_module("compliance.pdt")
    importlib.reload(mod)
    return mod


@pytest.fixture
def pdt_instance(pdt_module, tmp_path):
    """Return a fresh PDTCompliance with a temp log file."""
    log_file = str(tmp_path / "pdt_test.jsonl")
    return pdt_module.PDTCompliance(log_path=log_file)


# ===================================================================
# can_day_trade: equity above $25K (exempt)
# ===================================================================

class TestCanDayTradeAbove25K:
    """Accounts >= $25K are exempt from PDT limits."""

    def test_above_25k_always_allowed(self, pdt_instance):
        allowed, remaining, reason = pdt_instance.can_day_trade(equity=30_000.0)
        assert allowed is True
        assert remaining == 999
        assert "25k" in reason.lower()

    def test_exactly_25k_is_exempt(self, pdt_instance):
        allowed, remaining, reason = pdt_instance.can_day_trade(equity=25_000.0)
        assert allowed is True
        assert remaining == 999

    def test_high_equity_with_trades(self, pdt_instance):
        """Even with 3 day trades, high equity means no limit."""
        today = datetime.now(ET).date()
        for i in range(3):
            pdt_instance.record_day_trade(f"SYM{i}", trade_date=today)

        allowed, remaining, reason = pdt_instance.can_day_trade(equity=50_000.0)
        assert allowed is True
        assert remaining == 999


# ===================================================================
# can_day_trade: below $25K, within limit
# ===================================================================

class TestCanDayTradeBelow25K:
    """Accounts < $25K are subject to 3 day trades per 5 business days."""

    def test_no_trades_full_allowance(self, pdt_instance):
        allowed, remaining, reason = pdt_instance.can_day_trade(equity=20_000.0)
        assert allowed is True
        assert remaining == 3

    def test_one_trade_two_remaining(self, pdt_instance):
        today = datetime.now(ET).date()
        pdt_instance.record_day_trade("AAPL", trade_date=today)

        allowed, remaining, reason = pdt_instance.can_day_trade(equity=20_000.0)
        assert allowed is True
        assert remaining == 2

    def test_two_trades_one_remaining_with_caution(self, pdt_instance):
        today = datetime.now(ET).date()
        pdt_instance.record_day_trade("AAPL", trade_date=today)
        pdt_instance.record_day_trade("GOOG", trade_date=today)

        allowed, remaining, reason = pdt_instance.can_day_trade(equity=20_000.0)
        assert allowed is True
        assert remaining == 1
        assert "caution" in reason.lower() or "last" in reason.lower()


# ===================================================================
# can_day_trade: blocked at PDT limit
# ===================================================================

class TestCanDayTradeBlocked:
    """3 trades in the window blocks further day trades."""

    def test_three_trades_blocks(self, pdt_instance):
        today = datetime.now(ET).date()
        pdt_instance.record_day_trade("AAPL", trade_date=today)
        pdt_instance.record_day_trade("GOOG", trade_date=today)
        pdt_instance.record_day_trade("MSFT", trade_date=today)

        allowed, remaining, reason = pdt_instance.can_day_trade(equity=20_000.0)
        assert allowed is False
        assert remaining == 0
        assert "limit" in reason.lower()

    def test_trades_from_old_window_dont_count(self, pdt_instance):
        """Trades outside the 5-business-day window should not count."""
        # Record trades 10 business days ago (well outside window)
        old_date = datetime.now(ET).date() - timedelta(days=14)
        pdt_instance.record_day_trade("AAPL", trade_date=old_date)
        pdt_instance.record_day_trade("GOOG", trade_date=old_date)
        pdt_instance.record_day_trade("MSFT", trade_date=old_date)

        allowed, remaining, reason = pdt_instance.can_day_trade(equity=20_000.0)
        assert allowed is True
        assert remaining == 3


# ===================================================================
# PDT restriction
# ===================================================================

class TestPDTRestriction:
    """Exceeding 3 day trades triggers restriction."""

    def test_fourth_trade_triggers_restriction(self, pdt_instance):
        today = datetime.now(ET).date()
        for i in range(4):
            pdt_instance.record_day_trade(f"SYM{i}", trade_date=today)

        assert pdt_instance._pdt_restricted is True

    def test_restricted_account_is_blocked(self, pdt_instance):
        today = datetime.now(ET).date()
        for i in range(4):
            pdt_instance.record_day_trade(f"SYM{i}", trade_date=today)

        allowed, remaining, reason = pdt_instance.can_day_trade(equity=20_000.0)
        assert allowed is False
        assert "restriction" in reason.lower()

    def test_reset_restriction_clears_block(self, pdt_instance):
        today = datetime.now(ET).date()
        for i in range(4):
            pdt_instance.record_day_trade(f"SYM{i}", trade_date=today)
        assert pdt_instance._pdt_restricted is True

        pdt_instance.reset_restriction()
        assert pdt_instance._pdt_restricted is False

        # Now only 3 trades count in window, but the 4th made it restricted
        # After reset, should be blocked by trade count but not by restriction flag
        allowed, remaining, reason = pdt_instance.can_day_trade(equity=20_000.0)
        # 4 trades in window -> 0 remaining, still blocked by count
        assert allowed is False

    def test_restriction_does_not_affect_high_equity(self, pdt_instance):
        today = datetime.now(ET).date()
        for i in range(4):
            pdt_instance.record_day_trade(f"SYM{i}", trade_date=today)

        # Even with restriction flag, $30k equity should be exempt
        allowed, remaining, reason = pdt_instance.can_day_trade(equity=30_000.0)
        assert allowed is True


# ===================================================================
# Day trade recording
# ===================================================================

class TestDayTradeRecording:
    """record_day_trade persists and tracks trades correctly."""

    def test_record_increments_count(self, pdt_instance):
        today = datetime.now(ET).date()
        pdt_instance.record_day_trade("AAPL", trade_date=today)
        trades = pdt_instance.get_trades_in_window()
        assert len(trades) == 1
        assert trades[0].symbol == "AAPL"

    def test_record_with_details(self, pdt_instance):
        today = datetime.now(ET).date()
        pdt_instance.record_day_trade(
            "GOOG", trade_date=today, side="buy",
            qty=50.0, entry_price=175.0, exit_price=178.0,
        )
        trades = pdt_instance.get_trades_in_window()
        assert len(trades) == 1
        assert trades[0].side == "buy"
        assert trades[0].qty == 50.0
        assert trades[0].entry_price == 175.0

    def test_default_date_is_today(self, pdt_instance):
        pdt_instance.record_day_trade("AAPL")
        trades = pdt_instance.get_trades_in_window()
        assert len(trades) == 1
        assert trades[0].trade_date == datetime.now(ET).date()

    def test_old_records_are_pruned(self, pdt_instance):
        """Records older than 90 days should be pruned."""
        very_old = datetime.now(ET).date() - timedelta(days=100)
        pdt_instance.record_day_trade("OLD", trade_date=very_old)

        # Recording a new trade triggers pruning
        today = datetime.now(ET).date()
        pdt_instance.record_day_trade("NEW", trade_date=today)

        # OLD should be pruned, only NEW remains
        all_trades = pdt_instance._day_trades
        symbols = {t.symbol for t in all_trades}
        assert "OLD" not in symbols
        assert "NEW" in symbols


# ===================================================================
# get_status
# ===================================================================

class TestGetStatus:
    """get_status returns comprehensive PDT status."""

    def test_status_reflects_current_state(self, pdt_instance, pdt_module):
        today = datetime.now(ET).date()
        pdt_instance.record_day_trade("AAPL", trade_date=today)
        pdt_instance.record_day_trade("GOOG", trade_date=today)

        status = pdt_instance.get_status(equity=20_000.0)
        assert isinstance(status, pdt_module.PDTStatus)
        assert status.can_day_trade is True
        assert status.remaining_day_trades == 1
        assert status.day_trades_in_window == 2
        assert status.equity == 20_000.0
        assert status.is_pdt_restricted is False

    def test_status_when_restricted(self, pdt_instance, pdt_module):
        today = datetime.now(ET).date()
        for i in range(4):
            pdt_instance.record_day_trade(f"SYM{i}", trade_date=today)

        status = pdt_instance.get_status(equity=20_000.0)
        assert status.can_day_trade is False
        assert status.is_pdt_restricted is True

    def test_status_high_equity(self, pdt_instance, pdt_module):
        status = pdt_instance.get_status(equity=50_000.0)
        assert status.can_day_trade is True
        assert status.remaining_day_trades == 999


# ===================================================================
# Alert callback
# ===================================================================

class TestAlertCallback:
    """Alert callback fires on PDT violations and warnings."""

    def test_alert_on_violation(self, pdt_module, tmp_path):
        alert_mock = MagicMock()
        pdt = pdt_module.PDTCompliance(
            log_path=str(tmp_path / "pdt.jsonl"),
            alert_callback=alert_mock,
        )
        today = datetime.now(ET).date()
        for i in range(4):
            pdt.record_day_trade(f"SYM{i}", trade_date=today)

        # Should fire EMERGENCY alert on 4th trade
        calls = alert_mock.call_args_list
        emergency_calls = [c for c in calls if c[0][0] == "EMERGENCY"]
        assert len(emergency_calls) >= 1

    def test_warning_at_limit(self, pdt_module, tmp_path):
        alert_mock = MagicMock()
        pdt = pdt_module.PDTCompliance(
            log_path=str(tmp_path / "pdt.jsonl"),
            alert_callback=alert_mock,
        )
        today = datetime.now(ET).date()
        for i in range(3):
            pdt.record_day_trade(f"SYM{i}", trade_date=today)

        calls = alert_mock.call_args_list
        warning_calls = [c for c in calls if c[0][0] == "WARNING"]
        assert len(warning_calls) >= 1


# ===================================================================
# Thread safety
# ===================================================================

class TestThreadSafety:
    """PDTCompliance uses RLock for thread-safe access."""

    def test_concurrent_record_and_check(self, pdt_instance):
        """Multiple threads recording and checking simultaneously."""
        results = []

        def worker(sym_prefix, equity):
            for i in range(3):
                pdt_instance.record_day_trade(f"{sym_prefix}{i}")
                allowed, remaining, reason = pdt_instance.can_day_trade(equity)
                results.append(allowed)

        threads = [
            threading.Thread(target=worker, args=("A", 20_000.0)),
            threading.Thread(target=worker, args=("B", 20_000.0)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No crashes; all calls returned a bool
        assert all(isinstance(r, bool) for r in results)

    def test_lock_is_reentrant(self, pdt_instance):
        """can_day_trade + get_status should not deadlock (uses RLock)."""
        # get_status calls _check_inner internally, which also acquires lock
        status = pdt_instance.get_status(equity=20_000.0)
        assert status is not None
