"""Tests for oms/kill_switch.py — emergency halt, queue persistence, recovery."""

import os
os.environ.setdefault("TESTING", "1")

import json
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch, call

import pytest

import config


# ===================================================================
# activate
# ===================================================================

class TestKillSwitchActivate:
    """Tests for KillSwitch.activate — cancel orders + close positions."""

    @patch("oms.kill_switch.log_event")
    @patch("oms.kill_switch.os.path.exists", return_value=False)
    def test_activate_cancels_all_orders(self, mock_exists, mock_log):
        from oms.kill_switch import KillSwitch
        ks = KillSwitch()
        mock_om = MagicMock()
        mock_om.cancel_all.return_value = ["oms-1", "oms-2"]

        ks.activate(reason="test", order_manager=mock_om)

        assert ks.active is True
        assert ks.reason == "test"
        mock_om.cancel_all.assert_called_once()

    @patch("oms.kill_switch.log_event")
    @patch("oms.kill_switch.os.path.exists", return_value=False)
    def test_activate_closes_all_positions(self, mock_exists, mock_log):
        from oms.kill_switch import KillSwitch
        ks = KillSwitch()

        mock_rm = MagicMock()
        mock_rm.open_trades = {"AAPL": MagicMock(entry_price=150.0),
                               "MSFT": MagicMock(entry_price=400.0)}

        with patch("oms.kill_switch._time.sleep"), \
             patch("oms.kill_switch.KillSwitch._write_queue"), \
             patch("oms.kill_switch.KillSwitch._remove_from_queue"):
            with patch("execution.close_position", return_value=True) as mock_close:
                ks.activate(reason="drawdown", risk_manager=mock_rm)

            assert ks.active is True
            assert mock_close.call_count == 2
            # Verify close_trade was called for both symbols with correct params
            assert mock_rm.close_trade.call_count == 2
            close_trade_symbols = {c.args[0] for c in mock_rm.close_trade.call_args_list}
            assert close_trade_symbols == {"AAPL", "MSFT"}
            # All calls should have exit_reason="kill_switch"
            for c in mock_rm.close_trade.call_args_list:
                assert c.kwargs.get("exit_reason") == "kill_switch"

    @patch("oms.kill_switch.log_event")
    @patch("oms.kill_switch.os.path.exists", return_value=False)
    def test_activate_idempotent_when_already_active(self, mock_exists, mock_log):
        from oms.kill_switch import KillSwitch
        ks = KillSwitch()
        ks.active = True

        mock_om = MagicMock()
        ks.activate(reason="again", order_manager=mock_om)
        # Should return early without cancelling
        mock_om.cancel_all.assert_not_called()

    @patch("oms.kill_switch.log_event")
    @patch("oms.kill_switch.os.path.exists", return_value=False)
    def test_activate_handles_close_position_failure(self, mock_exists, mock_log):
        """When close_position returns False, the symbol is tracked as failed."""
        from oms.kill_switch import KillSwitch
        ks = KillSwitch()

        mock_rm = MagicMock()
        mock_rm.open_trades = {"FAIL_SYM": MagicMock(entry_price=100.0)}

        with patch("oms.kill_switch._time.sleep"), \
             patch("oms.kill_switch.KillSwitch._write_queue"), \
             patch("oms.kill_switch.KillSwitch._remove_from_queue"):
            with patch("execution.close_position", return_value=False):
                ks.activate(reason="test_fail", risk_manager=mock_rm)

            assert ks.active is True
            # close_trade should NOT be called for the failed symbol
            mock_rm.close_trade.assert_not_called()

    @patch("oms.kill_switch.log_event")
    @patch("oms.kill_switch.os.path.exists", return_value=False)
    def test_activate_handles_close_position_exception(self, mock_exists, mock_log):
        """When close_position raises, the symbol is tracked as failed."""
        from oms.kill_switch import KillSwitch
        ks = KillSwitch()

        mock_rm = MagicMock()
        mock_rm.open_trades = {"EXC_SYM": MagicMock(entry_price=100.0)}

        with patch("oms.kill_switch._time.sleep"), \
             patch("oms.kill_switch.KillSwitch._write_queue"), \
             patch("oms.kill_switch.KillSwitch._remove_from_queue"):
            with patch("execution.close_position", side_effect=Exception("boom")):
                ks.activate(reason="test_exc", risk_manager=mock_rm)

            assert ks.active is True
            mock_rm.close_trade.assert_not_called()


# ===================================================================
# deactivate
# ===================================================================

class TestKillSwitchDeactivate:
    """Tests for KillSwitch.deactivate — re-enable trading."""

    @patch("oms.kill_switch.os.path.exists", return_value=False)
    def test_deactivate_re_enables_trading(self, mock_exists):
        from oms.kill_switch import KillSwitch
        ks = KillSwitch()
        ks.active = True
        ks.activated_at = datetime.now(config.ET)
        ks.reason = "test"

        ks.deactivate()

        assert ks.active is False
        assert ks.activated_at is None
        assert ks.reason == ""

    @patch("oms.kill_switch.os.path.exists", return_value=False)
    def test_deactivate_noop_when_not_active(self, mock_exists):
        from oms.kill_switch import KillSwitch
        ks = KillSwitch()
        assert ks.active is False
        ks.deactivate()  # Should not raise
        assert ks.active is False

    @patch("oms.kill_switch.os.path.exists", return_value=False)
    def test_is_trading_allowed_reflects_state(self, mock_exists):
        from oms.kill_switch import KillSwitch
        ks = KillSwitch()
        assert ks.is_trading_allowed() is True

        ks.active = True
        assert ks.is_trading_allowed() is False

        ks.deactivate()
        assert ks.is_trading_allowed() is True


# ===================================================================
# Queue persistence (atomic write)
# ===================================================================

class TestQueuePersistence:
    """Tests for _write_queue, _load_queue, _remove_from_queue."""

    def test_write_and_load_queue(self, tmp_path):
        """Write a queue to disk and load it back."""
        from oms.kill_switch import KillSwitch

        queue_file = str(tmp_path / "kill_switch_queue.json")
        with patch("oms.kill_switch._QUEUE_FILE", queue_file):
            KillSwitch._write_queue(["AAPL", "MSFT", "GOOG"], reason="test_write")

            symbols, reason = KillSwitch._load_queue()
            assert symbols == ["AAPL", "MSFT", "GOOG"]
            assert reason == "test_write"

    def test_remove_from_queue(self, tmp_path):
        """Remove a single symbol and verify the file is updated."""
        from oms.kill_switch import KillSwitch

        queue_file = str(tmp_path / "kill_switch_queue.json")
        with patch("oms.kill_switch._QUEUE_FILE", queue_file):
            KillSwitch._write_queue(["AAPL", "MSFT", "GOOG"], reason="test_rm")
            KillSwitch._remove_from_queue("MSFT")

            symbols, reason = KillSwitch._load_queue()
            assert "MSFT" not in symbols
            assert "AAPL" in symbols
            assert "GOOG" in symbols

    def test_remove_last_symbol_deletes_file(self, tmp_path):
        """Removing the last symbol should delete the queue file."""
        from oms.kill_switch import KillSwitch

        queue_file = str(tmp_path / "kill_switch_queue.json")
        with patch("oms.kill_switch._QUEUE_FILE", queue_file):
            KillSwitch._write_queue(["AAPL"], reason="test_last")
            KillSwitch._remove_from_queue("AAPL")
            assert not os.path.exists(queue_file)

    def test_load_queue_missing_file_returns_empty(self, tmp_path):
        """If queue file doesn't exist, return empty list."""
        from oms.kill_switch import KillSwitch

        queue_file = str(tmp_path / "nonexistent.json")
        with patch("oms.kill_switch._QUEUE_FILE", queue_file):
            symbols, reason = KillSwitch._load_queue()
            assert symbols == []
            assert reason == ""

    def test_write_queue_atomic_via_rename(self, tmp_path):
        """Verify the queue file is written atomically (exists after write)."""
        from oms.kill_switch import KillSwitch

        queue_file = str(tmp_path / "kill_switch_queue.json")
        with patch("oms.kill_switch._QUEUE_FILE", queue_file):
            KillSwitch._write_queue(["SPY"], reason="atomic_test")
            assert os.path.exists(queue_file)
            with open(queue_file) as f:
                data = json.load(f)
            assert data["symbols"] == ["SPY"]
            assert data["reason"] == "atomic_test"
            assert "timestamp" in data


# ===================================================================
# Residual queue processing on restart
# ===================================================================

class TestResidualQueueProcessing:
    """Tests for _process_residual_queue — crash recovery."""

    def test_residual_queue_closes_positions(self, tmp_path):
        """On startup, close any positions left in the queue."""
        from oms.kill_switch import KillSwitch

        queue_file = str(tmp_path / "kill_switch_queue.json")
        with open(queue_file, "w") as f:
            json.dump({"symbols": ["AAPL", "MSFT"], "reason": "test_crash"}, f)

        with patch("oms.kill_switch._QUEUE_FILE", queue_file), \
             patch("execution.close_position", return_value=True) as mock_close:
            ks = KillSwitch()
            assert mock_close.call_count == 2

    def test_residual_queue_handles_close_failure(self, tmp_path):
        """If close_position fails during recovery, the symbol stays in queue."""
        from oms.kill_switch import KillSwitch

        queue_file = str(tmp_path / "kill_switch_queue.json")
        with open(queue_file, "w") as f:
            json.dump({"symbols": ["FAIL_SYM"], "reason": "crash"}, f)

        with patch("oms.kill_switch._QUEUE_FILE", queue_file), \
             patch("execution.close_position", return_value=False) as mock_close:
            ks = KillSwitch()
            mock_close.assert_called_once()
            # File should still exist since close failed
            assert os.path.exists(queue_file)

    def test_no_residual_queue_is_noop(self, tmp_path):
        """If no queue file exists, startup is a no-op."""
        from oms.kill_switch import KillSwitch

        queue_file = str(tmp_path / "nonexistent.json")
        with patch("oms.kill_switch._QUEUE_FILE", queue_file):
            ks = KillSwitch()  # Should not raise
            assert ks.active is False


# ===================================================================
# Status property
# ===================================================================

class TestKillSwitchStatus:
    """Tests for the status property."""

    @patch("oms.kill_switch.os.path.exists", return_value=False)
    def test_status_when_inactive(self, mock_exists):
        from oms.kill_switch import KillSwitch
        ks = KillSwitch()
        status = ks.status
        assert status["active"] is False
        assert status["activated_at"] is None
        assert status["reason"] == ""

    @patch("oms.kill_switch.os.path.exists", return_value=False)
    def test_status_when_active(self, mock_exists):
        from oms.kill_switch import KillSwitch
        ks = KillSwitch()
        ks.active = True
        ks.activated_at = datetime(2026, 4, 4, 10, 0, tzinfo=config.ET)
        ks.reason = "drawdown"

        status = ks.status
        assert status["active"] is True
        assert "2026-04-04" in status["activated_at"]
        assert status["reason"] == "drawdown"
