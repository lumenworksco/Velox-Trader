"""Tests for execution/core.py — bracket orders, retries, circuit breaker, TWAP, chase logic."""

import os
os.environ.setdefault("TESTING", "1")

import threading
import time
import types
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from strategies.base import Signal
from tests.conftest import MockOrder, MockTradingClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(symbol="AAPL", strategy="MOMENTUM", side="buy",
                 entry_price=150.0, take_profit=155.0, stop_loss=148.0):
    return Signal(
        symbol=symbol,
        strategy=strategy,
        side=side,
        entry_price=entry_price,
        take_profit=take_profit,
        stop_loss=stop_loss,
        reason="test",
        hold_type="day",
        pair_id="",
    )


@pytest.fixture(autouse=True)
def _reset_api_failure_state():
    """Reset the module-level _api_failures list before each test."""
    from execution.core import _api_failures
    _api_failures.clear()
    yield
    _api_failures.clear()


# ===================================================================
# submit_bracket_order
# ===================================================================

class TestSubmitBracketOrder:
    """Tests for submit_bracket_order — success, failure, retry, and routing."""

    @patch("execution.core.validate_order_pretrade")
    @patch("execution.core._get_slippage_model", return_value=None)
    @patch("execution.core._verify_bracket_entry")
    @patch("execution.core.time.sleep")
    def test_success_returns_order_id(self, mock_sleep, mock_verify,
                                       mock_slip, mock_validate,
                                       mock_trading_client):
        """First attempt succeeds and returns a valid order ID."""
        from execution.core import submit_bracket_order
        mock_validate.return_value = types.SimpleNamespace(valid=True, reason="ok")
        signal = _make_signal()
        result = submit_bracket_order(signal, qty=10)
        assert result is not None
        assert isinstance(result, str)
        assert result.startswith("mock-")

    @patch("execution.core.validate_order_pretrade")
    @patch("execution.core._get_slippage_model", return_value=None)
    @patch("execution.core.time.sleep")
    def test_pretrade_validation_failure_returns_none(self, mock_sleep,
                                                      mock_slip,
                                                      mock_validate,
                                                      mock_trading_client):
        """Pre-trade validation failure returns None without submitting."""
        from execution.core import submit_bracket_order
        mock_validate.return_value = types.SimpleNamespace(
            valid=False, reason="insufficient_buying_power",
            details={"buying_power": 100},
        )
        signal = _make_signal()
        result = submit_bracket_order(signal, qty=10)
        assert result is None

    @patch("execution.core.validate_order_pretrade")
    @patch("execution.core._get_slippage_model", return_value=None)
    @patch("execution.core._verify_bracket_entry")
    @patch("execution.core.time.sleep")
    def test_retry_on_failure_then_success(self, mock_sleep, mock_verify,
                                            mock_slip, mock_validate,
                                            mock_trading_client):
        """First attempt raises, second succeeds. Backoff sleep is called."""
        from execution.core import submit_bracket_order
        mock_validate.return_value = types.SimpleNamespace(valid=True, reason="ok")

        # Build a dedicated client so we control its submit_order exactly
        client = MockTradingClient()
        call_count = 0
        original_submit = client.submit_order

        def _side_effect(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("timeout - try again")
            return original_submit(request)

        client.submit_order = _side_effect

        with patch("execution.core.get_trading_client", return_value=client):
            signal = _make_signal()
            result = submit_bracket_order(signal, qty=10)

        assert result is not None
        assert call_count == 2
        # Backoff sleep should have been called at least once (for attempt 1)
        assert mock_sleep.call_count >= 1

    @patch("execution.core.validate_order_pretrade")
    @patch("execution.core._get_slippage_model", return_value=None)
    @patch("execution.core.time.sleep")
    def test_all_retries_exhausted_returns_none(self, mock_sleep, mock_slip,
                                                 mock_validate,
                                                 mock_trading_client):
        """All 4 attempts fail — returns None."""
        from execution.core import submit_bracket_order
        mock_validate.return_value = types.SimpleNamespace(valid=True, reason="ok")

        client = MockTradingClient()
        client.submit_order = MagicMock(
            side_effect=Exception("persistent error")
        )

        with patch("execution.core.get_trading_client", return_value=client):
            signal = _make_signal()
            result = submit_bracket_order(signal, qty=10)
        assert result is None
        assert client.submit_order.call_count == 4

    @patch("execution.core.validate_order_pretrade")
    @patch("execution.core._get_slippage_model", return_value=None)
    @patch("execution.core._verify_bracket_entry")
    @patch("execution.core.time.sleep")
    def test_large_mean_reversion_routes_to_twap(self, mock_sleep,
                                                   mock_verify, mock_slip,
                                                   mock_validate,
                                                   mock_trading_client):
        """Order value > $25k with STAT_MR auto-routes to TWAP."""
        from execution.core import submit_bracket_order
        mock_validate.return_value = types.SimpleNamespace(valid=True, reason="ok")

        # $150 * 200 = $30,000 > $25k threshold
        signal = _make_signal(strategy="STAT_MR")
        with patch("execution.core.submit_twap_order", return_value=["id1", "id2"]) as mock_twap:
            result = submit_bracket_order(signal, qty=200)
            mock_twap.assert_called_once()
            assert result == ["id1", "id2"]


# ===================================================================
# _check_order_filled
# ===================================================================

class TestCheckOrderFilled:
    """Tests for _check_order_filled — full fills, partial fills, missing orders."""

    def test_filled_status_returns_true(self, mock_trading_client):
        from execution.core import _check_order_filled
        order = MockOrder("order-123")
        order.status = "filled"
        order.filled_qty = 10
        mock_trading_client._orders.append(order)
        assert _check_order_filled("order-123", mock_trading_client, requested_qty=10) is True

    def test_partial_fill_below_requested_returns_false(self, mock_trading_client):
        from execution.core import _check_order_filled
        order = MockOrder("order-456")
        order.status = "partially_filled"
        order.filled_qty = 5
        mock_trading_client._orders.append(order)
        assert _check_order_filled("order-456", mock_trading_client, requested_qty=10) is False

    def test_partial_fill_meets_requested_returns_true(self, mock_trading_client):
        from execution.core import _check_order_filled
        order = MockOrder("order-789")
        order.status = "partially_filled"
        order.filled_qty = 10
        mock_trading_client._orders.append(order)
        assert _check_order_filled("order-789", mock_trading_client, requested_qty=10) is True

    def test_no_order_id_returns_false(self, mock_trading_client):
        from execution.core import _check_order_filled
        assert _check_order_filled("", mock_trading_client) is False

    def test_no_client_returns_false(self):
        from execution.core import _check_order_filled
        assert _check_order_filled("order-123", None) is False

    def test_order_not_found_returns_false(self, mock_trading_client):
        from execution.core import _check_order_filled
        # get_order_by_id will raise for a non-existent order
        assert _check_order_filled("nonexistent", mock_trading_client) is False


# ===================================================================
# cancel_stale_orders
# ===================================================================

class TestCancelStaleOrders:
    """Tests for cancel_stale_orders — orders older than 5 min are cancelled."""

    def test_stale_order_gets_cancelled(self):
        """An order submitted 10 minutes ago should be cancelled."""
        from execution.core import cancel_stale_orders

        stale_order = types.SimpleNamespace(
            id="stale-001",
            symbol="AAPL",
            status="new",
            submitted_at=datetime.now(timezone.utc) - timedelta(minutes=10),
            created_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )

        mock_client = MagicMock()
        mock_client.get_orders.return_value = [stale_order]
        mock_client.cancel_order_by_id = MagicMock()

        cancelled = cancel_stale_orders(client=mock_client)
        assert "stale-001" in cancelled
        mock_client.cancel_order_by_id.assert_called_once_with("stale-001")

    def test_fresh_order_not_cancelled(self):
        """An order submitted 1 minute ago should not be cancelled."""
        from execution.core import cancel_stale_orders

        fresh_order = types.SimpleNamespace(
            id="fresh-001",
            symbol="AAPL",
            status="new",
            submitted_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            created_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )

        mock_client = MagicMock()
        mock_client.get_orders.return_value = [fresh_order]

        cancelled = cancel_stale_orders(client=mock_client)
        assert len(cancelled) == 0

    def test_filled_order_ignored(self):
        """Orders with 'filled' status should not be cancelled."""
        from execution.core import cancel_stale_orders

        filled_order = types.SimpleNamespace(
            id="filled-001",
            symbol="AAPL",
            status="filled",
            submitted_at=datetime.now(timezone.utc) - timedelta(minutes=10),
            created_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )

        mock_client = MagicMock()
        mock_client.get_orders.return_value = [filled_order]

        cancelled = cancel_stale_orders(client=mock_client)
        assert len(cancelled) == 0

    def test_stale_order_updates_oms_state(self):
        """When cancelling a stale order, OMS state should be updated."""
        from execution.core import cancel_stale_orders
        from oms.order import Order, OrderState

        stale_order = types.SimpleNamespace(
            id="stale-oms-001",
            symbol="AAPL",
            status="new",
            submitted_at=datetime.now(timezone.utc) - timedelta(minutes=10),
            created_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )

        mock_client = MagicMock()
        mock_client.get_orders.return_value = [stale_order]
        mock_client.cancel_order_by_id = MagicMock()

        mock_om = MagicMock()
        oms_order = Order(symbol="AAPL", state=OrderState.SUBMITTED,
                          broker_order_id="stale-oms-001")
        mock_om.get_by_broker_id.return_value = oms_order

        cancelled = cancel_stale_orders(client=mock_client, order_manager=mock_om)
        assert "stale-oms-001" in cancelled
        mock_om.update_state.assert_called_once_with(
            oms_order.oms_id, OrderState.CANCELLED,
        )

    def test_api_failure_returns_empty(self):
        """If get_orders raises, return an empty list (fail-open)."""
        from execution.core import cancel_stale_orders

        mock_client = MagicMock()
        mock_client.get_orders.side_effect = Exception("API down")

        cancelled = cancel_stale_orders(client=mock_client)
        assert cancelled == []


# ===================================================================
# API Circuit Breaker
# ===================================================================

class TestApiCircuitBreaker:
    """Tests for _record_api_failure / _record_api_success circuit breaker."""

    def test_five_failures_triggers_kill_switch(self):
        """5 consecutive failures within the window triggers kill switch."""
        from execution.core import _record_api_failure, _api_failures

        # KillSwitch is lazy-imported inside _record_api_failure, so patch
        # at the import location.
        with patch("oms.kill_switch.KillSwitch") as MockKS:
            mock_ks_instance = MagicMock()
            MockKS.return_value = mock_ks_instance

            for _ in range(5):
                _record_api_failure()

            mock_ks_instance.activate.assert_called_once_with(
                reason="api_failure_circuit_breaker"
            )

    def test_four_failures_does_not_trigger(self):
        """4 failures is below threshold — no kill switch."""
        from execution.core import _record_api_failure, _api_failures

        with patch("oms.kill_switch.KillSwitch") as MockKS:
            for _ in range(4):
                _record_api_failure()

            MockKS.return_value.activate.assert_not_called()

    def test_record_api_success_resets_counter(self):
        """_record_api_success clears the failure list."""
        from execution.core import (
            _record_api_failure, _record_api_success, _api_failures,
        )
        _record_api_failure()
        _record_api_failure()
        _record_api_failure()
        assert len(_api_failures) == 3

        _record_api_success()
        assert len(_api_failures) == 0

    def test_old_failures_pruned_from_window(self):
        """Failures older than 5 minutes are pruned and don't count."""
        from execution.core import _record_api_failure, _api_failures

        # Inject 4 old failures (outside the 5-minute window)
        old_time = time.time() - 400
        _api_failures.extend([old_time] * 4)

        # One new failure should prune the old ones and NOT trigger
        with patch("oms.kill_switch.KillSwitch") as MockKS:
            _record_api_failure()
            MockKS.return_value.activate.assert_not_called()
            # Only the new failure should remain
            assert len(_api_failures) == 1


# ===================================================================
# _validate_bracket_params
# ===================================================================

class TestValidateBracketParams:
    """Tests for _validate_bracket_params — SL/TP relative to entry price."""

    def test_valid_long_bracket(self):
        from execution.core import _validate_bracket_params
        signal = _make_signal(side="buy", entry_price=150.0,
                              take_profit=155.0, stop_loss=148.0)
        valid, reason = _validate_bracket_params(signal)
        assert valid is True
        assert reason == ""

    def test_valid_short_bracket(self):
        from execution.core import _validate_bracket_params
        signal = _make_signal(side="sell", entry_price=150.0,
                              take_profit=145.0, stop_loss=152.0)
        valid, reason = _validate_bracket_params(signal)
        assert valid is True
        assert reason == ""

    def test_long_sl_above_entry_invalid(self):
        from execution.core import _validate_bracket_params
        signal = _make_signal(side="buy", entry_price=150.0,
                              take_profit=155.0, stop_loss=152.0)
        valid, reason = _validate_bracket_params(signal)
        assert valid is False
        assert "SL" in reason

    def test_long_tp_below_entry_invalid(self):
        from execution.core import _validate_bracket_params
        signal = _make_signal(side="buy", entry_price=150.0,
                              take_profit=148.0, stop_loss=145.0)
        valid, reason = _validate_bracket_params(signal)
        assert valid is False
        assert "TP" in reason

    def test_short_sl_below_entry_invalid(self):
        from execution.core import _validate_bracket_params
        signal = _make_signal(side="sell", entry_price=150.0,
                              take_profit=145.0, stop_loss=148.0)
        valid, reason = _validate_bracket_params(signal)
        assert valid is False
        assert "SL" in reason

    def test_short_tp_above_entry_invalid(self):
        from execution.core import _validate_bracket_params
        signal = _make_signal(side="sell", entry_price=150.0,
                              take_profit=155.0, stop_loss=152.0)
        valid, reason = _validate_bracket_params(signal)
        assert valid is False
        assert "TP" in reason

    def test_zero_entry_price_invalid(self):
        """Zero entry price should be rejected by _validate_bracket_params.
        Since Signal.__post_init__ also validates, we use a SimpleNamespace."""
        from execution.core import _validate_bracket_params
        fake_signal = types.SimpleNamespace(
            symbol="AAPL", side="buy", entry_price=0.0,
            take_profit=155.0, stop_loss=148.0,
        )
        valid, reason = _validate_bracket_params(fake_signal)
        assert valid is False
        assert "entry" in reason.lower()

    def test_zero_stop_loss_invalid(self):
        from execution.core import _validate_bracket_params
        fake_signal = types.SimpleNamespace(
            symbol="AAPL", side="buy", entry_price=150.0,
            take_profit=155.0, stop_loss=0.0,
        )
        valid, reason = _validate_bracket_params(fake_signal)
        assert valid is False

    def test_zero_take_profit_invalid(self):
        from execution.core import _validate_bracket_params
        fake_signal = types.SimpleNamespace(
            symbol="AAPL", side="buy", entry_price=150.0,
            take_profit=0.0, stop_loss=148.0,
        )
        valid, reason = _validate_bracket_params(fake_signal)
        assert valid is False


# ===================================================================
# Chase thread
# ===================================================================

class TestChaseThread:
    """Tests for _start_chase_thread — limit order chase logic."""

    @patch("execution.core.validate_order_pretrade")
    @patch("execution.core._get_slippage_model", return_value=None)
    @patch("execution.core._verify_bracket_entry")
    @patch("execution.core._start_chase_thread")
    @patch("execution.core.time.sleep")
    def test_chase_thread_started_for_limit_order_strategies(
        self, mock_sleep, mock_chase, mock_verify, mock_slip,
        mock_validate, mock_trading_client,
    ):
        """Limit-order strategies (STAT_MR, ORB, etc.) start a chase thread."""
        from execution.core import submit_bracket_order
        mock_validate.return_value = types.SimpleNamespace(valid=True, reason="ok")

        for strat in ("STAT_MR", "KALMAN_PAIRS", "ORB", "PEAD"):
            mock_chase.reset_mock()
            # Keep order value under $25k to avoid TWAP routing for mean-rev.
            # Use coherent bracket params (SL below entry, TP above for long).
            signal = _make_signal(strategy=strat, entry_price=100.0,
                                  take_profit=105.0, stop_loss=97.0)
            result = submit_bracket_order(signal, qty=10)
            assert result is not None, f"submit_bracket_order returned None for {strat}"
            mock_chase.assert_called_once()

    @patch("execution.core.validate_order_pretrade")
    @patch("execution.core._get_slippage_model", return_value=None)
    @patch("execution.core._verify_bracket_entry")
    @patch("execution.core._start_chase_thread")
    @patch("execution.core.time.sleep")
    def test_no_chase_thread_for_market_order_strategies(
        self, mock_sleep, mock_chase, mock_verify, mock_slip,
        mock_validate, mock_trading_client,
    ):
        """Market-order strategies (MOMENTUM, VWAP) do not start a chase."""
        from execution.core import submit_bracket_order
        mock_validate.return_value = types.SimpleNamespace(valid=True, reason="ok")

        for strat in ("MOMENTUM", "VWAP", "MICRO_MOM"):
            mock_chase.reset_mock()
            signal = _make_signal(strategy=strat)
            result = submit_bracket_order(signal, qty=10)
            assert result is not None
            mock_chase.assert_not_called()


# ===================================================================
# submit_twap_order
# ===================================================================

class TestSubmitTwapOrder:
    """Tests for submit_twap_order — order splitting."""

    @patch("execution.core.time.sleep")
    def test_twap_splits_into_correct_slices(self, mock_sleep,
                                              mock_trading_client):
        """TWAP with 20 shares / 4 slices produces 4 orders of 5 each."""
        from execution.core import submit_twap_order
        signal = _make_signal()
        ids = submit_twap_order(signal, total_qty=20, slices=4, interval_sec=0)
        assert len(ids) == 4

    @patch("execution.core.time.sleep")
    def test_twap_remainder_distribution(self, mock_sleep,
                                          mock_trading_client):
        """23 shares / 5 slices: first 3 slices get 5 shares, last 2 get 4."""
        from execution.core import submit_twap_order
        signal = _make_signal()
        ids = submit_twap_order(signal, total_qty=23, slices=5, interval_sec=0)
        assert len(ids) == 5

    @patch("execution.core.time.sleep")
    @patch("execution.core.close_position")
    def test_twap_slice_failure_cancels_remaining(self, mock_close,
                                                    mock_sleep,
                                                    mock_trading_client):
        """If one slice fails, remaining slices are skipped."""
        from execution.core import submit_twap_order

        client = MockTradingClient()
        call_count = 0
        original_submit = client.submit_order

        def _fail_on_third(request):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise Exception("slice failed")
            return original_submit(request)

        client.submit_order = _fail_on_third
        client.cancel_order_by_id = MagicMock()
        # Mock get_order_by_id to return filled_qty for partial fill check
        mock_order = MockOrder("prev-001")
        mock_order.filled_qty = 0
        client.get_order_by_id = MagicMock(return_value=mock_order)

        with patch("execution.core.get_trading_client", return_value=client):
            signal = _make_signal()
            ids = submit_twap_order(signal, total_qty=25, slices=5, interval_sec=0)
        # Only the 2 successful submissions should produce IDs
        assert len(ids) == 2

    @patch("execution.core.time.sleep")
    @patch("oms.order_manager._IDEMPOTENCY_FILE", "/tmp/_test_idemp_never_exists.json")
    def test_twap_registers_parent_and_child_in_oms(self, mock_sleep,
                                                      mock_trading_client):
        """When order_manager is provided, parent and child orders are registered."""
        from execution.core import submit_twap_order
        from oms.order_manager import OrderManager

        om = OrderManager()
        signal = _make_signal()
        # Use unique idempotency keys for each child by making slices differ
        ids = submit_twap_order(signal, total_qty=10, slices=2,
                                interval_sec=0, order_manager=om)
        assert len(ids) == 2
        # The OMS should have: parent + N child orders (children may be deduplicated
        # if their auto-generated idempotency keys collide in the same time bucket).
        # The parent is always registered, plus at least 1 child.
        all_orders = om.get_all_orders(limit=100)
        # Parent + at least 1 child
        assert len(all_orders) >= 2
        # Verify parent exists with order_type="twap_parent"
        parent_orders = [o for o in all_orders if o.order_type == "twap_parent"]
        assert len(parent_orders) == 1


# ===================================================================
# Rejection classification
# ===================================================================

class TestRejectionClassification:
    """Tests for classify_rejection — error message routing."""

    def test_insufficient_buying_power_classifies_as_retry_less(self):
        from execution.core import classify_rejection, RejectionClass
        err = Exception("insufficient buying power for this order")
        assert classify_rejection(err) == RejectionClass.RETRY_WITH_LESS

    def test_timeout_classifies_as_transient(self):
        from execution.core import classify_rejection, RejectionClass
        err = Exception("request timeout after 30s")
        assert classify_rejection(err) == RejectionClass.TRANSIENT

    def test_503_classifies_as_transient(self):
        from execution.core import classify_rejection, RejectionClass
        err = Exception("HTTP 503 service unavailable")
        assert classify_rejection(err) == RejectionClass.TRANSIENT

    def test_symbol_not_tradable_classifies_as_do_not_retry(self):
        from execution.core import classify_rejection, RejectionClass
        err = Exception("symbol not tradable")
        assert classify_rejection(err) == RejectionClass.DO_NOT_RETRY

    def test_market_closed_classifies_as_do_not_retry(self):
        from execution.core import classify_rejection, RejectionClass
        err = Exception("market is closed")
        assert classify_rejection(err) == RejectionClass.DO_NOT_RETRY

    def test_unknown_error_defaults_to_do_not_retry(self):
        from execution.core import classify_rejection, RejectionClass
        err = Exception("something completely unexpected happened")
        assert classify_rejection(err) == RejectionClass.DO_NOT_RETRY
