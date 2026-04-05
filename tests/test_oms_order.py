"""Tests for oms/order.py — Order dataclass and state machine."""

import os
os.environ.setdefault("TESTING", "1")

import pytest

from oms.order import Order, OrderState, _TRANSITIONS


# ===================================================================
# Order creation
# ===================================================================

class TestOrderCreation:
    """Tests for creating Order objects with all fields."""

    def test_default_order_has_pending_state(self):
        order = Order()
        assert order.state == OrderState.PENDING

    def test_order_with_all_fields(self):
        order = Order(
            symbol="AAPL",
            strategy="ORB",
            side="buy",
            order_type="bracket",
            qty=10,
            limit_price=150.0,
            take_profit=155.0,
            stop_loss=148.0,
            pair_id="pair-001",
        )
        assert order.symbol == "AAPL"
        assert order.strategy == "ORB"
        assert order.side == "buy"
        assert order.order_type == "bracket"
        assert order.qty == 10
        assert order.limit_price == 150.0
        assert order.take_profit == 155.0
        assert order.stop_loss == 148.0
        assert order.pair_id == "pair-001"

    def test_oms_id_is_auto_generated(self):
        order1 = Order()
        order2 = Order()
        assert order1.oms_id != order2.oms_id
        assert len(order1.oms_id) == 12

    def test_created_at_is_set_automatically(self):
        order = Order()
        assert order.created_at is not None

    def test_default_fill_fields(self):
        order = Order()
        assert order.filled_qty == 0
        assert order.filled_avg_price == 0.0
        assert order.submitted_at is None
        assert order.filled_at is None
        assert order.cancelled_at is None


# ===================================================================
# State transitions
# ===================================================================

class TestOrderStateTransitions:
    """Tests for valid and invalid state transitions."""

    def test_pending_to_submitted(self):
        order = Order()
        assert order.transition(OrderState.SUBMITTED) is True
        assert order.state == OrderState.SUBMITTED
        assert order.submitted_at is not None

    def test_pending_to_failed(self):
        order = Order()
        assert order.transition(OrderState.FAILED) is True
        assert order.state == OrderState.FAILED

    def test_pending_to_cancelled(self):
        order = Order()
        assert order.transition(OrderState.CANCELLED) is True
        assert order.state == OrderState.CANCELLED
        assert order.cancelled_at is not None

    def test_submitted_to_filled(self):
        order = Order()
        order.transition(OrderState.SUBMITTED)
        assert order.transition(OrderState.FILLED) is True
        assert order.state == OrderState.FILLED
        assert order.filled_at is not None

    def test_submitted_to_partial_fill(self):
        order = Order()
        order.transition(OrderState.SUBMITTED)
        assert order.transition(OrderState.PARTIAL_FILL) is True
        assert order.state == OrderState.PARTIAL_FILL

    def test_submitted_to_cancelled(self):
        order = Order()
        order.transition(OrderState.SUBMITTED)
        assert order.transition(OrderState.CANCELLED) is True
        assert order.state == OrderState.CANCELLED

    def test_submitted_to_rejected(self):
        order = Order()
        order.transition(OrderState.SUBMITTED)
        assert order.transition(OrderState.REJECTED) is True
        assert order.state == OrderState.REJECTED

    def test_submitted_to_expired(self):
        order = Order()
        order.transition(OrderState.SUBMITTED)
        assert order.transition(OrderState.EXPIRED) is True
        assert order.state == OrderState.EXPIRED

    def test_partial_fill_to_filled(self):
        order = Order()
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.PARTIAL_FILL)
        assert order.transition(OrderState.FILLED) is True
        assert order.state == OrderState.FILLED

    def test_partial_fill_to_cancelled(self):
        order = Order()
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.PARTIAL_FILL)
        assert order.transition(OrderState.CANCELLED) is True

    # --- Invalid transitions ---

    def test_pending_to_filled_invalid(self):
        order = Order()
        assert order.transition(OrderState.FILLED) is False
        assert order.state == OrderState.PENDING  # Unchanged

    def test_filled_is_terminal(self):
        order = Order()
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.FILLED)
        assert order.transition(OrderState.CANCELLED) is False
        assert order.state == OrderState.FILLED

    def test_cancelled_is_terminal(self):
        order = Order()
        order.transition(OrderState.CANCELLED)
        assert order.transition(OrderState.SUBMITTED) is False
        assert order.state == OrderState.CANCELLED

    def test_rejected_is_terminal(self):
        order = Order()
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.REJECTED)
        assert order.transition(OrderState.FILLED) is False

    def test_expired_is_terminal(self):
        order = Order()
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.EXPIRED)
        assert order.transition(OrderState.SUBMITTED) is False

    def test_failed_is_terminal(self):
        order = Order()
        order.transition(OrderState.FAILED)
        assert order.transition(OrderState.SUBMITTED) is False

    def test_submitted_to_pending_invalid(self):
        """Cannot go backwards from SUBMITTED to PENDING."""
        order = Order()
        order.transition(OrderState.SUBMITTED)
        assert order.transition(OrderState.PENDING) is False

    def test_pending_to_partial_fill_invalid(self):
        """Cannot go directly from PENDING to PARTIAL_FILL."""
        order = Order()
        assert order.transition(OrderState.PARTIAL_FILL) is False


# ===================================================================
# Properties
# ===================================================================

class TestOrderProperties:
    """Tests for is_terminal and is_active properties."""

    def test_pending_is_active(self):
        order = Order()
        assert order.is_active is True
        assert order.is_terminal is False

    def test_submitted_is_active(self):
        order = Order()
        order.transition(OrderState.SUBMITTED)
        assert order.is_active is True

    def test_partial_fill_is_active(self):
        order = Order()
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.PARTIAL_FILL)
        assert order.is_active is True

    def test_filled_is_terminal(self):
        order = Order()
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.FILLED)
        assert order.is_terminal is True
        assert order.is_active is False

    def test_cancelled_is_terminal(self):
        order = Order()
        order.transition(OrderState.CANCELLED)
        assert order.is_terminal is True
        assert order.is_active is False

    def test_rejected_is_terminal(self):
        order = Order()
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.REJECTED)
        assert order.is_terminal is True

    def test_expired_is_terminal(self):
        order = Order()
        order.transition(OrderState.SUBMITTED)
        order.transition(OrderState.EXPIRED)
        assert order.is_terminal is True

    def test_failed_is_terminal(self):
        order = Order()
        order.transition(OrderState.FAILED)
        assert order.is_terminal is True


# ===================================================================
# Idempotency key
# ===================================================================

class TestIdempotencyKey:
    """Tests for idempotency_key field on Order."""

    def test_idempotency_key_stored(self):
        order = Order(idempotency_key="test-key-001")
        assert order.idempotency_key == "test-key-001"

    def test_idempotency_key_defaults_to_empty(self):
        order = Order()
        assert order.idempotency_key == ""


# ===================================================================
# Transition table completeness
# ===================================================================

class TestTransitionTableCompleteness:
    """Verify the _TRANSITIONS table covers all OrderState values."""

    def test_all_states_have_transition_entries(self):
        for state in OrderState:
            assert state in _TRANSITIONS, f"Missing transition entry for {state}"

    def test_terminal_states_have_empty_transitions(self):
        terminal = (OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED,
                     OrderState.EXPIRED, OrderState.FAILED)
        for state in terminal:
            assert _TRANSITIONS[state] == set(), f"{state} should be terminal"
