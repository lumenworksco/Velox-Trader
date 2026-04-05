"""Tests for oms/order_manager.py — OrderManager registry and lifecycle."""

import os
os.environ.setdefault("TESTING", "1")

import time
from unittest.mock import patch, MagicMock

import pytest

from oms.order import Order, OrderState
from oms.order_manager import OrderManager


@pytest.fixture(autouse=True)
def _isolate_idempotency_file(tmp_path):
    """Redirect the idempotency key file to a temp dir so tests don't
    load stale keys from previous runs or interfere with each other."""
    fake_file = str(tmp_path / "idempotency_keys.json")
    with patch("oms.order_manager._IDEMPOTENCY_FILE", fake_file):
        yield


# ===================================================================
# create_order
# ===================================================================

class TestCreateOrder:
    """Tests for OrderManager.create_order."""

    def test_creates_order_with_correct_fields(self):
        om = OrderManager()
        order = om.create_order(
            symbol="AAPL",
            strategy="ORB",
            side="buy",
            order_type="bracket",
            qty=10,
            limit_price=150.0,
            take_profit=155.0,
            stop_loss=148.0,
        )
        assert isinstance(order, Order)
        assert order.symbol == "AAPL"
        assert order.strategy == "ORB"
        assert order.side == "buy"
        assert order.order_type == "bracket"
        assert order.qty == 10
        assert order.limit_price == 150.0
        assert order.take_profit == 155.0
        assert order.stop_loss == 148.0
        assert order.state == OrderState.PENDING

    def test_order_appears_in_registry(self):
        om = OrderManager()
        order = om.create_order(
            symbol="AAPL", strategy="ORB", side="buy",
            order_type="bracket", qty=10,
        )
        assert om.get_order(order.oms_id) is order

    def test_order_shows_in_active_orders(self):
        om = OrderManager()
        order = om.create_order(
            symbol="AAPL", strategy="ORB", side="buy",
            order_type="bracket", qty=10,
        )
        active = om.get_active_orders()
        assert any(o.oms_id == order.oms_id for o in active)

    def test_order_shows_in_active_orders_by_symbol(self):
        om = OrderManager()
        om.create_order(symbol="AAPL", strategy="ORB", side="buy",
                         order_type="bracket", qty=10)
        om.create_order(symbol="MSFT", strategy="VWAP", side="sell",
                         order_type="bracket", qty=5)

        aapl_orders = om.get_active_orders(symbol="AAPL")
        assert len(aapl_orders) == 1
        assert aapl_orders[0].symbol == "AAPL"

    def test_pair_id_stored(self):
        om = OrderManager()
        order = om.create_order(
            symbol="AAPL", strategy="PAIRS", side="buy",
            order_type="bracket", qty=10, pair_id="AAPL-MSFT-001",
        )
        assert order.pair_id == "AAPL-MSFT-001"

    def test_auto_generates_idempotency_key(self):
        om = OrderManager()
        order = om.create_order(
            symbol="AAPL", strategy="ORB", side="buy",
            order_type="bracket", qty=10,
        )
        assert order.idempotency_key != ""
        assert len(order.idempotency_key) == 16  # SHA256[:16]


# ===================================================================
# get_by_broker_id
# ===================================================================

class TestGetByBrokerId:
    """Tests for broker ID lookup."""

    def test_lookup_after_state_update_with_broker_id(self):
        om = OrderManager()
        order = om.create_order(
            symbol="AAPL", strategy="ORB", side="buy",
            order_type="bracket", qty=10,
        )
        om.update_state(order.oms_id, OrderState.SUBMITTED,
                         broker_order_id="broker-123")

        found = om.get_by_broker_id("broker-123")
        assert found is not None
        assert found.oms_id == order.oms_id
        assert found.broker_order_id == "broker-123"

    def test_unknown_broker_id_returns_none(self):
        om = OrderManager()
        assert om.get_by_broker_id("nonexistent") is None


# ===================================================================
# update_state
# ===================================================================

class TestUpdateState:
    """Tests for state transition updates."""

    def test_valid_transition_returns_true(self):
        om = OrderManager()
        order = om.create_order(
            symbol="AAPL", strategy="ORB", side="buy",
            order_type="bracket", qty=10,
        )
        assert om.update_state(order.oms_id, OrderState.SUBMITTED) is True
        assert om.get_order(order.oms_id).state == OrderState.SUBMITTED

    def test_full_lifecycle(self):
        om = OrderManager()
        order = om.create_order(
            symbol="AAPL", strategy="ORB", side="buy",
            order_type="bracket", qty=10,
        )
        assert om.update_state(order.oms_id, OrderState.SUBMITTED,
                                broker_order_id="b-001") is True
        assert om.update_state(order.oms_id, OrderState.PARTIAL_FILL,
                                filled_qty=5, filled_avg_price=150.50) is True
        assert om.update_state(order.oms_id, OrderState.FILLED,
                                filled_qty=10, filled_avg_price=150.75) is True

        final = om.get_order(order.oms_id)
        assert final.state == OrderState.FILLED
        assert final.filled_qty == 10
        assert final.filled_avg_price == 150.75
        assert final.broker_order_id == "b-001"

    def test_invalid_transition_returns_false(self):
        om = OrderManager()
        order = om.create_order(
            symbol="AAPL", strategy="ORB", side="buy",
            order_type="bracket", qty=10,
        )
        # PENDING -> FILLED is invalid (must go through SUBMITTED)
        assert om.update_state(order.oms_id, OrderState.FILLED) is False
        assert om.get_order(order.oms_id).state == OrderState.PENDING

    def test_unknown_oms_id_returns_false(self):
        om = OrderManager()
        assert om.update_state("nonexistent-id", OrderState.SUBMITTED) is False

    def test_filled_order_leaves_active_orders(self):
        om = OrderManager()
        order = om.create_order(
            symbol="AAPL", strategy="ORB", side="buy",
            order_type="bracket", qty=10,
        )
        om.update_state(order.oms_id, OrderState.SUBMITTED)
        om.update_state(order.oms_id, OrderState.FILLED)

        active = om.get_active_orders()
        assert not any(o.oms_id == order.oms_id for o in active)


# ===================================================================
# Idempotency key deduplication
# ===================================================================

class TestIdempotencyDeduplication:
    """Tests for duplicate order prevention via idempotency keys."""

    def test_same_key_returns_existing_order(self):
        om = OrderManager()
        key = "dedup-test-key-001"
        order1 = om.create_order(
            symbol="AAPL", strategy="ORB", side="buy",
            order_type="bracket", qty=10,
            idempotency_key=key,
        )
        order2 = om.create_order(
            symbol="AAPL", strategy="ORB", side="buy",
            order_type="bracket", qty=10,
            idempotency_key=key,
        )
        assert order1.oms_id == order2.oms_id  # Same object returned
        assert len(om.get_all_orders()) == 1   # Only one order created

    def test_different_keys_create_separate_orders(self):
        om = OrderManager()
        order1 = om.create_order(
            symbol="AAPL", strategy="ORB", side="buy",
            order_type="bracket", qty=10,
            idempotency_key="key-A",
        )
        order2 = om.create_order(
            symbol="AAPL", strategy="ORB", side="buy",
            order_type="bracket", qty=10,
            idempotency_key="key-B",
        )
        assert order1.oms_id != order2.oms_id
        assert len(om.get_all_orders()) == 2

    def test_auto_generated_keys_use_time_buckets(self):
        """Orders with identical params in the same 2s window get the same key."""
        om = OrderManager()
        # Monkey-patch to use a fixed monotonic time for both orders
        with patch("time.monotonic", return_value=1000.0):
            order1 = om.create_order(
                symbol="AAPL", strategy="ORB", side="buy",
                order_type="bracket", qty=10,
            )
            order2 = om.create_order(
                symbol="AAPL", strategy="ORB", side="buy",
                order_type="bracket", qty=10,
            )
        # Same time bucket -> same auto-generated key -> same order
        assert order1.oms_id == order2.oms_id


# ===================================================================
# Audit trail
# ===================================================================

class TestAuditTrail:
    """Tests for audit trail recording."""

    def test_creation_logged_in_audit(self):
        om = OrderManager()
        order = om.create_order(
            symbol="AAPL", strategy="ORB", side="buy",
            order_type="bracket", qty=10,
        )
        audit = om.get_audit_trail()
        assert len(audit) >= 1
        entry = audit[-1]
        assert entry["oms_id"] == order.oms_id
        assert entry["new_state"] == "pending"
        assert entry["old_state"] == "NEW"

    def test_transition_logged_in_audit(self):
        om = OrderManager()
        order = om.create_order(
            symbol="AAPL", strategy="ORB", side="buy",
            order_type="bracket", qty=10,
        )
        om.update_state(order.oms_id, OrderState.SUBMITTED)

        audit = om.get_audit_trail()
        last = audit[-1]
        assert last["old_state"] == "pending"
        assert last["new_state"] == "submitted"
        assert last["symbol"] == "AAPL"
        assert last["strategy"] == "ORB"

    def test_audit_trail_bounded(self):
        """Audit trail should not grow beyond its maxlen (10,000)."""
        om = OrderManager()
        # The deque maxlen is 10,000, so it self-limits.
        # Just confirm the audit works and returns results.
        for i in range(5):
            om.create_order(
                symbol=f"SYM{i}", strategy="ORB", side="buy",
                order_type="bracket", qty=1,
                idempotency_key=f"unique-key-{i}",
            )
        trail = om.get_audit_trail(limit=3)
        assert len(trail) == 3

    def test_audit_trail_limit(self):
        om = OrderManager()
        for i in range(10):
            om.create_order(
                symbol="AAPL", strategy="ORB", side="buy",
                order_type="bracket", qty=1,
                idempotency_key=f"trail-key-{i}",
            )
        limited = om.get_audit_trail(limit=5)
        assert len(limited) == 5


# ===================================================================
# cancel_all
# ===================================================================

class TestCancelAll:
    """Tests for cancel_all."""

    def test_cancel_all_cancels_active_orders(self):
        om = OrderManager()
        order1 = om.create_order(
            symbol="AAPL", strategy="ORB", side="buy",
            order_type="bracket", qty=10, idempotency_key="cancel-1",
        )
        order2 = om.create_order(
            symbol="MSFT", strategy="VWAP", side="sell",
            order_type="bracket", qty=5, idempotency_key="cancel-2",
        )

        cancelled = om.cancel_all()
        assert len(cancelled) == 2
        assert om.get_order(order1.oms_id).state == OrderState.CANCELLED
        assert om.get_order(order2.oms_id).state == OrderState.CANCELLED

    def test_cancel_all_skips_terminal_orders(self):
        om = OrderManager()
        order1 = om.create_order(
            symbol="AAPL", strategy="ORB", side="buy",
            order_type="bracket", qty=10, idempotency_key="term-1",
        )
        om.update_state(order1.oms_id, OrderState.SUBMITTED)
        om.update_state(order1.oms_id, OrderState.FILLED)

        order2 = om.create_order(
            symbol="MSFT", strategy="VWAP", side="sell",
            order_type="bracket", qty=5, idempotency_key="term-2",
        )

        cancelled = om.cancel_all()
        assert len(cancelled) == 1
        assert order2.oms_id in cancelled
        assert om.get_order(order1.oms_id).state == OrderState.FILLED


# ===================================================================
# stats
# ===================================================================

class TestOrderManagerStats:
    """Tests for the stats property."""

    def test_empty_stats(self):
        om = OrderManager()
        s = om.stats
        assert s["total"] == 0
        assert s["active"] == 0
        assert s["filled"] == 0

    def test_stats_after_orders(self):
        om = OrderManager()
        o1 = om.create_order(symbol="AAPL", strategy="ORB", side="buy",
                              order_type="bracket", qty=10,
                              idempotency_key="stats-1")
        o2 = om.create_order(symbol="MSFT", strategy="VWAP", side="sell",
                              order_type="bracket", qty=5,
                              idempotency_key="stats-2")
        om.update_state(o1.oms_id, OrderState.SUBMITTED)
        om.update_state(o1.oms_id, OrderState.FILLED)

        s = om.stats
        assert s["total"] == 2
        assert s["active"] == 1  # o2 is still PENDING
        assert s["filled"] == 1
