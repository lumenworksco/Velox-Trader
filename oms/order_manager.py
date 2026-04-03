"""V10 OMS — Central order registry and lifecycle manager."""

import hashlib
import json
import logging
import os
import threading
from collections import deque
from datetime import datetime

from oms.order import Order, OrderState

logger = logging.getLogger(__name__)

# V12 11.2: Persistent idempotency key file — survives restarts
_IDEMPOTENCY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "idempotency_keys.json"
)
_IDEMPOTENCY_PERSIST_LIMIT = 100  # Keep last N keys on disk


class OrderManager:
    """Central registry for all orders. Thread-safe.

    Provides:
    - Order creation with idempotency keys (prevents duplicate orders)
    - Order state tracking through full lifecycle
    - Active/historical order queries
    - Audit trail of all state transitions
    """

    def __init__(self):
        # MED-026: Wrap initialization with explicit error logging and propagation
        try:
            self._orders: dict[str, Order] = {}       # oms_id -> Order
            self._by_broker_id: dict[str, str] = {}   # broker_order_id -> oms_id
            self._by_symbol: dict[str, list[str]] = {} # symbol -> [oms_id, ...]
            self._idempotency: dict[str, str] = {}    # idempotency_key -> oms_id
            self._lock = threading.Lock()
            self._audit: deque[dict] = deque(maxlen=10_000)  # HIGH-006: bounded audit log
            # V12 11.2: Load persisted idempotency keys from previous session
            self._load_idempotency_keys()
            logger.info("OrderManager initialized successfully")
        except Exception as e:
            logger.critical("OrderManager initialization failed: %s", e, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # V12 11.2: Idempotency key persistence across restarts
    # ------------------------------------------------------------------

    def _load_idempotency_keys(self) -> None:
        """Load persisted idempotency keys from disk on startup."""
        try:
            if os.path.exists(_IDEMPOTENCY_FILE):
                with open(_IDEMPOTENCY_FILE, "r") as f:
                    data = json.load(f)
                keys = data if isinstance(data, dict) else {}
                self._idempotency.update(keys)
                logger.info(f"V12 11.2: Loaded {len(keys)} idempotency keys from disk")
        except Exception as e:
            logger.warning(f"V12 11.2: Failed to load idempotency keys: {e}")

    def _persist_idempotency_keys(self) -> None:
        """Persist the most recent idempotency keys to disk (last N entries)."""
        try:
            # Keep only the last N keys to bound file size
            items = list(self._idempotency.items())
            trimmed = dict(items[-_IDEMPOTENCY_PERSIST_LIMIT:])
            os.makedirs(os.path.dirname(_IDEMPOTENCY_FILE), exist_ok=True)
            with open(_IDEMPOTENCY_FILE, "w") as f:
                json.dump(trimmed, f)
        except Exception as e:
            logger.warning(f"V12 11.2: Failed to persist idempotency keys: {e}")

    @staticmethod
    def _generate_idempotency_key(symbol: str, side: str, qty: int,
                                   strategy: str) -> str:
        """Generate unique idempotency key using content hash + monotonic nonce.

        Uses a hash of (symbol, side, qty, strategy) combined with a 2-second
        timestamp bucket so that identical orders within the same window are
        treated as duplicates. Narrower window (was 5s) reduces race condition
        risk for rapid sequential orders.

        HIGH-024: Uses time.monotonic() to avoid issues with system clock changes.
        """
        import time
        # Content-based: same order params within 2-second window get same key
        content = f"{symbol}:{side}:{qty}:{strategy}"
        time_bucket = int(time.monotonic() * 1000) // 2000  # 2-second buckets (was 5s)
        raw = f"{content}:{time_bucket}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def create_order(self, symbol: str, strategy: str, side: str,
                     order_type: str, qty: int, limit_price: float = 0.0,
                     take_profit: float = 0.0, stop_loss: float = 0.0,
                     pair_id: str = "", idempotency_key: str = "") -> Order:
        """Create and register a new order."""
        with self._lock:
            # BUG-012: Auto-generate idempotency key if empty/missing
            if not idempotency_key:
                idempotency_key = self._generate_idempotency_key(
                    symbol, side, qty, strategy
                )
                logger.debug(f"OMS: Auto-generated idempotency_key={idempotency_key}")

            # Check idempotency
            if idempotency_key in self._idempotency:
                existing_id = self._idempotency[idempotency_key]
                logger.info(f"Duplicate order blocked (idempotency_key={idempotency_key})")
                return self._orders[existing_id]

            order = Order(
                symbol=symbol,
                strategy=strategy,
                side=side,
                order_type=order_type,
                qty=qty,
                limit_price=limit_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                pair_id=pair_id,
                idempotency_key=idempotency_key,
            )

            self._orders[order.oms_id] = order
            self._by_symbol.setdefault(symbol, []).append(order.oms_id)
            self._idempotency[idempotency_key] = order.oms_id

            # V12 11.2: Persist idempotency keys to disk after each new order
            self._persist_idempotency_keys()

            self._log_transition(order, None, OrderState.PENDING)
            logger.info(f"OMS: Created order {order.oms_id} {side} {qty} {symbol} ({strategy})")
            return order

    def update_state(self, oms_id: str, new_state: OrderState,
                     broker_order_id: str = "", filled_qty: int = 0,
                     filled_avg_price: float = 0.0) -> bool:
        """Update order state. Returns True if transition was valid."""
        with self._lock:
            order = self._orders.get(oms_id)
            if not order:
                logger.warning(f"OMS: Unknown order {oms_id}")
                return False

            old_state = order.state
            if not order.transition(new_state):
                return False

            if broker_order_id:
                order.broker_order_id = broker_order_id
                self._by_broker_id[broker_order_id] = oms_id
            if filled_qty:
                order.filled_qty = filled_qty
            if filled_avg_price:
                order.filled_avg_price = filled_avg_price

            self._log_transition(order, old_state, new_state)
            return True

    def get_order(self, oms_id: str) -> Order | None:
        """Get order by OMS ID."""
        return self._orders.get(oms_id)

    def get_by_broker_id(self, broker_order_id: str) -> Order | None:
        """Get order by broker order ID."""
        oms_id = self._by_broker_id.get(broker_order_id)
        return self._orders.get(oms_id) if oms_id else None

    def get_active_orders(self, symbol: str = None) -> list[Order]:
        """Get all active (non-terminal) orders, optionally filtered by symbol."""
        with self._lock:
            if symbol:
                oms_ids = self._by_symbol.get(symbol, [])
                return [self._orders[oid] for oid in oms_ids if self._orders[oid].is_active]
            return [o for o in self._orders.values() if o.is_active]

    def get_all_orders(self, limit: int = 100) -> list[Order]:
        """Get most recent orders."""
        orders = sorted(self._orders.values(), key=lambda o: o.created_at, reverse=True)
        return orders[:limit]

    def cancel_all(self) -> list[str]:
        """Cancel all active orders. Returns list of cancelled OMS IDs."""
        cancelled = []
        with self._lock:
            for order in list(self._orders.values()):
                if order.is_active:
                    old_state = order.state
                    if order.transition(OrderState.CANCELLED):
                        self._log_transition(order, old_state, OrderState.CANCELLED)
                        cancelled.append(order.oms_id)
        return cancelled

    def _log_transition(self, order: Order, old_state: OrderState | None, new_state: OrderState):
        """Log a state transition for audit trail (bounded by deque maxlen)."""
        self._audit.append({
            "timestamp": datetime.now().isoformat(),
            "oms_id": order.oms_id,
            "symbol": order.symbol,
            "strategy": order.strategy,
            "old_state": old_state.value if old_state else "NEW",
            "new_state": new_state.value,
        })

    def get_audit_trail(self, limit: int = 50) -> list[dict]:
        """Get recent audit trail entries."""
        return list(self._audit)[-limit:]

    @property
    def stats(self) -> dict:
        """Get OMS statistics."""
        with self._lock:
            active = sum(1 for o in self._orders.values() if o.is_active)
            filled = sum(1 for o in self._orders.values() if o.state == OrderState.FILLED)
            cancelled = sum(1 for o in self._orders.values() if o.state == OrderState.CANCELLED)
            rejected = sum(1 for o in self._orders.values() if o.state == OrderState.REJECTED)
            return {
                "total": len(self._orders),
                "active": active,
                "filled": filled,
                "cancelled": cancelled,
                "rejected": rejected,
            }
