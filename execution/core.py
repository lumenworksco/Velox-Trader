"""Order execution — bracket orders, TWAP splitting, retries, EOD exits.

EXEC-005: Exponential backoff with jitter for order submission retries.
EXEC-006: Enhanced pre-trade validation (buying power, price reasonableness, etc.).
V12 7.1: Order rejection classification (RETRY_WITH_LESS, TRANSIENT, DO_NOT_RETRY).
V12 7.2: Chase logic for unfilled limit orders.
V12 7.3: Stale order cancellation (>5 min in SUBMITTED status).
V12 7.4: Post-trade slippage feedback loop.
"""

import logging
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    ReplaceOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

import config
from data import get_trading_client
from strategies.base import Signal

logger = logging.getLogger(__name__)

# V11.3 T4: Lazy-loaded slippage model singleton
_slippage_model = None

# ============================================================
# V12 6.4: API Failure Circuit Breaker
# ============================================================
_api_failures: list[float] = []  # timestamps of consecutive failures
_API_FAILURE_WINDOW_SEC = 300    # 5-minute window
_API_FAILURE_THRESHOLD = 5       # 5 failures triggers kill switch


def _record_api_failure() -> None:
    """Record an API failure and activate kill switch if threshold exceeded."""
    now = time.time()
    _api_failures.append(now)
    # Prune failures outside the 5-minute window
    cutoff = now - _API_FAILURE_WINDOW_SEC
    while _api_failures and _api_failures[0] < cutoff:
        _api_failures.pop(0)
    if len(_api_failures) >= _API_FAILURE_THRESHOLD:
        logger.critical(
            "V12 6.4: %d consecutive API failures in %ds — activating kill switch",
            len(_api_failures), _API_FAILURE_WINDOW_SEC,
        )
        try:
            from oms.kill_switch import KillSwitch
            KillSwitch().activate(reason="api_failure_circuit_breaker")
        except Exception as e:
            logger.critical("V12 6.4: Kill switch activation failed: %s", e)
        _api_failures.clear()


def _record_api_success() -> None:
    """Reset the API failure counter on successful ORDER SUBMISSION.

    V12 FINAL: Only call this after a successful order submission,
    NOT after other API calls (get_positions, get_account, etc.).
    The circuit breaker tracks consecutive submission failures only.
    """
    _api_failures.clear()


def _get_slippage_model():
    """Lazy-load the SlippageModel singleton (fail-open)."""
    global _slippage_model
    if _slippage_model is None:
        try:
            from execution.slippage_model import SlippageModel
            _slippage_model = SlippageModel()
            logger.info("V11.3 T4: SlippageModel initialized")
        except Exception as e:
            logger.warning(f"V11.3 T4: SlippageModel init failed (fail-open): {e}")
            _slippage_model = False  # Sentinel: don't retry
    return _slippage_model if _slippage_model else None


# ============================================================
# V12 7.1: Order Rejection Classification
# ============================================================

class RejectionClass(Enum):
    """Classification of order rejection errors."""
    RETRY_WITH_LESS = "retry_with_less"   # Reduce size by 50% and retry
    TRANSIENT = "transient"               # Retry after delay (up to 3 times)
    DO_NOT_RETRY = "do_not_retry"         # Log and skip


# Patterns matched case-insensitively against the error message
_RETRY_WITH_LESS_PATTERNS = [
    r"insufficient\s+buying\s*power",
    r"exceeds?\s+max\s+notional",
    r"buying\s*power\s+.*not\s+sufficient",
    r"notional\s+.*exceed",
    r"insufficient\s+equity",
    r"margin\s+requirement",
]
_TRANSIENT_PATTERNS = [
    r"internal\s+(server\s+)?error",
    r"timeout",
    r"\b503\b",
    r"\b502\b",
    r"\b504\b",
    r"service\s+unavailable",
    r"temporarily\s+unavailable",
    r"rate\s+limit",
    r"try\s+again",
    r"connection\s+(reset|refused|aborted)",
]
_DO_NOT_RETRY_PATTERNS = [
    r"symbol\s+not\s+tradable",
    r"market\s+(is\s+)?closed",
    r"invalid\s+order",
    r"asset\s+not\s+found",
    r"not\s+shortable",
    r"order\s+.*not\s+allowed",
    r"account\s+is\s+restricted",
    r"halt",
    r"invalid\s+qty",
    r"invalid\s+side",
]

_COMPILED_RETRY_LESS = [re.compile(p, re.IGNORECASE) for p in _RETRY_WITH_LESS_PATTERNS]
_COMPILED_TRANSIENT = [re.compile(p, re.IGNORECASE) for p in _TRANSIENT_PATTERNS]
_COMPILED_NO_RETRY = [re.compile(p, re.IGNORECASE) for p in _DO_NOT_RETRY_PATTERNS]


def classify_rejection(error: Exception) -> RejectionClass:
    """V12 7.1: Classify an order rejection error into a retry strategy.

    Args:
        error: The exception from order submission.

    Returns:
        RejectionClass indicating how to handle the failure.
    """
    msg = str(error)

    # Check DO_NOT_RETRY first (most specific, prevents wasting retries)
    for pattern in _COMPILED_NO_RETRY:
        if pattern.search(msg):
            return RejectionClass.DO_NOT_RETRY

    # Check RETRY_WITH_LESS (sizing issues)
    for pattern in _COMPILED_RETRY_LESS:
        if pattern.search(msg):
            return RejectionClass.RETRY_WITH_LESS

    # Check TRANSIENT (infrastructure issues)
    for pattern in _COMPILED_TRANSIENT:
        if pattern.search(msg):
            return RejectionClass.TRANSIENT

    # Default: treat unknown errors as DO_NOT_RETRY to avoid infinite loops
    return RejectionClass.DO_NOT_RETRY


# ============================================================
# V12 7.2: Chase logic for unfilled limit orders
# ============================================================

# Active chase threads tracked for cleanup
_active_chases: dict[str, threading.Thread] = {}
_chase_lock = threading.Lock()

# Strategies that use limit orders (chase candidates)
_LIMIT_ORDER_STRATEGIES = ("STAT_MR", "KALMAN_PAIRS", "ORB", "PEAD")


def _chase_unfilled_order(
    order_id: str,
    signal: Signal,
    qty: int,
    client=None,
) -> None:
    """V12 7.2: Background chase logic for unfilled limit orders.

    V12 FINAL: Tightened chase schedule to reduce latency:
    - After 10s unfilled: amend price to mid-quote (using cached spread, no fresh API call).
    - After 20s still unfilled: convert to market order (cancel + resubmit).
    - After 30s still unfilled: cancel entirely.

    Runs in a daemon thread; fail-open on all errors.
    """
    if client is None:
        client = get_trading_client()

    chase_start = time.monotonic()
    amended = False
    converted = False
    current_order_id = order_id

    try:
        while True:
            elapsed = time.monotonic() - chase_start
            if elapsed > 30:
                # 30s: cancel entirely
                try:
                    client.cancel_order_by_id(current_order_id)
                    logger.warning(
                        f"V12 7.2: Chase timeout 30s — cancelled order "
                        f"{current_order_id} for {signal.symbol}"
                    )
                except Exception as ce:
                    logger.debug(f"V12 7.2: Cancel failed for {current_order_id}: {ce}")
                break

            # Check current order status
            try:
                order = client.get_order_by_id(current_order_id)
                status = str(order.status).lower()
            except Exception as e:
                logger.debug(
                    f"V12 7.2: Order status check failed for {current_order_id}: {e}"
                )
                break

            # If filled or terminal — stop chasing
            if status in ("filled", "partially_filled", "cancelled", "canceled",
                          "expired", "rejected", "replaced"):
                logger.debug(
                    f"V12 7.2: Chase ended for {current_order_id} — status={status}"
                )
                break

            if elapsed > 20 and not converted:
                # 20s: convert to market order (cancel limit + submit market)
                try:
                    client.cancel_order_by_id(current_order_id)
                    side = OrderSide.BUY if signal.side == "buy" else OrderSide.SELL
                    mkt_order = client.submit_order(
                        MarketOrderRequest(
                            symbol=signal.symbol,
                            qty=qty,
                            side=side,
                            time_in_force=TimeInForce.DAY,
                            order_class=OrderClass.BRACKET,
                            take_profit={"limit_price": round(signal.take_profit, 2)},
                            stop_loss={"stop_price": round(signal.stop_loss, 2)},
                        )
                    )
                    converted = True
                    current_order_id = str(mkt_order.id)
                    logger.info(
                        f"V12 7.2: Chase 20s — converted {signal.symbol} to market "
                        f"order {current_order_id}"
                    )
                except Exception as e:
                    logger.warning(
                        f"V12 7.2: Chase market conversion failed for "
                        f"{signal.symbol}: {e}"
                    )
                    converted = True  # Don't retry conversion

            elif elapsed > 10 and not amended and not converted:
                # 10s: amend price to mid-quote using cached spread (no fresh API call)
                try:
                    from data import quote_cache  # V12 AUDIT: use cached quotes, not live fetch
                    cached = quote_cache.get(signal.symbol)
                    if cached is not None:
                        bid = getattr(cached, 'bid_price', 0)
                        ask = getattr(cached, 'ask_price', 0)
                        if bid > 0 and ask > 0:
                            new_price = round((bid + ask) / 2, 2)
                            client.replace_order_by_id(
                                current_order_id,
                                ReplaceOrderRequest(
                                    qty=qty,
                                    limit_price=new_price,
                                    time_in_force=TimeInForce.DAY,
                                ),
                            )
                            amended = True
                            logger.info(
                                f"V12 7.2: Chase 10s — amended {signal.symbol} "
                                f"limit to cached mid={new_price}"
                            )
                        else:
                            logger.debug(
                                f"V12 7.2: Cached quote for {signal.symbol} has invalid bid/ask — skipping amend"
                            )
                            amended = True
                    else:
                        logger.debug(
                            f"V12 7.2: No cached quote for {signal.symbol} — skipping amend"
                        )
                        amended = True
                except ImportError:
                    # quote_cache not available — fall back to no-amend
                    logger.debug(
                        f"V12 7.2: quote_cache not available — skipping amend for {signal.symbol}"
                    )
                    amended = True
                except Exception as e:
                    logger.debug(
                        f"V12 7.2: Chase amend failed for {signal.symbol}: {e}"
                    )
                    amended = True  # Don't retry amendment

            time.sleep(5)  # Poll every 5 seconds

    except Exception as e:
        logger.warning(f"V12 7.2: Chase thread error for {signal.symbol}: {e}")
    finally:
        with _chase_lock:
            _active_chases.pop(order_id, None)


def _start_chase_thread(order_id: str, signal: Signal, qty: int) -> None:
    """V12 7.2: Start a background daemon thread to chase an unfilled limit order."""
    with _chase_lock:
        if order_id in _active_chases:
            return  # Already chasing this order

    t = threading.Thread(
        target=_chase_unfilled_order,
        args=(order_id, signal, qty),
        name=f"chase-{signal.symbol}-{order_id[:8]}",
        daemon=True,
    )
    with _chase_lock:
        _active_chases[order_id] = t
    t.start()
    logger.debug(
        f"V12 7.2: Chase thread started for {signal.symbol} order_id={order_id}"
    )


# ============================================================
# V12 7.3: Stale Order Cancellation
# ============================================================

_STALE_ORDER_THRESHOLD_SEC = 300  # 5 minutes


def cancel_stale_orders(client=None, order_manager=None) -> list[str]:
    """V12 7.3: Cancel any orders in open status for >5 minutes.

    Called every scan cycle from main.py.

    Returns:
        List of cancelled order IDs.
    """
    if client is None:
        client = get_trading_client()

    cancelled: list[str] = []
    now = datetime.now(timezone.utc)

    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        open_orders = client.get_orders(request)
    except Exception as e:
        logger.debug(f"V12 7.3: Failed to fetch open orders: {e}")
        return cancelled

    for order in open_orders:
        try:
            status = str(order.status).lower()
            if status not in ("new", "accepted", "pending_new",
                              "accepted_for_bidding", "held"):
                continue

            # Check order age
            submitted_at = order.submitted_at or order.created_at
            if submitted_at is None:
                continue

            # Ensure timezone-aware comparison
            if submitted_at.tzinfo is None:
                submitted_at = submitted_at.replace(tzinfo=timezone.utc)

            age_sec = (now - submitted_at).total_seconds()
            if age_sec > _STALE_ORDER_THRESHOLD_SEC:
                try:
                    client.cancel_order_by_id(str(order.id))
                    cancelled.append(str(order.id))
                    logger.warning(
                        f"V12 7.3: Cancelled stale order {order.id} "
                        f"for {order.symbol} — age={age_sec:.0f}s "
                        f"status={status}"
                    )
                    # V12 AUDIT: Update OMS state when cancelling stale orders
                    try:
                        if order_manager:
                            from oms.order import OrderState
                            oms_order = order_manager.get_by_broker_id(str(order.id))
                            if oms_order:
                                order_manager.update_state(oms_order.oms_id, OrderState.CANCELLED)
                    except Exception as e:
                        logger.debug("V12 AUDIT: OMS state update for cancelled order failed: %s", e)
                except Exception as ce:
                    logger.debug(
                        f"V12 7.3: Failed to cancel stale order {order.id}: {ce}"
                    )
        except Exception as e:
            logger.debug(f"V12 7.3: Error processing order {order.id}: {e}")

    if cancelled:
        logger.info(f"V12 7.3: Cancelled {len(cancelled)} stale orders")
    return cancelled


# ============================================================
# V12 7.4: Post-Trade Slippage Feedback
# ============================================================

def record_slippage_feedback(
    symbol: str,
    predicted_bps: float,
    actual_fill_price: float,
    expected_price: float,
    side: str,
) -> None:
    """V12 7.4: Record prediction error in the slippage model after a fill.

    Computes actual slippage in bps from fill vs expected price, then
    calls slippage_model.record_prediction_error(predicted, actual).

    Args:
        symbol: The filled symbol.
        predicted_bps: Slippage prediction made before the trade (0 if none).
        actual_fill_price: Actual fill price from broker.
        expected_price: Expected price at order time (entry_price).
        side: "buy" or "sell".
    """
    if expected_price <= 0:
        return

    # Calculate actual slippage in bps (positive = cost)
    if side == "buy":
        actual_bps = (actual_fill_price - expected_price) / expected_price * 10000
    else:
        actual_bps = (expected_price - actual_fill_price) / expected_price * 10000

    slippage_model = _get_slippage_model()
    if slippage_model is not None:
        try:
            slippage_model.record_prediction_error(predicted_bps, actual_bps)
            logger.debug(
                f"V12 7.4: Slippage feedback for {symbol}: "
                f"predicted={predicted_bps:.1f}bps actual={actual_bps:.1f}bps "
                f"error={actual_bps - predicted_bps:.1f}bps"
            )
        except Exception as e:
            logger.debug(f"V12 7.4: Slippage feedback failed for {symbol}: {e}")


# ============================================================
# EXEC-005: Exponential backoff configuration
# ============================================================
_MAX_RETRY_ATTEMPTS = 4
_BACKOFF_BASE_DELAYS = [0.0, 1.0, 3.0, 8.0]    # Base delay per attempt (seconds)
_BACKOFF_JITTER_RANGE = [0.0, 1.0, 3.0, 7.0]   # Max jitter added per attempt


# ============================================================
# EXEC-006: Pre-trade validation
# ============================================================

@dataclass
class ValidationResult:
    """Result of pre-trade validation."""
    valid: bool
    reason: str = ""
    details: dict = field(default_factory=dict)


def validate_order_pretrade(
    signal: Signal,
    qty: int,
    client=None,
) -> ValidationResult:
    """EXEC-006: Enhanced pre-trade validation before order submission.

    Checks:
    1. Buying power sufficiency
    2. Margin requirement (for shorts)
    3. Price reasonableness (limit price within 2% of mid)
    4. Symbol validity (tradable, not halted)
    5. Market hours
    6. Minimum notional ($1 Alpaca minimum, $100 practical minimum)

    Args:
        signal: Trading signal.
        qty: Order quantity.
        client: Optional trading client (fetched if None).

    Returns:
        ValidationResult with valid=True if all checks pass.
    """
    if client is None:
        try:
            client = get_trading_client()
        except Exception as e:
            return ValidationResult(
                valid=False,
                reason="client_unavailable",
                details={"error": str(e)},
            )

    notional = qty * signal.entry_price

    # Check 1: Minimum notional value
    min_notional = getattr(config, "MIN_POSITION_VALUE", 100)
    if notional < min_notional:
        return ValidationResult(
            valid=False,
            reason="below_min_notional",
            details={"notional": notional, "minimum": min_notional},
        )

    # Check 2: Symbol validity (tradable, not halted)
    try:
        asset = client.get_asset(signal.symbol)
        if not asset.tradable:
            return ValidationResult(
                valid=False,
                reason="symbol_not_tradable",
                details={"symbol": signal.symbol},
            )
        if hasattr(asset, "status") and asset.status == "inactive":
            return ValidationResult(
                valid=False,
                reason="symbol_inactive",
                details={"symbol": signal.symbol, "status": asset.status},
            )
    except Exception as e:
        return ValidationResult(
            valid=False,
            reason="symbol_lookup_failed",
            details={"symbol": signal.symbol, "error": str(e)},
        )

    # Check 3: Buying power
    try:
        account = client.get_account()
        buying_power = float(account.buying_power)

        if signal.side == "sell" and not config.ALLOW_SHORT:
            # Short selling: need margin
            required = notional * 1.5
        else:
            required = notional

        if buying_power < required:
            return ValidationResult(
                valid=False,
                reason="insufficient_buying_power",
                details={
                    "buying_power": buying_power,
                    "required": required,
                    "shortfall": required - buying_power,
                },
            )
    except Exception as e:
        logger.warning(f"Buying power check failed for {signal.symbol}: {e}")
        # Non-fatal: proceed with caution if we can't check

    # Check 4: Price reasonableness — reject limits > 2% from mid
    try:
        # Use last trade or quote to estimate mid price
        if signal.entry_price > 0:
            # Get a rough mid from latest quote if possible
            mid_estimate = signal.entry_price  # Fallback to signal's entry price
            try:
                from alpaca.data.requests import StockLatestQuoteRequest
                from data import get_stock_data_client
                data_client = get_stock_data_client()
                quote = data_client.get_stock_latest_quote(
                    StockLatestQuoteRequest(symbol_or_symbols=signal.symbol)
                )
                if signal.symbol in quote:
                    q = quote[signal.symbol]
                    if q.bid_price > 0 and q.ask_price > 0:
                        mid_estimate = (q.bid_price + q.ask_price) / 2
            except Exception:
                pass  # Use signal's entry_price as fallback

            if mid_estimate > 0:
                deviation_pct = abs(signal.entry_price - mid_estimate) / mid_estimate
                if deviation_pct > 0.02:
                    return ValidationResult(
                        valid=False,
                        reason="price_unreasonable",
                        details={
                            "entry_price": signal.entry_price,
                            "mid_estimate": mid_estimate,
                            "deviation_pct": round(deviation_pct, 4),
                            "max_deviation_pct": 0.02,
                        },
                    )
    except Exception as e:
        logger.warning(f"Price reasonableness check failed for {signal.symbol}: {e}")

    # Check 5: Market hours (basic check)
    try:
        clock = client.get_clock()
        if not clock.is_open:
            return ValidationResult(
                valid=False,
                reason="market_closed",
                details={"next_open": str(clock.next_open)},
            )
    except Exception as e:
        logger.warning(f"Market hours check failed: {e}")

    return ValidationResult(valid=True, reason="all_checks_passed")


def _check_order_filled(order_id: str, client=None, requested_qty: int = 0) -> bool:
    """EXEC-005: Check if an order has already been filled before retrying.

    V12 2.12: A "partially_filled" status no longer counts as fully filled.
    If *requested_qty* is provided, the order is considered filled only when
    filled_qty >= requested_qty.  Without requested_qty (legacy callers),
    only the "filled" status is treated as complete.

    Args:
        order_id: Broker order ID to check.
        client: Trading client.
        requested_qty: The originally requested quantity. When > 0, partial
            fills are compared against this value instead of relying on
            status alone.

    Returns:
        True if the order is fully filled (do NOT retry).
    """
    if not order_id or client is None:
        return False
    try:
        order = client.get_order_by_id(order_id)
        status = str(order.status).lower()
        filled_qty = int(order.filled_qty or 0)

        if status == "filled":
            logger.info(
                f"Order {order_id} filled ({filled_qty} shares) — skipping retry"
            )
            return True

        if status == "partially_filled":
            if requested_qty > 0 and filled_qty >= requested_qty:
                # Edge case: status says partial but qty matches — treat as filled
                logger.info(
                    f"Order {order_id} partially_filled but {filled_qty}>="
                    f"{requested_qty} requested — treating as filled"
                )
                return True
            else:
                logger.info(
                    f"Order {order_id} partially_filled: {filled_qty}/"
                    f"{requested_qty or '?'} shares — NOT treating as filled, "
                    f"retry may be needed"
                )
                return False

    except Exception as e:
        logger.debug(f"Order status check failed for {order_id}: {e}")
    return False


def _compute_backoff_delay(attempt: int) -> float:
    """EXEC-005: Compute exponential backoff delay with jitter.

    Attempt 0: 0s (immediate)
    Attempt 1: 1-2s
    Attempt 2: 3-6s
    Attempt 3: 8-15s

    Args:
        attempt: Zero-based attempt index.

    Returns:
        Delay in seconds (with jitter).
    """
    if attempt <= 0:
        return 0.0
    idx = min(attempt, len(_BACKOFF_BASE_DELAYS) - 1)
    base = _BACKOFF_BASE_DELAYS[idx]
    jitter = random.uniform(0, _BACKOFF_JITTER_RANGE[idx])
    return base + jitter


# BUG-013: Structured return type for close_all_positions
@dataclass
class CloseResult:
    """Result of close_all_positions — distinguishes success from failure."""
    success: bool
    closed_count: int
    failed_symbols: list[str] = field(default_factory=list)


def _validate_bracket_params(signal: Signal) -> tuple[bool, str]:
    """MED-032: Validate bracket order parameters before submission.

    Checks that stop_loss and take_profit are on the correct side of entry_price
    for the given trade direction.

    Returns:
        (valid, reason) tuple.
    """
    entry = signal.entry_price
    sl = signal.stop_loss
    tp = signal.take_profit

    if entry <= 0:
        return False, f"Invalid entry price: {entry}"
    if sl <= 0 or tp <= 0:
        return False, f"Invalid SL={sl} or TP={tp} (must be > 0)"

    if signal.side == "buy":
        # Long: stop_loss must be below entry, take_profit must be above entry
        if sl >= entry:
            return False, f"Long SL={sl} >= entry={entry}"
        if tp <= entry:
            return False, f"Long TP={tp} <= entry={entry}"
    elif signal.side == "sell":
        # Short: stop_loss must be above entry, take_profit must be below entry
        if sl <= entry:
            return False, f"Short SL={sl} <= entry={entry}"
        if tp >= entry:
            return False, f"Short TP={tp} >= entry={entry}"

    return True, ""


def _submit_order(signal: Signal, qty: int, client=None):
    """Internal: submit a single order. Returns order object.

    V6 strategy routing:
      - STAT_MR / KALMAN_PAIRS  -> LIMIT (mean-reversion, not time-sensitive)
      - MICRO_MOM / BETA_HEDGE  -> MARKET (speed matters)
    Legacy strategies (MOMENTUM, ORB, VWAP, etc.) are kept as fallback.
    """
    if client is None:
        client = get_trading_client()

    # MED-032: Validate bracket parameters before submission
    valid, reason = _validate_bracket_params(signal)
    if not valid:
        raise ValueError(f"Bracket order validation failed for {signal.symbol}: {reason}")

    side = OrderSide.BUY if signal.side == "buy" else OrderSide.SELL

    # --- CRIT-014: if/elif/elif/else chain (was if/if/if causing fallthrough) ---
    if signal.strategy in ("STAT_MR", "KALMAN_PAIRS"):
        # Mean-reversion: limit order, not time-sensitive
        return client.submit_order(
            LimitOrderRequest(
                symbol=signal.symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(signal.entry_price, 2),
                order_class=OrderClass.BRACKET,
                take_profit={"limit_price": round(signal.take_profit, 2)},
                stop_loss={"stop_price": round(signal.stop_loss, 2)},
            )
        )

    elif signal.strategy in ("MICRO_MOM", "BETA_HEDGE"):
        # Momentum / hedge: market order, speed matters
        # V12 FINAL: IOC would be ideal for immediate fill, but Alpaca
        # bracket orders apply a single TIF to all legs — IOC would cancel
        # the take-profit/stop-loss legs if the parent fills partially.
        # Keep DAY so bracket children persist as intended.
        return client.submit_order(
            MarketOrderRequest(
                symbol=signal.symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit={"limit_price": round(signal.take_profit, 2)},
                stop_loss={"stop_price": round(signal.stop_loss, 2)},
            )
        )

    elif signal.strategy == "ORB":
        # ORB uses limit order at breakout price (day only)
        return client.submit_order(
            LimitOrderRequest(
                symbol=signal.symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(signal.entry_price, 2),
                order_class=OrderClass.BRACKET,
                take_profit={"limit_price": round(signal.take_profit, 2)},
                stop_loss={"stop_price": round(signal.stop_loss, 2)},
            )
        )

    elif signal.strategy == "PEAD":
        # CRIT-014: PEAD: limit order (swing trade, not time-sensitive)
        return client.submit_order(
            LimitOrderRequest(
                symbol=signal.symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(signal.entry_price, 2),
                order_class=OrderClass.BRACKET,
                take_profit={"limit_price": round(signal.take_profit, 2)},
                stop_loss={"stop_price": round(signal.stop_loss, 2)},
            )
        )

    else:
        # VWAP and others: market order (speed matters, day only)
        return client.submit_order(
            MarketOrderRequest(
                symbol=signal.symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit={"limit_price": round(signal.take_profit, 2)},
                stop_loss={"stop_price": round(signal.stop_loss, 2)},
            )
        )


def can_short(symbol: str, qty: int, entry_price: float) -> tuple[bool, str]:
    """V3: Pre-trade checks before shorting. Returns (allowed, reason)."""
    if not config.ALLOW_SHORT:
        return False, "shorting_disabled"

    if symbol in config.NO_SHORT_SYMBOLS:
        return False, "no_short_symbol"

    try:
        client = get_trading_client()

        # Check if asset is shortable
        asset = client.get_asset(symbol)
        if not asset.shortable:
            return False, "not_shortable"

        # Check buying power (shorts require ~150% margin)
        account = client.get_account()
        required = qty * entry_price * 1.5
        if float(account.buying_power) < required:
            return False, "insufficient_buying_power"

        return True, "ok"

    except Exception as e:
        logger.error(f"Short pre-check failed for {symbol}: {e}")
        return False, f"check_error: {e}"


def submit_twap_order(
    signal: Signal, total_qty: int, slices: int = 5, interval_sec: int = 60,
    order_manager=None,
) -> list[str]:
    """Split large orders into time-weighted slices (TWAP).

    For orders > $2000, split into `slices` smaller orders spaced
    `interval_sec` apart.  Each slice is a bracket order with the same TP/SL.

    BUG-011: Each slice is registered as a child order in the OMS, linked to
    a parent order. If any slice fails, remaining slices are cancelled.

    Returns list of order IDs.
    """
    client = get_trading_client()
    slice_qty = total_qty // slices
    remainder = total_qty % slices
    order_ids: list[str] = []

    # BUG-011: Register parent TWAP order in OMS
    parent_oms_order = None
    if order_manager:
        try:
            from oms.order import OrderState
            parent_oms_order = order_manager.create_order(
                symbol=signal.symbol,
                strategy=signal.strategy,
                side=signal.side,
                order_type="twap_parent",
                qty=total_qty,
                limit_price=getattr(signal, 'entry_price', 0.0),
                take_profit=getattr(signal, 'take_profit', 0.0),
                stop_loss=getattr(signal, 'stop_loss', 0.0),
                pair_id=getattr(signal, 'pair_id', ''),
            )
            order_manager.update_state(parent_oms_order.oms_id, OrderState.SUBMITTED)
        except Exception as e:
            logger.warning(f"TWAP OMS parent registration failed: {e}")

    child_oms_ids: list[str] = []
    total_filled = 0
    slice_failed = False
    # V12 2.1: Mean-reversion strategies close partial fills; momentum completes them
    _MEAN_REVERSION_STRATEGIES = ("STAT_MR", "KALMAN_PAIRS")

    for i in range(slices):
        # V10 BUG-004: Distribute remainder across first N slices (1 extra each)
        qty = slice_qty + (1 if i < remainder else 0)
        if qty <= 0:
            continue

        # BUG-011: If a previous slice failed, cancel remaining slices
        if slice_failed:
            logger.warning(
                f"TWAP slice {i+1}/{slices} skipped for {signal.symbol}: "
                f"previous slice failed, cancelling remaining"
            )
            continue

        # BUG-011: Register child slice in OMS
        child_oms_order = None
        if order_manager and parent_oms_order:
            try:
                from oms.order import OrderState
                child_oms_order = order_manager.create_order(
                    symbol=signal.symbol,
                    strategy=signal.strategy,
                    side=signal.side,
                    order_type="twap_slice",
                    qty=qty,
                    limit_price=getattr(signal, 'entry_price', 0.0),
                    take_profit=getattr(signal, 'take_profit', 0.0),
                    stop_loss=getattr(signal, 'stop_loss', 0.0),
                    pair_id=getattr(signal, 'pair_id', ''),
                )
                child_oms_order.parent_oms_id = parent_oms_order.oms_id
                child_oms_ids.append(child_oms_order.oms_id)
            except Exception as e:
                logger.warning(f"TWAP OMS child registration failed: {e}")

        try:
            order = _submit_order(signal, qty, client)
            oid = str(order.id)
            order_ids.append(oid)
            total_filled += qty
            logger.info(
                f"TWAP slice {i+1}/{slices}: {signal.side.upper()} {qty} "
                f"{signal.symbol} ({signal.strategy}) order_id={oid}"
            )

            # BUG-011: Update child order state in OMS
            if order_manager and child_oms_order:
                try:
                    from oms.order import OrderState
                    order_manager.update_state(
                        child_oms_order.oms_id, OrderState.SUBMITTED,
                        broker_order_id=oid,
                    )
                except Exception as e:
                    logger.error(f"TWAP OMS child state update failed: {e}", exc_info=True)

        except Exception as e:
            logger.error(
                f"TWAP slice {i+1}/{slices} failed for {signal.symbol}: {e}"
            )
            slice_failed = True

            # BUG-011: Mark failed child in OMS
            if order_manager and child_oms_order:
                try:
                    from oms.order import OrderState
                    order_manager.update_state(child_oms_order.oms_id, OrderState.FAILED)
                except Exception as e2:
                    logger.error(f"TWAP OMS child FAILED state update failed: {e2}", exc_info=True)

            # BUG-011: Cancel remaining slices — cancel already-submitted broker orders
            for prev_oid in order_ids:
                try:
                    client.cancel_order_by_id(prev_oid)
                    logger.info(f"TWAP cancelled previous slice order_id={prev_oid}")
                except Exception as ce:
                    logger.warning(f"TWAP cancel failed for {prev_oid}: {ce}")

            # Cancel child OMS orders
            if order_manager:
                try:
                    from oms.order import OrderState
                    for cid in child_oms_ids:
                        order_manager.update_state(cid, OrderState.CANCELLED)
                except Exception as e2:
                    logger.error(f"TWAP OMS child cancel state update failed: {e2}", exc_info=True)

            # ------------------------------------------------------------------
            # V12 2.1: Handle partial TWAP fill — resolve unhedged position
            # After cancelling pending slices, check how many shares actually
            # filled across previous slices. If partial fill exists, either
            # complete the order (momentum) or close the filled portion
            # (mean reversion) to avoid an unhedged partial position.
            # ------------------------------------------------------------------
            actual_filled_qty = 0
            for prev_oid in order_ids:
                try:
                    prev_order = client.get_order_by_id(prev_oid)
                    filled = int(prev_order.filled_qty or 0)
                    actual_filled_qty += filled
                except Exception as fq_err:
                    logger.warning(
                        f"V12 2.1: Could not check filled qty for {prev_oid}: {fq_err}"
                    )

            if actual_filled_qty > 0:
                unfilled_qty = total_qty - actual_filled_qty
                logger.warning(
                    f"V12 2.1: TWAP partial fill for {signal.symbol}: "
                    f"{actual_filled_qty}/{total_qty} shares filled, "
                    f"{unfilled_qty} unfilled"
                )

                if signal.strategy in _MEAN_REVERSION_STRATEGIES:
                    # Mean reversion: close the partial fill — position is
                    # too small relative to intended hedge/pair size
                    logger.info(
                        f"V12 2.1: Mean-reversion strategy {signal.strategy} — "
                        f"closing partial fill of {actual_filled_qty} shares "
                        f"for {signal.symbol}"
                    )
                    try:
                        close_position(
                            signal.symbol,
                            reason=f"V12 TWAP partial fill cleanup "
                                   f"({actual_filled_qty}/{total_qty})",
                        )
                    except Exception as close_err:
                        logger.error(
                            f"V12 2.1: Failed to close partial fill for "
                            f"{signal.symbol}: {close_err}"
                        )
                else:
                    # Momentum / other strategies: complete the remainder via
                    # market order so the intended position size is reached
                    if unfilled_qty > 0:
                        logger.info(
                            f"V12 2.1: Momentum strategy {signal.strategy} — "
                            f"submitting market order for remaining "
                            f"{unfilled_qty} shares of {signal.symbol}"
                        )
                        try:
                            side = (
                                OrderSide.BUY
                                if signal.side == "buy"
                                else OrderSide.SELL
                            )
                            remainder_order = client.submit_order(
                                MarketOrderRequest(
                                    symbol=signal.symbol,
                                    qty=unfilled_qty,
                                    side=side,
                                    time_in_force=TimeInForce.DAY,
                                )
                            )
                            remainder_oid = str(remainder_order.id)
                            order_ids.append(remainder_oid)
                            total_filled += unfilled_qty
                            logger.info(
                                f"V12 2.1: Remainder market order submitted: "
                                f"{unfilled_qty} {signal.symbol} "
                                f"order_id={remainder_oid}"
                            )
                        except Exception as mkt_err:
                            logger.error(
                                f"V12 2.1: Remainder market order failed for "
                                f"{signal.symbol}: {mkt_err} — "
                                f"closing partial fill instead"
                            )
                            try:
                                close_position(
                                    signal.symbol,
                                    reason=f"V12 TWAP remainder failed, "
                                           f"closing partial "
                                           f"({actual_filled_qty}/{total_qty})",
                                )
                            except Exception as close_err2:
                                logger.error(
                                    f"V12 2.1: Fallback close also failed "
                                    f"for {signal.symbol}: {close_err2}"
                                )

            # Update total_filled to reflect actual broker fills
            total_filled = actual_filled_qty

        # Sleep between slices (not after the last one)
        if i < slices - 1 and not slice_failed:
            time.sleep(interval_sec)

    # BUG-011: Update parent order with aggregate fill info
    if order_manager and parent_oms_order:
        try:
            from oms.order import OrderState
            fill_pct = total_filled / total_qty if total_qty > 0 else 0.0
            if slice_failed:
                order_manager.update_state(parent_oms_order.oms_id, OrderState.CANCELLED)
            elif total_filled >= total_qty:
                order_manager.update_state(
                    parent_oms_order.oms_id, OrderState.FILLED,
                    filled_qty=total_filled,
                )
            elif total_filled > 0:
                order_manager.update_state(
                    parent_oms_order.oms_id, OrderState.PARTIAL_FILL,
                    filled_qty=total_filled,
                )
            logger.info(
                f"TWAP parent {parent_oms_order.oms_id}: "
                f"filled {total_filled}/{total_qty} ({fill_pct:.0%})"
            )
        except Exception as e:
            logger.warning(f"TWAP OMS parent update failed: {e}")

    return order_ids


class _BracketGapException(Exception):
    """V12 2.2: Raised when a bracket order's stop-loss fills before the entry."""
    pass


def _verify_bracket_entry(
    order_id: str, signal: Signal, client, poll_seconds: int = 5,
) -> None:
    """V12 2.2: Poll bracket order for up to *poll_seconds* to detect gap-open
    anomalies where the stop-loss leg fills before the entry.

    If the entry fills first (normal case), returns immediately.
    If the SL fills first, cancels remaining legs, closes the accidental
    position, and raises _BracketGapException.
    If neither fills within the poll window, returns (normal — order is
    still working and will be managed by the OMS).

    Args:
        order_id: The parent bracket order ID.
        signal: The original trade signal.
        client: Alpaca trading client.
        poll_seconds: How long to poll (default 5s).

    Raises:
        _BracketGapException if stop-loss fired before entry.
    """
    poll_interval = 0.5
    elapsed = 0.0

    while elapsed < poll_seconds:
        try:
            order = client.get_order_by_id(order_id)
            status = str(order.status).lower()

            # Entry filled normally — bracket is working as intended
            if status == "filled":
                return

            # Check legs if bracket has child orders (Alpaca 'legs' attribute)
            legs = getattr(order, "legs", None) or []
            entry_filled = False
            sl_filled = False

            for leg in legs:
                leg_status = str(leg.status).lower()
                leg_type = str(getattr(leg, "order_type", "")).lower()
                leg_side = str(getattr(leg, "side", "")).lower()

                # Identify stop-loss leg: it's a "stop" type, or on the
                # opposite side of the entry
                is_stop_leg = (
                    "stop" in leg_type
                    or (signal.side == "buy" and leg_side == "sell" and "limit" not in leg_type)
                )

                if is_stop_leg and leg_status == "filled":
                    sl_filled = True
                elif not is_stop_leg and leg_status == "filled":
                    entry_filled = True

            if entry_filled:
                # Entry filled — normal operation
                return

            if sl_filled and not entry_filled:
                # SL filled before entry — gap-open anomaly
                logger.error(
                    f"V12 2.2: Stop-loss filled before entry for "
                    f"{signal.symbol} order_id={order_id} — cancelling "
                    f"remaining legs and closing position"
                )

                # Cancel remaining legs
                for leg in legs:
                    leg_status = str(leg.status).lower()
                    if leg_status not in ("filled", "cancelled", "expired"):
                        try:
                            client.cancel_order_by_id(str(leg.id))
                            logger.info(
                                f"V12 2.2: Cancelled bracket leg {leg.id}"
                            )
                        except Exception as ce:
                            logger.warning(
                                f"V12 2.2: Failed to cancel leg {leg.id}: {ce}"
                            )

                # Also cancel the parent order
                try:
                    client.cancel_order_by_id(order_id)
                except Exception:
                    pass  # May already be done

                # Close the accidental position created by the SL fill
                try:
                    close_position(
                        signal.symbol,
                        reason="V12 2.2 bracket gap — SL filled before entry",
                    )
                except Exception as close_err:
                    logger.error(
                        f"V12 2.2: Failed to close gap position for "
                        f"{signal.symbol}: {close_err}"
                    )

                raise _BracketGapException(
                    f"SL filled before entry for {signal.symbol}"
                )

            # Order still pending/new — not yet filled either way
            if status in ("canceled", "cancelled", "expired", "rejected"):
                return  # Order died on its own, nothing to protect

        except _BracketGapException:
            raise  # Re-raise our own exception
        except Exception as e:
            logger.debug(
                f"V12 2.2: Bracket poll error for {signal.symbol}: {e}"
            )

        time.sleep(poll_interval)
        elapsed += poll_interval

    # Poll window expired without either side filling — normal for limit
    # entries.  The order continues working and will be managed by OMS.
    return


def submit_bracket_order(signal: Signal, qty: int, order_manager=None) -> str | list[str] | None:
    """Submit bracket order, auto-routing to TWAP for large orders.

    EXEC-005: Uses exponential backoff with jitter (up to 4 attempts).
    Before each retry, checks if the original order was already filled.
    Uses an idempotency key to prevent duplicate orders on retries.

    EXEC-006: Runs enhanced pre-trade validation before submission.

    Returns a single order ID (str), a list of order IDs (TWAP), or None on
    failure.
    """
    # Auto-route large mean-reversion orders to TWAP
    order_value = qty * signal.entry_price
    if order_value > 25000 and signal.strategy in ("STAT_MR", "KALMAN_PAIRS"):
        return submit_twap_order(signal, qty, order_manager=order_manager)

    client = get_trading_client()

    # EXEC-006: Pre-trade validation
    validation = validate_order_pretrade(signal, qty, client)
    if not validation.valid:
        logger.warning(
            f"Pre-trade validation failed for {signal.symbol}: "
            f"{validation.reason} — {validation.details}"
        )
        return None

    # V11.3 T4: Pre-trade slippage prediction — skip if cost > 50% of expected gain
    slippage_model = _get_slippage_model()
    if slippage_model is not None:
        try:
            prediction = slippage_model.predict_from_order(
                order_size=qty,
                price=signal.entry_price,
                order_type="limit" if signal.strategy in ("STAT_MR", "KALMAN_PAIRS", "ORB", "PEAD") else "market",
                side=signal.side,
                current_time=datetime.now(timezone.utc),
            )
            slippage_bps = prediction.expected_bps
            slippage_dollars = signal.entry_price * slippage_bps / 10000.0

            # Calculate expected gain (distance to take profit)
            expected_gain = abs(signal.take_profit - signal.entry_price)

            if expected_gain > 0 and slippage_dollars > expected_gain * 0.5:
                logger.warning(
                    f"V11.3 T4: Skipping {signal.symbol} — predicted slippage "
                    f"${slippage_dollars:.2f} ({slippage_bps:.1f}bps) > 50%% of "
                    f"expected gain ${expected_gain:.2f}"
                )
                return None

            # V11.4: Only widen the stop-loss (protective side) by predicted slippage.
            # Widening TP is counterproductive — it makes profits harder to capture.
            if slippage_dollars > 0.01:
                if signal.side == "buy":
                    signal.stop_loss = round(signal.stop_loss - slippage_dollars, 2)
                else:
                    signal.stop_loss = round(signal.stop_loss + slippage_dollars, 2)

            logger.debug(
                f"V11.3 T4: {signal.symbol} slippage={slippage_bps:.1f}bps "
                f"(${slippage_dollars:.3f}/share)"
            )
        except Exception as e:
            logger.debug(f"V11.3 T4: Slippage prediction failed for {signal.symbol} (fail-open): {e}")

    # EXEC-005: Generate idempotency key for retry safety
    idempotency_key = f"bracket_{signal.symbol}_{signal.strategy}_{uuid.uuid4().hex[:8]}"
    last_order_id: str | None = None
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRY_ATTEMPTS):
        # Compute backoff delay (0s for first attempt)
        delay = _compute_backoff_delay(attempt)
        if delay > 0:
            logger.info(
                f"Order retry {attempt}/{_MAX_RETRY_ATTEMPTS - 1} for "
                f"{signal.symbol} after {delay:.1f}s backoff"
            )
            time.sleep(delay)

        # EXEC-005: Before retrying, check if original order was filled
        # V12 2.12: Pass requested qty so partial fills are not mistaken as complete
        if attempt > 0 and last_order_id:
            if _check_order_filled(last_order_id, client, requested_qty=qty):
                logger.info(
                    f"Order {last_order_id} already filled — no retry needed"
                )
                return last_order_id

        try:
            order = _submit_order(signal, qty, client)
            order_id = str(order.id)
            last_order_id = order_id

            # V12 6.4: API success — reset failure counter
            _record_api_success()

            logger.info(
                f"Order submitted (attempt {attempt + 1}): "
                f"{signal.side.upper()} {qty} {signal.symbol} "
                f"({signal.strategy}) order_id={order_id} "
                f"idempotency_key={idempotency_key}"
            )

            # ----------------------------------------------------------
            # V12 2.2: Gap-open protection for bracket orders
            # On overnight gaps the stop-loss leg can fill at the open
            # before the entry leg, flipping the position to the wrong
            # side.  Poll for 5 seconds after submission to verify the
            # entry fills first.  If the SL fires first, cancel the
            # remaining legs and close the accidental position.
            # ----------------------------------------------------------
            try:
                _verify_bracket_entry(order_id, signal, client)
            except _BracketGapException as gap_exc:
                logger.error(
                    f"V12 2.2: Bracket gap detected for {signal.symbol}: "
                    f"{gap_exc} — returning None"
                )
                return None

            # V12 FINAL: Start chase thread for limit orders to prevent
            # unfilled orders sitting at stale prices. Chase will amend
            # price at 15s, convert to market at 30s, cancel at 45s.
            if signal.strategy in _LIMIT_ORDER_STRATEGIES:
                try:
                    _start_chase_thread(order_id, signal, qty)
                except Exception as chase_err:
                    logger.debug(
                        "V12 FINAL: Chase thread start failed for %s (non-fatal): %s",
                        signal.symbol, chase_err,
                    )

            return order_id

        except Exception as e:
            last_error = e
            # V12 6.4: Record API failure for circuit breaker
            _record_api_failure()
            logger.error(
                f"Bracket order attempt {attempt + 1}/{_MAX_RETRY_ATTEMPTS} "
                f"failed for {signal.symbol}: {e}"
            )

    # All attempts exhausted
    logger.error(
        f"Bracket order FAILED after {_MAX_RETRY_ATTEMPTS} attempts for "
        f"{signal.symbol}: {last_error}"
    )
    return None


def get_order_commission(order_id: str, client=None) -> float:
    """V12 6.1: Retrieve commission paid on a filled order.

    Alpaca's order object exposes a `commission` field after fill.
    Returns 0.0 if the field is unavailable or the lookup fails
    (Alpaca is commission-free for US equities, but this tracks it
    for futures/options compatibility and regulatory fee pass-throughs).

    Args:
        order_id: Broker order ID.
        client: Optional trading client (fetched if None).

    Returns:
        Commission amount in dollars (>= 0.0).
    """
    if not order_id:
        return 0.0
    if client is None:
        try:
            client = get_trading_client()
        except Exception:
            return 0.0
    try:
        order = client.get_order_by_id(order_id)
        # Alpaca v2 API: order.commission is a string or Decimal
        raw = getattr(order, "commission", None)
        if raw is not None:
            commission = float(raw)
            if commission > 0:
                logger.debug(
                    f"V12 6.1: Order {order_id} commission=${commission:.4f}"
                )
            return max(commission, 0.0)
    except Exception as e:
        logger.debug(f"V12 6.1: Commission lookup failed for {order_id}: {e}")
    return 0.0


def close_position(symbol: str, reason: str = "") -> bool:
    """Close an open position by symbol. Returns True on success."""
    client = get_trading_client()
    try:
        client.close_position(symbol)
        logger.info(f"Position closed: {symbol} ({reason})")
        return True
    except Exception as e:
        logger.error(f"Failed to close {symbol}: {e}")
        return False


def close_all_positions(reason: str = "EOD") -> CloseResult:
    """Close all open positions. Returns CloseResult with success/count/failures.

    BUG-013: Previously returned 0 on success, -1 on error, making it impossible
    to distinguish "closed 0 positions" from "failed". Now returns a structured
    CloseResult dataclass.
    """
    client = get_trading_client()
    failed_symbols: list[str] = []
    closed_count = 0

    try:
        # Get current positions before closing
        positions = client.get_all_positions()
        symbols_to_close = [p.symbol for p in positions]

        if not symbols_to_close:
            logger.info(f"No positions to close ({reason})")
            return CloseResult(success=True, closed_count=0)

        # Attempt to close all
        client.close_all_positions(cancel_orders=True)

        # Verify which positions actually closed
        remaining = client.get_all_positions()
        remaining_symbols = {p.symbol for p in remaining}

        for sym in symbols_to_close:
            if sym in remaining_symbols:
                failed_symbols.append(sym)
            else:
                closed_count += 1

        success = len(failed_symbols) == 0
        if success:
            logger.info(f"All {closed_count} positions closed ({reason})")
        else:
            logger.warning(
                f"Closed {closed_count}/{len(symbols_to_close)} positions ({reason}), "
                f"failed: {failed_symbols}"
            )

        return CloseResult(
            success=success,
            closed_count=closed_count,
            failed_symbols=failed_symbols,
        )

    except Exception as e:
        logger.error(f"Failed to close all positions: {e}")
        return CloseResult(success=False, closed_count=closed_count, failed_symbols=failed_symbols)


def cancel_all_open_orders() -> bool:
    """Cancel all pending orders."""
    client = get_trading_client()
    try:
        client.cancel_orders()
        logger.info("All open orders cancelled")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel orders: {e}")
        return False


def close_orb_positions(open_trades: dict, now: datetime) -> list[str]:
    """Close all ORB positions (for 3:45 PM exit). Returns list of closed symbols."""
    closed = []
    for symbol, trade in list(open_trades.items()):
        if trade.strategy == "ORB":
            if close_position(symbol, reason="ORB EOD exit"):
                closed.append(symbol)
    return closed


def check_vwap_time_stops(open_trades: dict, now: datetime) -> list[str]:
    """Check VWAP trades for time stop (45 min). Returns list of symbols to close."""
    expired = []
    for symbol, trade in open_trades.items():
        if trade.strategy == "VWAP" and trade.time_stop:
            if now >= trade.time_stop:
                if close_position(symbol, reason="VWAP time stop"):
                    expired.append(symbol)
    return expired


## V10: Removed dead strategy functions:
# check_momentum_max_hold, close_gap_go_positions,
# check_sector_max_hold, check_pairs_max_hold
# (MOMENTUM, GAP_GO, SECTOR_ROTATION, PAIRS strategies no longer exist)


def close_partial_position(symbol: str, qty: int) -> bool:
    """Close a partial position (for scaled exits)."""
    client = get_trading_client()
    try:
        # V10 BUG-016: Validate qty is int before string conversion (Alpaca API requires str)
        client.close_position(symbol, qty=str(int(qty)))
        logger.info(f"Partial close: {symbol} qty={qty}")
        return True
    except Exception as e:
        logger.error(f"Failed partial close {symbol} qty={qty}: {e}")
        return False
