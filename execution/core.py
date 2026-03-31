"""Order execution — bracket orders, TWAP splitting, retries, EOD exits.

EXEC-005: Exponential backoff with jitter for order submission retries.
EXEC-006: Enhanced pre-trade validation (buying power, price reasonableness, etc.).
"""

import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

import config
from data import get_trading_client
from strategies.base import Signal

logger = logging.getLogger(__name__)

# V11.3 T4: Lazy-loaded slippage model singleton
_slippage_model = None


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


def _check_order_filled(order_id: str, client=None) -> bool:
    """EXEC-005: Check if an order has already been filled before retrying.

    Args:
        order_id: Broker order ID to check.
        client: Trading client.

    Returns:
        True if the order is filled (do NOT retry).
    """
    if not order_id or client is None:
        return False
    try:
        order = client.get_order_by_id(order_id)
        status = str(order.status).lower()
        if status in ("filled", "partially_filled"):
            logger.info(
                f"Order {order_id} already {status} — skipping retry"
            )
            return True
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
        if attempt > 0 and last_order_id:
            if _check_order_filled(last_order_id, client):
                logger.info(
                    f"Order {last_order_id} already filled — no retry needed"
                )
                return last_order_id

        try:
            order = _submit_order(signal, qty, client)
            order_id = str(order.id)
            last_order_id = order_id

            logger.info(
                f"Order submitted (attempt {attempt + 1}): "
                f"{signal.side.upper()} {qty} {signal.symbol} "
                f"({signal.strategy}) order_id={order_id} "
                f"idempotency_key={idempotency_key}"
            )
            return order_id

        except Exception as e:
            last_error = e
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
