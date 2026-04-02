"""V10 Engine — Exit processing for strategy-driven and WebSocket-triggered closes.

Emits position.closed and position.partial_close events on the event bus.
"""

import logging
from datetime import datetime, timedelta

import config
from data import get_snapshot, get_snapshots, get_filled_exit_price
from execution import close_position, close_partial_position
from risk import RiskManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# V11.5: Advanced exit state tracking
# ---------------------------------------------------------------------------
# Tracks how many profit-tier partials have been taken per symbol (0, 1, or 2)
_profit_tier_exits: dict[str, int] = {}
# Tracks symbols that have already had a scale-out-on-loser partial exit
_scaled_out: set[str] = set()

# Day strategies eligible for dead-signal exit
_DAY_STRATEGIES = {"STAT_MR", "VWAP", "MICRO_MOM", "ORB"}

# Event bus integration (fail-open)
try:
    from engine.events import get_event_bus, Event, EventTypes
    _EVENTS_AVAILABLE = True
except ImportError:
    _EVENTS_AVAILABLE = False


def _emit_event(event_type: str, data: dict, source: str = "exit_processor"):
    if _EVENTS_AVAILABLE:
        try:
            bus = get_event_bus()
            bus.publish(Event(event_type, data, source=source))
        except Exception:
            pass

# Lazy-loaded optional modules
_notifications = None
_intraday_controls = None

# Adaptive exit manager for STAT_MR and VWAP positions (fail-open)
_adaptive_exit_mgr = None
_ADAPTIVE_EXIT_AVAILABLE = False
try:
    from adaptive_exit_manager import AdaptiveExitManager
    from risk import get_vix_level
    _adaptive_exit_mgr = AdaptiveExitManager()
    _ADAPTIVE_EXIT_AVAILABLE = True
except ImportError:
    pass

_ADAPTIVE_EXIT_STRATEGIES = {"STAT_MR", "VWAP"}


def set_intraday_controls(controls) -> None:
    """Set the shared IntradayRiskControls instance (called from main.py)."""
    global _intraday_controls
    _intraday_controls = controls


def _get_notifications():
    global _notifications
    if _notifications is None:
        try:
            import notifications as _n
            _notifications = _n
        except ImportError:
            _notifications = False
    return _notifications if _notifications else None


def now_et():
    return datetime.now(config.ET)


def check_profit_tiers(trade, current_price: float) -> list[dict]:
    """V11.5: Tiered profit-taking — partial exits at +1.5% and +2.5%.

    Tier 1 (+1.5%): exit 33% of position
    Tier 2 (+2.5%): exit another 33%
    Remaining 34% rides with trailing stop.

    Returns a list of exit action dicts (0 or 1 item).
    """
    symbol = trade.symbol
    direction = 1 if trade.side == "buy" else -1
    unrealized_pct = (current_price - trade.entry_price) / trade.entry_price * direction

    tiers_taken = _profit_tier_exits.get(symbol, 0)

    if tiers_taken >= 2:
        return []

    if tiers_taken == 0 and unrealized_pct >= 0.015:
        # Tier 1: exit 33%
        partial_qty = max(1, int(trade.qty * 0.33))
        _profit_tier_exits[symbol] = 1
        logger.info(
            "V11.5: Profit tier 1 hit for %s — %.2f%% unrealized, exiting %d of %d shares",
            symbol, unrealized_pct * 100, partial_qty, trade.qty,
        )
        return [{
            "symbol": symbol,
            "action": "partial",
            "qty": partial_qty,
            "reason": "profit_tier_1_1.5pct",
        }]

    if tiers_taken == 1 and unrealized_pct >= 0.025:
        # Tier 2: exit another 33%
        partial_qty = max(1, int(trade.qty * 0.33))
        _profit_tier_exits[symbol] = 2
        logger.info(
            "V11.5: Profit tier 2 hit for %s — %.2f%% unrealized, exiting %d of %d shares",
            symbol, unrealized_pct * 100, partial_qty, trade.qty,
        )
        return [{
            "symbol": symbol,
            "action": "partial",
            "qty": partial_qty,
            "reason": "profit_tier_2_2.5pct",
        }]

    return []


def _check_dead_signal(trade, current_price: float, now: datetime) -> list[dict]:
    """V11.5: Exit positions with no conviction after 30 minutes.

    If a day-strategy position has been held > 30 min with unrealized P&L
    between -0.3% and +0.3%, close it entirely — the signal had no conviction.
    """
    if trade.strategy not in _DAY_STRATEGIES:
        return []

    hold_duration = now - trade.entry_time
    if hold_duration < timedelta(minutes=30):
        return []

    direction = 1 if trade.side == "buy" else -1
    unrealized_pct = (current_price - trade.entry_price) / trade.entry_price * direction

    if -0.003 <= unrealized_pct <= 0.003:
        logger.info(
            "V11.5: Dead signal for %s (%s) — held %.1f min, unrealized %.2f%%, closing",
            trade.symbol, trade.strategy, hold_duration.total_seconds() / 60,
            unrealized_pct * 100,
        )
        return [{
            "symbol": trade.symbol,
            "action": "full",
            "reason": "dead_signal",
        }]
    return []


def _check_scale_out_loser(trade, current_price: float) -> list[dict]:
    """V11.5: Scale out 50% when unrealized loss exceeds -1.0%.

    Reduces exposure before the stop-loss is hit. Only triggers once per trade.
    """
    symbol = trade.symbol
    if symbol in _scaled_out:
        return []

    direction = 1 if trade.side == "buy" else -1
    unrealized_pct = (current_price - trade.entry_price) / trade.entry_price * direction

    if unrealized_pct <= -0.01:
        partial_qty = max(1, trade.qty // 2)
        _scaled_out.add(symbol)
        logger.info(
            "V11.5: Scale-out loser %s — %.2f%% unrealized, exiting %d of %d shares",
            symbol, unrealized_pct * 100, partial_qty, trade.qty,
        )
        return [{
            "symbol": symbol,
            "action": "partial",
            "qty": partial_qty,
            "reason": "scale_out_loser_1pct",
        }]
    return []


def check_advanced_exits(risk: RiskManager, now: datetime) -> list[dict]:
    """V11.5: Run all advanced exit checks on open positions.

    Called from the main loop alongside strategy check_exits().
    Returns a list of exit action dicts to be fed into handle_strategy_exits().
    """
    actions: list[dict] = []
    if not risk.open_trades:
        return actions

    # Fetch current prices for all open positions in one batch
    prices = get_current_prices(risk.open_trades)

    for symbol, trade in list(risk.open_trades.items()):
        current_price = prices.get(symbol)
        if current_price is None:
            continue

        # 1. Profit-taking tiers
        actions.extend(check_profit_tiers(trade, current_price))

        # 2. Dead signal exit (day strategies only, held > 30 min, flat P&L)
        actions.extend(_check_dead_signal(trade, current_price, now))

        # 3. Scale-out on losers approaching stop
        actions.extend(_check_scale_out_loser(trade, current_price))

        # 4. Adaptive exit for STAT_MR and VWAP positions (fail-open)
        if _ADAPTIVE_EXIT_AVAILABLE and trade.strategy in _ADAPTIVE_EXIT_STRATEGIES:
            try:
                vix = get_vix_level()
                ou_params = {
                    'mu': trade.entry_price,
                    'sigma': trade.entry_atr if trade.entry_atr > 0 else trade.entry_price * 0.02,
                    'half_life': 24.0,
                }
                exit_reason, exit_type = _adaptive_exit_mgr.should_exit(
                    position_side=trade.side,
                    position_entry_time=trade.entry_time,
                    current_price=current_price,
                    ou_params=ou_params,
                    vix=vix,
                    partial_exits=trade.partial_exits,
                )
                if exit_type == 'full':
                    actions.append({
                        "symbol": symbol,
                        "action": "full",
                        "reason": f"adaptive_exit_{exit_reason}",
                    })
                elif exit_type == 'partial':
                    partial_qty = max(1, trade.qty // 2)
                    actions.append({
                        "symbol": symbol,
                        "action": "partial",
                        "qty": partial_qty,
                        "reason": f"adaptive_exit_{exit_reason}",
                    })
            except Exception as e:
                logger.debug("Adaptive exit check failed for %s (fail-open): %s", symbol, e)

    return actions


def cleanup_exit_state(symbol: str) -> None:
    """Clean up advanced exit tracking when a position is fully closed."""
    _profit_tier_exits.pop(symbol, None)
    _scaled_out.discard(symbol)


def handle_strategy_exits(exit_actions: list[dict], risk: RiskManager, now: datetime, ws_monitor=None):
    """Process exit actions returned by strategy check_exits() methods.

    Each action dict: {symbol, action, reason, ...}
    action = "full" -> close_position, "partial" -> close_partial_position
    """
    notif = _get_notifications()

    for action in exit_actions:
        symbol = action["symbol"]
        if symbol not in risk.open_trades:
            continue
        trade = risk.open_trades[symbol]
        reason = action.get("reason", "strategy_exit")

        if action.get("action") == "partial":
            partial_qty = action.get("qty", max(1, trade.qty // 2))
            try:
                close_partial_position(symbol, partial_qty)
                logger.info(f"Partial exit {symbol}: {partial_qty} shares, reason={reason}")
                _emit_event(EventTypes.POSITION_PARTIAL_CLOSE if _EVENTS_AVAILABLE else "position.partial_close", {
                    "symbol": symbol, "strategy": trade.strategy, "qty": partial_qty, "reason": reason,
                })
            except Exception as e:
                logger.error(f"Partial close failed for {symbol}: {e}")
        else:
            try:
                close_position(symbol, reason=reason)
            except Exception as e:
                logger.error(f"Close failed for {symbol}: {e}")
                continue

            exit_price = get_filled_exit_price(symbol, side=trade.side)
            if exit_price is None:
                try:
                    snap = get_snapshot(symbol)
                    exit_price = float(snap.latest_trade.price) if snap and snap.latest_trade else trade.entry_price
                except Exception:
                    exit_price = trade.entry_price

            pnl = (exit_price - trade.entry_price) * trade.qty * (1 if trade.side == "buy" else -1)
            risk.close_trade(symbol, exit_price, now, exit_reason=reason)

            # V11.5: Clean up advanced exit tracking state
            cleanup_exit_state(symbol)

            # V11.3 T2: Feed intraday risk controls with P&L data
            if _intraday_controls is not None:
                try:
                    pnl_pct = pnl / max(risk.current_equity, 1)
                    is_stop = "stop" in reason.lower()
                    _intraday_controls.record_pnl(pnl_pct, is_stop_loss=is_stop,
                                                   is_loss=(pnl < 0), is_win=(pnl > 0), now=now)
                except Exception:
                    pass

            # Register cooldown if this was a stop-loss exit
            if "stop" in reason.lower():
                try:
                    from engine.signal_processor import register_stopout
                    register_stopout(symbol)
                except Exception:
                    pass
            _emit_event(EventTypes.POSITION_CLOSED if _EVENTS_AVAILABLE else "position.closed", {
                "symbol": symbol, "strategy": trade.strategy, "side": trade.side,
                "entry_price": trade.entry_price, "exit_price": exit_price,
                "qty": trade.qty, "pnl": round(pnl, 2), "reason": reason,
            })
            if ws_monitor:
                ws_monitor.unsubscribe(symbol)
            if notif and config.TELEGRAM_ENABLED:
                try:
                    notif.notify_trade_closed(trade)
                except Exception as e:
                    logger.error(f"Failed to send close notification for {symbol}: {e}")


def handle_ws_close(symbol: str, reason: str, risk: RiskManager, ws_monitor):
    """Callback for WebSocket-triggered position closes.

    Don't close for SL/TP hits (broker bracket order handles those).
    Only close for reasons the broker doesn't know about (time stops, hard stops).
    """
    if symbol not in risk.open_trades:
        return

    trade = risk.open_trades[symbol]

    # Bracket order SL/TP handled by broker — defer to broker_sync
    if reason in ("stop_loss_ws", "take_profit_ws"):
        logger.info(f"WS: {symbol} {reason} detected — deferring to broker bracket order")
        return

    try:
        close_position(symbol, reason=reason)
    except Exception as e:
        logger.error(f"WS close failed for {symbol}: {e}")
        return

    exit_price = get_filled_exit_price(symbol, side=trade.side)
    if exit_price is None:
        try:
            snap = get_snapshot(symbol)
            exit_price = float(snap.latest_trade.price) if snap and snap.latest_trade else trade.entry_price
        except Exception:
            exit_price = trade.entry_price

    risk.close_trade(symbol, exit_price, now_et(), exit_reason=reason)
    ws_monitor.unsubscribe(symbol)

    notif = _get_notifications()
    if notif and config.TELEGRAM_ENABLED:
        try:
            notif.notify_trade_closed(trade)
        except Exception as e:
            logger.error(f"Failed to send close notification for {symbol}: {e}")


def get_current_prices(open_trades: dict) -> dict[str, float]:
    """Fetch current prices for open trades (for beta calculation etc.)."""
    symbols = list(open_trades.keys())
    if not symbols:
        return {}
    prices: dict[str, float] = {}
    try:
        snapshots = get_snapshots(symbols)
        for sym, snap in snapshots.items():
            if snap and snap.latest_trade:
                prices[sym] = float(snap.latest_trade.price)
    except Exception as e:
        logger.warning(f"Price fetch failed: {e}")
    for sym, trade in open_trades.items():
        if sym not in prices:
            prices[sym] = trade.entry_price
    return prices
