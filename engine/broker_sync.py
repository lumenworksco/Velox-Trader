"""V10 Engine — Broker position synchronization and shadow trade management."""

import logging
import threading
from datetime import datetime, timedelta

import config
import database
from data import get_positions, get_account, get_snapshot, get_filled_exit_info
from risk import RiskManager, TradeRecord

logger = logging.getLogger(__name__)

# Track consecutive misses before closing (prevents false closes from transient API issues)
# V11.3 P8: Protected by lock for thread safety
_broker_miss_counts: dict[str, int] = {}
_broker_miss_lock = threading.Lock()

# Lazy-loaded optional modules
_notifications = None
_intraday_controls = None
_intraday_controls_checked = False


def set_intraday_controls(controls) -> None:
    """Set the shared IntradayRiskControls instance (called from main.py)."""
    global _intraday_controls
    _intraday_controls = controls


def _get_intraday_controls():
    return _intraday_controls


def _get_notifications():
    global _notifications
    if _notifications is None:
        try:
            import notifications as _n
            _notifications = _n
        except ImportError:
            _notifications = False
    return _notifications if _notifications else None


def sync_positions_with_broker(risk: RiskManager, now: datetime, ws_monitor=None):
    """Sync open trades with actual broker positions.

    - Require 2 consecutive misses before closing (transient API tolerance).
    - Re-adopts ANY broker position not in tracking (V10 BUG-027).
    - Uses snapshot price for exit (V10 BUG-008).
    """
    try:
        broker_positions = {p.symbol: p for p in get_positions()}
    except Exception as e:
        logger.error(f"Failed to fetch positions: {e}")
        return

    notif = _get_notifications()

    # Close DB records for positions the broker no longer has
    for symbol in list(risk.open_trades.keys()):
        if symbol not in broker_positions:
            with _broker_miss_lock:
                _broker_miss_counts[symbol] = _broker_miss_counts.get(symbol, 0) + 1
                miss_count = _broker_miss_counts[symbol]
            if miss_count < 2:
                logger.info(f"Position {symbol} missing from broker (miss {miss_count}/2) — will confirm next sync")
                continue

            trade = risk.open_trades[symbol]

            exit_price, broker_reason = get_filled_exit_info(symbol, side=trade.side)
            if exit_price is None:
                broker_reason = "broker_sync"
                # V11.3 T7: Improved fallback chain for exit price
                # 1. Try snapshot (real-time)
                exit_price_found = False
                try:
                    snap = get_snapshot(symbol)
                    if snap and snap.latest_trade:
                        exit_price = float(snap.latest_trade.price)
                        exit_price_found = True
                except Exception:
                    pass

                # 2. Try bar cache (last known close)
                if not exit_price_found:
                    try:
                        from data import get_intraday_bars
                        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
                        bars = get_intraday_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute),
                                                  start=now - timedelta(minutes=10), end=now)
                        if bars is not None and not bars.empty:
                            exit_price = float(bars["close"].iloc[-1])
                            exit_price_found = True
                            logger.info(f"V11.3: Using bar cache close for {symbol} exit: ${exit_price:.2f}")
                    except Exception:
                        pass

                # 3. Last resort: entry_price (log as CRITICAL for reconciliation)
                if not exit_price_found:
                    exit_price = trade.entry_price
                    logger.critical(
                        f"V11.3 T7: CRITICAL — using entry_price as exit for {symbol}. "
                        f"Real exit price unknown. Manual reconciliation needed."
                    )

            risk.close_trade(symbol, exit_price, now, exit_reason=broker_reason)
            logger.info(f"Position {symbol} confirmed gone from broker — {broker_reason} at ${exit_price:.2f} (entry ${trade.entry_price:.2f})")
            with _broker_miss_lock:
                _broker_miss_counts.pop(symbol, None)

            # BUG-022: Register cooldown when bracket stop fires at broker
            if broker_reason and "stop" in broker_reason.lower():
                try:
                    from engine.signal_processor import register_stopout
                    register_stopout(symbol)
                except Exception:
                    pass

            # V11.3 T2: Feed intraday risk controls with P&L data
            try:
                from risk.intraday_controls import IntradayRiskControls
                _irc = _get_intraday_controls()
                if _irc is not None:
                    pnl = (exit_price - trade.entry_price) * trade.qty * (1 if trade.side == "buy" else -1)
                    pnl_pct = pnl / max(risk.current_equity, 1)
                    is_stop = broker_reason and "stop" in broker_reason.lower()
                    _irc.record_pnl(pnl_pct, is_stop_loss=is_stop,
                                    is_loss=(pnl < 0), is_win=(pnl > 0), now=now)
            except Exception:
                pass

            if ws_monitor:
                ws_monitor.unsubscribe(symbol)
            if notif and config.TELEGRAM_ENABLED:
                try:
                    notif.notify_trade_closed(trade)
                except Exception as e:
                    logger.error(f"Failed to send close notification for {symbol}: {e}")
        else:
            with _broker_miss_lock:
                _broker_miss_counts.pop(symbol, None)

    # Re-adopt broker positions not in our tracking
    our_symbols = set(risk.open_trades.keys())
    for symbol in broker_positions:
        if symbol not in our_symbols and symbol not in getattr(config, "BROKER_SYNC_EXCLUDE_SYMBOLS", {"SPY"}):
            bp = broker_positions[symbol]
            qty = int(float(bp.qty))
            avg_price = float(bp.avg_entry_price)

            recent = database.get_recent_trades(days=1)
            recent_match = next(
                (t for t in recent if t["symbol"] == symbol),
                None,
            )

            side = "buy" if qty > 0 else "sell"
            original_strategy = recent_match["strategy"] if recent_match else "re-adopted"

            # Use ATR-based TP/SL for re-adopted positions (fall back to 2%)
            atr_val = avg_price * 0.02  # Default: 2% of price
            try:
                from data import get_intraday_bars
                from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
                bars = get_intraday_bars(symbol, TimeFrame(5, TimeFrameUnit.Minute),
                                         start=now - timedelta(hours=2), end=now)
                if bars is not None and len(bars) >= 10:
                    import pandas_ta as pta
                    atr = pta.atr(bars['high'], bars['low'], bars['close'], length=10)
                    if atr is not None and len(atr) > 0:
                        atr_val = float(atr.iloc[-1])
                    else:
                        # MED-027: Warn when ATR data is missing for re-adopted position
                        logger.warning(
                            f"ATR data unavailable for re-adopted position {symbol} — "
                            f"using default 2%% of price (${atr_val:.2f}) for TP/SL"
                        )
                else:
                    logger.warning(
                        f"Insufficient bars for ATR calculation on re-adopted position {symbol} "
                        f"(got {len(bars) if bars is not None else 0}, need 10) — using default 2%%"
                    )
            except Exception as e:
                logger.warning(
                    f"ATR calculation failed for re-adopted position {symbol}: {e} — using default 2%%"
                )

            tp_mult, sl_mult = 2.0, 1.5  # 2 ATR TP, 1.5 ATR SL
            if side == "buy":
                take_profit = avg_price + (atr_val * tp_mult)
                stop_loss = avg_price - (atr_val * sl_mult)
            else:
                take_profit = avg_price - (atr_val * tp_mult)
                stop_loss = avg_price + (atr_val * sl_mult)

            trade = TradeRecord(
                symbol=symbol,
                strategy=original_strategy,
                side=side,
                entry_price=avg_price,
                entry_time=now,
                qty=abs(qty),
                take_profit=round(take_profit, 2),
                stop_loss=round(stop_loss, 2),
                order_id="",
                hold_type="day",
            )
            risk.open_trades[symbol] = trade
            logger.warning(
                f"Re-adopted broker position: {symbol} qty={qty} @ ${avg_price:.2f} "
                f"(strategy={original_strategy})"
            )
            if ws_monitor:
                ws_monitor.subscribe(symbol)

    try:
        account = get_account()
        risk.update_equity(float(account.equity), float(account.cash))
    except Exception as e:
        logger.error(f"Failed to update account: {e}")


def check_shadow_exits(now: datetime):
    """Check shadow trades for TP/SL/time_stop hits and close them."""
    try:
        open_shadows = database.get_open_shadow_trades()
    except Exception as e:
        logger.warning(f"Failed to fetch shadow trades: {e}")
        return

    for shadow in open_shadows:
        try:
            snap = get_snapshot(shadow["symbol"])
            if not snap or not snap.latest_trade:
                continue
            price = float(snap.latest_trade.price)
            side = shadow["side"]
            tp = shadow["take_profit"]
            sl = shadow["stop_loss"]
            exit_reason = None

            if side == "buy":
                if price >= tp:
                    exit_reason = "take_profit"
                elif price <= sl:
                    exit_reason = "stop_loss"
            else:
                if price <= tp:
                    exit_reason = "take_profit"
                elif price >= sl:
                    exit_reason = "stop_loss"

            if exit_reason:
                database.close_shadow_trade(
                    shadow["id"], price, now.isoformat(), exit_reason
                )
                logger.info(
                    f"[SHADOW] Closed {shadow['symbol']} ({shadow['strategy']}) "
                    f"@ {price:.2f} reason={exit_reason}"
                )
        except Exception as e:
            logger.warning(f"Shadow exit check failed for {shadow.get('symbol', '?')}: {e}")
