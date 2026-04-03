"""V12-1.1: Unified Exit Orchestrator — single entry point for ALL exit decisions.

Consolidates three competing exit systems:
- exit_manager.py (scaled TP, ATR trailing, RSI momentum, vol expansion)
- adaptive_exit_manager.py (VIX-aware z-score exits for mean reversion)
- engine/exit_processor.py (profit tiers, dead signal, scale-out loser)

The orchestrator runs exits in priority order for each position:
  1. Hard stop loss (always first)
  2. Strategy-specific exits (from individual strategy check_exits)
  3. Adaptive VIX exit (MR strategies only)
  4. Scaled take-profit / profit tier exits
  5. ATR trailing stop
  6. RSI momentum exit
  7. Volatility expansion exit
  8. Dead signal detection
  9. Scale-out loser
  10. Time stops

All exit execution flows through handle_exit_action() which emits events,
updates risk state, and manages notifications.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import config

logger = logging.getLogger(__name__)

# Event bus integration (fail-open)
try:
    from engine.events import get_event_bus, Event, EventTypes
    _EVENTS_AVAILABLE = True
except ImportError:
    _EVENTS_AVAILABLE = False

# Adaptive exit manager (fail-open)
try:
    from adaptive_exit_manager import AdaptiveExitManager
    from risk import get_vix_level
    _ADAPTIVE_AVAILABLE = True
except ImportError:
    _ADAPTIVE_AVAILABLE = False

# Day strategies eligible for dead-signal exit
_DAY_STRATEGIES = {"STAT_MR", "VWAP", "MICRO_MOM", "ORB"}

# Strategies that use adaptive VIX-aware exits
_ADAPTIVE_EXIT_STRATEGIES = {"STAT_MR", "VWAP"}


@dataclass
class ExitAction:
    """A single exit decision returned by the orchestrator."""
    symbol: str
    action: str          # "full" or "partial"
    qty: Optional[int] = None   # None = full position
    reason: str = ""
    priority: int = 0    # Lower = higher priority (for deduplication)


@dataclass
class PositionExitState:
    """Single source of truth for exit state of one position.

    Replaces the scattered state in _profit_tier_exits, _scaled_out,
    and trade.partial_exits / trade.partial_closed_qty.
    """
    symbol: str
    # Profit tier tracking (0, 1, or 2 tiers taken)
    profit_tiers_taken: int = 0
    # Whether scale-out-on-loser has fired
    scaled_out: bool = False
    # Scaled TP levels taken (from ExitManager logic)
    scaled_tp_levels: int = 0
    # Total partial quantity closed
    partial_closed_qty: int = 0
    # Trailing stop price (ratchets only in profitable direction)
    trailing_stop_price: float = 0.0
    # Highest/lowest price seen (for trailing stops)
    highest_price_seen: float = 0.0
    lowest_price_seen: float = 0.0


class ExitOrchestrator:
    """Single entry point for all exit decisions.

    Replaces direct usage of ExitManager, AdaptiveExitManager, and the
    exit_processor module-level functions (check_profit_tiers, _check_dead_signal,
    _check_scale_out_loser, check_advanced_exits).
    """

    def __init__(self):
        self._lock = threading.Lock()
        # Per-position exit state: symbol -> PositionExitState
        self._state: dict[str, PositionExitState] = {}
        # Adaptive exit manager instance
        self._adaptive_mgr = AdaptiveExitManager() if _ADAPTIVE_AVAILABLE else None
        # Lazy-loaded optional modules
        self._notifications = None
        self._intraday_controls = None

    # ------------------------------------------------------------------ #
    #  State management
    # ------------------------------------------------------------------ #

    def _get_state(self, symbol: str) -> PositionExitState:
        """Get or create exit state for a position."""
        if symbol not in self._state:
            self._state[symbol] = PositionExitState(symbol=symbol)
        return self._state[symbol]

    def cleanup_state(self, symbol: str) -> None:
        """Remove exit state when a position is fully closed."""
        with self._lock:
            self._state.pop(symbol, None)

    def set_intraday_controls(self, controls) -> None:
        """Set the shared IntradayRiskControls instance."""
        self._intraday_controls = controls

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #

    def check_exits(
        self,
        risk_manager,
        now: datetime,
        strategies: Optional[dict] = None,
        ws_monitor=None,
    ) -> list[ExitAction]:
        """Single entry point for ALL exit decisions.

        Args:
            risk_manager: RiskManager with open_trades.
            now: Current time (ET-aware datetime).
            strategies: dict of {name: strategy_instance} for strategy-specific exits.
            ws_monitor: WebSocket monitor for unsubscribing on close.

        Returns:
            List of ExitAction to execute. Caller should pass these to
            execute_exits() for actual order submission.
        """
        if not risk_manager.open_trades:
            return []

        actions: list[ExitAction] = []
        # Track which symbols already have a full-close action to avoid duplicates
        full_close_symbols: set[str] = set()

        # Fetch current prices for all positions in one batch
        prices = _get_current_prices(risk_manager.open_trades)

        # --- Phase 1: Strategy-specific exits (from strategy.check_exits) ---
        if strategies:
            for name, strategy in strategies.items():
                if strategy is None:
                    continue
                try:
                    exits = strategy.check_exits(risk_manager.open_trades, now)
                    for ex in (exits or []):
                        sym = ex.get("symbol", "")
                        if sym in full_close_symbols:
                            continue
                        act = ExitAction(
                            symbol=sym,
                            action=ex.get("action", "full"),
                            qty=ex.get("qty"),
                            reason=ex.get("reason", f"{name}_exit"),
                            priority=10,
                        )
                        actions.append(act)
                        if act.action == "full":
                            full_close_symbols.add(sym)
                except Exception as e:
                    logger.error(f"ExitOrchestrator: {name} check_exits failed: {e}")

        # --- Phase 2: Per-position orchestrated exits ---
        for symbol, trade in list(risk_manager.open_trades.items()):
            if symbol in full_close_symbols:
                continue

            current_price = prices.get(symbol)
            if current_price is None or current_price <= 0:
                continue
            if trade.entry_price <= 0:
                continue

            state = self._get_state(symbol)

            # Sync state from trade object (for backward compat during migration)
            self._sync_state_from_trade(state, trade, current_price)

            direction = 1 if trade.side == "buy" else -1
            unrealized_pct = (current_price - trade.entry_price) / trade.entry_price * direction

            # Priority order within per-position checks:

            # 2a. Adaptive VIX exit (MR strategies only)
            if (self._adaptive_mgr is not None
                    and config.ADAPTIVE_EXITS_ENABLED
                    and trade.strategy in _ADAPTIVE_EXIT_STRATEGIES):
                act = self._check_adaptive_exit(trade, current_price, state)
                if act:
                    actions.append(act)
                    if act.action == "full":
                        full_close_symbols.add(symbol)
                        continue

            # 2b. Profit-tier exits (+1.5%, +2.5%)
            act = self._check_profit_tiers(trade, current_price, unrealized_pct, state)
            if act:
                actions.append(act)

            # 2c. ATR trailing stop
            if config.ATR_TRAILING_ENABLED and trade.entry_atr > 0:
                act = self._check_atr_trailing(trade, current_price, state)
                if act:
                    actions.append(act)
                    if act.action == "full":
                        full_close_symbols.add(symbol)
                        continue

            # 2d. Scaled take-profit (from ExitManager)
            if config.SCALED_TP_ENABLED and trade.qty > 1:
                act = self._check_scaled_tp(trade, current_price, state)
                if act:
                    actions.append(act)

            # 2e. RSI momentum exit (only if profitable)
            if unrealized_pct > 0.005:
                act = self._check_rsi_exit(trade, current_price, now)
                if act:
                    actions.append(act)
                    if act.action == "full":
                        full_close_symbols.add(symbol)
                        continue

            # 2f. Volatility expansion exit (only if losing)
            if unrealized_pct < 0 and trade.entry_atr > 0:
                act = self._check_volatility_exit(trade, current_price, now)
                if act:
                    actions.append(act)
                    if act.action == "full":
                        full_close_symbols.add(symbol)
                        continue

            # 2g. Dead signal detection
            act = self._check_dead_signal(trade, current_price, unrealized_pct, now)
            if act:
                actions.append(act)
                if act.action == "full":
                    full_close_symbols.add(symbol)
                    continue

            # 2h. Scale-out loser
            act = self._check_scale_out_loser(trade, current_price, unrealized_pct, state)
            if act:
                actions.append(act)

        return actions

    # ------------------------------------------------------------------ #
    #  Exit execution
    # ------------------------------------------------------------------ #

    def execute_exits(
        self,
        exit_actions: list[ExitAction],
        risk_manager,
        now: datetime,
        ws_monitor=None,
    ) -> None:
        """Execute a list of ExitActions — submit orders, update risk state, emit events.

        This is the SINGLE place where exit orders are actually submitted.
        Replaces handle_strategy_exits from exit_processor.py.
        """
        from execution import close_position, close_partial_position
        from data import get_snapshot, get_filled_exit_price

        notif = self._get_notifications()

        for action in exit_actions:
            symbol = action.symbol
            if symbol not in risk_manager.open_trades:
                continue
            trade = risk_manager.open_trades[symbol]

            if action.action == "partial":
                partial_qty = action.qty or max(1, trade.qty // 2)
                try:
                    close_partial_position(symbol, partial_qty)
                    logger.info(
                        "ExitOrchestrator: partial exit %s: %d shares, reason=%s",
                        symbol, partial_qty, action.reason,
                    )
                    # Update risk manager
                    try:
                        risk_manager.partial_close(symbol, partial_qty,
                                                   _get_price_for_symbol(symbol, trade),
                                                   now, action.reason)
                    except Exception:
                        pass

                    # Register partial in OMS (fail-open)
                    self._register_oms_partial(symbol, trade, partial_qty,
                                               _get_price_for_symbol(symbol, trade), now, action.reason)

                    _emit_event(
                        EventTypes.POSITION_PARTIAL_CLOSE if _EVENTS_AVAILABLE else "position.partial_close",
                        {"symbol": symbol, "strategy": trade.strategy,
                         "qty": partial_qty, "reason": action.reason},
                    )
                except Exception as e:
                    logger.error(f"ExitOrchestrator: partial close failed for {symbol}: {e}")

            else:
                # Full close
                try:
                    close_position(symbol, reason=action.reason)
                except Exception as e:
                    logger.error(f"ExitOrchestrator: close failed for {symbol}: {e}")
                    continue

                exit_price = get_filled_exit_price(symbol, side=trade.side)
                if exit_price is None:
                    try:
                        snap = get_snapshot(symbol)
                        exit_price = float(snap.latest_trade.price) if snap and snap.latest_trade else trade.entry_price
                    except Exception:
                        exit_price = trade.entry_price

                pnl = (exit_price - trade.entry_price) * trade.qty * (1 if trade.side == "buy" else -1)
                risk_manager.close_trade(symbol, exit_price, now, exit_reason=action.reason)

                # Clean up orchestrator state
                self.cleanup_state(symbol)

                # Feed intraday risk controls
                if self._intraday_controls is not None:
                    try:
                        pnl_pct = pnl / max(risk_manager.current_equity, 1)
                        is_stop = "stop" in action.reason.lower()
                        self._intraday_controls.record_pnl(
                            pnl_pct, is_stop_loss=is_stop,
                            is_loss=(pnl < 0), is_win=(pnl > 0), now=now,
                        )
                    except Exception:
                        pass

                # Register cooldown on stop-loss
                if "stop" in action.reason.lower():
                    try:
                        from engine.signal_processor import register_stopout
                        register_stopout(symbol)
                    except Exception:
                        pass

                _emit_event(
                    EventTypes.POSITION_CLOSED if _EVENTS_AVAILABLE else "position.closed",
                    {"symbol": symbol, "strategy": trade.strategy, "side": trade.side,
                     "entry_price": trade.entry_price, "exit_price": exit_price,
                     "qty": trade.qty, "pnl": round(pnl, 2), "reason": action.reason},
                )

                if ws_monitor:
                    ws_monitor.unsubscribe(symbol)

                if notif and config.TELEGRAM_ENABLED:
                    try:
                        notif.notify_trade_closed(trade)
                    except Exception as e:
                        logger.error(f"ExitOrchestrator: notification failed for {symbol}: {e}")

    # ------------------------------------------------------------------ #
    #  WebSocket-triggered closes (delegated from exit_processor)
    # ------------------------------------------------------------------ #

    def handle_ws_close(self, symbol: str, reason: str, risk_manager, ws_monitor) -> None:
        """Handle WebSocket-triggered position close.

        Bracket SL/TP handled by broker — only close for reasons the broker
        does not know about (time stops, hard stops).
        """
        if symbol not in risk_manager.open_trades:
            return
        trade = risk_manager.open_trades[symbol]

        if reason in ("stop_loss_ws", "take_profit_ws"):
            logger.info("WS: %s %s detected — deferring to broker bracket order", symbol, reason)
            return

        from execution import close_position
        from data import get_snapshot, get_filled_exit_price

        try:
            close_position(symbol, reason=reason)
        except Exception as e:
            logger.error(f"ExitOrchestrator WS close failed for {symbol}: {e}")
            return

        exit_price = get_filled_exit_price(symbol, side=trade.side)
        if exit_price is None:
            try:
                snap = get_snapshot(symbol)
                exit_price = float(snap.latest_trade.price) if snap and snap.latest_trade else trade.entry_price
            except Exception:
                exit_price = trade.entry_price

        now = datetime.now(config.ET)
        risk_manager.close_trade(symbol, exit_price, now, exit_reason=reason)
        self.cleanup_state(symbol)
        ws_monitor.unsubscribe(symbol)

        notif = self._get_notifications()
        if notif and config.TELEGRAM_ENABLED:
            try:
                notif.notify_trade_closed(trade)
            except Exception as e:
                logger.error(f"ExitOrchestrator: notification failed for {symbol}: {e}")

    # ------------------------------------------------------------------ #
    #  Price extreme updates (from WebSocket quotes)
    # ------------------------------------------------------------------ #

    def update_price_extremes(self, risk_manager, quotes: dict) -> None:
        """Update highest/lowest prices from live quote data.

        Thread-safe: acquires risk_manager lock.
        """
        lock = getattr(risk_manager, '_lock', None)
        if lock is not None:
            lock.acquire()
        try:
            for symbol, price in quotes.items():
                if symbol in risk_manager.open_trades:
                    trade = risk_manager.open_trades[symbol]
                    if trade.side == "buy" and price > trade.highest_price_seen:
                        trade.highest_price_seen = price
                    elif trade.side == "sell":
                        if trade.lowest_price_seen == 0 or price < trade.lowest_price_seen:
                            trade.lowest_price_seen = price
        finally:
            if lock is not None:
                lock.release()

    # ------------------------------------------------------------------ #
    #  Individual exit checks
    # ------------------------------------------------------------------ #

    def _sync_state_from_trade(self, state: PositionExitState, trade, current_price: float) -> None:
        """Sync orchestrator state from trade object fields for backward compat."""
        # Sync partial exit count
        if trade.partial_exits > state.scaled_tp_levels:
            state.scaled_tp_levels = trade.partial_exits
        # Sync price extremes
        if trade.side == "buy":
            if current_price > trade.highest_price_seen:
                trade.highest_price_seen = current_price
            state.highest_price_seen = trade.highest_price_seen
        else:
            if trade.lowest_price_seen == 0 or current_price < trade.lowest_price_seen:
                trade.lowest_price_seen = current_price
            state.lowest_price_seen = trade.lowest_price_seen

    def _check_adaptive_exit(
        self, trade, current_price: float, state: PositionExitState,
    ) -> Optional[ExitAction]:
        """VIX-aware z-score exit for mean-reversion strategies."""
        try:
            vix = get_vix_level()
            ou_params = {
                'mu': trade.entry_price,
                'sigma': trade.entry_atr if trade.entry_atr > 0 else trade.entry_price * 0.02,
                'half_life': 24.0,
            }
            exit_reason, exit_type = self._adaptive_mgr.should_exit(
                position_side=trade.side,
                position_entry_time=trade.entry_time,
                current_price=current_price,
                ou_params=ou_params,
                vix=vix,
                partial_exits=trade.partial_exits,
            )
            if exit_type == 'full':
                return ExitAction(
                    symbol=trade.symbol,
                    action="full",
                    reason=f"adaptive_exit_{exit_reason}",
                    priority=20,
                )
            elif exit_type == 'partial':
                partial_qty = max(1, trade.qty // 2)
                return ExitAction(
                    symbol=trade.symbol,
                    action="partial",
                    qty=partial_qty,
                    reason=f"adaptive_exit_{exit_reason}",
                    priority=20,
                )
        except Exception as e:
            logger.debug("ExitOrchestrator: adaptive exit check failed for %s: %s",
                         trade.symbol, e)
        return None

    def _check_profit_tiers(
        self, trade, current_price: float, unrealized_pct: float,
        state: PositionExitState,
    ) -> Optional[ExitAction]:
        """Tiered profit-taking: partial exits at +1.5% and +2.5%."""
        if state.profit_tiers_taken >= 2:
            return None

        if state.profit_tiers_taken == 0 and unrealized_pct >= 0.015:
            partial_qty = max(1, int(trade.qty * 0.33))
            state.profit_tiers_taken = 1
            logger.info(
                "ExitOrchestrator: Profit tier 1 for %s — %.2f%% unrealized, %d of %d shares",
                trade.symbol, unrealized_pct * 100, partial_qty, trade.qty,
            )
            return ExitAction(
                symbol=trade.symbol, action="partial",
                qty=partial_qty, reason="profit_tier_1_1.5pct", priority=30,
            )

        if state.profit_tiers_taken == 1 and unrealized_pct >= 0.025:
            partial_qty = max(1, int(trade.qty * 0.33))
            state.profit_tiers_taken = 2
            logger.info(
                "ExitOrchestrator: Profit tier 2 for %s — %.2f%% unrealized, %d of %d shares",
                trade.symbol, unrealized_pct * 100, partial_qty, trade.qty,
            )
            return ExitAction(
                symbol=trade.symbol, action="partial",
                qty=partial_qty, reason="profit_tier_2_2.5pct", priority=30,
            )

        return None

    def _check_atr_trailing(
        self, trade, current_price: float, state: PositionExitState,
    ) -> Optional[ExitAction]:
        """ATR-based trailing stop for all strategies."""
        atr_mult = config.ATR_TRAIL_MULT.get(trade.strategy)
        if atr_mult is None:
            return None

        trail_distance = trade.entry_atr * atr_mult
        activation_distance = trade.entry_atr * config.ATR_TRAIL_ACTIVATION

        if trade.side == "buy":
            if current_price < trade.entry_price + activation_distance:
                return None
            atr_trail_stop = trade.highest_price_seen - trail_distance
            if atr_trail_stop > trade.stop_loss:
                trade.stop_loss = atr_trail_stop
            if current_price <= trade.stop_loss:
                return ExitAction(
                    symbol=trade.symbol, action="full",
                    qty=trade.qty, reason="atr_trailing_stop", priority=25,
                )

        elif trade.side == "sell":
            if current_price > trade.entry_price - activation_distance:
                return None
            if trade.lowest_price_seen <= 0:
                trade.lowest_price_seen = current_price
            atr_trail_stop = trade.lowest_price_seen + trail_distance
            if atr_trail_stop < trade.stop_loss:
                trade.stop_loss = atr_trail_stop
            if current_price >= trade.stop_loss:
                return ExitAction(
                    symbol=trade.symbol, action="full",
                    qty=trade.qty, reason="atr_trailing_stop", priority=25,
                )

        return None

    def _check_scaled_tp(
        self, trade, current_price: float, state: PositionExitState,
    ) -> Optional[ExitAction]:
        """Scaled take-profit: 33% at 1/3 target, 50% of remaining at 2/3 target."""
        MAX_LEVELS = 2
        if state.scaled_tp_levels >= MAX_LEVELS:
            return None

        entry = trade.entry_price
        target = trade.take_profit

        if trade.side == "buy":
            target_range = target - entry
            if target_range <= 0:
                return None

            if current_price >= entry + target_range * 0.33 and state.scaled_tp_levels == 0:
                qty_to_close = max(1, int(trade.qty * 0.33))
                state.scaled_tp_levels = 1
                state.partial_closed_qty += qty_to_close
                return ExitAction(
                    symbol=trade.symbol, action="partial",
                    qty=qty_to_close, reason="scaled_tp_1", priority=30,
                )

            if current_price >= entry + target_range * 0.66 and state.scaled_tp_levels == 1:
                remaining_qty = trade.qty - state.partial_closed_qty
                qty_to_close = max(1, int(remaining_qty * 0.50))
                if config.BREAKEVEN_STOP_ENABLED:
                    trade.stop_loss = entry * 1.001
                state.scaled_tp_levels = 2
                state.partial_closed_qty += qty_to_close
                return ExitAction(
                    symbol=trade.symbol, action="partial",
                    qty=qty_to_close, reason="scaled_tp_2", priority=30,
                )

        elif trade.side == "sell":
            target_range = entry - target
            if target_range <= 0:
                return None

            if current_price <= entry - target_range * 0.33 and state.scaled_tp_levels == 0:
                qty_to_close = max(1, int(trade.qty * 0.33))
                state.scaled_tp_levels = 1
                state.partial_closed_qty += qty_to_close
                return ExitAction(
                    symbol=trade.symbol, action="partial",
                    qty=qty_to_close, reason="scaled_tp_1", priority=30,
                )

            if current_price <= entry - target_range * 0.66 and state.scaled_tp_levels == 1:
                remaining_qty = trade.qty - state.partial_closed_qty
                qty_to_close = max(1, int(remaining_qty * 0.50))
                if config.BREAKEVEN_STOP_ENABLED:
                    trade.stop_loss = entry * 0.999
                state.scaled_tp_levels = 2
                state.partial_closed_qty += qty_to_close
                return ExitAction(
                    symbol=trade.symbol, action="partial",
                    qty=qty_to_close, reason="scaled_tp_2", priority=30,
                )

        return None

    def _check_rsi_exit(
        self, trade, current_price: float, now: datetime,
    ) -> Optional[ExitAction]:
        """Exit if RSI is extremely overbought/oversold and trade is profitable."""
        try:
            import pandas_ta as ta
            from data import get_intraday_bars
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            market_open = datetime(now.year, now.month, now.day, 9, 30, tzinfo=config.ET)
            bars = get_intraday_bars(trade.symbol, TimeFrame(1, TimeFrameUnit.Minute),
                                     start=market_open, end=now)
            if bars.empty or len(bars) < 20:
                return None

            rsi_series = ta.rsi(bars["close"], length=14)
            if rsi_series is None or rsi_series.empty:
                return None
            rsi = rsi_series.iloc[-1]

            if trade.side == "buy" and rsi > config.RSI_EXIT_THRESHOLD:
                return ExitAction(
                    symbol=trade.symbol, action="full",
                    qty=trade.qty, reason="rsi_overbought", priority=40,
                )
            if trade.side == "sell" and rsi < (100 - config.RSI_EXIT_THRESHOLD):
                return ExitAction(
                    symbol=trade.symbol, action="full",
                    qty=trade.qty, reason="rsi_oversold", priority=40,
                )
        except Exception as e:
            logger.debug("ExitOrchestrator: RSI exit check failed for %s: %s", trade.symbol, e)
        return None

    def _check_volatility_exit(
        self, trade, current_price: float, now: datetime,
    ) -> Optional[ExitAction]:
        """Exit if ATR has expanded significantly since entry."""
        if trade.entry_atr <= 0:
            return None
        try:
            import pandas_ta as ta
            from data import get_intraday_bars
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

            market_open = datetime(now.year, now.month, now.day, 9, 30, tzinfo=config.ET)
            bars = get_intraday_bars(trade.symbol, TimeFrame(1, TimeFrameUnit.Minute),
                                     start=market_open, end=now)
            if bars.empty or len(bars) < 20:
                return None

            atr_series = ta.atr(bars["high"], bars["low"], bars["close"], length=14)
            if atr_series is None or atr_series.empty:
                return None
            current_atr = atr_series.iloc[-1]

            if current_atr > trade.entry_atr * config.ATR_EXPANSION_MULT:
                return ExitAction(
                    symbol=trade.symbol, action="full",
                    qty=trade.qty, reason="volatility_expansion", priority=40,
                )
        except Exception as e:
            logger.debug("ExitOrchestrator: volatility exit check failed for %s: %s",
                         trade.symbol, e)
        return None

    def _check_dead_signal(
        self, trade, current_price: float, unrealized_pct: float, now: datetime,
    ) -> Optional[ExitAction]:
        """Exit positions with no conviction after 30 minutes (day strategies)."""
        if trade.strategy not in _DAY_STRATEGIES:
            return None

        hold_duration = now - trade.entry_time
        if hold_duration < timedelta(minutes=30):
            return None

        if -0.003 <= unrealized_pct <= 0.003:
            logger.info(
                "ExitOrchestrator: Dead signal %s (%s) — held %.1f min, unrealized %.2f%%",
                trade.symbol, trade.strategy,
                hold_duration.total_seconds() / 60, unrealized_pct * 100,
            )
            return ExitAction(
                symbol=trade.symbol, action="full",
                reason="dead_signal", priority=50,
            )
        return None

    def _check_scale_out_loser(
        self, trade, current_price: float, unrealized_pct: float,
        state: PositionExitState,
    ) -> Optional[ExitAction]:
        """Scale out 50% when unrealized loss exceeds -1.0%. Once per trade."""
        if state.scaled_out:
            return None

        if unrealized_pct <= -0.01:
            partial_qty = max(1, trade.qty // 2)
            state.scaled_out = True
            logger.info(
                "ExitOrchestrator: Scale-out loser %s — %.2f%% unrealized, %d of %d shares",
                trade.symbol, unrealized_pct * 100, partial_qty, trade.qty,
            )
            return ExitAction(
                symbol=trade.symbol, action="partial",
                qty=partial_qty, reason="scale_out_loser_1pct", priority=35,
            )
        return None

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _get_notifications(self):
        if self._notifications is None:
            try:
                import notifications as _n
                self._notifications = _n
            except ImportError:
                self._notifications = False
        return self._notifications if self._notifications else None

    def _register_oms_partial(self, symbol, trade, qty, price, now, reason):
        """Register partial close in OMS (fail-open)."""
        try:
            from engine.signal_processor import _order_manager, _OMS_AVAILABLE
            if _OMS_AVAILABLE and _order_manager:
                _order_manager.create_order(
                    symbol=symbol,
                    strategy=trade.strategy,
                    side="sell" if trade.side == "buy" else "buy",
                    order_type="market",
                    qty=qty,
                    limit_price=price,
                    idempotency_key=f"partial_{symbol}_{reason}_{now.strftime('%Y%m%d%H%M%S')}",
                )
        except Exception as e:
            logger.debug("OMS registration for partial close failed (non-critical): %s", e)


# ====================================================================== #
#  Module-level helpers
# ====================================================================== #

def _emit_event(event_type: str, data: dict, source: str = "exit_orchestrator"):
    if _EVENTS_AVAILABLE:
        try:
            bus = get_event_bus()
            bus.publish(Event(event_type, data, source=source))
        except Exception:
            pass


def _get_current_prices(open_trades: dict) -> dict[str, float]:
    """Fetch current prices — delegates to exit_processor's implementation."""
    try:
        from engine.exit_processor import get_current_prices
        return get_current_prices(open_trades)
    except ImportError:
        # Fallback: use entry prices
        return {sym: t.entry_price for sym, t in open_trades.items()}


def _get_price_for_symbol(symbol: str, trade) -> float:
    """Get current price for a single symbol (for partial close recording)."""
    try:
        from data import get_snapshot
        snap = get_snapshot(symbol)
        if snap and snap.latest_trade:
            return float(snap.latest_trade.price)
    except Exception:
        pass
    return trade.entry_price


# ====================================================================== #
#  Singleton accessor
# ====================================================================== #

_orchestrator: Optional[ExitOrchestrator] = None
_orchestrator_lock = threading.Lock()


def get_exit_orchestrator() -> ExitOrchestrator:
    """Get or create the singleton ExitOrchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        with _orchestrator_lock:
            if _orchestrator is None:
                _orchestrator = ExitOrchestrator()
    return _orchestrator
