"""V10 OMS — Emergency kill switch: cancel all orders + close all positions."""

import json
import logging
import os
import time as _time
from datetime import datetime
from pathlib import Path

import config

from engine.event_log import log_event, EventType

logger = logging.getLogger(__name__)

# V12 11.1: Persistent queue file for crash recovery during batch close
_QUEUE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "kill_switch_queue.json")

# MED-031: Batch size and delay for position closes to avoid API rate limits
KILL_SWITCH_BATCH_SIZE = getattr(config, "KILL_SWITCH_BATCH_SIZE", 5)
KILL_SWITCH_BATCH_DELAY_SEC = getattr(config, "KILL_SWITCH_BATCH_DELAY_SEC", 0.5)


class KillSwitch:
    """Emergency halt: cancel all orders, close all positions, disable new trading.

    Can be activated via:
    - API endpoint (web dashboard)
    - CLI command
    - Auto-trigger on extreme drawdown (configurable)
    """

    def __init__(self):
        self.active = False
        self.activated_at: datetime | None = None
        self.reason: str = ""
        # V12 11.1: Process any leftover queue from a previous crash
        self._process_residual_queue()

    # ------------------------------------------------------------------
    # V12 11.1: Persistent queue helpers — survive crash mid-close
    # ------------------------------------------------------------------

    @staticmethod
    def _write_queue(symbols: list[str], reason: str) -> None:
        """Persist the list of symbols still pending close to disk."""
        try:
            Path(_QUEUE_FILE).parent.mkdir(parents=True, exist_ok=True)
            with open(_QUEUE_FILE, "w") as f:
                json.dump({"symbols": symbols, "reason": reason,
                           "ts": datetime.now(config.ET).isoformat()}, f)
            logger.debug(f"Kill-switch queue written: {len(symbols)} symbols")
        except Exception as e:
            logger.error(f"Failed to write kill-switch queue: {e}")

    @staticmethod
    def _remove_from_queue(symbol: str) -> None:
        """Remove a single symbol from the persistent queue after confirmed close."""
        try:
            if not os.path.exists(_QUEUE_FILE):
                return
            with open(_QUEUE_FILE, "r") as f:
                data = json.load(f)
            remaining = [s for s in data.get("symbols", []) if s != symbol]
            if remaining:
                data["symbols"] = remaining
                with open(_QUEUE_FILE, "w") as f:
                    json.dump(data, f)
            else:
                os.remove(_QUEUE_FILE)
                logger.info("Kill-switch queue cleared (all positions closed)")
        except Exception as e:
            logger.error(f"Failed to update kill-switch queue: {e}")

    @staticmethod
    def _load_queue() -> tuple[list[str], str]:
        """Load pending close queue from disk. Returns (symbols, reason)."""
        try:
            if os.path.exists(_QUEUE_FILE):
                with open(_QUEUE_FILE, "r") as f:
                    data = json.load(f)
                return data.get("symbols", []), data.get("reason", "crash_recovery")
        except Exception as e:
            logger.error(f"Failed to load kill-switch queue: {e}")
        return [], ""

    def _process_residual_queue(self) -> None:
        """On startup, close any positions left in the queue from a previous crash."""
        symbols, reason = self._load_queue()
        if not symbols:
            return
        logger.warning(
            f"V12 11.1: Residual kill-switch queue found ({len(symbols)} symbols "
            f"from reason={reason}). Attempting to close..."
        )
        from execution import close_position
        for symbol in list(symbols):
            try:
                close_position(symbol, reason=f"kill_switch_recovery_{reason}")
                self._remove_from_queue(symbol)
                logger.info(f"Kill-switch recovery: closed {symbol}")
            except Exception as e:
                logger.error(f"Kill-switch recovery: failed to close {symbol}: {e}")

    def activate(self, reason: str = "manual", risk_manager=None, order_manager=None):
        """Activate kill switch: cancel all orders and close all positions.

        Args:
            reason: Why the kill switch was activated
            risk_manager: RiskManager instance for position data
            order_manager: OrderManager instance for order cancellation
        """
        if self.active:
            logger.warning("Kill switch already active")
            return

        self.active = True
        self.activated_at = datetime.now(config.ET)
        self.reason = reason
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        log_event(EventType.KILL_SWITCH, "kill_switch",
                  details=f"reason={reason}", severity="CRITICAL")

        # 1. Cancel all pending/active orders
        if order_manager:
            cancelled = order_manager.cancel_all()
            logger.info(f"Kill switch: cancelled {len(cancelled)} orders")

        # 2. Close all positions via broker (MED-031: batch to avoid API rate limits)
        failed_closes = []
        if risk_manager:
            from execution import close_position
            symbols = list(risk_manager.open_trades.keys())

            # V12 11.1: Write full queue to disk before starting batch close
            self._write_queue(symbols, reason)

            for batch_start in range(0, len(symbols), KILL_SWITCH_BATCH_SIZE):
                batch = symbols[batch_start:batch_start + KILL_SWITCH_BATCH_SIZE]
                for symbol in batch:
                    try:
                        close_position(symbol, reason="kill_switch")
                        trade = risk_manager.open_trades.get(symbol)
                        if trade:
                            risk_manager.close_trade(
                                symbol, trade.entry_price,
                                datetime.now(config.ET),
                                exit_reason="kill_switch",
                            )
                        # V12 11.1: Remove from persistent queue after confirmed close
                        self._remove_from_queue(symbol)
                        logger.info(f"Kill switch: closed {symbol}")
                    except Exception as e:
                        failed_closes.append(symbol)
                        logger.error(f"Kill switch: failed to close {symbol}: {e}")
                # Sleep between batches to avoid API rate limits (skip after last batch)
                if batch_start + KILL_SWITCH_BATCH_SIZE < len(symbols):
                    _time.sleep(KILL_SWITCH_BATCH_DELAY_SEC)

            if failed_closes:
                logger.critical(
                    f"KILL SWITCH: {len(failed_closes)} positions FAILED to close: "
                    f"{failed_closes}. MANUAL INTERVENTION REQUIRED."
                )

        # 3. Send notification
        try:
            import notifications
            if config.TELEGRAM_ENABLED:
                notifications.send_alert(
                    f"KILL SWITCH ACTIVATED: {reason}\n"
                    f"All positions closed, all orders cancelled."
                )
        except Exception:
            pass

    def deactivate(self):
        """Deactivate kill switch, allowing trading to resume."""
        if not self.active:
            return
        self.active = False
        logger.info(f"Kill switch deactivated (was active since {self.activated_at})")
        self.activated_at = None
        self.reason = ""

    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed (kill switch not active)."""
        return not self.active

    @property
    def status(self) -> dict:
        return {
            "active": self.active,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "reason": self.reason,
        }
