"""RISK-005 + RISK-006: Enhanced intraday risk controls with multi-timeframe
rolling P&L limits, per-window pause durations, and velocity checks.

Monitors P&L on multiple rolling time windows and enforces progressive
throttling when losses cluster in short periods. Designed to catch
runaway strategies or adverse market microstructure before the daily
circuit breaker triggers.

Rolling windows (RISK-006: each window pauses trading for its own duration):
    - 5 min:  max -0.8% portfolio loss  -> pause for 5 min
    - 30 min: max -1.2% portfolio loss  -> pause for 30 min
    - 1 hour: max -1.8% portfolio loss  -> pause for 1 hour

Velocity controls:
    - 3+ stops hit in 15 min  -> pause all entries for 30 min
    - 4:1 loss/win ratio in 1 hr -> reduce sizing to 50%
"""

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import config

logger = logging.getLogger(__name__)


class ControlState(Enum):
    NORMAL = "normal"
    THROTTLED = "throttled"     # Reduced sizing
    PAUSED = "paused"           # No new entries allowed
    HALTED = "halted"           # Rolling window limit breached


@dataclass
class RiskControlState:
    """Snapshot of current intraday risk control status."""
    state: ControlState = ControlState.NORMAL
    size_multiplier: float = 1.0
    reason: str = ""
    pause_until: datetime | None = None
    rolling_5m_pnl: float = 0.0
    rolling_30m_pnl: float = 0.0
    rolling_1h_pnl: float = 0.0
    stops_last_15m: int = 0
    loss_win_ratio_1h: float = 0.0


@dataclass
class PnLTick:
    """Single P&L observation with timestamp."""
    timestamp: datetime
    pnl_pct: float            # Incremental P&L as fraction of equity
    is_stop_loss: bool = False  # Whether this was a stop-loss exit
    is_loss: bool = False       # Whether trade was a loss
    is_win: bool = False        # Whether trade was a win


# Rolling window limits
WINDOW_LIMITS = {
    timedelta(minutes=5): -0.008,    # V12 AUDIT: Widened from -0.3% to -0.8%
    timedelta(minutes=30): -0.012,   # V12 AUDIT: Widened from -0.5% to -1.2%
    timedelta(hours=1): -0.018,      # V12 AUDIT: Widened from -0.8% to -1.8%
}

# Velocity thresholds
STOPS_WINDOW = timedelta(minutes=15)
STOPS_THRESHOLD = 3
STOPS_PAUSE_DURATION = timedelta(minutes=30)

LOSS_RATIO_WINDOW = timedelta(hours=1)
LOSS_RATIO_THRESHOLD = 4.0
LOSS_RATIO_SIZE_MULT = 0.5


class IntradayRiskControls:
    """Real-time intraday P&L and velocity monitoring.

    Feed P&L ticks as trades close. The controller maintains rolling
    windows and triggers throttling/pausing when limits are breached.

    Thread-safe: all state mutations are protected by a lock.
    """

    def __init__(
        self,
        window_limits: dict[timedelta, float] | None = None,
        stops_threshold: int = STOPS_THRESHOLD,
        stops_pause_duration: timedelta = STOPS_PAUSE_DURATION,
        loss_ratio_threshold: float = LOSS_RATIO_THRESHOLD,
    ):
        self._window_limits = window_limits or dict(WINDOW_LIMITS)
        self._stops_threshold = stops_threshold
        self._stops_pause_duration = stops_pause_duration
        self._loss_ratio_threshold = loss_ratio_threshold

        # Rolling tick buffer (keeps up to 2 hours of ticks)
        self._ticks: deque[PnLTick] = deque(maxlen=5000)

        # Current state
        self._state = ControlState.NORMAL
        self._pause_until: datetime | None = None
        self._pause_reason: str = ""           # RISK-006: reason for current pause
        self._last_check: datetime | None = None
        self._lock = threading.Lock()

        # RISK-006: Cached rolling P&Ls for reporting during pause
        self._cached_rolling_5m: float = 0.0
        self._cached_rolling_30m: float = 0.0
        self._cached_rolling_1h: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_pnl(
        self,
        pnl_pct: float,
        is_stop_loss: bool = False,
        is_loss: bool = False,
        is_win: bool = False,
        now: datetime | None = None,
    ):
        """Record a P&L tick (typically when a trade closes or partial exit).

        Args:
            pnl_pct: P&L as fraction of portfolio equity (e.g., -0.002 = -0.2%)
            is_stop_loss: Whether exit was triggered by a stop loss
            is_loss: Whether the trade was a loss
            is_win: Whether the trade was a win
            now: Timestamp (defaults to now)
        """
        if now is None:
            now = datetime.now(config.ET)

        tick = PnLTick(
            timestamp=now,
            pnl_pct=pnl_pct,
            is_stop_loss=is_stop_loss,
            is_loss=is_loss,
            is_win=is_win,
        )

        with self._lock:
            self._ticks.append(tick)

    def record_stop_loss(self, now: datetime | None = None) -> None:
        """Record that a stop-loss was triggered.

        Convenience wrapper around record_pnl for stop-loss events.
        Records a zero P&L tick with the stop-loss flag set, which
        feeds into the velocity check (3+ stops in 15 min -> pause).

        Args:
            now: Timestamp (defaults to now).
        """
        self.record_pnl(
            pnl_pct=0.0,
            is_stop_loss=True,
            is_loss=True,
            now=now,
        )
        logger.debug("Stop-loss recorded at %s", now or "now")

    def get_sizing_multiplier(self, now: datetime | None = None) -> float:
        """Get the current position size multiplier (0.0, 0.5, or 1.0).

        Alias for get_size_multiplier, matching the spec interface name.
        """
        return self.get_size_multiplier(now)

    def check_controls(self, now: datetime | None = None) -> RiskControlState:
        """Evaluate all intraday controls and return current state.

        Call this before entering a new trade or periodically (e.g., every scan).

        Returns:
            RiskControlState with current throttle/pause status
        """
        if now is None:
            now = datetime.now(config.ET)

        with self._lock:
            return self._evaluate(now)

    def should_allow_trade(self, now: datetime | None = None) -> tuple[bool, str]:
        """Quick check: should a new trade be allowed right now?

        Returns:
            (allowed: bool, reason: str)
        """
        state = self.check_controls(now)

        if state.state == ControlState.HALTED:
            return False, f"Rolling P&L limit breached: {state.reason}"

        if state.state == ControlState.PAUSED:
            if state.pause_until:
                remaining = (state.pause_until - (now or datetime.now(config.ET))).total_seconds()
                return False, f"Velocity pause: {state.reason} ({remaining:.0f}s remaining)"
            return False, f"Paused: {state.reason}"

        if state.state == ControlState.THROTTLED:
            # Allow trade but at reduced size
            return True, f"Throttled ({state.size_multiplier:.0%} sizing): {state.reason}"

        return True, ""

    def get_size_multiplier(self, now: datetime | None = None) -> float:
        """Get the current position size multiplier (1.0 = normal)."""
        state = self.check_controls(now)
        return state.size_multiplier

    def reset_daily(self):
        """Reset all state for a new trading day."""
        with self._lock:
            self._ticks.clear()
            self._state = ControlState.NORMAL
            self._pause_until = None
            self._pause_reason = ""
            self._last_check = None
            self._cached_rolling_5m = 0.0
            self._cached_rolling_30m = 0.0
            self._cached_rolling_1h = 0.0
        logger.info("Intraday risk controls reset for new day")

    # ------------------------------------------------------------------
    # Internal evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, now: datetime) -> RiskControlState:
        """Evaluate all controls (must hold lock)."""
        self._last_check = now

        # Prune old ticks (keep last 2 hours)
        cutoff = now - timedelta(hours=2)
        while self._ticks and self._ticks[0].timestamp < cutoff:
            self._ticks.popleft()

        # RISK-006: Check if a per-window pause is still active
        if self._pause_until and now < self._pause_until:
            remaining = (self._pause_until - now).total_seconds()
            reason = self._pause_reason or "rolling_window_breach"
            return RiskControlState(
                state=ControlState.HALTED,
                size_multiplier=0.0,
                reason=f"{reason} (pause {remaining:.0f}s remaining)",
                pause_until=self._pause_until,
                rolling_5m_pnl=self._cached_rolling_5m,
                rolling_30m_pnl=self._cached_rolling_30m,
                rolling_1h_pnl=self._cached_rolling_1h,
            )
        elif self._pause_until and now >= self._pause_until:
            self._pause_until = None
            self._pause_reason = ""

        # 1. Check rolling P&L windows
        rolling_pnls = {}
        for window, limit in self._window_limits.items():
            window_start = now - window
            window_pnl = sum(
                t.pnl_pct for t in self._ticks if t.timestamp >= window_start
            )
            rolling_pnls[window] = window_pnl

            if window_pnl < limit:
                # RISK-006: Pause for the breached window's duration
                self._pause_until = now + window
                window_min = int(window.total_seconds() / 60)
                reason = (
                    f"{window_min}min P&L {window_pnl:.3%} "
                    f"< limit {limit:.3%}"
                )
                self._pause_reason = reason
                logger.warning(
                    f"RISK-006: Intraday control HALT: {reason} — "
                    f"pausing for {window_min}min until {self._pause_until.strftime('%H:%M:%S')}"
                )
                self._state = ControlState.HALTED

                # Cache rolling P&Ls for pause state
                self._cached_rolling_5m = rolling_pnls.get(timedelta(minutes=5), 0.0)
                self._cached_rolling_30m = rolling_pnls.get(timedelta(minutes=30), 0.0)
                self._cached_rolling_1h = rolling_pnls.get(timedelta(hours=1), 0.0)

                return RiskControlState(
                    state=ControlState.HALTED,
                    size_multiplier=0.0,
                    reason=reason,
                    pause_until=self._pause_until,
                    rolling_5m_pnl=rolling_pnls.get(timedelta(minutes=5), 0.0),
                    rolling_30m_pnl=rolling_pnls.get(timedelta(minutes=30), 0.0),
                    rolling_1h_pnl=rolling_pnls.get(timedelta(hours=1), 0.0),
                )

        # 2. Check stop-loss velocity
        stops_window_start = now - STOPS_WINDOW
        stops_count = sum(
            1 for t in self._ticks
            if t.timestamp >= stops_window_start and t.is_stop_loss
        )

        if stops_count >= self._stops_threshold:
            # Set or extend pause
            if self._pause_until is None or self._pause_until < now:
                self._pause_until = now + self._stops_pause_duration
                logger.warning(
                    f"Intraday PAUSE: {stops_count} stops in 15min. "
                    f"Paused until {self._pause_until.strftime('%H:%M:%S')}"
                )

        # Check if we're still in a pause
        if self._pause_until and now < self._pause_until:
            self._state = ControlState.PAUSED
            return RiskControlState(
                state=ControlState.PAUSED,
                size_multiplier=0.0,
                reason=f"{stops_count} stops in 15min",
                pause_until=self._pause_until,
                rolling_5m_pnl=rolling_pnls.get(timedelta(minutes=5), 0.0),
                rolling_30m_pnl=rolling_pnls.get(timedelta(minutes=30), 0.0),
                rolling_1h_pnl=rolling_pnls.get(timedelta(hours=1), 0.0),
                stops_last_15m=stops_count,
            )
        elif self._pause_until and now >= self._pause_until:
            self._pause_until = None  # Pause expired

        # 3. Check loss/win ratio velocity
        ratio_window_start = now - LOSS_RATIO_WINDOW
        recent_losses = sum(
            1 for t in self._ticks
            if t.timestamp >= ratio_window_start and t.is_loss
        )
        recent_wins = sum(
            1 for t in self._ticks
            if t.timestamp >= ratio_window_start and t.is_win
        )

        loss_ratio = (
            recent_losses / max(recent_wins, 1)
            if (recent_losses + recent_wins) >= 3
            else 0.0
        )

        if loss_ratio >= self._loss_ratio_threshold:
            self._state = ControlState.THROTTLED
            reason = f"Loss/win ratio {loss_ratio:.1f}:1 in 1hr"
            logger.info(f"Intraday THROTTLE: {reason} -> {LOSS_RATIO_SIZE_MULT:.0%} sizing")
            return RiskControlState(
                state=ControlState.THROTTLED,
                size_multiplier=LOSS_RATIO_SIZE_MULT,
                reason=reason,
                rolling_5m_pnl=rolling_pnls.get(timedelta(minutes=5), 0.0),
                rolling_30m_pnl=rolling_pnls.get(timedelta(minutes=30), 0.0),
                rolling_1h_pnl=rolling_pnls.get(timedelta(hours=1), 0.0),
                stops_last_15m=stops_count,
                loss_win_ratio_1h=loss_ratio,
            )

        # All clear
        self._state = ControlState.NORMAL
        return RiskControlState(
            state=ControlState.NORMAL,
            size_multiplier=1.0,
            rolling_5m_pnl=rolling_pnls.get(timedelta(minutes=5), 0.0),
            rolling_30m_pnl=rolling_pnls.get(timedelta(minutes=30), 0.0),
            rolling_1h_pnl=rolling_pnls.get(timedelta(hours=1), 0.0),
            stops_last_15m=stops_count,
            loss_win_ratio_1h=loss_ratio,
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def status(self) -> dict:
        with self._lock:
            now = datetime.now(config.ET)
            return {
                "state": self._state.value,
                "ticks_buffered": len(self._ticks),
                "pause_until": (
                    self._pause_until.isoformat() if self._pause_until else None
                ),
                "last_check": (
                    self._last_check.isoformat() if self._last_check else None
                ),
                "window_limits": {
                    f"{int(k.total_seconds() / 60)}min": f"{v:.3%}"
                    for k, v in self._window_limits.items()
                },
            }
