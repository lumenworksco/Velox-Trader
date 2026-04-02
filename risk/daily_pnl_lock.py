"""Daily P&L Lock — the key mechanism for consistent returns.

Gradual gain lock:
- At +1.5% daily P&L, reduce sizing to 70% (not 30%).
Gradual loss reduction:
- At -0.5% daily P&L, reduce sizing to 60%.
- At -1.0% daily P&L, halt all new trades.

This creates an asymmetric daily P&L distribution:
- Good days: capped at ~+1.5-2.0%
- Bad days: capped at ~-1.0%
- Result: consistently positive over time.
"""

import logging
import threading
from enum import Enum

import config

logger = logging.getLogger(__name__)


class LockState(str, Enum):
    NORMAL = "NORMAL"
    LOSS_CAUTION = "LOSS_CAUTION"  # Intermediate: -0.5% daily
    GAIN_LOCK = "GAIN_LOCK"
    LOSS_HALT = "LOSS_HALT"


class DailyPnLLock:
    """Tracks daily P&L and restricts trading based on thresholds."""

    def __init__(self):
        self._lock = threading.Lock()
        self.state = LockState.NORMAL
        self._day_pnl_pct = 0.0

    def update(self, day_pnl_pct: float) -> LockState:
        """Update with current day P&L percentage.

        Args:
            day_pnl_pct: Current day P&L as decimal (e.g., 0.015 = +1.5%)

        Returns:
            Current lock state
        """
        with self._lock:
            self._day_pnl_pct = day_pnl_pct

            # Loss thresholds (checked most severe first)
            if day_pnl_pct <= config.PNL_LOSS_HALT_PCT:
                if self.state != LockState.LOSS_HALT:
                    logger.warning(
                        f"PNL LOCK: LOSS HALT activated at {day_pnl_pct:.2%} "
                        f"(threshold: {config.PNL_LOSS_HALT_PCT:.2%})"
                    )
                self.state = LockState.LOSS_HALT
            elif day_pnl_pct <= -0.005:
                if self.state != LockState.LOSS_CAUTION:
                    logger.info(
                        f"PNL LOCK: LOSS CAUTION activated at {day_pnl_pct:.2%} "
                        f"(threshold: -0.50%)"
                    )
                self.state = LockState.LOSS_CAUTION
            elif day_pnl_pct >= config.PNL_GAIN_LOCK_PCT:
                if self.state != LockState.GAIN_LOCK:
                    logger.info(
                        f"PNL LOCK: GAIN LOCK activated at {day_pnl_pct:.2%} "
                        f"(threshold: {config.PNL_GAIN_LOCK_PCT:.2%})"
                    )
                self.state = LockState.GAIN_LOCK
            else:
                self.state = LockState.NORMAL

            return self.state

    def get_size_multiplier(self) -> float:
        """Get position size multiplier based on current lock state.

        Returns:
            1.0 for NORMAL, 0.7 for GAIN_LOCK, 0.6 for LOSS_CAUTION, 0.0 for LOSS_HALT
        """
        with self._lock:
            if self.state == LockState.LOSS_HALT:
                return 0.0
            elif self.state == LockState.LOSS_CAUTION:
                return 0.6
            elif self.state == LockState.GAIN_LOCK:
                return 0.7
            return 1.0

    def is_trading_allowed(self) -> bool:
        """Whether new trades are allowed."""
        with self._lock:
            return self.state != LockState.LOSS_HALT

    def reset_daily(self):
        """Reset state for new trading day."""
        with self._lock:
            self.state = LockState.NORMAL
            self._day_pnl_pct = 0.0

    @property
    def day_pnl_pct(self) -> float:
        with self._lock:
            return self._day_pnl_pct
