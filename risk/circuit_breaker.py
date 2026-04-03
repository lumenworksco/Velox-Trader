"""V10 Risk — Tiered intraday circuit breaker.

Replaces the single-threshold circuit breaker with 4 progressive tiers:
- Tier 1 (Yellow): Reduce new position sizes by 50%
- Tier 2 (Orange): Stop all new entries, manage existing only
- Tier 3 (Red):    Close all day-trade positions
- Tier 4 (Black):  Kill switch — close everything

Each tier is configurable via config constants.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum

import config

logger = logging.getLogger(__name__)

# WIRE-014: Drawdown risk (CDaR) as additional circuit breaker trigger (fail-open)
_drawdown_mgr = None
try:
    from ops.drawdown_risk import DrawdownRiskManager as _DRM
    _drawdown_mgr = _DRM()
except ImportError:
    _DRM = None


class CircuitTier(IntEnum):
    NORMAL = 0
    YELLOW = 1   # Reduce sizing
    ORANGE = 2   # Stop new entries
    RED = 3      # Close day trades
    BLACK = 4    # Kill switch


@dataclass
class TierConfig:
    threshold_pct: float     # Negative P&L % that triggers this tier
    size_multiplier: float   # Position size multiplier (1.0 = normal, 0 = blocked)
    allow_new_entries: bool  # Whether new trades are allowed
    close_day_trades: bool   # Whether to close all day-hold positions
    close_all: bool          # Whether to close ALL positions


# Default tier thresholds (negative values = loss)
DEFAULT_TIERS = {
    CircuitTier.NORMAL: TierConfig(0.0, 1.0, True, False, False),
    CircuitTier.YELLOW: TierConfig(-0.01, 0.5, True, False, False),    # -1%
    CircuitTier.ORANGE: TierConfig(-0.02, 0.0, False, False, False),   # -2%
    CircuitTier.RED: TierConfig(-0.03, 0.0, False, True, False),       # -3%
    CircuitTier.BLACK: TierConfig(-0.04, 0.0, False, False, True),     # -4%
}


class TieredCircuitBreaker:
    """Progressive circuit breaker with 4 severity tiers."""

    # Hysteresis buffer: P&L must recover by this much beyond the threshold
    # before de-escalating, to prevent rapid oscillation.
    HYSTERESIS_PCT = 0.002  # 0.2%

    def __init__(self, tiers: dict[CircuitTier, TierConfig] | None = None):
        self.tiers = tiers or DEFAULT_TIERS
        # MED-011: Validate thresholds are monotonically decreasing
        # (more severe tiers must have lower/more-negative thresholds)
        ordered_tiers = [CircuitTier.YELLOW, CircuitTier.ORANGE, CircuitTier.RED, CircuitTier.BLACK]
        for i in range(len(ordered_tiers) - 1):
            t_curr = ordered_tiers[i]
            t_next = ordered_tiers[i + 1]
            if t_curr in self.tiers and t_next in self.tiers:
                if self.tiers[t_curr].threshold_pct <= self.tiers[t_next].threshold_pct:
                    logger.warning(
                        "MED-011: Circuit breaker thresholds not monotonic: "
                        "%s (%.3f%%) should be > %s (%.3f%%). Falling back to defaults.",
                        t_curr.name, self.tiers[t_curr].threshold_pct,
                        t_next.name, self.tiers[t_next].threshold_pct,
                    )
                    self.tiers = dict(DEFAULT_TIERS)
                    break
        self.current_tier = CircuitTier.NORMAL
        self.last_update: datetime | None = None
        self.tier_history: list[tuple[datetime, CircuitTier]] = []
        self._lock = threading.Lock()

    def update(self, day_pnl_pct: float, now: datetime | None = None) -> CircuitTier:
        """Evaluate current P&L and determine the appropriate tier (thread-safe).

        Args:
            day_pnl_pct: Today's P&L as a decimal (e.g., -0.02 = -2%)
            now: Current timestamp

        Returns:
            The current CircuitTier
        """
        with self._lock:
            return self._update_locked(day_pnl_pct, now)

    def _update_locked(self, day_pnl_pct: float, now: datetime | None = None) -> CircuitTier:
        if now is None:
            now = datetime.now(config.ET)

        old_tier = self.current_tier

        # Determine tier based on P&L — check from most severe (BLACK) down
        # to least severe (YELLOW), then fall through to NORMAL.
        # This explicit ordering avoids IntEnum sort-order issues.
        new_tier = CircuitTier.NORMAL
        for tier in (CircuitTier.BLACK, CircuitTier.RED, CircuitTier.ORANGE, CircuitTier.YELLOW):
            if tier not in self.tiers:
                continue
            cfg = self.tiers[tier]
            if day_pnl_pct <= cfg.threshold_pct:
                new_tier = tier
                break

        # WIRE-014: CDaR-based escalation (fail-open)
        # If drawdown risk is elevated, escalate tier by one level
        try:
            if _drawdown_mgr is not None:
                dd_mult = _drawdown_mgr.get_exposure_multiplier(abs(day_pnl_pct))
                if dd_mult <= 0.0 and new_tier < CircuitTier.RED:
                    new_tier = CircuitTier.RED
                    logger.warning("WIRE-014: CDaR escalated circuit breaker to RED (dd_mult=0)")
                elif dd_mult <= 0.5 and new_tier < CircuitTier.ORANGE:
                    new_tier = CircuitTier.ORANGE
                    logger.warning("WIRE-014: CDaR escalated circuit breaker to ORANGE (dd_mult=%.2f)", dd_mult)
        except Exception as _e:
            logger.debug("WIRE-014: Drawdown risk check failed (fail-open): %s", _e)

        # Apply hysteresis buffer for de-escalation to prevent rapid
        # oscillation when P&L hovers near a threshold boundary.
        # Escalation applies immediately; de-escalation requires P&L to
        # recover beyond the old tier's threshold by HYSTERESIS_PCT.
        if new_tier < old_tier:
            old_cfg = self.tiers[old_tier]
            # P&L must be above (old threshold + hysteresis) to de-escalate.
            # E.g. RED threshold is -3%; must recover above -2.8% to leave RED.
            if day_pnl_pct < old_cfg.threshold_pct + self.HYSTERESIS_PCT:
                new_tier = old_tier  # stay at current tier

        if new_tier != old_tier:
            self.current_tier = new_tier
            self.tier_history.append((now, new_tier))
            if new_tier > old_tier:
                logger.warning(
                    f"Circuit breaker ESCALATED: {old_tier.name} -> {new_tier.name} "
                    f"(day P&L: {day_pnl_pct:.2%})"
                )
            else:
                logger.info(
                    f"Circuit breaker de-escalated: {old_tier.name} -> {new_tier.name} "
                    f"(day P&L: {day_pnl_pct:.2%})"
                )

        self.last_update = now
        return self.current_tier

    @property
    def config(self) -> TierConfig:
        """Get the configuration for the current tier."""
        return self.tiers[self.current_tier]

    @property
    def size_multiplier(self) -> float:
        """Get the position size multiplier for the current tier."""
        return self.config.size_multiplier

    @property
    def allow_new_entries(self) -> bool:
        """Whether new entries are allowed at the current tier."""
        return self.config.allow_new_entries

    @property
    def should_close_day_trades(self) -> bool:
        """Whether day-trade positions should be closed."""
        return self.config.close_day_trades

    @property
    def should_close_all(self) -> bool:
        """Whether ALL positions should be closed (kill switch)."""
        return self.config.close_all

    def escalate_to(self, tier: CircuitTier, reason: str = "") -> CircuitTier:
        """Force-escalate to *at least* the given tier (thread-safe).

        If the breaker is already at an equal or higher tier, this is a no-op.
        De-escalation is never performed — use ``update()`` for that.

        Args:
            tier: Minimum tier to escalate to.
            reason: Human-readable reason for the escalation.

        Returns:
            The resulting current tier.
        """
        with self._lock:
            if tier <= self.current_tier:
                return self.current_tier
            old = self.current_tier
            now = datetime.now(config.ET)
            self.current_tier = tier
            self.tier_history.append((now, tier))
            self.last_update = now
            logger.warning(
                "Circuit breaker ESCALATED (external): %s -> %s (%s)",
                old.name, tier.name, reason,
            )
            return self.current_tier

    def reset_daily(self):
        """Reset at start of new trading day (thread-safe)."""
        with self._lock:
            self.current_tier = CircuitTier.NORMAL
            self.tier_history.clear()
        logger.info("Circuit breaker reset for new day")

    @property
    def status(self) -> dict:
        return {
            "tier": self.current_tier.name,
            "tier_value": int(self.current_tier),
            "size_multiplier": self.size_multiplier,
            "allow_new_entries": self.allow_new_entries,
            "should_close_day_trades": self.should_close_day_trades,
            "should_close_all": self.should_close_all,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }
