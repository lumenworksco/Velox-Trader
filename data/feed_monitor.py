"""V12 Item 2.3 — Alpaca Outage Detection + Backup Stops.

DataFeedMonitor tracks per-cycle data fetch success/failure rates to detect
Alpaca API outages during market hours. When >50% of the symbol universe fails
in the same cycle, the monitor declares "feed_down" and:
  1. Sends a CRITICAL Telegram alert via the notifications module.
  2. Provides a local price cache so exit checks can still enforce stops
     using the last known prices.
  3. Marks OU parameters as "stale" so that signal confidence is reduced
     for 5 minutes after the feed recovers.

Usage (wired into main.py scan loop)::

    from data.feed_monitor import DataFeedMonitor
    feed_monitor = DataFeedMonitor()

    # After each data fetch cycle
    feed_monitor.report_cycle(succeeded=["AAPL", "MSFT"], failed=["TSLA"])

    if feed_monitor.is_feed_down:
        # Use cached prices for exit checks
        prices = feed_monitor.get_cached_prices(["AAPL", "TSLA"])
"""

import logging
import threading
import time as _time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Lazy-load notifications to avoid circular imports
_notifications = None


def _get_notifications():
    global _notifications
    if _notifications is None:
        try:
            import notifications as _n
            _notifications = _n
        except ImportError:
            _notifications = False
    return _notifications if _notifications else None


class DataFeedMonitor:
    """Monitors Alpaca data feed health and provides backup price cache.

    Attributes:
        is_feed_down: True when >50% of symbols failed in the last cycle.
        consecutive_down_cycles: Number of consecutive cycles where feed was down.
        recovery_until: datetime until which OU params should be treated as stale.
    """

    # Thresholds
    FAILURE_THRESHOLD_PCT = 0.50   # >50% failure triggers feed_down
    RECOVERY_STALE_MINUTES = 5     # OU params stay "stale" for 5 min after recovery
    ALERT_COOLDOWN_SECONDS = 300   # Don't spam alerts more than once per 5 min

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Feed state
        self.is_feed_down: bool = False
        self.consecutive_down_cycles: int = 0
        self._feed_down_since: Optional[datetime] = None
        self.recovery_until: Optional[datetime] = None

        # Per-symbol failure tracking
        self._consecutive_failures: Dict[str, int] = {}

        # Price cache: symbol -> (price, timestamp_monotonic)
        self._price_cache: Dict[str, tuple[float, float]] = {}
        self._price_cache_max_age = 3600.0  # 1 hour max staleness

        # Alert throttling
        self._last_alert_time: float = 0.0

        # Stats for last cycle
        self._last_cycle_succeeded: int = 0
        self._last_cycle_failed: int = 0
        self._last_cycle_total: int = 0

    # ------------------------------------------------------------------
    # Core: report each scan cycle's results
    # ------------------------------------------------------------------

    def report_cycle(
        self,
        succeeded: List[str],
        failed: List[str],
        prices: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Report the outcome of a data fetch cycle.

        Args:
            succeeded: Symbols for which data was fetched successfully.
            failed: Symbols for which data fetch failed or returned empty.
            prices: Optional dict of symbol -> latest price from successful fetches.
                    If provided, updates the backup price cache.

        Returns:
            True if the feed is currently declared down.
        """
        with self._lock:
            total = len(succeeded) + len(failed)
            self._last_cycle_succeeded = len(succeeded)
            self._last_cycle_failed = len(failed)
            self._last_cycle_total = total

            # Update per-symbol failure counters
            for sym in succeeded:
                self._consecutive_failures[sym] = 0
            for sym in failed:
                self._consecutive_failures[sym] = self._consecutive_failures.get(sym, 0) + 1

            # Update price cache with fresh data
            if prices:
                now_mono = _time.monotonic()
                for sym, price in prices.items():
                    if price and price > 0:
                        self._price_cache[sym] = (price, now_mono)

            # Determine feed health
            if total == 0:
                # No symbols fetched at all — treat as down
                was_down = self.is_feed_down
                self._declare_down()
                if not was_down:
                    self._send_alert(total, len(failed))
                return self.is_feed_down

            failure_pct = len(failed) / total if total > 0 else 0.0

            if failure_pct > self.FAILURE_THRESHOLD_PCT:
                was_down = self.is_feed_down
                self._declare_down()
                if not was_down:
                    logger.critical(
                        "V12-2.3: DATA FEED DOWN — %d/%d symbols failed (%.0f%%)",
                        len(failed), total, failure_pct * 100,
                    )
                    self._send_alert(total, len(failed))
                else:
                    # Already down — log at warning level, throttle alerts
                    logger.warning(
                        "V12-2.3: Feed still down — cycle %d, %d/%d failed",
                        self.consecutive_down_cycles, len(failed), total,
                    )
                    self._maybe_repeat_alert(total, len(failed))
            else:
                if self.is_feed_down:
                    # Recovery!
                    self._declare_recovered()
                    logger.info(
                        "V12-2.3: DATA FEED RECOVERED after %d down cycles — "
                        "OU params stale until %s",
                        self.consecutive_down_cycles,
                        self.recovery_until.strftime("%H:%M:%S") if self.recovery_until else "N/A",
                    )
                    self._send_recovery_alert()
                # Reset consecutive down counter
                self.consecutive_down_cycles = 0

            return self.is_feed_down

    def _declare_down(self) -> None:
        """Mark feed as down (called under lock)."""
        if not self.is_feed_down:
            import config
            self._feed_down_since = datetime.now(config.ET)
        self.is_feed_down = True
        self.consecutive_down_cycles += 1

    def _declare_recovered(self) -> None:
        """Mark feed as recovered with stale window (called under lock)."""
        import config
        self.is_feed_down = False
        self._feed_down_since = None
        self.recovery_until = datetime.now(config.ET) + timedelta(
            minutes=self.RECOVERY_STALE_MINUTES
        )

    # ------------------------------------------------------------------
    # Price cache for backup stop enforcement
    # ------------------------------------------------------------------

    def update_price(self, symbol: str, price: float) -> None:
        """Update the cached price for a single symbol."""
        if price and price > 0:
            with self._lock:
                self._price_cache[symbol] = (price, _time.monotonic())

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Bulk-update cached prices."""
        now_mono = _time.monotonic()
        with self._lock:
            for sym, price in prices.items():
                if price and price > 0:
                    self._price_cache[sym] = (price, now_mono)

    def get_cached_price(self, symbol: str) -> Optional[float]:
        """Get the last known price for a symbol from the backup cache.

        Returns None if the symbol has never been cached or the cache entry
        is older than _price_cache_max_age.
        """
        with self._lock:
            entry = self._price_cache.get(symbol)
            if entry is None:
                return None
            price, ts = entry
            age = _time.monotonic() - ts
            if age > self._price_cache_max_age:
                return None
            return price

    def get_cached_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get cached prices for multiple symbols. Skips missing/stale entries."""
        result = {}
        with self._lock:
            now_mono = _time.monotonic()
            for sym in symbols:
                entry = self._price_cache.get(sym)
                if entry is not None:
                    price, ts = entry
                    if (now_mono - ts) <= self._price_cache_max_age:
                        result[sym] = price
        return result

    def get_cache_age(self, symbol: str) -> Optional[float]:
        """Get the age in seconds of the cached price for a symbol."""
        with self._lock:
            entry = self._price_cache.get(symbol)
            if entry is None:
                return None
            _, ts = entry
            return _time.monotonic() - ts

    # ------------------------------------------------------------------
    # OU staleness check
    # ------------------------------------------------------------------

    def are_params_stale(self) -> bool:
        """Return True if OU parameters should be treated as stale.

        Stale = feed is currently down OR we are within the recovery window.
        """
        if self.is_feed_down:
            return True
        if self.recovery_until is not None:
            import config
            now = datetime.now(config.ET)
            if now < self.recovery_until:
                return True
            # Past the window — clear it
            self.recovery_until = None
        return False

    @property
    def confidence_multiplier(self) -> float:
        """Return a confidence multiplier for signal scoring.

        1.0 = normal, <1.0 = reduced due to stale data.
        During feed_down: 0.0 (no new signals).
        During recovery window: 0.6 (reduced confidence).
        """
        if self.is_feed_down:
            return 0.0
        if self.are_params_stale():
            return 0.6
        return 1.0

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def _send_alert(self, total: int, failed: int) -> None:
        """Send CRITICAL Telegram alert for feed outage."""
        self._last_alert_time = _time.monotonic()
        import config
        notif = _get_notifications()
        if notif and config.TELEGRAM_ENABLED:
            try:
                notif._send_telegram(
                    "\U0001f6a8 *CRITICAL: DATA FEED DOWN*\n"
                    f"Failed: {failed}/{total} symbols ({failed/total*100:.0f}%)\n"
                    f"Backup stops: ACTIVE (using cached prices)\n"
                    f"New signals: BLOCKED until recovery\n"
                    f"Time: {datetime.now(config.ET).strftime('%H:%M:%S ET')}"
                )
            except Exception as e:
                logger.warning("V12-2.3: Failed to send feed-down alert: %s", e)

    def _maybe_repeat_alert(self, total: int, failed: int) -> None:
        """Repeat alert every ALERT_COOLDOWN_SECONDS while feed is down."""
        now_mono = _time.monotonic()
        if (now_mono - self._last_alert_time) >= self.ALERT_COOLDOWN_SECONDS:
            self._last_alert_time = now_mono
            import config
            notif = _get_notifications()
            if notif and config.TELEGRAM_ENABLED:
                try:
                    down_duration = ""
                    if self._feed_down_since:
                        elapsed = datetime.now(config.ET) - self._feed_down_since
                        minutes = int(elapsed.total_seconds() / 60)
                        down_duration = f"\nDuration: {minutes} min"
                    notif._send_telegram(
                        "\U0001f6a8 *DATA FEED STILL DOWN*\n"
                        f"Cycle {self.consecutive_down_cycles}: "
                        f"{failed}/{total} symbols failing"
                        f"{down_duration}\n"
                        f"Backup stops remain active."
                    )
                except Exception as e:
                    logger.warning("V12-2.3: Failed to send repeat alert: %s", e)

    def _send_recovery_alert(self) -> None:
        """Send recovery notification."""
        import config
        notif = _get_notifications()
        if notif and config.TELEGRAM_ENABLED:
            try:
                down_cycles = self.consecutive_down_cycles
                stale_until = (
                    self.recovery_until.strftime('%H:%M:%S ET')
                    if self.recovery_until else "N/A"
                )
                notif._send_telegram(
                    "\u2705 *DATA FEED RECOVERED*\n"
                    f"Down for {down_cycles} scan cycles.\n"
                    f"OU params stale until: {stale_until}\n"
                    f"Signal confidence reduced to 60% during stale window."
                )
            except Exception as e:
                logger.warning("V12-2.3: Failed to send recovery alert: %s", e)

    # ------------------------------------------------------------------
    # Stats / status
    # ------------------------------------------------------------------

    @property
    def status(self) -> dict:
        """Return a status dict for dashboard/logging."""
        return {
            "feed_down": self.is_feed_down,
            "consecutive_down_cycles": self.consecutive_down_cycles,
            "params_stale": self.are_params_stale(),
            "confidence_multiplier": self.confidence_multiplier,
            "cached_symbols": len(self._price_cache),
            "last_cycle_succeeded": self._last_cycle_succeeded,
            "last_cycle_failed": self._last_cycle_failed,
            "feed_down_since": (
                self._feed_down_since.isoformat() if self._feed_down_since else None
            ),
        }

    def get_symbols_with_failures(self, min_consecutive: int = 3) -> List[str]:
        """Return symbols that have failed consecutively >= min_consecutive times."""
        with self._lock:
            return [
                sym for sym, count in self._consecutive_failures.items()
                if count >= min_consecutive
            ]

    def __repr__(self) -> str:
        state = "DOWN" if self.is_feed_down else "OK"
        stale = " STALE" if self.are_params_stale() else ""
        return (
            f"DataFeedMonitor(state={state}{stale}, "
            f"down_cycles={self.consecutive_down_cycles}, "
            f"cached={len(self._price_cache)})"
        )


# Module-level singleton
_feed_monitor: Optional[DataFeedMonitor] = None


def get_feed_monitor() -> DataFeedMonitor:
    """Return the singleton DataFeedMonitor instance."""
    global _feed_monitor
    if _feed_monitor is None:
        _feed_monitor = DataFeedMonitor()
    return _feed_monitor
