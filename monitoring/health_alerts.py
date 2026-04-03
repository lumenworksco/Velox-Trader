"""V12 10.1: System Health Alerts — centralized health monitoring with automatic detection.

Monitors 8 critical health conditions:
  1. Database write failure        — Any DB error           → CRITICAL
  2. API rate limited              — 3+ consecutive 429s    → WARNING
  3. Data feed stale               — No bars for 10+ min    → CRITICAL
  4. Order stuck                   — SUBMITTED for >5 min   → WARNING
  5. Broker disconnected           — Account API fails 3x   → CRITICAL
  6. ML model stale                — Model >30 days old     → WARNING
  7. Strategy silent               — 0 signals for 3+ days  → WARNING
  8. Position mismatch             — DB ≠ broker count      → CRITICAL

Usage:
    from monitoring.health_alerts import SystemHealthMonitor

    monitor = SystemHealthMonitor(alert_manager)
    monitor.run_all_checks(risk_manager)  # Call periodically from main loop
"""

import logging
import os
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Health check result
# ---------------------------------------------------------------------------

@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    name: str
    healthy: bool
    severity: str = "INFO"       # INFO, WARNING, CRITICAL
    message: str = ""
    last_checked: datetime = field(default_factory=lambda: datetime.now(config.ET))


@dataclass
class SystemHealthReport:
    """Aggregated health status across all checks."""

    timestamp: datetime
    checks: list[HealthCheckResult] = field(default_factory=list)
    overall_healthy: bool = True
    critical_count: int = 0
    warning_count: int = 0

    def summary(self) -> str:
        """One-line summary for logging."""
        failed = [c for c in self.checks if not c.healthy]
        if not failed:
            return "All health checks passed"
        parts = []
        for c in failed:
            parts.append(f"{c.name}({c.severity})")
        return f"{len(failed)} issue(s): {', '.join(parts)}"


# ---------------------------------------------------------------------------
# Tracker helpers for stateful checks
# ---------------------------------------------------------------------------

class _RateLimitTracker:
    """Track consecutive 429 responses from the Alpaca API."""

    def __init__(self, threshold: int = 3):
        self._threshold = threshold
        self._consecutive_429s = 0
        self._lock = threading.Lock()

    def record_response(self, status_code: int):
        with self._lock:
            if status_code == 429:
                self._consecutive_429s += 1
            else:
                self._consecutive_429s = 0

    def is_rate_limited(self) -> bool:
        with self._lock:
            return self._consecutive_429s >= self._threshold

    @property
    def count(self) -> int:
        with self._lock:
            return self._consecutive_429s

    def reset(self):
        with self._lock:
            self._consecutive_429s = 0


class _BrokerFailureTracker:
    """Track consecutive broker API failures."""

    def __init__(self, threshold: int = 3):
        self._threshold = threshold
        self._consecutive_failures = 0
        self._lock = threading.Lock()

    def record_success(self):
        with self._lock:
            self._consecutive_failures = 0

    def record_failure(self):
        with self._lock:
            self._consecutive_failures += 1

    def is_disconnected(self) -> bool:
        with self._lock:
            return self._consecutive_failures >= self._threshold

    @property
    def count(self) -> int:
        with self._lock:
            return self._consecutive_failures

    def reset(self):
        with self._lock:
            self._consecutive_failures = 0


class _DataFeedTracker:
    """Track last bar timestamp to detect stale data feeds."""

    def __init__(self, stale_threshold_sec: int = 600):
        self._stale_threshold = stale_threshold_sec  # 10 minutes
        self._last_bar_time: float = time.monotonic()
        self._lock = threading.Lock()

    def record_bar(self):
        with self._lock:
            self._last_bar_time = time.monotonic()

    def is_stale(self) -> bool:
        with self._lock:
            return (time.monotonic() - self._last_bar_time) > self._stale_threshold

    @property
    def seconds_since_bar(self) -> float:
        with self._lock:
            return time.monotonic() - self._last_bar_time

    def reset(self):
        with self._lock:
            self._last_bar_time = time.monotonic()


# ---------------------------------------------------------------------------
# SystemHealthMonitor
# ---------------------------------------------------------------------------

class SystemHealthMonitor:
    """Centralized health monitoring for the trading system.

    Runs periodic checks against all critical subsystems and dispatches
    alerts through the existing AlertManager.

    Args:
        alert_manager: The monitoring.alerting.AlertManager instance (or None).
        check_interval_sec: Minimum seconds between full health check runs.
    """

    def __init__(self, alert_manager=None, check_interval_sec: int = 300):
        self._alert_manager = alert_manager
        self._check_interval = check_interval_sec
        self._last_check: float = 0.0

        # Stateful trackers
        self.rate_limit_tracker = _RateLimitTracker(threshold=3)
        self.broker_tracker = _BrokerFailureTracker(threshold=3)
        self.data_feed_tracker = _DataFeedTracker(stale_threshold_sec=600)

        # DB write error tracking
        self._db_error_detected = False
        self._db_error_message = ""
        self._db_lock = threading.Lock()

        # Alert dedup: track which alerts have fired this cycle
        self._fired_alerts: dict[str, float] = {}
        self._alert_cooldown_sec = 600  # Re-alert after 10 minutes

        logger.info("V12 10.1: SystemHealthMonitor initialized (interval=%ds)", check_interval_sec)

    # ------------------------------------------------------------------
    # Public: record events from the main loop
    # ------------------------------------------------------------------

    def record_api_response(self, status_code: int):
        """Call after each API response to track rate limiting."""
        self.rate_limit_tracker.record_response(status_code)

    def record_broker_success(self):
        """Call after a successful broker API call."""
        self.broker_tracker.record_success()

    def record_broker_failure(self):
        """Call after a failed broker API call."""
        self.broker_tracker.record_failure()

    def record_bar_received(self):
        """Call when a market data bar is received."""
        self.data_feed_tracker.record_bar()

    def record_db_error(self, error_msg: str):
        """Call when a database write error occurs."""
        with self._db_lock:
            self._db_error_detected = True
            self._db_error_message = str(error_msg)[:200]

    def clear_db_error(self):
        """Call after a successful database write to clear error state."""
        with self._db_lock:
            self._db_error_detected = False
            self._db_error_message = ""

    # ------------------------------------------------------------------
    # Public: run checks
    # ------------------------------------------------------------------

    def should_run(self) -> bool:
        """Whether enough time has passed for another check cycle."""
        return (time.monotonic() - self._last_check) >= self._check_interval

    def run_all_checks(self, risk_manager=None) -> SystemHealthReport:
        """Execute all health checks and dispatch alerts.

        Args:
            risk_manager: The RiskManager instance (for position/order checks).

        Returns:
            SystemHealthReport with results of all checks.
        """
        self._last_check = time.monotonic()
        now = datetime.now(config.ET)
        report = SystemHealthReport(timestamp=now)

        # 1. Database write failure
        report.checks.append(self._check_database())

        # 2. API rate limited
        report.checks.append(self._check_rate_limiting())

        # 3. Data feed stale (only during market hours)
        report.checks.append(self._check_data_feed(now))

        # 4. Order stuck
        report.checks.append(self._check_stuck_orders(risk_manager))

        # 5. Broker disconnected
        report.checks.append(self._check_broker_connection())

        # 6. ML model stale
        report.checks.append(self._check_ml_model_age())

        # 7. Strategy silent
        report.checks.append(self._check_strategy_silence())

        # 8. Position mismatch
        report.checks.append(self._check_position_mismatch(risk_manager))

        # Aggregate
        for check in report.checks:
            if not check.healthy:
                if check.severity == "CRITICAL":
                    report.critical_count += 1
                elif check.severity == "WARNING":
                    report.warning_count += 1
                report.overall_healthy = False

                # Dispatch alert (with cooldown dedup)
                self._dispatch_alert(check)

        logger.info("V12 10.1: Health check: %s", report.summary())
        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_database(self) -> HealthCheckResult:
        """Check 1: Database write failure."""
        with self._db_lock:
            if self._db_error_detected:
                return HealthCheckResult(
                    name="database",
                    healthy=False,
                    severity="CRITICAL",
                    message=f"Database write failure: {self._db_error_message}",
                )
        return HealthCheckResult(name="database", healthy=True)

    def _check_rate_limiting(self) -> HealthCheckResult:
        """Check 2: API rate limited (3+ consecutive 429s)."""
        if self.rate_limit_tracker.is_rate_limited():
            return HealthCheckResult(
                name="api_rate_limit",
                healthy=False,
                severity="WARNING",
                message=f"API rate limited: {self.rate_limit_tracker.count} consecutive 429s",
            )
        return HealthCheckResult(name="api_rate_limit", healthy=True)

    def _check_data_feed(self, now: datetime) -> HealthCheckResult:
        """Check 3: Data feed stale (no bars for 10+ min during market hours)."""
        # Only check during market hours
        current_time = now.time()
        if not (config.MARKET_OPEN <= current_time <= config.MARKET_CLOSE):
            return HealthCheckResult(name="data_feed", healthy=True, message="Market closed")

        if self.data_feed_tracker.is_stale():
            elapsed = self.data_feed_tracker.seconds_since_bar
            return HealthCheckResult(
                name="data_feed",
                healthy=False,
                severity="CRITICAL",
                message=f"No market data bars for {elapsed / 60:.1f} minutes",
            )
        return HealthCheckResult(name="data_feed", healthy=True)

    def _check_stuck_orders(self, risk_manager) -> HealthCheckResult:
        """Check 4: Orders stuck in SUBMITTED state for >5 minutes."""
        if risk_manager is None:
            return HealthCheckResult(name="stuck_orders", healthy=True, message="No risk manager")

        now = datetime.now(config.ET)
        stuck_threshold = timedelta(minutes=5)
        stuck_orders = []

        try:
            open_trades = getattr(risk_manager, 'open_trades', {})
            for symbol, trade in open_trades.items():
                # Check if the trade has been in a "pending" state too long
                entry_time = getattr(trade, 'entry_time', None)
                status = getattr(trade, 'status', 'active')
                if status in ('submitted', 'pending') and entry_time:
                    if (now - entry_time) > stuck_threshold:
                        stuck_orders.append(symbol)
        except Exception as e:
            logger.debug("Stuck order check failed: %s", e)
            return HealthCheckResult(name="stuck_orders", healthy=True, message=f"Check error: {e}")

        # Also check OMS if available
        try:
            from container import Container
            ctr = Container.instance()
            oms = ctr.get("oms")
            if oms and hasattr(oms, 'get_active_orders'):
                for order in oms.get_active_orders():
                    created = getattr(order, 'created_at', None) or getattr(order, 'submitted_at', None)
                    if created and (now - created) > stuck_threshold:
                        sym = getattr(order, 'symbol', 'unknown')
                        if sym not in stuck_orders:
                            stuck_orders.append(sym)
        except Exception:
            pass

        if stuck_orders:
            return HealthCheckResult(
                name="stuck_orders",
                healthy=False,
                severity="WARNING",
                message=f"Orders stuck >5 min: {', '.join(stuck_orders[:5])}",
            )
        return HealthCheckResult(name="stuck_orders", healthy=True)

    def _check_broker_connection(self) -> HealthCheckResult:
        """Check 5: Broker disconnected (account API fails 3x)."""
        if self.broker_tracker.is_disconnected():
            return HealthCheckResult(
                name="broker_connection",
                healthy=False,
                severity="CRITICAL",
                message=f"Broker disconnected: {self.broker_tracker.count} consecutive API failures",
            )
        return HealthCheckResult(name="broker_connection", healthy=True)

    def _check_ml_model_age(self) -> HealthCheckResult:
        """Check 6: ML model stale (>30 days old)."""
        try:
            import glob as _g
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            ml_models = _g.glob(os.path.join(model_dir, "model_*.pkl"))
            if not ml_models:
                return HealthCheckResult(
                    name="ml_model",
                    healthy=False,
                    severity="WARNING",
                    message="No ML model found in models/",
                )

            # Check age of most recent model file
            latest_model = max(ml_models, key=os.path.getmtime)
            model_age_days = (time.time() - os.path.getmtime(latest_model)) / 86400

            if model_age_days > 30:
                return HealthCheckResult(
                    name="ml_model",
                    healthy=False,
                    severity="WARNING",
                    message=f"ML model is {model_age_days:.0f} days old (>30 day threshold)",
                )
            return HealthCheckResult(
                name="ml_model",
                healthy=True,
                message=f"Model age: {model_age_days:.0f} days",
            )
        except Exception as e:
            logger.debug("ML model age check failed: %s", e)
            return HealthCheckResult(name="ml_model", healthy=True, message=f"Check error: {e}")

    def _check_strategy_silence(self) -> HealthCheckResult:
        """Check 7: Strategy silent (0 signals for 3+ days)."""
        try:
            import database
            silent_strategies = []

            for strategy in config.STRATEGY_ALLOCATIONS:
                try:
                    signals = database.get_signals_by_strategy(strategy, days=3)
                    if not signals:
                        silent_strategies.append(strategy)
                except Exception:
                    pass  # Skip strategies with no signal table entries

            if silent_strategies:
                return HealthCheckResult(
                    name="strategy_silence",
                    healthy=False,
                    severity="WARNING",
                    message=f"No signals for 3+ days: {', '.join(silent_strategies)}",
                )
            return HealthCheckResult(name="strategy_silence", healthy=True)
        except Exception as e:
            logger.debug("Strategy silence check failed: %s", e)
            return HealthCheckResult(name="strategy_silence", healthy=True, message=f"Check error: {e}")

    def _check_position_mismatch(self, risk_manager) -> HealthCheckResult:
        """Check 8: Position mismatch (DB count != broker count)."""
        if risk_manager is None:
            return HealthCheckResult(name="position_mismatch", healthy=True, message="No risk manager")

        try:
            from data import get_positions

            # Get broker positions
            broker_positions = get_positions()
            broker_count = len(broker_positions) if broker_positions else 0

            # Get DB positions (risk manager's tracked positions)
            db_count = len(getattr(risk_manager, 'open_trades', {}))

            # Exclude known non-tracked symbols (e.g., SPY hedge)
            broker_symbols = set()
            if broker_positions:
                for pos in broker_positions:
                    sym = getattr(pos, 'symbol', None) or pos.get('symbol', '') if isinstance(pos, dict) else ''
                    if sym:
                        broker_symbols.add(sym)

            # Filter out broker-sync excluded symbols
            excluded = getattr(config, 'BROKER_SYNC_EXCLUDE_SYMBOLS', set())
            broker_symbols -= excluded
            adjusted_broker_count = len(broker_symbols)

            if abs(db_count - adjusted_broker_count) > 0:
                db_symbols = set(getattr(risk_manager, 'open_trades', {}).keys())
                only_db = db_symbols - broker_symbols
                only_broker = broker_symbols - db_symbols

                detail_parts = []
                if only_db:
                    detail_parts.append(f"only in DB: {', '.join(list(only_db)[:3])}")
                if only_broker:
                    detail_parts.append(f"only at broker: {', '.join(list(only_broker)[:3])}")

                return HealthCheckResult(
                    name="position_mismatch",
                    healthy=False,
                    severity="CRITICAL",
                    message=f"Position mismatch: DB={db_count}, broker={adjusted_broker_count}. {'; '.join(detail_parts)}",
                )
            return HealthCheckResult(name="position_mismatch", healthy=True)
        except Exception as e:
            logger.debug("Position mismatch check failed: %s", e)
            return HealthCheckResult(
                name="position_mismatch", healthy=True,
                message=f"Check error: {e}",
            )

    # ------------------------------------------------------------------
    # Alert dispatch with cooldown
    # ------------------------------------------------------------------

    def _dispatch_alert(self, check: HealthCheckResult):
        """Send alert through AlertManager with cooldown dedup."""
        now = time.monotonic()
        last_fired = self._fired_alerts.get(check.name, 0.0)

        if (now - last_fired) < self._alert_cooldown_sec:
            return  # Still in cooldown

        self._fired_alerts[check.name] = now

        if self._alert_manager:
            try:
                self._alert_manager.send_alert(
                    level=check.severity,
                    message=check.message,
                    source=f"health_monitor.{check.name}",
                )
            except Exception as e:
                logger.error("Failed to dispatch health alert: %s", e)
        else:
            # Fallback to direct Telegram if no AlertManager
            try:
                import notifications
                severity_emoji = {
                    "CRITICAL": "\U0001f6a8",
                    "WARNING": "\u26a0\ufe0f",
                }.get(check.severity, "\u2139\ufe0f")
                notifications._send_telegram(
                    f"{severity_emoji} *HEALTH ALERT: {check.severity}*\n"
                    f"Check: {check.name}\n"
                    f"{check.message}"
                )
            except Exception as e:
                logger.error("Health alert Telegram fallback failed: %s", e)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return current health monitor state for dashboard/API."""
        return {
            "rate_limit_429_count": self.rate_limit_tracker.count,
            "broker_failure_count": self.broker_tracker.count,
            "data_feed_stale": self.data_feed_tracker.is_stale(),
            "data_feed_seconds_since_bar": round(self.data_feed_tracker.seconds_since_bar, 1),
            "db_error_detected": self._db_error_detected,
            "active_alerts": len(self._fired_alerts),
        }
