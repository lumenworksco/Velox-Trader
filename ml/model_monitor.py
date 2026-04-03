"""V12 8.3: ML Model Accuracy Monitoring + Auto-Retrain Signal.

Tracks rolling accuracy of ML predictions vs actual trade outcomes.
Alerts if accuracy falls below 52% for 2 consecutive weeks.
"""

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MONITOR_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "model_monitor_state.json",
)

ROLLING_WINDOW = 50           # Last N trades for accuracy
ACCURACY_THRESHOLD = 0.52     # Minimum acceptable accuracy
STALE_MODEL_DAYS = 30         # Alert if model older than this
ALERT_COOLDOWN_SEC = 86400    # 24h between repeated alerts


@dataclass
class PredictionRecord:
    """Single ML prediction vs actual outcome."""
    symbol: str
    strategy: str
    prediction: float       # ML score (0-1, >0.5 = positive)
    actual_outcome: float   # 1.0 = profitable, 0.0 = loss
    timestamp: float = field(default_factory=time.time)


class ModelAccuracyMonitor:
    """Track ML prediction accuracy and emit alerts when degraded."""

    def __init__(self, alert_callback=None, state_file: str = _MONITOR_FILE):
        self._records: deque[PredictionRecord] = deque(maxlen=ROLLING_WINDOW)
        self._state_file = state_file
        self._alert_callback = alert_callback
        self._last_alert_time = 0.0
        self._below_threshold_since: Optional[float] = None
        self._load_state()

    def record_outcome(
        self, symbol: str, strategy: str, prediction: float, profitable: bool,
    ) -> None:
        """Record a completed trade's ML prediction vs actual outcome."""
        rec = PredictionRecord(
            symbol=symbol,
            strategy=strategy,
            prediction=prediction,
            actual_outcome=1.0 if profitable else 0.0,
        )
        self._records.append(rec)

        # Check accuracy after every record
        accuracy = self.get_accuracy()
        if accuracy is not None and accuracy < ACCURACY_THRESHOLD:
            if self._below_threshold_since is None:
                self._below_threshold_since = time.time()
            elif time.time() - self._below_threshold_since > 14 * 86400:
                # Below threshold for >2 weeks
                self._emit_alert(accuracy)
        else:
            self._below_threshold_since = None

        # Persist periodically (every 10 records)
        if len(self._records) % 10 == 0:
            self._save_state()

    def get_accuracy(self) -> Optional[float]:
        """Rolling accuracy over last ROLLING_WINDOW trades."""
        if len(self._records) < 10:
            return None
        correct = sum(
            1 for r in self._records
            if (r.prediction >= 0.5) == (r.actual_outcome >= 0.5)
        )
        return correct / len(self._records)

    def get_stats(self) -> dict:
        """Return current monitoring statistics."""
        accuracy = self.get_accuracy()
        return {
            "total_records": len(self._records),
            "rolling_accuracy": round(accuracy, 4) if accuracy else None,
            "below_threshold_since": self._below_threshold_since,
            "threshold": ACCURACY_THRESHOLD,
            "window_size": ROLLING_WINDOW,
        }

    def check_model_staleness(self, model_path: str) -> bool:
        """Check if the model file is older than STALE_MODEL_DAYS."""
        try:
            mtime = os.path.getmtime(model_path)
            age_days = (time.time() - mtime) / 86400
            if age_days > STALE_MODEL_DAYS:
                logger.warning(
                    "V12 8.3: ML model is %.0f days old (threshold: %d days) — "
                    "consider retraining",
                    age_days, STALE_MODEL_DAYS,
                )
                return True
        except (OSError, TypeError):
            pass
        return False

    def _emit_alert(self, accuracy: float) -> None:
        """Send alert if cooldown has elapsed."""
        now = time.time()
        if now - self._last_alert_time < ALERT_COOLDOWN_SEC:
            return
        self._last_alert_time = now
        msg = (
            f"V12 8.3 ALERT: ML model accuracy {accuracy:.1%} "
            f"< {ACCURACY_THRESHOLD:.0%} for >2 weeks. "
            f"Consider retraining. Window: {len(self._records)} trades."
        )
        logger.warning(msg)
        if self._alert_callback:
            try:
                self._alert_callback(msg, severity="WARNING")
            except Exception as e:
                logger.debug("Model monitor alert callback failed: %s", e)

    def _save_state(self) -> None:
        """Persist monitor state to disk."""
        try:
            state = {
                "records": [
                    {
                        "symbol": r.symbol,
                        "strategy": r.strategy,
                        "prediction": r.prediction,
                        "actual_outcome": r.actual_outcome,
                        "timestamp": r.timestamp,
                    }
                    for r in self._records
                ],
                "below_threshold_since": self._below_threshold_since,
                "last_alert_time": self._last_alert_time,
            }
            Path(self._state_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, "w") as f:
                json.dump(state, f)
        except Exception as e:
            logger.debug("Failed to save model monitor state: %s", e)

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        try:
            if os.path.exists(self._state_file):
                with open(self._state_file, "r") as f:
                    state = json.load(f)
                for r in state.get("records", []):
                    self._records.append(PredictionRecord(
                        symbol=r["symbol"],
                        strategy=r["strategy"],
                        prediction=r["prediction"],
                        actual_outcome=r["actual_outcome"],
                        timestamp=r.get("timestamp", 0),
                    ))
                self._below_threshold_since = state.get("below_threshold_since")
                self._last_alert_time = state.get("last_alert_time", 0)
                logger.info(
                    "V12 8.3: Loaded model monitor state — %d records, "
                    "accuracy=%.1f%%",
                    len(self._records),
                    (self.get_accuracy() or 0) * 100,
                )
        except Exception as e:
            logger.debug("Failed to load model monitor state: %s", e)


# Module-level singleton
_monitor: Optional[ModelAccuracyMonitor] = None


def get_model_monitor(alert_callback=None) -> ModelAccuracyMonitor:
    """Get or create the singleton ModelAccuracyMonitor."""
    global _monitor
    if _monitor is None:
        _monitor = ModelAccuracyMonitor(alert_callback=alert_callback)
    return _monitor
