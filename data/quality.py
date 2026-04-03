"""DATA-005: Enhanced Data Quality Framework.

Comprehensive data quality validation with per-bar scoring (0-1).
Extends the original data_quality.py with additional checks:
staleness, volume anomalies, price anomalies, gap detection, OHLC
consistency, corporate action detection, and a composite quality score
that strategies can use to weight their signals.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality check result types
# ---------------------------------------------------------------------------

@dataclass
class QualityIssue:
    """A single data quality issue found in a bar or bar series."""
    check_name: str
    severity: str  # "warning" or "critical"
    bar_index: Optional[int] = None  # index into the DataFrame, if per-bar
    message: str = ""
    penalty: float = 0.0  # score reduction (0-1)


@dataclass
class DataQualityScore:
    """Composite quality score for a bar series."""
    overall: float = 1.0  # 0.0 = unusable, 1.0 = perfect
    per_bar: Optional[np.ndarray] = None  # per-bar scores if computed
    issues: list[QualityIssue] = field(default_factory=list)
    is_tradeable: bool = True  # False if critical issues found

    def add_issue(self, issue: QualityIssue):
        self.issues.append(issue)
        self.overall = max(0.0, self.overall - issue.penalty)
        if issue.severity == "critical":
            self.is_tradeable = False

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")


# ---------------------------------------------------------------------------
# Individual quality checks
# ---------------------------------------------------------------------------

def _check_staleness(bars: pd.DataFrame, symbol: str, now: Optional[datetime],
                     max_staleness_sec: float) -> List[QualityIssue]:
    """Check if the latest bar is too old."""
    issues = []
    if now is None or bars.empty:
        return issues

    try:
        last_ts = bars.index[-1]
        # Handle timezone-aware vs naive
        if hasattr(last_ts, "tzinfo") and last_ts.tzinfo is not None:
            if now.tzinfo is None:
                from datetime import timezone
                now = now.replace(tzinfo=timezone.utc)
        age_sec = (now - last_ts).total_seconds()
        if age_sec > max_staleness_sec:
            severity = "critical" if age_sec > max_staleness_sec * 3 else "warning"
            issues.append(QualityIssue(
                check_name="staleness",
                severity=severity,
                message=f"{symbol}: latest bar is {age_sec:.0f}s old "
                        f"(threshold: {max_staleness_sec}s)",
                penalty=min(0.3, 0.1 * (age_sec / max_staleness_sec)),
            ))
    except Exception as e:
        logger.debug("Staleness check error for %s: %s", symbol, e)

    return issues


def _check_volume_anomalies(bars: pd.DataFrame, symbol: str,
                             low_pct: float = 0.10,
                             high_mult: float = 10.0,
                             lookback: int = 20) -> List[QualityIssue]:
    """Detect volume anomalies: too low (<10% of avg) or too high (>1000% of avg)."""
    issues = []
    if "volume" not in bars.columns or len(bars) < lookback + 1:
        return issues

    vol = bars["volume"]
    avg_vol = vol.iloc[:-1].tail(lookback).mean()
    if avg_vol <= 0:
        return issues

    current_vol = vol.iloc[-1]
    ratio = current_vol / avg_vol

    if ratio < low_pct:
        issues.append(QualityIssue(
            check_name="volume_low",
            severity="warning",
            bar_index=len(bars) - 1,
            message=f"{symbol}: volume {ratio:.1%} of 20-day avg "
                    f"(possible halt or thin trading)",
            penalty=0.15,
        ))

    if ratio > high_mult:
        issues.append(QualityIssue(
            check_name="volume_spike",
            severity="warning",
            bar_index=len(bars) - 1,
            message=f"{symbol}: volume {ratio:.0f}x of 20-day avg "
                    f"(possible news/event)",
            penalty=0.05,  # high volume is less dangerous, just notable
        ))

    return issues


def _check_price_anomalies(bars: pd.DataFrame, symbol: str,
                            max_single_move: float = 0.15) -> List[QualityIssue]:
    """Detect single-bar price moves exceeding threshold."""
    issues = []
    if "close" not in bars.columns or len(bars) < 2:
        return issues

    returns = bars["close"].pct_change().dropna()
    anomalous = returns.abs() > max_single_move

    if anomalous.any():
        worst_idx = returns.abs().idxmax()
        worst_move = returns.loc[worst_idx]
        bar_pos = bars.index.get_loc(worst_idx)
        issues.append(QualityIssue(
            check_name="price_anomaly",
            severity="warning",
            bar_index=bar_pos,
            message=f"{symbol}: Price anomaly, {worst_move:.1%} single-bar move at bar {bar_pos}",
            penalty=0.10,
        ))

    return issues


def _check_gaps(bars: pd.DataFrame, symbol: str,
                max_gap_pct: float = 0.05) -> List[QualityIssue]:
    """Detect large gaps between consecutive bars (close-to-open)."""
    issues = []
    if not all(c in bars.columns for c in ("open", "close")) or len(bars) < 2:
        return issues

    prev_close = bars["close"].shift(1)
    gap_pct = ((bars["open"] - prev_close) / prev_close).dropna()
    large_gaps = gap_pct.abs() > max_gap_pct

    if large_gaps.any():
        gap_count = large_gaps.sum()
        worst_idx = gap_pct.abs().idxmax()
        worst_gap = gap_pct.loc[worst_idx]
        issues.append(QualityIssue(
            check_name="gap_detection",
            severity="warning",
            message=f"{symbol}: {gap_count} gap(s) > {max_gap_pct:.0%}, "
                    f"worst: {worst_gap:.1%}",
            penalty=min(0.10, 0.03 * gap_count),
        ))

    return issues


def _check_ohlc_consistency(bars: pd.DataFrame, symbol: str) -> List[QualityIssue]:
    """Verify OHLC invariants: high >= max(open,close), low <= min(open,close)."""
    issues = []
    required = {"open", "high", "low", "close"}
    if not required.issubset(bars.columns) or bars.empty:
        return issues

    high_violation = (bars["high"] < bars[["open", "close"]].max(axis=1))
    low_violation = (bars["low"] > bars[["open", "close"]].min(axis=1))
    violations = high_violation | low_violation
    violation_count = violations.sum()

    if violation_count > 0:
        severity = "critical" if violation_count > len(bars) * 0.05 else "warning"
        issues.append(QualityIssue(
            check_name="ohlc_consistency",
            severity=severity,
            message=f"{symbol}: {violation_count} bar(s) with OHLC inconsistency",
            penalty=min(0.20, 0.05 * violation_count),
        ))

    return issues


def _check_corporate_actions(bars: pd.DataFrame, symbol: str,
                              overnight_threshold: float = 0.20) -> List[QualityIssue]:
    """Detect possible corporate actions (overnight change > 20% with volume spike).

    BUG-025: Enhanced to also check for volume spikes coinciding with large
    overnight moves, which strengthens the corporate action signal. Flags
    affected symbols for cached indicator invalidation.
    """
    issues = []
    if not all(c in bars.columns for c in ("open", "close")) or len(bars) < 2:
        return issues

    prev_close = bars["close"].shift(1)
    overnight_change = ((bars["open"] - prev_close) / prev_close).dropna()
    corp_action = overnight_change.abs() > overnight_threshold

    if corp_action.any():
        count = corp_action.sum()
        worst_idx = overnight_change.abs().idxmax()
        worst_change = overnight_change.loc[worst_idx]

        # BUG-025: Check for volume spike on the same bar (strengthens signal)
        has_volume_spike = False
        if "volume" in bars.columns:
            try:
                bar_pos = bars.index.get_loc(worst_idx)
                if bar_pos >= 5:
                    avg_vol = bars["volume"].iloc[max(0, bar_pos - 20):bar_pos].mean()
                    if avg_vol > 0 and bars["volume"].iloc[bar_pos] > avg_vol * 3:
                        has_volume_spike = True
            except Exception:
                pass

        volume_note = " with volume spike" if has_volume_spike else ""
        issues.append(QualityIssue(
            check_name="corporate_action",
            severity="critical",
            message=f"{symbol}: {count} possible corporate action(s){volume_note}, "
                    f"worst overnight change: {worst_change:.1%} "
                    f"(split/dividend/M&A?) — cached indicators should be invalidated",
            penalty=0.25 if not has_volume_spike else 0.40,
        ))

        # BUG-025: Flag for indicator invalidation
        _flag_indicator_invalidation(symbol, worst_idx)

    return issues


# BUG-025: Track symbols needing indicator cache invalidation
_invalidated_symbols: Dict[str, datetime] = {}
_invalidated_symbols_lock = threading.Lock()  # MED-007: protect concurrent access


def _flag_indicator_invalidation(symbol: str, event_timestamp) -> None:
    """BUG-025: Flag a symbol for cached indicator invalidation due to corporate action."""
    try:
        ts = pd.Timestamp(event_timestamp)
        with _invalidated_symbols_lock:
            _invalidated_symbols[symbol] = ts.to_pydatetime()
        logger.warning(f"BUG-025: {symbol} flagged for indicator invalidation at {ts}")
    except Exception:
        with _invalidated_symbols_lock:
            _invalidated_symbols[symbol] = datetime.now()


def get_invalidated_symbols() -> Dict[str, datetime]:
    """Return symbols flagged for indicator cache invalidation due to corporate actions."""
    with _invalidated_symbols_lock:
        return dict(_invalidated_symbols)


def clear_invalidation(symbol: str) -> None:
    """Clear the invalidation flag for a symbol after indicators have been recomputed."""
    with _invalidated_symbols_lock:
        _invalidated_symbols.pop(symbol, None)


def _check_duplicate_timestamps(bars: pd.DataFrame, symbol: str) -> List[QualityIssue]:
    """Check for duplicate timestamps."""
    issues = []
    if bars.empty:
        return issues

    dup_count = bars.index.duplicated().sum()
    if dup_count > 0:
        issues.append(QualityIssue(
            check_name="duplicate_timestamps",
            severity="warning",
            message=f"{symbol}: {dup_count} duplicate timestamp(s)",
            penalty=min(0.10, 0.02 * dup_count),
        ))

    return issues


def _check_minimum_bars(bars: pd.DataFrame, symbol: str,
                         min_bars: int) -> List[QualityIssue]:
    """Check that enough bars are present for analysis."""
    issues = []
    if len(bars) < min_bars:
        issues.append(QualityIssue(
            check_name="minimum_bars",
            severity="critical",
            message=f"{symbol}: Only {len(bars)} bars (need {min_bars})",
            penalty=0.30,
        ))
    return issues


def _check_zero_volume_bars(bars: pd.DataFrame, symbol: str,
                             max_zero_pct: float = 0.10) -> List[QualityIssue]:
    """Check for excessive zero-volume bars (halted symbol)."""
    issues = []
    if "volume" not in bars.columns or bars.empty:
        return issues

    zero_pct = (bars["volume"] == 0).mean()
    if zero_pct > max_zero_pct:
        issues.append(QualityIssue(
            check_name="zero_volume",
            severity="critical" if zero_pct > 0.30 else "warning",
            message=f"{symbol}: {zero_pct:.0%} zero-volume bars (possible halt)",
            penalty=min(0.25, zero_pct),
        ))

    return issues


# ---------------------------------------------------------------------------
# Per-bar quality scores
# ---------------------------------------------------------------------------

def _compute_per_bar_scores(bars: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    """Compute a quality score (0-1) for each bar in the DataFrame.

    Factors:
    - Volume normality (vs rolling average)
    - Return normality (vs rolling std)
    - OHLC consistency
    """
    n = len(bars)
    scores = np.ones(n, dtype=np.float64)

    if n < 2:
        return scores

    # Volume score: penalise bars with volume < 10% or > 1000% of rolling avg
    if "volume" in bars.columns:
        vol = bars["volume"].values.astype(np.float64)
        for i in range(lookback, n):
            window = vol[max(0, i - lookback):i]
            avg = window.mean()
            if avg > 0:
                ratio = vol[i] / avg
                if ratio < 0.10:
                    scores[i] -= 0.15
                elif ratio > 10.0:
                    scores[i] -= 0.05

    # Return score: penalise bars with returns > 3 sigma
    if "close" in bars.columns:
        close = bars["close"].values.astype(np.float64)
        for i in range(lookback + 1, n):
            window_rets = np.diff(np.log(close[max(0, i - lookback):i + 1]))
            if len(window_rets) > 1:
                std = window_rets[:-1].std()
                if std > 0:
                    z = abs(window_rets[-1]) / std
                    if z > 3.0:
                        scores[i] -= min(0.20, 0.05 * (z - 3.0))

    # OHLC consistency
    if all(c in bars.columns for c in ("open", "high", "low", "close")):
        o = bars["open"].values
        h = bars["high"].values
        l = bars["low"].values
        c = bars["close"].values
        for i in range(n):
            if h[i] < max(o[i], c[i]) or l[i] > min(o[i], c[i]):
                scores[i] -= 0.20

    return np.clip(scores, 0.0, 1.0)


# ---------------------------------------------------------------------------
# DataQualityFramework
# ---------------------------------------------------------------------------

class DataQualityFramework:
    """Enhanced data quality validation with configurable thresholds.

    Usage:
        dqf = DataQualityFramework()
        score = dqf.check(bars, "AAPL", now=datetime.now(tz=...))
        if not score.is_tradeable:
            logger.warning("Data quality too low for %s", symbol)
        # Weight signals by score.overall
        signal_confidence *= score.overall
    """

    def __init__(
        self,
        max_staleness_sec: float = 300.0,
        max_single_move: float = 0.15,
        max_gap_pct: float = 0.05,
        overnight_threshold: float = 0.20,
        max_zero_vol_pct: float = 0.10,
        volume_low_pct: float = 0.10,
        volume_high_mult: float = 10.0,
        volume_lookback: int = 20,
        min_bars: int = 50,
        min_tradeable_score: float = 0.50,
        compute_per_bar: bool = True,
    ):
        self.max_staleness_sec = max_staleness_sec
        self.max_single_move = max_single_move
        self.max_gap_pct = max_gap_pct
        self.overnight_threshold = overnight_threshold
        self.max_zero_vol_pct = max_zero_vol_pct
        self.volume_low_pct = volume_low_pct
        self.volume_high_mult = volume_high_mult
        self.volume_lookback = volume_lookback
        self.min_bars = min_bars
        self.min_tradeable_score = min_tradeable_score
        self.compute_per_bar = compute_per_bar

        # Stats
        self._checks_run = 0
        self._symbols_flagged = 0

        logger.info("DataQualityFramework initialised (min_score=%.2f, "
                     "staleness=%ds, min_bars=%d)",
                     min_tradeable_score, int(max_staleness_sec), min_bars)

    def check(self, bars: pd.DataFrame, symbol: str,
              now: Optional[datetime] = None) -> DataQualityScore:
        """Run all quality checks on a bar DataFrame.

        Args:
            bars: DataFrame with OHLCV data, datetime-indexed.
            symbol: Ticker symbol (for logging).
            now: Current time for staleness check. If None, staleness
                 check is skipped.

        Returns:
            DataQualityScore with overall score, per-bar scores, and issues.
        """
        self._checks_run += 1
        score = DataQualityScore()

        if bars is None or bars.empty:
            score.add_issue(QualityIssue(
                check_name="empty_data",
                severity="critical",
                message=f"{symbol}: no bar data available",
                penalty=1.0,
            ))
            return score

        # Run all checks
        all_issues: List[QualityIssue] = []

        all_issues.extend(_check_minimum_bars(bars, symbol, self.min_bars))
        all_issues.extend(_check_staleness(bars, symbol, now, self.max_staleness_sec))
        all_issues.extend(_check_volume_anomalies(
            bars, symbol, self.volume_low_pct, self.volume_high_mult, self.volume_lookback))
        all_issues.extend(_check_zero_volume_bars(bars, symbol, self.max_zero_vol_pct))
        all_issues.extend(_check_price_anomalies(bars, symbol, self.max_single_move))
        all_issues.extend(_check_gaps(bars, symbol, self.max_gap_pct))
        all_issues.extend(_check_ohlc_consistency(bars, symbol))
        all_issues.extend(_check_corporate_actions(bars, symbol, self.overnight_threshold))
        all_issues.extend(_check_duplicate_timestamps(bars, symbol))

        for issue in all_issues:
            score.add_issue(issue)

        # Override tradeable flag based on minimum score
        if score.overall < self.min_tradeable_score:
            score.is_tradeable = False

        # Compute per-bar scores
        if self.compute_per_bar and not bars.empty:
            score.per_bar = _compute_per_bar_scores(bars, self.volume_lookback)

        if score.issue_count > 0:
            self._symbols_flagged += 1
            log_fn = logger.warning if not score.is_tradeable else logger.debug
            log_fn("DataQuality %s: score=%.2f tradeable=%s issues=%d (%d critical)",
                   symbol, score.overall, score.is_tradeable,
                   score.issue_count, score.critical_count)

        return score

    def check_and_clean(self, bars: pd.DataFrame, symbol: str,
                        now: Optional[datetime] = None,
                        min_bar_score: float = 0.5) -> Tuple[pd.DataFrame, DataQualityScore]:
        """HIGH-026: Run quality checks and return cleaned dataset with flagged bars.

        Instead of rejecting entire datasets, flags individual bad bars and
        returns a cleaned dataset. Adds a '_quality_score' column (0-1 per bar)
        and a '_flagged' boolean column indicating bars below min_bar_score.

        Args:
            bars: DataFrame with OHLCV data, datetime-indexed.
            symbol: Ticker symbol.
            now: Current time for staleness check.
            min_bar_score: Bars with per-bar score below this are flagged.

        Returns:
            Tuple of (cleaned_bars, DataQualityScore).
            cleaned_bars has '_quality_score' and '_flagged' columns added.
            Flagged bars remain in the DataFrame but are marked for exclusion.
        """
        score = self.check(bars, symbol, now=now)

        cleaned = bars.copy()
        if score.per_bar is not None and len(score.per_bar) == len(cleaned):
            cleaned["_quality_score"] = score.per_bar
        else:
            cleaned["_quality_score"] = 1.0

        cleaned["_flagged"] = cleaned["_quality_score"] < min_bar_score

        flagged_count = int(cleaned["_flagged"].sum())
        if flagged_count > 0:
            logger.info(
                "DataQuality %s: %d/%d bars flagged (score < %.2f), keeping in dataset",
                symbol, flagged_count, len(cleaned), min_bar_score,
            )

        return cleaned, score

    def check_multiple(self, bars_dict: Dict[str, pd.DataFrame],
                       now: Optional[datetime] = None
                       ) -> Dict[str, DataQualityScore]:
        """Run quality checks on multiple symbols.

        Args:
            bars_dict: Mapping of symbol -> OHLCV DataFrame.
            now: Current time for staleness checks.

        Returns:
            Mapping of symbol -> DataQualityScore.
        """
        results = {}
        for symbol, bars in bars_dict.items():
            results[symbol] = self.check(bars, symbol, now=now)
        return results

    def get_tradeable_symbols(self, bars_dict: Dict[str, pd.DataFrame],
                              now: Optional[datetime] = None
                              ) -> List[str]:
        """Return only symbols that pass quality checks.

        Args:
            bars_dict: Mapping of symbol -> OHLCV DataFrame.
            now: Current time for staleness checks.

        Returns:
            List of symbols with is_tradeable=True.
        """
        scores = self.check_multiple(bars_dict, now=now)
        return [sym for sym, s in scores.items() if s.is_tradeable]

    def stats(self) -> dict:
        """Return framework statistics."""
        return {
            "checks_run": self._checks_run,
            "symbols_flagged": self._symbols_flagged,
            "config": {
                "max_staleness_sec": self.max_staleness_sec,
                "max_single_move": self.max_single_move,
                "min_bars": self.min_bars,
                "min_tradeable_score": self.min_tradeable_score,
            },
        }


# ---------------------------------------------------------------------------
# Convenience: backward-compatible wrapper
# ---------------------------------------------------------------------------

def check_bar_quality(bars: pd.DataFrame, symbol: str,
                      now: Optional[datetime] = None,
                      max_staleness_sec: int = 300,
                      max_single_move: float = 0.15,
                      max_zero_vol_pct: float = 0.10,
                      min_bars: int = 50) -> DataQualityScore:
    """Drop-in replacement for the original data_quality.check_bar_quality.

    Returns a DataQualityScore instead of the old DataQualityResult.
    The DataQualityScore has .is_tradeable (same semantics as .is_clean)
    and .issues list.
    """
    framework = DataQualityFramework(
        max_staleness_sec=max_staleness_sec,
        max_single_move=max_single_move,
        max_zero_vol_pct=max_zero_vol_pct,
        min_bars=min_bars,
        compute_per_bar=False,
    )
    return framework.check(bars, symbol, now=now)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_dq_framework: Optional[DataQualityFramework] = None


def get_quality_framework() -> DataQualityFramework:
    """Get or create the global DataQualityFramework singleton."""
    global _dq_framework
    if _dq_framework is None:
        _dq_framework = DataQualityFramework(
            max_staleness_sec=getattr(config, "DATA_QUALITY_MAX_STALENESS_SEC", 300),
            max_single_move=getattr(config, "DATA_QUALITY_MAX_SINGLE_MOVE", 0.15),
            min_bars=getattr(config, "DATA_QUALITY_MIN_BARS", 50),
        )
    return _dq_framework
