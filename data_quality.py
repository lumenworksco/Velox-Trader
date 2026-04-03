"""V12 shim: data_quality.py -> data/quality.py (the comprehensive implementation).

All functionality now lives in data/quality.py which has per-bar scoring,
corporate action detection, and a full DataQualityFramework. This module
provides backward-compatible wrappers so existing imports keep working.
"""

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

# Re-export the comprehensive framework for any code that wants it
from data.quality import (  # noqa: F401
    DataQualityFramework,
    DataQualityScore,
    QualityIssue,
    get_quality_framework,
    get_invalidated_symbols,
    clear_invalidation,
)


@dataclass
class DataQualityResult:
    """Backward-compatible result type matching the original V8 API.

    Maps to DataQualityScore internally but exposes .is_clean and
    .issues as a list of plain strings (not QualityIssue objects).
    """
    is_clean: bool
    issues: list[str] = field(default_factory=list)

    def add_issue(self, issue: str):
        self.issues.append(issue)
        self.is_clean = False


def check_bar_quality(bars: pd.DataFrame, symbol: str,
                      now: datetime | None = None,
                      max_staleness_sec: int = 300,
                      max_single_move: float = 0.15,
                      max_zero_vol_pct: float = 0.10,
                      min_bars: int = 50) -> DataQualityResult:
    """Backward-compatible wrapper around data.quality.DataQualityFramework.

    Returns a DataQualityResult with .is_clean and .issues (list of strings),
    matching the original V8 API that existing tests and callers expect.
    """
    from data.quality import DataQualityFramework as DQF

    framework = DQF(
        max_staleness_sec=max_staleness_sec,
        max_single_move=max_single_move,
        max_zero_vol_pct=max_zero_vol_pct,
        min_bars=min_bars,
        compute_per_bar=False,
    )
    score = framework.check(bars, symbol, now=now)

    # Convert DataQualityScore -> DataQualityResult with string issues.
    # The old API set is_clean=False if ANY issues were found, whereas the
    # new DataQualityScore.is_tradeable only goes False for critical issues
    # or low overall score.  Preserve the old stricter behavior here.
    has_issues = len(score.issues) > 0
    result = DataQualityResult(is_clean=not has_issues)
    for qi in score.issues:
        result.issues.append(qi.message)

    return result
