"""V12 12.2: Overnight Gap Analysis.

At 9:25 AM ET, computes overnight gaps for all symbols in the universe:
- 1-3% gaps  -> mean reversion candidates (gap fill strategy)
- >3% gaps   -> breakout candidates

Results are cached for the trading day and exposed via get_gap_flags().
"""

import logging
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import config

logger = logging.getLogger(__name__)


class GapType(str, Enum):
    """Classification of overnight gap magnitude."""
    NONE = "none"           # < 1% gap
    MR_CANDIDATE = "mr"    # 1-3% gap -> mean reversion / gap fill
    BREAKOUT = "breakout"  # > 3% gap -> momentum / breakout


@dataclass
class GapInfo:
    """Gap analysis result for a single symbol."""
    symbol: str
    gap_pct: float          # Signed gap percentage (positive = gap up)
    gap_type: GapType
    prev_close: float       # Previous session close
    open_price: float       # Current session open (or pre-market indicative)
    timestamp: datetime | None = None


# Module-level cache (thread-safe)
_gap_lock = threading.Lock()
_gap_cache: dict[str, GapInfo] = {}
_gap_cache_date: str = ""


def compute_gaps(symbols: list[str], now: datetime | None = None) -> dict[str, GapInfo]:
    """Compute overnight gaps for all symbols.

    Should be called around 9:25 AM ET (before market open) for best results.
    Uses previous day's close and current snapshot (pre-market or opening price).

    Args:
        symbols: List of ticker symbols to analyze.
        now: Current datetime (defaults to now in ET).

    Returns:
        Dict of symbol -> GapInfo for all symbols with gaps >= 1%.
    """
    global _gap_cache, _gap_cache_date

    if now is None:
        now = datetime.now(config.ET)

    today = now.strftime("%Y-%m-%d")

    # Return cached results if already computed today
    with _gap_lock:
        if _gap_cache_date == today and _gap_cache:
            return _gap_cache

    logger.info("V12 12.2: Computing overnight gaps for %d symbols...", len(symbols))
    gaps: dict[str, GapInfo] = {}

    try:
        from data.fetcher import get_daily_bars, get_snapshots_batch

        # Batch-fetch snapshots for current prices
        snapshots = get_snapshots_batch(symbols)

        for symbol in symbols:
            try:
                # Get previous close from daily bars
                daily = get_daily_bars(symbol, days=5)
                if daily is None or daily.empty or len(daily) < 1:
                    continue

                prev_close = float(daily["close"].iloc[-1])
                if prev_close <= 0:
                    continue

                # Get current/pre-market price from snapshot
                snap = snapshots.get(symbol)
                if snap is None:
                    continue

                # Use latest trade price; fall back to daily bar open
                open_price = None
                if hasattr(snap, "latest_trade") and snap.latest_trade:
                    open_price = float(snap.latest_trade.price)
                elif hasattr(snap, "daily_bar") and snap.daily_bar:
                    open_price = float(snap.daily_bar.open)

                if open_price is None or open_price <= 0:
                    continue

                gap_pct = (open_price - prev_close) / prev_close

                # Classify gap
                abs_gap = abs(gap_pct)
                if abs_gap < 0.01:
                    gap_type = GapType.NONE
                elif abs_gap <= 0.03:
                    gap_type = GapType.MR_CANDIDATE
                else:
                    gap_type = GapType.BREAKOUT

                info = GapInfo(
                    symbol=symbol,
                    gap_pct=gap_pct,
                    gap_type=gap_type,
                    prev_close=prev_close,
                    open_price=open_price,
                    timestamp=now,
                )

                if gap_type != GapType.NONE:
                    gaps[symbol] = info

            except Exception as e:
                logger.debug("V12 12.2: Gap calc failed for %s: %s", symbol, e)

    except ImportError as e:
        logger.warning("V12 12.2: Gap analysis import failed: %s", e)
        return gaps
    except Exception as e:
        logger.warning("V12 12.2: Gap analysis batch failed: %s", e)
        return gaps

    # Cache results
    with _gap_lock:
        _gap_cache = gaps
        _gap_cache_date = today

    mr_count = sum(1 for g in gaps.values() if g.gap_type == GapType.MR_CANDIDATE)
    bo_count = sum(1 for g in gaps.values() if g.gap_type == GapType.BREAKOUT)
    logger.info(
        "V12 12.2: Gap analysis complete — %d gapped symbols "
        "(%d MR candidates, %d breakout candidates)",
        len(gaps), mr_count, bo_count,
    )
    return gaps


def get_gap_flags() -> dict[str, GapInfo]:
    """Get cached gap analysis results for the current day.

    Returns empty dict if compute_gaps() hasn't been called yet today.
    """
    with _gap_lock:
        return dict(_gap_cache)


def get_gap_info(symbol: str) -> GapInfo | None:
    """Get gap info for a specific symbol, or None if no significant gap."""
    with _gap_lock:
        return _gap_cache.get(symbol)


def get_mr_candidates() -> list[GapInfo]:
    """Get all mean reversion gap candidates (1-3% gaps)."""
    with _gap_lock:
        return [g for g in _gap_cache.values() if g.gap_type == GapType.MR_CANDIDATE]


def get_breakout_candidates() -> list[GapInfo]:
    """Get all breakout gap candidates (>3% gaps)."""
    with _gap_lock:
        return [g for g in _gap_cache.values() if g.gap_type == GapType.BREAKOUT]


def clear_cache():
    """Clear the gap analysis cache."""
    global _gap_cache, _gap_cache_date
    with _gap_lock:
        _gap_cache = {}
        _gap_cache_date = ""
