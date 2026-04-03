"""Earnings filter — skip trades near earnings announcements.

V12 12.3: Enhanced with auto-fetch via yfinance Ticker.calendar.
- Caches earnings dates (not just booleans) for richer downstream use.
- Persists daily cache to disk to avoid redundant API calls on restart.
- Exposes get_earnings_date() for other modules (gap analysis, etc.).
"""

import json
import logging
import os
from datetime import datetime, date, timedelta
from pathlib import Path

import config

logger = logging.getLogger(__name__)

# Daily cache: symbol -> earnings date string (YYYY-MM-DD) or None
_earnings_dates: dict[str, str | None] = {}
# Boolean cache derived from _earnings_dates (for backward compat)
_earnings_cache: dict[str, bool] = {}
_cache_date: str = ""

# Disk cache path (survives restarts within the same day)
_CACHE_DIR = Path(os.path.dirname(os.path.dirname(__file__))) / "cache"
_CACHE_FILE = _CACHE_DIR / "earnings_dates.json"

# ETFs don't report earnings — skip them to avoid yfinance 404 spam
_KNOWN_ETFS = {
    # Core ETFs
    "SPY", "QQQ", "IWM", "DIA", "ARKK", "GLD", "SLV", "TLT",
    # Sector ETFs
    "XLF", "XLE", "XLK", "XLV", "XLB", "XLI", "XLU", "XLRE", "XLC", "XLP", "XLY",
    "SOXX", "SMH", "IBB",
    # Leveraged / Inverse ETFs
    "SOXL", "TQQQ", "SPXL", "UVXY", "SQQQ", "TNA", "FAS", "FAZ", "LABU", "LABD",
}


def _load_disk_cache() -> dict:
    """Load earnings date cache from disk if it exists and is from today."""
    try:
        if _CACHE_FILE.exists():
            data = json.loads(_CACHE_FILE.read_text())
            if data.get("date") == datetime.now(config.ET).strftime("%Y-%m-%d"):
                return data.get("earnings", {})
    except Exception as e:
        logger.debug("V12 12.3: Failed to load disk earnings cache: %s", e)
    return {}


def _save_disk_cache():
    """Persist earnings date cache to disk."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "date": _cache_date,
            "earnings": _earnings_dates,
        }
        _CACHE_FILE.write_text(json.dumps(data, default=str))
        logger.debug("V12 12.3: Saved earnings cache to disk (%d symbols)", len(_earnings_dates))
    except Exception as e:
        logger.debug("V12 12.3: Failed to save disk earnings cache: %s", e)


def _fetch_earnings_date(symbol: str) -> str | None:
    """Fetch the next earnings date for a symbol via yfinance Ticker.calendar.

    Returns date string (YYYY-MM-DD) or None if not found.
    """
    try:
        import yfinance as yf
        import pandas as pd

        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        if cal is None or (hasattr(cal, 'empty') and cal.empty):
            return None

        # yfinance calendar format varies by version
        earnings_date = None
        if isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.columns:
                earnings_date = cal["Earnings Date"].iloc[0]
            elif len(cal.columns) > 0:
                earnings_date = cal.iloc[0, 0]
        elif isinstance(cal, dict):
            ed_list = cal.get("Earnings Date")
            if ed_list:
                earnings_date = ed_list[0] if isinstance(ed_list, list) else ed_list

        if earnings_date is None:
            return None

        if isinstance(earnings_date, str):
            earnings_date = pd.Timestamp(earnings_date)

        if hasattr(earnings_date, 'date'):
            return str(earnings_date.date())
        return str(earnings_date)

    except Exception:
        return None


def load_earnings_cache(symbols: list[str]):
    """Refresh earnings cache for all symbols. Call once per day.

    V12 12.3: Now fetches and caches actual earnings dates (not just booleans).
    Loads from disk cache first to avoid redundant yfinance calls on restart.
    """
    global _earnings_cache, _earnings_dates, _cache_date

    today = datetime.now(config.ET).strftime("%Y-%m-%d")
    if _cache_date == today and _earnings_cache:
        return  # Already loaded today

    # V12 12.3: Try disk cache first
    disk_cache = _load_disk_cache()
    if disk_cache:
        logger.info("V12 12.3: Loaded %d earnings dates from disk cache", len(disk_cache))
        _earnings_dates = disk_cache
        # Rebuild boolean cache from dates
        _earnings_cache.clear()
        today_date = datetime.now(config.ET).date()
        for sym, edate_str in _earnings_dates.items():
            if edate_str is None:
                _earnings_cache[sym] = False
            else:
                try:
                    edate = date.fromisoformat(edate_str)
                    days_until = (edate - today_date).days
                    _earnings_cache[sym] = 0 <= days_until <= config.EARNINGS_FILTER_DAYS
                except (ValueError, TypeError):
                    _earnings_cache[sym] = False
        # Check if all symbols are covered
        missing = [s for s in symbols if s not in _earnings_dates]
        if not missing:
            _cache_date = today
            excluded = sum(1 for v in _earnings_cache.values() if v)
            logger.info("V12 12.3: Earnings cache fully loaded from disk — %d excluded", excluded)
            return
        # Only fetch missing symbols
        symbols = missing
        logger.info("V12 12.3: Fetching earnings for %d symbols not in disk cache", len(missing))

    logger.info("Loading earnings calendar for %d symbols...", len(symbols))
    excluded = 0

    # Suppress yfinance's noisy internal logging during batch check
    yf_logger = logging.getLogger("yfinance")
    prev_level = yf_logger.level
    yf_logger.setLevel(logging.CRITICAL)

    try:
        for symbol in symbols:
            # ETFs don't have earnings — skip to avoid 404 errors
            if symbol in _KNOWN_ETFS:
                _earnings_dates[symbol] = None
                _earnings_cache[symbol] = False
                continue
            try:
                edate_str = _fetch_earnings_date(symbol)
                _earnings_dates[symbol] = edate_str

                if edate_str is None:
                    _earnings_cache[symbol] = False
                else:
                    today_date = datetime.now(config.ET).date()
                    edate = date.fromisoformat(edate_str)
                    days_until = (edate - today_date).days
                    has_earnings = 0 <= days_until <= config.EARNINGS_FILTER_DAYS
                    _earnings_cache[symbol] = has_earnings
                    if has_earnings:
                        excluded += 1
            except Exception:
                _earnings_dates[symbol] = None
                _earnings_cache[symbol] = False  # If we can't determine, allow
    finally:
        yf_logger.setLevel(prev_level)

    _cache_date = today
    _save_disk_cache()
    logger.info("Earnings filter: %d symbols excluded (%d ETFs auto-skipped)", excluded, len(_KNOWN_ETFS))


def has_earnings_soon(symbol: str) -> bool:
    """Check if symbol has earnings soon. Uses cache if available."""
    if symbol in _earnings_cache:
        return _earnings_cache[symbol]
    # Not in cache — check live (shouldn't happen if cache is loaded)
    edate_str = _fetch_earnings_date(symbol)
    if edate_str is None:
        return False
    try:
        today_date = datetime.now(config.ET).date()
        edate = date.fromisoformat(edate_str)
        days_until = (edate - today_date).days
        return 0 <= days_until <= config.EARNINGS_FILTER_DAYS
    except (ValueError, TypeError):
        return False


def get_earnings_date(symbol: str) -> str | None:
    """V12 12.3: Get the next earnings date for a symbol (YYYY-MM-DD string or None).

    Returns cached value if available, fetches live otherwise.
    """
    if symbol in _earnings_dates:
        return _earnings_dates[symbol]
    # Live fetch for uncached symbol
    edate_str = _fetch_earnings_date(symbol)
    _earnings_dates[symbol] = edate_str
    return edate_str


def get_earnings_calendar() -> dict[str, str]:
    """V12 12.3: Get all cached earnings dates as {symbol: date_string}.

    Only includes symbols with known upcoming earnings dates.
    """
    return {sym: d for sym, d in _earnings_dates.items() if d is not None}


def get_excluded_count() -> int:
    """Get count of symbols excluded due to earnings."""
    return sum(1 for v in _earnings_cache.values() if v)
