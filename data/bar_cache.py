"""T4-001: Shared bar cache across strategies — eliminates ~1,440 redundant API calls/day.

BarCache is a singleton with LRU + TTL eviction. All bar fetches go through
BarCache.get() which returns cached data when fresh and delegates to the
underlying fetcher on cache miss.

Key: (symbol, timeframe_value, lookback_bars)
TTL: 60s for intraday bars, 24h for daily bars.
"""

import logging
import threading
import time as _time
from datetime import datetime, time, timedelta
from typing import Optional

import pandas as pd

try:
    import config as _cfg
    _ET = _cfg.ET
except Exception:
    import pytz as _pytz
    _ET = _pytz.timezone("US/Eastern")

logger = logging.getLogger(__name__)

# TTL constants (seconds)
_INTRADAY_TTL = 60            # 1 minute for intraday data
_DAILY_TTL = 86400            # 24 hours for daily data (after market close)
_DAILY_TTL_INTRADAY = 120    # 2 minutes for daily bars still forming during market hours
_MAX_CACHE_SIZE = 2000        # Max entries before LRU eviction


def _daily_bar_ttl() -> float:
    """Return the appropriate TTL for daily bars.

    During market hours (before 4pm ET), daily bars are still forming so we
    use a short 120-second TTL.  After market close, the full 24-hour TTL
    applies since the bar is final.
    """
    try:
        now_et = datetime.now(_ET).time()
        market_close = time(16, 0)
        if now_et < market_close:
            return _DAILY_TTL_INTRADAY
    except Exception:
        pass
    return _DAILY_TTL


class _CacheEntry:
    """Single cache entry with value, timestamp, and access tracking."""
    __slots__ = ("value", "created_at", "last_access", "ttl")

    def __init__(self, value: pd.DataFrame, ttl: float):
        now = _time.monotonic()
        self.value = value
        self.created_at = now
        self.last_access = now
        self.ttl = ttl

    @property
    def is_expired(self) -> bool:
        return (_time.monotonic() - self.created_at) > self.ttl

    def touch(self):
        self.last_access = _time.monotonic()


class BarCache:
    """Singleton bar cache with LRU + TTL eviction.

    Thread-safe. All bar fetches should go through get() / get_daily() /
    get_intraday() which transparently cache results.

    Usage:
        from data.bar_cache import bar_cache
        df = bar_cache.get_daily("AAPL", days=30)
    """

    _instance: Optional["BarCache"] = None
    _instance_lock = threading.Lock()

    def __init__(self, max_size: int = _MAX_CACHE_SIZE):
        self._cache: dict[tuple, _CacheEntry] = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        logger.info("T4-001: BarCache initialized (max_size=%d)", max_size)

    @classmethod
    def instance(cls) -> "BarCache":
        """Get or create the singleton BarCache."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_daily(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get daily bars, using cache if fresh (TTL=24h).

        Delegates to data.fetcher.get_daily_bars on cache miss.
        """
        key = (symbol, "Day", days)
        return self._get_or_fetch(key, ttl=_daily_bar_ttl(), fetch_fn=self._fetch_daily, symbol=symbol, days=days)

    def get_intraday(self, symbol: str, timeframe, start: datetime,
                     end: datetime | None = None) -> pd.DataFrame:
        """Get intraday bars, using cache if fresh (TTL=60s).

        Delegates to data.fetcher.get_intraday_bars on cache miss.
        """
        # Use start timestamp as part of the key for uniqueness
        tf_str = str(timeframe)
        start_str = start.isoformat() if start else ""
        end_str = end.isoformat() if end else ""
        key = (symbol, tf_str, start_str, end_str)
        return self._get_or_fetch(
            key, ttl=_INTRADAY_TTL, fetch_fn=self._fetch_intraday,
            symbol=symbol, timeframe=timeframe, start=start, end=end,
        )

    def get_bars(self, symbol: str, timeframe, start: datetime,
                 end: datetime | None = None, limit: int | None = None) -> pd.DataFrame:
        """Generic bar fetch with caching.

        TTL is auto-selected: 60s for intraday, 24h for daily.
        """
        tf_str = str(timeframe)
        is_daily = "day" in tf_str.lower()
        ttl = _daily_bar_ttl() if is_daily else _INTRADAY_TTL
        key = (symbol, tf_str, start.isoformat() if start else "", limit or 0)
        return self._get_or_fetch(
            key, ttl=ttl, fetch_fn=self._fetch_bars,
            symbol=symbol, timeframe=timeframe, start=start, end=end, limit=limit,
        )

    def pre_warm(self, symbols: list[str], days: int = 30):
        """Pre-warm cache for all universe symbols (call at scan start).

        Uses multi-symbol fetch for efficiency.
        """
        uncached = []
        for sym in symbols:
            key = (sym, "Day", days)
            with self._lock:
                entry = self._cache.get(key)
                if entry is None or entry.is_expired:
                    uncached.append(sym)

        if not uncached:
            logger.debug("T4-001: Pre-warm skipped — all %d symbols cached", len(symbols))
            return

        logger.info("T4-001: Pre-warming cache for %d/%d symbols", len(uncached), len(symbols))
        try:
            from data.fetcher import get_bars_multi
            from alpaca.data.timeframe import TimeFrame
            import config

            start = datetime.now(config.ET) - timedelta(days=days + 5)
            results = get_bars_multi(uncached, TimeFrame.Day, start=start)

            with self._lock:
                for sym, df in results.items():
                    key = (sym, "Day", days)
                    self._cache[key] = _CacheEntry(df, _DAILY_TTL)

                self._evict_if_needed()

            logger.info("T4-001: Pre-warmed %d symbols into cache", len(results))
        except Exception as e:
            logger.warning("T4-001: Pre-warm failed, strategies will fetch individually: %s", e)

    def invalidate(self, symbol: str | None = None):
        """Invalidate cache entries. If symbol is None, clear all."""
        with self._lock:
            if symbol is None:
                self._cache.clear()
                logger.info("T4-001: Cache fully invalidated")
            else:
                keys_to_remove = [k for k in self._cache if k[0] == symbol]
                for k in keys_to_remove:
                    del self._cache[k]
                if keys_to_remove:
                    logger.debug("T4-001: Invalidated %d entries for %s", len(keys_to_remove), symbol)

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / max(total, 1), 3),
            }

    # --- Internal methods ---

    def _get_or_fetch(self, key: tuple, ttl: float, fetch_fn, **kwargs) -> pd.DataFrame:
        """Check cache; on miss or expiry, call fetch_fn and cache result."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None and not entry.is_expired:
                entry.touch()
                self._hits += 1
                return entry.value.copy()

        # Cache miss — fetch outside lock to avoid blocking other threads
        self._misses += 1
        try:
            df = fetch_fn(**kwargs)
        except Exception as e:
            logger.warning("T4-001: Fetch failed for key %s: %s", key[:2], e)
            return pd.DataFrame()

        with self._lock:
            self._cache[key] = _CacheEntry(df, ttl)
            self._evict_if_needed()

        return df.copy() if not df.empty else df

    def _evict_if_needed(self):
        """Evict expired entries first, then LRU if still over max_size. Must hold _lock."""
        # Phase 1: Remove expired
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for k in expired:
            del self._cache[k]

        # Phase 2: LRU eviction if still over limit
        if len(self._cache) > self._max_size:
            # Sort by last_access ascending, remove oldest
            sorted_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k].last_access)
            to_remove = len(self._cache) - self._max_size
            for k in sorted_keys[:to_remove]:
                del self._cache[k]

    @staticmethod
    def _fetch_daily(symbol: str, days: int) -> pd.DataFrame:
        from data.fetcher import get_daily_bars
        return get_daily_bars(symbol, days=days)

    @staticmethod
    def _fetch_intraday(symbol: str, timeframe, start: datetime,
                        end: datetime | None = None) -> pd.DataFrame:
        from data.fetcher import get_intraday_bars
        return get_intraday_bars(symbol, timeframe, start=start, end=end)

    @staticmethod
    def _fetch_bars(symbol: str, timeframe, start: datetime,
                    end: datetime | None = None, limit: int | None = None) -> pd.DataFrame:
        from data.fetcher import get_bars
        return get_bars(symbol, timeframe, start=start, end=end, limit=limit)


# Module-level singleton accessor
bar_cache = BarCache.instance()
