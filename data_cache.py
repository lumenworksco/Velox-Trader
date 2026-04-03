"""V12 shim: data_cache.py -> data.bar_cache (the superior implementation).

Production code should use data.bar_cache.bar_cache (the singleton) or
data.bar_cache.BarCache directly. This module preserves the legacy BarCache
with its simpler .put()/.get_bars()/.cache_stats() API so that existing
callers and tests keep working.

FillMonitor note: FillMonitor was never in this module.
"""

import logging
import threading
from collections import OrderedDict
from datetime import datetime

import pandas as pd

# Re-export the production bar_cache singleton for any code that wants it
from data.bar_cache import bar_cache as _production_bar_cache  # noqa: F401
from data.bar_cache import BarCache as ProductionBarCache  # noqa: F401

logger = logging.getLogger(__name__)


class BarCache:
    """Legacy thread-safe LRU cache for bar data.

    NOTE: The production cache is data.bar_cache.BarCache which has
    pre-warming, monotonic time, and better eviction. This class is kept
    for backward compatibility with code that uses .put()/.get_bars().
    """

    DEFAULT_TTL = {
        "1Min": 60,
        "2Min": 60,
        "5Min": 120,
        "15Min": 300,
        "1Hour": 600,
        "1Day": 3600,
    }

    def __init__(self, max_size: int = 500):
        self._cache: OrderedDict[tuple[str, str], dict] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get_bars(self, symbol: str, timeframe: str, bars: pd.DataFrame | None = None,
                 fetch_fn=None, **fetch_kwargs) -> pd.DataFrame | None:
        """Get cached bars or fetch fresh ones."""
        key = (symbol, timeframe)
        need_fetch = False

        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                ttl = self.DEFAULT_TTL.get(timeframe, 120)
                age = (datetime.now() - entry["last_fetch"]).total_seconds()

                if age < ttl:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return entry["bars"].copy() if entry["bars"] is not None else None

            self._misses += 1
            need_fetch = True

        if not need_fetch:
            return None

        if bars is not None:
            result = bars
        elif fetch_fn is not None:
            try:
                result = fetch_fn(**fetch_kwargs)
            except Exception as e:
                logger.debug(f"Cache fetch failed for {symbol}/{timeframe}: {e}")
                with self._lock:
                    if key in self._cache:
                        return self._cache[key]["bars"].copy()
                return None
        else:
            return None

        with self._lock:
            self._cache[key] = {
                "bars": result,
                "last_fetch": datetime.now(),
            }
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

        return result.copy() if result is not None else None

    def put(self, symbol: str, timeframe: str, bars: pd.DataFrame):
        """Directly store bars in cache."""
        key = (symbol, timeframe)
        with self._lock:
            self._cache[key] = {
                "bars": bars,
                "last_fetch": datetime.now(),
            }
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def invalidate(self, symbol: str, timeframe: str | None = None):
        """Remove cached data for a symbol."""
        with self._lock:
            if timeframe:
                self._cache.pop((symbol, timeframe), None)
            else:
                keys_to_remove = [k for k in self._cache if k[0] == symbol]
                for k in keys_to_remove:
                    del self._cache[k]

    def clear(self):
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def cache_stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
                "symbols_cached": len(set(k[0] for k in self._cache)),
            }


# Module-level singleton (backward-compatible)
_bar_cache: BarCache | None = None
_bar_cache_lock = threading.Lock()


def get_bar_cache() -> BarCache:
    """Get or create the global bar cache singleton (thread-safe)."""
    global _bar_cache
    if _bar_cache is None:
        with _bar_cache_lock:
            if _bar_cache is None:
                _bar_cache = BarCache(max_size=500)
    return _bar_cache
