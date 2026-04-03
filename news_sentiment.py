"""Alpaca News API sentiment filter for pre-trade screening.

Fetches recent headlines for a symbol, scores them via keyword matching,
and returns a position-size multiplier that the execution layer applies
before submitting orders.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple

import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword dictionaries
# ---------------------------------------------------------------------------

HALT_KEYWORDS = frozenset({
    "bankruptcy", "fraud", "sec charges", "trading halted",
    "fda rejection", "criminal charges", "delisted",
})

POSITIVE_STRONG = frozenset({
    "beat", "beats", "upgrade", "raises guidance", "record",
    "partnership", "buyback", "acquisition approved",
})

NEGATIVE_STRONG = frozenset({
    "miss", "misses", "downgrade", "cuts guidance", "loss",
    "layoffs", "recall", "investigation", "disappoints",
})

# Per-keyword scores
_POSITIVE_SCORE = +2
_NEGATIVE_SCORE = -2


class AlpacaNewsSentiment:
    """Checks recent Alpaca news headlines and returns a sizing multiplier."""

    def __init__(self):
        from alpaca.data.historical.news import NewsClient

        self._client = NewsClient(
            api_key=config.API_KEY,
            secret_key=config.API_SECRET,
        )
        # symbol -> (timestamp, multiplier, reason)
        self._cache: Dict[str, Tuple[datetime, float, str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sentiment_size_mult(self, symbol: str) -> Tuple[float, str]:
        """Return ``(size_multiplier, reason)`` for *symbol*.

        Multiplier values:
            0.0  — halt (catastrophic headline detected)
            0.5  — very negative (net score <= -3)
            0.75 — negative (net score <= -1)
            1.0  — neutral
            1.1  — positive (net score >= 2)
        """
        try:
            cached = self._cache_lookup(symbol)
            if cached is not None:
                return cached

            headlines = self._fetch_headlines(symbol)
            if not headlines:
                result = (1.0, "no_news")
                self._cache_store(symbol, result)
                return result

            result = self._score_headlines(headlines)
            self._cache_store(symbol, result)
            return result

        except Exception as exc:
            log.warning("News sentiment fetch failed for %s: %s", symbol, exc)
            return (1.0, "news_unavailable")

    def get_recent_headlines(self, symbol: str) -> list[str]:
        """Return recent headlines for *symbol* (up to 5, last 6 hours).

        V12 8.1: Added so that FinBERT can score the same headlines that
        the keyword scorer uses.  Returns an empty list on failure.
        """
        try:
            return self._fetch_headlines(symbol)
        except Exception as exc:
            log.debug("get_recent_headlines failed for %s: %s", symbol, exc)
            return []

    def clear_daily_cache(self) -> None:
        """Drop all cached entries (call at start of trading day)."""
        self._cache.clear()
        log.info("News sentiment cache cleared")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cache_lookup(self, symbol: str) -> Tuple[float, str] | None:
        if symbol not in self._cache:
            return None
        ts, mult, reason = self._cache[symbol]
        age_min = (datetime.now(timezone.utc) - ts).total_seconds() / 60
        if age_min > config.NEWS_CACHE_TTL_MIN:
            del self._cache[symbol]
            return None
        return (mult, reason)

    def _cache_store(self, symbol: str, result: Tuple[float, str]) -> None:
        self._cache[symbol] = (datetime.now(timezone.utc), result[0], result[1])

    def _fetch_headlines(self, symbol: str) -> list[str]:
        from alpaca.data.requests import NewsRequest

        start = datetime.now(timezone.utc) - timedelta(hours=6)
        request = NewsRequest(
            symbols=symbol,
            start=start,
            limit=5,
            sort="desc",
        )
        news = self._client.get_news(request)

        headlines: list[str] = []
        if news and hasattr(news, "news"):
            for article in news.news:
                if hasattr(article, "headline") and article.headline:
                    headlines.append(article.headline)
        return headlines

    def _score_headlines(self, headlines: list[str]) -> Tuple[float, str]:
        """Score a list of headline strings and return (multiplier, reason)."""
        net_score = 0
        halt_found = False
        matched_keywords: list[str] = []

        for headline in headlines:
            lower = headline.lower()

            # Check halt keywords first
            for kw in HALT_KEYWORDS:
                if kw in lower:
                    halt_found = True
                    matched_keywords.append(f"HALT:{kw}")
                    break

            if halt_found:
                reason = f"halt_keyword: {', '.join(matched_keywords)}"
                return (0.0, reason)

            # Score positive / negative keywords
            for kw in POSITIVE_STRONG:
                if kw in lower:
                    net_score += _POSITIVE_SCORE
                    matched_keywords.append(f"+{kw}")

            for kw in NEGATIVE_STRONG:
                if kw in lower:
                    net_score += _NEGATIVE_SCORE
                    matched_keywords.append(f"-{kw}")

        # Map net score to multiplier
        kw_str = ", ".join(matched_keywords) if matched_keywords else "none"

        if net_score <= -3:
            return (0.5, f"very_negative (score={net_score}, kw={kw_str})")
        elif net_score <= -1:
            return (0.75, f"negative (score={net_score}, kw={kw_str})")
        elif net_score >= 2:
            return (1.1, f"positive (score={net_score}, kw={kw_str})")
        else:
            return (1.0, f"neutral (score={net_score}, kw={kw_str})")
