"""V12 8.1 — FinBERT Local Sentiment Scoring.

Loads ProsusAI/finbert (MIT-licensed) locally via HuggingFace transformers.
Scores Alpaca/Benzinga news headlines per symbol and returns a sentiment
multiplier (0.5-1.2x) used as a conviction filter in the signal processor.

Fail-open: if the model or torch is unavailable, all methods return neutral
(multiplier=1.0) so the trading engine continues without sentiment scoring.

Performance: ~50ms per headline on CPU, batched for efficiency.
"""

import logging
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports — fail-open
# ---------------------------------------------------------------------------

_HAS_TRANSFORMERS = False
_HAS_TORCH = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _HAS_TRANSFORMERS = True
except ImportError:
    pass

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "ProsusAI/finbert"
# FinBERT label mapping: index 0=positive, 1=negative, 2=neutral
LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}


class FinBERTSentimentScorer:
    """V12 8.1: Local FinBERT sentiment scorer for news headlines.

    Loads ProsusAI/finbert on first use and caches per-symbol sentiment
    results for 15 minutes.  Returns a multiplier in [0.5, 1.2] that
    scales position size based on headline sentiment.

    All operations are fail-open: if the model cannot be loaded or
    inference fails, neutral multiplier (1.0) is returned.
    """

    def __init__(self, cache_ttl_min: int = 15):
        self._tokenizer = None
        self._model = None
        self._loaded = False
        self._load_attempted = False
        self._load_error: Optional[str] = None
        self._cache_ttl_sec = cache_ttl_min * 60
        # symbol -> (timestamp, multiplier, reason)
        self._cache: Dict[str, Tuple[float, float, str]] = {}
        # Performance tracking
        self._total_scored = 0
        self._total_errors = 0
        self._avg_latency_ms = 0.0

    # ------------------------------------------------------------------
    # Model loading (lazy, singleton)
    # ------------------------------------------------------------------

    def _load_model(self) -> bool:
        """Load ProsusAI/finbert model and tokenizer. Returns True on success."""
        if self._load_attempted:
            return self._loaded
        self._load_attempted = True

        if not _HAS_TRANSFORMERS or not _HAS_TORCH:
            self._load_error = (
                "transformers or torch not installed — "
                "install with: pip install transformers torch"
            )
            logger.info("V12 8.1: FinBERT unavailable: %s", self._load_error)
            return False

        try:
            logger.info("V12 8.1: Loading ProsusAI/finbert model...")
            start = _time.perf_counter()
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self._model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            self._model.eval()
            elapsed = (_time.perf_counter() - start) * 1000
            self._loaded = True
            logger.info(
                "V12 8.1: FinBERT loaded successfully in %.0fms (device=cpu)",
                elapsed,
            )
            return True
        except Exception as e:
            self._load_error = f"Failed to load FinBERT: {e}"
            logger.warning("V12 8.1: %s", self._load_error)
            return False

    @property
    def is_available(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        if not self._load_attempted:
            self._load_model()
        return self._loaded

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def score_headlines(self, headlines: List[str]) -> List[Dict[str, float]]:
        """Score a batch of headlines using FinBERT.

        Args:
            headlines: List of news headline strings.

        Returns:
            List of dicts with keys: positive, negative, neutral, score.
            score is in [-1.0, 1.0] where positive = bullish, negative = bearish.
            Returns neutral scores if model is unavailable.
        """
        neutral = {"positive": 0.33, "negative": 0.33, "neutral": 0.34, "score": 0.0}
        if not headlines:
            return []
        if not self._load_model():
            return [neutral.copy() for _ in headlines]

        try:
            start = _time.perf_counter()
            inputs = self._tokenizer(
                headlines,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            results = []
            for i in range(len(headlines)):
                p = probs[i].tolist()
                pos_prob = p[0]
                neg_prob = p[1]
                neu_prob = p[2]
                # Composite score: positive - negative, range [-1, 1]
                score = pos_prob - neg_prob
                results.append({
                    "positive": round(pos_prob, 4),
                    "negative": round(neg_prob, 4),
                    "neutral": round(neu_prob, 4),
                    "score": round(score, 4),
                })

            elapsed_ms = (_time.perf_counter() - start) * 1000
            self._total_scored += len(headlines)
            self._avg_latency_ms = (
                0.9 * self._avg_latency_ms + 0.1 * (elapsed_ms / len(headlines))
                if self._avg_latency_ms > 0
                else elapsed_ms / len(headlines)
            )

            logger.debug(
                "V12 8.1: Scored %d headlines in %.0fms (%.1fms/headline)",
                len(headlines), elapsed_ms, elapsed_ms / len(headlines),
            )
            return results

        except Exception as e:
            self._total_errors += 1
            logger.warning("V12 8.1: FinBERT inference failed (fail-open): %s", e)
            return [neutral.copy() for _ in headlines]

    # ------------------------------------------------------------------
    # Trading integration
    # ------------------------------------------------------------------

    def get_sentiment_multiplier(
        self,
        symbol: str,
        headlines: List[str],
    ) -> Tuple[float, str]:
        """Score headlines for a symbol and return a position-size multiplier.

        Args:
            symbol: Ticker symbol (for caching).
            headlines: Recent news headlines for the symbol.

        Returns:
            (multiplier, reason) tuple.
            Multiplier values:
                0.5  — strongly bearish (avg score < -0.4)
                0.7  — bearish (avg score < -0.2)
                1.0  — neutral
                1.1  — mildly bullish (avg score > 0.2)
                1.2  — strongly bullish (avg score > 0.4)
        """
        # Check cache first
        now = _time.time()
        if symbol in self._cache:
            cached_ts, cached_mult, cached_reason = self._cache[symbol]
            if (now - cached_ts) < self._cache_ttl_sec:
                return cached_mult, cached_reason

        if not headlines:
            result = (1.0, "finbert_no_headlines")
            self._cache[symbol] = (now, result[0], result[1])
            return result

        # Model not available — fail-open
        if not self.is_available:
            return 1.0, "finbert_unavailable"

        scores = self.score_headlines(headlines)
        if not scores:
            return 1.0, "finbert_no_scores"

        avg_score = sum(s["score"] for s in scores) / len(scores)

        # Map average score to multiplier
        if avg_score > 0.4:
            mult = 1.2
            reason = f"finbert_strong_bullish_{avg_score:.2f}"
        elif avg_score > 0.2:
            mult = 1.1
            reason = f"finbert_bullish_{avg_score:.2f}"
        elif avg_score < -0.4:
            mult = 0.5
            reason = f"finbert_strong_bearish_{avg_score:.2f}"
        elif avg_score < -0.2:
            mult = 0.7
            reason = f"finbert_bearish_{avg_score:.2f}"
        else:
            mult = 1.0
            reason = f"finbert_neutral_{avg_score:.2f}"

        self._cache[symbol] = (now, mult, reason)

        logger.info(
            "V12 8.1: FinBERT %s — avg_score=%.3f mult=%.1f (%d headlines)",
            symbol, avg_score, mult, len(headlines),
        )
        return mult, reason

    def clear_cache(self) -> None:
        """Clear the sentiment cache (call at start of trading day)."""
        self._cache.clear()
        logger.debug("V12 8.1: FinBERT sentiment cache cleared")

    def get_stats(self) -> Dict[str, float]:
        """Return performance statistics."""
        return {
            "total_scored": self._total_scored,
            "total_errors": self._total_errors,
            "avg_latency_ms": round(self._avg_latency_ms, 1),
            "is_available": self.is_available,
            "cache_size": len(self._cache),
        }


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_instance: Optional[FinBERTSentimentScorer] = None


def get_finbert_scorer() -> FinBERTSentimentScorer:
    """Get or create the module-level FinBERT scorer singleton."""
    global _instance
    if _instance is None:
        _instance = FinBERTSentimentScorer()
    return _instance
