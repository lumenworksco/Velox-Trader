"""Market regime detection — HMM-based with EMA20 fallback.

When HMM_REGIME_ENABLED is True, uses the HMM detector from analytics.hmm_regime
for probabilistic 5-state classification. Falls back to the legacy EMA20 slope
detector if HMM fails or is disabled.
"""

import logging
from datetime import datetime

import pandas_ta as ta

import config
from data import get_daily_bars

logger = logging.getLogger(__name__)

# Lazy imports for HMM — fail-open if not available
_hmm_detector = None
_hmm_import_failed = False


def _get_hmm_detector():
    """Lazy-init the HMM detector (singleton)."""
    global _hmm_detector, _hmm_import_failed
    if _hmm_import_failed:
        return None
    if _hmm_detector is not None:
        return _hmm_detector
    try:
        from analytics.hmm_regime import HMMRegimeDetector
        _hmm_detector = HMMRegimeDetector()
        return _hmm_detector
    except Exception as e:
        logger.warning(f"HMM regime detector unavailable: {e}")
        _hmm_import_failed = True
        return None


class MarketRegime:
    """Market regime detection with HMM + EMA20 fallback.

    Attributes:
        regime: Legacy string regime ("BULLISH", "BEARISH", "UNKNOWN").
        hmm_regime: HMM MarketRegimeState enum (if enabled).
        hmm_probabilities: Probability distribution across HMM states.
    """

    def __init__(self):
        self.regime: str = "UNKNOWN"
        self.last_check: datetime | None = None
        self.spy_price: float = 0.0
        self.spy_ema: float = 0.0

        # HMM state
        self.hmm_regime = None
        self.hmm_probabilities: dict = {}
        self._hmm_last_fit: datetime | None = None

    def update(self, now: datetime) -> str:
        """Update regime detection. Returns legacy string regime.

        If HMM is enabled and fitted, also updates hmm_regime and
        hmm_probabilities attributes.
        """
        if (
            self.last_check is not None
            and (now - self.last_check).total_seconds() < config.REGIME_CHECK_INTERVAL_MIN * 60
        ):
            return self.regime

        # BUG-FIX: VIX override — if VIX > 30, force BEARISH regardless
        # of HMM or EMA.  During tariff-panic/crash scenarios, the HMM
        # can remain stuck on a stale BULLISH classification (trained on
        # older data) and the EMA20 is too slow to react to intraday
        # crashes.  VIX > 30 is unambiguous high-fear territory.
        try:
            from risk import get_vix_level
            vix = get_vix_level()
            if vix > 30:
                self.regime = "BEARISH"
                self.last_check = now
                # Also update HMM state if available.
                # V12 HOTFIX (CodeQL): explicit fail-open — HMM is optional,
                # so a missing module should not break the VIX override path.
                try:
                    from analytics.hmm_regime import MarketRegimeState
                    self.hmm_regime = MarketRegimeState.HIGH_VOL_BEAR
                except ImportError as e:
                    logger.debug(
                        "HMM MarketRegimeState unavailable during VIX override; "
                        "skipping hmm_regime sync: %s", e,
                    )
                logger.warning(
                    "VIX override: VIX=%.1f > 30 — forcing regime to BEARISH",
                    vix,
                )
                return self.regime
        except Exception as e:
            logger.debug("VIX override check failed (non-fatal): %s", e)

        # Try HMM first
        hmm_enabled = getattr(config, "HMM_REGIME_ENABLED", False)
        if hmm_enabled:
            try:
                regime_str = self._update_hmm(now)
                if regime_str is not None:
                    self.regime = regime_str
                    self.last_check = now
                    return self.regime
            except Exception as e:
                logger.warning(f"HMM regime update failed, falling back to EMA: {e}")

        # Fallback: EMA20 slope
        return self._update_ema(now)

    def _update_hmm(self, now: datetime) -> str | None:
        """Update regime via HMM detector. Returns legacy string or None on failure."""
        from analytics.hmm_regime import map_hmm_to_legacy, MarketRegimeState

        detector = _get_hmm_detector()
        if detector is None:
            return None

        # Check if model needs fitting/refitting (weekly on Sunday)
        retrain_day = getattr(config, "HMM_RETRAIN_DAY", "sunday")
        day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        target_weekday = day_names.index(retrain_day.lower()) if retrain_day.lower() in day_names else 6

        needs_fit = (
            not detector.is_fitted
            or (now.weekday() == target_weekday and self._hmm_last_fit != now.date())
        )

        if needs_fit:
            try:
                training_years = getattr(config, "HMM_TRAINING_YEARS", 3)
                spy_data = get_daily_bars("SPY", days=training_years * 252 + 30)
                if spy_data is not None and len(spy_data) >= 200:
                    detector.fit(spy_data)
                    self._hmm_last_fit = now.date()
                    logger.info("HMM regime model retrained")
            except Exception as e:
                logger.warning(f"HMM training failed: {e}")
                if not detector.is_fitted:
                    return None

        if not detector.is_fitted:
            return None

        # Predict current regime using recent data
        try:
            recent_data = get_daily_bars("SPY", days=30)
            if recent_data is None or len(recent_data) < 25:
                return None

            regime, probs = detector.predict_regime(recent_data)
            self.hmm_regime = regime
            self.hmm_probabilities = probs

            # Also update SPY price for dashboard
            self.spy_price = float(recent_data["close"].iloc[-1])

            # Map to legacy string
            legacy = map_hmm_to_legacy(regime)
            min_prob = getattr(config, "HMM_MIN_PROBABILITY", 0.4)
            max_prob = max(probs.values()) if probs else 0.0

            if max_prob < min_prob:
                # Uncertain — treat as UNKNOWN for conservative sizing
                legacy = "UNKNOWN"

            logger.info(
                f"HMM regime: {regime.value} (p={max_prob:.2f}) → {legacy} | "
                f"probs: {', '.join(f'{r.value}={p:.2f}' for r, p in sorted(probs.items(), key=lambda x: -x[1]))}"
            )
            return legacy

        except Exception as e:
            logger.warning(f"HMM prediction failed: {e}")
            return None

    def _update_ema(self, now: datetime) -> str:
        """Legacy EMA20 slope regime detection."""
        try:
            df = get_daily_bars("SPY", days=config.REGIME_EMA_PERIOD + 5)
            if df.empty or len(df) < config.REGIME_EMA_PERIOD:
                # V10 BUG-018: Default to UNKNOWN (conservative) instead of BULLISH
                logger.warning("Not enough SPY data for regime check, defaulting to UNKNOWN")
                self.regime = "UNKNOWN"
                self.last_check = now
                return self.regime

            ema = ta.ema(df["close"], length=config.REGIME_EMA_PERIOD)
            if ema is None or ema.empty:
                self.regime = "UNKNOWN"
                self.last_check = now
                return self.regime

            self.spy_price = df["close"].iloc[-1]
            self.spy_ema = ema.iloc[-1]

            self.regime = "BULLISH" if self.spy_price > self.spy_ema else "BEARISH"
            self.last_check = now
            logger.info(f"Market regime (EMA): {self.regime} (SPY={self.spy_price:.2f}, EMA={self.spy_ema:.2f})")

        except Exception as e:
            logger.error(f"Regime check failed: {e}")
            # V10: Keep UNKNOWN on exception (don't default to BULLISH)

        return self.regime

    def is_spy_positive_today(self) -> bool:
        """Check if SPY is positive on the day (for momentum filter)."""
        try:
            df = get_daily_bars("SPY", days=2)
            if df.empty or len(df) < 2:
                return False
            return df["close"].iloc[-1] > df["close"].iloc[-2]
        except Exception:
            return False

    def get_regime_affinity(self, strategy: str) -> float:
        """Get the regime affinity multiplier for a strategy.

        Returns 1.0 if HMM is not enabled/fitted.
        """
        if self.hmm_regime is None:
            return 1.0
        try:
            from analytics.hmm_regime import get_regime_size_multiplier
            return get_regime_size_multiplier(
                strategy, self.hmm_regime, self.hmm_probabilities
            )
        except Exception:
            return 1.0
