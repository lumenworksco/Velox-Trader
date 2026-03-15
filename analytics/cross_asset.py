"""Cross-Asset Signal Layer — macro regime detection via intermarket analysis.

Monitors VIX, TLT, HYG, UUP, GLD and VIX3M to produce a composite
risk-appetite score, term-structure signal, dollar regime, credit stress
level, and flight-to-safety flag.  All methods are fail-open: if data is
unavailable the module returns neutral defaults so it never blocks trading.
"""

import logging
import time as _time
from datetime import datetime, timedelta

import numpy as np

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instrument mapping (Yahoo Finance tickers)
# ---------------------------------------------------------------------------
INSTRUMENTS = {
    "vix": "^VIX",
    "tlt": "TLT",
    "hyg": "HYG",
    "uup": "UUP",
    "gld": "GLD",
    "vix3m": "^VIX3M",
}

# Neutral defaults returned when data is unavailable
_NEUTRAL_SIGNALS = {
    "risk_appetite": 0.0,
    "vix_term_structure": "contango",
    "dollar_regime": "neutral",
    "credit_stress": 0.0,
    "flight_to_safety": False,
}


class CrossAssetMonitor:
    """Fetch and interpret cross-asset macro signals.

    Usage::

        monitor = CrossAssetMonitor()
        monitor.update(datetime.now())
        signals = monitor.compute_signals()
        bias = monitor.get_equity_bias()
    """

    def __init__(self):
        self._cache: dict[str, dict] = {}  # ticker -> {"hist": DataFrame, "price": float}
        self._last_update: float = 0.0  # epoch seconds

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def update(self, now: datetime) -> None:
        """Refresh instrument data if the cache has expired.

        Uses *yfinance* to pull recent daily bars.  Any fetch error is
        logged and swallowed (fail-open).
        """
        elapsed = _time.time() - self._last_update
        if elapsed < config.CROSS_ASSET_UPDATE_INTERVAL and self._cache:
            return  # cache still valid

        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed — cross-asset signals disabled")
            return

        for key, ticker in INSTRUMENTS.items():
            try:
                data = yf.download(
                    ticker,
                    period="25d",
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )
                if data is None or data.empty:
                    logger.debug("No data returned for %s", ticker)
                    continue

                # Flatten MultiIndex columns if present (yfinance >= 0.2.36)
                if hasattr(data.columns, 'levels') and data.columns.nlevels > 1:
                    data.columns = data.columns.get_level_values(0)

                self._cache[key] = {
                    "hist": data,
                    "price": float(data["Close"].iloc[-1]),
                }
            except Exception as exc:
                logger.debug("Failed to fetch %s (%s): %s", key, ticker, exc)

        self._last_update = _time.time()

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def compute_signals(self) -> dict:
        """Derive cross-asset signals from cached data.

        Returns a dict with keys:
            risk_appetite      float  -1.0 … +1.0
            vix_term_structure str    "contango" | "backwardation"
            dollar_regime      str    "strengthening" | "weakening" | "neutral"
            credit_stress      float  0.0 … 1.0
            flight_to_safety   bool
        """
        if not self._cache:
            return dict(_NEUTRAL_SIGNALS)

        try:
            risk_appetite = self._compute_risk_appetite()
            vix_ts = self._compute_vix_term_structure()
            dollar = self._compute_dollar_regime()
            credit = self._compute_credit_stress()
            flight = self._compute_flight_to_safety()

            return {
                "risk_appetite": risk_appetite,
                "vix_term_structure": vix_ts,
                "dollar_regime": dollar,
                "credit_stress": credit,
                "flight_to_safety": flight,
            }
        except Exception as exc:
            logger.warning("compute_signals failed: %s — returning neutral", exc)
            return dict(_NEUTRAL_SIGNALS)

    # ------------------------------------------------------------------
    # Individual signal helpers
    # ------------------------------------------------------------------

    def _safe_return(self, key: str, periods: int = 1) -> float | None:
        """Compute percentage return over *periods* days for *key*.

        Returns None if data is missing or insufficient.
        """
        entry = self._cache.get(key)
        if entry is None:
            return None
        hist = entry["hist"]
        if hist is None or len(hist) < periods + 1:
            return None
        try:
            close = hist["Close"]
            return float((close.iloc[-1] - close.iloc[-1 - periods]) / close.iloc[-1 - periods])
        except Exception:
            return None

    def _compute_risk_appetite(self) -> float:
        """Composite risk-appetite score (-1 to +1).

        Positive = risk-on:  TLT falling + HYG rising + VIX falling + USD falling.
        Negative = risk-off: opposite.
        """
        components: list[float] = []

        tlt_ret = self._safe_return("tlt", 5)
        if tlt_ret is not None:
            # TLT falling = risk-on → positive contribution
            components.append(-tlt_ret * 10)

        hyg_ret = self._safe_return("hyg", 5)
        if hyg_ret is not None:
            # HYG rising = risk-on → positive
            components.append(hyg_ret * 20)

        vix_entry = self._cache.get("vix")
        if vix_entry is not None:
            vix_level = vix_entry["price"]
            # VIX below 20 = risk-on, above 30 = risk-off
            vix_score = (25 - vix_level) / 15  # 10→+1, 25→0, 40→-1
            components.append(vix_score)

        uup_ret = self._safe_return("uup", 5)
        if uup_ret is not None:
            # USD falling = risk-on → positive
            components.append(-uup_ret * 15)

        if not components:
            return 0.0

        raw = float(np.mean(components))
        return float(np.clip(raw, -1.0, 1.0))

    def _compute_vix_term_structure(self) -> str:
        """'contango' if VIX < VIX3M, else 'backwardation'."""
        vix_entry = self._cache.get("vix")
        vix3m_entry = self._cache.get("vix3m")
        if vix_entry is None or vix3m_entry is None:
            return "contango"  # neutral default
        if vix_entry["price"] > vix3m_entry["price"]:
            return "backwardation"
        return "contango"

    def _compute_dollar_regime(self) -> str:
        """UUP 5-day return → 'strengthening' / 'weakening' / 'neutral'."""
        ret = self._safe_return("uup", 5)
        if ret is None:
            return "neutral"
        if ret > 0.005:
            return "strengthening"
        if ret < -0.005:
            return "weakening"
        return "neutral"

    def _compute_credit_stress(self) -> float:
        """HYG drawdown from 20-day high normalised by ATR.  0–1 range."""
        entry = self._cache.get("hyg")
        if entry is None:
            return 0.0
        hist = entry["hist"]
        if hist is None or len(hist) < 15:
            return 0.0
        try:
            close = hist["Close"]
            high_20 = close.rolling(20, min_periods=5).max().iloc[-1]
            current = close.iloc[-1]
            drawdown = float(high_20 - current)

            # ATR (14-period, or whatever length available)
            hi = hist["High"]
            lo = hist["Low"]
            tr = np.maximum(hi - lo, np.abs(hi - close.shift(1)))
            tr = np.maximum(tr, np.abs(lo - close.shift(1)))
            atr = float(tr.rolling(14, min_periods=5).mean().iloc[-1])

            if atr < 1e-6:
                return 0.0

            stress = drawdown / atr
            return float(np.clip(stress, 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_flight_to_safety(self) -> bool:
        """True if TLT up > 1% AND GLD up > 0.5% AND VIX up > 10% today."""
        tlt_ret = self._safe_return("tlt", 1)
        gld_ret = self._safe_return("gld", 1)
        vix_ret = self._safe_return("vix", 1)

        if tlt_ret is None or gld_ret is None or vix_ret is None:
            return False

        return tlt_ret > 0.01 and gld_ret > 0.005 and vix_ret > 0.10

    # ------------------------------------------------------------------
    # Equity bias — single composite score
    # ------------------------------------------------------------------

    def get_equity_bias(self) -> float:
        """Single -1.0 to +1.0 score used as a position-size multiplier.

        Maps to effective sizing via:  ``0.5 + (bias * 0.5)``
        giving a range of 0.0 (fully risk-off) to 1.0 (fully risk-on).
        """
        signals = self.compute_signals()

        components: list[float] = []

        # 1. Risk appetite (weight 40%)
        components.append(signals["risk_appetite"] * 0.40)

        # 2. VIX term structure (weight 20%)
        ts_score = 0.2 if signals["vix_term_structure"] == "contango" else -0.2
        components.append(ts_score)

        # 3. Credit stress (weight 20%) — higher stress → negative
        components.append(-signals["credit_stress"] * 0.20)

        # 4. Flight to safety override (weight 20%)
        if signals["flight_to_safety"]:
            components.append(-0.20)
        else:
            components.append(0.0)

        # 5. Dollar regime (small nudge)
        dr = signals["dollar_regime"]
        if dr == "weakening":
            components.append(0.05)
        elif dr == "strengthening":
            components.append(-0.05)

        raw = sum(components)
        return float(np.clip(raw, -1.0, 1.0))
