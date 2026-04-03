"""V9: Signal ranking with multi-dimensional scoring and alpha-weighted expected value.

Scores each signal on regime affinity, historical win rate, and R:R confidence.
Returns signals sorted best-first so the execution layer can prioritise when
capital is constrained.
"""

import logging

import numpy as np
import pandas as pd

from strategies.base import Signal
from analytics.hmm_regime import (
    MarketRegimeState,
    STRATEGY_REGIME_AFFINITY,
)

logger = logging.getLogger(__name__)

# Scoring weights (must sum to 1.0)
_WEIGHTS = {
    "regime":      0.45,    # V12 AUDIT: Increased from 0.30 — HMM regime is most predictive
    "win_rate":    0.25,    # V12 AUDIT: Increased from 0.20 — historical performance matters
    "confidence":  0.30,    # V12 AUDIT: Increased from 0.20 — strategy-assigned conviction
    # V12 AUDIT: Removed placeholder scores (obv, seasonality, liquidity) that returned 0.5
}

# Defaults used when data is unavailable
_DEFAULT_WIN_RATE = 0.50
_DEFAULT_AVG_WIN = 0.01   # 1 %
_DEFAULT_AVG_LOSS = 0.01  # 1 %


class SignalRanker:
    """Score and rank trading signals on multiple dimensions."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self,
        signals: list[Signal],
        regime_probs: dict[MarketRegimeState, float] | None = None,
        trade_history: pd.DataFrame | None = None,
    ) -> list[Signal]:
        """Score each signal, attach score to *reason*, return sorted (best first).

        Scoring dimensions (each normalised to 0-1, then weighted):
          1. Strategy-regime affinity  (45 %)
          2. Historical win rate       (25 %)
          3. Signal confidence / R:R   (30 %)

        Pair signals (non-empty *pair_id*) are kept adjacent after sorting so
        both legs stay together, scored by the max of the pair.
        """
        if not signals:
            return []

        scored: list[tuple[float, Signal]] = []
        for sig in signals:
            score = self._composite_score(sig, regime_probs, trade_history)
            # Append score to reason
            sig.reason = f"{sig.reason} [score={score:.2f}]"
            scored.append((score, sig))

        # Sort by score descending
        scored.sort(key=lambda t: t[0], reverse=True)

        # Re-group pair legs so both legs are adjacent (use max score)
        result = self._regroup_pairs(scored)
        return result

    def get_expected_value(
        self,
        signal: Signal,
        trade_history: pd.DataFrame | None = None,
    ) -> float:
        """EV = (win_rate * avg_win) - ((1 - win_rate) * avg_loss).

        Uses per-symbol-per-strategy stats first, falls back to strategy-level,
        then global defaults.
        """
        win_rate, avg_win, avg_loss = self._trade_stats(
            signal.symbol, signal.strategy, trade_history,
        )
        ev = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)
        return round(ev, 6)

    # ------------------------------------------------------------------
    # Scoring components
    # ------------------------------------------------------------------

    def _composite_score(
        self,
        signal: Signal,
        regime_probs: dict[MarketRegimeState, float] | None,
        trade_history: pd.DataFrame | None,
    ) -> float:
        scores = {
            "regime":      self._regime_score(signal.strategy, regime_probs),
            "win_rate":    self._win_rate_score(signal.symbol, signal.strategy, trade_history),
            "confidence":  self._confidence_score(signal),
        }
        composite = sum(scores[k] * _WEIGHTS[k] for k in _WEIGHTS)
        return round(min(max(composite, 0.0), 1.0), 4)

    @staticmethod
    def _regime_score(
        strategy: str,
        regime_probs: dict[MarketRegimeState, float] | None,
    ) -> float:
        """Weighted regime affinity for this strategy (0-1 normalised)."""
        affinity = STRATEGY_REGIME_AFFINITY.get(strategy)
        if affinity is None or regime_probs is None:
            return 0.5  # neutral default

        raw = sum(affinity.get(r, 1.0) * p for r, p in regime_probs.items())
        # Affinity values range roughly 0.2-1.3; normalise to 0-1
        return min(max(raw / 1.3, 0.0), 1.0)

    def _win_rate_score(
        self,
        symbol: str,
        strategy: str,
        trade_history: pd.DataFrame | None,
    ) -> float:
        """Historical win rate for symbol+strategy, normalised to 0-1."""
        wr, _, _ = self._trade_stats(symbol, strategy, trade_history)
        return min(max(wr, 0.0), 1.0)

    @staticmethod
    def _confidence_score(signal: Signal) -> float:
        """Derived from reward-to-risk ratio (R:R).

        R:R = |take_profit - entry| / |entry - stop_loss|
        Normalised: score = min(rr / 5, 1.0)  (5:1 or better -> 1.0)
        """
        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.entry_price)
        if risk < 1e-8:
            return 0.0
        rr = reward / risk
        return min(rr / 5.0, 1.0)

    @staticmethod
    def _obv_score() -> float:
        """Placeholder — returns neutral 0.5."""
        return 0.5

    @staticmethod
    def _seasonality_score() -> float:
        """Placeholder — returns neutral 0.5."""
        return 0.5

    @staticmethod
    def _liquidity_score() -> float:
        """Placeholder — returns neutral 0.5."""
        return 0.5

    # ------------------------------------------------------------------
    # Trade history helpers
    # ------------------------------------------------------------------

    def _trade_stats(
        self,
        symbol: str,
        strategy: str,
        trade_history: pd.DataFrame | None,
    ) -> tuple[float, float, float]:
        """Return (win_rate, avg_win_pct, avg_loss_pct).

        Tries symbol+strategy first, then strategy-level, then defaults.
        """
        if trade_history is None or trade_history.empty:
            return _DEFAULT_WIN_RATE, _DEFAULT_AVG_WIN, _DEFAULT_AVG_LOSS

        # Ensure pnl_pct column exists
        if "pnl_pct" not in trade_history.columns:
            return _DEFAULT_WIN_RATE, _DEFAULT_AVG_WIN, _DEFAULT_AVG_LOSS

        # Try symbol + strategy
        mask = (trade_history["symbol"] == symbol) & (trade_history["strategy"] == strategy)
        subset = trade_history.loc[mask]

        if len(subset) < 5:
            # Fallback: strategy level
            mask = trade_history["strategy"] == strategy
            subset = trade_history.loc[mask]

        if len(subset) < 5:
            return _DEFAULT_WIN_RATE, _DEFAULT_AVG_WIN, _DEFAULT_AVG_LOSS

        wins = subset[subset["pnl_pct"] > 0]
        losses = subset[subset["pnl_pct"] <= 0]

        win_rate = len(wins) / len(subset) if len(subset) > 0 else _DEFAULT_WIN_RATE
        avg_win = float(wins["pnl_pct"].mean()) if len(wins) > 0 else _DEFAULT_AVG_WIN
        avg_loss = float(abs(losses["pnl_pct"].mean())) if len(losses) > 0 else _DEFAULT_AVG_LOSS

        return win_rate, avg_win, avg_loss

    # ------------------------------------------------------------------
    # Pair regrouping
    # ------------------------------------------------------------------

    @staticmethod
    def _regroup_pairs(scored: list[tuple[float, Signal]]) -> list[Signal]:
        """Keep pair legs adjacent, ordered by the max score of the pair."""
        pair_groups: dict[str, list[tuple[float, Signal]]] = {}
        singles: list[tuple[float, Signal]] = []

        for score, sig in scored:
            if sig.pair_id:
                pair_groups.setdefault(sig.pair_id, []).append((score, sig))
            else:
                singles.append((score, sig))

        # Build result: singles stay in their sorted order
        result: list[Signal] = [sig for _, sig in singles]

        # Insert pair groups at appropriate position based on max score
        pair_entries: list[tuple[float, list[Signal]]] = []
        for pid, items in pair_groups.items():
            max_score = max(s for s, _ in items)
            sigs = [sig for _, sig in items]
            pair_entries.append((max_score, sigs))

        pair_entries.sort(key=lambda t: t[0], reverse=True)

        # Merge singles and pairs by score
        merged: list[tuple[float, list[Signal]]] = []
        for score, sig in singles:
            merged.append((score, [sig]))
        for max_score, sigs in pair_entries:
            merged.append((max_score, sigs))

        merged.sort(key=lambda t: t[0], reverse=True)

        result = []
        for _, sigs in merged:
            result.extend(sigs)

        return result
