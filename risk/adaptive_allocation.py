"""Adaptive Strategy Allocation — dynamically reweight strategies based on performance.

Combines three signals to shift capital toward strategies that are performing
well in the current market regime:

  1. Rolling Sortino ratio per strategy (40%) — reward risk-adjusted returns
  2. HMM regime affinity (30%) — tilt toward strategies that suit the regime
  3. Return correlation penalty (30%) — diversify away from correlated strategies

Constraints:
  - Min 3% per strategy, max 60%
  - Max +/-10% change per rebalance vs. previous weights
  - Weights always sum to 1.0
  - Fail-open: returns base_weights on any error
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config
from analytics.hmm_regime import MarketRegimeState, STRATEGY_REGIME_AFFINITY

logger = logging.getLogger(__name__)


class AdaptiveAllocator:
    """Dynamically adjust strategy allocation weights."""

    def __init__(self, strategies: list[str], base_weights: dict[str, float]):
        self.strategies = list(strategies)
        self.base_weights = dict(base_weights)
        self._previous_weights: dict[str, float] | None = None
        self._change_reasons: list[str] = []

    def compute_weights(
        self,
        trade_history: pd.DataFrame,
        regime_probs: dict[MarketRegimeState, float] | None = None,
    ) -> dict[str, float]:
        """Compute adaptive allocation weights from performance signals.

        Args:
            trade_history: DataFrame with at least columns 'strategy' and 'pnl'.
                Each row is a closed trade.
            regime_probs: Probability distribution across MarketRegimeState.
                If None, regime signal is skipped and its weight redistributed.

        Returns:
            dict mapping strategy name to weight (sums to 1.0).
        """
        try:
            return self._compute_weights_inner(trade_history, regime_probs)
        except Exception as e:
            logger.error(f"Adaptive allocation failed, using base weights: {e}")
            self._change_reasons = [f"Fallback to base weights due to error: {e}"]
            return dict(self.base_weights)

    def _compute_weights_inner(
        self,
        trade_history: pd.DataFrame,
        regime_probs: dict[MarketRegimeState, float] | None,
    ) -> dict[str, float]:
        self._change_reasons = []
        n = len(self.strategies)

        if n == 0:
            return {}

        # Config parameters
        min_weight = getattr(config, "ADAPTIVE_MIN_WEIGHT", 0.03)
        max_weight = getattr(config, "ADAPTIVE_MAX_WEIGHT", 0.60)
        max_daily_change = getattr(config, "ADAPTIVE_MAX_DAILY_CHANGE", 0.10)
        sortino_lookback = getattr(config, "ADAPTIVE_SORTINO_LOOKBACK", 30)
        sortino_w = getattr(config, "ADAPTIVE_SORTINO_WEIGHT", 0.40)
        regime_w = getattr(config, "ADAPTIVE_REGIME_WEIGHT", 0.30)
        corr_w = getattr(config, "ADAPTIVE_CORRELATION_WEIGHT", 0.30)

        # If trade history is empty or missing required columns, return base
        if trade_history is None or trade_history.empty:
            self._change_reasons.append("No trade history available, using base weights")
            return dict(self.base_weights)

        required_cols = {"strategy", "pnl"}
        if not required_cols.issubset(set(trade_history.columns)):
            self._change_reasons.append("Trade history missing required columns")
            return dict(self.base_weights)

        # --- Signal 1: Rolling Sortino per strategy (40%) ---
        sortino_scores = self._compute_sortino_signal(trade_history, sortino_lookback)

        # --- Signal 2: Regime affinity (30%) ---
        regime_scores = self._compute_regime_signal(regime_probs)

        # If regime probs not available, redistribute weight to other signals
        if regime_probs is None:
            effective_sortino_w = sortino_w + regime_w * (sortino_w / (sortino_w + corr_w))
            effective_regime_w = 0.0
            effective_corr_w = corr_w + regime_w * (corr_w / (sortino_w + corr_w))
        else:
            effective_sortino_w = sortino_w
            effective_regime_w = regime_w
            effective_corr_w = corr_w

        # --- Signal 3: Correlation penalty (30%) ---
        corr_scores = self._compute_correlation_signal(trade_history, sortino_lookback)

        # --- Combine signals into raw weights ---
        raw_weights = {}
        for s in self.strategies:
            base = self.base_weights.get(s, 1.0 / n)
            score = (
                sortino_scores.get(s, 1.0) * effective_sortino_w
                + regime_scores.get(s, 1.0) * effective_regime_w
                + corr_scores.get(s, 1.0) * effective_corr_w
            )
            raw_weights[s] = base * score

        # --- Normalize to sum to 1.0 ---
        total = sum(raw_weights.values())
        if total < 1e-10:
            return dict(self.base_weights)
        weights = {s: w / total for s, w in raw_weights.items()}

        # --- Apply min/max constraints ---
        weights = self._apply_bounds(weights, min_weight, max_weight)

        # --- Apply daily change cap ---
        if self._previous_weights is not None:
            weights = self._apply_change_cap(weights, max_daily_change)

        # Store for next call
        self._previous_weights = dict(weights)

        # Log changes
        for s in self.strategies:
            base = self.base_weights.get(s, 0)
            new = weights.get(s, 0)
            diff = new - base
            if abs(diff) > 0.005:
                direction = "increased" if diff > 0 else "decreased"
                self._change_reasons.append(
                    f"{s}: {direction} {abs(diff):.1%} "
                    f"({base:.1%} -> {new:.1%})"
                )

        if not self._change_reasons:
            self._change_reasons.append("Weights unchanged from base allocation")

        return weights

    def _compute_sortino_signal(
        self, trade_history: pd.DataFrame, lookback: int
    ) -> dict[str, float]:
        """Compute per-strategy Sortino scores; higher = better."""
        from analytics.metrics import compute_sortino

        scores = {}
        for s in self.strategies:
            strat_trades = trade_history[trade_history["strategy"] == s]
            if len(strat_trades) < 3:
                scores[s] = 1.0  # Neutral
                continue
            recent = strat_trades.tail(lookback)
            returns = recent["pnl"].values.astype(float)
            # Use per-trade Sortino (not annualized daily)
            sortino = compute_sortino(returns, risk_free_rate=0.0, periods_per_year=1)
            # Clamp to avoid extreme values
            sortino = max(-5.0, min(10.0, sortino))
            # Map to a multiplier: 0 Sortino -> 1.0x, positive -> >1, negative -> <1
            scores[s] = max(0.1, 1.0 + sortino * 0.1)

        # Normalize scores so average = 1.0
        if scores:
            avg = sum(scores.values()) / len(scores)
            if avg > 1e-8:
                scores = {s: v / avg for s, v in scores.items()}

        return scores

    def _compute_regime_signal(
        self, regime_probs: dict[MarketRegimeState, float] | None
    ) -> dict[str, float]:
        """Compute per-strategy regime affinity scores."""
        if regime_probs is None:
            return {s: 1.0 for s in self.strategies}

        scores = {}
        for s in self.strategies:
            affinity = STRATEGY_REGIME_AFFINITY.get(s)
            if affinity is None:
                scores[s] = 1.0
                continue
            # Weighted average affinity across regime probabilities
            weighted = sum(
                affinity.get(regime, 1.0) * prob
                for regime, prob in regime_probs.items()
            )
            scores[s] = max(0.1, weighted)

        # Normalize
        if scores:
            avg = sum(scores.values()) / len(scores)
            if avg > 1e-8:
                scores = {s: v / avg for s, v in scores.items()}

        return scores

    def _compute_correlation_signal(
        self, trade_history: pd.DataFrame, lookback: int
    ) -> dict[str, float]:
        """Penalize strategies whose returns are highly correlated with others.

        Strategies that provide unique/uncorrelated returns get a boost.
        """
        n = len(self.strategies)
        if n < 2:
            return {s: 1.0 for s in self.strategies}

        # Build return series per strategy (align by trade index)
        return_series = {}
        for s in self.strategies:
            strat_trades = trade_history[trade_history["strategy"] == s]
            if len(strat_trades) < 3:
                continue
            recent = strat_trades.tail(lookback)
            return_series[s] = recent["pnl"].values.astype(float)

        if len(return_series) < 2:
            return {s: 1.0 for s in self.strategies}

        # Compute pairwise correlations
        # Use min length for fair comparison
        min_len = min(len(v) for v in return_series.values())
        if min_len < 3:
            return {s: 1.0 for s in self.strategies}

        strat_names = list(return_series.keys())
        trimmed = {s: return_series[s][-min_len:] for s in strat_names}
        matrix = np.array([trimmed[s] for s in strat_names])

        # Correlation matrix
        try:
            corr_matrix = np.corrcoef(matrix)
        except Exception:
            return {s: 1.0 for s in self.strategies}

        # For each strategy, compute average absolute correlation with others
        scores = {}
        for i, s in enumerate(strat_names):
            others_corr = []
            for j in range(len(strat_names)):
                if i != j:
                    c = corr_matrix[i, j]
                    if not np.isnan(c):
                        others_corr.append(abs(c))
            if others_corr:
                avg_corr = np.mean(others_corr)
                # Low correlation -> boost (1.3x), high correlation -> penalty (0.7x)
                scores[s] = max(0.5, 1.3 - 0.6 * avg_corr)
            else:
                scores[s] = 1.0

        # Strategies not in return_series get neutral score
        for s in self.strategies:
            if s not in scores:
                scores[s] = 1.0

        # Normalize
        if scores:
            avg = sum(scores.values()) / len(scores)
            if avg > 1e-8:
                scores = {s: v / avg for s, v in scores.items()}

        return scores

    def _apply_bounds(
        self, weights: dict[str, float], min_w: float, max_w: float
    ) -> dict[str, float]:
        """Enforce min/max weight constraints while maintaining sum = 1.0."""
        n = len(weights)
        if n == 0:
            return weights

        # Iteratively clamp and redistribute
        for _ in range(10):
            clamped = {}
            excess = 0.0
            free_total = 0.0

            for s, w in weights.items():
                if w < min_w:
                    excess += min_w - w
                    clamped[s] = min_w
                elif w > max_w:
                    excess -= w - max_w
                    clamped[s] = max_w
                else:
                    clamped[s] = w
                    free_total += w

            if abs(excess) < 1e-8:
                weights = clamped
                break

            # Redistribute excess proportionally among unclamped strategies
            if free_total > 1e-8:
                for s in clamped:
                    if min_w < clamped[s] < max_w:
                        clamped[s] -= excess * (clamped[s] / free_total)
                        clamped[s] = max(min_w, min(max_w, clamped[s]))

            weights = clamped

        # Final normalization to ensure sum = 1.0
        total = sum(weights.values())
        if total > 1e-8:
            weights = {s: w / total for s, w in weights.items()}

        return weights

    def _apply_change_cap(
        self, weights: dict[str, float], max_change: float
    ) -> dict[str, float]:
        """Cap weight changes to max_change per rebalance period."""
        prev = self._previous_weights
        if prev is None:
            return weights

        capped = {}
        for s in self.strategies:
            new_w = weights.get(s, 0.0)
            old_w = prev.get(s, new_w)
            diff = new_w - old_w
            if abs(diff) > max_change:
                capped[s] = old_w + max_change * (1.0 if diff > 0 else -1.0)
            else:
                capped[s] = new_w

        # Re-normalize after capping
        total = sum(capped.values())
        if total > 1e-8:
            capped = {s: w / total for s, w in capped.items()}

        return capped

    def get_allocation_change_reason(self) -> str:
        """Return a human-readable explanation of the latest weight changes."""
        if not self._change_reasons:
            return "No allocation changes computed yet"
        return "; ".join(self._change_reasons)

    @property
    def previous_weights(self) -> dict[str, float] | None:
        """Return the previous weights for inspection."""
        return self._previous_weights
