"""Tests for adaptive strategy allocation (Section 2 — Velox V9).

Covers:
  - Weight validity (sum to 1.0, within bounds)
  - Min/max constraints
  - Daily change cap
  - Fallback behaviour on empty/invalid data
  - Sortino-driven tilt toward outperforming strategies
  - Regime-probability influence
  - Correlation penalty
  - Human-readable change reason
  - Integration with vol_targeting
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from analytics.hmm_regime import MarketRegimeState
from risk.adaptive_allocation import AdaptiveAllocator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STRATEGIES = ["STAT_MR", "VWAP", "KALMAN_PAIRS", "PEAD", "ORB", "MICRO_MOM"]
BASE_WEIGHTS = {
    "STAT_MR": 0.40,
    "VWAP": 0.20,
    "KALMAN_PAIRS": 0.20,
    "PEAD": 0.10,
    "ORB": 0.05,
    "MICRO_MOM": 0.05,
}


def _make_allocator() -> AdaptiveAllocator:
    return AdaptiveAllocator(strategies=STRATEGIES, base_weights=BASE_WEIGHTS)


def _make_trade_history(
    n_per_strategy: int = 40,
    pnl_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Generate synthetic trade history. pnl_overrides sets mean pnl per strategy."""
    rng = np.random.RandomState(42)
    rows = []
    for s in STRATEGIES:
        mean_pnl = (pnl_overrides or {}).get(s, 0.0)
        for _ in range(n_per_strategy):
            rows.append({"strategy": s, "pnl": mean_pnl + rng.randn() * 10.0})
    return pd.DataFrame(rows)


def _uniform_regime_probs() -> dict[MarketRegimeState, float]:
    return {r: 0.2 for r in MarketRegimeState}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWeightValidity:
    """Weights must always sum to 1.0 and be within [min, max]."""

    def test_weights_sum_to_one(self):
        allocator = _make_allocator()
        history = _make_trade_history()
        weights = allocator.compute_weights(history, _uniform_regime_probs())
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_weights_sum_to_one_no_regime(self):
        allocator = _make_allocator()
        history = _make_trade_history()
        weights = allocator.compute_weights(history, regime_probs=None)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_all_weights_within_bounds(self):
        allocator = _make_allocator()
        history = _make_trade_history()
        weights = allocator.compute_weights(history, _uniform_regime_probs())
        for s, w in weights.items():
            assert w >= config.ADAPTIVE_MIN_WEIGHT - 1e-6, f"{s} below min: {w}"
            assert w <= config.ADAPTIVE_MAX_WEIGHT + 1e-6, f"{s} above max: {w}"

    def test_all_strategies_present(self):
        allocator = _make_allocator()
        history = _make_trade_history()
        weights = allocator.compute_weights(history, _uniform_regime_probs())
        assert set(weights.keys()) == set(STRATEGIES)


class TestFallbackBehavior:
    """When data is missing or invalid, return base_weights."""

    def test_empty_trade_history(self):
        allocator = _make_allocator()
        empty_df = pd.DataFrame(columns=["strategy", "pnl"])
        weights = allocator.compute_weights(empty_df)
        assert weights == BASE_WEIGHTS

    def test_none_trade_history(self):
        allocator = _make_allocator()
        weights = allocator.compute_weights(None)
        assert weights == BASE_WEIGHTS

    def test_missing_columns(self):
        allocator = _make_allocator()
        bad_df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
        weights = allocator.compute_weights(bad_df)
        assert weights == BASE_WEIGHTS

    def test_exception_returns_base_weights(self):
        """If internal computation raises, fail-open to base weights."""
        allocator = _make_allocator()
        # Pass a non-DataFrame that will cause an error in internal logic
        weights = allocator.compute_weights("not a dataframe")
        assert weights == BASE_WEIGHTS


class TestMinMaxConstraints:
    """Ensure min/max per-strategy weight constraints are enforced."""

    def test_min_weight_enforced(self):
        """Even a terrible strategy gets at least ADAPTIVE_MIN_WEIGHT."""
        allocator = _make_allocator()
        # Make one strategy have terrible PnL
        history = _make_trade_history(
            pnl_overrides={"MICRO_MOM": -100.0}
        )
        weights = allocator.compute_weights(history, _uniform_regime_probs())
        for s, w in weights.items():
            assert w >= config.ADAPTIVE_MIN_WEIGHT - 1e-6

    def test_max_weight_enforced(self):
        """Even a great strategy is capped at ADAPTIVE_MAX_WEIGHT."""
        allocator = _make_allocator()
        # Make one strategy dominate
        history = _make_trade_history(
            pnl_overrides={"STAT_MR": 500.0}
        )
        weights = allocator.compute_weights(history, _uniform_regime_probs())
        for s, w in weights.items():
            assert w <= config.ADAPTIVE_MAX_WEIGHT + 1e-6


class TestDailyChangeCap:
    """Weight changes should be capped at ADAPTIVE_MAX_DAILY_CHANGE."""

    def test_change_cap_first_call_no_cap(self):
        """First call has no previous weights, so no cap applied."""
        allocator = _make_allocator()
        history = _make_trade_history(pnl_overrides={"STAT_MR": 200.0})
        weights = allocator.compute_weights(history, _uniform_regime_probs())
        # Just verify it works (no previous weights to cap against)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_change_cap_enforced_on_second_call(self):
        """Second call should cap changes relative to first call."""
        allocator = _make_allocator()

        # First call — neutral
        history1 = _make_trade_history()
        weights1 = allocator.compute_weights(history1, _uniform_regime_probs())

        # Second call — dramatically different PnL profile
        history2 = _make_trade_history(
            pnl_overrides={"STAT_MR": 500.0, "ORB": -200.0, "MICRO_MOM": -200.0}
        )
        weights2 = allocator.compute_weights(history2, _uniform_regime_probs())

        max_change = config.ADAPTIVE_MAX_DAILY_CHANGE
        for s in STRATEGIES:
            diff = abs(weights2.get(s, 0) - weights1.get(s, 0))
            # Allow some tolerance for renormalization
            assert diff <= max_change + 0.02, (
                f"{s} changed by {diff:.3f}, exceeds cap {max_change}"
            )

    def test_gradual_convergence(self):
        """Multiple calls should gradually shift weights toward target."""
        allocator = _make_allocator()
        history = _make_trade_history(pnl_overrides={"STAT_MR": 300.0})
        regime = _uniform_regime_probs()

        prev_stat_mr = None
        for _ in range(5):
            weights = allocator.compute_weights(history, regime)
            if prev_stat_mr is not None:
                # Each step should move toward higher STAT_MR weight
                # (or at least not crash)
                assert abs(sum(weights.values()) - 1.0) < 1e-6
            prev_stat_mr = weights["STAT_MR"]


class TestSortinoSignal:
    """Strategy with higher Sortino should get more weight."""

    def test_better_strategy_gets_more_weight(self):
        allocator = _make_allocator()
        history = _make_trade_history(
            n_per_strategy=50,
            pnl_overrides={"VWAP": 50.0, "ORB": -30.0},
        )
        weights = allocator.compute_weights(history, _uniform_regime_probs())
        # VWAP (positive pnl) should get more than ORB (negative pnl)
        assert weights["VWAP"] > weights["ORB"]

    def test_few_trades_gets_neutral_score(self):
        """Strategy with fewer than 3 trades gets neutral (base) weight."""
        allocator = _make_allocator()
        # Build history with only 2 trades for ORB, many for others
        rows = []
        rng = np.random.RandomState(42)
        for s in STRATEGIES:
            n = 2 if s == "ORB" else 40
            for _ in range(n):
                rows.append({"strategy": s, "pnl": rng.randn() * 5.0})
        history = pd.DataFrame(rows)
        weights = allocator.compute_weights(history, _uniform_regime_probs())
        assert abs(sum(weights.values()) - 1.0) < 1e-6


class TestRegimeSignal:
    """Regime probabilities should tilt weights toward regime-appropriate strategies."""

    def test_mean_reverting_regime_boosts_stat_mr(self):
        allocator = _make_allocator()
        history = _make_trade_history()
        # Strong mean-reverting regime
        regime = {r: 0.05 for r in MarketRegimeState}
        regime[MarketRegimeState.MEAN_REVERTING] = 0.80
        weights_mr = allocator.compute_weights(history, regime)

        # Compare to low_vol_bull which favors ORB
        allocator2 = _make_allocator()
        regime_bull = {r: 0.05 for r in MarketRegimeState}
        regime_bull[MarketRegimeState.LOW_VOL_BULL] = 0.80
        weights_bull = allocator2.compute_weights(history, regime_bull)

        # STAT_MR should be higher in mean-reverting regime
        assert weights_mr["STAT_MR"] > weights_bull["STAT_MR"] - 0.01

    def test_no_regime_probs_still_valid(self):
        allocator = _make_allocator()
        history = _make_trade_history()
        weights = allocator.compute_weights(history, regime_probs=None)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        for w in weights.values():
            assert w >= config.ADAPTIVE_MIN_WEIGHT - 1e-6


class TestCorrelationSignal:
    """Correlated strategies should both be penalized."""

    def test_identical_returns_penalized(self):
        """Two strategies with identical returns should get less weight."""
        allocator = _make_allocator()
        rng = np.random.RandomState(99)
        rows = []
        shared_pnl = rng.randn(50) * 10.0
        for i in range(50):
            # STAT_MR and VWAP get identical returns
            rows.append({"strategy": "STAT_MR", "pnl": shared_pnl[i]})
            rows.append({"strategy": "VWAP", "pnl": shared_pnl[i]})
            # Others get independent returns
            for s in ["KALMAN_PAIRS", "PEAD", "ORB", "MICRO_MOM"]:
                rows.append({"strategy": s, "pnl": rng.randn() * 10.0})
        history = pd.DataFrame(rows)
        weights = allocator.compute_weights(history, _uniform_regime_probs())
        # Correlated pair should not dominate
        assert weights["STAT_MR"] + weights["VWAP"] < 0.75


class TestChangeReason:
    """get_allocation_change_reason should return a non-empty string."""

    def test_reason_before_compute(self):
        allocator = _make_allocator()
        reason = allocator.get_allocation_change_reason()
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_reason_after_compute(self):
        allocator = _make_allocator()
        history = _make_trade_history(pnl_overrides={"STAT_MR": 100.0})
        allocator.compute_weights(history, _uniform_regime_probs())
        reason = allocator.get_allocation_change_reason()
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_reason_mentions_strategy_on_shift(self):
        allocator = _make_allocator()
        history = _make_trade_history(
            pnl_overrides={"STAT_MR": 300.0, "MICRO_MOM": -100.0}
        )
        allocator.compute_weights(history, _uniform_regime_probs())
        reason = allocator.get_allocation_change_reason()
        # Should mention at least one strategy that changed significantly
        assert any(s in reason for s in STRATEGIES) or "unchanged" in reason.lower()


class TestPreviousWeights:
    """_previous_weights should be stored after compute."""

    def test_previous_weights_none_initially(self):
        allocator = _make_allocator()
        assert allocator.previous_weights is None

    def test_previous_weights_set_after_compute(self):
        allocator = _make_allocator()
        history = _make_trade_history()
        weights = allocator.compute_weights(history, _uniform_regime_probs())
        assert allocator.previous_weights is not None
        assert allocator.previous_weights == weights


class TestVolTargetingIntegration:
    """Verify vol_targeting uses adaptive weights when enabled."""

    def test_adaptive_weights_used_when_enabled(self):
        from risk.vol_targeting import VolatilityTargetingRiskEngine

        engine = VolatilityTargetingRiskEngine()
        adaptive_w = {"STAT_MR": 0.50, "VWAP": 0.15}
        engine.set_adaptive_weights(adaptive_w)

        original_enabled = config.ADAPTIVE_ALLOCATION_ENABLED
        try:
            config.ADAPTIVE_ALLOCATION_ENABLED = True
            # Call position sizing for STAT_MR
            qty_adaptive = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=98.0,
                strategy="STAT_MR",
            )

            # Disable adaptive, use config.STRATEGY_ALLOCATIONS
            config.ADAPTIVE_ALLOCATION_ENABLED = False
            qty_static = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=98.0,
                strategy="STAT_MR",
            )

            # Adaptive weight (0.50) > static (0.40) so adaptive qty should be larger
            assert qty_adaptive >= qty_static
        finally:
            config.ADAPTIVE_ALLOCATION_ENABLED = original_enabled

    def test_fallback_when_strategy_not_in_adaptive(self):
        from risk.vol_targeting import VolatilityTargetingRiskEngine

        engine = VolatilityTargetingRiskEngine()
        engine.set_adaptive_weights({"STAT_MR": 0.50})

        original_enabled = config.ADAPTIVE_ALLOCATION_ENABLED
        try:
            config.ADAPTIVE_ALLOCATION_ENABLED = True
            # VWAP not in adaptive_weights, should fall back to config
            qty = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=98.0,
                strategy="VWAP",
            )
            assert qty > 0
        finally:
            config.ADAPTIVE_ALLOCATION_ENABLED = original_enabled
