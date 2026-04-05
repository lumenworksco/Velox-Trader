"""Comprehensive tests for risk/circuit_breaker.py.

Covers:
- Tiered escalation (Yellow -> Orange -> Red -> Black)
- Hysteresis (de-escalation requires recovery beyond threshold + buffer)
- escalate_to() forces minimum tier
- VIX spike escalation (via CDaR module, mocked)
- reset_daily()
- Properties (size_multiplier, allow_new_entries, should_close_*)
- Tier history tracking
"""

from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from risk.circuit_breaker import (
    CircuitTier,
    TierConfig,
    TieredCircuitBreaker,
    DEFAULT_TIERS,
)

ET = ZoneInfo("America/New_York")


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def cb():
    """Fresh TieredCircuitBreaker with defaults."""
    return TieredCircuitBreaker()


@pytest.fixture
def now():
    return datetime(2026, 4, 1, 10, 30, tzinfo=ET)


# ===================================================================
# Tiered escalation
# ===================================================================

class TestTieredEscalation:

    def test_initial_state_is_normal(self, cb):
        assert cb.current_tier == CircuitTier.NORMAL

    def test_escalation_to_yellow(self, cb, now):
        """P&L at -1% should trigger Yellow."""
        tier = cb.update(-0.01, now)
        assert tier == CircuitTier.YELLOW

    def test_escalation_to_orange(self, cb, now):
        """P&L at -2% should trigger Orange."""
        tier = cb.update(-0.02, now)
        assert tier == CircuitTier.ORANGE

    def test_escalation_to_red(self, cb, now):
        """P&L at -3% should trigger Red."""
        tier = cb.update(-0.03, now)
        assert tier == CircuitTier.RED

    def test_escalation_to_black(self, cb, now):
        """P&L at -4% should trigger Black."""
        tier = cb.update(-0.04, now)
        assert tier == CircuitTier.BLACK

    def test_escalation_beyond_black(self, cb, now):
        """P&L at -10% should still be Black (most severe)."""
        tier = cb.update(-0.10, now)
        assert tier == CircuitTier.BLACK

    def test_progressive_escalation(self, cb, now):
        """Step through tiers with worsening P&L."""
        cb.update(-0.005, now)
        assert cb.current_tier == CircuitTier.NORMAL

        cb.update(-0.01, now)
        assert cb.current_tier == CircuitTier.YELLOW

        cb.update(-0.02, now)
        assert cb.current_tier == CircuitTier.ORANGE

        cb.update(-0.03, now)
        assert cb.current_tier == CircuitTier.RED

        cb.update(-0.04, now)
        assert cb.current_tier == CircuitTier.BLACK

    def test_stays_at_current_tier_if_no_change(self, cb, now):
        """Same P&L level should keep the same tier."""
        cb.update(-0.015, now)
        assert cb.current_tier == CircuitTier.YELLOW
        cb.update(-0.015, now)
        assert cb.current_tier == CircuitTier.YELLOW

    def test_immediate_jump_to_high_tier(self, cb, now):
        """A sudden large loss can skip intermediate tiers."""
        tier = cb.update(-0.05, now)
        assert tier == CircuitTier.BLACK

    def test_positive_pnl_stays_normal(self, cb, now):
        tier = cb.update(0.02, now)
        assert tier == CircuitTier.NORMAL

    def test_zero_pnl_stays_normal(self, cb, now):
        tier = cb.update(0.0, now)
        assert tier == CircuitTier.NORMAL

    def test_slightly_above_threshold_stays_lower(self, cb, now):
        """P&L at -0.99% (just above -1%) should stay NORMAL."""
        tier = cb.update(-0.009, now)
        assert tier == CircuitTier.NORMAL


# ===================================================================
# Hysteresis — de-escalation requires extra recovery
# ===================================================================

class TestHysteresis:

    def test_de_escalation_blocked_by_hysteresis(self, cb, now):
        """After reaching Orange (-2%), must recover above -2% + 0.2% = -1.8%
        before de-escalating."""
        cb.update(-0.025, now)
        assert cb.current_tier == CircuitTier.ORANGE

        # Recover to -1.9% — still within hysteresis buffer
        cb.update(-0.019, now)
        assert cb.current_tier == CircuitTier.ORANGE  # stuck due to hysteresis

    def test_de_escalation_allowed_after_full_recovery(self, cb, now):
        """After recovering sufficiently past threshold + hysteresis."""
        cb.update(-0.025, now)
        assert cb.current_tier == CircuitTier.ORANGE

        # Recover above -2% + 0.2% = -1.8%
        cb.update(-0.017, now)
        assert cb.current_tier == CircuitTier.YELLOW  # de-escalated to Yellow

    def test_de_escalation_from_red_to_orange(self, cb, now):
        """RED threshold is -3%. Need to recover above -3% + 0.2% = -2.8% to leave RED."""
        cb.update(-0.035, now)
        assert cb.current_tier == CircuitTier.RED

        # At -2.9%: still in hysteresis zone for RED
        cb.update(-0.029, now)
        assert cb.current_tier == CircuitTier.RED

        # At -2.7%: past hysteresis, should de-escalate to ORANGE (still below -2%)
        cb.update(-0.027, now)
        assert cb.current_tier == CircuitTier.ORANGE

    def test_de_escalation_from_yellow_to_normal(self, cb, now):
        """YELLOW threshold is -1%. Need to recover above -1% + 0.2% = -0.8%."""
        cb.update(-0.012, now)
        assert cb.current_tier == CircuitTier.YELLOW

        # Still in hysteresis zone
        cb.update(-0.009, now)
        assert cb.current_tier == CircuitTier.YELLOW

        # Past the hysteresis buffer
        cb.update(-0.007, now)
        assert cb.current_tier == CircuitTier.NORMAL

    def test_escalation_is_immediate(self, cb, now):
        """Escalation should happen immediately (no hysteresis delay)."""
        tier = cb.update(-0.005, now)
        assert tier == CircuitTier.NORMAL

        tier = cb.update(-0.025, now)
        assert tier == CircuitTier.ORANGE  # immediate escalation

    def test_hysteresis_buffer_value(self, cb):
        """Verify the hysteresis buffer is 0.2%."""
        assert cb.HYSTERESIS_PCT == 0.002

    def test_full_recovery_to_normal(self, cb, now):
        """Recover from BLACK all the way back to NORMAL."""
        cb.update(-0.05, now)
        assert cb.current_tier == CircuitTier.BLACK

        # Step 1: recover past BLACK hysteresis (-4% + 0.2% = -3.8%)
        cb.update(-0.037, now)
        assert cb.current_tier == CircuitTier.RED

        # Step 2: recover past RED hysteresis (-3% + 0.2% = -2.8%)
        cb.update(-0.027, now)
        assert cb.current_tier == CircuitTier.ORANGE

        # Step 3: recover past ORANGE hysteresis (-2% + 0.2% = -1.8%)
        cb.update(-0.017, now)
        assert cb.current_tier == CircuitTier.YELLOW

        # Step 4: recover past YELLOW hysteresis (-1% + 0.2% = -0.8%)
        cb.update(-0.007, now)
        assert cb.current_tier == CircuitTier.NORMAL


# ===================================================================
# escalate_to() — force minimum tier
# ===================================================================

class TestEscalateTo:

    def test_escalate_from_normal_to_red(self, cb):
        result = cb.escalate_to(CircuitTier.RED, reason="VIX spike")
        assert result == CircuitTier.RED
        assert cb.current_tier == CircuitTier.RED

    def test_escalate_to_same_tier_noop(self, cb, now):
        cb.update(-0.02, now)
        assert cb.current_tier == CircuitTier.ORANGE
        result = cb.escalate_to(CircuitTier.ORANGE, reason="test")
        assert result == CircuitTier.ORANGE

    def test_escalate_to_lower_tier_noop(self, cb, now):
        """Cannot de-escalate via escalate_to()."""
        cb.update(-0.03, now)
        assert cb.current_tier == CircuitTier.RED
        result = cb.escalate_to(CircuitTier.YELLOW, reason="try to de-escalate")
        assert result == CircuitTier.RED  # stays at RED

    def test_escalate_records_history(self, cb):
        cb.escalate_to(CircuitTier.ORANGE, reason="test")
        assert len(cb.tier_history) == 1
        _, tier = cb.tier_history[0]
        assert tier == CircuitTier.ORANGE

    def test_escalate_to_black(self, cb):
        result = cb.escalate_to(CircuitTier.BLACK, reason="kill switch")
        assert result == CircuitTier.BLACK
        assert cb.should_close_all is True


# ===================================================================
# VIX spike escalation (CDaR module mock)
# ===================================================================

class TestVixSpikeEscalation:

    @patch("risk.circuit_breaker._drawdown_mgr")
    def test_cdar_escalates_to_red(self, mock_ddm, cb, now):
        """CDaR multiplier <= 0.0 should escalate to RED."""
        mock_ddm.get_exposure_multiplier.return_value = 0.0
        # P&L is only at Yellow level (-1%)
        tier = cb.update(-0.01, now)
        assert tier == CircuitTier.RED

    @patch("risk.circuit_breaker._drawdown_mgr")
    def test_cdar_escalates_to_orange(self, mock_ddm, cb, now):
        """CDaR multiplier <= 0.5 should escalate to ORANGE."""
        mock_ddm.get_exposure_multiplier.return_value = 0.4
        # P&L is at Normal level
        tier = cb.update(-0.005, now)
        assert tier == CircuitTier.ORANGE

    @patch("risk.circuit_breaker._drawdown_mgr")
    def test_cdar_does_not_lower_tier(self, mock_ddm, cb, now):
        """CDaR cannot reduce tier below what P&L dictates."""
        mock_ddm.get_exposure_multiplier.return_value = 0.8  # no escalation
        tier = cb.update(-0.03, now)
        assert tier == CircuitTier.RED  # P&L-driven

    @patch("risk.circuit_breaker._drawdown_mgr")
    def test_cdar_failure_is_fail_open(self, mock_ddm, cb, now):
        """If CDaR check throws, circuit breaker should continue normally."""
        mock_ddm.get_exposure_multiplier.side_effect = Exception("CDaR error")
        tier = cb.update(-0.015, now)
        assert tier == CircuitTier.YELLOW  # P&L-only


# ===================================================================
# Properties
# ===================================================================

class TestProperties:

    def test_normal_tier_properties(self, cb):
        assert cb.size_multiplier == 1.0
        assert cb.allow_new_entries is True
        assert cb.should_close_day_trades is False
        assert cb.should_close_all is False

    def test_yellow_tier_properties(self, cb, now):
        cb.update(-0.01, now)
        assert cb.size_multiplier == 0.5
        assert cb.allow_new_entries is True
        assert cb.should_close_day_trades is False
        assert cb.should_close_all is False

    def test_orange_tier_properties(self, cb, now):
        cb.update(-0.02, now)
        assert cb.size_multiplier == 0.0
        assert cb.allow_new_entries is False
        assert cb.should_close_day_trades is False
        assert cb.should_close_all is False

    def test_red_tier_properties(self, cb, now):
        cb.update(-0.03, now)
        assert cb.size_multiplier == 0.0
        assert cb.allow_new_entries is False
        assert cb.should_close_day_trades is True
        assert cb.should_close_all is False

    def test_black_tier_properties(self, cb, now):
        cb.update(-0.04, now)
        assert cb.size_multiplier == 0.0
        assert cb.allow_new_entries is False
        assert cb.should_close_day_trades is False
        assert cb.should_close_all is True

    def test_status_dict(self, cb, now):
        cb.update(-0.02, now)
        status = cb.status
        assert status["tier"] == "ORANGE"
        assert status["tier_value"] == 2
        assert status["size_multiplier"] == 0.0
        assert status["allow_new_entries"] is False
        assert status["last_update"] is not None


# ===================================================================
# Tier history
# ===================================================================

class TestTierHistory:

    def test_history_records_escalations(self, cb, now):
        cb.update(-0.01, now)
        cb.update(-0.03, now)
        assert len(cb.tier_history) == 2
        assert cb.tier_history[0][1] == CircuitTier.YELLOW
        assert cb.tier_history[1][1] == CircuitTier.RED

    def test_history_records_de_escalations(self, cb, now):
        cb.update(-0.02, now)
        cb.update(-0.007, now)  # recover past yellow hysteresis
        assert len(cb.tier_history) >= 2


# ===================================================================
# reset_daily()
# ===================================================================

class TestResetDaily:

    def test_reset_returns_to_normal(self, cb, now):
        cb.update(-0.04, now)
        assert cb.current_tier == CircuitTier.BLACK
        cb.reset_daily()
        assert cb.current_tier == CircuitTier.NORMAL

    def test_reset_clears_history(self, cb, now):
        cb.update(-0.02, now)
        cb.reset_daily()
        assert len(cb.tier_history) == 0


# ===================================================================
# Custom tiers
# ===================================================================

class TestCustomTiers:

    def test_custom_thresholds(self):
        custom = {
            CircuitTier.NORMAL: TierConfig(0.0, 1.0, True, False, False),
            CircuitTier.YELLOW: TierConfig(-0.005, 0.7, True, False, False),
            CircuitTier.ORANGE: TierConfig(-0.01, 0.3, False, False, False),
            CircuitTier.RED: TierConfig(-0.015, 0.0, False, True, False),
            CircuitTier.BLACK: TierConfig(-0.02, 0.0, False, False, True),
        }
        cb = TieredCircuitBreaker(tiers=custom)
        now = datetime(2026, 4, 1, 10, 0, tzinfo=ET)
        tier = cb.update(-0.005, now)
        assert tier == CircuitTier.YELLOW

    def test_non_monotonic_thresholds_fallback_to_defaults(self):
        """If thresholds aren't monotonically decreasing, fall back to defaults."""
        invalid = {
            CircuitTier.NORMAL: TierConfig(0.0, 1.0, True, False, False),
            CircuitTier.YELLOW: TierConfig(-0.03, 0.5, True, False, False),   # -3%
            CircuitTier.ORANGE: TierConfig(-0.02, 0.0, False, False, False),  # -2% (should be more negative)
            CircuitTier.RED: TierConfig(-0.04, 0.0, False, True, False),
            CircuitTier.BLACK: TierConfig(-0.05, 0.0, False, False, True),
        }
        cb = TieredCircuitBreaker(tiers=invalid)
        # Should have fallen back to DEFAULT_TIERS
        assert cb.tiers[CircuitTier.YELLOW].threshold_pct == DEFAULT_TIERS[CircuitTier.YELLOW].threshold_pct


# ===================================================================
# Thread safety
# ===================================================================

class TestThreadSafety:

    def test_concurrent_updates(self, cb):
        """Multiple threads updating simultaneously should not crash."""
        import threading

        def _update(pnl):
            for _ in range(50):
                cb.update(pnl, datetime(2026, 4, 1, 10, 0, tzinfo=ET))

        threads = [
            threading.Thread(target=_update, args=(-0.005 * i,))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Just verify no crash and tier is valid
        assert cb.current_tier in CircuitTier

    def test_concurrent_escalate_to(self, cb):
        import threading

        def _escalate(tier):
            cb.escalate_to(tier, reason="concurrent test")

        threads = [
            threading.Thread(target=_escalate, args=(CircuitTier.YELLOW,)),
            threading.Thread(target=_escalate, args=(CircuitTier.RED,)),
            threading.Thread(target=_escalate, args=(CircuitTier.BLACK,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Final state should be BLACK (highest)
        assert cb.current_tier == CircuitTier.BLACK
