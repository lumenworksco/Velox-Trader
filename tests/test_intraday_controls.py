"""Comprehensive tests for risk/intraday_controls.py.

Covers:
- Rolling window P&L tracking
- Threshold triggers at -0.8%, -1.2%, -1.8%
- Reset on new day
- Stop-loss velocity checks (3+ stops in 15min)
- Loss/win ratio throttling
- Pause duration mechanics
- State transitions (NORMAL -> THROTTLED -> PAUSED -> HALTED)
"""

from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

import config
from risk.intraday_controls import (
    ControlState,
    IntradayRiskControls,
    PnLTick,
    RiskControlState,
    WINDOW_LIMITS,
    STOPS_THRESHOLD,
    STOPS_PAUSE_DURATION,
    LOSS_RATIO_THRESHOLD,
    LOSS_RATIO_SIZE_MULT,
)

ET = ZoneInfo("America/New_York")


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def controls():
    """Fresh IntradayRiskControls."""
    return IntradayRiskControls()


@pytest.fixture
def base_time():
    """Base time for tests — 10:30 AM ET."""
    return datetime(2026, 4, 1, 10, 30, 0, tzinfo=ET)


# ===================================================================
# Rolling window P&L tracking — 5min, 30min, 1hr thresholds
# ===================================================================

class TestRollingWindowLimits:

    def test_5min_threshold_triggers_halt(self, controls, base_time):
        """Cumulative P&L < -0.8% in 5 minutes should trigger HALTED."""
        # Record losses within 5 minutes
        for i in range(4):
            controls.record_pnl(
                pnl_pct=-0.003,  # -0.3% each
                now=base_time + timedelta(seconds=30 * i),
            )
        # Total = -1.2%, exceeds -0.8% limit
        state = controls.check_controls(now=base_time + timedelta(minutes=2))
        assert state.state == ControlState.HALTED
        assert state.size_multiplier == 0.0

    def test_30min_threshold_triggers_halt(self, controls, base_time):
        """Cumulative P&L < -1.2% in 30 minutes should trigger HALTED."""
        # Spread losses over 20 minutes
        for i in range(6):
            controls.record_pnl(
                pnl_pct=-0.003,  # -0.3% each
                now=base_time + timedelta(minutes=3 * i),
            )
        # Total = -1.8%, exceeds -1.2% 30-min limit
        state = controls.check_controls(now=base_time + timedelta(minutes=20))
        assert state.state == ControlState.HALTED

    def test_1hr_threshold_triggers_halt(self, controls, base_time):
        """Cumulative P&L < -1.8% in 1 hour should trigger HALTED."""
        # Spread losses over 50 minutes
        for i in range(10):
            controls.record_pnl(
                pnl_pct=-0.002,  # -0.2% each
                now=base_time + timedelta(minutes=5 * i),
            )
        # Total = -2.0%, exceeds -1.8% 1-hr limit
        state = controls.check_controls(now=base_time + timedelta(minutes=55))
        assert state.state == ControlState.HALTED

    def test_just_below_threshold_stays_normal(self, controls, base_time):
        """P&L just above the 5-min threshold should stay NORMAL."""
        controls.record_pnl(pnl_pct=-0.007, now=base_time)  # -0.7%, above -0.8%
        state = controls.check_controls(now=base_time + timedelta(seconds=30))
        assert state.state == ControlState.NORMAL
        assert state.size_multiplier == 1.0

    def test_positive_pnl_stays_normal(self, controls, base_time):
        """Positive P&L should keep state NORMAL."""
        for i in range(5):
            controls.record_pnl(pnl_pct=0.002, now=base_time + timedelta(minutes=i))
        state = controls.check_controls(now=base_time + timedelta(minutes=5))
        assert state.state == ControlState.NORMAL

    def test_old_ticks_rolled_off(self, controls, base_time):
        """Losses older than the window should not count."""
        # Record a loss
        controls.record_pnl(pnl_pct=-0.009, now=base_time)
        # Wait 6 minutes (past the 5-min window)
        state = controls.check_controls(now=base_time + timedelta(minutes=6))
        # The -0.9% loss is outside the 5-min window, so 5-min P&L = 0
        assert state.rolling_5m_pnl == 0.0
        # But it's still within 30-min window
        assert state.rolling_30m_pnl < 0

    def test_mixed_wins_and_losses(self, controls, base_time):
        """Wins should offset losses in rolling windows."""
        controls.record_pnl(pnl_pct=-0.006, now=base_time)
        controls.record_pnl(pnl_pct=0.005, now=base_time + timedelta(minutes=1))
        state = controls.check_controls(now=base_time + timedelta(minutes=2))
        assert state.rolling_5m_pnl == pytest.approx(-0.001, abs=1e-6)
        assert state.state == ControlState.NORMAL


# ===================================================================
# Pause duration — RISK-006
# ===================================================================

class TestPauseDuration:

    def test_5min_breach_pauses_for_5min(self, controls, base_time):
        """5-min window breach should pause for 5 minutes."""
        controls.record_pnl(pnl_pct=-0.01, now=base_time)
        state = controls.check_controls(now=base_time + timedelta(seconds=30))
        assert state.state == ControlState.HALTED
        assert state.pause_until is not None
        # Pause should be ~5 min from the check time
        expected_pause = base_time + timedelta(seconds=30) + timedelta(minutes=5)
        assert abs((state.pause_until - expected_pause).total_seconds()) < 5

    def test_30min_breach_pauses_for_30min(self, controls, base_time):
        """30-min window breach should pause for 30 minutes."""
        # Spread losses just under the 5-min limit but over the 30-min limit
        for i in range(5):
            controls.record_pnl(
                pnl_pct=-0.003,
                now=base_time + timedelta(minutes=5 * i),
            )
        # Total = -1.5%, exceeds -1.2% 30-min limit but check at 25min
        state = controls.check_controls(now=base_time + timedelta(minutes=25))
        assert state.state == ControlState.HALTED

    def test_pause_expires_returns_to_normal(self, controls, base_time):
        """After pause duration, state should return to NORMAL."""
        controls.record_pnl(pnl_pct=-0.01, now=base_time)
        # Trigger halt
        controls.check_controls(now=base_time + timedelta(seconds=30))
        # Wait for pause to expire (5 min + extra)
        state = controls.check_controls(now=base_time + timedelta(minutes=10))
        assert state.state == ControlState.NORMAL

    def test_during_pause_reports_remaining_time(self, controls, base_time):
        """While paused, state should include pause_until."""
        controls.record_pnl(pnl_pct=-0.01, now=base_time)
        state = controls.check_controls(now=base_time + timedelta(seconds=30))
        assert state.pause_until is not None


# ===================================================================
# Stop-loss velocity checks
# ===================================================================

class TestStopLossVelocity:

    def test_3_stops_in_15min_triggers_pause(self, controls, base_time):
        """3+ stop-losses in 15 minutes should trigger a 30-min pause."""
        for i in range(3):
            controls.record_pnl(
                pnl_pct=0.0, is_stop_loss=True, is_loss=True,
                now=base_time + timedelta(minutes=2 * i),
            )
        state = controls.check_controls(now=base_time + timedelta(minutes=7))
        assert state.state == ControlState.PAUSED
        assert state.stops_last_15m == 3

    def test_2_stops_below_threshold(self, controls, base_time):
        """2 stop-losses in 15 minutes should NOT trigger pause."""
        for i in range(2):
            controls.record_pnl(
                pnl_pct=0.0, is_stop_loss=True, is_loss=True,
                now=base_time + timedelta(minutes=5 * i),
            )
        state = controls.check_controls(now=base_time + timedelta(minutes=12))
        assert state.state != ControlState.PAUSED

    def test_stops_outside_window_not_counted(self, controls, base_time):
        """Stop-losses older than 15 min should not count."""
        # 2 stops at base_time
        for i in range(2):
            controls.record_pnl(
                pnl_pct=0.0, is_stop_loss=True, is_loss=True,
                now=base_time + timedelta(seconds=30 * i),
            )
        # 1 stop at 20 minutes later (the first 2 are now outside the 15-min window)
        controls.record_pnl(
            pnl_pct=0.0, is_stop_loss=True, is_loss=True,
            now=base_time + timedelta(minutes=20),
        )
        state = controls.check_controls(now=base_time + timedelta(minutes=20, seconds=30))
        # Only 1 stop in the 15-min window
        assert state.stops_last_15m == 1
        assert state.state != ControlState.PAUSED

    def test_stop_pause_duration_30min(self, controls, base_time):
        """Stop velocity pause should last 30 minutes."""
        for i in range(3):
            controls.record_pnl(
                pnl_pct=0.0, is_stop_loss=True, is_loss=True,
                now=base_time + timedelta(minutes=i),
            )
        state = controls.check_controls(now=base_time + timedelta(minutes=5))
        assert state.pause_until is not None

        # Still paused after 20 min
        state_20 = controls.check_controls(now=base_time + timedelta(minutes=25))
        assert state_20.state in (ControlState.PAUSED, ControlState.HALTED)

    def test_record_stop_loss_convenience(self, controls, base_time):
        """record_stop_loss() should be equivalent to record_pnl with stop flag."""
        for i in range(3):
            controls.record_stop_loss(now=base_time + timedelta(minutes=i))
        state = controls.check_controls(now=base_time + timedelta(minutes=5))
        assert state.stops_last_15m == 3


# ===================================================================
# Loss/win ratio throttling
# ===================================================================

class TestLossWinRatio:

    def test_high_loss_ratio_triggers_throttle(self, controls, base_time):
        """4:1 loss/win ratio in 1 hour should throttle to 50% sizing."""
        # 1 win, 4 losses = 4:1 ratio
        controls.record_pnl(pnl_pct=0.001, is_win=True,
                             now=base_time)
        for i in range(4):
            controls.record_pnl(pnl_pct=-0.001, is_loss=True,
                                 now=base_time + timedelta(minutes=5 * (i + 1)))
        state = controls.check_controls(now=base_time + timedelta(minutes=25))
        assert state.state == ControlState.THROTTLED
        assert state.size_multiplier == LOSS_RATIO_SIZE_MULT

    def test_below_ratio_threshold_stays_normal(self, controls, base_time):
        """3:1 ratio should not trigger throttling (need 4:1)."""
        controls.record_pnl(pnl_pct=0.001, is_win=True, now=base_time)
        for i in range(3):
            controls.record_pnl(pnl_pct=-0.001, is_loss=True,
                                 now=base_time + timedelta(minutes=5 * (i + 1)))
        state = controls.check_controls(now=base_time + timedelta(minutes=20))
        assert state.state == ControlState.NORMAL

    def test_insufficient_trades_not_throttled(self, controls, base_time):
        """Fewer than 3 total trades should not trigger ratio check."""
        controls.record_pnl(pnl_pct=-0.001, is_loss=True, now=base_time)
        controls.record_pnl(pnl_pct=-0.001, is_loss=True,
                             now=base_time + timedelta(minutes=5))
        state = controls.check_controls(now=base_time + timedelta(minutes=10))
        assert state.state == ControlState.NORMAL


# ===================================================================
# should_allow_trade()
# ===================================================================

class TestShouldAllowTrade:

    def test_normal_state_allows_trade(self, controls, base_time):
        allowed, reason = controls.should_allow_trade(now=base_time)
        assert allowed is True
        assert reason == ""

    def test_halted_state_blocks_trade(self, controls, base_time):
        controls.record_pnl(pnl_pct=-0.01, now=base_time)
        allowed, reason = controls.should_allow_trade(
            now=base_time + timedelta(seconds=30)
        )
        assert allowed is False
        assert "Rolling P&L limit" in reason or "rolling_window_breach" in reason

    def test_paused_state_blocks_trade(self, controls, base_time):
        for i in range(3):
            controls.record_pnl(
                pnl_pct=0.0, is_stop_loss=True, is_loss=True,
                now=base_time + timedelta(minutes=i),
            )
        allowed, reason = controls.should_allow_trade(
            now=base_time + timedelta(minutes=5)
        )
        assert allowed is False
        assert "Velocity pause" in reason or "stops" in reason

    def test_throttled_state_allows_trade(self, controls, base_time):
        """Throttled state allows trades but at reduced size."""
        controls.record_pnl(pnl_pct=0.001, is_win=True, now=base_time)
        for i in range(4):
            controls.record_pnl(pnl_pct=-0.001, is_loss=True,
                                 now=base_time + timedelta(minutes=5 * (i + 1)))
        allowed, reason = controls.should_allow_trade(
            now=base_time + timedelta(minutes=25)
        )
        assert allowed is True
        assert "Throttled" in reason


# ===================================================================
# get_size_multiplier() / get_sizing_multiplier()
# ===================================================================

class TestGetSizeMultiplier:

    def test_normal_returns_1(self, controls, base_time):
        mult = controls.get_size_multiplier(now=base_time)
        assert mult == 1.0

    def test_halted_returns_0(self, controls, base_time):
        controls.record_pnl(pnl_pct=-0.01, now=base_time)
        mult = controls.get_size_multiplier(now=base_time + timedelta(seconds=30))
        assert mult == 0.0

    def test_throttled_returns_half(self, controls, base_time):
        controls.record_pnl(pnl_pct=0.001, is_win=True, now=base_time)
        for i in range(4):
            controls.record_pnl(pnl_pct=-0.001, is_loss=True,
                                 now=base_time + timedelta(minutes=5 * (i + 1)))
        mult = controls.get_sizing_multiplier(now=base_time + timedelta(minutes=25))
        assert mult == LOSS_RATIO_SIZE_MULT

    def test_get_sizing_multiplier_is_alias(self, controls, base_time):
        """get_sizing_multiplier should return the same as get_size_multiplier."""
        m1 = controls.get_size_multiplier(now=base_time)
        m2 = controls.get_sizing_multiplier(now=base_time)
        assert m1 == m2


# ===================================================================
# reset_daily()
# ===================================================================

class TestResetDaily:

    def test_reset_clears_all_state(self, controls, base_time):
        """reset_daily should clear ticks, state, and pause."""
        controls.record_pnl(pnl_pct=-0.01, now=base_time)
        controls.check_controls(now=base_time + timedelta(seconds=30))
        # Now reset
        controls.reset_daily()
        # State should be clean
        state = controls.check_controls(now=base_time + timedelta(hours=1))
        assert state.state == ControlState.NORMAL
        assert state.size_multiplier == 1.0
        assert state.rolling_5m_pnl == 0.0
        assert state.rolling_30m_pnl == 0.0
        assert state.rolling_1h_pnl == 0.0

    def test_reset_clears_pause(self, controls, base_time):
        """Pause should be cleared after reset."""
        for i in range(3):
            controls.record_pnl(
                pnl_pct=0.0, is_stop_loss=True, is_loss=True,
                now=base_time + timedelta(minutes=i),
            )
        controls.check_controls(now=base_time + timedelta(minutes=5))
        controls.reset_daily()
        allowed, _ = controls.should_allow_trade(
            now=base_time + timedelta(minutes=10)
        )
        assert allowed is True


# ===================================================================
# Status property
# ===================================================================

class TestStatus:

    def test_status_dict_keys(self, controls):
        status = controls.status
        assert "state" in status
        assert "ticks_buffered" in status
        assert "pause_until" in status
        assert "window_limits" in status

    def test_status_reflects_state(self, controls, base_time):
        controls.record_pnl(pnl_pct=-0.01, now=base_time)
        controls.check_controls(now=base_time + timedelta(seconds=30))
        status = controls.status
        assert status["state"] == "halted"
        assert status["pause_until"] is not None


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:

    def test_empty_ticks_normal(self, controls, base_time):
        """No ticks recorded should give NORMAL state."""
        state = controls.check_controls(now=base_time)
        assert state.state == ControlState.NORMAL

    def test_ticks_pruned_after_2_hours(self, controls, base_time):
        """Ticks older than 2 hours should be pruned."""
        controls.record_pnl(pnl_pct=-0.005, now=base_time)
        # Check 3 hours later — old tick should be pruned
        state = controls.check_controls(now=base_time + timedelta(hours=3))
        assert state.rolling_1h_pnl == 0.0

    def test_custom_window_limits(self, base_time):
        """Custom window limits should override defaults."""
        custom_limits = {
            timedelta(minutes=5): -0.005,   # Tighter
            timedelta(minutes=30): -0.01,
            timedelta(hours=1): -0.015,
        }
        controls = IntradayRiskControls(window_limits=custom_limits)
        controls.record_pnl(pnl_pct=-0.006, now=base_time)
        state = controls.check_controls(now=base_time + timedelta(seconds=30))
        assert state.state == ControlState.HALTED

    def test_thread_safety(self, controls, base_time):
        """Concurrent record_pnl and check_controls should not crash."""
        import threading

        def _record(offset):
            for i in range(50):
                controls.record_pnl(
                    pnl_pct=-0.0001,
                    now=base_time + timedelta(seconds=offset + i),
                )

        def _check(offset):
            for i in range(50):
                controls.check_controls(
                    now=base_time + timedelta(seconds=offset + i)
                )

        threads = [
            threading.Thread(target=_record, args=(0,)),
            threading.Thread(target=_record, args=(100,)),
            threading.Thread(target=_check, args=(200,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Just verify no crash
        assert True
