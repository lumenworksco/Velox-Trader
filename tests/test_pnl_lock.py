"""Tests for Daily P&L Lock."""

import pytest
from unittest.mock import patch


class TestDailyPnLLock:
    def test_normal_state(self):
        with patch.dict('config.__dict__', {
            'PNL_GAIN_LOCK_PCT': 0.015,
            'PNL_LOSS_HALT_PCT': -0.01,
            'PNL_GAIN_LOCK_SIZE_MULT': 0.3,
        }):
            from risk.daily_pnl_lock import DailyPnLLock, LockState
            lock = DailyPnLLock()
            state = lock.update(0.005)
            assert state == LockState.NORMAL
            assert lock.get_size_multiplier() == 1.0
            assert lock.is_trading_allowed()

    def test_gain_lock_activation(self):
        with patch.dict('config.__dict__', {
            'PNL_GAIN_LOCK_PCT': 0.015,
            'PNL_LOSS_HALT_PCT': -0.01,
            'PNL_GAIN_LOCK_SIZE_MULT': 0.70,  # V12: less aggressive gain lock
        }):
            from risk.daily_pnl_lock import DailyPnLLock, LockState
            lock = DailyPnLLock()
            state = lock.update(0.016)
            assert state == LockState.GAIN_LOCK
            assert lock.get_size_multiplier() == 0.70  # V12: 70% not 30%
            assert lock.is_trading_allowed()

    def test_loss_halt_activation(self):
        with patch.dict('config.__dict__', {
            'PNL_GAIN_LOCK_PCT': 0.015,
            'PNL_LOSS_HALT_PCT': -0.01,
            'PNL_GAIN_LOCK_SIZE_MULT': 0.3,
        }):
            from risk.daily_pnl_lock import DailyPnLLock, LockState
            lock = DailyPnLLock()
            state = lock.update(-0.012)
            assert state == LockState.LOSS_HALT
            assert lock.get_size_multiplier() == 0.0
            assert not lock.is_trading_allowed()

    def test_daily_reset(self):
        with patch.dict('config.__dict__', {
            'PNL_GAIN_LOCK_PCT': 0.015,
            'PNL_LOSS_HALT_PCT': -0.01,
            'PNL_GAIN_LOCK_SIZE_MULT': 0.3,
        }):
            from risk.daily_pnl_lock import DailyPnLLock, LockState
            lock = DailyPnLLock()
            lock.update(-0.02)
            assert lock.state == LockState.LOSS_HALT
            lock.reset_daily()
            assert lock.state == LockState.NORMAL
            assert lock.get_size_multiplier() == 1.0

    def test_transition_gain_to_normal(self):
        """P&L dropping back below gain lock returns to NORMAL."""
        with patch.dict('config.__dict__', {
            'PNL_GAIN_LOCK_PCT': 0.015,
            'PNL_LOSS_HALT_PCT': -0.01,
            'PNL_GAIN_LOCK_SIZE_MULT': 0.3,
        }):
            from risk.daily_pnl_lock import DailyPnLLock, LockState
            lock = DailyPnLLock()
            lock.update(0.02)
            assert lock.state == LockState.GAIN_LOCK
            lock.update(0.01)
            assert lock.state == LockState.NORMAL
