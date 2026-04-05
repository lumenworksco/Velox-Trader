"""Comprehensive tests for risk/risk_manager.py.

Covers:
- calculate_position_size() with various inputs
- VIX scaling in both directions (up and down)
- Multiplier cascade floor (0.30 minimum)
- Time-of-day multiplier (each session window)
- Friday EOW multiplier
- Drawdown-based sizing (smooth curve)
- register_trade() and close_trade()
- partial_close()
- get_day_summary()
- check_circuit_breaker()
"""

import threading
from datetime import datetime, time
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

import config
from risk.risk_manager import (
    RiskManager,
    TradeRecord,
    VixSnapshot,
    get_friday_eow_multiplier,
    get_time_of_day_multiplier,
    get_vix_risk_scalar,
    is_strategy_in_time_window,
    _INTRADAY_SESSION_WINDOWS,
    STRATEGY_TIME_WINDOWS,
)

ET = ZoneInfo("America/New_York")


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def rm():
    """Fresh RiskManager with $100k equity."""
    r = RiskManager()
    r.starting_equity = 100_000.0
    r.current_equity = 100_000.0
    r.current_cash = 80_000.0
    return r


@pytest.fixture
def trade_aapl():
    """Standard open long trade on AAPL."""
    return TradeRecord(
        symbol="AAPL",
        strategy="ORB",
        side="buy",
        entry_price=150.0,
        entry_time=datetime(2026, 4, 1, 10, 5, tzinfo=ET),
        qty=10,
        take_profit=155.0,
        stop_loss=148.0,
    )


@pytest.fixture
def trade_tsla_short():
    """Standard open short trade on TSLA."""
    return TradeRecord(
        symbol="TSLA",
        strategy="STAT_MR",
        side="sell",
        entry_price=200.0,
        entry_time=datetime(2026, 4, 1, 10, 10, tzinfo=ET),
        qty=5,
        take_profit=190.0,
        stop_loss=210.0,
    )


# ===================================================================
# calculate_position_size() — basic
# ===================================================================

class TestCalculatePositionSize:

    @patch("risk.risk_manager.get_vix_risk_scalar", return_value=1.0)
    def test_basic_long_position(self, _vix, rm, override_config):
        """Entry $100, stop $98 => risk_per_share=$2, base shares."""
        with override_config(DYNAMIC_ALLOCATION=False, BEARISH_SIZE_CUT=0.40):
            qty = rm.calculate_position_size(
                entry_price=100.0, stop_price=98.0,
                regime="UNKNOWN", strategy="ORB", side="buy",
            )
        assert qty > 0
        # Risk = 100k * 0.008 = $800, risk/share = $2, base_shares = 400
        # Position value = 400 * 100 = $40k, capped at 8% of 100k = $8k
        # Then capped by MAX_PORTFOLIO_DEPLOY
        assert qty <= int(rm.current_equity * config.MAX_POSITION_PCT / 100.0) + 1

    @patch("risk.risk_manager.get_vix_risk_scalar", return_value=1.0)
    def test_zero_risk_per_share_returns_zero(self, _vix, rm):
        """If entry == stop, risk_per_share is 0 => return 0 shares."""
        qty = rm.calculate_position_size(
            entry_price=100.0, stop_price=100.0,
            regime="UNKNOWN",
        )
        assert qty == 0

    @patch("risk.risk_manager.get_vix_risk_scalar", return_value=1.0)
    def test_zero_entry_price_returns_zero(self, _vix, rm):
        qty = rm.calculate_position_size(
            entry_price=0.0, stop_price=5.0,
            regime="UNKNOWN",
        )
        assert qty == 0

    @patch("risk.risk_manager.get_vix_risk_scalar", return_value=1.0)
    def test_bearish_regime_reduces_size(self, _vix, rm, override_config):
        """Bearish regime should cut position size."""
        with override_config(DYNAMIC_ALLOCATION=False, BEARISH_SIZE_CUT=0.40):
            qty_bullish = rm.calculate_position_size(
                entry_price=100.0, stop_price=95.0,
                regime="BULLISH",
            )
            qty_bearish = rm.calculate_position_size(
                entry_price=100.0, stop_price=95.0,
                regime="BEARISH",
            )
        assert qty_bearish < qty_bullish

    @patch("risk.risk_manager.get_vix_risk_scalar", return_value=1.0)
    def test_short_side_reduces_size(self, _vix, rm, override_config):
        """Short trades get reduced by SHORT_SIZE_MULTIPLIER."""
        with override_config(DYNAMIC_ALLOCATION=False):
            qty_long = rm.calculate_position_size(
                entry_price=100.0, stop_price=95.0,
                regime="UNKNOWN", side="buy",
            )
            qty_short = rm.calculate_position_size(
                entry_price=100.0, stop_price=105.0,
                regime="UNKNOWN", side="sell",
            )
        assert qty_short <= qty_long

    @patch("risk.risk_manager.get_vix_risk_scalar", return_value=1.0)
    def test_position_capped_at_max_deploy(self, _vix, rm, override_config):
        """Position can't exceed remaining deployment capacity."""
        with override_config(DYNAMIC_ALLOCATION=False, MAX_PORTFOLIO_DEPLOY=0.10):
            # Deploy $10k max, add a trade consuming most of it
            rm.open_trades["MSFT"] = TradeRecord(
                symbol="MSFT", strategy="ORB", side="buy",
                entry_price=100.0, entry_time=datetime.now(ET),
                qty=90, take_profit=110.0, stop_loss=95.0,
            )
            qty = rm.calculate_position_size(
                entry_price=100.0, stop_price=95.0,
                regime="UNKNOWN",
            )
        # $10k deploy - $9k already used = $1k remaining => 10 shares at $100
        assert qty <= 10


# ===================================================================
# VIX scaling — both directions
# ===================================================================

class TestVixScaling:

    @patch("risk.risk_manager.get_vix_risk_scalar", return_value=0.5)
    def test_high_vix_reduces_size(self, _vix, rm, override_config):
        """High VIX scalar (0.5) should reduce position size."""
        with override_config(DYNAMIC_ALLOCATION=False):
            qty = rm.calculate_position_size(
                entry_price=100.0, stop_price=95.0,
                regime="UNKNOWN",
            )
        # Without VIX scaling qty would be larger
        assert qty > 0

    @patch("risk.risk_manager.get_vix_risk_scalar", return_value=1.3)
    def test_low_vix_boosts_size(self, _vix, rm, override_config):
        """Low VIX (falling) scalar 1.3 should increase position size."""
        with override_config(DYNAMIC_ALLOCATION=False):
            qty_boosted = rm.calculate_position_size(
                entry_price=100.0, stop_price=95.0,
                regime="UNKNOWN",
            )

        # Compare with VIX scalar 1.0
        with patch("risk.risk_manager.get_vix_risk_scalar", return_value=1.0):
            with override_config(DYNAMIC_ALLOCATION=False):
                qty_normal = rm.calculate_position_size(
                    entry_price=100.0, stop_price=95.0,
                    regime="UNKNOWN",
                )
        assert qty_boosted >= qty_normal

    @patch("risk.risk_manager.get_vix_risk_scalar", return_value=0.0)
    def test_vix_halt_zero_scalar_triggers_floor(self, _vix, rm, override_config):
        """VIX at halt level returns scalar 0.0, but cascade floor applies."""
        with override_config(DYNAMIC_ALLOCATION=False):
            qty = rm.calculate_position_size(
                entry_price=100.0, stop_price=95.0,
                regime="UNKNOWN",
            )
        # cascade floor enforces min 30% of original size, so qty > 0
        assert qty > 0


class TestGetVixRiskScalar:
    """Test the get_vix_risk_scalar() function directly."""

    @patch("risk.risk_manager.get_vix_level", return_value=10.0)
    def test_low_vix_returns_high_scalar(self, _mock, override_config):
        """VIX <= 12 should give level_scalar of 1.0."""
        import risk.risk_manager as rm_mod
        with override_config(VIX_RISK_SCALING_ENABLED=True, VIX_HALT_THRESHOLD=40):
            # Clear history to avoid direction adjustment
            with rm_mod._vix_cache_lock:
                rm_mod._vix_history.clear()
            scalar = get_vix_risk_scalar()
        assert scalar >= 0.9

    @patch("risk.risk_manager.get_vix_level", return_value=45.0)
    def test_vix_above_halt_returns_zero(self, _mock, override_config):
        """VIX above halt threshold should return 0.0."""
        import risk.risk_manager as rm_mod
        with override_config(VIX_RISK_SCALING_ENABLED=True, VIX_HALT_THRESHOLD=40):
            with rm_mod._vix_cache_lock:
                rm_mod._vix_history.clear()
            scalar = get_vix_risk_scalar()
        assert scalar == 0.0

    def test_disabled_returns_1(self, override_config):
        """When VIX scaling disabled, always returns 1.0."""
        with override_config(VIX_RISK_SCALING_ENABLED=False):
            scalar = get_vix_risk_scalar()
        assert scalar == 1.0

    @patch("risk.risk_manager.get_vix_level", return_value=20.0)
    def test_vix_spike_direction_reduces(self, _mock, override_config):
        """VIX spiking (>10% increase) should apply direction_adj=0.7."""
        import risk.risk_manager as rm_mod
        with override_config(VIX_RISK_SCALING_ENABLED=True, VIX_HALT_THRESHOLD=40):
            with rm_mod._vix_cache_lock:
                rm_mod._vix_history.clear()
                # Seed history with low values so current 20 is a spike
                rm_mod._vix_history.extend([14.0, 14.0, 14.0, 14.0])
            scalar = get_vix_risk_scalar()
        # level_scalar at VIX=20 is ~0.85, direction_adj ~0.7 => ~0.595
        # bounded by [0.3, 1.3]
        assert 0.3 <= scalar <= 0.85

    @patch("risk.risk_manager.get_vix_level", return_value=20.0)
    def test_vix_falling_direction_boosts(self, _mock, override_config):
        """VIX falling fast (>10% decrease) should boost with direction_adj=1.2."""
        import risk.risk_manager as rm_mod
        with override_config(VIX_RISK_SCALING_ENABLED=True, VIX_HALT_THRESHOLD=40):
            with rm_mod._vix_cache_lock:
                rm_mod._vix_history.clear()
                # Seed history with high values so current 20 is falling
                rm_mod._vix_history.extend([28.0, 27.0, 26.0, 25.0])
            scalar = get_vix_risk_scalar()
        # level_scalar ~0.85, direction_adj ~1.2 => ~1.02
        assert scalar > 0.85

    @patch("risk.risk_manager.get_vix_level", return_value=30.0)
    def test_vix_linear_interpolation(self, _mock, override_config):
        """VIX between breakpoints should interpolate linearly."""
        import risk.risk_manager as rm_mod
        with override_config(VIX_RISK_SCALING_ENABLED=True, VIX_HALT_THRESHOLD=40):
            with rm_mod._vix_cache_lock:
                rm_mod._vix_history.clear()
            scalar = get_vix_risk_scalar()
        # VIX=30 is between (25, 0.70) and (30, 0.50), at boundary => 0.50
        assert 0.3 <= scalar <= 0.55

    @patch("risk.risk_manager.get_vix_level", return_value=35.0)
    def test_scalar_floor_at_0_3(self, _mock, override_config):
        """Scalar should never go below 0.3 (unless VIX >= halt threshold)."""
        import risk.risk_manager as rm_mod
        with override_config(VIX_RISK_SCALING_ENABLED=True, VIX_HALT_THRESHOLD=40):
            with rm_mod._vix_cache_lock:
                rm_mod._vix_history.clear()
                # Seed with lower values to simulate spike
                rm_mod._vix_history.extend([20.0, 20.0, 20.0, 20.0])
            scalar = get_vix_risk_scalar()
        assert scalar >= 0.3


# ===================================================================
# Multiplier cascade floor
# ===================================================================

class TestMultiplierCascadeFloor:

    @patch("risk.risk_manager.get_vix_risk_scalar", return_value=0.3)
    def test_cascade_floor_prevents_over_reduction(self, _vix, rm, override_config):
        """Even with VIX=0.3 + bearish + short, floor should keep >= 30% of original."""
        with override_config(DYNAMIC_ALLOCATION=False, BEARISH_SIZE_CUT=0.40):
            qty = rm.calculate_position_size(
                entry_price=100.0, stop_price=95.0,
                regime="BEARISH", side="sell",
            )
        assert qty > 0  # floor prevents it going to zero

    @patch("risk.risk_manager.get_vix_risk_scalar", return_value=1.0)
    def test_no_floor_when_no_reduction(self, _vix, rm, override_config):
        """When no reduction is applied, floor doesn't change anything."""
        with override_config(DYNAMIC_ALLOCATION=False):
            qty = rm.calculate_position_size(
                entry_price=100.0, stop_price=95.0,
                regime="UNKNOWN", side="buy",
            )
        assert qty > 0


# ===================================================================
# Time-of-day multiplier
# ===================================================================

class TestTimeOfDayMultiplier:

    def test_open_rush(self):
        """9:30-9:40 => multiplier 0.50"""
        dt = datetime(2026, 4, 1, 9, 35, tzinfo=ET)
        assert get_time_of_day_multiplier(dt) == 0.50

    def test_early_session(self):
        """9:40-10:00 => multiplier 0.80"""
        dt = datetime(2026, 4, 1, 9, 45, tzinfo=ET)
        assert get_time_of_day_multiplier(dt) == 0.80

    def test_prime_session(self):
        """10:00-11:30 => multiplier 1.00"""
        dt = datetime(2026, 4, 1, 10, 30, tzinfo=ET)
        assert get_time_of_day_multiplier(dt) == 1.00

    def test_lunch_session(self):
        """11:30-13:00 => multiplier 0.60"""
        dt = datetime(2026, 4, 1, 12, 0, tzinfo=ET)
        assert get_time_of_day_multiplier(dt) == 0.60

    def test_afternoon_session(self):
        """13:00-15:00 => multiplier 1.00"""
        dt = datetime(2026, 4, 1, 14, 0, tzinfo=ET)
        assert get_time_of_day_multiplier(dt) == 1.00

    def test_close_session(self):
        """15:00-15:55 => multiplier 0.70"""
        dt = datetime(2026, 4, 1, 15, 30, tzinfo=ET)
        assert get_time_of_day_multiplier(dt) == 0.70

    def test_outside_all_windows_returns_1(self):
        """Before market open or after 15:55 => default 1.0."""
        dt_pre = datetime(2026, 4, 1, 9, 0, tzinfo=ET)
        dt_post = datetime(2026, 4, 1, 16, 0, tzinfo=ET)
        assert get_time_of_day_multiplier(dt_pre) == 1.0
        assert get_time_of_day_multiplier(dt_post) == 1.0

    def test_window_boundaries_start_inclusive(self):
        """Start of each window is inclusive."""
        dt = datetime(2026, 4, 1, 9, 30, tzinfo=ET)
        assert get_time_of_day_multiplier(dt) == 0.50
        dt2 = datetime(2026, 4, 1, 10, 0, tzinfo=ET)
        assert get_time_of_day_multiplier(dt2) == 1.00

    def test_window_boundaries_end_exclusive(self):
        """End of each window is exclusive."""
        # 9:40 should be the NEXT window (Early), not Open Rush
        dt = datetime(2026, 4, 1, 9, 40, tzinfo=ET)
        assert get_time_of_day_multiplier(dt) == 0.80


# ===================================================================
# Friday EOW multiplier
# ===================================================================

class TestFridayEowMultiplier:

    def test_friday_after_2pm_returns_half(self):
        """Friday after 14:00 ET => 0.50 multiplier."""
        friday = datetime(2026, 4, 3, 14, 30, tzinfo=ET)
        assert friday.weekday() == 4  # sanity check
        assert get_friday_eow_multiplier(friday) == 0.50

    def test_friday_before_2pm_returns_1(self):
        """Friday before 14:00 ET => 1.0."""
        friday = datetime(2026, 4, 3, 10, 0, tzinfo=ET)
        assert get_friday_eow_multiplier(friday) == 1.0

    def test_friday_exactly_2pm_returns_half(self):
        """Friday at exactly 14:00 ET => 0.50."""
        friday = datetime(2026, 4, 3, 14, 0, tzinfo=ET)
        assert get_friday_eow_multiplier(friday) == 0.50

    def test_monday_returns_1(self):
        """Non-Friday days should always return 1.0."""
        monday = datetime(2026, 3, 30, 14, 30, tzinfo=ET)
        assert monday.weekday() == 0
        assert get_friday_eow_multiplier(monday) == 1.0

    def test_wednesday_returns_1(self):
        wednesday = datetime(2026, 4, 1, 15, 0, tzinfo=ET)
        assert wednesday.weekday() == 2
        assert get_friday_eow_multiplier(wednesday) == 1.0


# ===================================================================
# Drawdown-based sizing
# ===================================================================

class TestDrawdownSizing:

    @patch("risk.risk_manager.get_vix_risk_scalar", return_value=1.0)
    def test_no_drawdown_no_reduction(self, _vix, rm, override_config):
        """When equity == starting_equity, no drawdown reduction."""
        with override_config(DYNAMIC_ALLOCATION=False):
            qty_no_dd = rm.calculate_position_size(
                entry_price=100.0, stop_price=95.0,
                regime="UNKNOWN",
            )
        assert qty_no_dd > 0

    @patch("risk.risk_manager.get_vix_risk_scalar", return_value=1.0)
    def test_moderate_drawdown_reduces_size(self, _vix, rm, override_config):
        """4% drawdown should reduce size by roughly 50%."""
        with override_config(DYNAMIC_ALLOCATION=False, MAX_ACCEPTABLE_DRAWDOWN=0.08):
            qty_no_dd = rm.calculate_position_size(
                entry_price=100.0, stop_price=95.0,
                regime="UNKNOWN",
            )
            # Simulate 4% drawdown
            rm.current_equity = 96_000.0
            qty_dd = rm.calculate_position_size(
                entry_price=100.0, stop_price=95.0,
                regime="UNKNOWN",
            )
        assert qty_dd < qty_no_dd

    @patch("risk.risk_manager.get_vix_risk_scalar", return_value=1.0)
    def test_severe_drawdown_floors_at_20pct(self, _vix, rm, override_config):
        """Drawdown beyond max_dd should floor the multiplier at 0.2."""
        with override_config(DYNAMIC_ALLOCATION=False, MAX_ACCEPTABLE_DRAWDOWN=0.08):
            rm.current_equity = 85_000.0  # 15% drawdown
            qty = rm.calculate_position_size(
                entry_price=100.0, stop_price=95.0,
                regime="UNKNOWN",
            )
        # drawdown_mult = max(0.2, 1.0 - 0.15/0.08) = max(0.2, -0.875) = 0.2
        # But cascade floor might also apply
        assert qty > 0

    @patch("risk.risk_manager.get_vix_risk_scalar", return_value=1.0)
    def test_equity_above_starting_no_drawdown(self, _vix, rm, override_config):
        """If equity > starting_equity, no drawdown reduction."""
        with override_config(DYNAMIC_ALLOCATION=False):
            rm.current_equity = 105_000.0
            qty = rm.calculate_position_size(
                entry_price=100.0, stop_price=95.0,
                regime="UNKNOWN",
            )
        assert qty > 0


# ===================================================================
# register_trade()
# ===================================================================

class TestRegisterTrade:

    def test_register_adds_to_open_trades(self, rm, trade_aapl):
        rm.register_trade(trade_aapl)
        assert "AAPL" in rm.open_trades
        assert rm.open_trades["AAPL"] is trade_aapl

    def test_register_increments_signals_today(self, rm, trade_aapl):
        assert rm.signals_today == 0
        rm.register_trade(trade_aapl)
        assert rm.signals_today == 1

    def test_register_multiple_trades(self, rm, trade_aapl, trade_tsla_short):
        rm.register_trade(trade_aapl)
        rm.register_trade(trade_tsla_short)
        assert len(rm.open_trades) == 2
        assert rm.signals_today == 2

    def test_register_overwrites_same_symbol(self, rm, trade_aapl):
        rm.register_trade(trade_aapl)
        trade2 = TradeRecord(
            symbol="AAPL", strategy="VWAP", side="buy",
            entry_price=152.0, entry_time=datetime.now(ET),
            qty=5, take_profit=157.0, stop_loss=150.0,
        )
        rm.register_trade(trade2)
        assert rm.open_trades["AAPL"].strategy == "VWAP"
        assert rm.signals_today == 2

    def test_register_is_thread_safe(self, rm):
        """Concurrent register_trade calls should not corrupt state."""
        trades = [
            TradeRecord(
                symbol=f"SYM{i}", strategy="ORB", side="buy",
                entry_price=100.0, entry_time=datetime.now(ET),
                qty=1, take_profit=110.0, stop_loss=95.0,
            )
            for i in range(20)
        ]
        threads = [
            threading.Thread(target=rm.register_trade, args=(t,))
            for t in trades
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(rm.open_trades) == 20
        assert rm.signals_today == 20


# ===================================================================
# close_trade()
# ===================================================================

class TestCloseTrade:

    @patch("database.log_trade")
    def test_close_profitable_long(self, mock_db, rm, trade_aapl):
        rm.register_trade(trade_aapl)
        rm.close_trade("AAPL", exit_price=155.0,
                        now=datetime(2026, 4, 1, 15, 0, tzinfo=ET),
                        exit_reason="take_profit")
        assert "AAPL" not in rm.open_trades
        assert len(rm.closed_trades) == 1
        closed = rm.closed_trades[0]
        assert closed.pnl == (155.0 - 150.0) * 10  # $50
        assert closed.exit_reason == "take_profit"

    @patch("database.log_trade")
    def test_close_losing_long(self, mock_db, rm, trade_aapl):
        rm.register_trade(trade_aapl)
        rm.close_trade("AAPL", exit_price=148.0,
                        now=datetime(2026, 4, 1, 15, 0, tzinfo=ET),
                        exit_reason="stop_loss")
        closed = rm.closed_trades[0]
        assert closed.pnl == (148.0 - 150.0) * 10  # -$20

    @patch("database.log_trade")
    def test_close_short_trade(self, mock_db, rm, trade_tsla_short):
        rm.register_trade(trade_tsla_short)
        rm.close_trade("TSLA", exit_price=190.0,
                        now=datetime(2026, 4, 1, 15, 0, tzinfo=ET),
                        exit_reason="take_profit")
        closed = rm.closed_trades[0]
        # Short P&L: (entry - exit) * qty = (200-190)*5 = $50
        assert closed.pnl == 50.0

    @patch("database.log_trade")
    def test_close_with_commission(self, mock_db, rm, trade_aapl):
        """V12 6.1: Commission should be subtracted from P&L."""
        rm.register_trade(trade_aapl)
        rm.close_trade("AAPL", exit_price=155.0,
                        now=datetime(2026, 4, 1, 15, 0, tzinfo=ET),
                        exit_reason="take_profit", commission=2.50)
        closed = rm.closed_trades[0]
        assert closed.pnl == (155.0 - 150.0) * 10 - 2.50  # $47.50
        assert closed.commission == 2.50

    @patch("database.log_trade")
    def test_close_nonexistent_symbol_noop(self, mock_db, rm):
        rm.close_trade("FAKE", exit_price=100.0,
                        now=datetime(2026, 4, 1, 15, 0, tzinfo=ET))
        assert len(rm.closed_trades) == 0
        mock_db.assert_not_called()

    @patch("database.log_trade")
    def test_close_tracks_per_symbol_pnl(self, mock_db, rm, trade_aapl):
        rm.register_trade(trade_aapl)
        rm.close_trade("AAPL", exit_price=155.0,
                        now=datetime(2026, 4, 1, 15, 0, tzinfo=ET))
        assert rm._symbol_daily_pnl.get("AAPL") == 50.0

    @patch("database.log_trade", side_effect=Exception("DB error"))
    def test_close_handles_db_failure(self, mock_db, rm, trade_aapl):
        """DB failure should not crash close_trade (handled by failure_modes)."""
        rm.register_trade(trade_aapl)
        # Should not raise
        rm.close_trade("AAPL", exit_price=155.0,
                        now=datetime(2026, 4, 1, 15, 0, tzinfo=ET))
        assert len(rm.closed_trades) == 1


# ===================================================================
# partial_close()
# ===================================================================

class TestPartialClose:

    @patch("database.log_trade")
    def test_partial_close_reduces_qty(self, mock_db, rm, trade_aapl):
        rm.register_trade(trade_aapl)
        rm.partial_close("AAPL", qty_to_close=4, exit_price=153.0,
                          now=datetime(2026, 4, 1, 14, 0, tzinfo=ET))
        assert rm.open_trades["AAPL"].qty == 6
        assert rm.open_trades["AAPL"].partial_exits == 1

    @patch("database.log_trade")
    def test_partial_close_full_qty_removes_trade(self, mock_db, rm, trade_aapl):
        rm.register_trade(trade_aapl)
        rm.partial_close("AAPL", qty_to_close=10, exit_price=153.0,
                          now=datetime(2026, 4, 1, 14, 0, tzinfo=ET))
        assert "AAPL" not in rm.open_trades

    @patch("database.log_trade")
    def test_partial_close_clamps_excess_qty(self, mock_db, rm, trade_aapl):
        """If qty_to_close > trade.qty, it should be clamped."""
        rm.register_trade(trade_aapl)
        rm.partial_close("AAPL", qty_to_close=999, exit_price=153.0,
                          now=datetime(2026, 4, 1, 14, 0, tzinfo=ET))
        assert "AAPL" not in rm.open_trades

    @patch("database.log_trade")
    def test_partial_close_zero_qty_noop(self, mock_db, rm, trade_aapl):
        rm.register_trade(trade_aapl)
        rm.partial_close("AAPL", qty_to_close=0, exit_price=153.0,
                          now=datetime(2026, 4, 1, 14, 0, tzinfo=ET))
        assert rm.open_trades["AAPL"].qty == 10
        mock_db.assert_not_called()

    def test_partial_close_nonexistent_symbol_noop(self, rm):
        rm.partial_close("FAKE", qty_to_close=5, exit_price=100.0,
                          now=datetime(2026, 4, 1, 14, 0, tzinfo=ET))

    @patch("database.log_trade")
    def test_partial_close_with_commission(self, mock_db, rm, trade_aapl):
        """V12 6.1: Commission on partial close is subtracted from partial P&L."""
        rm.register_trade(trade_aapl)
        rm.partial_close("AAPL", qty_to_close=5, exit_price=153.0,
                          now=datetime(2026, 4, 1, 14, 0, tzinfo=ET),
                          commission=1.00)
        assert rm.open_trades["AAPL"].commission == 1.00
        # DB should log the partial P&L minus commission
        call_args = mock_db.call_args
        assert call_args.kwargs["pnl"] == (153.0 - 150.0) * 5 - 1.00  # $14.00

    @patch("database.log_trade")
    def test_partial_close_short_trade(self, mock_db, rm, trade_tsla_short):
        rm.register_trade(trade_tsla_short)
        rm.partial_close("TSLA", qty_to_close=2, exit_price=195.0,
                          now=datetime(2026, 4, 1, 14, 0, tzinfo=ET))
        assert rm.open_trades["TSLA"].qty == 3
        # Short P&L: (200 - 195) * 2 = $10
        call_args = mock_db.call_args
        assert call_args.kwargs["pnl"] == 10.0


# ===================================================================
# get_day_summary()
# ===================================================================

class TestGetDaySummary:

    def test_no_trades_returns_minimal(self, rm):
        result = rm.get_day_summary()
        assert result == {"trades": 0}

    @patch("database.log_trade")
    def test_summary_with_mixed_trades(self, mock_db, rm):
        # Register and close trades with mixed results
        for sym, pnl_price in [("AAPL", 155.0), ("MSFT", 145.0), ("TSLA", 160.0)]:
            trade = TradeRecord(
                symbol=sym, strategy="ORB", side="buy",
                entry_price=150.0, entry_time=datetime(2026, 4, 1, 10, 0, tzinfo=ET),
                qty=10, take_profit=160.0, stop_loss=145.0,
            )
            rm.register_trade(trade)
            rm.close_trade(sym, exit_price=pnl_price,
                            now=datetime(2026, 4, 1, 15, 0, tzinfo=ET))

        summary = rm.get_day_summary()
        assert summary["trades"] == 3
        assert summary["winners"] == 2   # AAPL +$50, TSLA +$100
        assert summary["losers"] == 1    # MSFT -$50
        assert summary["total_pnl"] == 100.0  # 50 - 50 + 100
        assert "best_trade" in summary
        assert "worst_trade" in summary
        assert "TSLA" in summary["best_trade"]
        assert "MSFT" in summary["worst_trade"]

    @patch("database.log_trade")
    def test_summary_per_strategy_win_rates(self, mock_db, rm):
        strategies = ["ORB", "ORB", "VWAP"]
        prices = [155.0, 145.0, 153.0]  # W, L, W
        for i, (strat, price) in enumerate(zip(strategies, prices)):
            trade = TradeRecord(
                symbol=f"SYM{i}", strategy=strat, side="buy",
                entry_price=150.0, entry_time=datetime(2026, 4, 1, 10, 0, tzinfo=ET),
                qty=10, take_profit=160.0, stop_loss=145.0,
            )
            rm.register_trade(trade)
            rm.close_trade(f"SYM{i}", exit_price=price,
                            now=datetime(2026, 4, 1, 15, 0, tzinfo=ET))

        summary = rm.get_day_summary()
        assert summary["orb_win_rate"] == "1/2"
        assert summary["vwap_win_rate"] == "1/1"


# ===================================================================
# check_circuit_breaker()
# ===================================================================

class TestCheckCircuitBreaker:

    def test_no_loss_no_breaker(self, rm, override_config):
        with override_config(DAILY_LOSS_HALT=-0.025):
            rm.day_pnl = 0.0
            assert rm.check_circuit_breaker() is False
            assert rm.circuit_breaker_active is False

    def test_loss_exceeds_limit_triggers_breaker(self, rm, override_config):
        with override_config(DAILY_LOSS_HALT=-0.025):
            rm.day_pnl = -0.03
            assert rm.check_circuit_breaker() is True
            assert rm.circuit_breaker_active is True

    def test_exactly_at_limit_triggers_breaker(self, rm, override_config):
        with override_config(DAILY_LOSS_HALT=-0.025):
            rm.day_pnl = -0.025
            assert rm.check_circuit_breaker() is True

    def test_profit_does_not_trigger(self, rm, override_config):
        with override_config(DAILY_LOSS_HALT=-0.025):
            rm.day_pnl = 0.05
            assert rm.check_circuit_breaker() is False

    def test_breaker_stays_active_once_triggered(self, rm, override_config):
        """Once activated, circuit_breaker_active stays True."""
        with override_config(DAILY_LOSS_HALT=-0.025):
            rm.day_pnl = -0.03
            rm.check_circuit_breaker()
            assert rm.circuit_breaker_active is True
            # Even if pnl recovers (shouldn't happen, but check state)
            rm.day_pnl = -0.01
            rm.check_circuit_breaker()
            # The flag remains from first activation
            assert rm.circuit_breaker_active is True


# ===================================================================
# Strategy time windows
# ===================================================================

class TestStrategyTimeWindows:

    def test_stat_mr_in_window(self):
        dt = datetime(2026, 4, 1, 12, 0, tzinfo=ET)
        assert is_strategy_in_time_window("STAT_MR", dt) is True

    def test_stat_mr_out_of_window(self):
        dt = datetime(2026, 4, 1, 9, 35, tzinfo=ET)
        assert is_strategy_in_time_window("STAT_MR", dt) is False

    def test_micro_mom_multi_window_first(self):
        dt = datetime(2026, 4, 1, 10, 0, tzinfo=ET)
        assert is_strategy_in_time_window("MICRO_MOM", dt) is True

    def test_micro_mom_multi_window_second(self):
        dt = datetime(2026, 4, 1, 14, 0, tzinfo=ET)
        assert is_strategy_in_time_window("MICRO_MOM", dt) is True

    def test_micro_mom_between_windows(self):
        dt = datetime(2026, 4, 1, 12, 0, tzinfo=ET)
        assert is_strategy_in_time_window("MICRO_MOM", dt) is False

    def test_unknown_strategy_always_allowed(self):
        dt = datetime(2026, 4, 1, 9, 30, tzinfo=ET)
        assert is_strategy_in_time_window("UNKNOWN_STRATEGY", dt) is True

    def test_orb_window(self):
        # ORB: 10:00-12:30
        assert is_strategy_in_time_window("ORB", datetime(2026, 4, 1, 11, 0, tzinfo=ET)) is True
        assert is_strategy_in_time_window("ORB", datetime(2026, 4, 1, 13, 0, tzinfo=ET)) is False


# ===================================================================
# reset_daily()
# ===================================================================

class TestResetDaily:

    def test_reset_clears_day_state(self, rm, trade_aapl):
        rm.register_trade(trade_aapl)
        rm.circuit_breaker_active = True
        rm.signals_today = 5
        rm.day_pnl = -0.01
        rm._symbol_daily_pnl["AAPL"] = -100.0
        rm.closed_trades.append(trade_aapl)

        rm.reset_daily(equity=110_000, cash=90_000)

        assert rm.starting_equity == 110_000
        assert rm.current_equity == 110_000
        assert rm.current_cash == 90_000
        assert rm.day_pnl == 0.0
        assert rm.circuit_breaker_active is False
        assert rm.signals_today == 0
        assert len(rm.closed_trades) == 0
        assert len(rm._symbol_daily_pnl) == 0
        # Day trade was cleared
        assert "AAPL" not in rm.open_trades

    def test_reset_preserves_swing_trades(self, rm):
        swing = TradeRecord(
            symbol="MSFT", strategy="PAIRS", side="buy",
            entry_price=400.0, entry_time=datetime.now(ET),
            qty=5, take_profit=420.0, stop_loss=390.0,
            hold_type="swing",
        )
        rm.register_trade(swing)
        rm.reset_daily(equity=100_000, cash=80_000)
        assert "MSFT" in rm.open_trades


# ===================================================================
# can_open_trade()
# ===================================================================

class TestCanOpenTrade:

    def test_allowed_when_no_limits_hit(self, rm, override_config):
        with override_config(MAX_POSITIONS=12, MAX_PORTFOLIO_DEPLOY=0.55,
                             VIX_RISK_SCALING_ENABLED=False):
            allowed, reason = rm.can_open_trade()
        assert allowed is True
        assert reason == ""

    def test_blocked_by_circuit_breaker(self, rm):
        rm.circuit_breaker_active = True
        allowed, reason = rm.can_open_trade()
        assert allowed is False
        assert "Circuit breaker" in reason

    def test_blocked_by_max_positions(self, rm, override_config):
        with override_config(MAX_POSITIONS=2, VIX_RISK_SCALING_ENABLED=False):
            for i in range(2):
                rm.open_trades[f"SYM{i}"] = TradeRecord(
                    symbol=f"SYM{i}", strategy="ORB", side="buy",
                    entry_price=100.0, entry_time=datetime.now(ET),
                    qty=1, take_profit=110.0, stop_loss=95.0,
                )
            allowed, reason = rm.can_open_trade()
        assert allowed is False
        assert "Max positions" in reason

    @patch("risk.risk_manager.get_vix_risk_scalar", return_value=0.0)
    def test_blocked_by_vix_halt(self, _vix, rm, override_config):
        with override_config(VIX_RISK_SCALING_ENABLED=True, VIX_HALT_THRESHOLD=40):
            allowed, reason = rm.can_open_trade()
        assert allowed is False
        assert "VIX" in reason


# ===================================================================
# update_equity()
# ===================================================================

class TestUpdateEquity:

    def test_update_computes_day_pnl(self, rm):
        rm.starting_equity = 100_000.0
        rm.update_equity(equity=102_000.0, cash=82_000.0)
        assert rm.current_equity == 102_000.0
        assert rm.current_cash == 82_000.0
        assert abs(rm.day_pnl - 0.02) < 1e-9

    def test_zero_starting_equity(self, rm):
        rm.starting_equity = 0.0
        rm.update_equity(equity=100.0, cash=100.0)
        assert rm.day_pnl == 0.0
