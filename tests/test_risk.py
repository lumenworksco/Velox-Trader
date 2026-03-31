"""Tests for risk.py — RiskManager, position sizing, circuit breaker, VIX scaling."""

from datetime import datetime
from unittest.mock import patch

import pytest

from risk import RiskManager, TradeRecord, get_vix_risk_scalar
from conftest import _make_trade, ET


# ===================================================================
# Circuit Breaker
# ===================================================================

class TestCircuitBreaker:
    def test_circuit_breaker_triggers(self, override_config):
        """day_pnl <= DAILY_LOSS_HALT activates the circuit breaker."""
        with override_config(DAILY_LOSS_HALT=-0.025):
            rm = RiskManager(starting_equity=100_000, current_equity=97_000)
            rm.day_pnl = -0.03  # -3%, below the -2.5% threshold

            result = rm.check_circuit_breaker()

            assert result is True
            assert rm.circuit_breaker_active is True

    def test_circuit_breaker_inactive(self, override_config):
        """day_pnl above threshold keeps breaker inactive."""
        with override_config(DAILY_LOSS_HALT=-0.025):
            rm = RiskManager(starting_equity=100_000, current_equity=99_000)
            rm.day_pnl = -0.01  # -1%, above the -2.5% threshold

            result = rm.check_circuit_breaker()

            assert result is False
            assert rm.circuit_breaker_active is False

    def test_circuit_breaker_exactly_at_limit(self, override_config):
        """day_pnl exactly at DAILY_LOSS_HALT triggers the breaker."""
        with override_config(DAILY_LOSS_HALT=-0.025):
            rm = RiskManager(starting_equity=100_000)
            rm.day_pnl = -0.025

            result = rm.check_circuit_breaker()

            assert result is True
            assert rm.circuit_breaker_active is True


# ===================================================================
# can_open_trade — position limits
# ===================================================================

class TestCanOpenTrade:
    def test_can_open_trade_max_positions(self, override_config):
        """Returns False when MAX_POSITIONS is reached."""
        with override_config(MAX_POSITIONS=2, VIX_RISK_SCALING_ENABLED=False):
            rm = RiskManager(current_equity=100_000)
            rm.open_trades["AAPL"] = _make_trade(symbol="AAPL")
            rm.open_trades["MSFT"] = _make_trade(symbol="MSFT")

            allowed, reason = rm.can_open_trade()

            assert allowed is False
            assert "Max positions" in reason

    def test_can_open_trade_vix_halt(self, override_config):
        """Returns False when VIX scalar is 0.0 (VIX above halt threshold)."""
        with override_config(MAX_POSITIONS=10, VIX_RISK_SCALING_ENABLED=True,
                             VIX_HALT_THRESHOLD=40):
            rm = RiskManager(current_equity=100_000)

            with patch("risk.risk_manager.get_vix_risk_scalar", return_value=0.0):
                allowed, reason = rm.can_open_trade()

            assert allowed is False
            assert "VIX" in reason

    def test_can_open_trade_allowed(self, override_config):
        """Allows trade when all limits are clear."""
        with override_config(MAX_POSITIONS=6, VIX_RISK_SCALING_ENABLED=False,
                             MAX_PORTFOLIO_DEPLOY=0.90):
            rm = RiskManager(current_equity=100_000)

            allowed, reason = rm.can_open_trade()

            assert allowed is True
            assert reason == ""

    def test_can_open_trade_circuit_breaker_blocks(self, override_config):
        """Circuit breaker active blocks new trades."""
        with override_config(VIX_RISK_SCALING_ENABLED=False):
            rm = RiskManager(current_equity=100_000)
            rm.circuit_breaker_active = True

            allowed, reason = rm.can_open_trade()

            assert allowed is False
            assert "Circuit breaker" in reason


# ===================================================================
# Position Sizing
# ===================================================================

class TestPositionSize:
    def test_position_size_basic(self, override_config):
        """Basic ATR-based sizing: risk 1% of portfolio."""
        with override_config(RISK_PER_TRADE_PCT=0.01, MAX_POSITION_PCT=0.20,
                             MIN_POSITION_VALUE=100, BEARISH_SIZE_CUT=0.40,
                             VIX_RISK_SCALING_ENABLED=False,
                             DYNAMIC_ALLOCATION=False,
                             MAX_PORTFOLIO_DEPLOY=0.90,
                             SHORT_SIZE_MULTIPLIER=0.75):
            rm = RiskManager(current_equity=100_000)
            # entry=100, stop=98 => risk_per_share=$2
            # risk_per_trade = 100_000 * 0.01 = $1000
            # shares = 1000 / 2 = 500 => position_value = 500 * 100 = 50,000
            # cap: 100_000 * 0.20 = 20,000 => capped to 20,000
            # qty = 20,000 / 100 = 200
            qty = rm.calculate_position_size(
                entry_price=100.0, stop_price=98.0, regime="BULLISH")

            assert qty == 200

    def test_position_size_bearish_cut(self, override_config):
        """Bearish regime reduces position size by BEARISH_SIZE_CUT."""
        with override_config(RISK_PER_TRADE_PCT=0.01, MAX_POSITION_PCT=0.20,
                             MIN_POSITION_VALUE=100, BEARISH_SIZE_CUT=0.40,
                             VIX_RISK_SCALING_ENABLED=False,
                             DYNAMIC_ALLOCATION=False,
                             MAX_PORTFOLIO_DEPLOY=0.90,
                             SHORT_SIZE_MULTIPLIER=0.75):
            rm = RiskManager(current_equity=100_000)

            qty_bull = rm.calculate_position_size(
                entry_price=100.0, stop_price=98.0, regime="BULLISH")
            qty_bear = rm.calculate_position_size(
                entry_price=100.0, stop_price=98.0, regime="BEARISH")

            # Bearish should be 60% of bullish (40% cut)
            assert qty_bear < qty_bull
            assert qty_bear == int(200 * 0.60)  # 120

    def test_position_size_vix_scaling(self, override_config):
        """VIX scalar < 1.0 reduces position size proportionally."""
        with override_config(RISK_PER_TRADE_PCT=0.01, MAX_POSITION_PCT=0.20,
                             MIN_POSITION_VALUE=100, BEARISH_SIZE_CUT=0.40,
                             VIX_RISK_SCALING_ENABLED=True,
                             DYNAMIC_ALLOCATION=False,
                             MAX_PORTFOLIO_DEPLOY=0.90,
                             SHORT_SIZE_MULTIPLIER=0.75):
            rm = RiskManager(current_equity=100_000)

            with patch("risk.risk_manager.get_vix_risk_scalar", return_value=0.50):
                qty = rm.calculate_position_size(
                    entry_price=100.0, stop_price=98.0, regime="BULLISH")

            # Base capped to 200 shares ($20k), VIX * 0.5 = $10k => 100 shares
            assert qty == 100

    def test_position_size_zero_risk_per_share(self):
        """Returns 0 when entry == stop (division by zero guard)."""
        rm = RiskManager(current_equity=100_000)
        qty = rm.calculate_position_size(
            entry_price=100.0, stop_price=100.0, regime="BULLISH")
        assert qty == 0

    def test_position_size_zero_equity(self):
        """Returns 0 when equity is 0."""
        rm = RiskManager(current_equity=0)
        qty = rm.calculate_position_size(
            entry_price=100.0, stop_price=98.0, regime="BULLISH")
        assert qty == 0


# ===================================================================
# Trade Lifecycle
# ===================================================================

class TestTradeLifecycle:
    def test_register_and_close_trade(self):
        """Full lifecycle: register -> close -> PnL recorded."""
        rm = RiskManager(current_equity=100_000, starting_equity=100_000)
        trade = _make_trade(symbol="AAPL", entry_price=150.0, qty=10,
                            side="buy")

        rm.register_trade(trade)
        assert "AAPL" in rm.open_trades
        assert rm.signals_today == 1

        now = datetime(2026, 3, 13, 14, 0, tzinfo=ET)
        rm.close_trade("AAPL", exit_price=155.0, now=now,
                        exit_reason="take_profit")

        assert "AAPL" not in rm.open_trades
        assert len(rm.closed_trades) == 1
        closed = rm.closed_trades[0]
        assert closed.pnl == 50.0  # (155 - 150) * 10
        assert closed.exit_reason == "take_profit"

    def test_close_trade_sell_side(self):
        """Short trade PnL: (entry - exit) * qty."""
        rm = RiskManager(current_equity=100_000, starting_equity=100_000)
        trade = _make_trade(symbol="MSFT", entry_price=400.0, qty=5,
                            side="sell")
        rm.register_trade(trade)

        now = datetime(2026, 3, 13, 14, 0, tzinfo=ET)
        rm.close_trade("MSFT", exit_price=390.0, now=now,
                        exit_reason="take_profit")

        closed = rm.closed_trades[0]
        assert closed.pnl == 50.0  # (400 - 390) * 5

    def test_partial_close(self):
        """V4: partial close reduces qty and increments partial_exits."""
        rm = RiskManager(current_equity=100_000, starting_equity=100_000)
        trade = _make_trade(symbol="AAPL", entry_price=150.0, qty=30,
                            side="buy")
        rm.register_trade(trade)

        now = datetime(2026, 3, 13, 14, 0, tzinfo=ET)
        rm.partial_close("AAPL", qty_to_close=10, exit_price=155.0,
                          now=now, exit_reason="partial_tp_1")

        assert rm.open_trades["AAPL"].qty == 20
        assert rm.open_trades["AAPL"].partial_exits == 1

    def test_partial_close_full_qty_removes_trade(self):
        """Partial closing entire qty removes the trade."""
        rm = RiskManager(current_equity=100_000, starting_equity=100_000)
        trade = _make_trade(symbol="AAPL", entry_price=150.0, qty=10,
                            side="buy")
        rm.register_trade(trade)

        now = datetime(2026, 3, 13, 14, 0, tzinfo=ET)
        rm.partial_close("AAPL", qty_to_close=10, exit_price=155.0, now=now)

        assert "AAPL" not in rm.open_trades


# ===================================================================
# VIX Risk Scalar
# ===================================================================

class TestVixRiskScalar:
    """Test VIX level scaling for get_vix_risk_scalar.

    V11.4: Updated to smooth linear interpolation between breakpoints:
    (12, 1.0), (15, 0.95), (20, 0.85), (25, 0.70), (30, 0.50), (35, 0.35)
    direction_adj is 1.0 when VIX history has <3 readings (fresh start).
    """

    @pytest.mark.parametrize("vix_level, expected_scalar", [
        (10.0, 1.0),    # Below first breakpoint
        (12.0, 1.0),    # At first breakpoint
        (15.0, 0.95),   # At breakpoint
        (20.0, 0.85),   # At breakpoint
        (25.0, 0.70),   # At breakpoint
        (30.0, 0.50),   # At breakpoint
        (35.0, 0.35),   # At last breakpoint
        (40.0, 0.0),    # VIX >= 40 (halt)
        (50.0, 0.0),    # VIX >= 40 (halt)
    ])
    def test_get_vix_risk_scalar_thresholds(self, vix_level, expected_scalar,
                                             override_config):
        import risk.risk_manager as _rm
        _rm._vix_history.clear()

        with override_config(VIX_RISK_SCALING_ENABLED=True,
                             VIX_HALT_THRESHOLD=40):
            with patch("risk.risk_manager.get_vix_level", return_value=vix_level):
                scalar = get_vix_risk_scalar()
                assert abs(scalar - expected_scalar) < 0.01, (
                    f"VIX={vix_level}: expected ~{expected_scalar}, got {scalar}"
                )

    def test_vix_scaling_disabled_returns_1(self, override_config):
        """When VIX_RISK_SCALING_ENABLED=False, scalar is always 1.0."""
        with override_config(VIX_RISK_SCALING_ENABLED=False):
            scalar = get_vix_risk_scalar()
            assert scalar == 1.0


# ===================================================================
# update_equity and reset_daily
# ===================================================================

class TestEquityUpdates:
    def test_update_equity_calculates_day_pnl(self):
        rm = RiskManager(starting_equity=100_000)
        rm.update_equity(equity=101_000, cash=80_000)

        assert rm.current_equity == 101_000
        assert rm.current_cash == 80_000
        assert rm.day_pnl == pytest.approx(0.01)

    def test_reset_daily_preserves_swing_trades(self):
        rm = RiskManager()
        rm.open_trades["AAPL"] = _make_trade(symbol="AAPL", hold_type="day")
        rm.open_trades["NVDA"] = _make_trade(symbol="NVDA", hold_type="swing")

        rm.reset_daily(equity=100_000, cash=80_000)

        assert "AAPL" not in rm.open_trades
        assert "NVDA" in rm.open_trades
