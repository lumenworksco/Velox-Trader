"""Comprehensive tests for engine/exit_orchestrator.py — unified exit decision engine.

Tests cover:
- check_exits() with various position states
- Double-exit prevention (same symbol not exited twice)
- Profit tier progression (1.5% and 2.5%)
- Dead signal detection (day strategy, >30 min, <0.3% move)
- Scale-out loser logic (-1% triggers 50% partial)
- ATR trailing stop updates
- execute_exits() with mock close functions
- State cleanup after full close
"""

import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers: lightweight Trade and RiskManager stand-ins
# ---------------------------------------------------------------------------

@dataclass
class FakeTrade:
    symbol: str = "AAPL"
    strategy: str = "STAT_MR"
    side: str = "buy"
    entry_price: float = 100.0
    qty: int = 100
    take_profit: float = 105.0
    stop_loss: float = 95.0
    entry_time: datetime = field(default_factory=lambda: datetime(2026, 4, 3, 10, 0, tzinfo=ET))
    entry_atr: float = 1.5
    partial_exits: int = 0
    partial_closed_qty: int = 0
    highest_price_seen: float = 0.0
    lowest_price_seen: float = 0.0
    order_id: str = "mock-001"
    hold_type: str = "day"
    pnl: float = 0.0
    exit_price: float | None = None


@dataclass
class FakeRiskManager:
    open_trades: dict = field(default_factory=dict)
    current_equity: float = 100_000.0
    starting_equity: float = 100_000.0
    day_pnl: float = 0.0
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def close_trade(self, symbol, exit_price, now, exit_reason="", commission=0.0):
        self.open_trades.pop(symbol, None)

    def partial_close(self, symbol, qty, price, now, reason=""):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_config():
    """Provide a minimal mock config module with all attrs exit_orchestrator needs."""
    import types
    cfg = types.ModuleType("config")
    cfg.ET = ET
    cfg.ATR_TRAILING_ENABLED = True
    cfg.ATR_TRAIL_MULT = {"STAT_MR": 2.0, "VWAP": 2.0, "ORB": 2.5}
    cfg.ATR_TRAIL_ACTIVATION = 0.5
    cfg.SCALED_TP_ENABLED = False
    cfg.BREAKEVEN_STOP_ENABLED = True
    cfg.ADAPTIVE_EXITS_ENABLED = False
    cfg.RSI_EXIT_THRESHOLD = 70
    cfg.ATR_EXPANSION_MULT = 2.5
    cfg.TELEGRAM_ENABLED = False
    cfg.MAX_NEW_ENTRIES_PER_SCAN = 5
    cfg.MICRO_BETA_TABLE = {}
    cfg.UNIVERSE = []
    cfg.ALLOW_SHORT = True
    cfg.MAX_SECTOR_EXPOSURE = 0.30
    cfg.SECTOR_MAP = {}

    with patch.dict(sys.modules, {"config": cfg}):
        yield cfg


@pytest.fixture
def emod(_mock_config):
    """Import and return a freshly-reloaded exit_orchestrator module."""
    with patch.dict(sys.modules, {
        "adaptive_exit_manager": MagicMock(),
        "risk": MagicMock(),
        "engine.events": MagicMock(),
    }):
        import importlib
        mod = importlib.import_module("engine.exit_orchestrator")
        importlib.reload(mod)
        yield mod


@pytest.fixture
def orchestrator(emod):
    """Return a fresh ExitOrchestrator with adaptive exits disabled."""
    orch = emod.ExitOrchestrator()
    orch._adaptive_mgr = None
    return orch


def _patch_prices(emod, prices: dict):
    """Context manager to mock _get_current_prices on the loaded module."""
    return patch.object(emod, "_get_current_prices", return_value=prices)


# ===================================================================
# check_exits -- basic
# ===================================================================

class TestCheckExitsBasic:
    """Basic check_exits behavior: empty positions, price fetch fallback."""

    def test_no_open_trades_returns_empty(self, orchestrator):
        rm = FakeRiskManager()
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)
        actions = orchestrator.check_exits(rm, now)
        assert actions == []

    def test_zero_current_price_is_skipped(self, orchestrator, emod):
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = FakeTrade()
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)
        with _patch_prices(emod, {"AAPL": 0.0}):
            actions = orchestrator.check_exits(rm, now)
        assert actions == []

    def test_negative_entry_price_is_skipped(self, orchestrator, emod):
        trade = FakeTrade(entry_price=-1.0)
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)
        with _patch_prices(emod, {"AAPL": 100.0}):
            actions = orchestrator.check_exits(rm, now)
        assert actions == []


# ===================================================================
# Double-exit prevention
# ===================================================================

class TestDoubleExitPrevention:
    """A symbol with a full-close from Phase 1 must not appear again in Phase 2."""

    def test_strategy_full_close_prevents_phase2_exits(self, orchestrator, emod):
        trade = FakeTrade(strategy="STAT_MR", entry_price=100.0)
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = datetime(2026, 4, 3, 12, 0, tzinfo=ET)

        mock_strategy = MagicMock()
        mock_strategy.check_exits.return_value = [
            {"symbol": "AAPL", "action": "full", "reason": "strategy_exit"}
        ]

        with _patch_prices(emod, {"AAPL": 100.0}):
            actions = orchestrator.check_exits(
                rm, now, strategies={"STAT_MR": mock_strategy}
            )

        full_exits = [a for a in actions if a.symbol == "AAPL" and a.action == "full"]
        assert len(full_exits) == 1
        assert full_exits[0].reason == "strategy_exit"

    def test_two_positions_one_full_close_other_still_checked(self, orchestrator, emod):
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = FakeTrade(symbol="AAPL", entry_price=100.0)
        rm.open_trades["GOOG"] = FakeTrade(
            symbol="GOOG", strategy="STAT_MR", entry_price=200.0,
        )
        now = datetime(2026, 4, 3, 12, 0, tzinfo=ET)

        mock_strategy = MagicMock()
        mock_strategy.check_exits.return_value = [
            {"symbol": "AAPL", "action": "full", "reason": "strategy_tp"}
        ]

        with _patch_prices(emod, {"AAPL": 100.0, "GOOG": 200.2}):
            actions = orchestrator.check_exits(
                rm, now, strategies={"STAT_MR": mock_strategy}
            )

        symbols_exited = {a.symbol for a in actions}
        assert "AAPL" in symbols_exited
        # GOOG should NOT be blocked by AAPL's full_close_symbols set


# ===================================================================
# Profit tier progression
# ===================================================================

class TestProfitTiers:
    """Profit tier partial exits at +1.5% and +2.5%."""

    def test_tier1_at_1_5pct(self, orchestrator, emod):
        trade = FakeTrade(entry_price=100.0, qty=100, side="buy")
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = datetime(2026, 4, 3, 10, 5, tzinfo=ET)

        with _patch_prices(emod, {"AAPL": 101.60}):
            actions = orchestrator.check_exits(rm, now)

        tier1 = [a for a in actions if "profit_tier_1" in a.reason]
        assert len(tier1) == 1
        assert tier1[0].action == "partial"
        assert tier1[0].qty == 33

    def test_tier2_only_after_tier1(self, orchestrator, emod):
        trade = FakeTrade(entry_price=100.0, qty=100, side="buy")
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = datetime(2026, 4, 3, 10, 5, tzinfo=ET)

        with _patch_prices(emod, {"AAPL": 102.60}):
            actions = orchestrator.check_exits(rm, now)

        tier1 = [a for a in actions if "profit_tier_1" in a.reason]
        tier2 = [a for a in actions if "profit_tier_2" in a.reason]
        assert len(tier1) == 1
        assert len(tier2) == 0

    def test_tier2_fires_after_tier1_state(self, orchestrator, emod):
        trade = FakeTrade(entry_price=100.0, qty=100, side="buy")
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = datetime(2026, 4, 3, 10, 5, tzinfo=ET)

        with _patch_prices(emod, {"AAPL": 101.60}):
            orchestrator.check_exits(rm, now)

        with _patch_prices(emod, {"AAPL": 102.60}):
            actions = orchestrator.check_exits(rm, now)

        tier2 = [a for a in actions if "profit_tier_2" in a.reason]
        assert len(tier2) == 1
        assert tier2[0].action == "partial"

    def test_no_tier_below_threshold(self, orchestrator, emod):
        trade = FakeTrade(entry_price=100.0, qty=100, side="buy")
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = datetime(2026, 4, 3, 10, 5, tzinfo=ET)

        with _patch_prices(emod, {"AAPL": 101.00}):
            actions = orchestrator.check_exits(rm, now)

        tier_actions = [a for a in actions if "profit_tier" in a.reason]
        assert len(tier_actions) == 0

    def test_sell_side_profit_tiers(self, orchestrator, emod):
        trade = FakeTrade(entry_price=100.0, qty=100, side="sell")
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = datetime(2026, 4, 3, 10, 5, tzinfo=ET)

        with _patch_prices(emod, {"AAPL": 98.40}):
            actions = orchestrator.check_exits(rm, now)

        tier1 = [a for a in actions if "profit_tier_1" in a.reason]
        assert len(tier1) == 1

    def test_no_more_tiers_after_tier2(self, orchestrator, emod):
        trade = FakeTrade(entry_price=100.0, qty=100, side="buy")
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = datetime(2026, 4, 3, 10, 5, tzinfo=ET)

        with _patch_prices(emod, {"AAPL": 101.60}):
            orchestrator.check_exits(rm, now)
        with _patch_prices(emod, {"AAPL": 102.60}):
            orchestrator.check_exits(rm, now)
        with _patch_prices(emod, {"AAPL": 104.00}):
            actions = orchestrator.check_exits(rm, now)

        tier_actions = [a for a in actions if "profit_tier" in a.reason]
        assert len(tier_actions) == 0


# ===================================================================
# Dead signal detection
# ===================================================================

class TestDeadSignal:
    """Positions with no conviction after 30 minutes get closed."""

    def test_dead_signal_fires_after_30_min(self, orchestrator, emod):
        entry_time = datetime(2026, 4, 3, 10, 0, tzinfo=ET)
        trade = FakeTrade(strategy="STAT_MR", entry_price=100.0, entry_time=entry_time)
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = entry_time + timedelta(minutes=35)

        with _patch_prices(emod, {"AAPL": 100.10}):
            actions = orchestrator.check_exits(rm, now)

        dead = [a for a in actions if a.reason == "dead_signal"]
        assert len(dead) == 1
        assert dead[0].action == "full"

    def test_no_dead_signal_before_30_min(self, orchestrator, emod):
        entry_time = datetime(2026, 4, 3, 10, 0, tzinfo=ET)
        trade = FakeTrade(strategy="STAT_MR", entry_price=100.0, entry_time=entry_time)
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = entry_time + timedelta(minutes=20)

        with _patch_prices(emod, {"AAPL": 100.10}):
            actions = orchestrator.check_exits(rm, now)

        dead = [a for a in actions if a.reason == "dead_signal"]
        assert len(dead) == 0

    def test_no_dead_signal_for_swing_strategy(self, orchestrator, emod):
        entry_time = datetime(2026, 4, 3, 10, 0, tzinfo=ET)
        trade = FakeTrade(strategy="PEAD", entry_price=100.0, entry_time=entry_time)
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = entry_time + timedelta(minutes=60)

        with _patch_prices(emod, {"AAPL": 100.10}):
            actions = orchestrator.check_exits(rm, now)

        dead = [a for a in actions if a.reason == "dead_signal"]
        assert len(dead) == 0

    def test_no_dead_signal_if_big_move(self, orchestrator, emod):
        entry_time = datetime(2026, 4, 3, 10, 0, tzinfo=ET)
        trade = FakeTrade(strategy="VWAP", entry_price=100.0, entry_time=entry_time)
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = entry_time + timedelta(minutes=45)

        with _patch_prices(emod, {"AAPL": 101.00}):
            actions = orchestrator.check_exits(rm, now)

        dead = [a for a in actions if a.reason == "dead_signal"]
        assert len(dead) == 0

    def test_dead_signal_for_all_day_strategies(self, orchestrator, emod):
        """STAT_MR, VWAP, MICRO_MOM, ORB should all detect dead signals."""
        entry_time = datetime(2026, 4, 3, 10, 0, tzinfo=ET)
        now = entry_time + timedelta(minutes=35)

        for strat in ["STAT_MR", "VWAP", "MICRO_MOM", "ORB"]:
            orchestrator._state.clear()
            trade = FakeTrade(symbol="AAPL", strategy=strat, entry_price=100.0,
                              entry_time=entry_time)
            rm = FakeRiskManager()
            rm.open_trades["AAPL"] = trade

            with _patch_prices(emod, {"AAPL": 100.10}):
                actions = orchestrator.check_exits(rm, now)

            dead = [a for a in actions if a.reason == "dead_signal"]
            assert len(dead) == 1, f"Dead signal should fire for {strat}"


# ===================================================================
# Scale-out loser
# ===================================================================

class TestScaleOutLoser:
    """Scale out 50% when unrealized loss exceeds -1.0%."""

    def test_scale_out_at_minus_1pct(self, orchestrator, emod):
        trade = FakeTrade(entry_price=100.0, qty=100, side="buy")
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = datetime(2026, 4, 3, 10, 5, tzinfo=ET)

        with _patch_prices(emod, {"AAPL": 98.90}):
            actions = orchestrator.check_exits(rm, now)

        loser = [a for a in actions if "scale_out_loser" in a.reason]
        assert len(loser) == 1
        assert loser[0].action == "partial"
        assert loser[0].qty == 50

    def test_scale_out_fires_only_once(self, orchestrator, emod):
        trade = FakeTrade(entry_price=100.0, qty=100, side="buy")
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = datetime(2026, 4, 3, 10, 5, tzinfo=ET)

        with _patch_prices(emod, {"AAPL": 98.50}):
            actions1 = orchestrator.check_exits(rm, now)

        with _patch_prices(emod, {"AAPL": 98.00}):
            actions2 = orchestrator.check_exits(rm, now)

        loser1 = [a for a in actions1 if "scale_out_loser" in a.reason]
        loser2 = [a for a in actions2 if "scale_out_loser" in a.reason]
        assert len(loser1) == 1
        assert len(loser2) == 0

    def test_no_scale_out_above_threshold(self, orchestrator, emod):
        trade = FakeTrade(entry_price=100.0, qty=100, side="buy")
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = datetime(2026, 4, 3, 10, 5, tzinfo=ET)

        with _patch_prices(emod, {"AAPL": 99.50}):
            actions = orchestrator.check_exits(rm, now)

        loser = [a for a in actions if "scale_out_loser" in a.reason]
        assert len(loser) == 0

    def test_sell_side_scale_out(self, orchestrator, emod):
        trade = FakeTrade(entry_price=100.0, qty=80, side="sell")
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = datetime(2026, 4, 3, 10, 5, tzinfo=ET)

        with _patch_prices(emod, {"AAPL": 101.20}):
            actions = orchestrator.check_exits(rm, now)

        loser = [a for a in actions if "scale_out_loser" in a.reason]
        assert len(loser) == 1
        assert loser[0].qty == 40


# ===================================================================
# ATR trailing stop
# ===================================================================

class TestATRTrailing:
    """ATR-based trailing stop updates and triggers."""

    def test_atr_trailing_triggers_full_close(self, orchestrator, emod):
        trade = FakeTrade(
            strategy="STAT_MR", entry_price=100.0, qty=50,
            entry_atr=1.0, side="buy", stop_loss=97.0,
            highest_price_seen=103.0,
        )
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        with _patch_prices(emod, {"AAPL": 100.9}):
            actions = orchestrator.check_exits(rm, now)

        atr_exits = [a for a in actions if "atr_trailing" in a.reason]
        assert len(atr_exits) == 1
        assert atr_exits[0].action == "full"

    def test_atr_trailing_not_active_below_activation(self, orchestrator, emod):
        trade = FakeTrade(
            strategy="STAT_MR", entry_price=100.0, qty=50,
            entry_atr=1.0, side="buy", stop_loss=97.0,
        )
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        with _patch_prices(emod, {"AAPL": 100.3}):
            actions = orchestrator.check_exits(rm, now)

        atr_exits = [a for a in actions if "atr_trailing" in a.reason]
        assert len(atr_exits) == 0

    def test_atr_trailing_ratchets_stop_upward(self, orchestrator, emod):
        trade = FakeTrade(
            strategy="STAT_MR", entry_price=100.0, qty=50,
            entry_atr=1.0, side="buy", stop_loss=97.0,
            highest_price_seen=102.0,
        )
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        with _patch_prices(emod, {"AAPL": 101.5}):
            actions = orchestrator.check_exits(rm, now)

        atr_exits = [a for a in actions if "atr_trailing" in a.reason]
        assert len(atr_exits) == 0
        assert trade.stop_loss == 100.0

    def test_no_atr_trailing_without_config(self, orchestrator, emod, _mock_config):
        _mock_config.ATR_TRAILING_ENABLED = False
        trade = FakeTrade(
            strategy="STAT_MR", entry_price=100.0, entry_atr=1.0,
            highest_price_seen=105.0, stop_loss=97.0,
        )
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        with _patch_prices(emod, {"AAPL": 99.0}):
            actions = orchestrator.check_exits(rm, now)

        atr_exits = [a for a in actions if "atr_trailing" in a.reason]
        assert len(atr_exits) == 0


# ===================================================================
# execute_exits -- mock order submission
# ===================================================================

class TestExecuteExits:
    """execute_exits submits orders, updates risk state, cleans up."""

    def _make_exec_and_data_mocks(self):
        """Create mock execution and data modules for lazy imports inside execute_exits."""
        mock_execution = MagicMock()
        mock_data = MagicMock()
        mock_data.get_filled_exit_price = MagicMock(return_value=101.0)
        mock_data.get_snapshot = MagicMock(return_value=None)
        return mock_execution, mock_data

    def test_full_close_calls_close_position(self, orchestrator, emod):
        trade = FakeTrade(symbol="AAPL", entry_price=100.0, qty=50, side="buy")
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        actions = [
            emod.ExitAction(symbol="AAPL", action="full", reason="test_exit", priority=10)
        ]

        mock_execution, mock_data = self._make_exec_and_data_mocks()

        with patch.dict(sys.modules, {
            "execution": mock_execution,
            "data": mock_data,
        }), patch.object(emod, "_emit_event"):
            orchestrator.execute_exits(actions, rm, now)

        mock_execution.close_position.assert_called_once_with("AAPL", reason="test_exit")

    def test_partial_close_calls_close_partial(self, orchestrator, emod):
        trade = FakeTrade(symbol="GOOG", entry_price=200.0, qty=60, side="buy")
        rm = FakeRiskManager()
        rm.open_trades["GOOG"] = trade
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        actions = [
            emod.ExitAction(symbol="GOOG", action="partial", qty=20, reason="profit_tier_1", priority=30)
        ]

        mock_execution, mock_data = self._make_exec_and_data_mocks()

        with patch.dict(sys.modules, {
            "execution": mock_execution,
            "data": mock_data,
        }), patch.object(emod, "_emit_event"), \
             patch.object(emod, "_get_price_for_symbol", return_value=202.0):
            orchestrator.execute_exits(actions, rm, now)

        mock_execution.close_partial_position.assert_called_once_with("GOOG", 20)


# ===================================================================
# State cleanup
# ===================================================================

class TestStateCleanup:
    """cleanup_state removes per-position exit tracking."""

    def test_cleanup_removes_state(self, orchestrator):
        orchestrator._get_state("AAPL").profit_tiers_taken = 1
        orchestrator._get_state("AAPL").scaled_out = True
        assert "AAPL" in orchestrator._state

        orchestrator.cleanup_state("AAPL")
        assert "AAPL" not in orchestrator._state

    def test_cleanup_nonexistent_symbol_is_safe(self, orchestrator):
        orchestrator.cleanup_state("NONEXISTENT")

    def test_full_close_cleans_up_state(self, orchestrator, emod):
        """After execute_exits full close, orchestrator state should be gone."""
        trade = FakeTrade(symbol="MSFT", entry_price=300.0, qty=10, side="buy")
        rm = FakeRiskManager()
        rm.open_trades["MSFT"] = trade
        orchestrator._get_state("MSFT").profit_tiers_taken = 2
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        actions = [emod.ExitAction(symbol="MSFT", action="full", reason="test", priority=10)]

        mock_execution = MagicMock()
        mock_data = MagicMock()
        mock_data.get_filled_exit_price = MagicMock(return_value=302.0)
        mock_data.get_snapshot = MagicMock(return_value=None)

        with patch.dict(sys.modules, {
            "execution": mock_execution,
            "data": mock_data,
        }), patch.object(emod, "_emit_event"):
            orchestrator.execute_exits(actions, rm, now)

        assert "MSFT" not in orchestrator._state


# ===================================================================
# Price extreme updates
# ===================================================================

class TestPriceExtremes:
    """update_price_extremes tracks highest/lowest for trailing stops."""

    def test_updates_highest_for_long(self, orchestrator):
        trade = FakeTrade(side="buy", highest_price_seen=100.0)
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade

        orchestrator.update_price_extremes(rm, {"AAPL": 105.0})
        assert trade.highest_price_seen == 105.0

    def test_does_not_lower_highest(self, orchestrator):
        trade = FakeTrade(side="buy", highest_price_seen=110.0)
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade

        orchestrator.update_price_extremes(rm, {"AAPL": 105.0})
        assert trade.highest_price_seen == 110.0

    def test_updates_lowest_for_short(self, orchestrator):
        trade = FakeTrade(side="sell", lowest_price_seen=100.0)
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade

        orchestrator.update_price_extremes(rm, {"AAPL": 95.0})
        assert trade.lowest_price_seen == 95.0

    def test_does_not_raise_lowest(self, orchestrator):
        trade = FakeTrade(side="sell", lowest_price_seen=90.0)
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = trade

        orchestrator.update_price_extremes(rm, {"AAPL": 95.0})
        assert trade.lowest_price_seen == 90.0
