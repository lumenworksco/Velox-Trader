"""Comprehensive tests for engine/exit_processor.py — V11.5 advanced exit logic.

Tests cover:
- check_advanced_exits() returns proper actions
- Profit tier tracking (_profit_tier_exits state)
- Dead signal detection (held > 30 min with < 0.3% move)
- Scale-out at -1% unrealized loss
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
# Lightweight stand-ins
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
def _mock_deps():
    """Mock all external deps so exit_processor can import cleanly."""
    import types
    cfg = types.ModuleType("config")
    cfg.ET = ET
    cfg.TELEGRAM_ENABLED = False
    cfg.ADAPTIVE_EXITS_ENABLED = False

    mock_data = MagicMock()
    mock_exec = MagicMock()
    mock_exec_core = MagicMock()
    mock_exec_core.get_order_commission = MagicMock(return_value=0.0)
    mock_risk = MagicMock()

    with patch.dict(sys.modules, {
        "config": cfg,
        "data": mock_data,
        "execution": mock_exec,
        "execution.core": mock_exec_core,
        "risk": mock_risk,
        "adaptive_exit_manager": MagicMock(),
        "engine.events": MagicMock(),
    }):
        # Set attributes the module expects on mock_risk
        mock_risk.RiskManager = FakeRiskManager
        mock_risk.get_vix_level = MagicMock(return_value=20.0)
        yield cfg


@pytest.fixture
def exit_processor(_mock_deps):
    """Import and return a fresh exit_processor module."""
    import importlib
    mod = importlib.import_module("engine.exit_processor")
    importlib.reload(mod)
    # Reset module-level state
    mod._profit_tier_exits.clear()
    mod._scaled_out.clear()
    return mod


# ===================================================================
# check_profit_tiers
# ===================================================================

class TestCheckProfitTiers:
    """Tiered profit taking at +1.5% and +2.5%."""

    def test_tier1_fires_at_1_5pct_for_long(self, exit_processor):
        trade = FakeTrade(entry_price=100.0, qty=100, side="buy")
        # 1.6% above entry
        actions = exit_processor.check_profit_tiers(trade, 101.60)
        assert len(actions) == 1
        assert actions[0]["reason"] == "profit_tier_1_1.5pct"
        assert actions[0]["action"] == "partial"
        assert actions[0]["qty"] == 33

    def test_tier1_fires_at_1_5pct_for_short(self, exit_processor):
        trade = FakeTrade(entry_price=100.0, qty=100, side="sell")
        # Price drops 1.6% — profit for shorts
        actions = exit_processor.check_profit_tiers(trade, 98.40)
        assert len(actions) == 1
        assert actions[0]["reason"] == "profit_tier_1_1.5pct"

    def test_no_tier1_below_threshold(self, exit_processor):
        trade = FakeTrade(entry_price=100.0, qty=100, side="buy")
        actions = exit_processor.check_profit_tiers(trade, 101.00)
        assert actions == []

    def test_tier2_requires_tier1_first(self, exit_processor):
        trade = FakeTrade(entry_price=100.0, qty=100, side="buy")
        # Jump to 2.6% without having triggered tier 1
        actions = exit_processor.check_profit_tiers(trade, 102.60)
        # This triggers tier 1 (not tier 2) because tiers_taken starts at 0
        assert len(actions) == 1
        assert actions[0]["reason"] == "profit_tier_1_1.5pct"

    def test_tier2_fires_after_tier1(self, exit_processor):
        trade = FakeTrade(entry_price=100.0, qty=100, side="buy")
        # Trigger tier 1
        exit_processor.check_profit_tiers(trade, 101.60)
        assert exit_processor._profit_tier_exits["AAPL"] == 1

        # Now trigger tier 2
        actions = exit_processor.check_profit_tiers(trade, 102.60)
        assert len(actions) == 1
        assert actions[0]["reason"] == "profit_tier_2_2.5pct"
        assert exit_processor._profit_tier_exits["AAPL"] == 2

    def test_no_tiers_after_both_taken(self, exit_processor):
        trade = FakeTrade(entry_price=100.0, qty=100, side="buy")
        exit_processor.check_profit_tiers(trade, 101.60)
        exit_processor.check_profit_tiers(trade, 102.60)
        # Both tiers taken
        actions = exit_processor.check_profit_tiers(trade, 104.00)
        assert actions == []

    def test_state_tracking_persists(self, exit_processor):
        """_profit_tier_exits dict maintains state across calls."""
        trade = FakeTrade(entry_price=100.0, qty=50, side="buy")
        exit_processor.check_profit_tiers(trade, 101.60)
        assert "AAPL" in exit_processor._profit_tier_exits
        assert exit_processor._profit_tier_exits["AAPL"] == 1

    def test_different_symbols_tracked_independently(self, exit_processor):
        trade_a = FakeTrade(symbol="AAPL", entry_price=100.0, qty=100, side="buy")
        trade_g = FakeTrade(symbol="GOOG", entry_price=200.0, qty=50, side="buy")

        exit_processor.check_profit_tiers(trade_a, 101.60)
        exit_processor.check_profit_tiers(trade_g, 200.50)  # Not enough for tier 1

        assert exit_processor._profit_tier_exits.get("AAPL") == 1
        assert exit_processor._profit_tier_exits.get("GOOG", 0) == 0


# ===================================================================
# _check_dead_signal
# ===================================================================

class TestCheckDeadSignal:
    """Dead signal: day strategy held >30 min with <0.3% move."""

    def test_dead_signal_long(self, exit_processor):
        entry_time = datetime(2026, 4, 3, 10, 0, tzinfo=ET)
        trade = FakeTrade(strategy="STAT_MR", entry_price=100.0, entry_time=entry_time)
        now = entry_time + timedelta(minutes=35)

        actions = exit_processor._check_dead_signal(trade, 100.20, now)
        assert len(actions) == 1
        assert actions[0]["reason"] == "dead_signal"
        assert actions[0]["action"] == "full"

    def test_dead_signal_short(self, exit_processor):
        entry_time = datetime(2026, 4, 3, 10, 0, tzinfo=ET)
        trade = FakeTrade(strategy="VWAP", entry_price=100.0, side="sell",
                          entry_time=entry_time)
        now = entry_time + timedelta(minutes=35)
        # Price barely moved up 0.2% — for a short, unrealized is -0.2%
        actions = exit_processor._check_dead_signal(trade, 100.20, now)
        assert len(actions) == 1

    def test_no_dead_signal_before_30_min(self, exit_processor):
        entry_time = datetime(2026, 4, 3, 10, 0, tzinfo=ET)
        trade = FakeTrade(strategy="STAT_MR", entry_price=100.0, entry_time=entry_time)
        now = entry_time + timedelta(minutes=25)

        actions = exit_processor._check_dead_signal(trade, 100.10, now)
        assert actions == []

    def test_no_dead_signal_with_significant_move(self, exit_processor):
        entry_time = datetime(2026, 4, 3, 10, 0, tzinfo=ET)
        trade = FakeTrade(strategy="STAT_MR", entry_price=100.0, entry_time=entry_time)
        now = entry_time + timedelta(minutes=45)

        # 0.5% move — outside the -0.3% to +0.3% dead zone
        actions = exit_processor._check_dead_signal(trade, 100.50, now)
        assert actions == []

    def test_no_dead_signal_for_non_day_strategy(self, exit_processor):
        entry_time = datetime(2026, 4, 3, 10, 0, tzinfo=ET)
        trade = FakeTrade(strategy="KALMAN_PAIRS", entry_price=100.0,
                          entry_time=entry_time)
        now = entry_time + timedelta(minutes=60)

        actions = exit_processor._check_dead_signal(trade, 100.10, now)
        assert actions == []

    def test_dead_signal_exact_boundary_30_min(self, exit_processor):
        """Exactly 30 min triggers (hold_duration < 30 is False, so check proceeds)."""
        entry_time = datetime(2026, 4, 3, 10, 0, tzinfo=ET)
        trade = FakeTrade(strategy="ORB", entry_price=100.0, entry_time=entry_time)
        now = entry_time + timedelta(minutes=30)

        actions = exit_processor._check_dead_signal(trade, 100.10, now)
        assert len(actions) == 1

    def test_dead_signal_does_not_fire_at_29_min(self, exit_processor):
        """29 min should NOT trigger (hold_duration < 30 is True)."""
        entry_time = datetime(2026, 4, 3, 10, 0, tzinfo=ET)
        trade = FakeTrade(strategy="ORB", entry_price=100.0, entry_time=entry_time)
        now = entry_time + timedelta(minutes=29)

        actions = exit_processor._check_dead_signal(trade, 100.10, now)
        assert actions == []

    def test_dead_signal_boundary_0_3pct(self, exit_processor):
        """Exactly 0.3% should trigger (within -0.3% to +0.3% inclusive)."""
        entry_time = datetime(2026, 4, 3, 10, 0, tzinfo=ET)
        trade = FakeTrade(strategy="MICRO_MOM", entry_price=100.0, entry_time=entry_time)
        now = entry_time + timedelta(minutes=31)

        # +0.3% exactly
        actions = exit_processor._check_dead_signal(trade, 100.30, now)
        assert len(actions) == 1

    def test_negative_dead_signal_boundary(self, exit_processor):
        """At -0.3% should also trigger."""
        entry_time = datetime(2026, 4, 3, 10, 0, tzinfo=ET)
        trade = FakeTrade(strategy="STAT_MR", entry_price=100.0, entry_time=entry_time)
        now = entry_time + timedelta(minutes=31)

        actions = exit_processor._check_dead_signal(trade, 99.70, now)
        assert len(actions) == 1


# ===================================================================
# _check_scale_out_loser
# ===================================================================

class TestCheckScaleOutLoser:
    """Scale out 50% at -1% unrealized loss, once per trade."""

    def test_scale_out_triggers(self, exit_processor):
        trade = FakeTrade(entry_price=100.0, qty=100, side="buy")
        actions = exit_processor._check_scale_out_loser(trade, 98.90)
        assert len(actions) == 1
        assert actions[0]["qty"] == 50
        assert actions[0]["reason"] == "scale_out_loser_1pct"
        assert actions[0]["action"] == "partial"

    def test_scale_out_only_once(self, exit_processor):
        trade = FakeTrade(entry_price=100.0, qty=100, side="buy")
        actions1 = exit_processor._check_scale_out_loser(trade, 98.90)
        assert len(actions1) == 1

        actions2 = exit_processor._check_scale_out_loser(trade, 98.50)
        assert actions2 == []

    def test_no_scale_out_above_threshold(self, exit_processor):
        trade = FakeTrade(entry_price=100.0, qty=100, side="buy")
        actions = exit_processor._check_scale_out_loser(trade, 99.50)
        assert actions == []

    def test_scale_out_short_side(self, exit_processor):
        trade = FakeTrade(entry_price=100.0, qty=80, side="sell")
        # Price up 1.1% — loss for short
        actions = exit_processor._check_scale_out_loser(trade, 101.10)
        assert len(actions) == 1
        assert actions[0]["qty"] == 40

    def test_scale_out_minimum_1_share(self, exit_processor):
        trade = FakeTrade(entry_price=100.0, qty=1, side="buy")
        actions = exit_processor._check_scale_out_loser(trade, 98.50)
        assert len(actions) == 1
        assert actions[0]["qty"] == 1  # max(1, 1//2) = 1

    def test_scale_out_state_per_symbol(self, exit_processor):
        """_scaled_out is per-symbol."""
        trade_a = FakeTrade(symbol="AAPL", entry_price=100.0, qty=100, side="buy")
        trade_g = FakeTrade(symbol="GOOG", entry_price=200.0, qty=50, side="buy")

        exit_processor._check_scale_out_loser(trade_a, 98.50)
        assert "AAPL" in exit_processor._scaled_out
        assert "GOOG" not in exit_processor._scaled_out

        actions = exit_processor._check_scale_out_loser(trade_g, 197.50)
        assert len(actions) == 1
        assert "GOOG" in exit_processor._scaled_out


# ===================================================================
# check_advanced_exits (integration)
# ===================================================================

class TestCheckAdvancedExits:
    """check_advanced_exits aggregates all advanced exit checks."""

    def test_returns_empty_on_no_trades(self, exit_processor):
        rm = FakeRiskManager()
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        with patch.object(exit_processor, "get_current_prices", return_value={}):
            actions = exit_processor.check_advanced_exits(rm, now)
        assert actions == []

    def test_combines_multiple_exit_types(self, exit_processor):
        """One position triggers profit tier, another triggers dead signal."""
        entry_time = datetime(2026, 4, 3, 10, 0, tzinfo=ET)
        now = entry_time + timedelta(minutes=35)

        trade_profit = FakeTrade(symbol="GOOG", entry_price=200.0, qty=100, side="buy")
        trade_dead = FakeTrade(
            symbol="MSFT", strategy="STAT_MR", entry_price=300.0,
            qty=50, side="buy", entry_time=entry_time,
        )

        rm = FakeRiskManager()
        rm.open_trades["GOOG"] = trade_profit
        rm.open_trades["MSFT"] = trade_dead

        prices = {"GOOG": 203.10, "MSFT": 300.50}

        with patch.object(exit_processor, "get_current_prices", return_value=prices):
            actions = exit_processor.check_advanced_exits(rm, now)

        symbols = {a["symbol"] for a in actions}
        # GOOG should get profit tier, MSFT should get dead signal
        assert "GOOG" in symbols
        assert "MSFT" in symbols
