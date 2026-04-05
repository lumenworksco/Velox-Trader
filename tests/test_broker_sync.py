"""Comprehensive tests for engine/broker_sync.py — broker position synchronization.

Tests cover:
- Position sync with mock broker
- Discrepancy detection (broker has position we lack and vice versa)
- Shadow exit checking (TP/SL hits on shadow trades)
- 2-consecutive-miss tolerance before closing
- Re-adoption of broker positions not in tracking
"""

import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call
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

    def add_trade(self, trade):
        self.open_trades[trade.symbol] = trade


class FakeBrokerPosition:
    """Mock broker position matching Alpaca Position fields."""
    def __init__(self, symbol, qty, avg_entry_price):
        self.symbol = symbol
        self.qty = str(qty)
        self.avg_entry_price = str(avg_entry_price)
        self.market_value = str(float(qty) * float(avg_entry_price))
        self.unrealized_pl = "0.0"


class FakeSnapshot:
    def __init__(self, price):
        self.latest_trade = MagicMock(price=price)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_deps():
    """Mock all external deps so broker_sync can import cleanly."""
    import types
    cfg = types.ModuleType("config")
    cfg.ET = ET
    cfg.TELEGRAM_ENABLED = False
    cfg.BROKER_SYNC_EXCLUDE_SYMBOLS = {"SPY"}

    mock_database = MagicMock()
    mock_database.get_recent_trades = MagicMock(return_value=[])
    mock_database.get_open_shadow_trades = MagicMock(return_value=[])
    mock_database.close_shadow_trade = MagicMock()

    mock_data = MagicMock()
    mock_data.get_positions = MagicMock(return_value=[])
    mock_data.get_account = MagicMock()
    mock_data.get_snapshot = MagicMock(return_value=None)
    mock_data.get_filled_exit_info = MagicMock(return_value=(None, None))

    mock_exec_core = MagicMock()
    mock_exec_core.get_order_commission = MagicMock(return_value=0.0)

    mock_risk_mod = MagicMock()
    mock_risk_mod.RiskManager = FakeRiskManager
    mock_risk_mod.TradeRecord = FakeTrade

    with patch.dict(sys.modules, {
        "config": cfg,
        "database": mock_database,
        "data": mock_data,
        "execution.core": mock_exec_core,
        "risk": mock_risk_mod,
        "engine.events": MagicMock(),
        "engine.signal_processor": MagicMock(),
    }):
        yield cfg, mock_database, mock_data


@pytest.fixture
def broker_sync(_mock_deps):
    """Import and return a fresh broker_sync module."""
    import importlib
    mod = importlib.import_module("engine.broker_sync")
    importlib.reload(mod)
    # Reset module-level state
    mod._broker_miss_counts.clear()
    mod._notifications = None
    return mod


# ===================================================================
# Position sync: basic matching
# ===================================================================

class TestPositionSyncBasic:
    """Sync detects when broker and local state agree or diverge."""

    def test_matching_positions_no_action(self, broker_sync, _mock_deps):
        _, mock_db, mock_data = _mock_deps
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = FakeTrade(symbol="AAPL")

        # Broker also has AAPL
        mock_data.get_positions.return_value = [
            FakeBrokerPosition("AAPL", 100, 100.0)
        ]
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        broker_sync.sync_positions_with_broker(rm, now)
        assert "AAPL" in rm.open_trades  # Still tracked

    def test_api_failure_does_not_crash(self, broker_sync, _mock_deps):
        _, mock_db, mock_data = _mock_deps
        mock_data.get_positions.side_effect = Exception("API timeout")
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = FakeTrade(symbol="AAPL")
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        # Should not raise
        broker_sync.sync_positions_with_broker(rm, now)
        assert "AAPL" in rm.open_trades


# ===================================================================
# Discrepancy detection: 2-miss tolerance
# ===================================================================

class TestDiscrepancyDetection:
    """Broker missing a position requires 2 consecutive misses to close."""

    def test_first_miss_does_not_close(self, broker_sync, _mock_deps):
        _, mock_db, mock_data = _mock_deps
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = FakeTrade(symbol="AAPL")

        # Broker has NO positions
        mock_data.get_positions.return_value = []
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        broker_sync.sync_positions_with_broker(rm, now)
        # First miss — should NOT close yet
        assert "AAPL" in rm.open_trades

    def test_second_miss_closes_position(self, broker_sync, _mock_deps):
        _, mock_db, mock_data = _mock_deps
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = FakeTrade(symbol="AAPL")

        mock_data.get_positions.return_value = []
        mock_data.get_filled_exit_info.return_value = (101.0, "take_profit")
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        # First sync — miss 1
        broker_sync.sync_positions_with_broker(rm, now)
        assert "AAPL" in rm.open_trades

        # Re-add trade (close_trade removes it, but we want to test the flow)
        rm.open_trades["AAPL"] = FakeTrade(symbol="AAPL")

        # Second sync — miss 2 → close
        broker_sync.sync_positions_with_broker(rm, now)
        assert "AAPL" not in rm.open_trades

    def test_miss_count_resets_when_position_reappears(self, broker_sync, _mock_deps):
        _, mock_db, mock_data = _mock_deps
        rm = FakeRiskManager()
        rm.open_trades["AAPL"] = FakeTrade(symbol="AAPL")

        # Miss 1: broker has nothing
        mock_data.get_positions.return_value = []
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)
        broker_sync.sync_positions_with_broker(rm, now)
        assert "AAPL" in rm.open_trades

        # Next sync: broker has AAPL again — miss count should reset
        mock_data.get_positions.return_value = [
            FakeBrokerPosition("AAPL", 100, 100.0)
        ]
        broker_sync.sync_positions_with_broker(rm, now)

        # The miss count for AAPL should be cleared
        assert broker_sync._broker_miss_counts.get("AAPL", 0) == 0


# ===================================================================
# Re-adoption of unknown broker positions
# ===================================================================

class TestReAdoption:
    """Broker has positions not in our tracking — re-adopt them."""

    def test_readopt_broker_position(self, broker_sync, _mock_deps):
        _, mock_db, mock_data = _mock_deps
        rm = FakeRiskManager()
        # We have no positions, but broker has GOOG
        mock_data.get_positions.return_value = [
            FakeBrokerPosition("GOOG", 50, 175.0)
        ]
        mock_db.get_recent_trades.return_value = []
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        broker_sync.sync_positions_with_broker(rm, now)
        # GOOG should be re-adopted into risk manager
        assert "GOOG" in rm.open_trades

    def test_excluded_symbols_not_readopted(self, broker_sync, _mock_deps):
        cfg, mock_db, mock_data = _mock_deps
        cfg.BROKER_SYNC_EXCLUDE_SYMBOLS = {"SPY"}
        rm = FakeRiskManager()

        mock_data.get_positions.return_value = [
            FakeBrokerPosition("SPY", 200, 500.0)
        ]
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        broker_sync.sync_positions_with_broker(rm, now)
        assert "SPY" not in rm.open_trades


# ===================================================================
# Shadow exit checking
# ===================================================================

class TestShadowExits:
    """check_shadow_exits closes shadow trades when TP/SL is hit."""

    def test_shadow_buy_tp_hit(self, broker_sync, _mock_deps):
        _, mock_db, mock_data = _mock_deps
        mock_db.get_open_shadow_trades.return_value = [
            {"id": 1, "symbol": "AAPL", "strategy": "STAT_MR",
             "side": "buy", "take_profit": 105.0, "stop_loss": 95.0},
        ]
        mock_data.get_snapshot.return_value = FakeSnapshot(106.0)
        now = datetime(2026, 4, 3, 12, 0, tzinfo=ET)

        broker_sync.check_shadow_exits(now)
        mock_db.close_shadow_trade.assert_called_once_with(
            1, 106.0, now.isoformat(), "take_profit"
        )

    def test_shadow_buy_sl_hit(self, broker_sync, _mock_deps):
        _, mock_db, mock_data = _mock_deps
        mock_db.get_open_shadow_trades.return_value = [
            {"id": 2, "symbol": "GOOG", "strategy": "ORB",
             "side": "buy", "take_profit": 180.0, "stop_loss": 160.0},
        ]
        mock_data.get_snapshot.return_value = FakeSnapshot(158.0)
        now = datetime(2026, 4, 3, 12, 0, tzinfo=ET)

        broker_sync.check_shadow_exits(now)
        mock_db.close_shadow_trade.assert_called_once_with(
            2, 158.0, now.isoformat(), "stop_loss"
        )

    def test_shadow_sell_tp_hit(self, broker_sync, _mock_deps):
        _, mock_db, mock_data = _mock_deps
        mock_db.get_open_shadow_trades.return_value = [
            {"id": 3, "symbol": "TSLA", "strategy": "VWAP",
             "side": "sell", "take_profit": 90.0, "stop_loss": 110.0},
        ]
        mock_data.get_snapshot.return_value = FakeSnapshot(88.0)
        now = datetime(2026, 4, 3, 12, 0, tzinfo=ET)

        broker_sync.check_shadow_exits(now)
        mock_db.close_shadow_trade.assert_called_once_with(
            3, 88.0, now.isoformat(), "take_profit"
        )

    def test_shadow_sell_sl_hit(self, broker_sync, _mock_deps):
        _, mock_db, mock_data = _mock_deps
        mock_db.get_open_shadow_trades.return_value = [
            {"id": 4, "symbol": "META", "strategy": "STAT_MR",
             "side": "sell", "take_profit": 280.0, "stop_loss": 320.0},
        ]
        mock_data.get_snapshot.return_value = FakeSnapshot(325.0)
        now = datetime(2026, 4, 3, 12, 0, tzinfo=ET)

        broker_sync.check_shadow_exits(now)
        mock_db.close_shadow_trade.assert_called_once_with(
            4, 325.0, now.isoformat(), "stop_loss"
        )

    def test_shadow_no_exit_within_range(self, broker_sync, _mock_deps):
        _, mock_db, mock_data = _mock_deps
        mock_db.get_open_shadow_trades.return_value = [
            {"id": 5, "symbol": "AAPL", "strategy": "ORB",
             "side": "buy", "take_profit": 110.0, "stop_loss": 90.0},
        ]
        mock_data.get_snapshot.return_value = FakeSnapshot(100.0)
        now = datetime(2026, 4, 3, 12, 0, tzinfo=ET)

        broker_sync.check_shadow_exits(now)
        mock_db.close_shadow_trade.assert_not_called()

    def test_shadow_no_snapshot_skipped(self, broker_sync, _mock_deps):
        _, mock_db, mock_data = _mock_deps
        mock_db.get_open_shadow_trades.return_value = [
            {"id": 6, "symbol": "NVDA", "strategy": "VWAP",
             "side": "buy", "take_profit": 500.0, "stop_loss": 400.0},
        ]
        # No snapshot available
        mock_data.get_snapshot.return_value = None
        now = datetime(2026, 4, 3, 12, 0, tzinfo=ET)

        broker_sync.check_shadow_exits(now)
        mock_db.close_shadow_trade.assert_not_called()

    def test_shadow_multiple_trades(self, broker_sync, _mock_deps):
        """Multiple shadow trades: one hits TP, one in range, one hits SL."""
        _, mock_db, mock_data = _mock_deps
        mock_db.get_open_shadow_trades.return_value = [
            {"id": 10, "symbol": "AAPL", "strategy": "ORB",
             "side": "buy", "take_profit": 105.0, "stop_loss": 95.0},
            {"id": 11, "symbol": "GOOG", "strategy": "STAT_MR",
             "side": "buy", "take_profit": 200.0, "stop_loss": 180.0},
            {"id": 12, "symbol": "MSFT", "strategy": "VWAP",
             "side": "buy", "take_profit": 350.0, "stop_loss": 300.0},
        ]

        def snapshot_for(symbol):
            prices = {"AAPL": 106.0, "GOOG": 190.0, "MSFT": 298.0}
            return FakeSnapshot(prices[symbol])

        mock_data.get_snapshot.side_effect = snapshot_for
        now = datetime(2026, 4, 3, 12, 0, tzinfo=ET)

        broker_sync.check_shadow_exits(now)

        calls = mock_db.close_shadow_trade.call_args_list
        closed_ids = {c[0][0] for c in calls}
        assert 10 in closed_ids  # AAPL TP hit
        assert 12 in closed_ids  # MSFT SL hit
        assert 11 not in closed_ids  # GOOG in range

    def test_shadow_db_failure_does_not_crash(self, broker_sync, _mock_deps):
        _, mock_db, mock_data = _mock_deps
        mock_db.get_open_shadow_trades.side_effect = Exception("DB error")
        now = datetime(2026, 4, 3, 12, 0, tzinfo=ET)

        # Should not raise
        broker_sync.check_shadow_exits(now)
