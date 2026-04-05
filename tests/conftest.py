"""Shared fixtures for the trading bot test suite.

All Alpaca API access is mocked — no real API calls are ever made.
The database module is patched to use an in-memory SQLite connection.
"""

import sqlite3
import sys
import types
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

# ---------------------------------------------------------------------------
# Ensure the trading_bot package is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

ET = ZoneInfo("America/New_York")


# ===================================================================
# Mock Alpaca Trading Client
# ===================================================================

class MockOrder:
    """Minimal order object returned by submit_order."""
    def __init__(self, order_id: str = "mock-order-001"):
        self.id = order_id
        self.status = "new"
        self.symbol = ""


class MockAccount:
    """Minimal Alpaca account object."""
    def __init__(self, equity: float = 100_000.0, cash: float = 80_000.0,
                 buying_power: float = 200_000.0):
        self.equity = str(equity)
        self.cash = str(cash)
        self.buying_power = str(buying_power)
        self.portfolio_value = str(equity)


class MockPosition:
    """Minimal Alpaca position object."""
    def __init__(self, symbol: str, qty: str, market_value: str,
                 unrealized_pl: str = "0.0"):
        self.symbol = symbol
        self.qty = qty
        self.market_value = market_value
        self.unrealized_pl = unrealized_pl


class MockAsset:
    """Minimal Alpaca asset object."""
    def __init__(self, symbol: str = "AAPL", tradable: bool = True,
                 shortable: bool = True, easy_to_borrow: bool = True):
        self.symbol = symbol
        self.tradable = tradable
        self.shortable = shortable
        self.easy_to_borrow = easy_to_borrow


class MockTradingClient:
    """Drop-in replacement for alpaca.trading.client.TradingClient.

    Returns canned data for all methods used by the bot.
    """
    def __init__(self, **kwargs):
        self._orders: list[MockOrder] = []
        self._positions: list[MockPosition] = []
        self._account = MockAccount()

    def submit_order(self, request) -> MockOrder:
        order = MockOrder(order_id=f"mock-{len(self._orders)+1:03d}")
        order.symbol = getattr(request, "symbol", "")
        self._orders.append(order)
        return order

    def get_account(self) -> MockAccount:
        return self._account

    def get_all_positions(self) -> list[MockPosition]:
        return self._positions

    def get_asset(self, symbol: str) -> MockAsset:
        return MockAsset(symbol=symbol)

    def close_position(self, symbol: str, **kwargs):
        return True

    def close_all_positions(self, cancel_orders: bool = False):
        self._positions.clear()
        return True

    def get_clock(self):
        return types.SimpleNamespace(is_open=True, next_open="2026-03-22T09:30:00")

    def get_order_by_id(self, order_id: str) -> MockOrder:
        for order in self._orders:
            if str(order.id) == order_id:
                return order
        raise Exception(f"Order {order_id} not found")

    def cancel_orders(self):
        self._orders.clear()
        return True


# ===================================================================
# Mock Data Client / Bars
# ===================================================================

class MockBar:
    """Single bar of OHLCV data."""
    def __init__(self, o=100.0, h=102.0, l=99.0, c=101.0, v=1_000_000,
                 t=None):
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v
        self.timestamp = t or datetime(2026, 3, 13, 10, 0, tzinfo=ET)


class MockSnapshot:
    """Minimal snapshot with latest_trade."""
    def __init__(self, price: float = 150.0):
        self.latest_trade = types.SimpleNamespace(price=price)


@pytest.fixture
def mock_trading_client():
    """Return a MockTradingClient instance."""
    return MockTradingClient()


@pytest.fixture
def mock_snapshot():
    """Return a factory for MockSnapshot objects."""
    def _factory(price: float = 150.0):
        return MockSnapshot(price=price)
    return _factory


# ===================================================================
# In-memory SQLite database fixture
# ===================================================================

@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite DB with the bot's schema.

    Yields a sqlite3.Connection.  The database module's _get_conn is
    patched so every call returns this connection.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def patch_database_connection(in_memory_db):
    """Auto-patch the database module so it uses the in-memory DB.

    This runs for every test automatically, ensuring no on-disk DB
    is touched.
    """
    with patch("database._get_conn", return_value=in_memory_db), \
         patch("database._conn", in_memory_db):
        import database
        database._conn = in_memory_db
        # Initialize schema in the in-memory DB
        database.init_db()
        yield


# ===================================================================
# Config overrides fixture
# ===================================================================

@pytest.fixture
def override_config():
    """Provide a context manager to temporarily override config values.

    Usage:
        def test_something(override_config):
            with override_config(MAX_POSITIONS=2, DAILY_LOSS_HALT=-0.01):
                ...
    """
    import config as cfg

    class _Override:
        def __init__(self, **overrides):
            self._overrides = overrides
            self._originals = {}

        def __enter__(self):
            for key, value in self._overrides.items():
                self._originals[key] = getattr(cfg, key)
                setattr(cfg, key, value)
            return self

        def __exit__(self, *exc):
            for key, value in self._originals.items():
                setattr(cfg, key, value)

    def _factory(**overrides):
        return _Override(**overrides)

    return _factory


# ===================================================================
# Signal fixtures
# ===================================================================

def _make_signal(symbol="AAPL", strategy="ORB", side="buy",
                 entry_price=150.0, take_profit=155.0, stop_loss=148.0,
                 reason="test", hold_type="day", pair_id=""):
    from strategies.base import Signal
    return Signal(
        symbol=symbol,
        strategy=strategy,
        side=side,
        entry_price=entry_price,
        take_profit=take_profit,
        stop_loss=stop_loss,
        reason=reason,
        hold_type=hold_type,
        pair_id=pair_id,
    )


@pytest.fixture
def orb_buy_signal():
    return _make_signal(symbol="AAPL", strategy="ORB", side="buy",
                        entry_price=150.0, take_profit=155.0, stop_loss=148.0,
                        reason="ORB breakout long")


@pytest.fixture
def vwap_sell_signal():
    return _make_signal(symbol="MSFT", strategy="VWAP", side="sell",
                        entry_price=400.0, take_profit=395.0, stop_loss=404.0,
                        reason="VWAP reversion short")


@pytest.fixture
def momentum_buy_signal():
    return _make_signal(symbol="NVDA", strategy="MOMENTUM", side="buy",
                        entry_price=800.0, take_profit=850.0, stop_loss=780.0,
                        reason="Momentum continuation", hold_type="swing")


@pytest.fixture
def sector_rotation_buy_signal():
    return _make_signal(symbol="XLK", strategy="SECTOR_ROTATION", side="buy",
                        entry_price=200.0, take_profit=214.0, stop_loss=193.0,
                        reason="Sector rotation rank #1", hold_type="swing")


@pytest.fixture
def pairs_buy_signal():
    return _make_signal(symbol="AAPL", strategy="PAIRS", side="buy",
                        entry_price=150.0, take_profit=155.0, stop_loss=145.0,
                        reason="Pairs long leg", hold_type="swing",
                        pair_id="AAPL-MSFT-001")


@pytest.fixture
def pairs_sell_signal():
    return _make_signal(symbol="MSFT", strategy="PAIRS", side="sell",
                        entry_price=400.0, take_profit=390.0, stop_loss=410.0,
                        reason="Pairs short leg", hold_type="swing",
                        pair_id="AAPL-MSFT-001")


# ===================================================================
# TradeRecord fixtures
# ===================================================================

def _make_trade(symbol="AAPL", strategy="ORB", side="buy",
                entry_price=150.0, qty=10, take_profit=155.0,
                stop_loss=148.0, status="open", pnl=0.0,
                exit_price=None, exit_reason="", hold_type="day",
                pair_id="", partial_exits=0, highest_price_seen=0.0,
                entry_atr=0.0, entry_time=None):
    from risk import TradeRecord
    return TradeRecord(
        symbol=symbol,
        strategy=strategy,
        side=side,
        entry_price=entry_price,
        entry_time=entry_time or datetime(2026, 3, 13, 10, 5, tzinfo=ET),
        qty=qty,
        take_profit=take_profit,
        stop_loss=stop_loss,
        status=status,
        pnl=pnl,
        exit_price=exit_price,
        exit_reason=exit_reason,
        hold_type=hold_type,
        pair_id=pair_id,
        partial_exits=partial_exits,
        highest_price_seen=highest_price_seen,
        entry_atr=entry_atr,
    )


@pytest.fixture
def open_trade():
    return _make_trade()


@pytest.fixture
def profitable_trade():
    return _make_trade(status="closed", pnl=50.0, exit_price=155.0,
                       exit_reason="take_profit")


@pytest.fixture
def losing_trade():
    return _make_trade(status="closed", pnl=-20.0, exit_price=148.0,
                       exit_reason="stop_loss")


# ===================================================================
# Patch get_trading_client so no real Alpaca connections happen
# ===================================================================

@pytest.fixture(autouse=True)
def patch_trading_client(mock_trading_client):
    """Auto-patch data.get_trading_client to return the mock client."""
    with patch("data.get_trading_client", return_value=mock_trading_client):
        yield mock_trading_client


@pytest.fixture(autouse=True)
def isolate_oms_state(tmp_path):
    """Isolate OMS idempotency keys and OrderManager state between tests.

    Without this, the persistent idempotency_keys.json file and
    module-level OrderManager singleton leak state across tests.
    """
    # Redirect idempotency file to temp directory
    tmp_keys = str(tmp_path / "idempotency_keys.json")
    with patch("oms.order_manager._IDEMPOTENCY_FILE", tmp_keys):
        # Also reset the OrderManager singleton if it exists
        try:
            import oms.order_manager as _om
            if hasattr(_om, "_instance"):
                _om._instance = None
        except Exception:
            pass
        yield
