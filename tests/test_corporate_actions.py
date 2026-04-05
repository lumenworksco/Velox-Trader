"""Comprehensive tests for risk/corporate_actions.py.

Covers:
- Split detection and application
- Reverse split handling
- Dividend detection and price target adjustment
- Position adjustment calculations
- Idempotency (processed actions not re-applied)
- Error handling (fail-open)
- Edge cases (zero ratio, negative dividend, missing symbols)
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from risk.corporate_actions import (
    ActionType,
    CorporateAction,
    CorporateActionDetector,
    PositionAdjustment,
)

ET = ZoneInfo("America/New_York")


# ===================================================================
# Helpers — mock TradeRecord
# ===================================================================

@dataclass
class MockTradeRecord:
    """Minimal trade record for testing corporate action adjustments."""
    symbol: str
    qty: int
    entry_price: float
    stop_loss: float
    take_profit: float
    side: str = "buy"
    highest_price_seen: float = 0.0
    lowest_price_seen: float = 0.0


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def detector():
    return CorporateActionDetector()


@pytest.fixture
def aapl_trade():
    return MockTradeRecord(
        symbol="AAPL",
        qty=100,
        entry_price=200.0,
        stop_loss=190.0,
        take_profit=220.0,
        highest_price_seen=210.0,
        lowest_price_seen=195.0,
    )


@pytest.fixture
def tsla_trade_short():
    return MockTradeRecord(
        symbol="TSLA",
        qty=50,
        entry_price=300.0,
        stop_loss=320.0,
        take_profit=280.0,
        side="sell",
        highest_price_seen=310.0,
        lowest_price_seen=290.0,
    )


# ===================================================================
# Split detection and application
# ===================================================================

class TestStockSplit:

    def test_4_to_1_split_adjusts_correctly(self, detector, aapl_trade):
        """4:1 split: qty *= 4, all prices /= 4."""
        action = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.STOCK_SPLIT,
            effective_date=date(2026, 4, 1),
            split_ratio=4.0,
        )
        open_trades = {"AAPL": aapl_trade}
        adjustments = detector.apply_adjustments([action], open_trades)

        assert len(adjustments) == 1
        adj = adjustments[0]
        assert adj.new_qty == 400
        assert adj.old_qty == 100
        assert adj.new_entry_price == pytest.approx(50.0)
        assert adj.new_stop_loss == pytest.approx(47.5)
        assert adj.new_take_profit == pytest.approx(55.0)

        # Verify trade object was mutated
        assert aapl_trade.qty == 400
        assert aapl_trade.entry_price == pytest.approx(50.0)
        assert aapl_trade.stop_loss == pytest.approx(47.5)
        assert aapl_trade.take_profit == pytest.approx(55.0)
        assert aapl_trade.highest_price_seen == pytest.approx(52.5)
        assert aapl_trade.lowest_price_seen == pytest.approx(48.75)

    def test_2_to_1_split(self, detector, aapl_trade):
        """2:1 split: qty *= 2, prices /= 2."""
        action = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.STOCK_SPLIT,
            effective_date=date(2026, 4, 1),
            split_ratio=2.0,
        )
        adjustments = detector.apply_adjustments([action], {"AAPL": aapl_trade})

        assert adjustments[0].new_qty == 200
        assert aapl_trade.entry_price == pytest.approx(100.0)

    def test_3_to_1_split(self, detector, aapl_trade):
        """3:1 split."""
        action = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.STOCK_SPLIT,
            effective_date=date(2026, 4, 1),
            split_ratio=3.0,
        )
        adjustments = detector.apply_adjustments([action], {"AAPL": aapl_trade})
        assert adjustments[0].new_qty == 300
        assert aapl_trade.entry_price == pytest.approx(200.0 / 3.0)

    def test_split_reason_string(self, detector, aapl_trade):
        action = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.STOCK_SPLIT,
            effective_date=date(2026, 4, 1),
            split_ratio=4.0,
        )
        adjustments = detector.apply_adjustments([action], {"AAPL": aapl_trade})
        assert "4.0:1 split" in adjustments[0].reason


class TestReverseSplit:

    def test_reverse_1_to_10_split(self, detector, aapl_trade):
        """1:10 reverse split: ratio=0.1, qty *= 0.1, prices /= 0.1."""
        action = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.REVERSE_SPLIT,
            effective_date=date(2026, 4, 1),
            split_ratio=0.1,
        )
        adjustments = detector.apply_adjustments([action], {"AAPL": aapl_trade})

        assert len(adjustments) == 1
        assert adjustments[0].new_qty == 10  # 100 * 0.1
        assert aapl_trade.entry_price == pytest.approx(2000.0)  # 200 / 0.1
        assert aapl_trade.stop_loss == pytest.approx(1900.0)
        assert aapl_trade.take_profit == pytest.approx(2200.0)

    def test_reverse_1_to_2_split(self, detector, aapl_trade):
        """1:2 reverse split."""
        action = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.REVERSE_SPLIT,
            effective_date=date(2026, 4, 1),
            split_ratio=0.5,
        )
        adjustments = detector.apply_adjustments([action], {"AAPL": aapl_trade})
        assert adjustments[0].new_qty == 50  # 100 * 0.5
        assert aapl_trade.entry_price == pytest.approx(400.0)  # 200 / 0.5


class TestSplitEdgeCases:

    def test_ratio_1_is_noop(self, detector, aapl_trade):
        """Split ratio of 1.0 should be a no-op."""
        action = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.STOCK_SPLIT,
            effective_date=date(2026, 4, 1),
            split_ratio=1.0,
        )
        adjustments = detector.apply_adjustments([action], {"AAPL": aapl_trade})
        assert len(adjustments) == 0
        assert aapl_trade.qty == 100  # unchanged

    def test_ratio_zero_is_noop(self, detector, aapl_trade):
        """Split ratio of 0 should be a no-op."""
        action = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.STOCK_SPLIT,
            effective_date=date(2026, 4, 1),
            split_ratio=0.0,
        )
        adjustments = detector.apply_adjustments([action], {"AAPL": aapl_trade})
        assert len(adjustments) == 0

    def test_negative_ratio_is_noop(self, detector, aapl_trade):
        """Negative split ratio should be a no-op."""
        action = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.STOCK_SPLIT,
            effective_date=date(2026, 4, 1),
            split_ratio=-2.0,
        )
        adjustments = detector.apply_adjustments([action], {"AAPL": aapl_trade})
        assert len(adjustments) == 0

    def test_split_on_missing_symbol(self, detector, aapl_trade):
        """Split for symbol not in open_trades should be skipped."""
        action = CorporateAction(
            symbol="MSFT",
            action_type=ActionType.STOCK_SPLIT,
            effective_date=date(2026, 4, 1),
            split_ratio=4.0,
        )
        adjustments = detector.apply_adjustments([action], {"AAPL": aapl_trade})
        assert len(adjustments) == 0


# ===================================================================
# Dividend detection and application
# ===================================================================

class TestDividend:

    def test_dividend_adjusts_price_targets_down(self, detector, aapl_trade):
        """On ex-date, TP and SL should be reduced by dividend amount."""
        action = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.DIVIDEND,
            effective_date=date(2026, 4, 1),
            dividend_amount=0.50,
            ex_date=date(2026, 4, 1),
        )
        adjustments = detector.apply_adjustments([action], {"AAPL": aapl_trade})

        assert len(adjustments) == 1
        adj = adjustments[0]
        assert adj.new_take_profit == pytest.approx(219.50)
        assert adj.new_stop_loss == pytest.approx(189.50)
        assert aapl_trade.take_profit == pytest.approx(219.50)
        assert aapl_trade.stop_loss == pytest.approx(189.50)

    def test_dividend_on_short_position(self, detector, tsla_trade_short):
        """Dividend adjustments apply the same way for shorts (prices move down)."""
        action = CorporateAction(
            symbol="TSLA",
            action_type=ActionType.DIVIDEND,
            effective_date=date(2026, 4, 1),
            dividend_amount=1.00,
        )
        adjustments = detector.apply_adjustments([action], {"TSLA": tsla_trade_short})

        assert len(adjustments) == 1
        assert tsla_trade_short.take_profit == pytest.approx(279.0)
        assert tsla_trade_short.stop_loss == pytest.approx(319.0)

    def test_large_dividend(self, detector, aapl_trade):
        """Large special dividend should adjust targets significantly."""
        action = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.DIVIDEND,
            effective_date=date(2026, 4, 1),
            dividend_amount=5.00,
        )
        adjustments = detector.apply_adjustments([action], {"AAPL": aapl_trade})
        assert aapl_trade.take_profit == pytest.approx(215.0)
        assert aapl_trade.stop_loss == pytest.approx(185.0)

    def test_zero_dividend_is_noop(self, detector, aapl_trade):
        """Zero dividend amount should be a no-op."""
        action = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.DIVIDEND,
            effective_date=date(2026, 4, 1),
            dividend_amount=0.0,
        )
        adjustments = detector.apply_adjustments([action], {"AAPL": aapl_trade})
        assert len(adjustments) == 0

    def test_negative_dividend_is_noop(self, detector, aapl_trade):
        """Negative dividend amount should be a no-op."""
        action = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.DIVIDEND,
            effective_date=date(2026, 4, 1),
            dividend_amount=-1.0,
        )
        adjustments = detector.apply_adjustments([action], {"AAPL": aapl_trade})
        assert len(adjustments) == 0

    def test_dividend_reason_string(self, detector, aapl_trade):
        action = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.DIVIDEND,
            effective_date=date(2026, 4, 1),
            dividend_amount=0.50,
        )
        adjustments = detector.apply_adjustments([action], {"AAPL": aapl_trade})
        assert "Ex-div" in adjustments[0].reason
        assert "$0.5000" in adjustments[0].reason


# ===================================================================
# Idempotency — processed actions
# ===================================================================

class TestIdempotency:

    def test_processed_action_filtered_by_check(self, detector, aapl_trade):
        """Once an action is processed via apply_adjustments, check_actions
        should filter it out so it is not returned again."""
        action = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.STOCK_SPLIT,
            effective_date=date(2026, 4, 1),
            split_ratio=4.0,
        )
        # First application
        adj1 = detector.apply_adjustments([action], {"AAPL": aapl_trade})
        assert len(adj1) == 1
        assert aapl_trade.qty == 400

        # Processed key should now be in the set
        key = "AAPL_stock_split_2026-04-01"
        assert key in detector._processed_actions

    @patch.object(CorporateActionDetector, "_check_alpaca")
    def test_check_actions_filters_processed(self, mock_alpaca, detector, aapl_trade):
        """check_actions should not return actions already in _processed_actions."""
        action = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.STOCK_SPLIT,
            effective_date=date(2026, 4, 1),
            split_ratio=4.0,
        )
        # Apply once to mark it as processed
        detector.apply_adjustments([action], {"AAPL": aapl_trade})

        # Now check_actions should filter it out
        mock_alpaca.return_value = [action]
        actions = detector.check_actions(["AAPL"])
        assert len(actions) == 0


# ===================================================================
# check_actions() — action detection
# ===================================================================

class TestCheckActions:

    def test_empty_symbols_returns_empty(self, detector):
        actions = detector.check_actions([])
        assert actions == []

    @patch.object(CorporateActionDetector, "_check_alpaca")
    @patch.object(CorporateActionDetector, "_check_yfinance")
    def test_alpaca_returns_actions(self, mock_yf, mock_alpaca, detector):
        """If Alpaca returns actions, yfinance should not be called."""
        split = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.STOCK_SPLIT,
            effective_date=date(2026, 4, 1),
            split_ratio=4.0,
        )
        mock_alpaca.return_value = [split]

        actions = detector.check_actions(["AAPL"])
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.STOCK_SPLIT
        mock_yf.assert_not_called()

    @patch.object(CorporateActionDetector, "_check_alpaca")
    @patch.object(CorporateActionDetector, "_check_yfinance")
    def test_falls_back_to_yfinance(self, mock_yf, mock_alpaca, detector):
        """If Alpaca returns nothing, yfinance should be tried."""
        mock_alpaca.return_value = []
        div = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.DIVIDEND,
            effective_date=date(2026, 4, 1),
            dividend_amount=0.50,
        )
        mock_yf.return_value = [div]

        actions = detector.check_actions(["AAPL"])
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.DIVIDEND

    @patch.object(CorporateActionDetector, "_check_alpaca")
    @patch.object(CorporateActionDetector, "_check_yfinance")
    def test_both_sources_fail_returns_empty(self, mock_yf, mock_alpaca, detector):
        """If both sources fail (exception), should return empty (fail-open)."""
        mock_alpaca.side_effect = Exception("Alpaca down")
        mock_yf.side_effect = Exception("yfinance down")

        actions = detector.check_actions(["AAPL"])
        assert actions == []

    @patch.object(CorporateActionDetector, "_check_alpaca")
    def test_already_processed_action_filtered(self, mock_alpaca, detector):
        """Actions that were already processed should be filtered out."""
        split = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.STOCK_SPLIT,
            effective_date=date(2026, 4, 1),
            split_ratio=4.0,
        )

        # Mark as processed
        detector._processed_actions.add("AAPL_stock_split_2026-04-01")

        mock_alpaca.return_value = [split]
        actions = detector.check_actions(["AAPL"])
        assert len(actions) == 0

    @patch.object(CorporateActionDetector, "_check_alpaca")
    def test_multiple_symbols_checked(self, mock_alpaca, detector):
        mock_alpaca.return_value = []
        detector.check_actions(["AAPL", "MSFT", "TSLA"])
        assert mock_alpaca.call_count == 3


# ===================================================================
# Multiple actions
# ===================================================================

class TestMultipleActions:

    def test_split_and_dividend_same_day(self, detector, aapl_trade):
        """Both a split and dividend applied to same position."""
        actions = [
            CorporateAction(
                symbol="AAPL",
                action_type=ActionType.STOCK_SPLIT,
                effective_date=date(2026, 4, 1),
                split_ratio=2.0,
            ),
            CorporateAction(
                symbol="AAPL",
                action_type=ActionType.DIVIDEND,
                effective_date=date(2026, 4, 1),
                dividend_amount=0.50,
            ),
        ]
        adjustments = detector.apply_adjustments(actions, {"AAPL": aapl_trade})
        assert len(adjustments) == 2
        # After split: qty=200, entry=100, tp=110, sl=95
        # After dividend: tp=109.50, sl=94.50
        assert aapl_trade.qty == 200
        assert aapl_trade.take_profit == pytest.approx(109.50)
        assert aapl_trade.stop_loss == pytest.approx(94.50)

    def test_actions_for_different_symbols(self, detector, aapl_trade, tsla_trade_short):
        actions = [
            CorporateAction(
                symbol="AAPL",
                action_type=ActionType.STOCK_SPLIT,
                effective_date=date(2026, 4, 1),
                split_ratio=2.0,
            ),
            CorporateAction(
                symbol="TSLA",
                action_type=ActionType.DIVIDEND,
                effective_date=date(2026, 4, 1),
                dividend_amount=1.00,
            ),
        ]
        open_trades = {"AAPL": aapl_trade, "TSLA": tsla_trade_short}
        adjustments = detector.apply_adjustments(actions, open_trades)
        assert len(adjustments) == 2
        assert aapl_trade.qty == 200
        assert tsla_trade_short.stop_loss == pytest.approx(319.0)


# ===================================================================
# Error handling
# ===================================================================

class TestErrorHandling:

    def test_apply_adjustments_handles_exception(self, detector):
        """If adjustment throws internally, it should be caught (fail-open)."""
        # Create a trade-like object that will cause an error
        bad_trade = MagicMock()
        bad_trade.qty = 100
        bad_trade.entry_price = 200.0
        bad_trade.stop_loss = 190.0
        bad_trade.take_profit = 220.0
        # Make setting qty raise
        type(bad_trade).qty = property(lambda self: 100, lambda self, v: (_ for _ in ()).throw(RuntimeError("fail")))

        action = CorporateAction(
            symbol="BAD",
            action_type=ActionType.STOCK_SPLIT,
            effective_date=date(2026, 4, 1),
            split_ratio=2.0,
        )
        # Should not raise
        adjustments = detector.apply_adjustments([action], {"BAD": bad_trade})
        assert len(adjustments) == 0  # failed, so no adjustment recorded


# ===================================================================
# Status
# ===================================================================

class TestStatus:

    def test_status_dict_keys(self, detector):
        status = detector.status
        assert "processed_actions_count" in status
        assert "last_check" in status
        assert "check_interval_sec" in status

    def test_status_after_processing(self, detector, aapl_trade):
        action = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.STOCK_SPLIT,
            effective_date=date(2026, 4, 1),
            split_ratio=2.0,
        )
        detector.apply_adjustments([action], {"AAPL": aapl_trade})
        status = detector.status
        assert status["processed_actions_count"] == 1


# ===================================================================
# ActionType enum
# ===================================================================

class TestActionType:

    def test_enum_values(self):
        assert ActionType.STOCK_SPLIT.value == "stock_split"
        assert ActionType.REVERSE_SPLIT.value == "reverse_split"
        assert ActionType.DIVIDEND.value == "dividend"


# ===================================================================
# CorporateAction dataclass
# ===================================================================

class TestCorporateActionDataclass:

    def test_default_values(self):
        action = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.STOCK_SPLIT,
            effective_date=date(2026, 4, 1),
        )
        assert action.split_ratio == 1.0
        assert action.dividend_amount == 0.0
        assert action.ex_date is None

    def test_full_construction(self):
        action = CorporateAction(
            symbol="AAPL",
            action_type=ActionType.DIVIDEND,
            effective_date=date(2026, 4, 1),
            dividend_amount=0.82,
            ex_date=date(2026, 3, 28),
        )
        assert action.dividend_amount == 0.82
        assert action.ex_date == date(2026, 3, 28)
