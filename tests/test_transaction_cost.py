"""Tests for oms/transaction_cost.py — cost estimation, EV checks, spread costs."""

import os
os.environ.setdefault("TESTING", "1")

from unittest.mock import patch, MagicMock

import pytest

from oms.transaction_cost import (
    estimate_round_trip_cost,
    is_trade_profitable_after_costs,
    DEFAULT_SPREAD_BPS,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_COMMISSION_PER_SHARE,
    MIN_EXPECTED_RETURN_BPS,
)


# ===================================================================
# estimate_round_trip_cost
# ===================================================================

class TestEstimateRoundTripCost:
    """Tests for round-trip transaction cost estimation."""

    def test_basic_cost_estimation(self):
        """Standard cost estimation with default parameters."""
        result = estimate_round_trip_cost(entry_price=100.0, qty=100)

        assert "spread_cost" in result
        assert "slippage_cost" in result
        assert "commission_cost" in result
        assert "total_cost" in result
        assert "cost_pct" in result
        assert "cost_bps" in result

        # Total should be sum of components
        expected_total = (result["spread_cost"] + result["slippage_cost"]
                         + result["commission_cost"])
        assert abs(result["total_cost"] - expected_total) < 0.01

    def test_spread_cost_calculation(self):
        """Spread cost = trade_value * spread_bps/10000 * 2 (both legs)."""
        result = estimate_round_trip_cost(
            entry_price=100.0, qty=100, spread_bps=5.0,
            slippage_bps=0.0, commission_per_share=0.0,
        )
        # trade_value = $10,000
        # spread_cost = 10000 * (5/10000) * 2 = $10.00
        assert result["spread_cost"] == 10.0
        assert result["slippage_cost"] == 0.0
        assert result["commission_cost"] == 0.0
        assert result["total_cost"] == 10.0

    def test_slippage_cost_calculation(self):
        """Slippage cost = trade_value * slippage_bps/10000 * 2."""
        result = estimate_round_trip_cost(
            entry_price=200.0, qty=50, spread_bps=0.0,
            slippage_bps=10.0, commission_per_share=0.0,
        )
        # trade_value = $10,000
        # slippage_cost = 10000 * (10/10000) * 2 = $20.00
        assert result["slippage_cost"] == 20.0
        assert result["total_cost"] == 20.0

    def test_commission_cost_calculation(self):
        """Commission cost = commission_per_share * qty * 2."""
        result = estimate_round_trip_cost(
            entry_price=100.0, qty=100, spread_bps=0.0,
            slippage_bps=0.0, commission_per_share=0.01,
        )
        # commission_cost = 0.01 * 100 * 2 = $2.00
        assert result["commission_cost"] == 2.0
        assert result["total_cost"] == 2.0

    def test_cost_bps_matches_cost_pct(self):
        """cost_bps should be cost_pct * 10000."""
        result = estimate_round_trip_cost(entry_price=150.0, qty=50)
        assert abs(result["cost_bps"] - result["cost_pct"] * 10000) < 0.1

    def test_different_trade_sizes(self):
        """Larger trades should have proportionally larger absolute costs."""
        small = estimate_round_trip_cost(entry_price=100.0, qty=10)
        large = estimate_round_trip_cost(entry_price=100.0, qty=100)

        # Total cost scales linearly with qty (same entry price)
        ratio = large["total_cost"] / small["total_cost"]
        assert abs(ratio - 10.0) < 0.01

    def test_zero_trade_value(self):
        """Zero trade value should not cause division errors."""
        result = estimate_round_trip_cost(entry_price=0.0, qty=0)
        assert result["total_cost"] == 0.0
        assert result["cost_pct"] == 0
        assert result["cost_bps"] == 0

    def test_custom_parameters_override_defaults(self):
        """Explicit spread/slippage/commission override config defaults."""
        result = estimate_round_trip_cost(
            entry_price=100.0, qty=100,
            spread_bps=1.0, slippage_bps=1.0, commission_per_share=0.005,
        )
        # trade_value = $10,000
        # spread = 10000 * 1/10000 * 2 = $2.00
        # slippage = 10000 * 1/10000 * 2 = $2.00
        # commission = 0.005 * 100 * 2 = $1.00
        assert result["spread_cost"] == 2.0
        assert result["slippage_cost"] == 2.0
        assert result["commission_cost"] == 1.0
        assert result["total_cost"] == 5.0

    def test_high_price_stock(self):
        """Cost estimation works for high-priced stocks."""
        result = estimate_round_trip_cost(
            entry_price=5000.0, qty=10,
            spread_bps=3.0, slippage_bps=2.0, commission_per_share=0.0,
        )
        # trade_value = $50,000
        # spread = 50000 * 3/10000 * 2 = $30.00
        # slippage = 50000 * 2/10000 * 2 = $20.00
        assert result["spread_cost"] == 30.0
        assert result["slippage_cost"] == 20.0
        assert result["total_cost"] == 50.0

    def test_penny_stock(self):
        """Cost estimation works for low-priced stocks."""
        result = estimate_round_trip_cost(
            entry_price=1.0, qty=1000,
            spread_bps=3.0, slippage_bps=2.0, commission_per_share=0.0,
        )
        # trade_value = $1,000
        # spread = 1000 * 3/10000 * 2 = $0.60
        # slippage = 1000 * 2/10000 * 2 = $0.40
        assert result["spread_cost"] == 0.6
        assert result["slippage_cost"] == 0.4


# ===================================================================
# is_trade_profitable_after_costs (negative-EV rejection)
# ===================================================================

class TestIsTradeProfitableAfterCosts:
    """Tests for EV-based trade profitability checks."""

    def test_clearly_profitable_trade(self):
        """Large TP distance with high win rate should be profitable."""
        profitable, details = is_trade_profitable_after_costs(
            entry_price=100.0,
            take_profit=110.0,  # $10 profit per share
            stop_loss=98.0,     # $2 loss per share
            qty=100,
            side="buy",
            win_rate=0.60,
        )
        assert profitable is True
        assert details["expected_value"] > 0
        assert details["gross_profit"] == 1000.0  # (110-100) * 100
        assert details["gross_loss"] == 200.0      # (100-98) * 100

    def test_negative_ev_trade_rejected(self):
        """A trade with tiny profit target and large stop should be rejected."""
        profitable, details = is_trade_profitable_after_costs(
            entry_price=100.0,
            take_profit=100.10,  # Only $0.10 profit per share
            stop_loss=95.0,      # $5 loss per share
            qty=100,
            side="buy",
            win_rate=0.50,
        )
        assert profitable is False
        assert details["expected_value"] < 0

    def test_short_side_calculation(self):
        """Short trades compute profit/loss correctly."""
        profitable, details = is_trade_profitable_after_costs(
            entry_price=100.0,
            take_profit=90.0,   # Short profits when price drops
            stop_loss=105.0,    # Short stops when price rises
            qty=100,
            side="sell",
            win_rate=0.55,
        )
        assert details["gross_profit"] == 1000.0  # (100-90) * 100
        assert details["gross_loss"] == 500.0      # (105-100) * 100
        assert profitable is True

    def test_details_include_all_cost_fields(self):
        """The returned details dict should contain all expected fields."""
        _, details = is_trade_profitable_after_costs(
            entry_price=100.0,
            take_profit=105.0,
            stop_loss=98.0,
            qty=50,
            side="buy",
            win_rate=0.55,
        )
        expected_keys = {
            "spread_cost", "slippage_cost", "commission_cost",
            "total_cost", "cost_pct", "cost_bps",
            "gross_profit", "gross_loss", "net_profit", "net_loss",
            "expected_value", "win_rate", "profitable",
        }
        assert expected_keys.issubset(set(details.keys()))

    def test_net_profit_accounts_for_costs(self):
        """net_profit = gross_profit - total_cost."""
        _, details = is_trade_profitable_after_costs(
            entry_price=100.0,
            take_profit=110.0,
            stop_loss=95.0,
            qty=100,
            side="buy",
            win_rate=0.55,
        )
        expected_net_profit = details["gross_profit"] - details["total_cost"]
        assert abs(details["net_profit"] - expected_net_profit) < 0.01

    def test_net_loss_includes_costs(self):
        """net_loss = gross_loss + total_cost."""
        _, details = is_trade_profitable_after_costs(
            entry_price=100.0,
            take_profit=110.0,
            stop_loss=95.0,
            qty=100,
            side="buy",
            win_rate=0.55,
        )
        expected_net_loss = details["gross_loss"] + details["total_cost"]
        assert abs(details["net_loss"] - expected_net_loss) < 0.01

    def test_win_rate_in_details(self):
        """The win_rate should be reflected in the details."""
        _, details = is_trade_profitable_after_costs(
            entry_price=100.0,
            take_profit=105.0,
            stop_loss=98.0,
            qty=50,
            side="buy",
            win_rate=0.65,
        )
        assert details["win_rate"] == 0.65

    def test_breakeven_trade_near_threshold(self):
        """A near-breakeven trade may be rejected due to minimum return threshold."""
        # Tiny profit: 1 cent per share, 100 shares = $1 gross profit
        profitable, details = is_trade_profitable_after_costs(
            entry_price=100.0,
            take_profit=100.01,
            stop_loss=99.99,
            qty=100,
            side="buy",
            win_rate=0.55,
        )
        # The $1 profit is likely eaten by costs
        assert profitable is False

    @patch("oms.transaction_cost._get_strategy_win_rate", return_value=0.70)
    def test_strategy_specific_win_rate_used(self, mock_wr):
        """When strategy is provided, _get_strategy_win_rate is called."""
        profitable, details = is_trade_profitable_after_costs(
            entry_price=100.0,
            take_profit=110.0,
            stop_loss=95.0,
            qty=100,
            side="buy",
            win_rate=0.50,       # Default, should be overridden
            strategy="ORB",
        )
        mock_wr.assert_called_once_with("ORB", default_win_rate=0.50)
        assert details["win_rate"] == 0.70


# ===================================================================
# Spread cost verification
# ===================================================================

class TestSpreadCostCalculation:
    """Focused tests on spread cost behavior."""

    def test_spread_is_paid_on_both_legs(self):
        """Spread cost multiplied by 2 (entry + exit)."""
        result = estimate_round_trip_cost(
            entry_price=100.0, qty=1,
            spread_bps=10.0, slippage_bps=0.0, commission_per_share=0.0,
        )
        # trade_value = $100
        # One leg: 100 * 10/10000 = $0.10
        # Both legs: $0.20
        assert result["spread_cost"] == 0.2

    def test_zero_spread(self):
        """Zero spread should result in zero spread cost."""
        result = estimate_round_trip_cost(
            entry_price=100.0, qty=100,
            spread_bps=0.0, slippage_bps=2.0, commission_per_share=0.0,
        )
        assert result["spread_cost"] == 0.0

    def test_high_spread_environment(self):
        """High-spread environment (e.g., 20 bps) correctly calculated."""
        result = estimate_round_trip_cost(
            entry_price=50.0, qty=200,
            spread_bps=20.0, slippage_bps=0.0, commission_per_share=0.0,
        )
        # trade_value = $10,000
        # spread = 10000 * 20/10000 * 2 = $40
        assert result["spread_cost"] == 40.0


# ===================================================================
# Edge cases
# ===================================================================

class TestTransactionCostEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_share_trade(self):
        """Cost estimation for a single share."""
        result = estimate_round_trip_cost(
            entry_price=150.0, qty=1,
            spread_bps=3.0, slippage_bps=2.0, commission_per_share=0.0,
        )
        assert result["total_cost"] > 0
        assert result["cost_bps"] > 0

    def test_large_position(self):
        """Cost estimation for a very large position."""
        result = estimate_round_trip_cost(
            entry_price=100.0, qty=100_000,
            spread_bps=3.0, slippage_bps=2.0, commission_per_share=0.0,
        )
        # trade_value = $10,000,000
        # spread = 10M * 3/10000 * 2 = $6,000
        # slippage = 10M * 2/10000 * 2 = $4,000
        assert result["spread_cost"] == 6000.0
        assert result["slippage_cost"] == 4000.0
        assert result["total_cost"] == 10000.0

    def test_cost_pct_is_independent_of_position_size(self):
        """cost_pct (and cost_bps) should be the same regardless of qty."""
        small = estimate_round_trip_cost(
            entry_price=100.0, qty=10,
            spread_bps=3.0, slippage_bps=2.0, commission_per_share=0.0,
        )
        large = estimate_round_trip_cost(
            entry_price=100.0, qty=10_000,
            spread_bps=3.0, slippage_bps=2.0, commission_per_share=0.0,
        )
        assert small["cost_bps"] == large["cost_bps"]
        assert small["cost_pct"] == large["cost_pct"]
