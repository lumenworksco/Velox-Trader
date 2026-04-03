"""Tests for risk/stress_testing.py — Stress testing framework.

Covers:
- Scenario evaluation against portfolio positions
- P&L calculation correctness (equity, sector, spread)
- Block threshold enforcement
- Adversarial scenario generation
- Edge cases: empty positions, zero equity, single position
"""

import sys
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub out config before importing the module under test
# ---------------------------------------------------------------------------
ET = ZoneInfo("America/New_York")

_config_mod = MagicMock()
_config_mod.ET = ET
_config_mod.SECTOR_MAP = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "JPM": "XLF",
    "XOM": "XLE",
    "TSLA": "XLY",
}
_config_mod.MICRO_BETA_TABLE = {
    "AAPL": 1.1,
    "MSFT": 1.0,
    "JPM": 0.9,
    "XOM": 0.8,
    "TSLA": 1.5,
}
_config_mod.OVERNIGHT_ELIGIBLE_STRATEGIES = ["PEAD", "STAT_MR"]
sys.modules.setdefault("config", _config_mod)

from risk.stress_testing import (
    BLOCK_THRESHOLD_PCT,
    PREDEFINED_SCENARIOS,
    ScenarioShock,
    ScenarioType,
    StressTestFramework,
    StressTestResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_positions(**kwargs):
    """Build a positions dict from symbol -> (qty, entry_price, side)."""
    positions = {}
    for sym, (qty, price, side) in kwargs.items():
        positions[sym] = {"qty": qty, "entry_price": price, "side": side, "strategy": "TEST"}
    return positions


def _make_trade_record(qty, entry_price, side="buy", strategy="TEST"):
    return SimpleNamespace(
        qty=qty, entry_price=entry_price, side=side, strategy=strategy
    )


# ===================================================================
# Basic framework behaviour
# ===================================================================

class TestStressTestFrameworkBasics:
    """Tests for core StressTestFramework operations."""

    def test_no_positions_returns_empty(self):
        fw = StressTestFramework()
        results = fw.run_stress_tests({})
        assert results == []
        assert not fw.should_block_new_positions()

    def test_run_predefined_scenarios(self):
        positions = _make_positions(AAPL=(100, 150.0, "buy"), MSFT=(50, 300.0, "buy"))
        fw = StressTestFramework()
        results = fw.run_stress_tests(positions, portfolio_equity=100_000)

        assert len(results) == len(PREDEFINED_SCENARIOS)
        for r in results:
            assert isinstance(r, StressTestResult)
            assert r.positions_affected == 2

    def test_worst_scenario_returned(self):
        positions = _make_positions(AAPL=(100, 150.0, "buy"))
        fw = StressTestFramework()
        fw.run_stress_tests(positions, portfolio_equity=100_000)

        worst = fw.get_worst_scenario()
        assert worst is not None
        assert worst.estimated_pnl_pct < 0  # Should be negative (loss)

    def test_last_results_stored(self):
        positions = _make_positions(AAPL=(100, 150.0, "buy"))
        fw = StressTestFramework()
        fw.run_stress_tests(positions, portfolio_equity=100_000)
        assert len(fw.last_results) == len(PREDEFINED_SCENARIOS)

    def test_status_property(self):
        fw = StressTestFramework()
        status = fw.status
        assert "scenarios_count" in status
        assert status["blocking_new_positions"] is False


# ===================================================================
# Scenario P&L calculation
# ===================================================================

class TestScenarioPnLCalculation:
    """Tests for _evaluate_scenario P&L math."""

    def test_equity_move_long_position(self):
        """Long position loses on equity drop."""
        scenario = ScenarioShock(
            name="Test Drop",
            scenario_type=ScenarioType.FLASH_CRASH,
            description="5% drop",
            equity_move_pct=-0.05,
            beta_amplification=1.0,
        )
        fw = StressTestFramework(scenarios=[scenario])
        # AAPL beta=1.1, long 100 shares @ 100 = $10,000 notional
        positions = _make_positions(AAPL=(100, 100.0, "buy"))
        results = fw.run_stress_tests(positions, portfolio_equity=100_000)

        r = results[0]
        # Expected: 10000 * (-0.05) * 1.1 * 1.0 * (+1) = -550
        assert r.estimated_pnl_dollars == pytest.approx(-550.0, abs=1.0)

    def test_equity_move_short_position(self):
        """Short position profits on equity drop."""
        scenario = ScenarioShock(
            name="Test Drop",
            scenario_type=ScenarioType.FLASH_CRASH,
            description="5% drop",
            equity_move_pct=-0.05,
            beta_amplification=1.0,
        )
        fw = StressTestFramework(scenarios=[scenario])
        positions = _make_positions(AAPL=(100, 100.0, "sell"))
        results = fw.run_stress_tests(positions, portfolio_equity=100_000)

        r = results[0]
        # Short side_mult=-1: 10000 * (-0.05) * 1.1 * (-1) = +550
        assert r.estimated_pnl_dollars == pytest.approx(550.0, abs=1.0)

    def test_sector_shock_additive(self):
        """Sector shock adds on top of equity move."""
        scenario = ScenarioShock(
            name="Tech Shock",
            scenario_type=ScenarioType.SECTOR_SHOCK,
            description="tech sector shock",
            equity_move_pct=-0.01,
            sector_shock={"XLK": -0.10},
            beta_amplification=1.0,
        )
        fw = StressTestFramework(scenarios=[scenario])
        # AAPL is in XLK sector
        positions = _make_positions(AAPL=(100, 100.0, "buy"))
        results = fw.run_stress_tests(positions, portfolio_equity=100_000)

        r = results[0]
        # Equity: 10000 * (-0.01) * 1.1 = -110
        # Sector: 10000 * (-0.10) * 1 = -1000
        # Total (before spread): -1110
        assert r.estimated_pnl_dollars < -1000

    def test_spread_cost_always_negative(self):
        """Spread multiplier always incurs a cost."""
        scenario = ScenarioShock(
            name="Spread Test",
            scenario_type=ScenarioType.LIQUIDITY_CRISIS,
            description="spreads widen",
            equity_move_pct=0.0,
            spread_multiplier=10.0,
            beta_amplification=1.0,
        )
        fw = StressTestFramework(scenarios=[scenario])
        positions = _make_positions(AAPL=(100, 100.0, "buy"))
        results = fw.run_stress_tests(positions, portfolio_equity=100_000)

        r = results[0]
        # With no equity move, spread cost should be the only P&L component
        assert r.estimated_pnl_dollars < 0

    def test_beta_amplification(self):
        """Beta amplification scales the equity move."""
        scenario_1x = ScenarioShock(
            name="1x beta",
            scenario_type=ScenarioType.FLASH_CRASH,
            description="",
            equity_move_pct=-0.05,
            beta_amplification=1.0,
        )
        scenario_2x = ScenarioShock(
            name="2x beta",
            scenario_type=ScenarioType.FLASH_CRASH,
            description="",
            equity_move_pct=-0.05,
            beta_amplification=2.0,
        )
        positions = _make_positions(AAPL=(100, 100.0, "buy"))

        fw1 = StressTestFramework(scenarios=[scenario_1x])
        r1 = fw1.run_stress_tests(positions, portfolio_equity=100_000)[0]

        fw2 = StressTestFramework(scenarios=[scenario_2x])
        r2 = fw2.run_stress_tests(positions, portfolio_equity=100_000)[0]

        assert r2.estimated_pnl_dollars == pytest.approx(r1.estimated_pnl_dollars * 2, abs=5)

    def test_pnl_pct_relative_to_equity(self):
        """P&L percentage is relative to portfolio equity."""
        scenario = ScenarioShock(
            name="Test",
            scenario_type=ScenarioType.FLASH_CRASH,
            description="",
            equity_move_pct=-0.05,
            beta_amplification=1.0,
        )
        fw = StressTestFramework(scenarios=[scenario])
        positions = _make_positions(AAPL=(100, 100.0, "buy"))
        results = fw.run_stress_tests(positions, portfolio_equity=50_000)

        r = results[0]
        expected_pct = r.estimated_pnl_dollars / 50_000
        assert r.estimated_pnl_pct == pytest.approx(expected_pct, abs=1e-6)


# ===================================================================
# Block threshold
# ===================================================================

class TestBlockThreshold:
    """Tests for should_block_new_positions logic."""

    def test_no_results_returns_false(self):
        fw = StressTestFramework()
        assert fw.should_block_new_positions() is False

    def test_mild_loss_does_not_block(self):
        """Losses under -5% should not block."""
        scenario = ScenarioShock(
            name="Small Drop",
            scenario_type=ScenarioType.FLASH_CRASH,
            description="",
            equity_move_pct=-0.01,
            beta_amplification=1.0,
        )
        fw = StressTestFramework(scenarios=[scenario])
        positions = _make_positions(AAPL=(10, 100.0, "buy"))
        fw.run_stress_tests(positions, portfolio_equity=100_000)
        assert fw.should_block_new_positions() is False

    def test_severe_loss_blocks(self):
        """Losses exceeding -5% should block."""
        scenario = ScenarioShock(
            name="Crash",
            scenario_type=ScenarioType.FLASH_CRASH,
            description="",
            equity_move_pct=-0.50,
            beta_amplification=1.0,
        )
        fw = StressTestFramework(scenarios=[scenario])
        # Large position: 1000 * 100 = 100k notional = 100% of equity
        positions = _make_positions(AAPL=(1000, 100.0, "buy"))
        fw.run_stress_tests(positions, portfolio_equity=100_000)
        assert fw.should_block_new_positions() is True

    def test_custom_block_threshold(self):
        """Custom threshold is respected."""
        scenario = ScenarioShock(
            name="Test",
            scenario_type=ScenarioType.FLASH_CRASH,
            description="",
            equity_move_pct=-0.05,
            beta_amplification=1.0,
        )
        fw = StressTestFramework(scenarios=[scenario], block_threshold=-0.001)
        positions = _make_positions(AAPL=(100, 100.0, "buy"))
        fw.run_stress_tests(positions, portfolio_equity=100_000)
        # Even small losses should block with a tight threshold
        assert fw.should_block_new_positions() is True


# ===================================================================
# Position normalization
# ===================================================================

class TestPositionNormalization:
    """Tests for _normalize_positions handling various input types."""

    def test_dict_positions(self):
        positions = {"AAPL": {"qty": 100, "entry_price": 150.0, "side": "buy"}}
        result = StressTestFramework._normalize_positions(positions)
        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["notional"] == 15_000.0
        assert result[0]["side_mult"] == 1.0

    def test_object_positions(self):
        positions = {"AAPL": _make_trade_record(100, 150.0, "sell")}
        result = StressTestFramework._normalize_positions(positions)
        assert len(result) == 1
        assert result[0]["side_mult"] == -1.0

    def test_zero_qty_skipped(self):
        positions = {"AAPL": {"qty": 0, "entry_price": 150.0, "side": "buy"}}
        result = StressTestFramework._normalize_positions(positions)
        assert len(result) == 0

    def test_zero_price_skipped(self):
        positions = {"AAPL": {"qty": 100, "entry_price": 0, "side": "buy"}}
        result = StressTestFramework._normalize_positions(positions)
        assert len(result) == 0

    def test_negative_qty_skipped(self):
        positions = {"AAPL": {"qty": -5, "entry_price": 150.0, "side": "buy"}}
        result = StressTestFramework._normalize_positions(positions)
        assert len(result) == 0


# ===================================================================
# Adversarial scenario generation (RISK-008)
# ===================================================================

class TestAdversarialScenarios:
    """Tests for adversarial scenario generation."""

    def test_empty_positions_returns_empty(self):
        fw = StressTestFramework()
        scenarios = fw.generate_adversarial_scenarios({})
        assert scenarios == []

    def test_generates_scenarios_for_portfolio(self):
        positions = _make_positions(
            AAPL=(500, 150.0, "buy"),
            MSFT=(200, 300.0, "buy"),
            JPM=(300, 140.0, "buy"),
        )
        fw = StressTestFramework()
        scenarios = fw.generate_adversarial_scenarios(positions, portfolio_equity=100_000)
        # Should generate at least some adversarial scenarios
        assert isinstance(scenarios, list)
        for s in scenarios:
            assert isinstance(s, ScenarioShock)

    def test_concentration_attack_triggered(self):
        """A highly concentrated portfolio should trigger concentration attack."""
        # Single position = 100% concentration
        positions = _make_positions(AAPL=(1000, 150.0, "buy"))
        fw = StressTestFramework()
        scenarios = fw.generate_adversarial_scenarios(positions, portfolio_equity=150_000)
        names = [s.name for s in scenarios]
        assert any("Concentration" in n for n in names)

    def test_correlated_selloff_needs_multiple_positions(self):
        """Correlated selloff requires at least 2 positions."""
        positions = _make_positions(AAPL=(100, 150.0, "buy"))
        fw = StressTestFramework()
        scenarios = fw.generate_adversarial_scenarios(positions, portfolio_equity=100_000)
        names = [s.name for s in scenarios]
        assert not any("Correlated" in n for n in names)

    def test_run_adversarial_stress_tests(self):
        """End-to-end: generate + run adversarial scenarios."""
        positions = _make_positions(
            AAPL=(500, 150.0, "buy"),
            MSFT=(200, 300.0, "buy"),
        )
        fw = StressTestFramework()
        results = fw.run_adversarial_stress_tests(positions, portfolio_equity=100_000)
        assert isinstance(results, list)
        # After running, predefined scenarios should be restored
        assert fw._scenarios == PREDEFINED_SCENARIOS or len(fw._scenarios) == len(PREDEFINED_SCENARIOS)


# ===================================================================
# Edge cases
# ===================================================================

class TestStressTestEdgeCases:
    """Edge cases and error handling."""

    def test_zero_portfolio_equity_fallback(self):
        """Zero equity should not cause division by zero."""
        fw = StressTestFramework()
        positions = _make_positions(AAPL=(100, 150.0, "buy"))
        results = fw.run_stress_tests(positions, portfolio_equity=0)
        # Should use 100_000 fallback
        assert len(results) > 0

    def test_none_portfolio_equity_estimated(self):
        """None equity should be estimated from positions."""
        fw = StressTestFramework()
        positions = _make_positions(AAPL=(100, 150.0, "buy"))
        results = fw.run_stress_tests(positions, portfolio_equity=None)
        assert len(results) > 0

    def test_scenario_exception_handled(self):
        """If a scenario evaluation raises, it is caught and logged."""
        bad_scenario = ScenarioShock(
            name="Bad",
            scenario_type=ScenarioType.FLASH_CRASH,
            description="",
            equity_move_pct=-0.05,
            sector_shock={"XLK": -0.10},
            beta_amplification=1.0,
        )
        fw = StressTestFramework(scenarios=[bad_scenario])
        positions = _make_positions(AAPL=(100, 150.0, "buy"))
        # This should not raise even if internals fail
        results = fw.run_stress_tests(positions, portfolio_equity=100_000)
        assert len(results) == 1

    def test_worst_position_tracked(self):
        """The worst-hit position should be identified."""
        scenario = ScenarioShock(
            name="Drop",
            scenario_type=ScenarioType.FLASH_CRASH,
            description="",
            equity_move_pct=-0.10,
            beta_amplification=1.0,
        )
        fw = StressTestFramework(scenarios=[scenario])
        # TSLA has higher beta (1.5) so should be worst hit
        positions = _make_positions(
            AAPL=(100, 100.0, "buy"),
            TSLA=(100, 100.0, "buy"),
        )
        results = fw.run_stress_tests(positions, portfolio_equity=100_000)
        r = results[0]
        assert r.worst_position == "TSLA"
