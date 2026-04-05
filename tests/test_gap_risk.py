"""Tests for risk/gap_risk.py — Overnight gap risk management.

Covers:
- Gap statistics computation (update_gap_stats, update_gap_stats_from_ohlc)
- Gap risk estimation (compute_gap_risk)
- Position assessment (assess_position)
- should_close_before_eod() logic
- Overnight sizing multiplier
- Edge cases: insufficient data, unknown symbols, extreme gaps
"""

import sys
from datetime import datetime, time, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub out config before importing the module under test
# ---------------------------------------------------------------------------
ET = ZoneInfo("America/New_York")

_config_mod = MagicMock()
_config_mod.ET = ET
_config_mod.SECTOR_MAP = {}
sys.modules.setdefault("config", _config_mod)
# Ensure ET is always set correctly even if another test loaded config first
sys.modules["config"].ET = ET

from risk.gap_risk import (
    GapRiskManager,
    GapRiskResult,
    GapStats,
    HIGH_GAP_RISK_THRESHOLD,
    MAX_OVERNIGHT_EXPOSURE_PCT,
    MAX_SINGLE_OVERNIGHT_PCT,
    CLOSE_BY_TIME,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gap_returns(n=50, mean=0.0, std=0.01, seed=42):
    """Generate random gap returns."""
    rng = np.random.default_rng(seed)
    return list(rng.normal(mean, std, n))


def _make_high_risk_gaps(n=50, seed=42):
    """Generate gap returns with mean |gap| > HIGH_GAP_RISK_THRESHOLD (2%)."""
    rng = np.random.default_rng(seed)
    # Generate large gaps centered around 3%
    return list(rng.normal(0.0, 0.04, n))


def _make_manager(**kwargs):
    return GapRiskManager(**kwargs)


# ---------------------------------------------------------------------------
# Gap statistics
# ---------------------------------------------------------------------------

class TestGapStatistics:

    def test_update_gap_stats_basic(self):
        mgr = _make_manager()
        gaps = _make_gap_returns(n=50, mean=0.001, std=0.015)
        mgr.update_gap_stats("AAPL", gaps)

        stats = mgr._gap_stats.get("AAPL")
        assert stats is not None
        assert stats.symbol == "AAPL"
        assert stats.sample_count == 50
        assert stats.mean_gap == pytest.approx(np.mean(gaps), abs=1e-6)
        assert stats.mean_abs_gap == pytest.approx(np.mean(np.abs(gaps)), abs=1e-6)
        assert stats.std_gap == pytest.approx(np.std(gaps), abs=1e-6)

    def test_max_negative_and_positive_gaps(self):
        mgr = _make_manager()
        gaps = [-0.05, -0.02, 0.0, 0.01, 0.03, 0.04, -0.01, 0.02, 0.0, -0.03]
        mgr.update_gap_stats("TSLA", gaps)

        stats = mgr._gap_stats["TSLA"]
        assert stats.max_negative_gap == -0.05
        assert stats.max_positive_gap == 0.04

    def test_gap_95_percentile(self):
        mgr = _make_manager()
        gaps = _make_gap_returns(n=100, std=0.02)
        mgr.update_gap_stats("SPY", gaps)

        stats = mgr._gap_stats["SPY"]
        expected_95 = float(np.percentile(np.abs(gaps), 95))
        assert stats.gap_95_pct == pytest.approx(expected_95, abs=1e-6)

    def test_insufficient_data_skipped(self):
        mgr = _make_manager()
        mgr.update_gap_stats("TINY", [0.01, 0.02, -0.01])  # Only 3 gaps
        assert "TINY" not in mgr._gap_stats

    def test_exactly_5_gaps_accepted(self):
        mgr = _make_manager()
        mgr.update_gap_stats("FIVE", [0.01, 0.02, -0.01, 0.005, -0.005])
        assert "FIVE" in mgr._gap_stats
        assert mgr._gap_stats["FIVE"].sample_count == 5

    def test_update_gap_stats_from_ohlc(self):
        mgr = _make_manager()
        closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0,
                           105.0, 106.0, 107.0, 108.0, 109.0])
        opens = np.array([100.5, 101.5, 102.5, 103.5, 104.5,
                          105.5, 106.5, 107.5, 108.5, 109.5])
        mgr.update_gap_stats_from_ohlc("SPY", opens, closes)
        assert "SPY" in mgr._gap_stats

    def test_update_gap_stats_from_ohlc_insufficient_data(self):
        mgr = _make_manager()
        mgr.update_gap_stats_from_ohlc("SPY", np.array([100.0, 101.0]), np.array([100.0, 101.0]))
        assert "SPY" not in mgr._gap_stats


# ---------------------------------------------------------------------------
# Gap risk computation
# ---------------------------------------------------------------------------

class TestComputeGapRisk:

    def test_known_symbol_uses_95th_percentile(self):
        mgr = _make_manager()
        gaps = _make_gap_returns(n=100, std=0.02)
        mgr.update_gap_stats("AAPL", gaps)

        position_size = 10_000.0
        risk = mgr.compute_gap_risk("AAPL", position_size)

        expected_gap_95 = float(np.percentile(np.abs(gaps), 95))
        expected_risk = position_size * expected_gap_95
        assert risk == pytest.approx(expected_risk, abs=1.0)

    def test_unknown_symbol_uses_default_2pct(self):
        mgr = _make_manager()
        risk = mgr.compute_gap_risk("UNKNOWN", 10_000.0)
        assert risk == pytest.approx(10_000 * 0.02, abs=0.01)

    def test_negative_position_uses_absolute(self):
        mgr = _make_manager()
        risk = mgr.compute_gap_risk("UNKNOWN", -10_000.0)
        assert risk == pytest.approx(10_000 * 0.02, abs=0.01)

    def test_zero_position_zero_risk(self):
        mgr = _make_manager()
        risk = mgr.compute_gap_risk("UNKNOWN", 0.0)
        assert risk == 0.0


# ---------------------------------------------------------------------------
# Position assessment
# ---------------------------------------------------------------------------

class TestAssessPosition:

    def test_low_risk_symbol_hold_recommendation(self):
        mgr = _make_manager()
        gaps = _make_gap_returns(n=100, mean=0.0, std=0.005)  # Small gaps
        mgr.update_gap_stats("AAPL", gaps)

        result = mgr.assess_position("AAPL", 5_000.0, portfolio_equity=100_000.0)
        assert isinstance(result, GapRiskResult)
        assert result.recommended_action == "hold"
        assert not result.is_high_risk

    def test_high_risk_symbol_flagged(self):
        mgr = _make_manager()
        gaps = _make_high_risk_gaps(n=100)
        mgr.update_gap_stats("TSLA", gaps)

        result = mgr.assess_position("TSLA", 10_000.0, portfolio_equity=100_000.0)
        assert result.is_high_risk

    def test_oversize_position_reduce_recommendation(self):
        mgr = _make_manager()
        gaps = _make_high_risk_gaps(n=100)
        mgr.update_gap_stats("TSLA", gaps)

        # Position is 12% of portfolio, over the 8% max single limit
        result = mgr.assess_position("TSLA", 12_000.0, portfolio_equity=100_000.0)
        assert result.recommended_action in ("reduce", "close")

    def test_gap_risk_pct_calculation(self):
        mgr = _make_manager()
        gaps = _make_gap_returns(n=50, std=0.02)
        mgr.update_gap_stats("AAPL", gaps)

        result = mgr.assess_position("AAPL", 10_000.0, portfolio_equity=100_000.0)
        assert result.gap_risk_pct == pytest.approx(
            result.gap_risk_dollars / 100_000.0, abs=1e-6
        )

    def test_zero_equity_safe(self):
        mgr = _make_manager()
        result = mgr.assess_position("AAPL", 10_000.0, portfolio_equity=0.0)
        assert result.gap_risk_pct == 0.0


# ---------------------------------------------------------------------------
# should_close_before_eod
# ---------------------------------------------------------------------------

class TestShouldCloseBeforeEOD:

    def test_before_close_time_returns_false(self):
        mgr = _make_manager()
        early_time = datetime(2026, 4, 4, 10, 0, 0, tzinfo=ET)
        trade = {"hold_type": "day"}
        assert not mgr.should_close_before_eod(trade, current_time=early_time)

    def test_after_close_time_day_trade_returns_true(self):
        mgr = _make_manager()
        late_time = datetime(2026, 4, 4, 15, 56, 0, tzinfo=ET)
        trade = {"hold_type": "day"}
        assert mgr.should_close_before_eod(trade, current_time=late_time)

    def test_after_close_time_swing_trade_returns_false(self):
        mgr = _make_manager()
        late_time = datetime(2026, 4, 4, 15, 56, 0, tzinfo=ET)
        trade = {"hold_type": "swing"}
        assert not mgr.should_close_before_eod(trade, current_time=late_time)

    def test_exactly_at_close_time_day_trade_returns_true(self):
        mgr = _make_manager()
        at_close = datetime(2026, 4, 4, 15, 55, 0, tzinfo=ET)
        trade = {"hold_type": "day"}
        assert mgr.should_close_before_eod(trade, current_time=at_close)

    def test_trade_object_with_attribute(self):
        """should_close_before_eod also supports objects with hold_type attribute."""
        mgr = _make_manager()
        late_time = datetime(2026, 4, 4, 15, 56, 0, tzinfo=ET)

        class MockTrade:
            hold_type = "day"

        assert mgr.should_close_before_eod(MockTrade(), current_time=late_time)

    def test_missing_hold_type_defaults_to_day(self):
        mgr = _make_manager()
        late_time = datetime(2026, 4, 4, 15, 56, 0, tzinfo=ET)
        trade = {}  # No hold_type key
        assert mgr.should_close_before_eod(trade, current_time=late_time)

    def test_custom_close_by_time(self):
        mgr = _make_manager(close_by=time(15, 45))
        # 3:50 PM is after 3:45 PM
        late_time = datetime(2026, 4, 4, 15, 50, 0, tzinfo=ET)
        trade = {"hold_type": "day"}
        assert mgr.should_close_before_eod(trade, current_time=late_time)


# ---------------------------------------------------------------------------
# Overnight sizing multiplier
# ---------------------------------------------------------------------------

class TestOvernightSizingMultiplier:

    def test_unknown_symbol_gets_conservative(self):
        mgr = _make_manager()
        mult = mgr.get_overnight_sizing_multiplier("UNKNOWN")
        assert mult == 0.5

    def test_low_gap_symbol_full_sizing(self):
        mgr = _make_manager()
        gaps = _make_gap_returns(n=50, std=0.005)  # avg |gap| ~ 0.4%, well below 1%
        mgr.update_gap_stats("SPY", gaps)
        mult = mgr.get_overnight_sizing_multiplier("SPY")
        assert mult == 1.0

    def test_high_gap_symbol_reduced_sizing(self):
        mgr = _make_manager()
        gaps = _make_high_risk_gaps(n=100)  # avg |gap| >> 2%
        mgr.update_gap_stats("TSLA", gaps)
        mult = mgr.get_overnight_sizing_multiplier("TSLA")
        assert mult < 1.0
        assert mult >= 0.25

    def test_multiplier_bounds(self):
        """Multiplier should always be in [0.25, 1.0]."""
        mgr = _make_manager()
        # Extreme gaps
        extreme_gaps = list(np.ones(50) * 0.10)  # 10% gaps every day
        mgr.update_gap_stats("EXTREME", extreme_gaps)
        mult = mgr.get_overnight_sizing_multiplier("EXTREME")
        assert 0.25 <= mult <= 1.0

    def test_insufficient_data_conservative(self):
        """Symbols with < 5 gap samples should get conservative sizing."""
        mgr = _make_manager()
        mgr._gap_stats["TINY"] = GapStats(symbol="TINY", sample_count=3)
        mult = mgr.get_overnight_sizing_multiplier("TINY")
        assert mult == 0.5
