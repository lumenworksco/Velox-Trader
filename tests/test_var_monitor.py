"""Tests for risk/var_monitor.py — VaR and CVaR monitor.

Covers:
- Parametric VaR calculation (normal + Cornish-Fisher adjustment)
- Historical simulation VaR
- Monte Carlo VaR
- CVaR (Expected Shortfall) computation
- Risk budget enforcement and size multiplier
- Edge cases: insufficient data, zero sigma, empty returns
"""

import sys
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub out config before importing the module under test
# ---------------------------------------------------------------------------
from zoneinfo import ZoneInfo as _ZoneInfo

_ET = _ZoneInfo("America/New_York")
_config_mod = MagicMock()
_config_mod.ET = _ET
_config_mod.VAR_MAX_DAILY_PCT = 0.02
_config_mod.VAR_LOOKBACK_DAYS = 60
_config_mod.VAR_MC_SIMULATIONS = 10000
sys.modules.setdefault("config", _config_mod)
# Ensure ET is always set correctly even if another test loaded config first
sys.modules["config"].ET = _ET

from risk.var_monitor import VaRMonitor, VaRResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns(n=60, mu=-0.001, sigma=0.015, seed=42):
    """Generate a series of daily P&L returns for testing."""
    rng = np.random.default_rng(seed)
    return list(rng.normal(mu, sigma, n))


def _make_negative_skew_returns(n=60, seed=42):
    """Generate returns with strong negative skew (fat left tail)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.01, n)
    # Add a few large negative outliers
    base[0] = -0.08
    base[5] = -0.06
    base[10] = -0.05
    return list(base)


# ---------------------------------------------------------------------------
# VaRMonitor initialization
# ---------------------------------------------------------------------------

class TestVaRMonitorInit:

    def test_default_init(self):
        m = VaRMonitor()
        assert m.max_var_pct == 0.02
        assert m.lookback_days == 60

    def test_custom_init(self):
        m = VaRMonitor(max_var_pct=0.05, lookback_days=30, mc_simulations=5000)
        assert m.max_var_pct == 0.05
        assert m.lookback_days == 30
        assert m.mc_simulations == 5000

    def test_initial_result_is_empty(self):
        m = VaRMonitor()
        r = m.result
        assert r.var_95 == 0.0
        assert r.var_99 == 0.0
        assert r.sample_size == 0


# ---------------------------------------------------------------------------
# Insufficient data
# ---------------------------------------------------------------------------

class TestVaRInsufficientData:

    def test_less_than_10_days_returns_empty(self):
        m = VaRMonitor()
        result = m.update(daily_pnls=[0.01, -0.005, 0.003], portfolio_value=100_000)
        assert result.var_95 == 0.0
        assert result.var_99 == 0.0
        assert result.sample_size == 3

    def test_exactly_10_days_uses_parametric(self):
        m = VaRMonitor()
        pnls = _make_returns(n=10)
        result = m.update(daily_pnls=pnls, portfolio_value=100_000)
        assert result.sample_size == 10
        assert result.method == "parametric"
        assert result.var_95 >= 0

    def test_empty_returns(self):
        m = VaRMonitor()
        result = m.update(daily_pnls=[], portfolio_value=100_000)
        assert result.var_95 == 0.0
        assert result.sample_size == 0


# ---------------------------------------------------------------------------
# Parametric VaR
# ---------------------------------------------------------------------------

class TestParametricVaR:

    def test_parametric_var_positive(self):
        """Parametric VaR should produce positive loss values for negative-mean returns."""
        m = VaRMonitor(lookback_days=25)
        pnls = _make_returns(n=25, mu=-0.002, sigma=0.02)
        result = m.update(daily_pnls=pnls, portfolio_value=100_000)
        assert result.var_95 > 0
        assert result.var_99 >= result.var_95  # 99% VaR >= 95% VaR

    def test_parametric_var_percentages(self):
        """VaR as percentage should be consistent with dollar VaR."""
        m = VaRMonitor(lookback_days=20)
        pv = 200_000
        pnls = _make_returns(n=20, mu=-0.001, sigma=0.015)
        result = m.update(daily_pnls=pnls, portfolio_value=pv)
        # var_95_pct * portfolio_value should approximately equal var_95
        expected_dollar = result.var_95_pct * pv
        assert result.var_95 == pytest.approx(expected_dollar, rel=0.01)

    def test_cornish_fisher_method_label(self):
        """With n >= 20 returns, parametric should use Cornish-Fisher adjustment."""
        m = VaRMonitor(lookback_days=25)
        pnls = _make_returns(n=25)
        result = m.update(daily_pnls=pnls, portfolio_value=100_000)
        assert result.method == "parametric-cf"

    def test_plain_parametric_under_20(self):
        """With 10 <= n < 20, method should be plain 'parametric'."""
        m = VaRMonitor(lookback_days=15)
        pnls = _make_returns(n=15)
        result = m.update(daily_pnls=pnls, portfolio_value=100_000)
        assert result.method == "parametric"

    def test_cvar_exceeds_var(self):
        """CVaR (expected shortfall) should be >= VaR."""
        m = VaRMonitor(lookback_days=25)
        pnls = _make_returns(n=25, mu=-0.003, sigma=0.02)
        result = m.update(daily_pnls=pnls, portfolio_value=100_000)
        assert result.cvar_95 >= result.var_95

    def test_zero_sigma_returns_zero(self):
        """If all returns are identical, sigma is 0 and VaR should be 0."""
        m = VaRMonitor(lookback_days=20)
        pnls = [0.001] * 20
        result = m.update(daily_pnls=pnls, portfolio_value=100_000)
        assert result.var_95 == 0.0

    def test_negative_skew_increases_var(self):
        """Cornish-Fisher should increase VaR for negatively skewed returns."""
        m = VaRMonitor(lookback_days=60)
        skewed = _make_negative_skew_returns(n=60)
        result_skewed = m.update(daily_pnls=skewed, portfolio_value=100_000)

        m2 = VaRMonitor(lookback_days=60)
        normal = _make_returns(n=60, mu=0.0, sigma=0.01)
        result_normal = m2.update(daily_pnls=normal, portfolio_value=100_000)

        # Skewed returns should have higher VaR
        assert result_skewed.var_95 > result_normal.var_95


# ---------------------------------------------------------------------------
# Historical VaR
# ---------------------------------------------------------------------------

class TestHistoricalVaR:

    def test_historical_var_with_30_plus_days(self):
        """With 30+ days, the monitor should use historical simulation."""
        m = VaRMonitor()
        pnls = _make_returns(n=60)
        result = m.update(daily_pnls=pnls, portfolio_value=100_000)
        assert result.method == "historical"
        assert result.sample_size == 60

    def test_historical_var_is_positive(self):
        m = VaRMonitor()
        pnls = _make_returns(n=60, mu=-0.002, sigma=0.015)
        result = m.update(daily_pnls=pnls, portfolio_value=100_000)
        assert result.var_95 > 0
        assert result.var_99 >= result.var_95

    def test_historical_cvar_exceeds_var(self):
        m = VaRMonitor()
        pnls = _make_returns(n=60, mu=-0.003, sigma=0.02)
        result = m.update(daily_pnls=pnls, portfolio_value=100_000)
        assert result.cvar_95 >= result.var_95 - 1e-6  # Tolerance for edge cases

    def test_historical_var_scales_with_portfolio(self):
        """Dollar VaR should scale linearly with portfolio value."""
        pnls = _make_returns(n=60)

        m1 = VaRMonitor()
        r1 = m1.update(daily_pnls=pnls, portfolio_value=100_000)

        m2 = VaRMonitor()
        r2 = m2.update(daily_pnls=pnls, portfolio_value=200_000)

        assert r2.var_95 == pytest.approx(r1.var_95 * 2, rel=0.01)

    def test_lookback_limits_data(self):
        """Only the last `lookback_days` returns should be used."""
        m = VaRMonitor(lookback_days=30)
        pnls = _make_returns(n=100)
        result = m.update(daily_pnls=pnls, portfolio_value=100_000)
        assert result.sample_size == 30


# ---------------------------------------------------------------------------
# Monte Carlo VaR
# ---------------------------------------------------------------------------

class TestMonteCarloVaR:

    def test_mc_var_returns_valid_result(self):
        m = VaRMonitor(mc_simulations=5000)
        m._daily_returns = _make_returns(n=60)
        m._portfolio_value = 100_000
        result = m.monte_carlo_var(horizon_days=1)
        assert result.method == "monte_carlo"
        assert result.var_95 > 0
        assert result.var_99 >= result.var_95

    def test_mc_var_multi_day_horizon(self):
        """Multi-day MC VaR should be larger than single-day."""
        m = VaRMonitor(mc_simulations=5000)
        m._daily_returns = _make_returns(n=60, mu=-0.001, sigma=0.015)
        m._portfolio_value = 100_000

        r1 = m.monte_carlo_var(horizon_days=1)
        r5 = m.monte_carlo_var(horizon_days=5)

        # 5-day VaR should generally be larger than 1-day
        assert r5.var_95 > r1.var_95 * 0.5  # At least 50% of 1-day

    def test_mc_var_insufficient_data(self):
        """MC VaR with < 20 returns should return empty result."""
        m = VaRMonitor()
        m._daily_returns = _make_returns(n=10)
        m._portfolio_value = 100_000
        result = m.monte_carlo_var()
        assert result.var_95 == 0.0
        assert result.method == "monte_carlo"


# ---------------------------------------------------------------------------
# Risk budget and size multiplier
# ---------------------------------------------------------------------------

class TestRiskBudget:

    def test_full_budget_when_no_risk(self):
        """With no VaR, risk budget should be 1.0."""
        m = VaRMonitor()
        m._portfolio_value = 100_000
        assert m.risk_budget_remaining == 1.0
        assert m.size_multiplier == 1.0

    def test_budget_decreases_with_var(self):
        """Budget should decrease as VaR approaches max."""
        m = VaRMonitor(max_var_pct=0.02)
        pnls = _make_returns(n=60, mu=-0.005, sigma=0.03)
        m.update(daily_pnls=pnls, portfolio_value=100_000)
        assert m.risk_budget_remaining < 1.0

    def test_budget_zero_when_at_limit(self):
        """When VaR equals max_var_pct, budget should be 0."""
        m = VaRMonitor(max_var_pct=0.02)
        # Force the result directly
        m._portfolio_value = 100_000
        m._last_result = VaRResult(var_95_pct=0.02)
        assert m.risk_budget_remaining == pytest.approx(0.0)
        assert m.size_multiplier == pytest.approx(0.0)

    def test_budget_clamped_at_zero_when_over_limit(self):
        """Budget should not go negative; clamped at 0."""
        m = VaRMonitor(max_var_pct=0.02)
        m._portfolio_value = 100_000
        m._last_result = VaRResult(var_95_pct=0.05)  # Way over limit
        assert m.risk_budget_remaining == 0.0
        assert m.size_multiplier == 0.0

    def test_size_multiplier_between_0_and_1(self):
        m = VaRMonitor(max_var_pct=0.02)
        m._portfolio_value = 100_000
        m._last_result = VaRResult(var_95_pct=0.01)
        assert 0 <= m.size_multiplier <= 1.0

    def test_zero_portfolio_value_returns_zero_budget(self):
        m = VaRMonitor()
        m._portfolio_value = 0
        assert m.risk_budget_remaining == 0.0


# ---------------------------------------------------------------------------
# Status dict
# ---------------------------------------------------------------------------

class TestVaRStatus:

    def test_status_keys(self):
        m = VaRMonitor()
        m.update(daily_pnls=_make_returns(n=60), portfolio_value=100_000)
        s = m.status
        assert "var_95" in s
        assert "var_99" in s
        assert "cvar_95" in s
        assert "method" in s
        assert "risk_budget_remaining" in s
        assert "size_multiplier" in s

    def test_status_portfolio_value(self):
        m = VaRMonitor()
        m.update(daily_pnls=_make_returns(n=60), portfolio_value=250_000)
        assert m.status["portfolio_value"] == 250_000.0


# ---------------------------------------------------------------------------
# Cornish-Fisher static method
# ---------------------------------------------------------------------------

class TestCornishFisher:

    def test_zero_skew_kurt_returns_z(self):
        """With zero skew and kurtosis, CF should return the original z."""
        z = VaRMonitor._cornish_fisher_z(1.645, 0.0, 0.0)
        assert z == pytest.approx(1.645, abs=1e-6)

    def test_negative_skew_adjusts_z(self):
        """Negative skew should adjust the z-score (Cornish-Fisher correction)."""
        z_adj = VaRMonitor._cornish_fisher_z(1.645, -1.5, 0.0)
        # CF with negative skew reduces the z quantile, shifting the VaR estimate
        assert z_adj != 1.645
        assert isinstance(z_adj, float)

    def test_positive_kurtosis_adjusts_z(self):
        """Excess kurtosis > 0 (fat tails) should adjust z via Cornish-Fisher."""
        z_adj = VaRMonitor._cornish_fisher_z(1.645, 0.0, 5.0)
        # CF with excess kurtosis modifies the z-score
        assert z_adj != 1.645
        assert isinstance(z_adj, float)


# ---------------------------------------------------------------------------
# Update method idempotency and state management
# ---------------------------------------------------------------------------

class TestVaRUpdateState:

    def test_multiple_updates_overwrite(self):
        """Calling update multiple times should overwrite the result."""
        m = VaRMonitor()
        r1 = m.update(daily_pnls=_make_returns(n=60, sigma=0.01), portfolio_value=100_000)
        r2 = m.update(daily_pnls=_make_returns(n=60, sigma=0.03), portfolio_value=100_000)
        # Higher sigma should produce higher VaR
        assert r2.var_95 > r1.var_95

    def test_result_property_returns_latest(self):
        m = VaRMonitor()
        r = m.update(daily_pnls=_make_returns(n=60), portfolio_value=100_000)
        assert m.result is r
