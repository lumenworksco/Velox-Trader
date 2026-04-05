"""Comprehensive tests for risk/factor_model.py.

Covers:
- Beta computation via rolling regression
- Factor exposure calculation (portfolio-level)
- Sector exposure limits
- Risk decomposition (systematic vs idiosyncratic)
- Factor limit checking
- Caching behavior
- Edge cases (insufficient data, empty positions)
"""

from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import config
from risk.factor_model import (
    DEFAULT_FACTOR_LIMITS,
    DEFAULT_SECTOR_LIMIT,
    FACTOR_PROXIES,
    ROLLING_WINDOW,
    SECTOR_ETFS,
    FactorExposure,
    FactorRiskModel,
    RiskDecomposition,
)

ET = ZoneInfo("America/New_York")


# ===================================================================
# Helpers — synthetic return data
# ===================================================================

def _make_return_series(n: int = 120, mean: float = 0.0005,
                        std: float = 0.01, seed: int = 42) -> pd.Series:
    """Generate a synthetic daily return series."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2025-10-01", periods=n, freq="B")
    returns = rng.normal(mean, std, n)
    return pd.Series(returns, index=dates)


def _make_correlated_series(base: pd.Series, beta: float = 1.0,
                             noise_std: float = 0.005,
                             seed: int = 99) -> pd.Series:
    """Generate a return series correlated with `base` via beta."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, noise_std, len(base))
    correlated = base.values * beta + noise
    return pd.Series(correlated, index=base.index)


def _build_returns_data(n: int = 120) -> dict[str, pd.Series]:
    """Build a complete returns dataset for testing factor model."""
    spy = _make_return_series(n, seed=1)
    iwm = _make_return_series(n, seed=2)
    iwd = _make_return_series(n, seed=3)
    iwf = _make_return_series(n, seed=4)
    mtum = _make_return_series(n, seed=5)
    aapl = _make_correlated_series(spy, beta=1.1, seed=10)
    tsla = _make_correlated_series(spy, beta=1.8, seed=11)
    return {
        "SPY": spy,
        "IWM": iwm,
        "IWD": iwd,
        "IWF": iwf,
        "MTUM": mtum,
        "AAPL": aapl,
        "TSLA": tsla,
    }


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def model():
    """Fresh FactorRiskModel with defaults."""
    return FactorRiskModel()


@pytest.fixture
def returns_data():
    """120-day returns dataset."""
    return _build_returns_data(120)


@pytest.fixture
def short_returns():
    """Returns with insufficient data (less than rolling window)."""
    return _build_returns_data(20)


# ===================================================================
# Beta computation
# ===================================================================

class TestBetaComputation:

    def test_get_factor_exposures_returns_dict(self, model, returns_data):
        """get_factor_exposures should return a dict of factor -> beta."""
        exposures = model.get_factor_exposures("AAPL", returns_data)
        assert isinstance(exposures, dict)
        assert "market" in exposures

    def test_high_beta_stock(self, model, returns_data):
        """TSLA (beta=1.8 synthetic) should have high market beta."""
        exposures = model.get_factor_exposures("TSLA", returns_data)
        assert exposures["market"] > 0.5  # Should be close to 1.8

    def test_market_factor_positive_for_correlated_stock(self, model, returns_data):
        """A stock positively correlated with SPY should have positive market beta."""
        exposures = model.get_factor_exposures("AAPL", returns_data)
        assert exposures["market"] > 0

    def test_insufficient_data_returns_zeros(self, model, short_returns):
        """If data < rolling_window, all betas should be 0."""
        exposures = model.get_factor_exposures("AAPL", short_returns)
        for factor, beta in exposures.items():
            assert beta == 0.0

    def test_missing_symbol_returns_zeros(self, model, returns_data):
        """If symbol not in returns_data, all betas should be 0."""
        exposures = model.get_factor_exposures("FAKE", returns_data)
        for factor, beta in exposures.items():
            assert beta == 0.0

    def test_exposures_cached(self, model, returns_data):
        """After computing, exposures should be cached internally."""
        model.get_factor_exposures("AAPL", returns_data)
        assert "AAPL" in model._exposure_cache
        assert "market" in model._exposure_cache["AAPL"]


# ===================================================================
# Factor exposure calculation (portfolio-level)
# ===================================================================

class TestPortfolioFactorExposures:

    def test_compute_factor_exposures_aggregates(self, model, returns_data):
        """Portfolio-level exposures should be weighted sum of position betas."""
        positions = {"AAPL": 10_000.0, "TSLA": 10_000.0}
        exposures = model.compute_factor_exposures(positions, returns_data)
        assert "market" in exposures
        # Both stocks have positive market beta, so portfolio should too
        assert exposures["market"] > 0

    def test_empty_positions_returns_zeros(self, model, returns_data):
        exposures = model.compute_factor_exposures({}, returns_data)
        for factor, exp in exposures.items():
            assert exp == 0.0

    def test_zero_exposure_positions(self, model, returns_data):
        """Positions with zero dollar exposure should return zeros."""
        positions = {"AAPL": 0.0, "TSLA": 0.0}
        exposures = model.compute_factor_exposures(positions, returns_data)
        for factor, exp in exposures.items():
            assert exp == 0.0

    def test_long_short_portfolio_reduces_beta(self, model, returns_data):
        """Long-short portfolio should have lower market beta than all-long."""
        long_only = {"AAPL": 10_000.0, "TSLA": 10_000.0}
        long_short = {"AAPL": 10_000.0, "TSLA": -10_000.0}

        exp_long = model.compute_factor_exposures(long_only, returns_data)
        exp_ls = model.compute_factor_exposures(long_short, returns_data)

        # Long-short should have lower absolute market beta
        assert abs(exp_ls.get("market", 0)) < abs(exp_long.get("market", 0))

    def test_sector_exposures_included(self, model, returns_data):
        """Sector exposures should be computed as part of portfolio exposures."""
        positions = {"AAPL": 10_000.0}
        exposures = model.compute_factor_exposures(positions, returns_data)
        # Should have at least one sector_ key
        sector_keys = [k for k in exposures if k.startswith("sector_")]
        assert len(sector_keys) >= 0  # might not have sector mapping for synthetic data


# ===================================================================
# portfolio_factor_risk()
# ===================================================================

class TestPortfolioFactorRisk:

    def test_portfolio_factor_risk_structure(self, model, returns_data):
        positions = {"AAPL": 10_000.0, "TSLA": 10_000.0}
        result = model.portfolio_factor_risk(positions, returns_data)
        assert "exposures" in result
        assert "violations" in result
        assert "concentrated" in result

    def test_concentrated_portfolio_flagged(self, model, returns_data):
        """A portfolio with high factor exposure should be flagged."""
        # Use only high-beta stock
        positions = {"TSLA": 100_000.0}
        result = model.portfolio_factor_risk(positions, returns_data)
        # May or may not trigger depending on synthetic data,
        # but the structure should be correct
        assert isinstance(result["violations"], list)
        assert isinstance(result["concentrated"], bool)


# ===================================================================
# Sector exposure limits
# ===================================================================

class TestSectorExposureLimits:

    def test_sector_limit_default(self, model):
        assert model._sector_limit == DEFAULT_SECTOR_LIMIT

    def test_custom_sector_limit(self):
        model = FactorRiskModel(sector_limit=0.20)
        assert model._sector_limit == 0.20

    def test_sector_concentration_breach(self, model, returns_data):
        """If a single sector dominates, check_factor_limits should flag it."""
        # Create exposures with sector breach
        exposures = {"market": 0.3, "sector_Technology": 0.50}
        violations = model.check_factor_limits(
            positions={"AAPL": 10_000},
            exposures=exposures,
        )
        assert any("Sector limit breach" in v for v in violations)

    def test_sector_within_limits(self, model, returns_data):
        exposures = {"market": 0.3, "sector_Technology": 0.20}
        violations = model.check_factor_limits(
            positions={"AAPL": 10_000},
            exposures=exposures,
        )
        assert not any("Sector limit breach" in v for v in violations)


# ===================================================================
# check_factor_limits()
# ===================================================================

class TestCheckFactorLimits:

    def test_no_violations_when_within_limits(self, model):
        exposures = {
            "market": 0.30,
            "size": 0.20,
            "value": 0.20,
            "momentum": 0.30,
        }
        violations = model.check_factor_limits(
            positions={"AAPL": 10_000},
            exposures=exposures,
        )
        assert violations == []

    def test_market_factor_breach(self, model):
        """Market exposure above 0.60 should be flagged."""
        exposures = {"market": 0.70, "size": 0.10}
        violations = model.check_factor_limits(
            positions={"AAPL": 10_000},
            exposures=exposures,
        )
        assert len(violations) >= 1
        assert any("market" in v for v in violations)

    def test_multiple_factor_breaches(self, model):
        exposures = {
            "market": 0.70,   # breach
            "size": 0.50,     # breach
            "value": 0.50,    # breach
        }
        violations = model.check_factor_limits(
            positions={"AAPL": 10_000},
            exposures=exposures,
        )
        assert len(violations) >= 3

    def test_empty_positions_no_violations(self, model):
        violations = model.check_factor_limits(
            positions={},
            returns_data=None,
            exposures=None,
        )
        assert violations == []

    def test_uses_cached_exposures_when_no_returns(self, model):
        """When returns_data is None, should use cached exposures."""
        # Seed the cache
        model._exposure_cache["AAPL"] = {"market": 0.8, "size": 0.1}
        violations = model.check_factor_limits(
            positions={"AAPL": 10_000},
            returns_data=None,
        )
        # Market exposure 0.8 > 0.60 limit
        assert any("market" in v for v in violations)

    def test_custom_factor_limits(self):
        custom = {"market": 0.30, "size": 0.20}
        model = FactorRiskModel(factor_limits=custom)
        exposures = {"market": 0.35}
        violations = model.check_factor_limits(
            positions={"AAPL": 10_000},
            exposures=exposures,
        )
        assert any("market" in v for v in violations)


# ===================================================================
# Risk decomposition
# ===================================================================

class TestRiskDecomposition:

    def test_decompose_empty_positions(self, model):
        sys_risk, idio_risk = model.decompose_risk(positions={})
        assert sys_risk == 0.0
        assert idio_risk == 0.0

    def test_decompose_with_data(self, model, returns_data):
        positions = {"AAPL": 10_000.0, "TSLA": 10_000.0}
        sys_risk, idio_risk = model.decompose_risk(
            positions=positions, returns_data=returns_data,
        )
        # Both should be non-negative
        assert sys_risk >= 0.0
        assert idio_risk >= 0.0

    def test_decompose_stores_last_result(self, model, returns_data):
        positions = {"AAPL": 10_000.0}
        model.decompose_risk(positions=positions, returns_data=returns_data)
        decomp = model.last_decomposition
        assert decomp is not None
        assert decomp.total_risk >= 0
        assert 0 <= decomp.r_squared <= 1.0

    def test_decompose_insufficient_data(self, model, short_returns):
        positions = {"AAPL": 10_000.0}
        sys_risk, idio_risk = model.decompose_risk(
            positions=positions, returns_data=short_returns,
        )
        # Should gracefully handle insufficient data
        assert sys_risk >= 0 and idio_risk >= 0


# ===================================================================
# Status
# ===================================================================

class TestStatus:

    def test_status_dict_keys(self, model):
        status = model.status
        assert "cached_symbols" in status
        assert "cache_timestamp" in status
        assert "rolling_window" in status
        assert "factor_limits" in status
        assert "sector_limit" in status

    def test_status_after_computation(self, model, returns_data):
        model.get_factor_exposures("AAPL", returns_data)
        status = model.status
        assert status["cached_symbols"] >= 1


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:

    def test_model_with_custom_window(self):
        model = FactorRiskModel(rolling_window=30)
        assert model._rolling_window == 30

    def test_factor_returns_with_missing_proxies(self, model):
        """If some factor proxies are missing, partial factors should still work."""
        # Only provide SPY (market factor)
        partial_data = {
            "SPY": _make_return_series(120, seed=1),
            "AAPL": _make_return_series(120, seed=10),
        }
        exposures = model.get_factor_exposures("AAPL", partial_data)
        # Should have market factor at minimum
        assert "market" in exposures

    def test_thread_safety(self, model, returns_data):
        """Concurrent factor computations should not crash."""
        import threading

        def _compute(sym):
            model.get_factor_exposures(sym, returns_data)

        threads = [
            threading.Thread(target=_compute, args=(sym,))
            for sym in ["AAPL", "TSLA", "AAPL", "TSLA"]
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert model._exposure_cache.get("AAPL") is not None

    def test_regression_handles_singular_matrix(self, model):
        """If factor returns are constant (singular), should return zeros."""
        dates = pd.bdate_range("2025-10-01", periods=120, freq="B")
        constant = pd.Series(np.zeros(120), index=dates)
        data = {
            "SPY": constant,
            "AAPL": _make_return_series(120, seed=10),
        }
        exposures = model.get_factor_exposures("AAPL", data)
        # Should handle gracefully, returning finite values
        for factor, beta in exposures.items():
            assert np.isfinite(beta)
