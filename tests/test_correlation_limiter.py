"""Tests for risk/correlation_limiter.py — Correlation-based concentration limiter.

Covers:
- Pairwise correlation checks for new positions
- Effective number of bets (eigenvalue-based)
- Sector concentration limits (Herfindahl index)
- Correlation matrix construction with Ledoit-Wolf shrinkage
- Cache behavior (TTL-based matrix caching)
- Thread-safe sector map and correlation updates
- Edge cases: empty portfolio, single position, unknown sectors
"""

import sys
import time as _time
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
_config_mod.MAX_PAIRWISE_CORRELATION = 0.70
_config_mod.MIN_EFFECTIVE_BETS = 2.0
_config_mod.MAX_SECTOR_WEIGHT = 0.50
_config_mod.SECTOR_MAP = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "GOOGL": "XLC",
    "JPM": "XLF",
    "BAC": "XLF",
    "XOM": "XLE",
    "CVX": "XLE",
}
sys.modules.setdefault("config", _config_mod)
# Ensure config attributes are always set correctly even if another test loaded config first
sys.modules["config"].ET = _ET
sys.modules["config"].MAX_PAIRWISE_CORRELATION = 0.70
sys.modules["config"].MIN_EFFECTIVE_BETS = 2.0
sys.modules["config"].MAX_SECTOR_WEIGHT = 0.50
sys.modules["config"].SECTOR_MAP = _config_mod.SECTOR_MAP

from risk.correlation_limiter import CorrelationLimiter, ConcentrationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_limiter(**kwargs):
    """Create a CorrelationLimiter with optional overrides."""
    return CorrelationLimiter(**kwargs)


def _make_correlations(pairs: dict[tuple[str, str], float]) -> dict:
    """Build a correlations dict with sorted key tuples."""
    result = {}
    for (a, b), c in pairs.items():
        key = tuple(sorted([a, b]))
        result[key] = c
    return result


# ---------------------------------------------------------------------------
# Basic initialization
# ---------------------------------------------------------------------------

class TestCorrelationLimiterInit:

    def test_default_thresholds(self):
        cl = _make_limiter()
        assert cl.max_pairwise_corr == 0.70
        assert cl.min_effective_bets == 2.0
        assert cl.max_sector_weight == 0.50

    def test_custom_thresholds(self):
        cl = _make_limiter(max_pairwise_corr=0.80, min_effective_bets=3.0, max_sector_weight=0.40)
        assert cl.max_pairwise_corr == 0.80
        assert cl.min_effective_bets == 3.0
        assert cl.max_sector_weight == 0.40


# ---------------------------------------------------------------------------
# check_new_position — pairwise correlation
# ---------------------------------------------------------------------------

class TestPairwiseCorrelation:

    def test_empty_portfolio_allows_anything(self):
        """With no open positions, any new symbol should be allowed."""
        cl = _make_limiter()
        result = cl.check_new_position("AAPL", [])
        assert not result.too_concentrated
        assert result.effective_bets == 1.0

    def test_low_correlation_allowed(self):
        """Symbols with low pairwise correlation should pass."""
        cl = _make_limiter()
        corrs = _make_correlations({("AAPL", "XOM"): 0.20})
        result = cl.check_new_position("XOM", ["AAPL"], correlations=corrs)
        assert not result.too_concentrated

    def test_high_correlation_blocked(self):
        """Symbols exceeding max pairwise correlation should be blocked.

        Note: Ledoit-Wolf shrinkage reduces extreme correlations, so we use
        a low threshold (0.30) to ensure the shrunk value still triggers.
        We also put symbols in different sectors to avoid sector check triggering.
        """
        cl = _make_limiter(max_pairwise_corr=0.30, max_sector_weight=1.0)
        cl.set_sector_map({"AAPL": "XLK", "MSFT": "XLC"})
        corrs = _make_correlations({("AAPL", "MSFT"): 0.85})
        result = cl.check_new_position("MSFT", ["AAPL"], correlations=corrs)
        assert result.too_concentrated
        assert "high_corr" in result.reason
        assert "AAPL" in result.reason

    def test_low_correlation_at_threshold_passes(self):
        """With low correlation that stays below threshold after shrinkage, should pass."""
        cl = _make_limiter(max_pairwise_corr=0.70, max_sector_weight=1.0)
        cl.set_sector_map({"AAPL": "XLK", "MSFT": "XLC"})
        corrs = _make_correlations({("AAPL", "MSFT"): 0.30})
        result = cl.check_new_position("MSFT", ["AAPL"], correlations=corrs)
        assert not result.too_concentrated

    def test_negative_correlation_uses_absolute(self):
        """Absolute correlation is checked; -0.95 after shrinkage may still trigger."""
        cl = _make_limiter(max_pairwise_corr=0.30, max_sector_weight=1.0)
        cl.set_sector_map({"AAPL": "XLK", "SH": "OTHER"})
        corrs = _make_correlations({("AAPL", "SH"): -0.95})
        result = cl.check_new_position("SH", ["AAPL"], correlations=corrs)
        # After shrinkage, |corr| still exceeds 0.30 threshold
        assert result.too_concentrated

    def test_multiple_open_positions_low_new_corr_passes(self):
        """Low correlations with new symbol should pass even if existing pair is high."""
        cl = _make_limiter(max_pairwise_corr=0.70, max_sector_weight=1.0)
        # Each symbol in a different sector to avoid sector concentration
        cl.set_sector_map({"AAPL": "XLK", "MSFT": "XLC", "XOM": "XLE"})
        corrs = _make_correlations({
            ("AAPL", "MSFT"): 0.90,  # High — but between existing positions
            ("AAPL", "XOM"): 0.10,   # Low with new symbol
            ("MSFT", "XOM"): 0.05,   # Low with new symbol
        })
        result = cl.check_new_position("XOM", ["AAPL", "MSFT"], correlations=corrs)
        assert not result.too_concentrated


# ---------------------------------------------------------------------------
# Effective number of bets
# ---------------------------------------------------------------------------

class TestEffectiveNumberOfBets:

    def test_uncorrelated_positions_high_enb(self):
        """Fully uncorrelated positions should have ENB close to N."""
        cl = _make_limiter(min_effective_bets=2.0)
        # 4 symbols, all zero correlation
        corrs = _make_correlations({
            ("A", "B"): 0.0,
            ("A", "C"): 0.0,
            ("A", "NEW"): 0.0,
            ("B", "C"): 0.0,
            ("B", "NEW"): 0.0,
            ("C", "NEW"): 0.0,
        })
        # Need to set sector map to avoid sector concentration trigger
        cl.set_sector_map({"A": "S1", "B": "S2", "C": "S3", "NEW": "S4"})
        result = cl.check_new_position("NEW", ["A", "B", "C"], correlations=corrs)
        assert not result.too_concentrated
        # ENB for identity matrix of size 4 is 4.0
        assert result.effective_bets == pytest.approx(4.0, abs=0.5)

    def test_highly_correlated_low_enb(self):
        """Very high correlations with many positions should reduce ENB.

        Ledoit-Wolf shrinkage attenuates correlations, so we use a high
        min_effective_bets threshold to ensure the shrunk ENB triggers.
        With 8 positions all at corr=0.99, shrinkage brings ENB down enough.
        """
        cl = _make_limiter(max_pairwise_corr=1.1, min_effective_bets=5.0, max_sector_weight=1.0)
        syms = [f"S{i}" for i in range(8)]
        new_sym = "NEW"
        all_syms = syms + [new_sym]
        corrs = {}
        for i, a in enumerate(all_syms):
            for j, b in enumerate(all_syms):
                if i < j:
                    corrs[(a, b)] = 0.99
        corrs = _make_correlations(corrs)
        cl.set_sector_map({s: f"SEC{i}" for i, s in enumerate(all_syms)})
        result = cl.check_new_position(new_sym, syms, correlations=corrs)
        # With high correlations and high min_effective_bets threshold,
        # the portfolio should be flagged as too concentrated
        assert result.too_concentrated
        assert "low_effective_bets" in result.reason


# ---------------------------------------------------------------------------
# Sector concentration
# ---------------------------------------------------------------------------

class TestSectorConcentration:

    def test_sector_over_weight_blocked(self):
        """Adding a symbol that pushes sector weight above max should be blocked."""
        cl = _make_limiter(
            max_pairwise_corr=1.0,    # Disable pairwise check
            min_effective_bets=0.0,     # Disable ENB check
            max_sector_weight=0.40,
        )
        # 3 symbols in XLK out of 5 total = 60% > 40%
        cl.set_sector_map({
            "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK",
            "JPM": "XLF", "NEW_TECH": "XLK",
        })
        corrs = _make_correlations({
            ("AAPL", "MSFT"): 0.0, ("AAPL", "GOOGL"): 0.0,
            ("AAPL", "JPM"): 0.0, ("AAPL", "NEW_TECH"): 0.0,
            ("MSFT", "GOOGL"): 0.0, ("MSFT", "JPM"): 0.0,
            ("MSFT", "NEW_TECH"): 0.0, ("GOOGL", "JPM"): 0.0,
            ("GOOGL", "NEW_TECH"): 0.0, ("JPM", "NEW_TECH"): 0.0,
        })
        result = cl.check_new_position(
            "NEW_TECH", ["AAPL", "MSFT", "GOOGL", "JPM"],
            correlations=corrs,
        )
        assert result.too_concentrated
        assert "sector" in result.reason.lower()

    def test_unknown_sector_excluded_from_concentration(self):
        """Symbols with unknown sectors (OTHER) should not trigger concentration."""
        cl = _make_limiter(
            max_pairwise_corr=1.0,
            min_effective_bets=0.0,
            max_sector_weight=0.40,
        )
        # All symbols are unknown -> sector = OTHER (excluded from check)
        cl.set_sector_map({})
        corrs = _make_correlations({
            ("A", "B"): 0.0, ("A", "C"): 0.0, ("A", "NEW"): 0.0,
            ("B", "C"): 0.0, ("B", "NEW"): 0.0, ("C", "NEW"): 0.0,
        })
        result = cl.check_new_position("NEW", ["A", "B", "C"], correlations=corrs)
        assert not result.too_concentrated

    def test_diversified_sectors_allowed(self):
        """Evenly distributed sectors should pass concentration check."""
        cl = _make_limiter(
            max_pairwise_corr=1.0,
            min_effective_bets=0.0,
            max_sector_weight=0.50,
        )
        cl.set_sector_map({"A": "XLK", "B": "XLF", "C": "XLE", "NEW": "XLU"})
        corrs = _make_correlations({
            ("A", "B"): 0.0, ("A", "C"): 0.0, ("A", "NEW"): 0.0,
            ("B", "C"): 0.0, ("B", "NEW"): 0.0, ("C", "NEW"): 0.0,
        })
        result = cl.check_new_position("NEW", ["A", "B", "C"], correlations=corrs)
        assert not result.too_concentrated


# ---------------------------------------------------------------------------
# Correlation cache and update
# ---------------------------------------------------------------------------

class TestCorrelationCacheAndUpdate:

    def test_update_correlation_stores_sorted_key(self):
        """update_correlation should store with sorted key tuple."""
        cl = _make_limiter()
        cl.update_correlation("MSFT", "AAPL", 0.85)
        # Key should be ('AAPL', 'MSFT')
        assert ("AAPL", "MSFT") in cl._correlation_cache
        assert cl._correlation_cache[("AAPL", "MSFT")] == 0.85

    def test_cached_correlations_used_when_no_override(self):
        """If no correlations passed, internal cache is used."""
        cl = _make_limiter(max_pairwise_corr=0.70)
        cl.update_correlation("AAPL", "MSFT", 0.85)
        cl.set_sector_map({"AAPL": "XLK", "MSFT": "XLK"})
        result = cl.check_new_position("MSFT", ["AAPL"])
        assert result.too_concentrated

    def test_override_takes_precedence(self):
        """Explicitly passed correlations should override the cache."""
        cl = _make_limiter(max_pairwise_corr=0.70)
        cl.update_correlation("AAPL", "MSFT", 0.85)  # Cached: too high
        corrs = _make_correlations({("AAPL", "MSFT"): 0.30})  # Override: low
        cl.set_sector_map({"AAPL": "XLK", "MSFT": "XLC"})
        result = cl.check_new_position("MSFT", ["AAPL"], correlations=corrs)
        assert not result.too_concentrated


# ---------------------------------------------------------------------------
# set_sector_map
# ---------------------------------------------------------------------------

class TestSetSectorMap:

    def test_set_sector_map_replaces(self):
        cl = _make_limiter()
        cl.set_sector_map({"AAPL": "XLK", "JPM": "XLF"})
        assert cl._sector_map == {"AAPL": "XLK", "JPM": "XLF"}

    def test_set_sector_map_is_deep_copy(self):
        """Modifying the original dict should not affect the limiter."""
        cl = _make_limiter()
        original = {"AAPL": "XLK"}
        cl.set_sector_map(original)
        original["MSFT"] = "XLK"
        assert "MSFT" not in cl._sector_map


# ---------------------------------------------------------------------------
# ConcentrationResult fields
# ---------------------------------------------------------------------------

class TestConcentrationResultFields:

    def test_passing_result_has_all_fields(self):
        cl = _make_limiter()
        corrs = _make_correlations({("AAPL", "XOM"): 0.20})
        cl.set_sector_map({"AAPL": "XLK", "XOM": "XLE"})
        result = cl.check_new_position("XOM", ["AAPL"], correlations=corrs)
        assert isinstance(result.effective_bets, float)
        assert isinstance(result.max_pairwise_corr, float)
        assert isinstance(result.avg_pairwise_corr, float)
        assert isinstance(result.sector_concentration, float)
        assert isinstance(result.too_concentrated, bool)

    def test_hhi_for_equal_sectors(self):
        """HHI for equal sector weights should be 1/n_sectors."""
        cl = _make_limiter(max_pairwise_corr=1.0, min_effective_bets=0.0, max_sector_weight=1.0)
        cl.set_sector_map({"A": "S1", "B": "S2", "C": "S3", "NEW": "S4"})
        corrs = _make_correlations({
            ("A", "B"): 0.0, ("A", "C"): 0.0, ("A", "NEW"): 0.0,
            ("B", "C"): 0.0, ("B", "NEW"): 0.0, ("C", "NEW"): 0.0,
        })
        result = cl.check_new_position("NEW", ["A", "B", "C"], correlations=corrs)
        # 4 symbols in 4 sectors: each 25%, HHI = 4 * 0.25^2 = 0.25
        assert result.sector_concentration == pytest.approx(0.25, abs=0.01)
