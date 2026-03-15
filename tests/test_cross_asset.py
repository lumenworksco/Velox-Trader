"""Tests for CrossAssetMonitor — cross-asset signal layer."""

import time as _time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helper: build a fake yfinance-style DataFrame
# ---------------------------------------------------------------------------

def _make_hist(closes: list[float], highs: list[float] | None = None,
               lows: list[float] | None = None, days: int | None = None):
    """Return a DataFrame mimicking yfinance daily output."""
    n = len(closes)
    if highs is None:
        highs = [c * 1.01 for c in closes]
    if lows is None:
        lows = [c * 0.99 for c in closes]
    dates = pd.date_range(end=datetime(2026, 3, 13), periods=n, freq="D")
    return pd.DataFrame({
        "Open": closes,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": [1_000_000] * n,
    }, index=dates)


def _build_cache(vix=20.0, vix3m=22.0, tlt=90.0, hyg=75.0,
                 uup=28.0, gld=190.0, tlt_closes=None, hyg_closes=None,
                 uup_closes=None, gld_closes=None, vix_closes=None):
    """Build a _cache dict for CrossAssetMonitor with controlled prices."""
    def _entry(key, price, closes=None):
        if closes is None:
            closes = [price * 0.99] * 19 + [price]
        return {"hist": _make_hist(closes), "price": price}

    cache = {
        "vix": _entry("vix", vix, vix_closes),
        "vix3m": _entry("vix3m", vix3m),
        "tlt": _entry("tlt", tlt, tlt_closes),
        "hyg": _entry("hyg", hyg, hyg_closes),
        "uup": _entry("uup", uup, uup_closes),
        "gld": _entry("gld", gld, gld_closes),
    }
    return cache


# ===================================================================
# Tests
# ===================================================================


class TestComputeSignalsKeys:
    """compute_signals() must return all expected keys."""

    def test_returns_all_keys(self):
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        m._cache = _build_cache()
        m._last_update = _time.time()
        signals = m.compute_signals()
        assert "risk_appetite" in signals
        assert "vix_term_structure" in signals
        assert "dollar_regime" in signals
        assert "credit_stress" in signals
        assert "flight_to_safety" in signals

    def test_returns_exactly_five_keys(self):
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        m._cache = _build_cache()
        signals = m.compute_signals()
        assert len(signals) == 5


class TestRiskAppetite:
    """risk_appetite must be in [-1, +1]."""

    def test_range_normal(self):
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        m._cache = _build_cache()
        sig = m.compute_signals()
        assert -1.0 <= sig["risk_appetite"] <= 1.0

    def test_risk_on_scenario(self):
        """Low VIX, falling TLT, rising HYG, falling USD => positive."""
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        # TLT falling over 5 days
        tlt_closes = [95.0] * 15 + [95.0, 94.5, 94.0, 93.0, 90.0]
        # HYG rising
        hyg_closes = [72.0] * 15 + [72.0, 73.0, 74.0, 74.5, 76.0]
        # UUP falling
        uup_closes = [29.0] * 15 + [29.0, 28.8, 28.5, 28.2, 27.5]
        m._cache = _build_cache(
            vix=12.0, tlt=90.0, hyg=76.0, uup=27.5,
            tlt_closes=tlt_closes, hyg_closes=hyg_closes, uup_closes=uup_closes,
        )
        sig = m.compute_signals()
        assert sig["risk_appetite"] > 0.0

    def test_risk_off_scenario(self):
        """High VIX, rising TLT, falling HYG, rising USD => negative."""
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        tlt_closes = [85.0] * 15 + [85.0, 87.0, 89.0, 91.0, 95.0]
        hyg_closes = [78.0] * 15 + [78.0, 77.0, 76.0, 75.0, 73.0]
        uup_closes = [27.0] * 15 + [27.0, 27.5, 28.0, 28.5, 29.5]
        m._cache = _build_cache(
            vix=35.0, tlt=95.0, hyg=73.0, uup=29.5,
            tlt_closes=tlt_closes, hyg_closes=hyg_closes, uup_closes=uup_closes,
        )
        sig = m.compute_signals()
        assert sig["risk_appetite"] < 0.0


class TestCreditStress:
    """credit_stress must be in [0, 1]."""

    def test_range(self):
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        m._cache = _build_cache()
        sig = m.compute_signals()
        assert 0.0 <= sig["credit_stress"] <= 1.0

    def test_no_drawdown_is_zero(self):
        """If HYG is at its 20-day high, credit_stress should be 0."""
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        # Monotonically rising HYG
        hyg_closes = list(np.linspace(70, 80, 20))
        m._cache = _build_cache(hyg=80.0, hyg_closes=hyg_closes)
        sig = m.compute_signals()
        assert sig["credit_stress"] == pytest.approx(0.0, abs=0.01)


class TestVixTermStructure:
    """vix_term_structure must be 'contango' or 'backwardation'."""

    def test_contango(self):
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        m._cache = _build_cache(vix=18.0, vix3m=22.0)
        sig = m.compute_signals()
        assert sig["vix_term_structure"] == "contango"

    def test_backwardation(self):
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        m._cache = _build_cache(vix=30.0, vix3m=22.0)
        sig = m.compute_signals()
        assert sig["vix_term_structure"] == "backwardation"


class TestDollarRegime:
    """dollar_regime must be one of three valid strings."""

    def test_strengthening(self):
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        uup_closes = [27.0] * 15 + [27.0, 27.2, 27.5, 27.8, 28.5]
        m._cache = _build_cache(uup=28.5, uup_closes=uup_closes)
        sig = m.compute_signals()
        assert sig["dollar_regime"] == "strengthening"

    def test_weakening(self):
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        uup_closes = [29.0] * 15 + [29.0, 28.5, 28.0, 27.5, 27.0]
        m._cache = _build_cache(uup=27.0, uup_closes=uup_closes)
        sig = m.compute_signals()
        assert sig["dollar_regime"] == "weakening"

    def test_neutral(self):
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        # Flat UUP
        uup_closes = [28.0] * 20
        m._cache = _build_cache(uup=28.0, uup_closes=uup_closes)
        sig = m.compute_signals()
        assert sig["dollar_regime"] == "neutral"


class TestFlightToSafety:
    """flight_to_safety detection."""

    def test_flight_detected(self):
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        # TLT up > 1%, GLD up > 0.5%, VIX up > 10% in one day
        tlt_closes = [90.0] * 19 + [92.0]   # +2.2%
        gld_closes = [190.0] * 19 + [192.0]  # +1.05%
        vix_closes = [20.0] * 19 + [23.0]    # +15%
        m._cache = _build_cache(
            tlt=92.0, gld=192.0, vix=23.0,
            tlt_closes=tlt_closes, gld_closes=gld_closes, vix_closes=vix_closes,
        )
        sig = m.compute_signals()
        assert sig["flight_to_safety"] is True

    def test_no_flight_normal_day(self):
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        # Small moves — no flight
        tlt_closes = [90.0] * 19 + [90.2]
        gld_closes = [190.0] * 19 + [190.3]
        vix_closes = [20.0] * 19 + [20.5]
        m._cache = _build_cache(
            tlt=90.2, gld=190.3, vix=20.5,
            tlt_closes=tlt_closes, gld_closes=gld_closes, vix_closes=vix_closes,
        )
        sig = m.compute_signals()
        assert sig["flight_to_safety"] is False


class TestEquityBias:
    """get_equity_bias must return -1.0 to +1.0."""

    def test_range(self):
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        m._cache = _build_cache()
        bias = m.get_equity_bias()
        assert -1.0 <= bias <= 1.0

    def test_neutral_when_empty(self):
        """Empty cache => neutral bias."""
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        bias = m.get_equity_bias()
        # Neutral signals → contango contributes +0.2, everything else 0
        # so bias should be small positive
        assert -1.0 <= bias <= 1.0

    def test_sizing_multiplier_range(self):
        """0.5 + (bias * 0.5) must be between 0.0 and 1.0."""
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        m._cache = _build_cache()
        bias = m.get_equity_bias()
        mult = 0.5 + (bias * 0.5)
        assert 0.0 <= mult <= 1.0


class TestFailOpen:
    """All methods must return safe defaults when data is missing."""

    def test_compute_signals_empty_cache(self):
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        sig = m.compute_signals()
        assert sig["risk_appetite"] == 0.0
        assert sig["vix_term_structure"] == "contango"
        assert sig["dollar_regime"] == "neutral"
        assert sig["credit_stress"] == 0.0
        assert sig["flight_to_safety"] is False

    def test_partial_cache_no_crash(self):
        """Only some instruments cached — should not crash."""
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        m._cache = {"vix": {"hist": _make_hist([20.0] * 20), "price": 20.0}}
        sig = m.compute_signals()
        assert -1.0 <= sig["risk_appetite"] <= 1.0

    def test_get_equity_bias_empty_cache(self):
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        bias = m.get_equity_bias()
        assert isinstance(bias, float)
        assert -1.0 <= bias <= 1.0


class TestCaching:
    """Cache should prevent re-fetching within the update interval."""

    @patch("analytics.cross_asset.config")
    def test_no_refetch_within_interval(self, mock_config):
        mock_config.CROSS_ASSET_UPDATE_INTERVAL = 900
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()

        # Populate cache and set last_update to now
        m._cache = _build_cache()
        m._last_update = _time.time()

        with patch("analytics.cross_asset.yf", create=True) as mock_yf:
            m.update(datetime.now())
            # yfinance should NOT be called — cache is fresh
            mock_yf.download.assert_not_called()

    def test_refetch_after_interval_expires(self):
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        m._cache = _build_cache()
        # Simulate cache expired 20 min ago
        m._last_update = _time.time() - 1200

        mock_yf = MagicMock()
        mock_yf.download.return_value = _make_hist([100.0] * 20)
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            m.update(datetime.now())
            assert mock_yf.download.call_count > 0


class TestUpdateFetchErrors:
    """update() should swallow errors (fail-open)."""

    def test_yfinance_import_error(self):
        """If yfinance is not installed, update() logs and returns."""
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        with patch.dict("sys.modules", {"yfinance": None}):
            # Should not raise
            m.update(datetime.now())
        # Cache should remain empty
        assert m._cache == {}

    def test_download_exception(self):
        from analytics.cross_asset import CrossAssetMonitor
        m = CrossAssetMonitor()
        m._last_update = 0.0
        mock_yf = MagicMock()
        mock_yf.download.side_effect = Exception("Network error")
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            m.update(datetime.now())
        # Should not crash; cache still has whatever was there before


class TestConfigValues:
    """Verify the new config constants exist."""

    def test_cross_asset_enabled(self):
        import config
        assert hasattr(config, "CROSS_ASSET_ENABLED")
        assert config.CROSS_ASSET_ENABLED is True

    def test_update_interval(self):
        import config
        assert config.CROSS_ASSET_UPDATE_INTERVAL == 900

    def test_flight_reduction(self):
        import config
        assert config.CROSS_ASSET_FLIGHT_REDUCTION == 0.30

    def test_credit_stress_threshold(self):
        import config
        assert config.CROSS_ASSET_CREDIT_STRESS_THRESHOLD == 0.70
