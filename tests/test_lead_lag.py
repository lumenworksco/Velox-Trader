"""Tests for analytics/lead_lag.py — lead-lag detection and Granger causality.

Covers:
- Cross-correlation computation at various lags
- Granger causality F-test
- Information coefficient (rank IC)
- LeadLagAnalyzer.find_leaders()
- LeadLagAnalyzer.compute_lead_lag_matrix()
- Rolling lead-lag correlation
- Predictive signal generation
- Real-time lead-lag signal helpers (T5-013)
- Edge cases: insufficient data, NaN handling, empty inputs
"""

import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from analytics.lead_lag import (
    LeadLagAnalyzer,
    LeadLagResult,
    LeadLagMatrix,
    _cross_correlation,
    _granger_causality_test,
    _information_coefficient,
    _f_survival,
    _erf,
    get_lead_lag_signal,
    get_lead_lag_size_multiplier,
    clear_lead_lag_cache,
    MIN_OBSERVATIONS,
    DEFAULT_MAX_LAG,
    GRANGER_SIGNIFICANCE,
    MIN_CORRELATION,
    MAX_LEADERS,
    _active_biases,
    _bias_lock,
    _sector_move_cache,
    _sector_cache_lock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns(n: int = 200, seed: int = 42) -> pd.Series:
    """Generate a synthetic return series."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2025-01-01", periods=n)
    returns = rng.normal(0.0005, 0.02, size=n)
    return pd.Series(returns, index=dates, name="returns")


def _make_lagged_pair(n: int = 200, lag: int = 2, corr: float = 0.5,
                      seed: int = 42):
    """Generate two series where x leads y by `lag` periods with given correlation."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2025-01-01", periods=n)

    noise_x = rng.normal(0, 0.02, size=n)
    noise_y = rng.normal(0, 0.02, size=n)

    x = noise_x
    y = np.zeros(n)
    for i in range(lag, n):
        y[i] = corr * x[i - lag] + (1 - abs(corr)) * noise_y[i]

    return pd.Series(x, index=dates), pd.Series(y, index=dates)


# ===================================================================
# Cross-correlation tests
# ===================================================================

class TestCrossCorrelation:
    def test_basic_computation(self):
        """Cross-correlation returns a dict of lag -> correlation."""
        x = np.random.RandomState(1).randn(100)
        y = np.random.RandomState(2).randn(100)
        result = _cross_correlation(x, y, max_lag=5)
        assert isinstance(result, dict)
        assert 0 in result
        for lag in range(-5, 6):
            assert lag in result

    def test_positive_lag_meaning(self):
        """Positive lag means x leads y: high correlation at positive lag."""
        n = 300
        rng = np.random.RandomState(10)
        x = rng.randn(n)
        y = np.zeros(n)
        # y is a shifted copy of x (x leads y by 3)
        for i in range(3, n):
            y[i] = 0.8 * x[i - 3] + 0.2 * rng.randn()
        result = _cross_correlation(x, y, max_lag=5)
        # Lag 3 should have the highest absolute correlation among positive lags
        positive_lags = {k: abs(v) for k, v in result.items() if k > 0}
        best_lag = max(positive_lags, key=positive_lags.get)
        assert best_lag == 3

    def test_insufficient_data_returns_empty(self):
        """Too few observations returns empty dict."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        result = _cross_correlation(x, y, max_lag=5)
        assert result == {}

    def test_nan_handling(self):
        """NaN values in correlation get filtered out (not stored)."""
        x = np.ones(100)  # constant -> correlation is NaN
        y = np.random.RandomState(5).randn(100)
        result = _cross_correlation(x, y, max_lag=3)
        # NaN correlations should be excluded
        for v in result.values():
            assert np.isfinite(v)


# ===================================================================
# Granger causality tests
# ===================================================================

class TestGrangerCausality:
    def test_returns_tuple_of_three(self):
        """Returns (f_stat, p_value, best_lag)."""
        x = np.random.RandomState(1).randn(200)
        y = np.random.RandomState(2).randn(200)
        f_stat, p_val, best_lag = _granger_causality_test(x, y, max_lag=3)
        assert isinstance(f_stat, float)
        assert isinstance(p_val, float)
        assert isinstance(best_lag, int)
        assert 0 <= p_val <= 1

    def test_causal_series_significant(self):
        """When x truly Granger-causes y, p-value should be small."""
        n = 500
        rng = np.random.RandomState(42)
        x = rng.randn(n)
        y = np.zeros(n)
        for i in range(2, n):
            y[i] = 0.7 * x[i - 2] + 0.3 * rng.randn()
        f_stat, p_val, best_lag = _granger_causality_test(x, y, max_lag=5)
        assert p_val < 0.05, f"Expected significant p-value, got {p_val}"

    def test_independent_series_not_significant(self):
        """Independent series should generally not be Granger-significant."""
        rng = np.random.RandomState(99)
        x = rng.randn(500)
        y = rng.randn(500)
        f_stat, p_val, _ = _granger_causality_test(x, y, max_lag=3)
        # With independent data, p-value should typically be > 0.05
        # (not guaranteed, but very likely with this seed)
        assert p_val > 0.01

    def test_insufficient_data(self):
        """Too little data returns neutral (p=1.0)."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        f_stat, p_val, best_lag = _granger_causality_test(x, y, max_lag=3)
        assert p_val == 1.0
        assert f_stat == 0.0


# ===================================================================
# F-distribution survival function
# ===================================================================

class TestFSurvival:
    def test_basic_output(self):
        """Returns a float between 0 and 1."""
        p = _f_survival(3.0, 2, 50)
        assert 0 <= p <= 1

    def test_large_f_stat_small_pvalue(self):
        """Very large F-stat should give small p-value."""
        p = _f_survival(100.0, 2, 100)
        assert p < 0.01

    def test_small_f_stat_large_pvalue(self):
        """F-stat near 0 should give large p-value."""
        p = _f_survival(0.01, 2, 100)
        assert p > 0.5


# ===================================================================
# Error function approximation
# ===================================================================

class TestErf:
    def test_erf_zero(self):
        assert abs(_erf(0)) < 1e-6  # Abramowitz-Stegun approx: ~1e-9 at 0

    def test_erf_large_positive(self):
        assert abs(_erf(5.0) - 1.0) < 1e-5

    def test_erf_large_negative(self):
        assert abs(_erf(-5.0) + 1.0) < 1e-5

    def test_erf_symmetry(self):
        assert abs(_erf(1.0) + _erf(-1.0)) < 1e-10


# ===================================================================
# Information coefficient
# ===================================================================

class TestInformationCoefficient:
    def test_perfect_rank_correlation(self):
        signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        fwd_ret = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        ic = _information_coefficient(signal, fwd_ret)
        assert ic > 0.95

    def test_inverse_rank_correlation(self):
        signal = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
        fwd_ret = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        ic = _information_coefficient(signal, fwd_ret)
        assert ic < -0.95

    def test_insufficient_data(self):
        signal = np.array([1.0, 2.0])
        fwd_ret = np.array([1.0, 2.0])
        ic = _information_coefficient(signal, fwd_ret)
        assert ic == 0.0

    def test_length_mismatch(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        fwd_ret = np.array([1.0, 2.0, 3.0])
        ic = _information_coefficient(signal, fwd_ret)
        assert ic == 0.0


# ===================================================================
# LeadLagResult dataclass
# ===================================================================

class TestLeadLagResult:
    def test_summary_positive_correlation(self):
        r = LeadLagResult(
            leader="XLF", follower="SPY", optimal_lag=2,
            correlation=0.35, granger_pvalue=0.02,
            granger_significant=True,
        )
        s = r.summary
        assert "XLF" in s
        assert "SPY" in s
        assert "positively" in s
        assert "Granger significant" in s

    def test_summary_negative_correlation(self):
        r = LeadLagResult(
            leader="VIX", follower="SPY", optimal_lag=1,
            correlation=-0.4, granger_pvalue=0.10,
            granger_significant=False,
        )
        s = r.summary
        assert "negatively" in s
        assert "Granger significant" not in s


# ===================================================================
# LeadLagMatrix
# ===================================================================

class TestLeadLagMatrix:
    def test_top_leaders(self):
        m = LeadLagMatrix(
            symbols=["A", "B", "C"],
            net_leadership_scores={"A": 0.5, "B": -0.3, "C": 0.8},
        )
        leaders = m.get_top_leaders(2)
        assert leaders[0][0] == "C"
        assert leaders[1][0] == "A"

    def test_top_followers(self):
        m = LeadLagMatrix(
            symbols=["A", "B", "C"],
            net_leadership_scores={"A": 0.5, "B": -0.3, "C": 0.8},
        )
        followers = m.get_top_followers(2)
        assert followers[0][0] == "B"


# ===================================================================
# LeadLagAnalyzer — find_leaders
# ===================================================================

class TestFindLeaders:
    def test_finds_leading_asset(self):
        """When x truly leads target, it appears in results."""
        x, target = _make_lagged_pair(n=250, lag=2, corr=0.6, seed=42)
        analyzer = LeadLagAnalyzer(max_lag=5, min_correlation=0.05)
        results = analyzer.find_leaders(
            target=target,
            candidates={"LEADER": x},
            target_name="TARGET",
        )
        assert len(results) > 0
        assert results[0].leader == "LEADER"
        assert results[0].optimal_lag > 0

    def test_insufficient_target_data(self):
        """Returns empty when target has fewer than MIN_OBSERVATIONS points."""
        target = pd.Series([0.01] * 10, index=pd.bdate_range("2025-01-01", periods=10))
        x = _make_returns(n=200)
        analyzer = LeadLagAnalyzer()
        results = analyzer.find_leaders(
            target=target, candidates={"X": x},
        )
        assert results == []

    def test_multiple_candidates_sorted_by_correlation(self):
        """Results are sorted by absolute correlation (descending)."""
        target = _make_returns(n=250, seed=1)
        candidates = {
            f"SYM_{i}": _make_returns(n=250, seed=i + 10)
            for i in range(5)
        }
        analyzer = LeadLagAnalyzer(max_lag=3, min_correlation=0.0)
        results = analyzer.find_leaders(
            target=target, candidates=candidates,
        )
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert abs(results[i].correlation) >= abs(results[i + 1].correlation)

    def test_max_leaders_limit(self):
        """No more than MAX_LEADERS results returned."""
        target = _make_returns(n=300, seed=1)
        candidates = {
            f"SYM_{i}": _make_returns(n=300, seed=i + 100)
            for i in range(30)
        }
        analyzer = LeadLagAnalyzer(max_lag=3, min_correlation=0.0)
        results = analyzer.find_leaders(target=target, candidates=candidates)
        assert len(results) <= MAX_LEADERS

    def test_custom_max_lag(self):
        """Custom max_lag parameter is respected."""
        x, target = _make_lagged_pair(n=250, lag=2, corr=0.5, seed=42)
        analyzer = LeadLagAnalyzer(max_lag=10)
        results = analyzer.find_leaders(
            target=target, candidates={"X": x}, max_lag=3,
        )
        # All lag correlations should be within [-3, 3]
        if results:
            for lag in results[0].all_lag_correlations:
                assert abs(lag) <= 3


# ===================================================================
# LeadLagAnalyzer — compute_lead_lag_matrix
# ===================================================================

class TestComputeLeadLagMatrix:
    def test_matrix_structure(self):
        """Matrix contains all symbol pairs (excluding self)."""
        returns = {
            "A": _make_returns(n=200, seed=1),
            "B": _make_returns(n=200, seed=2),
            "C": _make_returns(n=200, seed=3),
        }
        analyzer = LeadLagAnalyzer(max_lag=3)
        matrix = analyzer.compute_lead_lag_matrix(returns)
        assert set(matrix.symbols) == {"A", "B", "C"}
        assert isinstance(matrix.net_leadership_scores, dict)
        assert len(matrix.net_leadership_scores) == 3

    def test_leadership_scores_sum_to_zero(self):
        """Net leadership scores should approximately sum to zero."""
        returns = {
            "A": _make_returns(n=200, seed=1),
            "B": _make_returns(n=200, seed=2),
        }
        analyzer = LeadLagAnalyzer(max_lag=3, min_correlation=0.0)
        matrix = analyzer.compute_lead_lag_matrix(returns)
        total = sum(matrix.net_leadership_scores.values())
        assert abs(total) < 1e-10


# ===================================================================
# Predictive signal and rolling lead-lag
# ===================================================================

class TestPredictiveSignal:
    def test_signal_shift(self):
        """Predictive signal shifts returns forward by optimal lag."""
        leader = _make_returns(n=100, seed=1)
        analyzer = LeadLagAnalyzer()
        signal = analyzer.compute_predictive_signal(leader, optimal_lag=3, correlation_sign=1.0)
        # Signal should be shorter than original (NaN dropped)
        assert len(signal) <= len(leader) - 3

    def test_negative_correlation_sign(self):
        """Negative correlation_sign inverts the signal."""
        leader = _make_returns(n=100, seed=1)
        analyzer = LeadLagAnalyzer()
        sig_pos = analyzer.compute_predictive_signal(leader, optimal_lag=2, correlation_sign=1.0)
        sig_neg = analyzer.compute_predictive_signal(leader, optimal_lag=2, correlation_sign=-1.0)
        # They should be opposite signs
        common_idx = sig_pos.index.intersection(sig_neg.index)
        if len(common_idx) > 0:
            np.testing.assert_array_almost_equal(
                sig_pos.loc[common_idx].values,
                -sig_neg.loc[common_idx].values,
            )


class TestRollingLeadLag:
    def test_basic_output(self):
        """Returns a Series of rolling correlations."""
        x = _make_returns(n=200, seed=1)
        y = _make_returns(n=200, seed=2)
        result = LeadLagAnalyzer.rolling_lead_lag(x, y, window=30, lag=1)
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_insufficient_data(self):
        """Returns empty Series when data is too short."""
        x = _make_returns(n=10, seed=1)
        y = _make_returns(n=10, seed=2)
        result = LeadLagAnalyzer.rolling_lead_lag(x, y, window=60, lag=1)
        assert len(result) == 0


# ===================================================================
# T5-013: Real-time lead-lag signal helpers
# ===================================================================

class TestGetLeadLagSignal:
    def setup_method(self):
        clear_lead_lag_cache()

    def test_disabled_returns_zero(self):
        """When LEAD_LAG_ENABLED is False, returns 0.0."""
        with patch("analytics.lead_lag.config") as mock_cfg:
            mock_cfg.LEAD_LAG_ENABLED = False
            assert get_lead_lag_signal("AAPL") == 0.0

    def test_active_bias_returned(self):
        """When an active bias exists and hasn't expired, it is returned."""
        with _bias_lock:
            _active_biases["AAPL"] = (0.15, time.time() + 600)
        with patch("analytics.lead_lag.config") as mock_cfg:
            mock_cfg.LEAD_LAG_ENABLED = True
            result = get_lead_lag_signal("AAPL")
        assert result == 0.15

    def test_expired_bias_cleared(self):
        """An expired bias is cleared and not returned."""
        with _bias_lock:
            _active_biases["MSFT"] = (0.10, time.time() - 10)
        with patch("analytics.lead_lag.config") as mock_cfg:
            mock_cfg.LEAD_LAG_ENABLED = True
            mock_cfg.SECTOR_MAP = {}
            result = get_lead_lag_signal("MSFT")
        assert result == 0.0

    def test_no_sector_mapping_returns_zero(self):
        """When symbol has no sector ETF mapping, returns 0.0."""
        with patch("analytics.lead_lag.config") as mock_cfg:
            mock_cfg.LEAD_LAG_ENABLED = True
            mock_cfg.SECTOR_MAP = {}
            result = get_lead_lag_signal("AAPL")
        assert result == 0.0

    def test_never_raises(self):
        """get_lead_lag_signal never raises (fail-open)."""
        with patch("analytics.lead_lag.config", side_effect=Exception("boom")):
            result = get_lead_lag_signal("AAPL")
        assert result == 0.0


class TestGetLeadLagSizeMultiplier:
    def setup_method(self):
        clear_lead_lag_cache()

    def test_no_signal_returns_one(self):
        """When there's no lead-lag signal, multiplier is 1.0."""
        with patch("analytics.lead_lag.get_lead_lag_signal", return_value=0.0):
            assert get_lead_lag_size_multiplier("AAPL") == 1.0

    def test_positive_bias(self):
        """Positive bias increases multiplier."""
        with patch("analytics.lead_lag.get_lead_lag_signal", return_value=0.15):
            m = get_lead_lag_size_multiplier("AAPL")
            assert m == pytest.approx(1.15)

    def test_negative_bias(self):
        """Negative bias decreases multiplier."""
        with patch("analytics.lead_lag.get_lead_lag_signal", return_value=-0.2):
            m = get_lead_lag_size_multiplier("AAPL")
            assert m == pytest.approx(0.8)


class TestClearLeadLagCache:
    def test_clears_both_caches(self):
        with _sector_cache_lock:
            _sector_move_cache["XLF"] = (0.5, time.time())
        with _bias_lock:
            _active_biases["AAPL"] = (0.1, time.time() + 300)
        clear_lead_lag_cache()
        assert len(_sector_move_cache) == 0
        assert len(_active_biases) == 0
