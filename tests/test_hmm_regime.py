"""Tests for HMM regime detection (analytics/hmm_regime.py + strategies/regime.py)."""

import pickle
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from analytics.hmm_regime import (
    HMMRegimeDetector,
    MarketRegimeState,
    STRATEGY_REGIME_AFFINITY,
    _compute_features,
    _label_states,
    get_regime_size_multiplier,
    get_strategy_regime_affinity,
    map_hmm_to_legacy,
)


# ===================================================================
# Feature computation
# ===================================================================

def _make_spy_data(n_days=100, seed=42):
    """Generate synthetic SPY daily OHLCV data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    close = 400.0 + np.cumsum(rng.normal(0.1, 2.0, n_days))
    high = close + rng.uniform(0, 3, n_days)
    low = close - rng.uniform(0, 3, n_days)
    open_ = close + rng.normal(0, 1, n_days)
    volume = rng.integers(50_000_000, 150_000_000, n_days)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


class TestComputeFeatures:
    def test_output_shape(self):
        df = _make_spy_data(100)
        features = _compute_features(df)
        # First ~20 rows are NaN due to rolling windows
        assert features.shape[1] == 5
        assert features.shape[0] > 50  # Should have 70-80 valid rows

    def test_no_nans_in_output(self):
        df = _make_spy_data(200)
        features = _compute_features(df)
        assert not np.isnan(features).any()

    def test_insufficient_data(self):
        df = _make_spy_data(10)
        features = _compute_features(df)
        # Very few valid rows with 10 days of data
        assert features.shape[0] <= 10

    def test_features_reasonable_range(self):
        df = _make_spy_data(500)
        # Use raw features for range checks (normalized will be z-scored)
        from analytics.hmm_regime import _compute_raw_features
        raw = _compute_raw_features(df)
        # Daily returns should be small
        assert np.abs(raw[:, 0]).max() < 0.5
        # Vol should be positive
        assert (raw[:, 1] >= 0).all()
        assert (raw[:, 2] >= 0).all()
        # Volume ratio should be positive
        assert (raw[:, 3] > 0).all()
        # Normalized features should have mean ~0
        features = _compute_features(df)
        assert np.abs(features.mean(axis=0)).max() < 0.1


# ===================================================================
# State labeling
# ===================================================================

class TestLabelStates:
    def test_all_five_states_assigned(self):
        df = _make_spy_data(500)
        features = _compute_features(df)

        from hmmlearn.hmm import GaussianHMM
        model = GaussianHMM(n_components=5, covariance_type="full", n_iter=50, random_state=42)
        model.fit(features)

        label_map = _label_states(model, features)
        assert len(label_map) == 5
        # All 5 regime states should be represented
        assigned_regimes = set(label_map.values())
        assert len(assigned_regimes) == 5

    def test_fewer_states_fallback(self):
        df = _make_spy_data(300)
        features = _compute_features(df)

        from hmmlearn.hmm import GaussianHMM
        model = GaussianHMM(n_components=3, covariance_type="full", n_iter=50, random_state=42)
        model.fit(features)

        label_map = _label_states(model, features)
        assert len(label_map) == 3


# ===================================================================
# HMMRegimeDetector
# ===================================================================

class TestHMMRegimeDetector:
    def test_init_defaults(self):
        with patch("analytics.hmm_regime.MODEL_PATH") as mock_path:
            mock_path.exists.return_value = False
            detector = HMMRegimeDetector(n_states=5)
        assert detector.n_states == 5
        assert not detector.is_fitted
        assert detector.current_regime == MarketRegimeState.MEAN_REVERTING

    def test_fit_with_sufficient_data(self):
        df = _make_spy_data(500)
        with patch("analytics.hmm_regime.MODEL_PATH") as mock_path:
            mock_path.exists.return_value = False
            mock_path.parent.mkdir = MagicMock()
            detector = HMMRegimeDetector(n_states=5)
            with patch("builtins.open", MagicMock()):
                result = detector.fit(df)
        assert result is True
        assert detector.is_fitted

    def test_fit_with_insufficient_data(self):
        df = _make_spy_data(20)
        with patch("analytics.hmm_regime.MODEL_PATH") as mock_path:
            mock_path.exists.return_value = False
            detector = HMMRegimeDetector(n_states=5)
        result = detector.fit(df)
        assert result is False
        assert not detector.is_fitted

    def test_predict_regime_returns_valid_state(self):
        df = _make_spy_data(500)
        with patch("analytics.hmm_regime.MODEL_PATH") as mock_path:
            mock_path.exists.return_value = False
            mock_path.parent.mkdir = MagicMock()
            detector = HMMRegimeDetector(n_states=5)
            with patch("builtins.open", MagicMock()):
                detector.fit(df)

        # Predict using recent subset
        recent = df.iloc[-30:]
        regime, probs = detector.predict_regime(recent)

        assert isinstance(regime, MarketRegimeState)
        assert len(probs) > 0
        assert abs(sum(probs.values()) - 1.0) < 0.01  # Probabilities sum to ~1

    def test_predict_regime_unfitted_returns_defaults(self):
        with patch("analytics.hmm_regime.MODEL_PATH") as mock_path:
            mock_path.exists.return_value = False
            detector = HMMRegimeDetector(n_states=5)

        recent = _make_spy_data(30)
        regime, probs = detector.predict_regime(recent)

        assert regime == MarketRegimeState.MEAN_REVERTING
        assert len(probs) == 5

    def test_get_transition_matrix(self):
        df = _make_spy_data(500)
        with patch("analytics.hmm_regime.MODEL_PATH") as mock_path:
            mock_path.exists.return_value = False
            mock_path.parent.mkdir = MagicMock()
            detector = HMMRegimeDetector(n_states=5)
            with patch("builtins.open", MagicMock()):
                detector.fit(df)

        matrix = detector.get_transition_matrix()
        assert matrix is not None
        assert matrix.shape == (5, 5)
        # Each row sums to 1
        for row in matrix:
            assert abs(sum(row) - 1.0) < 0.01

    def test_transition_matrix_unfitted_returns_none(self):
        with patch("analytics.hmm_regime.MODEL_PATH") as mock_path:
            mock_path.exists.return_value = False
            detector = HMMRegimeDetector(n_states=5)
        assert detector.get_transition_matrix() is None


# ===================================================================
# Regime size multiplier
# ===================================================================

class TestRegimeSizeMultiplier:
    def test_known_strategy_known_regime(self):
        mult = get_regime_size_multiplier("STAT_MR", MarketRegimeState.MEAN_REVERTING)
        assert mult == 1.5  # V11.4: Updated from 1.3 to 1.5

    def test_unknown_strategy_returns_one(self):
        mult = get_regime_size_multiplier("UNKNOWN_STRAT", MarketRegimeState.LOW_VOL_BULL)
        assert mult == 1.0

    def test_with_probabilities_weighted(self):
        probs = {
            MarketRegimeState.LOW_VOL_BULL: 0.5,
            MarketRegimeState.MEAN_REVERTING: 0.5,
        }
        mult = get_regime_size_multiplier("STAT_MR", MarketRegimeState.LOW_VOL_BULL, probs)
        # V11.4: 0.5 * 0.8 + 0.5 * 1.5 = 1.15
        assert abs(mult - 1.15) < 0.01

    def test_clamped_to_range(self):
        # Even with extreme probabilities, output is clamped
        probs = {MarketRegimeState.HIGH_VOL_BEAR: 1.0}
        mult = get_regime_size_multiplier("STAT_MR", MarketRegimeState.HIGH_VOL_BEAR, probs)
        assert 0.0 <= mult <= 1.5


# ===================================================================
# Legacy mapping
# ===================================================================

class TestMapHmmToLegacy:
    def test_bull_regimes(self):
        assert map_hmm_to_legacy(MarketRegimeState.LOW_VOL_BULL) == "BULLISH"
        assert map_hmm_to_legacy(MarketRegimeState.HIGH_VOL_BULL) == "BULLISH"

    def test_bear_regimes(self):
        assert map_hmm_to_legacy(MarketRegimeState.LOW_VOL_BEAR) == "BEARISH"
        assert map_hmm_to_legacy(MarketRegimeState.HIGH_VOL_BEAR) == "BEARISH"

    def test_mean_reverting(self):
        assert map_hmm_to_legacy(MarketRegimeState.MEAN_REVERTING) == "UNKNOWN"


# ===================================================================
# Strategy regime affinity
# ===================================================================

class TestStrategyRegimeAffinity:
    def test_all_strategies_present(self):
        affinity = get_strategy_regime_affinity()
        for strat in ["STAT_MR", "VWAP", "KALMAN_PAIRS", "ORB", "MICRO_MOM", "PEAD"]:
            assert strat in affinity

    def test_all_regimes_covered_per_strategy(self):
        affinity = get_strategy_regime_affinity()
        for strat, regime_map in affinity.items():
            for regime in MarketRegimeState:
                assert regime in regime_map, f"{strat} missing {regime}"


# ===================================================================
# MarketRegime wrapper (strategies/regime.py)
# ===================================================================

class TestMarketRegimeWrapper:
    @patch("strategies.regime.get_daily_bars")
    def test_update_ema_fallback(self, mock_bars):
        """When HMM is disabled, falls back to EMA."""
        df = _make_spy_data(30)
        mock_bars.return_value = df

        with patch("strategies.regime.config") as mock_config:
            mock_config.HMM_REGIME_ENABLED = False
            mock_config.REGIME_CHECK_INTERVAL_MIN = 0  # Always update
            mock_config.REGIME_EMA_PERIOD = 20

            from strategies.regime import MarketRegime
            regime = MarketRegime()
            result = regime.update(datetime(2026, 3, 15, 10, 0))

            assert result in ("BULLISH", "BEARISH")

    @patch("strategies.regime.get_daily_bars")
    def test_hmm_updates_probabilities(self, mock_bars):
        """When HMM succeeds, hmm_regime and hmm_probabilities are set."""
        df = _make_spy_data(500)
        mock_bars.return_value = df

        from strategies.regime import MarketRegime
        regime = MarketRegime()

        # Manually set up HMM detector
        from analytics.hmm_regime import HMMRegimeDetector
        with patch("analytics.hmm_regime.MODEL_PATH") as mock_path:
            mock_path.exists.return_value = False
            mock_path.parent.mkdir = MagicMock()
            detector = HMMRegimeDetector(n_states=5)
            with patch("builtins.open", MagicMock()):
                detector.fit(df)

        with patch("strategies.regime._get_hmm_detector", return_value=detector):
            with patch("strategies.regime.config") as mock_config:
                mock_config.HMM_REGIME_ENABLED = True
                mock_config.REGIME_CHECK_INTERVAL_MIN = 0
                mock_config.HMM_RETRAIN_DAY = "sunday"
                mock_config.HMM_TRAINING_YEARS = 3
                mock_config.HMM_MIN_PROBABILITY = 0.4

                result = regime.update(datetime(2026, 3, 15, 10, 0))

                assert result in ("BULLISH", "BEARISH", "UNKNOWN")
                assert regime.hmm_regime is not None
                assert len(regime.hmm_probabilities) > 0

    def test_get_regime_affinity_no_hmm(self):
        """When HMM is not set, regime affinity returns 1.0."""
        from strategies.regime import MarketRegime
        regime = MarketRegime()
        assert regime.get_regime_affinity("STAT_MR") == 1.0

    def test_get_regime_affinity_with_hmm(self):
        """When HMM regime is set, returns proper affinity."""
        from strategies.regime import MarketRegime
        regime = MarketRegime()
        regime.hmm_regime = MarketRegimeState.MEAN_REVERTING
        regime.hmm_probabilities = {MarketRegimeState.MEAN_REVERTING: 1.0}

        affinity = regime.get_regime_affinity("STAT_MR")
        assert affinity == 1.5  # V11.4: STAT_MR in MEAN_REVERTING = 1.5


# ===================================================================
# MarketRegimeState enum
# ===================================================================

class TestMarketRegimeState:
    def test_enum_values(self):
        assert MarketRegimeState.LOW_VOL_BULL.value == "low_vol_bull"
        assert MarketRegimeState.HIGH_VOL_BEAR.value == "high_vol_bear"
        assert MarketRegimeState.MEAN_REVERTING.value == "mean_reverting"

    def test_five_states(self):
        assert len(MarketRegimeState) == 5
