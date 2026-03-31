"""Hidden Markov Model regime detection — probabilistic market state classification.

Replaces the simple EMA20 slope regime detector with a 5-state HMM that
identifies distinct market regimes using daily returns, realized volatility,
volume, and VIX. Detects regime transitions BEFORE they become obvious.

Features fed to HMM (computed daily from SPY):
  - Daily return (close-to-close)
  - 5-day realized volatility (fast vol)
  - 20-day realized volatility (slow vol)
  - Volume ratio (today / 20-day average)
  - VIX level

Training: Fit on 3 years of daily SPY data. Retrain weekly on Sunday.
Model persisted to models/hmm_regime.pkl.
"""

import logging
import threading
import joblib
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# Model persistence path
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hmm_regime.pkl"


class MarketRegimeState(Enum):
    """Five distinct market regimes identified by the HMM."""
    LOW_VOL_BULL = "low_vol_bull"        # Trending up, low volatility — best for momentum/breakouts
    HIGH_VOL_BULL = "high_vol_bull"      # Trending up, high volatility — reduce size, widen stops
    LOW_VOL_BEAR = "low_vol_bear"        # Grinding down, low vol — best for mean reversion
    HIGH_VOL_BEAR = "high_vol_bear"      # Crisis/crash — halt most trading
    MEAN_REVERTING = "mean_reverting"    # Range-bound, choppy — best for StatMR/VWAP/Pairs


# V11.4: Strategy-regime affinity multipliers — more aggressive gating.
# Strategies are now effectively disabled (0.0) in their worst regimes,
# and boosted (up to 1.5) in their best regimes.
STRATEGY_REGIME_AFFINITY = {
    "STAT_MR": {
        # Mean-reversion thrives in range-bound, low-vol markets
        MarketRegimeState.LOW_VOL_BULL: 0.8,
        MarketRegimeState.HIGH_VOL_BULL: 0.4,
        MarketRegimeState.LOW_VOL_BEAR: 1.2,
        MarketRegimeState.HIGH_VOL_BEAR: 0.0,     # Don't mean-revert in crashes
        MarketRegimeState.MEAN_REVERTING: 1.5,
    },
    "VWAP": {
        # VWAP bands work best in mean-reverting regimes
        MarketRegimeState.LOW_VOL_BULL: 0.7,
        MarketRegimeState.HIGH_VOL_BULL: 0.3,
        MarketRegimeState.LOW_VOL_BEAR: 1.1,
        MarketRegimeState.HIGH_VOL_BEAR: 0.0,     # Bands break in crashes
        MarketRegimeState.MEAN_REVERTING: 1.4,
    },
    "KALMAN_PAIRS": {
        # Pairs are market-neutral, work in most regimes except extreme vol
        MarketRegimeState.LOW_VOL_BULL: 1.0,
        MarketRegimeState.HIGH_VOL_BULL: 0.7,
        MarketRegimeState.LOW_VOL_BEAR: 1.0,
        MarketRegimeState.HIGH_VOL_BEAR: 0.2,     # Correlations break down
        MarketRegimeState.MEAN_REVERTING: 1.2,
    },
    "ORB": {
        # Breakouts need directional momentum — bullish bias
        MarketRegimeState.LOW_VOL_BULL: 1.4,
        MarketRegimeState.HIGH_VOL_BULL: 1.0,
        MarketRegimeState.LOW_VOL_BEAR: 0.3,
        MarketRegimeState.HIGH_VOL_BEAR: 0.5,     # Can short breakdowns
        MarketRegimeState.MEAN_REVERTING: 0.2,     # Breakouts fail in ranges
    },
    "MICRO_MOM": {
        # Micro-momentum needs volatility for opportunities
        MarketRegimeState.LOW_VOL_BULL: 0.8,
        MarketRegimeState.HIGH_VOL_BULL: 1.5,
        MarketRegimeState.LOW_VOL_BEAR: 0.5,
        MarketRegimeState.HIGH_VOL_BEAR: 1.0,     # Vol creates micro-events
        MarketRegimeState.MEAN_REVERTING: 0.6,
    },
    "PEAD": {
        # Earnings drift is regime-independent but better in calm markets
        MarketRegimeState.LOW_VOL_BULL: 1.3,
        MarketRegimeState.HIGH_VOL_BULL: 0.9,
        MarketRegimeState.LOW_VOL_BEAR: 0.8,
        MarketRegimeState.HIGH_VOL_BEAR: 0.3,     # Everything moves together
        MarketRegimeState.MEAN_REVERTING: 1.0,
    },
}


def _compute_raw_features(df: pd.DataFrame) -> np.ndarray:
    """Compute raw HMM feature matrix from daily OHLCV data (unnormalized).

    Args:
        df: DataFrame with columns [open, high, low, close, volume].

    Returns:
        np.ndarray of shape (n_samples, 5) with columns:
        [daily_return, vol_5d, vol_20d, volume_ratio, vix_proxy]
        NaN rows are already dropped.
    """
    close = df["close"].values.astype(float)
    volume = df["volume"].values.astype(float)

    # Daily returns (log)
    log_returns = np.diff(np.log(np.maximum(close, 1e-8)))

    # MED-040: Vectorized rolling std using cumulative sums (replaces Python loops)
    n = len(log_returns)

    def _rolling_std_vec(arr: np.ndarray, window: int) -> np.ndarray:
        """Vectorized rolling standard deviation via cumulative sums."""
        result = np.full(len(arr), np.nan)
        if len(arr) < window:
            return result
        cs = np.concatenate([[0], np.cumsum(arr)])
        cs2 = np.concatenate([[0], np.cumsum(arr ** 2)])
        win_sum = cs[window:] - cs[:-window]
        win_sum2 = cs2[window:] - cs2[:-window]
        variance = win_sum2 / window - (win_sum / window) ** 2
        variance = np.maximum(variance, 0)
        result[window - 1:] = np.sqrt(variance)
        return result

    vol_5d = _rolling_std_vec(log_returns, 5)
    vol_20d = _rolling_std_vec(log_returns, 20)

    # Volume ratio (today / 20-day average) — vectorized
    vol_shifted = volume[1:]  # Align with returns
    vol_ratio = np.full(n, np.nan)
    if n >= 20:
        vcs = np.concatenate([[0], np.cumsum(vol_shifted)])
        avg_vol_20 = (vcs[20:] - vcs[:-20]) / 20.0
        safe_avg = np.where(avg_vol_20 > 0, avg_vol_20, 1.0)
        vol_ratio[19:] = vol_shifted[19:] / safe_avg

    # VIX proxy: 20-day annualized vol (since we may not have VIX in historical data)
    vix_proxy = vol_20d * np.sqrt(252) * 100  # Convert to VIX-like scale

    # Stack features
    features = np.column_stack([log_returns, vol_5d, vol_20d, vol_ratio, vix_proxy])

    # Drop rows with NaN (first ~20 rows)
    valid_mask = ~np.isnan(features).any(axis=1)
    return features[valid_mask]


def _normalize_features(features: np.ndarray,
                         means: np.ndarray | None = None,
                         stds: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize features. Returns (normalized, means, stds)."""
    if means is None:
        means = features.mean(axis=0)
    if stds is None:
        stds = features.std(axis=0)
    stds_safe = stds.copy()
    stds_safe[stds_safe < 1e-10] = 1.0
    normalized = (features - means) / stds_safe
    return normalized, means, stds


def _compute_features(df: pd.DataFrame) -> np.ndarray:
    """Compute normalized HMM features. Convenience wrapper for backward compat."""
    raw = _compute_raw_features(df)
    if len(raw) == 0:
        return raw
    normalized, _, _ = _normalize_features(raw)
    return normalized


def _label_states(model, features: np.ndarray) -> dict[int, MarketRegimeState]:
    """Map HMM state indices to MarketRegimeState enum based on fitted means.

    Labeling logic:
    - Compute each state's mean return and mean volatility
    - Sort by return (ascending): lowest = bear, highest = bull
    - Within bears, high vol = HIGH_VOL_BEAR, low vol = LOW_VOL_BEAR
    - Within bulls, high vol = HIGH_VOL_BULL, low vol = LOW_VOL_BULL
    - Middle state = MEAN_REVERTING
    """
    n_states = model.n_components
    means = model.means_  # shape (n_states, n_features)

    # Feature columns: [daily_return, vol_5d, vol_20d, volume_ratio, vix_proxy]
    state_return = means[:, 0]  # Mean daily return
    state_vol = means[:, 2]     # Mean 20-day vol

    # Sort states by return
    sorted_by_return = np.argsort(state_return)

    label_map = {}

    if n_states >= 5:
        # Bottom 2 = bears, top 2 = bulls, middle = mean-reverting
        bears = sorted_by_return[:2]
        bulls = sorted_by_return[-2:]
        mid = sorted_by_return[2]

        # Bears: sort by vol
        if state_vol[bears[0]] > state_vol[bears[1]]:
            label_map[bears[0]] = MarketRegimeState.HIGH_VOL_BEAR
            label_map[bears[1]] = MarketRegimeState.LOW_VOL_BEAR
        else:
            label_map[bears[0]] = MarketRegimeState.LOW_VOL_BEAR
            label_map[bears[1]] = MarketRegimeState.HIGH_VOL_BEAR

        # Bulls: sort by vol
        if state_vol[bulls[0]] > state_vol[bulls[1]]:
            label_map[bulls[0]] = MarketRegimeState.HIGH_VOL_BULL
            label_map[bulls[1]] = MarketRegimeState.LOW_VOL_BULL
        else:
            label_map[bulls[0]] = MarketRegimeState.LOW_VOL_BULL
            label_map[bulls[1]] = MarketRegimeState.HIGH_VOL_BULL

        label_map[mid] = MarketRegimeState.MEAN_REVERTING
    else:
        # Fallback for fewer states
        for i in range(n_states):
            if state_return[i] > 0 and state_vol[i] < np.median(state_vol):
                label_map[i] = MarketRegimeState.LOW_VOL_BULL
            elif state_return[i] > 0:
                label_map[i] = MarketRegimeState.HIGH_VOL_BULL
            elif state_return[i] < 0 and state_vol[i] > np.median(state_vol):
                label_map[i] = MarketRegimeState.HIGH_VOL_BEAR
            elif state_return[i] < 0:
                label_map[i] = MarketRegimeState.LOW_VOL_BEAR
            else:
                label_map[i] = MarketRegimeState.MEAN_REVERTING

    return label_map


class HMMRegimeDetector:
    """Hidden Markov Model-based market regime detector.

    Uses GaussianHMM to identify 5 distinct market states from daily
    SPY data. Provides probabilistic regime classification and transition
    matrix for forward-looking regime estimation.
    """

    def __init__(self, n_states: int | None = None):
        self.n_states = n_states or getattr(config, "HMM_N_STATES", 5)
        self.model = None
        self._label_map: dict[int, MarketRegimeState] = {}
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None
        self._current_regime = MarketRegimeState.MEAN_REVERTING
        self._regime_probabilities: dict[MarketRegimeState, float] = {
            r: 0.2 for r in MarketRegimeState
        }
        self._fitted = False
        self._lock = threading.Lock()  # MED-003: protect regime state

        # Try to load saved model
        self._load_model()

    def _load_model(self) -> bool:
        """Load a previously fitted model from disk."""
        try:
            if MODEL_PATH.exists():
                saved = joblib.load(MODEL_PATH)
                self.model = saved["model"]
                self._label_map = saved["label_map"]
                self._feature_means = saved.get("feature_means")
                self._feature_stds = saved.get("feature_stds")
                self._fitted = True
                logger.info("HMM regime model loaded from disk")
                return True
        except Exception as e:
            logger.warning(f"Failed to load HMM model: {e}")
        return False

    def _save_model(self):
        """Save fitted model to disk."""
        try:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                "model": self.model,
                "label_map": self._label_map,
                "feature_means": self._feature_means,
                "feature_stds": self._feature_stds,
            }, MODEL_PATH)
            logger.info(f"HMM regime model saved to {MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Failed to save HMM model: {e}")

    def fit(self, market_data: pd.DataFrame) -> bool:
        """Fit the HMM on historical daily SPY data.

        Args:
            market_data: DataFrame with columns [open, high, low, close, volume].
                Should contain 2+ years of daily data for robust fitting.

        Returns:
            True if fitting succeeded, False otherwise.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.error("hmmlearn not installed — HMM regime detection unavailable")
            return False

        raw_features = _compute_raw_features(market_data)
        if len(raw_features) < 100:
            logger.warning(f"Insufficient data for HMM fitting: {len(raw_features)} samples (need 100+)")
            return False

        try:
            features, means, stds = _normalize_features(raw_features)

            # V10 BUG-017: Train/validation split (80/20) to detect overfitting
            split_idx = int(len(features) * 0.8)
            train_features = features[:split_idx]
            val_features = features[split_idx:]

            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=200,
                random_state=42,
                verbose=False,
            )
            model.fit(train_features)

            # Validate: check that validation score is not drastically worse
            train_score = model.score(train_features)
            val_score = model.score(val_features) if len(val_features) >= 10 else train_score
            score_ratio = val_score / train_score if train_score != 0 else 1.0

            if score_ratio < 0.5:
                logger.warning(
                    f"HMM overfitting detected: train_score={train_score:.1f}, "
                    f"val_score={val_score:.1f} (ratio={score_ratio:.2f})"
                )

            # HIGH-003: Reorder HMM states so state 0 = lowest mean return
            sorted_indices = np.argsort(model.means_[:, 0])  # sort by mean daily return
            model.means_ = model.means_[sorted_indices]
            # Use internal _covars_ to bypass covariance_type validation during reorder
            if hasattr(model, '_covars_'):
                model._covars_ = model._covars_[sorted_indices]
            else:
                model.covars_ = model.covars_[sorted_indices]
            model.startprob_ = model.startprob_[sorted_indices]
            model.transmat_ = model.transmat_[sorted_indices][:, sorted_indices]

            self.model = model
            self._label_map = _label_states(model, features)
            self._feature_means = means
            self._feature_stds = stds
            self._fitted = True
            self._save_model()

            logger.info(
                f"HMM fitted on {len(train_features)} train + {len(val_features)} val samples, "
                f"{self.n_states} states, train_score={train_score:.1f}, val_score={val_score:.1f}"
            )
            return True

        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            return False

    def predict_regime(
        self, recent_data: pd.DataFrame
    ) -> tuple[MarketRegimeState, dict[MarketRegimeState, float]]:
        """Predict current regime from recent market data.

        Args:
            recent_data: DataFrame with columns [open, high, low, close, volume].
                Should contain at least 25 rows (20 for vol computation + 5 buffer).

        Returns:
            Tuple of (most_likely_regime, probability_distribution).
        """
        if not self._fitted or self.model is None:
            return self._current_regime, self._regime_probabilities

        try:
            raw = _compute_raw_features(recent_data)
            if len(raw) == 0:
                return self._current_regime, self._regime_probabilities
            features, _, _ = _normalize_features(raw, self._feature_means, self._feature_stds)
            if len(features) == 0:
                return self._current_regime, self._regime_probabilities

            # Get state probabilities for the last observation
            log_prob, posteriors = self.model.score_samples(features)

            # CRIT-012: Guard against NaN scores from degenerate features
            if np.isnan(log_prob).any() or np.isnan(posteriors).any():
                logger.warning("HMM produced NaN scores — returning fallback regime")
                return self._current_regime, self._regime_probabilities

            last_posteriors = posteriors[-1]  # Probabilities for most recent day

            # Map HMM states to regime labels
            probs: dict[MarketRegimeState, float] = {}
            for state_idx, prob in enumerate(last_posteriors):
                regime = self._label_map.get(state_idx, MarketRegimeState.MEAN_REVERTING)
                probs[regime] = probs.get(regime, 0.0) + prob

            # Determine most likely regime with 10% hysteresis
            best_regime = max(probs, key=lambda r: probs[r])

            # HIGH-003: Only switch regime if new regime probability exceeds
            # the current regime's probability by at least 10%
            current_prob = probs.get(self._current_regime, 0.0)
            best_prob = probs.get(best_regime, 0.0)
            if best_regime != self._current_regime:
                if best_prob < current_prob + 0.10:
                    # Not enough conviction to switch — stay in current regime
                    best_regime = self._current_regime

            # MED-003: atomically update regime state under lock
            with self._lock:
                self._current_regime = best_regime
                self._regime_probabilities = probs

            return best_regime, probs

        except Exception as e:
            logger.error(f"HMM prediction failed: {e}")
            return self._current_regime, self._regime_probabilities

    def retrain_weekly(self, lookback_days: int = 252) -> bool:
        """Retrain the HMM on recent market data (called weekly on Sunday).

        IMPL-008: Fetches the last `lookback_days` of daily SPY data and
        refits the HMM. This keeps the model adapted to current market
        conditions while using enough history for stable estimation.

        Args:
            lookback_days: Number of trading days of history to use for
                retraining. Defaults to 252 (approximately 1 year).

        Returns:
            True if retraining succeeded, False otherwise.
        """
        logger.info(
            f"HMM weekly retrain: fetching {lookback_days} days of SPY data"
        )

        try:
            from data import get_daily_bars
        except ImportError:
            logger.error("Cannot import data module for HMM retrain")
            return False

        try:
            spy_data = get_daily_bars("SPY", days=lookback_days)
            if spy_data is None or len(spy_data) < 100:
                logger.warning(
                    f"Insufficient SPY data for retrain: "
                    f"{len(spy_data) if spy_data is not None else 0} bars "
                    f"(need 100+)"
                )
                return False

            # Normalize column names to lowercase
            spy_data.columns = [c.lower() for c in spy_data.columns]

            old_regime = self._current_regime
            success = self.fit(spy_data)

            if success:
                # Run prediction on recent data to update current regime
                recent = spy_data.tail(30)
                new_regime, probs = self.predict_regime(recent)
                logger.info(
                    f"HMM retrain complete: regime {old_regime.value} -> "
                    f"{new_regime.value}, {len(spy_data)} training samples"
                )
            else:
                logger.warning("HMM retrain: fit() returned False")

            return success

        except Exception as e:
            logger.error(f"HMM weekly retrain failed: {e}", exc_info=True)
            return False

    def get_transition_matrix(self) -> np.ndarray | None:
        """Return the fitted transition probability matrix.

        Returns:
            np.ndarray of shape (n_states, n_states) or None if not fitted.
        """
        if self._fitted and self.model is not None:
            return self.model.transmat_
        return None

    @property
    def current_regime(self) -> MarketRegimeState:
        """Current most-likely regime (thread-safe read)."""
        with self._lock:
            return self._current_regime

    @property
    def regime_probabilities(self) -> dict[MarketRegimeState, float]:
        """Full probability distribution across regimes (thread-safe snapshot)."""
        with self._lock:
            return dict(self._regime_probabilities)

    @property
    def is_fitted(self) -> bool:
        """Whether the HMM has been fitted."""
        return self._fitted


# ---------------------------------------------------------------------------
# T5-002: Intraday Regime States (3-state HMM on 5-minute SPY data)
# ---------------------------------------------------------------------------

class IntradayRegimeState(Enum):
    """Three intraday market regimes identified by the 5-minute HMM."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"


INTRADAY_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hmm_intraday.pkl"


def _compute_intraday_features(df: pd.DataFrame) -> np.ndarray:
    """Compute features for the intraday 5-minute HMM.

    Features (from 5-minute SPY bars):
      - 5-minute return (log)
      - 12-bar rolling volatility (1 hour)
      - Volume ratio (current / 12-bar average)

    Args:
        df: DataFrame with columns [open, high, low, close, volume].

    Returns:
        np.ndarray of shape (n_samples, 3). NaN rows dropped.
    """
    close = df["close"].values.astype(float)
    volume = df["volume"].values.astype(float)

    # 5-minute log returns
    log_returns = np.diff(np.log(np.maximum(close, 1e-8)))
    n = len(log_returns)
    if n < 12:
        return np.array([]).reshape(0, 3)

    # 12-bar rolling std (1 hour of 5-min bars)
    vol_12 = np.full(n, np.nan)
    cs = np.concatenate([[0], np.cumsum(log_returns)])
    cs2 = np.concatenate([[0], np.cumsum(log_returns ** 2)])
    window = 12
    if n >= window:
        win_sum = cs[window:] - cs[:-window]
        win_sum2 = cs2[window:] - cs2[:-window]
        variance = win_sum2 / window - (win_sum / window) ** 2
        variance = np.maximum(variance, 0)
        vol_12[window - 1:] = np.sqrt(variance)

    # Volume ratio
    vol_shifted = volume[1:]
    vol_ratio = np.full(n, np.nan)
    if n >= 12:
        vcs = np.concatenate([[0], np.cumsum(vol_shifted)])
        avg_vol = (vcs[12:] - vcs[:-12]) / 12.0
        safe_avg = np.where(avg_vol > 0, avg_vol, 1.0)
        vol_ratio[11:] = vol_shifted[11:] / safe_avg

    features = np.column_stack([log_returns, vol_12, vol_ratio])
    valid_mask = ~np.isnan(features).any(axis=1)
    return features[valid_mask]


def _label_intraday_states(model) -> dict[int, IntradayRegimeState]:
    """Map 3-state HMM indices to IntradayRegimeState.

    Sort by mean return:
      - Lowest mean return -> TRENDING_DOWN
      - Highest mean return -> TRENDING_UP
      - Middle -> MEAN_REVERTING
    """
    means = model.means_
    sorted_indices = np.argsort(means[:, 0])

    label_map = {}
    if len(sorted_indices) >= 3:
        label_map[sorted_indices[0]] = IntradayRegimeState.TRENDING_DOWN
        label_map[sorted_indices[1]] = IntradayRegimeState.MEAN_REVERTING
        label_map[sorted_indices[2]] = IntradayRegimeState.TRENDING_UP
    else:
        for i in range(len(sorted_indices)):
            if means[sorted_indices[i], 0] > 0:
                label_map[sorted_indices[i]] = IntradayRegimeState.TRENDING_UP
            elif means[sorted_indices[i], 0] < 0:
                label_map[sorted_indices[i]] = IntradayRegimeState.TRENDING_DOWN
            else:
                label_map[sorted_indices[i]] = IntradayRegimeState.MEAN_REVERTING

    return label_map


class IntradayRegimeDetector:
    """T5-002: Intraday HMM regime detector trained on 5-minute SPY data.

    Uses a 3-state GaussianHMM to classify the current intraday regime
    as TRENDING_UP, TRENDING_DOWN, or MEAN_REVERTING. Updates every 5 minutes.
    """

    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.model = None
        self._label_map: dict[int, IntradayRegimeState] = {}
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None
        self._current_regime = IntradayRegimeState.MEAN_REVERTING
        self._regime_probabilities: dict[IntradayRegimeState, float] = {
            r: 1.0 / 3.0 for r in IntradayRegimeState
        }
        self._fitted = False
        self._lock = threading.Lock()
        self._last_update_ts: float = 0.0

        # Try to load saved model
        self._load_model()

    def _load_model(self) -> bool:
        """Load a previously fitted intraday model from disk."""
        try:
            if INTRADAY_MODEL_PATH.exists():
                saved = joblib.load(INTRADAY_MODEL_PATH)
                self.model = saved["model"]
                self._label_map = saved["label_map"]
                self._feature_means = saved.get("feature_means")
                self._feature_stds = saved.get("feature_stds")
                self._fitted = True
                logger.info("Intraday HMM regime model loaded from disk")
                return True
        except Exception as e:
            logger.warning(f"Failed to load intraday HMM model: {e}")
        return False

    def _save_model(self):
        """Save fitted intraday model to disk."""
        try:
            INTRADAY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                "model": self.model,
                "label_map": self._label_map,
                "feature_means": self._feature_means,
                "feature_stds": self._feature_stds,
            }, INTRADAY_MODEL_PATH)
            logger.info(f"Intraday HMM model saved to {INTRADAY_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Failed to save intraday HMM model: {e}")

    def fit(self, intraday_data: pd.DataFrame) -> bool:
        """Fit the intraday HMM on 5-minute SPY data.

        Args:
            intraday_data: DataFrame with columns [open, high, low, close, volume].
                Should contain multiple days of 5-minute bars.

        Returns:
            True if fitting succeeded, False otherwise.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.error("hmmlearn not installed — intraday HMM unavailable")
            return False

        raw_features = _compute_intraday_features(intraday_data)
        if len(raw_features) < 50:
            logger.warning(f"Insufficient intraday data for HMM: {len(raw_features)} (need 50+)")
            return False

        try:
            features, means, stds = _normalize_features(raw_features)

            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=150,
                random_state=42,
                verbose=False,
            )
            model.fit(features)

            # Reorder states by mean return
            sorted_indices = np.argsort(model.means_[:, 0])
            model.means_ = model.means_[sorted_indices]
            if hasattr(model, '_covars_'):
                model._covars_ = model._covars_[sorted_indices]
            else:
                model.covars_ = model.covars_[sorted_indices]
            model.startprob_ = model.startprob_[sorted_indices]
            model.transmat_ = model.transmat_[sorted_indices][:, sorted_indices]

            self.model = model
            self._label_map = _label_intraday_states(model)
            self._feature_means = means
            self._feature_stds = stds
            self._fitted = True
            self._save_model()

            logger.info(f"Intraday HMM fitted on {len(features)} samples, {self.n_states} states")
            return True

        except Exception as e:
            logger.error(f"Intraday HMM fitting failed: {e}")
            return False

    def update_intraday_regime(self, recent_5min_data: pd.DataFrame) -> IntradayRegimeState:
        """Update intraday regime based on recent 5-minute bars.

        Called every 5 minutes. Uses the fitted HMM to classify the
        current intraday state.

        Args:
            recent_5min_data: Recent 5-minute bars (at least 20 bars).

        Returns:
            Current IntradayRegimeState.
        """
        if not getattr(config, "INTRADAY_REGIME_ENABLED", False):
            return self._current_regime

        if not self._fitted or self.model is None:
            return self._current_regime

        try:
            raw = _compute_intraday_features(recent_5min_data)
            if len(raw) == 0:
                return self._current_regime

            features, _, _ = _normalize_features(raw, self._feature_means, self._feature_stds)
            if len(features) == 0:
                return self._current_regime

            log_prob, posteriors = self.model.score_samples(features)

            if np.isnan(log_prob).any() or np.isnan(posteriors).any():
                logger.warning("Intraday HMM produced NaN — returning fallback")
                return self._current_regime

            last_posteriors = posteriors[-1]

            probs: dict[IntradayRegimeState, float] = {}
            for state_idx, prob in enumerate(last_posteriors):
                regime = self._label_map.get(state_idx, IntradayRegimeState.MEAN_REVERTING)
                probs[regime] = probs.get(regime, 0.0) + prob

            best_regime = max(probs, key=lambda r: probs[r])

            with self._lock:
                self._current_regime = best_regime
                self._regime_probabilities = probs
                self._last_update_ts = __import__('time').time()

            return best_regime

        except Exception as e:
            logger.error(f"Intraday HMM update failed: {e}")
            return self._current_regime

    def get_intraday_regime(self) -> str:
        """Get current intraday regime as string.

        Returns:
            One of 'trending_up', 'trending_down', 'mean_reverting'.
        """
        with self._lock:
            return self._current_regime.value

    @property
    def intraday_regime(self) -> IntradayRegimeState:
        """Current intraday regime (thread-safe)."""
        with self._lock:
            return self._current_regime

    @property
    def intraday_probabilities(self) -> dict[IntradayRegimeState, float]:
        """Intraday regime probability distribution (thread-safe snapshot)."""
        with self._lock:
            return dict(self._regime_probabilities)

    @property
    def is_fitted(self) -> bool:
        return self._fitted


def get_strategy_regime_affinity() -> dict[str, dict[MarketRegimeState, float]]:
    """Return strategy-regime affinity mapping."""
    return STRATEGY_REGIME_AFFINITY


def get_regime_size_multiplier(strategy: str, regime: MarketRegimeState,
                                probabilities: dict[MarketRegimeState, float] | None = None) -> float:
    """Compute a position size multiplier based on regime and strategy.

    If probabilities are provided, computes a weighted average across all
    regimes. Otherwise uses the single regime's affinity score.

    Args:
        strategy: Strategy name (e.g., "STAT_MR").
        regime: Current most-likely regime.
        probabilities: Optional probability distribution across regimes.

    Returns:
        Multiplier in range [0.2, 1.5].
    """
    affinity = STRATEGY_REGIME_AFFINITY.get(strategy)
    if affinity is None:
        return 1.0

    if probabilities:
        # Weighted average across all regime probabilities
        multiplier = sum(
            affinity.get(r, 1.0) * p
            for r, p in probabilities.items()
        )
    else:
        multiplier = affinity.get(regime, 1.0)

    # Clamp to reasonable range
    return max(0.2, min(1.5, multiplier))


def map_hmm_to_legacy(regime: MarketRegimeState) -> str:
    """Map HMM MarketRegimeState to legacy string regime for backward compatibility.

    The old regime detector returned "BULLISH", "BEARISH", or "UNKNOWN".
    This maps the 5-state HMM output to those strings so existing code
    that checks `regime == "BEARISH"` continues to work.
    """
    if regime in (MarketRegimeState.LOW_VOL_BULL, MarketRegimeState.HIGH_VOL_BULL):
        return "BULLISH"
    elif regime in (MarketRegimeState.LOW_VOL_BEAR, MarketRegimeState.HIGH_VOL_BEAR):
        return "BEARISH"
    else:
        return "UNKNOWN"
