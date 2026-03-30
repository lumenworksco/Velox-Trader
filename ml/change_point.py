"""ADVML-003: Bayesian Online Change-Point Detection (BOCPD).

Implements the Adams & MacKay (2007) algorithm for detecting abrupt
changes in a streaming time series.  Maintains a posterior distribution
over *run lengths* (time since the last change-point) and updates it in
O(1) amortised time per observation via the hazard function and a
conjugate-exponential predictive model.

Typical applications in a trading context:
    - Volatility regime shifts (trigger position sizing adjustment)
    - Correlation breakdown detection (invalidate pairs/clusters)
    - Strategy performance degradation (trigger reallocation)
    - Structural breaks in mean/variance (re-estimate parameters)

Usage:
    detector = BayesianChangePointDetector(hazard_lambda=200)
    for obs in streaming_data:
        cp_prob = detector.update(obs)
        if detector.detect_change(threshold=0.5):
            handle_regime_change()

    # Multi-series:
    monitor = ChangePointMonitor(["volatility", "correlation", "sharpe"])
    results = monitor.update({"volatility": vol_t, "correlation": corr_t, ...})

References:
    - Adams, R. P. & MacKay, D. J. C. (2007).
      "Bayesian Online Changepoint Detection."
      arXiv:0710.3742.
    - Marcos Lopez de Prado, *Advances in Financial Machine Learning*.

Dependencies: numpy (always available).
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conjugate predictive model (Normal-Inverse-Gamma)
# ---------------------------------------------------------------------------


class _NormalInverseGamma:
    """Sufficient statistics for a Normal-Inverse-Gamma conjugate model.

    Tracks vectorised sufficient statistics so that each run-length
    hypothesis has its own posterior predictive.

    The predictive distribution is Student-t, but we only need its
    log-PDF for the BOCPD likelihood term.

    Parameters
    ----------
    prior_mu : float
        Prior mean.
    prior_kappa : float
        Prior precision scaling (number of pseudo-observations for the
        mean).
    prior_alpha : float
        Prior shape for the inverse-gamma on the variance.
    prior_beta : float
        Prior scale for the inverse-gamma on the variance.
    """

    def __init__(
        self,
        prior_mu: float = 0.0,
        prior_kappa: float = 1.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        self.mu0 = prior_mu
        self.kappa0 = prior_kappa
        self.alpha0 = prior_alpha
        self.beta0 = prior_beta

        # Vectorised sufficient stats — one entry per run-length hypothesis.
        # Initialised with a single entry for run-length 0 (new segment).
        self.mu = np.array([prior_mu])
        self.kappa = np.array([prior_kappa])
        self.alpha = np.array([prior_alpha])
        self.beta = np.array([prior_beta])

    def log_predictive(self, x: float) -> np.ndarray:
        """Log-probability of observation *x* under each run-length's
        predictive Student-t distribution.

        Returns an array of length ``len(self.mu)``.
        """
        df = 2.0 * self.alpha
        scale_sq = self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa)
        scale_sq = np.maximum(scale_sq, 1e-12)

        # Student-t log-pdf (vectorised over run-length hypotheses)
        z = (x - self.mu) ** 2 / scale_sq
        log_p = (
            _lgamma_vec(0.5 * (df + 1.0))
            - _lgamma_vec(0.5 * df)
            - 0.5 * np.log(df * math.pi * scale_sq)
            - 0.5 * (df + 1.0) * np.log1p(z / df)
        )
        return log_p

    def update(self, x: float) -> None:
        """Bayesian update: extend sufficient statistics by one observation.

        After calling, ``len(self.mu)`` grows by 1 (the new run-length=0
        entry is prepended by the caller, not here).
        """
        kappa_new = self.kappa + 1.0
        mu_new = (self.kappa * self.mu + x) / kappa_new
        alpha_new = self.alpha + 0.5
        beta_new = (
            self.beta
            + 0.5 * self.kappa * (x - self.mu) ** 2 / kappa_new
        )

        self.mu = mu_new
        self.kappa = kappa_new
        self.alpha = alpha_new
        self.beta = beta_new

    def prepend_prior(self) -> None:
        """Prepend a fresh prior entry (run-length = 0) for the new
        change-point hypothesis."""
        self.mu = np.concatenate([[self.mu0], self.mu])
        self.kappa = np.concatenate([[self.kappa0], self.kappa])
        self.alpha = np.concatenate([[self.alpha0], self.alpha])
        self.beta = np.concatenate([[self.beta0], self.beta])

    def truncate(self, max_len: int) -> None:
        """Keep at most *max_len* run-length hypotheses to bound memory."""
        self.mu = self.mu[:max_len]
        self.kappa = self.kappa[:max_len]
        self.alpha = self.alpha[:max_len]
        self.beta = self.beta[:max_len]


def _lgamma_vec(x: np.ndarray) -> np.ndarray:
    """Vectorised log-gamma, handling both scalars and arrays."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    for i in np.ndindex(x.shape):
        out[i] = math.lgamma(float(x[i]))
    return out


# ---------------------------------------------------------------------------
# BOCPD
# ---------------------------------------------------------------------------


class BOCPD:
    """Bayesian Online Change-Point Detector.

    Parameters
    ----------
    hazard_lambda : float
        Expected run length between change-points.  The hazard function
        ``H(r) = 1 / hazard_lambda`` is constant (geometric prior on
        segment length).  Default 100.
    prior_mu : float
        Prior mean for the Normal-Inverse-Gamma observation model.
    prior_kappa : float
        Prior precision scaling.
    prior_alpha : float
        Prior shape parameter.
    prior_beta : float
        Prior scale parameter.
    max_run_length : int
        Maximum run-length hypothesis to track.  Older hypotheses are
        truncated to bound memory at O(max_run_length).  Default 500.
    """

    def __init__(
        self,
        hazard_lambda: float = 100.0,
        prior_mu: float = 0.0,
        prior_kappa: float = 1.0,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        max_run_length: int = 500,
    ) -> None:
        if hazard_lambda <= 0:
            raise ValueError("hazard_lambda must be positive")

        self.hazard_lambda = hazard_lambda
        self.max_run_length = max_run_length

        self._model = _NormalInverseGamma(
            prior_mu=prior_mu,
            prior_kappa=prior_kappa,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
        )

        # Log joint probabilities: log P(r_t, x_{1:t})
        # Start with a single entry: log P(r_0 = 0) = 0  (certainty)
        self._log_joint = np.array([0.0])

        self._t = 0  # number of observations processed
        self._change_point_prob_history: List[float] = []

    # ----- public API -------------------------------------------------------

    def update(self, obs: float) -> float:
        """Process one observation and return P(change-point at this step).

        Parameters
        ----------
        obs : float
            New scalar observation.

        Returns
        -------
        float
            Posterior probability that a change-point occurred at this
            time step, i.e. ``P(r_t = 0 | x_{1:t})``.
        """
        obs = float(obs)
        if not math.isfinite(obs):
            logger.warning("Non-finite observation (%.4g) — skipping.", obs)
            cp_prob = 0.0
            self._change_point_prob_history.append(cp_prob)
            return cp_prob

        self._t += 1

        # --- step 1: predictive log-likelihoods -----------------------------
        log_pred = self._model.log_predictive(obs)

        # --- step 2: hazard function ----------------------------------------
        log_h = math.log(1.0 / self.hazard_lambda)
        log_1mh = math.log(1.0 - 1.0 / self.hazard_lambda)

        # --- step 3: growth & change-point probabilities --------------------
        # Growth: existing run lengths survive
        log_growth = self._log_joint + log_pred + log_1mh
        # Change-point: all run lengths collapse to 0
        log_cp = _logsumexp(self._log_joint + log_pred + log_h)

        # --- step 4: new joint (prepend change-point hypothesis) ------------
        new_log_joint = np.concatenate([[log_cp], log_growth])

        # --- step 5: normalise to get posterior run-length distribution ------
        log_evidence = _logsumexp(new_log_joint)
        log_posterior = new_log_joint - log_evidence

        # P(change-point) = P(r_t = 0)
        cp_prob = float(np.exp(log_posterior[0]))

        # --- step 6: update sufficient statistics ---------------------------
        self._model.update(obs)
        self._model.prepend_prior()

        # --- step 7: truncate to bound memory -------------------------------
        if len(new_log_joint) > self.max_run_length:
            new_log_joint = new_log_joint[: self.max_run_length]
            self._model.truncate(self.max_run_length)

        self._log_joint = new_log_joint
        self._change_point_prob_history.append(cp_prob)

        if cp_prob > 0.3:
            logger.debug(
                "t=%d  P(cp)=%.4f  (elevated change-point probability)",
                self._t,
                cp_prob,
            )

        return cp_prob

    def detect_change(self, threshold: float = 0.5) -> bool:
        """Return *True* if the most recent change-point probability
        exceeds *threshold*.

        Parameters
        ----------
        threshold : float
            Detection threshold in [0, 1].  Default 0.5.

        Returns
        -------
        bool
        """
        if not self._change_point_prob_history:
            return False
        return self._change_point_prob_history[-1] > threshold

    def get_run_length_posterior(self) -> np.ndarray:
        """Current posterior distribution over run lengths.

        Returns
        -------
        np.ndarray
            1-D array of probabilities; index *i* corresponds to
            run-length *i*.
        """
        log_evidence = _logsumexp(self._log_joint)
        return np.exp(self._log_joint - log_evidence)

    def get_map_run_length(self) -> int:
        """Maximum-a-posteriori run length (excluding r=0)."""
        posterior = self.get_run_length_posterior()
        if len(posterior) <= 1:
            return 0
        # Exclude r=0 for MAP (it spikes at every change-point)
        return int(np.argmax(posterior[1:]) + 1)

    def get_history(self) -> List[float]:
        """Full history of P(change-point) values."""
        return list(self._change_point_prob_history)

    def reset(self) -> None:
        """Reset the detector to its initial state."""
        self._model = _NormalInverseGamma(
            prior_mu=self._model.mu0,
            prior_kappa=self._model.kappa0,
            prior_alpha=self._model.alpha0,
            prior_beta=self._model.beta0,
        )
        self._log_joint = np.array([0.0])
        self._t = 0
        self._change_point_prob_history.clear()

    def process_batch(
        self,
        observations: np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[List[float], List[int]]:
        """Process a batch of observations and return change-point probabilities.

        Parameters
        ----------
        observations : array-like
            1-D array of observations.
        threshold : float
            Detection threshold.

        Returns
        -------
        cp_probs : list of float
            Change-point probability at each step.
        change_points : list of int
            Indices where cp_prob > threshold.
        """
        obs = np.asarray(observations, dtype=np.float64).ravel()
        cp_probs: List[float] = []
        change_points: List[int] = []

        for i, val in enumerate(obs):
            prob = self.update(val)
            cp_probs.append(prob)
            if prob > threshold:
                change_points.append(i)

        logger.info(
            "BOCPD batch: %d observations, %d change points (threshold=%.2f)",
            len(obs), len(change_points), threshold,
        )
        return cp_probs, change_points

    @property
    def n_observations(self) -> int:
        """Number of observations processed so far."""
        return self._t


# ---------------------------------------------------------------------------
# Multi-Series Change-Point Monitor
# ---------------------------------------------------------------------------


class ChangePointMonitor:
    """Convenience wrapper running BOCPD on multiple named series.

    Useful for simultaneously monitoring volatility, correlation,
    and strategy performance streams.

    Parameters
    ----------
    series_names : list of str
        Names of the series to track.
    hazard_lambda : float
        Shared hazard parameter.
    threshold : float
        Default detection threshold.
    """

    def __init__(
        self,
        series_names: List[str],
        hazard_lambda: float = 100.0,
        threshold: float = 0.5,
        **bocpd_kwargs,
    ) -> None:
        self.threshold = threshold
        self._detectors: Dict[str, BOCPD] = {}
        for name in series_names:
            self._detectors[name] = BOCPD(
                hazard_lambda=hazard_lambda, **bocpd_kwargs
            )

    def update(self, observations: Dict[str, float]) -> Dict[str, float]:
        """Update detectors with a dict of ``{series_name: value}``.

        Returns
        -------
        dict
            ``{series_name: P(change-point)}`` for each updated series.
        """
        results: Dict[str, float] = {}
        for name, value in observations.items():
            det = self._detectors.get(name)
            if det is None:
                logger.warning("Unknown series %r — skipping.", name)
                continue
            results[name] = det.update(value)
        return results

    def detect_changes(
        self,
        threshold: Optional[float] = None,
    ) -> Dict[str, bool]:
        """Return ``{series_name: bool}`` for each tracked series."""
        t = threshold if threshold is not None else self.threshold
        return {
            name: det.detect_change(t)
            for name, det in self._detectors.items()
        }

    def summary(self) -> Dict[str, Dict]:
        """Per-series summary with latest cp probability and MAP run length."""
        out: Dict[str, Dict] = {}
        for name, det in self._detectors.items():
            hist = det.get_history()
            out[name] = {
                "n_observations": det.n_observations,
                "latest_cp_prob": hist[-1] if hist else 0.0,
                "map_run_length": det.get_map_run_length(),
                "change_detected": det.detect_change(self.threshold),
            }
        return out


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _logsumexp(log_x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    if len(log_x) == 0:
        return -np.inf
    c = float(np.max(log_x))
    if not math.isfinite(c):
        return -np.inf
    return c + math.log(float(np.sum(np.exp(log_x - c))))


def detect_volatility_regime_change(
    returns: np.ndarray,
    hazard_lambda: float = 100.0,
    threshold: float = 0.5,
) -> Tuple[List[float], List[int]]:
    """Convenience function: run BOCPD on squared returns to detect
    volatility regime shifts.

    Parameters
    ----------
    returns : array-like
        1-D array of returns.
    hazard_lambda : float
        Expected regime duration.
    threshold : float
        Detection threshold.

    Returns
    -------
    cp_probs : list of float
        Change-point probability at each time step.
    change_points : list of int
        Indices where a change-point was detected.
    """
    returns = np.asarray(returns, dtype=np.float64).ravel()
    squared = returns ** 2

    det = BOCPD(hazard_lambda=hazard_lambda)
    cp_probs: List[float] = []
    change_points: List[int] = []

    for i, val in enumerate(squared):
        prob = det.update(val)
        cp_probs.append(prob)
        if prob > threshold:
            change_points.append(i)

    logger.info(
        "Volatility regime detection: %d change-points in %d observations.",
        len(change_points),
        len(returns),
    )
    return cp_probs, change_points


# ---------------------------------------------------------------------------
# COMP-012: O(1) Online BOCPD Extension
# ---------------------------------------------------------------------------


class OnlineBOCPD(BOCPD):
    """BOCPD with O(1) amortised online updates that avoid reprocessing history.

    Extends the base BOCPD with an efficient ``update_online()`` method that
    maintains a compact running state.  Instead of storing the full history
    of run-length distributions, it keeps only the current posterior and a
    fixed-size summary of recent change-point activity.

    Key differences from base ``update()``:
        - Maintains a running exponential moving average of change-point
          probability for smoother regime detection.
        - Keeps a bounded circular buffer of recent cp probabilities.
        - Provides ``get_regime_stability()`` — how long since the last
          detected change-point, without scanning full history.

    Parameters
    ----------
    hazard_lambda : float
        Expected run length between change-points.
    ema_alpha : float
        Smoothing factor for the exponential moving average of change-point
        probability.  Lower values = more smoothing.  Default 0.1.
    buffer_size : int
        Size of the circular buffer for recent cp probabilities.
    **kwargs
        Additional arguments passed to ``BOCPD.__init__``.

    Usage:
        detector = OnlineBOCPD(hazard_lambda=200)
        for obs in streaming_data:
            result = detector.update_online(obs)
            if result["change_detected"]:
                handle_regime_change()
            print(result["ema_cp_prob"], result["steps_since_cp"])
    """

    def __init__(
        self,
        hazard_lambda: float = 100.0,
        ema_alpha: float = 0.1,
        buffer_size: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(hazard_lambda=hazard_lambda, **kwargs)

        self._ema_alpha = ema_alpha
        self._ema_cp_prob = 0.0
        self._buffer_size = buffer_size
        self._cp_buffer = np.zeros(buffer_size, dtype=np.float64)
        self._buffer_idx = 0
        self._buffer_count = 0
        self._steps_since_cp = 0
        self._last_cp_step = -1
        self._total_steps = 0
        self._online_threshold = 0.5

    def update_online(
        self,
        observation: float,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process one observation with O(1) state maintenance.

        This wraps the base ``update()`` and maintains additional running
        state without reprocessing history.

        Parameters
        ----------
        observation : float
            New scalar observation.
        threshold : float, optional
            Detection threshold.  Defaults to 0.5.

        Returns
        -------
        dict
            Keys:
            - ``cp_prob``: raw change-point probability at this step
            - ``ema_cp_prob``: exponentially smoothed cp probability
            - ``change_detected``: bool, whether cp_prob > threshold
            - ``steps_since_cp``: steps since last detected change-point
            - ``total_steps``: total observations processed
            - ``map_run_length``: MAP estimate of current run length
            - ``buffer_mean``: mean cp probability over recent buffer
        """
        if threshold is None:
            threshold = self._online_threshold

        # Core BOCPD update — O(max_run_length), amortised O(1) per step
        cp_prob = self.update(observation)

        # O(1) running state updates
        self._total_steps += 1

        # EMA update
        self._ema_cp_prob = (
            self._ema_alpha * cp_prob
            + (1.0 - self._ema_alpha) * self._ema_cp_prob
        )

        # Circular buffer update
        self._cp_buffer[self._buffer_idx] = cp_prob
        self._buffer_idx = (self._buffer_idx + 1) % self._buffer_size
        self._buffer_count = min(self._buffer_count + 1, self._buffer_size)

        # Change-point tracking
        detected = cp_prob > threshold
        if detected:
            self._steps_since_cp = 0
            self._last_cp_step = self._total_steps
        else:
            self._steps_since_cp += 1

        # Buffer mean (only over filled portion)
        filled = self._cp_buffer[:self._buffer_count]
        buffer_mean = float(filled.mean()) if self._buffer_count > 0 else 0.0

        return {
            "cp_prob": cp_prob,
            "ema_cp_prob": self._ema_cp_prob,
            "change_detected": detected,
            "steps_since_cp": self._steps_since_cp,
            "total_steps": self._total_steps,
            "map_run_length": self.get_map_run_length(),
            "buffer_mean": buffer_mean,
        }

    def get_regime_stability(self) -> float:
        """Estimate regime stability as a score in [0, 1].

        Higher values indicate a more stable regime (longer since last
        change-point, lower recent cp probability).

        Returns
        -------
        float
            Stability score in [0, 1].
        """
        if self._total_steps == 0:
            return 1.0

        # Time decay: more steps since cp = more stable
        time_factor = 1.0 - math.exp(-self._steps_since_cp / self.hazard_lambda)

        # Low recent cp probability = more stable
        prob_factor = 1.0 - min(self._ema_cp_prob * 2.0, 1.0)

        return float(0.6 * time_factor + 0.4 * prob_factor)

    def get_recent_cp_rate(self) -> float:
        """Fraction of recent observations that exceeded default threshold.

        Returns
        -------
        float
            Rate in [0, 1].
        """
        if self._buffer_count == 0:
            return 0.0
        filled = self._cp_buffer[:self._buffer_count]
        return float((filled > self._online_threshold).mean())

    def set_threshold(self, threshold: float) -> None:
        """Update the default online detection threshold."""
        self._online_threshold = max(0.0, min(1.0, threshold))

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Return a compact snapshot of the online state.

        Useful for checkpointing without serialising the full BOCPD
        internal state.
        """
        return {
            "total_steps": self._total_steps,
            "ema_cp_prob": self._ema_cp_prob,
            "steps_since_cp": self._steps_since_cp,
            "last_cp_step": self._last_cp_step,
            "buffer_mean": float(self._cp_buffer[:self._buffer_count].mean())
            if self._buffer_count > 0
            else 0.0,
            "regime_stability": self.get_regime_stability(),
            "recent_cp_rate": self.get_recent_cp_rate(),
            "map_run_length": self.get_map_run_length(),
        }

    def reset_online(self) -> None:
        """Reset both the base BOCPD and the online state."""
        self.reset()
        self._ema_cp_prob = 0.0
        self._cp_buffer[:] = 0.0
        self._buffer_idx = 0
        self._buffer_count = 0
        self._steps_since_cp = 0
        self._last_cp_step = -1
        self._total_steps = 0


# Alias for spec compatibility
BayesianChangePointDetector = BOCPD
