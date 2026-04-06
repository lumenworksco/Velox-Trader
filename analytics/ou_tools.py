"""Ornstein-Uhlenbeck process tools for mean reversion signal generation."""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fit_ou_params(prices: pd.Series) -> dict:
    """
    Fit Ornstein-Uhlenbeck process to a price series via OLS regression.

    OU process: dX = kappa*(mu - X)dt + sigma*dW

    Uses lagged regression: ΔX = a + b*X_{t-1} + ε
    Then: kappa = -b, mu = a/kappa, sigma = std(residuals)

    Args:
        prices: Price series (daily or intraday)

    Returns:
        dict with keys: kappa, mu, sigma, half_life
        Returns empty dict if fitting fails.
    """
    if len(prices) < 20:
        return {}

    try:
        prices_clean = prices.dropna()
        if len(prices_clean) < 20:
            return {}

        # Lagged regression: ΔX = a + b * X_{t-1}
        delta_x = prices_clean.diff().iloc[1:].values
        x_lag = prices_clean.iloc[:-1].values

        # OLS: delta_x = a + b * x_lag
        X = np.column_stack([np.ones(len(x_lag)), x_lag])
        try:
            params = np.linalg.lstsq(X, delta_x, rcond=None)[0]
        except np.linalg.LinAlgError:
            return {}

        a, b = params

        # OU parameters
        if b >= 0:
            return {}  # Not mean-reverting (kappa must be positive)

        kappa = -b
        mu = a / kappa
        residuals = delta_x - (a + b * x_lag)
        sigma = float(np.std(residuals))

        # CRIT-013: Guard against near-zero sigma causing infinite z-scores
        if kappa <= 0 or sigma < 1e-8:
            return {}

        half_life = np.log(2) / kappa  # In units of bar intervals

        return {
            'kappa': float(kappa),
            'mu': float(mu),
            'sigma': float(sigma),
            'half_life': float(half_life),
        }
    except Exception as e:
        logger.warning(f"OU fitting failed: {e}")
        return {}


def compute_zscore(current_price: float, mu: float, sigma: float) -> float:
    """
    Compute z-score: how many std devs current price is from OU mean.

    Positive z-score = price above mean (potential short)
    Negative z-score = price below mean (potential long)

    V12 FINAL: Clips to [-5, 5] to prevent extreme signals, and guards
    against near-zero sigma producing infinite z-scores.
    """
    if sigma < 1e-8:
        return 0.0
    z = (current_price - mu) / sigma
    return max(-5.0, min(5.0, z))


def compute_zscore_series(prices: pd.Series, mu: float, sigma: float) -> pd.Series:
    """Compute z-score for entire price series."""
    if sigma <= 0:
        return pd.Series(0.0, index=prices.index)
    return (prices - mu) / sigma
