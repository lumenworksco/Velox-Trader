"""Kalman Pairs Trader — 25% of capital allocation.

Trades cointegrated pairs within sector groups using a Kalman filter
for dynamic hedge ratio estimation. Dollar-neutral positioning.

Target: 0.3-0.5% per trade, high win rate (70%+).
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config
from data import get_intraday_bars, get_daily_bars
from strategies.base import Signal
from database import (
    save_kalman_pair, get_active_kalman_pairs,
    deactivate_kalman_pair, deactivate_all_kalman_pairs,
)

logger = logging.getLogger(__name__)


class KalmanPairsTrader:
    """Trade cointegrated pairs with Kalman-estimated hedge ratios.

    Workflow:
    1. select_pairs_weekly() — test cointegration within sector groups, select top pairs
    2. scan() every 2 min — update Kalman filter, compute spread z-score, enter at |z| > 2.0
    3. check_exits() every cycle — convergence at |z| < 0.2, stop at |z| > 3.0, max hold 10 days

    CRITICAL: Kalman state persists across scans (theta, P matrices per pair).
    """

    def __init__(self):
        self.active_pairs: list[dict] = []  # [{symbol1, symbol2, hedge_ratio, ...}]
        self.kalman_state: dict[str, dict] = {}  # "SYM1_SYM2" -> {theta, P, spread_mean, spread_std}
        self._pairs_ready = False
        # V12 2.6: Track consecutive data-fetch failures per pair for halt detection
        self._pair_fetch_failures: dict[str, int] = {}  # pair_id -> consecutive failure count
        self._HALT_FAILURE_THRESHOLD = 3  # Close pair after this many consecutive failures

    def reset_daily(self):
        """Don't clear pairs — they persist for up to a week."""
        pass  # Only reset on weekly pair selection

    def select_pairs_weekly(self, now: datetime) -> list[dict]:
        """Select cointegrated pairs from sector groups.

        Called weekly (Sunday night/Monday morning).
        Tests all possible pairs within each sector group.

        For each pair:
        1. Get 60 days of daily close data
        2. Test cointegration (Engle-Granger: OLS residuals -> ADF test)
        3. Check correlation > PAIRS_MIN_CORRELATION (0.80)
        4. Check cointegration p-value < PAIRS_COINT_PVALUE (0.05)
        5. Initialize Kalman filter state
        6. Save to database
        """
        deactivate_all_kalman_pairs()
        self.active_pairs = []
        self.kalman_state = {}

        all_candidates = []

        # MED-039: Collect all pairs to test, then run cointegration tests in parallel
        all_pairs_to_test: list[tuple[str, str, str]] = []  # (sym1, sym2, group_name)
        for group_name, members in config.SECTOR_GROUPS.items():
            if group_name == 'etf_pairs':
                pairs_to_test = members
            else:
                pairs_to_test = [
                    (members[i], members[j])
                    for i in range(len(members))
                    for j in range(i + 1, len(members))
                ]
            for sym1, sym2 in pairs_to_test:
                all_pairs_to_test.append((sym1, sym2, group_name))

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _test_one(args: tuple[str, str, str]) -> dict | None:
            sym1, sym2, group = args
            try:
                return self._test_pair(sym1, sym2, group)
            except Exception as e:
                logger.debug(f"Pair test error {sym1}/{sym2}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_test_one, p): p for p in all_pairs_to_test}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_candidates.append(result)

        # Sort by cointegration quality (low p-value)
        all_candidates.sort(key=lambda c: c['coint_pvalue'])
        top_pairs = all_candidates[:config.PAIRS_MAX_ACTIVE]

        for pair in top_pairs:
            # BUG-014: Explicitly cast hedge_ratio to scalar float before
            # initializing Kalman state to prevent dimension mismatch
            hedge_ratio = float(pair['hedge_ratio'])
            assert np.isfinite(hedge_ratio), (
                f"Non-finite hedge_ratio for {pair['symbol1']}/{pair['symbol2']}: {hedge_ratio}"
            )

            # IMPL-009: Fit OU parameters via MLE for each selected pair
            spread = pair.get('_spread')
            ou_params = {}
            if spread is not None and len(spread) >= 10:
                ou_params = self._fit_ou_parameters(spread)
                logger.info(
                    f"Pair {pair['symbol1']}/{pair['symbol2']} OU calibration: "
                    f"theta={ou_params.get('theta', 0):.4f}, "
                    f"mu={ou_params.get('mu', 0):.4f}, "
                    f"sigma={ou_params.get('sigma', 0):.4f}, "
                    f"half_life={ou_params.get('half_life', 0):.1f}d"
                )

            # Initialize Kalman state, using OU-calibrated spread stats if available
            pair_key = f"{pair['symbol1']}_{pair['symbol2']}"
            self.kalman_state[pair_key] = {
                'theta': np.array([hedge_ratio, 0.0]),  # [hedge_ratio, intercept]
                'P': np.eye(2) * 1.0,  # Covariance matrix
                'spread_mean': ou_params.get('mu', pair['spread_mean']),
                'spread_std': ou_params.get('sigma', pair['spread_std']) if ou_params.get('sigma', 0) > 0 else pair['spread_std'],
                'ou_params': ou_params,  # Store full calibration for reference
            }

            # Use OU half-life if available, otherwise estimate from autocorrelation
            half_life = ou_params.get('half_life', 0.0)
            if half_life <= 0 or half_life > 100:
                half_life = self._estimate_half_life(spread)

            # Store calibration results in the pair dict for downstream use
            pair['ou_params'] = ou_params
            pair['half_life'] = half_life

            # Save to DB
            save_kalman_pair(
                pair['symbol1'], pair['symbol2'],
                pair['hedge_ratio'], pair['spread_mean'],
                pair['spread_std'], pair['correlation'],
                pair['coint_pvalue'], half_life,
                pair.get('sector_group', 'unknown'),
            )

            self.active_pairs.append(pair)

        self._pairs_ready = True
        logger.info(f"Pairs selected: {len(self.active_pairs)} from {len(all_candidates)} candidates")
        return self.active_pairs

    @staticmethod
    def _estimate_half_life(spread: np.ndarray | None) -> float:
        """Estimate mean-reversion half-life from spread series."""
        if spread is None or len(spread) < 10:
            return 5.0  # Default
        spread_lag = spread[:-1]
        spread_delta = np.diff(spread)
        if np.std(spread_lag) < 1e-8:
            return 5.0
        beta = np.sum(spread_lag * spread_delta) / np.sum(spread_lag ** 2)
        if beta >= 0:
            return 10.0  # Not mean-reverting, return max
        return max(1.0, -np.log(2) / beta)

    @staticmethod
    def _fit_ou_parameters(spread: np.ndarray) -> dict:
        """Fit Ornstein-Uhlenbeck parameters to a spread series via MLE.

        IMPL-009: Maximum Likelihood Estimation of OU process parameters:
            dS = theta * (mu - S) * dt + sigma * dW

        Parameters:
            theta (kappa): Speed of mean reversion
            mu: Long-run mean of the spread
            sigma: Volatility of the spread

        Also computes derived quantities:
            half_life: -ln(2) / ln(1 - theta*dt)
            equilibrium_variance: sigma^2 / (2 * theta)

        Args:
            spread: Array of spread values (log-prices or raw prices).

        Returns:
            Dict with keys: theta, mu, sigma, half_life, eq_variance, log_likelihood.
        """
        n = len(spread)
        if n < 10:
            return {
                "theta": 0.0, "mu": 0.0, "sigma": 0.0,
                "half_life": 10.0, "eq_variance": 0.0, "log_likelihood": 0.0,
            }

        dt = 1.0  # Daily frequency

        # MLE for OU process (exact discrete-time formulation):
        # S_{t+1} = a + b * S_t + eps, where eps ~ N(0, sigma_eps^2)
        # a = mu * (1 - exp(-theta*dt))
        # b = exp(-theta*dt)
        # sigma_eps = sigma * sqrt((1 - exp(-2*theta*dt)) / (2*theta))

        # OLS regression: S_{t+1} = a + b * S_t
        s_t = spread[:-1]
        s_tp1 = spread[1:]

        n_obs = len(s_t)
        if n_obs < 5:
            return {
                "theta": 0.0, "mu": float(np.mean(spread)), "sigma": float(np.std(spread)),
                "half_life": 10.0, "eq_variance": 0.0, "log_likelihood": 0.0,
            }

        # Fit OLS: s_{t+1} = a + b * s_t
        X = np.column_stack([np.ones(n_obs), s_t])
        params = np.linalg.lstsq(X, s_tp1, rcond=None)[0]
        a_hat = float(params[0])
        b_hat = float(params[1])

        # Residual variance
        residuals = s_tp1 - (a_hat + b_hat * s_t)
        sigma_eps_sq = float(np.var(residuals, ddof=2))

        # Recover OU parameters from discrete-time estimates
        if b_hat <= 0 or b_hat >= 1.0:
            # Not mean-reverting or degenerate
            theta = 0.0
            mu = float(np.mean(spread))
            sigma = float(np.std(spread))
            half_life = 10.0
        else:
            theta = -np.log(b_hat) / dt
            mu = a_hat / (1.0 - b_hat)
            # sigma from sigma_eps: sigma_eps^2 = sigma^2 * (1 - exp(-2*theta*dt)) / (2*theta)
            denom = (1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta)
            if denom > 0:
                sigma = float(np.sqrt(sigma_eps_sq / denom))
            else:
                sigma = float(np.sqrt(sigma_eps_sq))
            half_life = float(np.log(2) / theta) if theta > 0 else 10.0

        # Equilibrium variance
        eq_variance = (sigma ** 2) / (2.0 * theta) if theta > 0 else sigma ** 2

        # Log-likelihood (Gaussian)
        if sigma_eps_sq > 0:
            log_lik = -0.5 * n_obs * (np.log(2 * np.pi * sigma_eps_sq) + 1.0)
        else:
            log_lik = 0.0

        return {
            "theta": round(float(theta), 6),
            "mu": round(float(mu), 6),
            "sigma": round(float(sigma), 6),
            "half_life": round(float(half_life), 2),
            "eq_variance": round(float(eq_variance), 6),
            "log_likelihood": round(float(log_lik), 2),
        }

    @staticmethod
    def _has_corporate_action(close: pd.Series, threshold: float = 0.20) -> bool:
        """BUG-019: Detect large overnight price changes (>20%) suggesting corporate actions."""
        if len(close) < 2:
            return False
        pct_changes = close.pct_change().dropna()
        return bool((pct_changes.abs() > threshold).any())

    def _test_pair(self, sym1: str, sym2: str, sector_group: str = "unknown") -> dict | None:
        """Test if two symbols form a cointegrated pair."""
        bars1 = get_daily_bars(sym1, days=60)
        bars2 = get_daily_bars(sym2, days=60)

        if bars1 is None or bars2 is None:
            return None
        if len(bars1) < 40 or len(bars2) < 40:
            return None

        # Align dates
        close1 = bars1["close"]
        close2 = bars2["close"]

        # BUG-019: Skip pairs with suspected corporate actions (>20% overnight move)
        if self._has_corporate_action(close1) or self._has_corporate_action(close2):
            logger.info(f"Pair {sym1}/{sym2} skipped: suspected corporate action (>20% move)")
            return None

        # Ensure same length (take the shorter)
        min_len = min(len(close1), len(close2))
        close1 = close1.iloc[-min_len:]
        close2 = close2.iloc[-min_len:]

        # 1. Correlation check
        corr = close1.corr(close2)
        if abs(corr) < config.PAIRS_MIN_CORRELATION:
            return None

        # 2. OLS hedge ratio: close1 = beta * close2 + alpha + epsilon
        X = np.column_stack([close2.values, np.ones(len(close2))])
        params = np.linalg.lstsq(X, close1.values, rcond=None)[0]
        # BUG-014: Ensure hedge_ratio is scalar float (lstsq can return array elements)
        hedge_ratio = float(params[0])
        intercept = float(params[1])

        # 3. Compute spread
        spread = close1.values - hedge_ratio * close2.values - intercept
        spread_mean = float(np.mean(spread))
        spread_std = float(np.std(spread))

        if spread_std < 1e-6:
            return None

        # 4. V10: Proper ADF test using statsmodels (replaces hand-rolled approximation)
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(spread, maxlag=None, autolag='AIC')
            t_stat = adf_result[0]
            approx_pvalue = adf_result[1]
        except Exception:
            return None

        if approx_pvalue > config.PAIRS_COINT_PVALUE:
            return None

        return {
            'symbol1': sym1,
            'symbol2': sym2,
            'hedge_ratio': float(hedge_ratio),
            'correlation': float(corr),
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'coint_pvalue': approx_pvalue,
            'sector_group': sector_group,
            '_spread': spread,  # Kept for half-life estimation, not persisted
        }

    def _update_kalman(self, pair_key: str, price1: float, price2: float) -> float:
        """Update Kalman filter and return current spread z-score.

        Kalman filter tracks dynamic hedge ratio:
        observation: price1 = theta[0] * price2 + theta[1] + noise

        Returns z-score of the spread.
        """
        state = self.kalman_state.get(pair_key)
        if not state:
            return 0.0

        theta = state['theta']
        P = state['P']

        # Prediction step
        # State transition: theta doesn't change (random walk model)
        Q = np.eye(2) * config.KALMAN_DELTA  # Process noise
        P = P + Q

        # Observation
        x = np.array([price2, 1.0])  # Observation vector
        y = price1  # Observed value

        # Innovation
        y_hat = x @ theta
        e = y - y_hat  # Spread (innovation)

        # V12 FINAL: Check P condition BEFORE gain computation to prevent K divergence
        initial_variance = 1.0
        if np.linalg.cond(P) > 1e6:
            P = np.eye(2) * initial_variance
            logger.warning("V12 FINAL: P matrix pre-update reset for %s (cond > 1e6)", pair_key)

        # Innovation covariance
        R = config.KALMAN_OBS_NOISE  # Observation noise
        S = x @ P @ x.T + R
        S = max(S, 1e-8)  # V10: prevent division by near-zero

        # Kalman gain
        K = P @ x / S

        # V10: Clamp Kalman gain to prevent divergence
        k_norm = np.linalg.norm(K)
        if k_norm > 1.0:
            K = K / k_norm

        # Update step
        theta = theta + K * e
        # CRIT-010: Soft-clip hedge ratio — V11.3 T5 widened from [0.1, 10.0]
        # The old tight clamp biased spread calculations when true hedge ratios
        # were outside range, creating false high-z signals.
        theta[0] = np.clip(theta[0], 0.01, 100.0)
        P = P - np.outer(K, x) @ P

        # V10: Enforce P symmetry and positive semi-definiteness
        P = (P + P.T) / 2
        eigvals = np.linalg.eigvalsh(P)
        if np.any(eigvals < 0):
            P += np.eye(P.shape[0]) * (abs(min(eigvals)) + 1e-8)

        # V12 2.7: P matrix regularization — if condition number explodes
        # (spread near zero for extended periods), Kalman gain diverges.
        # Reset P to identity * initial_variance to restore numerical stability.
        try:
            cond = np.linalg.cond(P)
            if cond > 1e6:
                logger.warning(
                    f"V12 2.7: P matrix ill-conditioned for {pair_key} "
                    f"(cond={cond:.2e}), resetting to identity"
                )
                P = np.eye(2) * 1.0
        except np.linalg.LinAlgError:
            logger.warning(
                f"V12 2.7: P matrix singular for {pair_key}, resetting to identity"
            )
            P = np.eye(2) * 1.0

        # BUG-014: Assert dimensions after Kalman update to catch corruption early
        assert theta.shape == (2,), (
            f"Kalman theta dimension mismatch for {pair_key}: "
            f"expected (2,), got {theta.shape}"
        )
        assert P.shape == (2, 2), (
            f"Kalman P dimension mismatch for {pair_key}: "
            f"expected (2, 2), got {P.shape}"
        )

        # T1-007: Ensure hedge_ratio is scalar float at every Kalman update boundary
        hedge_ratio = float(np.squeeze(theta[0]))
        assert np.isfinite(hedge_ratio), f"Non-finite hedge_ratio after Kalman update for {pair_key}: {hedge_ratio}"
        theta[0] = hedge_ratio

        # Store updated state
        state['theta'] = theta
        state['P'] = P

        # Update running mean/std of spread
        alpha = 0.02  # Exponential decay for running stats
        state['spread_mean'] = (1 - alpha) * state['spread_mean'] + alpha * e
        state['spread_std'] = max(
            0.001,
            np.sqrt((1 - alpha) * state['spread_std'] ** 2 + alpha * e ** 2)
        )

        self.kalman_state[pair_key] = state

        # Z-score of spread
        zscore = (e - state['spread_mean']) / state['spread_std']
        return float(zscore)

    def scan(self, now: datetime, regime: str = "UNKNOWN") -> list[Signal]:
        """Scan active pairs for entry signals.

        Called every SCAN_INTERVAL_SEC (120s).

        Entry: |z-score| > PAIRS_ZSCORE_ENTRY (2.0)
        Generates TWO linked signals (dollar-neutral pair):
        - If spread too wide (z > 2): short symbol1, long symbol2
        - If spread too narrow (z < -2): long symbol1, short symbol2
        """
        signals = []

        if not self._pairs_ready or not self.active_pairs:
            return signals

        for pair in self.active_pairs:
            sym1, sym2 = pair['symbol1'], pair['symbol2']
            pair_key = f"{sym1}_{sym2}"

            try:
                # Get latest prices (use 2-min bars, last bar)
                lookback = now - timedelta(minutes=10)
                bars1 = get_intraday_bars(sym1, TimeFrame(2, TimeFrameUnit.Minute), start=lookback, end=now)
                bars2 = get_intraday_bars(sym2, TimeFrame(2, TimeFrameUnit.Minute), start=lookback, end=now)

                if bars1 is None or bars2 is None or bars1.empty or bars2.empty:
                    continue

                price1 = bars1["close"].iloc[-1]
                price2 = bars2["close"].iloc[-1]

                # Update Kalman and get z-score
                zscore = self._update_kalman(pair_key, price1, price2)

                state = self.kalman_state.get(pair_key, {})
                # T1-007: Ensure hedge_ratio is scalar float (theta[0] may be ndarray element)
                hedge_ratio = float(np.squeeze(state.get('theta', [1.0, 0.0])[0]))

                # --- Spread too wide (z > entry): short sym1, long sym2
                if zscore > config.PAIRS_ZSCORE_ENTRY:
                    pair_id = f"PAIR_{sym1}_{sym2}_{now.strftime('%H%M')}"

                    # Short leg (sym1)
                    if config.ALLOW_SHORT and sym1 not in config.NO_SHORT_SYMBOLS:
                        signals.append(Signal(
                            symbol=sym1,
                            strategy="KALMAN_PAIRS",
                            side="sell",
                            entry_price=round(price1, 2),
                            take_profit=round(price1 * (1 - config.PAIRS_TP_PCT), 2),
                            stop_loss=round(price1 * (1 + config.PAIRS_SL_PCT), 2),
                            reason=f"Pairs short z={zscore:.2f} vs {sym2}",
                            hold_type="day",
                            pair_id=pair_id,
                        ))

                    # Long leg (sym2)
                    signals.append(Signal(
                        symbol=sym2,
                        strategy="KALMAN_PAIRS",
                        side="buy",
                        entry_price=round(price2, 2),
                        take_profit=round(price2 * (1 + config.PAIRS_TP_PCT), 2),
                        stop_loss=round(price2 * (1 - config.PAIRS_SL_PCT), 2),
                        reason=f"Pairs long z={zscore:.2f} vs {sym1}",
                        hold_type="day",
                        pair_id=pair_id,
                    ))

                # --- Spread too narrow (z < -entry): long sym1, short sym2
                elif zscore < -config.PAIRS_ZSCORE_ENTRY:
                    pair_id = f"PAIR_{sym1}_{sym2}_{now.strftime('%H%M')}"

                    # Long leg (sym1)
                    signals.append(Signal(
                        symbol=sym1,
                        strategy="KALMAN_PAIRS",
                        side="buy",
                        entry_price=round(price1, 2),
                        take_profit=round(price1 * (1 + config.PAIRS_TP_PCT), 2),
                        stop_loss=round(price1 * (1 - config.PAIRS_SL_PCT), 2),
                        reason=f"Pairs long z={zscore:.2f} vs {sym2}",
                        hold_type="day",
                        pair_id=pair_id,
                    ))

                    # Short leg (sym2)
                    if config.ALLOW_SHORT and sym2 not in config.NO_SHORT_SYMBOLS:
                        signals.append(Signal(
                            symbol=sym2,
                            strategy="KALMAN_PAIRS",
                            side="sell",
                            entry_price=round(price2, 2),
                            take_profit=round(price2 * (1 - config.PAIRS_TP_PCT), 2),
                            stop_loss=round(price2 * (1 + config.PAIRS_SL_PCT), 2),
                            reason=f"Pairs short z={zscore:.2f} vs {sym1}",
                            hold_type="day",
                            pair_id=pair_id,
                        ))

            except Exception as e:
                logger.debug(f"Pairs scan error for {sym1}/{sym2}: {e}")

        return signals

    def check_exits(self, open_trades: dict, now: datetime) -> list[dict]:
        """Check pairs trades for exit conditions.

        Returns list of exit actions with pair_id for coordinated exits.

        Exit conditions:
        - Convergence: |z| < PAIRS_ZSCORE_EXIT (0.2)
        - Divergence stop: |z| > PAIRS_ZSCORE_STOP (3.0)
        - Max hold: PAIRS_MAX_HOLD_DAYS (10 days)
        """
        exits = []
        checked_pairs = set()

        for symbol, trade in open_trades.items():
            if trade.strategy != "KALMAN_PAIRS":
                continue
            if not trade.pair_id or trade.pair_id in checked_pairs:
                continue

            checked_pairs.add(trade.pair_id)

            # Find both legs of the pair
            parts = trade.pair_id.split('_')  # PAIR_SYM1_SYM2_HHMM
            if len(parts) < 4:
                continue

            sym1, sym2 = parts[1], parts[2]
            pair_key = f"{sym1}_{sym2}"

            try:
                # Get current prices
                lookback = now - timedelta(minutes=10)
                bars1 = get_intraday_bars(sym1, TimeFrame(2, TimeFrameUnit.Minute), start=lookback, end=now)
                bars2 = get_intraday_bars(sym2, TimeFrame(2, TimeFrameUnit.Minute), start=lookback, end=now)

                if bars1 is None or bars2 is None or bars1.empty or bars2.empty:
                    # V12 2.6: Track consecutive data-fetch failures per pair.
                    # If one leg is halted, we can't get price data. After
                    # _HALT_FAILURE_THRESHOLD consecutive failures, close the
                    # entire pair with market orders to avoid sitting unhedged.
                    self._pair_fetch_failures[trade.pair_id] = (
                        self._pair_fetch_failures.get(trade.pair_id, 0) + 1
                    )
                    consec = self._pair_fetch_failures[trade.pair_id]
                    leg1_ok = bars1 is not None and not bars1.empty
                    leg2_ok = bars2 is not None and not bars2.empty
                    failed_leg = sym1 if not leg1_ok else sym2
                    logger.warning(
                        f"V12 2.6: Pair {trade.pair_id} data fetch failure "
                        f"({consec}/{self._HALT_FAILURE_THRESHOLD}) — "
                        f"{failed_leg} data unavailable (possible halt)"
                    )
                    if consec >= self._HALT_FAILURE_THRESHOLD:
                        logger.warning(
                            f"V12 2.6: EMERGENCY CLOSE pair {trade.pair_id} — "
                            f"{failed_leg} halted for {consec} consecutive cycles"
                        )
                        # Close ALL legs of this pair
                        for s, t in open_trades.items():
                            if t.pair_id == trade.pair_id:
                                exits.append({
                                    "symbol": s,
                                    "action": "full",
                                    "reason": f"Pairs leg halted ({failed_leg} data unavail {consec}x)",
                                    "pair_id": trade.pair_id,
                                })
                        self._pair_fetch_failures.pop(trade.pair_id, None)
                    continue

                # V12 2.6: Reset failure counter on successful data fetch
                self._pair_fetch_failures.pop(trade.pair_id, None)

                price1 = bars1["close"].iloc[-1]
                price2 = bars2["close"].iloc[-1]

                zscore = self._update_kalman(pair_key, price1, price2)

                # Max hold check
                if hasattr(trade, 'entry_time') and trade.entry_time:
                    days_held = (now - trade.entry_time).total_seconds() / 86400
                    if days_held > config.PAIRS_MAX_HOLD_DAYS:
                        exits.append({
                            "symbol": symbol,
                            "action": "full",
                            "reason": f"Pairs max hold ({days_held:.1f}d)",
                            "pair_id": trade.pair_id,
                        })
                        continue

                # Convergence exit
                if abs(zscore) < config.PAIRS_ZSCORE_EXIT:
                    exits.append({
                        "symbol": symbol,
                        "action": "full",
                        "reason": f"Pairs converged z={zscore:.2f}",
                        "pair_id": trade.pair_id,
                    })

                # Divergence stop
                elif abs(zscore) > config.PAIRS_ZSCORE_STOP:
                    exits.append({
                        "symbol": symbol,
                        "action": "full",
                        "reason": f"Pairs diverged z={zscore:.2f}",
                        "pair_id": trade.pair_id,
                    })

            except Exception as e:
                logger.debug(f"Pairs exit check error for {pair_key}: {e}")

        return exits
