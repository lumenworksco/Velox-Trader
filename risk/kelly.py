"""V8: Kelly Criterion position sizing engine.

Replaces flat RISK_PER_TRADE_PCT with per-strategy Kelly-optimal sizing
based on historical win rate and avg win/loss ratio.

T7-004 addition: Multi-regime Bayesian Kelly sizing using HMM regime
probabilities to compute probability-weighted Kelly fractions.
"""

import logging
from datetime import datetime
from typing import Any

import numpy as np

import config
import database
from utils import safe_divide

logger = logging.getLogger(__name__)


class KellyEngine:
    """Compute half-Kelly fractions per strategy from trade history."""

    def __init__(self):
        self._fractions: dict[str, float] = {}
        self._params: dict[str, dict] = {}
        self._last_computed: datetime | None = None

    def compute_fractions(self):
        """Recalculate Kelly fractions from trade database. Call daily at market open."""
        # MED-015: Validate Kelly risk bounds at computation time
        if config.KELLY_MIN_RISK >= config.KELLY_MAX_RISK:
            logger.error(
                "KELLY_MIN_RISK (%.4f) >= KELLY_MAX_RISK (%.4f) — "
                "using flat RISK_PER_TRADE_PCT for all strategies",
                config.KELLY_MIN_RISK, config.KELLY_MAX_RISK,
            )
            for s in config.STRATEGY_ALLOCATIONS:
                self._fractions[s] = config.RISK_PER_TRADE_PCT
            return

        strategies = list(config.STRATEGY_ALLOCATIONS.keys())

        # MED-041: Single batch query for all trade history, then partition locally
        # to avoid N sequential database roundtrips
        try:
            all_trades = database.get_recent_trades(days=365)
        except Exception as e:
            logger.warning(f"Kelly batch query failed, falling back to per-strategy: {e}")
            all_trades = None

        trades_by_strategy: dict[str, list[dict]] = {}
        if all_trades is not None:
            for t in all_trades:
                strat = t.get("strategy", "")
                if strat in config.STRATEGY_ALLOCATIONS:
                    trades_by_strategy.setdefault(strat, []).append(t)

        for strategy in strategies:
            try:
                if all_trades is not None:
                    trades = trades_by_strategy.get(strategy, [])
                else:
                    trades = database.get_recent_trades_by_strategy(strategy, days=365)
                # Use last KELLY_LOOKBACK trades
                trades = trades[-config.KELLY_LOOKBACK:] if len(trades) > config.KELLY_LOOKBACK else trades

                if len(trades) < config.KELLY_MIN_TRADES:
                    self._fractions[strategy] = config.RISK_PER_TRADE_PCT
                    logger.debug(f"Kelly {strategy}: insufficient trades ({len(trades)}/{config.KELLY_MIN_TRADES}), using flat {config.RISK_PER_TRADE_PCT}")
                    continue

                wins = [t for t in trades if t.get("pnl", 0) > 0]
                losses = [t for t in trades if t.get("pnl", 0) <= 0]

                win_rate = len(wins) / len(trades) if trades else 0
                avg_win = sum(t.get("pnl_pct", 0) for t in wins) / len(wins) if wins else 0
                avg_loss = safe_divide(
                    sum(abs(t.get("pnl_pct", 0)) for t in losses),
                    len(losses),
                    default=1e-6,
                )

                if avg_loss < 1e-8:
                    avg_loss = 1e-6

                win_loss_ratio = safe_divide(avg_win, avg_loss, default=0.0)

                # HIGH-002: Cap extreme values to prevent outsized Kelly fractions
                win_loss_ratio = max(0.1, min(win_loss_ratio, 10.0))
                win_rate = min(win_rate, 0.95)

                # Guard against near-zero win/loss ratio
                if win_loss_ratio < 1e-6:
                    self._fractions[strategy] = config.KELLY_MIN_RISK
                    logger.warning(f"Kelly {strategy}: near-zero win/loss ratio={win_loss_ratio:.6f}, using min risk")
                    continue

                # Kelly fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
                kelly_f = win_rate - safe_divide(1 - win_rate, win_loss_ratio, default=0.0)

                # Half-Kelly for safety
                half_kelly = kelly_f * config.KELLY_FRACTION_MULT

                # Clamp to [KELLY_MIN_RISK, KELLY_MAX_RISK]
                half_kelly = max(config.KELLY_MIN_RISK, min(config.KELLY_MAX_RISK, half_kelly))

                # If Kelly is negative (losing strategy), use minimum
                if kelly_f <= 0:
                    half_kelly = config.KELLY_MIN_RISK
                    logger.warning(f"Kelly {strategy}: negative kelly_f={kelly_f:.4f}, using min risk {config.KELLY_MIN_RISK}")

                self._fractions[strategy] = half_kelly
                self._params[strategy] = {
                    "win_rate": round(win_rate, 4),
                    "avg_win_loss": round(win_loss_ratio, 4),
                    "kelly_f": round(kelly_f, 4),
                    "half_kelly_f": round(half_kelly, 4),
                    "sample_size": len(trades),
                }

                logger.info(f"Kelly {strategy}: wr={win_rate:.2%} w/l={win_loss_ratio:.2f} kelly={kelly_f:.4f} half={half_kelly:.4f} (n={len(trades)})")

                # Save to database
                try:
                    database.save_kelly_params(
                        strategy=strategy,
                        win_rate=win_rate,
                        avg_win_loss=win_loss_ratio,
                        kelly_f=kelly_f,
                        half_kelly_f=half_kelly,
                        sample_size=len(trades),
                    )
                except Exception as e:
                    logger.debug(f"Failed to save kelly params for {strategy}: {e}")

            except Exception as e:
                logger.warning(f"Kelly computation failed for {strategy}: {e}")
                self._fractions[strategy] = config.RISK_PER_TRADE_PCT

        self._last_computed = datetime.now(config.ET)

    def get_fraction(self, strategy: str) -> float:
        """Get the Kelly fraction for a strategy.

        Returns half-Kelly if sufficient data, else falls back to RISK_PER_TRADE_PCT.
        """
        if not config.KELLY_ENABLED:
            return config.RISK_PER_TRADE_PCT

        return self._fractions.get(strategy, config.RISK_PER_TRADE_PCT)

    @property
    def params(self) -> dict[str, dict]:
        """Return computed Kelly parameters for all strategies."""
        return self._params.copy()

    @property
    def last_computed(self) -> datetime | None:
        return self._last_computed

    # ------------------------------------------------------------------
    # T7-004: Multi-Regime Bayesian Kelly Sizing
    # ------------------------------------------------------------------

    def compute_regime_kelly(
        self,
        strategy: str,
        regime_probabilities: dict[int, float] | list[float],
        regime_params: dict[int, dict] | list[dict] | None = None,
    ) -> float:
        """T7-004: Compute Bayesian Kelly fraction weighted by regime probabilities.

        Uses HMM regime probabilities to compute a probability-weighted Kelly
        fraction: E[K] = Sum_i P(regime_i) * K_i, then applies fractional
        Kelly (0.5x) for safety.

        Each regime has its own win rate and win/loss ratio (from historical
        trades partitioned by the regime that was active when they occurred).
        If regime-specific parameters are not available, falls back to the
        standard Kelly fraction scaled by a regime-dependent risk multiplier.

        Args:
            strategy: Strategy name (e.g., "STAT_MR").
            regime_probabilities: Dict or list of regime probabilities.
                Dict: {regime_id: probability}
                List: [p_0, p_1, ..., p_n] where index is regime_id.
            regime_params: Optional per-regime trade statistics.
                Dict: {regime_id: {"win_rate": float, "avg_win_loss": float}}
                List: [{"win_rate": ..., "avg_win_loss": ...}, ...]
                If None, uses default regime risk multipliers.

        Returns:
            Bayesian Kelly fraction, clamped to [KELLY_MIN_RISK, KELLY_MAX_RISK].
        """
        bayesian_enabled = getattr(config, "BAYESIAN_KELLY_ENABLED", False)
        if not bayesian_enabled:
            return self.get_fraction(strategy)

        # Normalize regime_probabilities to a dict
        if isinstance(regime_probabilities, (list, tuple)):
            probs = {i: p for i, p in enumerate(regime_probabilities)}
        else:
            probs = dict(regime_probabilities)

        # Validate probabilities sum to ~1.0
        total_prob = sum(probs.values())
        if total_prob < 1e-6:
            logger.warning("T7-004: Regime probabilities sum to ~0, falling back to standard Kelly")
            return self.get_fraction(strategy)

        # Normalize if needed
        if abs(total_prob - 1.0) > 0.01:
            probs = {k: v / total_prob for k, v in probs.items()}

        # Normalize regime_params to a dict
        if regime_params is not None:
            if isinstance(regime_params, (list, tuple)):
                r_params = {i: p for i, p in enumerate(regime_params)}
            else:
                r_params = dict(regime_params)
        else:
            r_params = None

        # Default regime risk multipliers (used when no regime-specific params)
        # Based on typical HMM regimes: 0=calm, 1=trending, 2=volatile,
        # 3=crisis, 4=recovery
        _DEFAULT_REGIME_MULT = {
            0: 1.0,    # Calm/normal
            1: 1.2,    # Trending (favorable)
            2: 0.6,    # High volatility
            3: 0.2,    # Crisis
            4: 0.8,    # Recovery
        }

        weighted_kelly = 0.0

        for regime_id, prob in probs.items():
            if prob < 1e-8:
                continue

            if r_params and regime_id in r_params:
                # Use regime-specific parameters
                rp = r_params[regime_id]
                win_rate = rp.get("win_rate", 0.5)
                avg_win_loss = rp.get("avg_win_loss", 1.0)

                # Cap extreme values (same as standard Kelly)
                win_rate = min(win_rate, 0.95)
                avg_win_loss = min(avg_win_loss, 10.0)

                if avg_win_loss < 1e-6:
                    k_i = 0.0
                else:
                    # K_i = (p_i * b_i - q_i) / b_i
                    # where p_i = win_rate, b_i = avg_win_loss, q_i = 1 - p_i
                    k_i = win_rate - (1.0 - win_rate) / avg_win_loss
            else:
                # Fall back to standard Kelly scaled by regime multiplier
                base_kelly = self._params.get(strategy, {}).get("kelly_f", 0.0)
                mult = _DEFAULT_REGIME_MULT.get(regime_id, 0.5)
                k_i = base_kelly * mult

            # Clamp individual regime Kelly to avoid extreme negatives
            k_i = max(-0.5, k_i)

            weighted_kelly += prob * k_i

        # If the weighted Kelly is negative (overall losing), use minimum
        if weighted_kelly <= 0:
            logger.info(
                f"T7-004: Regime Kelly for {strategy} is negative "
                f"({weighted_kelly:.4f}), using min risk"
            )
            return config.KELLY_MIN_RISK

        # Apply fractional Kelly (0.5x for safety)
        fraction_mult = getattr(config, "KELLY_FRACTION_MULT", 0.5)
        half_kelly = weighted_kelly * fraction_mult

        # Clamp to bounds
        result = max(config.KELLY_MIN_RISK, min(config.KELLY_MAX_RISK, half_kelly))

        logger.info(
            f"T7-004: Regime Kelly {strategy}: weighted_kelly={weighted_kelly:.4f}, "
            f"half_kelly={half_kelly:.4f}, clamped={result:.4f}, "
            f"regimes={len(probs)}, dominant={max(probs, key=probs.get)}"
        )

        return result

    def compute_all_regime_kelly(
        self,
        regime_probabilities: dict[int, float] | list[float],
        regime_params_by_strategy: dict[str, dict[int, dict]] | None = None,
    ) -> dict[str, float]:
        """T7-004: Compute regime Kelly fractions for all active strategies.

        Args:
            regime_probabilities: Current HMM regime probabilities.
            regime_params_by_strategy: Optional per-strategy, per-regime params.
                {strategy: {regime_id: {"win_rate": float, "avg_win_loss": float}}}

        Returns:
            Dict of strategy -> regime-weighted Kelly fraction.
        """
        result = {}
        strategies = list(config.STRATEGY_ALLOCATIONS.keys())

        for strategy in strategies:
            r_params = None
            if regime_params_by_strategy and strategy in regime_params_by_strategy:
                r_params = regime_params_by_strategy[strategy]

            result[strategy] = self.compute_regime_kelly(
                strategy=strategy,
                regime_probabilities=regime_probabilities,
                regime_params=r_params,
            )

        return result
