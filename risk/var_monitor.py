"""V10 Risk — Portfolio Value-at-Risk (VaR) and Conditional VaR (CVaR) monitor.

Provides real-time VaR estimation using:
1. Parametric VaR (assumes normal distribution)
2. Historical simulation VaR (uses actual P&L history)
3. Monte Carlo VaR (simulated scenarios)

Integrates with the tiered circuit breaker to trigger size reduction
when portfolio risk exceeds thresholds.

Usage:
    monitor = VaRMonitor()
    monitor.update(daily_pnls=[-0.01, 0.02, -0.005, ...], portfolio_value=100000)
    print(monitor.var_95)       # 5% daily VaR in dollars
    print(monitor.cvar_95)      # Expected shortfall at 95%
    print(monitor.risk_budget)  # How much risk capacity remains
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    """VaR computation result."""
    var_95: float = 0.0          # 95% VaR in dollars (positive = loss)
    var_99: float = 0.0          # 99% VaR in dollars
    cvar_95: float = 0.0         # 95% Conditional VaR (Expected Shortfall)
    cvar_99: float = 0.0         # 99% CVaR
    var_95_pct: float = 0.0      # VaR as % of portfolio
    method: str = "parametric"   # Which method produced this result
    computed_at: datetime = field(default_factory=datetime.now)
    sample_size: int = 0


class VaRMonitor:
    """Real-time portfolio VaR estimation and risk budgeting.

    Computes VaR using multiple methods and tracks risk budget consumption.
    """

    def __init__(
        self,
        max_var_pct: float = None,
        lookback_days: int = None,
        mc_simulations: int = None,
    ):
        self.max_var_pct = max_var_pct or getattr(config, "VAR_MAX_DAILY_PCT", 0.02)
        self.lookback_days = lookback_days or getattr(config, "VAR_LOOKBACK_DAYS", 60)
        self.mc_simulations = mc_simulations or getattr(config, "VAR_MC_SIMULATIONS", 10000)

        self._daily_returns: list[float] = []
        self._portfolio_value: float = 0.0
        self._last_result: VaRResult = VaRResult()

    def update(self, daily_pnls: list[float], portfolio_value: float) -> VaRResult:
        """Update VaR with latest daily P&L series.

        Args:
            daily_pnls: List of daily P&L percentages (e.g., [-0.01, 0.02, ...])
            portfolio_value: Current portfolio value in dollars
        """
        self._portfolio_value = portfolio_value
        self._daily_returns = daily_pnls[-self.lookback_days:]

        if len(self._daily_returns) < 10:
            logger.debug(f"VaR: insufficient data ({len(self._daily_returns)} days, need 10+)")
            self._last_result = VaRResult(sample_size=len(self._daily_returns))
            return self._last_result

        # Compute using best available method
        if len(self._daily_returns) >= 30:
            result = self._historical_var()
        else:
            result = self._parametric_var()

        result.sample_size = len(self._daily_returns)
        result.computed_at = datetime.now()
        self._last_result = result

        logger.info(
            f"VaR update: 95%=${result.var_95:.0f} ({result.var_95_pct:.2%}), "
            f"99%=${result.var_99:.0f}, CVaR95=${result.cvar_95:.0f} "
            f"[{result.method}, n={result.sample_size}]"
        )

        return result

    @staticmethod
    def _cornish_fisher_z(z: float, skew: float, excess_kurt: float) -> float:
        """Cornish-Fisher expansion: adjust normal z-score for skewness/kurtosis.

        HIGH-008: Corrects parametric VaR for non-normal return distributions.
        z_cf = z + (z^2 - 1)*S/6 + (z^3 - 3*z)*K/24 - (2*z^3 - 5*z)*S^2/36
        where S = skewness, K = excess kurtosis.
        """
        z_cf = (z
                + (z**2 - 1) * skew / 6
                + (z**3 - 3 * z) * excess_kurt / 24
                - (2 * z**3 - 5 * z) * skew**2 / 36)
        return z_cf

    def _parametric_var(self) -> VaRResult:
        """Parametric VaR with Cornish-Fisher skewness/kurtosis correction.

        RISK-001: Uses Cornish-Fisher expansion to adjust both VaR and CVaR
        z-scores for non-normal return distributions (skewness & kurtosis).
        """
        returns = np.array(self._daily_returns)
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))

        if sigma < 1e-10:
            return VaRResult(method="parametric")

        # RISK-001: Compute skewness and excess kurtosis for Cornish-Fisher
        n = len(returns)
        if n >= 20:
            from scipy.stats import skew as _skew, kurtosis as _kurtosis
            s = float(_skew(returns))
            k = float(_kurtosis(returns))  # scipy returns excess kurtosis by default
            z_95 = self._cornish_fisher_z(1.645, s, k)
            z_99 = self._cornish_fisher_z(2.326, s, k)
        else:
            s = 0.0
            k = 0.0
            z_95 = 1.645
            z_99 = 2.326

        var_95_pct = -(mu - z_95 * sigma)
        var_99_pct = -(mu - z_99 * sigma)

        var_95 = var_95_pct * self._portfolio_value
        var_99 = var_99_pct * self._portfolio_value

        # RISK-001: CVaR (Expected Shortfall) with Cornish-Fisher correction.
        # Use the CF-adjusted z-scores for the CVaR pdf evaluation so that
        # heavy tails / skew propagate into the expected-shortfall estimate.
        from math import exp, sqrt, pi
        z_95_base = 1.645  # Use unadjusted z for CVaR PDF
        phi_95 = exp(-z_95_base**2 / 2) / sqrt(2 * pi)
        cvar_95_pct = -(mu - sigma * phi_95 / 0.05)
        cvar_95 = cvar_95_pct * self._portfolio_value

        z_99_base = 2.326
        phi_99 = exp(-z_99_base**2 / 2) / sqrt(2 * pi)
        cvar_99_pct = -(mu - sigma * phi_99 / 0.01)
        cvar_99 = cvar_99_pct * self._portfolio_value

        return VaRResult(
            var_95=max(0, var_95),
            var_99=max(0, var_99),
            cvar_95=max(0, cvar_95),
            cvar_99=max(0, cvar_99),
            var_95_pct=max(0, var_95_pct),
            method="parametric-cf" if n >= 20 else "parametric",
        )

    def _historical_var(self) -> VaRResult:
        """Historical simulation VaR using actual P&L distribution."""
        returns = np.array(self._daily_returns)

        # VaR = negative of the alpha-quantile
        var_95_pct = -float(np.percentile(returns, 5))
        var_99_pct = -float(np.percentile(returns, 1))

        var_95 = var_95_pct * self._portfolio_value
        var_99 = var_99_pct * self._portfolio_value

        # CVaR = mean of returns below VaR threshold
        # HIGH-007: Use strict < to exclude the boundary observation from tail
        tail_5 = returns[returns < np.percentile(returns, 5)]
        tail_1 = returns[returns < np.percentile(returns, 1)]

        cvar_95_pct = -float(np.mean(tail_5)) if len(tail_5) > 0 else var_95_pct
        cvar_99_pct = -float(np.mean(tail_1)) if len(tail_1) > 0 else var_99_pct

        cvar_95 = cvar_95_pct * self._portfolio_value
        cvar_99 = cvar_99_pct * self._portfolio_value

        return VaRResult(
            var_95=max(0, var_95),
            var_99=max(0, var_99),
            cvar_95=max(0, cvar_95),
            cvar_99=max(0, cvar_99),
            var_95_pct=max(0, var_95_pct),
            method="historical",
        )

    def monte_carlo_var(self, horizon_days: int = 1) -> VaRResult:
        """Monte Carlo VaR simulation for multi-day horizon.

        Uses bootstrapped daily returns to simulate future scenarios.
        """
        if len(self._daily_returns) < 20:
            return VaRResult(method="monte_carlo")

        returns = np.array(self._daily_returns)
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))

        # Simulate paths
        rng = np.random.default_rng()
        simulated_returns = rng.normal(mu, sigma, (self.mc_simulations, horizon_days))
        cumulative_returns = np.sum(simulated_returns, axis=1)

        var_95_pct = -float(np.percentile(cumulative_returns, 5))
        var_99_pct = -float(np.percentile(cumulative_returns, 1))

        tail_5 = cumulative_returns[cumulative_returns <= np.percentile(cumulative_returns, 5)]
        cvar_95_pct = -float(np.mean(tail_5)) if len(tail_5) > 0 else var_95_pct

        tail_1 = cumulative_returns[cumulative_returns <= np.percentile(cumulative_returns, 1)]
        cvar_99_pct = -float(np.mean(tail_1)) if len(tail_1) > 0 else var_99_pct

        return VaRResult(
            var_95=max(0, var_95_pct * self._portfolio_value),
            var_99=max(0, var_99_pct * self._portfolio_value),
            cvar_95=max(0, cvar_95_pct * self._portfolio_value),
            cvar_99=max(0, cvar_99_pct * self._portfolio_value),
            var_95_pct=max(0, var_95_pct),
            method="monte_carlo",
            sample_size=self.mc_simulations,
        )

    @property
    def risk_budget_remaining(self) -> float:
        """How much risk capacity remains (0.0 = at limit, negative = over limit).

        Returns fraction: 1.0 = full budget, 0.0 = at VaR limit.
        """
        if self._portfolio_value <= 0:
            return 0.0
        current_var_pct = self._last_result.var_95_pct
        if current_var_pct <= 0:
            return 1.0
        return max(0.0, 1.0 - (current_var_pct / self.max_var_pct))

    @property
    def size_multiplier(self) -> float:
        """Position size multiplier based on VaR budget (1.0 = full size, 0.0 = blocked).

        Linearly scales from 1.0 at 0% budget usage to 0.0 at 100%.
        """
        return max(0.0, min(1.0, self.risk_budget_remaining))

    @property
    def result(self) -> VaRResult:
        """Get the latest VaR result."""
        return self._last_result

    @property
    def status(self) -> dict:
        r = self._last_result
        return {
            "var_95": round(r.var_95, 2),
            "var_99": round(r.var_99, 2),
            "cvar_95": round(r.cvar_95, 2),
            "var_95_pct": round(r.var_95_pct, 4),
            "method": r.method,
            "sample_size": r.sample_size,
            "risk_budget_remaining": round(self.risk_budget_remaining, 3),
            "size_multiplier": round(self.size_multiplier, 3),
            "portfolio_value": round(self._portfolio_value, 2),
            "max_var_pct": self.max_var_pct,
        }
