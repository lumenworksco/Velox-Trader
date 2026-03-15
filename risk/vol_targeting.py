"""Volatility-targeting position sizing engine.

Professional funds target a specific daily volatility, not specific returns.
By controlling volatility, consistent returns follow naturally.

Target: 1% daily portfolio volatility.
"""

import logging
import numpy as np

import config

logger = logging.getLogger(__name__)


class VolatilityTargetingRiskEngine:
    """Scale all position sizes so expected daily portfolio volatility = target.

    On high-volatility days: size down automatically.
    On low-volatility days: size up slightly.
    """

    def __init__(self):
        self.target_vol = config.VOL_TARGET_DAILY
        self.max_vol = config.VOL_TARGET_MAX
        self._last_scalar = 1.0
        self._kelly_engine = None
        self._adaptive_weights: dict[str, float] | None = None

    def set_kelly_engine(self, kelly_engine):
        """Set the Kelly engine for dynamic risk sizing."""
        self._kelly_engine = kelly_engine

    def set_adaptive_weights(self, weights: dict[str, float]):
        """Store current adaptive allocation weights for position sizing."""
        self._adaptive_weights = dict(weights)

    def compute_vol_scalar(
        self,
        vix: float = 20.0,
        portfolio_atr_vol: float = 0.01,
        rolling_pnl_std: float = 0.01,
    ) -> float:
        """
        How much to scale all position sizes.

        Combines three volatility estimates:
        - VIX (market-wide vol proxy, 30%)
        - Portfolio ATR-based vol (position-level, 40%)
        - Rolling std of recent daily P&L (realized, 30%)

        Returns:
            Scalar clamped between VOL_SCALAR_MIN and VOL_SCALAR_MAX
        """
        # Convert VIX to daily vol estimate
        vix_daily = vix / 100.0 / np.sqrt(252)

        # Weighted average
        estimated_vol = (
            vix_daily * 0.3
            + portfolio_atr_vol * 0.4
            + rolling_pnl_std * 0.3
        )

        if estimated_vol < 1e-6:
            self._last_scalar = 1.0
            return 1.0

        scalar = self.target_vol / estimated_vol
        scalar = max(config.VOL_SCALAR_MIN, min(config.VOL_SCALAR_MAX, scalar))

        self._last_scalar = scalar
        return scalar

    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        vol_scalar: float = 1.0,
        strategy: str = "",
        side: str = "buy",
        pnl_lock_mult: float = 1.0,
    ) -> int:
        """
        Position sizing with volatility targeting.

        Base: risk RISK_PER_TRADE_PCT of portfolio per trade.
        Then: scale by vol_scalar, strategy allocation, and PnL lock multiplier.

        Args:
            equity: Current portfolio equity
            entry_price: Expected entry price
            stop_price: Stop loss price
            vol_scalar: From compute_vol_scalar()
            strategy: Strategy name for allocation weighting
            side: 'buy' or 'sell'
            pnl_lock_mult: From DailyPnLLock.get_size_multiplier()

        Returns:
            Number of shares (int), 0 if position is too small
        """
        if equity <= 0 or entry_price <= 0:
            return 0

        # 1. Base risk per trade (Kelly or flat)
        if self._kelly_engine is not None:
            risk_pct = self._kelly_engine.get_fraction(strategy)
        else:
            risk_pct = config.RISK_PER_TRADE_PCT
        risk_dollars = equity * risk_pct

        # 2. Risk per share (distance to stop)
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share < 0.001:
            return 0

        shares = risk_dollars / risk_per_share
        position_value = shares * entry_price

        # 3. Apply volatility targeting scalar
        position_value *= vol_scalar

        # 4. Apply strategy allocation weight
        if (getattr(config, "ADAPTIVE_ALLOCATION_ENABLED", False)
                and self._adaptive_weights
                and strategy in self._adaptive_weights):
            allocation = self._adaptive_weights[strategy]
        else:
            allocation = config.STRATEGY_ALLOCATIONS.get(strategy, 0.33)
        # Normalize: if equal weight would be 0.33, scale by allocation / 0.33
        position_value *= allocation / 0.33

        # 5. Apply PnL lock multiplier
        position_value *= pnl_lock_mult

        # 6. Short selling size reduction
        if side == "sell":
            position_value *= config.SHORT_SIZE_MULTIPLIER

        # 7. Hard caps
        max_position = equity * config.MAX_POSITION_PCT
        min_position = config.MIN_POSITION_VALUE
        position_value = max(0, min(position_value, max_position))

        if position_value < min_position:
            return 0

        qty = int(position_value / entry_price)
        return max(qty, 0)

    @property
    def last_scalar(self) -> float:
        return self._last_scalar
