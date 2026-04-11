"""Volatility-targeting position sizing engine.

Professional funds target a specific daily volatility, not specific returns.
By controlling volatility, consistent returns follow naturally.

Target: 1% daily portfolio volatility.
"""

import logging
import threading
from datetime import datetime

import numpy as np

import config
from risk.risk_manager import get_time_of_day_multiplier, get_friday_eow_multiplier
from utils import safe_divide

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
        self._weights_lock = threading.Lock()  # MED-005: protect _adaptive_weights

    def set_kelly_engine(self, kelly_engine):
        """Set the Kelly engine for dynamic risk sizing."""
        self._kelly_engine = kelly_engine

    def set_adaptive_weights(self, weights: dict[str, float]):
        """Store current adaptive allocation weights for position sizing (thread-safe)."""
        with self._weights_lock:
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
            self._last_scalar = config.VOL_SCALAR_MIN
            return config.VOL_SCALAR_MIN

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
        now: datetime | None = None,
        symbol: str = "",
    ) -> int:
        """
        Position sizing with volatility targeting.

        Base: risk RISK_PER_TRADE_PCT of portfolio per trade.
        Then: scale by vol_scalar, strategy allocation, PnL lock multiplier,
        intraday session multiplier (V12 4.1), and Friday EOW reduction (V12 4.3).

        Args:
            equity: Current portfolio equity
            entry_price: Expected entry price
            stop_price: Stop loss price
            vol_scalar: From compute_vol_scalar()
            strategy: Strategy name for allocation weighting
            side: 'buy' or 'sell'
            pnl_lock_mult: From DailyPnLLock.get_size_multiplier()
            now: Current datetime for intraday/Friday adjustments (optional)

        Returns:
            Number of shares (int), 0 if position is too small
        """
        if equity <= 0 or entry_price <= 0:
            return 0

        # 1. Base risk per trade (Kelly or flat)
        if self._kelly_engine is not None:
            risk_pct = self._kelly_engine.get_fraction(strategy)
            sizing_path = "kelly"
        else:
            risk_pct = config.RISK_PER_TRADE_PCT
            sizing_path = "flat"
        risk_dollars = equity * risk_pct

        # 2. Risk per share (distance to stop)
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share < 0.001:
            return 0

        shares = safe_divide(risk_dollars, risk_per_share, default=0.0)
        if shares <= 0:
            return 0
        position_value = shares * entry_price

        # 3. Apply volatility targeting scalar
        position_value *= vol_scalar

        # 4. Apply strategy allocation weight
        # MED-005: snapshot adaptive weights under lock
        with self._weights_lock:
            _aw_snapshot = dict(self._adaptive_weights) if self._adaptive_weights else None
        if (getattr(config, "ADAPTIVE_ALLOCATION_ENABLED", False)
                and _aw_snapshot
                and strategy in _aw_snapshot):
            allocation = _aw_snapshot[strategy]
        else:
            active_strategies = config.STRATEGY_ALLOCATIONS
            equal_weight = 1.0 / len(active_strategies) if active_strategies else 1.0
            allocation = active_strategies.get(strategy, equal_weight)
        # Normalize: scale by allocation relative to equal weight
        active_strategies = config.STRATEGY_ALLOCATIONS
        equal_weight = 1.0 / len(active_strategies) if active_strategies else 1.0
        position_value *= allocation / equal_weight

        # 5. Apply PnL lock multiplier
        position_value *= pnl_lock_mult

        # 6. Short selling size reduction
        if side == "sell":
            position_value *= config.SHORT_SIZE_MULTIPLIER

        # 6b. V12 4.1: Intraday session-based position sizing
        # 6c. V12 4.3: Friday end-of-week risk reduction
        tod_mult = 1.0
        friday_mult = 1.0
        if now is not None:
            tod_mult = get_time_of_day_multiplier(now)
            friday_mult = get_friday_eow_multiplier(now)
            position_value *= tod_mult * friday_mult

        # 6d. BUG-FIX: ATR-based per-stock volatility cap.  High-ATR stocks
        # (e.g. LYFT with ATR > 5% of price) can produce outsized losses
        # even when within the portfolio max-position limit.  Halve size
        # for very volatile stocks so a single adverse move doesn't wipe
        # out more than the intended risk budget.
        # V12 HOTFIX (CodeQL): entry_price > 0 is already guaranteed by the
        # early return at line 115, so the extra comparison is redundant.
        if symbol:
            try:
                # BUG-FIX: Use intraday 5-min bars (last 3 hours) for ATR cap —
                # daily bars miss intraday volatility spikes (e.g. LYFT at 4.2%)
                _atr_bars = None
                try:
                    from data import get_intraday_bars as _get_intraday_bars
                    from alpaca.data.timeframe import TimeFrame as _TF, TimeFrameUnit as _TFU
                    from datetime import timedelta as _td, timezone as _tz
                    _lookback = now - _td(hours=3) if now else datetime.now(_tz.utc) - _td(hours=3)
                    _atr_bars = _get_intraday_bars(symbol, _TF(5, _TFU.Minute), start=_lookback)
                    if _atr_bars is None or len(_atr_bars) < 10:
                        _atr_bars = None  # Fall through to daily fallback
                except Exception:
                    _atr_bars = None  # Fall through to daily fallback

                if _atr_bars is None:
                    from data import get_daily_bars as _get_daily_bars
                    _atr_bars = _get_daily_bars(symbol, days=20)

                if _atr_bars is not None and len(_atr_bars) >= 10:
                    _high = _atr_bars["high"]
                    _low = _atr_bars["low"]
                    _prev_close = _atr_bars["close"].shift(1)
                    _tr = np.maximum(
                        _high - _low,
                        np.maximum(
                            np.abs(_high - _prev_close),
                            np.abs(_low - _prev_close),
                        ),
                    )
                    _atr_14 = _tr.iloc[-14:].mean()
                    _atr_pct = _atr_14 / entry_price
                    # BUG-FIX: Tiered ATR cap — 5% was too loose (LYFT at 4.2% slipped through)
                    if _atr_pct > 0.07:  # ATR > 7% = extremely volatile — cut to 25%
                        vol_cap = 0.25
                        position_value *= vol_cap
                        logger.info(
                            "Extreme ATR %.1f%% for %s — size cut to 25%%",
                            _atr_pct * 100, symbol,
                        )
                    elif _atr_pct > 0.03:  # ATR > 3% = high vol — cut to 50%
                        vol_cap = 0.5
                        position_value *= vol_cap
                        logger.info(
                            "High ATR %.1f%% for %s — size halved",
                            _atr_pct * 100, symbol,
                        )
            except Exception:
                pass  # Fail-open: if ATR fetch fails, use original size

        # 7. Hard caps
        max_position = equity * config.MAX_POSITION_PCT
        min_position = config.MIN_POSITION_VALUE
        position_value = max(0, min(position_value, max_position))

        if position_value < min_position:
            return 0

        qty = int(safe_divide(position_value, entry_price, default=0.0))
        qty = max(qty, 0)

        # V11.3 T13: Log sizing path for diagnostics
        logger.debug(
            f"VolTarget sizing: {strategy} → {sizing_path} risk_pct={risk_pct:.4f} "
            f"vol_scalar={vol_scalar:.2f} pnl_lock={pnl_lock_mult:.2f} "
            f"alloc={allocation:.3f} tod={tod_mult:.2f} fri={friday_mult:.2f} "
            f"→ qty={qty} (${qty * entry_price:.0f})"
        )
        return qty

    @property
    def last_scalar(self) -> float:
        return self._last_scalar
