"""Risk management — position sizing, circuit breaker, portfolio limits, VIX scaling."""

import logging
import threading
import time as time_mod
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple, Optional

import config
import database
from analytics.metrics import compute_sharpe as _compute_sharpe_fn

from engine.event_log import log_event, EventType
from engine.failure_modes import FailureMode, handle_failure

logger = logging.getLogger(__name__)

# WIRE-006: Factor model for exposure limit checks (fail-open)
_factor_model = None
try:
    from risk.factor_model import FactorRiskModel as _FRM
    _factor_model = _FRM()
except ImportError:
    _FRM = None

# --- VIX Risk Scaling ---

# T1-003: Replace bare tuple with immutable NamedTuple for atomic snapshot reads
class VixSnapshot(NamedTuple):
    value: float
    timestamp: float

_vix_cache: Optional[VixSnapshot] = None
_vix_cache_lock = threading.Lock()  # MED-001: protect concurrent VIX cache access


def get_vix_level() -> float:
    """Get current VIX level, cached for VIX_CACHE_SECONDS."""
    global _vix_cache

    now = time_mod.time()
    # T1-003: Atomic snapshot read under lock, use outside lock
    with _vix_cache_lock:
        snapshot = _vix_cache

    if snapshot and (now - snapshot.timestamp) < config.VIX_CACHE_SECONDS:
        return snapshot.value

    try:
        import yfinance as yf
        import math
        vix = yf.Ticker("^VIX").fast_info.get("last_price", 0)
        # CRIT-015: Guard against None or NaN VIX values
        if vix is None or not isinstance(vix, (int, float)) or math.isnan(vix):
            vix = 0
        if vix > 0:
            new_snapshot = VixSnapshot(value=float(vix), timestamp=now)
            with _vix_cache_lock:
                _vix_cache = new_snapshot
            return new_snapshot.value
    except Exception as e:
        logger.warning(f"Failed to fetch VIX: {e}")

    # T1-003: Atomic snapshot read for fallback
    with _vix_cache_lock:
        snapshot = _vix_cache
    return snapshot.value if snapshot else 20.0  # Default to 20 if unavailable


_vix_history: list[float] = []  # Rolling VIX readings for rate-of-change


def get_vix_risk_scalar() -> float:
    """Return 0.0-1.5 multiplier for position sizing based on VIX level AND direction.

    V11.3 T6: Uses VIX rate-of-change, not just level.
    - VIX falling (vol compression) → scale UP (good for mean-reversion)
    - VIX spiking (vol expansion) → scale DOWN more aggressively
    - Level-based floor remains as safety net
    """
    if not config.VIX_RISK_SCALING_ENABLED:
        return 1.0

    vix = get_vix_level()

    # Track VIX history for rate-of-change (keep last 10 readings ≈ 20 min at 120s scan)
    # V11.3 P1: Protected by _vix_cache_lock for thread safety
    with _vix_cache_lock:
        _vix_history.append(vix)
        if len(_vix_history) > 10:
            _vix_history.pop(0)

    # Level-based floor (safety net)
    if vix >= config.VIX_HALT_THRESHOLD:
        return 0.0  # Halt all new positions

    # V11.4: Smooth linear interpolation instead of step discontinuities.
    # Breakpoints: VIX 12→1.0, 15→0.95, 20→0.85, 25→0.70, 30→0.50, 35→0.35
    _vix_breakpoints = [(12, 1.0), (15, 0.95), (20, 0.85), (25, 0.70), (30, 0.50), (35, 0.35)]
    if vix <= _vix_breakpoints[0][0]:
        level_scalar = _vix_breakpoints[0][1]
    elif vix >= _vix_breakpoints[-1][0]:
        level_scalar = _vix_breakpoints[-1][1]
    else:
        for i in range(len(_vix_breakpoints) - 1):
            v0, s0 = _vix_breakpoints[i]
            v1, s1 = _vix_breakpoints[i + 1]
            if v0 <= vix < v1:
                t = (vix - v0) / (v1 - v0)
                level_scalar = s0 + t * (s1 - s0)
                break

    # Rate-of-change adjustment: compare current VIX to average of last readings
    # V11.3 P1: Snapshot history under lock for thread safety
    with _vix_cache_lock:
        _history_snapshot = list(_vix_history)
    if len(_history_snapshot) >= 3:
        vix_avg = sum(_history_snapshot[:-1]) / len(_history_snapshot[:-1])
        vix_change_pct = (vix - vix_avg) / max(vix_avg, 1.0)

        if vix_change_pct > 0.10:
            # VIX spiking (>10% increase) → reduce further
            direction_adj = 0.7
        elif vix_change_pct > 0.05:
            # VIX rising moderately → slight reduction
            direction_adj = 0.85
        elif vix_change_pct < -0.10:
            # VIX falling fast (vol compression) → boost sizing
            direction_adj = 1.2
        elif vix_change_pct < -0.05:
            # VIX falling moderately → slight boost
            direction_adj = 1.1
        else:
            direction_adj = 1.0
    else:
        direction_adj = 1.0

    scalar = level_scalar * direction_adj
    # Bound to [0.3, 1.3] — never zero from direction alone (level handles halt)
    return max(0.3, min(scalar, 1.3))


@dataclass
class TradeRecord:
    symbol: str
    strategy: str
    side: str
    entry_price: float
    entry_time: datetime
    qty: int
    take_profit: float
    stop_loss: float
    pnl: float = 0.0
    exit_price: float | None = None
    exit_time: datetime | None = None
    exit_reason: str = ""            # V2: 'take_profit', 'stop_loss', 'time_stop', 'eod_close', 'max_hold'
    status: str = "open"             # "open", "closed"
    order_id: str = ""
    time_stop: datetime | None = None
    hold_type: str = "day"           # V2: "day" or "swing" (multi-day)
    max_hold_date: datetime | None = None  # V2: for momentum max hold
    pair_id: str = ""                # V4: links two legs of a pairs trade
    partial_exits: int = 0           # V4: count of partial exits taken
    highest_price_seen: float = 0.0  # V4: for trailing stop tracking (longs)
    lowest_price_seen: float = 0.0   # V10: for trailing stop tracking (shorts)
    entry_atr: float = 0.0           # V4: ATR at time of entry
    partial_closed_qty: int = 0      # V10 BUG-004: cumulative qty closed via partial exits


@dataclass
class RiskManager:
    starting_equity: float = 0.0
    current_equity: float = 0.0
    current_cash: float = 0.0
    day_pnl: float = 0.0
    circuit_breaker_active: bool = False
    open_trades: dict = field(default_factory=dict)     # symbol -> TradeRecord
    closed_trades: list = field(default_factory=list)    # today's closed trades
    signals_today: int = 0
    _strategy_weights: dict = field(default_factory=dict)  # V3: strategy -> weight
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # Per-symbol daily P&L tracking
    _symbol_daily_pnl: dict = field(default_factory=dict)  # symbol -> cumulative day pnl $

    def reset_daily(self, equity: float, cash: float):
        """Reset daily state. Preserves swing (multi-day) trades."""
        self.starting_equity = equity
        self.current_equity = equity
        self.current_cash = cash
        self.day_pnl = 0.0
        self.circuit_breaker_active = False
        self.closed_trades.clear()
        self.signals_today = 0
        self._symbol_daily_pnl.clear()

        # Preserve swing trades, clear day trades
        day_trades = [s for s, t in self.open_trades.items() if t.hold_type == "day"]
        for symbol in day_trades:
            self.open_trades.pop(symbol)

    def update_equity(self, equity: float, cash: float):
        self.current_equity = equity
        self.current_cash = cash
        if self.starting_equity > 0:
            self.day_pnl = (equity - self.starting_equity) / self.starting_equity
        else:
            self.day_pnl = 0.0

    def check_circuit_breaker(self) -> bool:
        """Check if daily loss limit hit. Returns True if trading should halt."""
        if self.day_pnl <= config.DAILY_LOSS_HALT:
            if not self.circuit_breaker_active:
                logger.warning(
                    f"CIRCUIT BREAKER ACTIVATED: Day P&L {self.day_pnl:.2%} "
                    f"hit limit of {config.DAILY_LOSS_HALT:.2%}"
                )
                log_event(EventType.CIRCUIT_BREAKER, "risk_manager",
                          details=f"day_pnl={self.day_pnl:.2%} limit={config.DAILY_LOSS_HALT:.2%}",
                          severity="WARNING")
            self.circuit_breaker_active = True
            return True
        return False

    def can_open_trade(self, strategy: str = "") -> tuple[bool, str]:
        """Check if we can open a new trade. Returns (allowed, reason)."""
        if self.circuit_breaker_active:
            return False, "Circuit breaker active"

        if len(self.open_trades) >= config.MAX_POSITIONS:
            return False, f"Max positions ({config.MAX_POSITIONS}) reached"

        # V4: VIX halt check
        if config.VIX_RISK_SCALING_ENABLED and get_vix_risk_scalar() == 0.0:
            return False, f"VIX > {config.VIX_HALT_THRESHOLD} — trading halted"

        # Check total deployed capital
        deployed = sum(
            t.entry_price * t.qty for t in self.open_trades.values()
        )
        max_deploy = self.current_equity * config.MAX_PORTFOLIO_DEPLOY
        if deployed >= max_deploy:
            return False, f"Max portfolio deployment ({config.MAX_PORTFOLIO_DEPLOY:.0%}) reached"

        # Per-symbol daily loss cap — block if we've already lost too much on this symbol today
        # (checked per-symbol in _process_single_signal via symbol parameter)

        # WIRE-006: Factor exposure limit check (fail-open)
        try:
            if _factor_model is not None and self.open_trades:
                violations = _factor_model.check_factor_limits(
                    positions=self.open_trades,
                    returns_data=None,
                    exposures=None,
                )
                if violations:
                    return False, f"Factor limit breach: {violations[0]}"
        except Exception as _e:
            logger.debug("WIRE-006: Factor model check failed (fail-open): %s", _e)

        return True, ""

    def calculate_position_size(self, entry_price: float, stop_price: float,
                               regime: str, strategy: str = "",
                               side: str = "buy") -> int:
        """Legacy base position sizing (used by RiskManager tests).

        For production sizing, use VolatilityTargetingRiskEngine.calculate_position_size()
        which adds Kelly, vol-scalar, and PnL-lock adjustments.

        Applies dynamic capital allocation weights per strategy and
        short selling multiplier for sell-side trades.

        Args:
            entry_price: Expected entry price
            stop_price: Stop loss price
            regime: Market regime ('BULLISH', 'BEARISH', 'UNKNOWN')
            strategy: Strategy name for capital allocation weighting
            side: 'buy' or 'sell' — shorts get reduced sizing
        """
        # Risk per trade = 1% of portfolio
        risk_per_trade = self.current_equity * config.RISK_PER_TRADE_PCT

        # Distance to stop in dollars per share
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share == 0 or entry_price <= 0:
            return 0

        # Shares = how many to risk exactly 1%
        shares = risk_per_trade / risk_per_share
        position_value = shares * entry_price

        # V3: Apply dynamic capital allocation weight
        # T1-004: Use immutable snapshot via get_strategy_weights() to avoid lock contention
        if config.DYNAMIC_ALLOCATION and strategy:
            weights = self.get_strategy_weights()  # Thread-safe snapshot copy
            if weights:
                weight = weights.get(strategy, 1.0)
                # Scale position by strategy weight relative to equal weight
                n_strategies = len(weights)
                equal_weight = 1.0 / max(n_strategies, 1)
                weight_factor = weight / equal_weight if equal_weight > 0 else 1.0
                position_value *= weight_factor

        # Hard caps
        max_position = self.current_equity * config.MAX_POSITION_PCT
        position_value = max(config.MIN_POSITION_VALUE, min(position_value, max_position))

        # Cut size in bearish regime
        if regime == "BEARISH":
            position_value *= (1 - config.BEARISH_SIZE_CUT)

        # V3: Short selling size reduction
        if side == "sell":
            position_value *= config.SHORT_SIZE_MULTIPLIER

        # V4: VIX-based risk scaling
        vix_scalar = get_vix_risk_scalar()
        if vix_scalar < 1.0:
            position_value *= vix_scalar

        # Check we don't exceed max deployment
        deployed = sum(t.entry_price * t.qty for t in self.open_trades.values())
        remaining_deploy = self.current_equity * config.MAX_PORTFOLIO_DEPLOY - deployed
        position_value = min(position_value, remaining_deploy)

        if position_value <= 0:
            return 0

        qty = int(position_value / entry_price)
        return max(qty, 0)

    def register_trade(self, trade: TradeRecord):
        """Register a new open trade (thread-safe)."""
        with self._lock:
            self.open_trades[trade.symbol] = trade
            self.signals_today += 1
        logger.info(
            f"Trade opened: {trade.side.upper()} {trade.qty} {trade.symbol} "
            f"@ {trade.entry_price:.2f} ({trade.strategy}/{trade.hold_type}) "
            f"TP={trade.take_profit:.2f} SL={trade.stop_loss:.2f}"
        )

    def close_trade(self, symbol: str, exit_price: float, now: datetime,
                    exit_reason: str = ""):
        """Close a trade, record P&L, and log to database (thread-safe)."""
        with self._lock:
            if symbol not in self.open_trades:
                return
            trade = self.open_trades.pop(symbol)
        trade.exit_price = exit_price
        trade.exit_time = now
        trade.status = "closed"
        trade.exit_reason = exit_reason

        if trade.side == "buy":
            trade.pnl = (exit_price - trade.entry_price) * trade.qty
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.qty

        pnl_pct = trade.pnl / (trade.entry_price * trade.qty) if trade.entry_price * trade.qty > 0 else 0

        self.closed_trades.append(trade)

        # Track per-symbol daily P&L
        self._symbol_daily_pnl[symbol] = self._symbol_daily_pnl.get(symbol, 0.0) + trade.pnl

        logger.info(
            f"Trade closed: {trade.symbol} ({trade.strategy}) "
            f"P&L=${trade.pnl:+.2f} ({pnl_pct:.1%}) reason={exit_reason}"
        )
        log_event(EventType.POSITION_CLOSED, "risk_manager",
                  symbol=trade.symbol, strategy=trade.strategy,
                  details=f"pnl={trade.pnl:+.2f} pnl_pct={pnl_pct:.1%} reason={exit_reason}")

        # Log to database
        try:
            database.log_trade(
                symbol=trade.symbol,
                strategy=trade.strategy,
                side=trade.side,
                entry_price=trade.entry_price,
                exit_price=exit_price,
                qty=trade.qty,
                entry_time=trade.entry_time,
                exit_time=now,
                exit_reason=exit_reason,
                pnl=trade.pnl,
                pnl_pct=pnl_pct,
            )
        except Exception as e:
            handle_failure(FailureMode.SKIP_SIGNAL, "risk_manager.log_trade_to_db", e,
                           symbol=trade.symbol, strategy=trade.strategy)

        # V10 + T5-008: Record PDT if same-day close (both trackers)
        try:
            if (trade.hold_type == "day" and trade.entry_time
                    and trade.entry_time.date() == now.date()):
                if hasattr(self, '_pdt') and self._pdt:
                    self._pdt.record_day_trade(trade.symbol, now.date())
                # T5-008: Also record with PDTCompliance for persistent tracking
                if hasattr(self, '_pdt_compliance') and self._pdt_compliance:
                    self._pdt_compliance.record_day_trade(
                        symbol=trade.symbol,
                        trade_date=now.date(),
                        side=trade.side,
                        qty=float(trade.qty),
                        entry_price=trade.entry_price,
                        exit_price=exit_price,
                    )
        except Exception as e:
            handle_failure(FailureMode.DEGRADE_GRACEFULLY, "risk_manager.pdt_recording", e,
                           symbol=trade.symbol, strategy=trade.strategy)

    def partial_close(self, symbol: str, qty_to_close: int, exit_price: float,
                      now: datetime, exit_reason: str = "partial_tp"):
        """Close a portion of a position. Reduces qty, logs partial P&L."""
        if symbol not in self.open_trades:
            return

        trade = self.open_trades[symbol]

        # V10 BUG-002: Clamp and assert cumulative closes never exceed original qty
        qty_to_close = min(qty_to_close, trade.qty)
        if qty_to_close <= 0:
            logger.warning(f"Partial close {symbol}: qty_to_close={qty_to_close} <= 0, skipping")
            return
        assert qty_to_close <= trade.qty, (
            f"Partial close {symbol}: qty_to_close={qty_to_close} > remaining={trade.qty}"
        )

        # Calculate P&L on closed portion
        if trade.side == "buy":
            partial_pnl = (exit_price - trade.entry_price) * qty_to_close
        else:
            partial_pnl = (trade.entry_price - exit_price) * qty_to_close

        pnl_pct = partial_pnl / (trade.entry_price * qty_to_close) if trade.entry_price > 0 else 0

        # Update remaining qty
        trade.qty -= qty_to_close
        trade.partial_exits += 1

        logger.info(
            f"Partial exit: {symbol} ({trade.strategy}) closed {qty_to_close} shares "
            f"P&L=${partial_pnl:+.2f} ({pnl_pct:.1%}) reason={exit_reason} "
            f"remaining={trade.qty}"
        )

        # Log partial to DB as a trade
        try:
            database.log_trade(
                symbol=symbol,
                strategy=trade.strategy,
                side=trade.side,
                entry_price=trade.entry_price,
                exit_price=exit_price,
                qty=qty_to_close,
                entry_time=trade.entry_time,
                exit_time=now,
                exit_reason=exit_reason,
                pnl=partial_pnl,
                pnl_pct=pnl_pct,
            )
        except Exception as e:
            logger.error(f"Failed to log partial trade to DB: {e}")

        # If no shares remaining, remove from open trades
        if trade.qty <= 0:
            self.open_trades.pop(symbol)

    def get_day_summary(self) -> dict:
        """Generate end-of-day summary stats."""
        all_trades = self.closed_trades
        if not all_trades:
            return {"trades": 0}

        winners = [t for t in all_trades if t.pnl > 0]
        losers = [t for t in all_trades if t.pnl <= 0]

        # Per-strategy breakdown
        strategies = {}
        for t in all_trades:
            s = t.strategy
            if s not in strategies:
                strategies[s] = {"total": 0, "winners": 0}
            strategies[s]["total"] += 1
            if t.pnl > 0:
                strategies[s]["winners"] += 1

        total_pnl = sum(t.pnl for t in all_trades)
        best = max(all_trades, key=lambda t: t.pnl)
        worst = min(all_trades, key=lambda t: t.pnl)

        result = {
            "trades": len(all_trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(all_trades) if all_trades else 0,
            "total_pnl": total_pnl,
            "pnl_pct": total_pnl / self.starting_equity if self.starting_equity else 0,
            "best_trade": f"{best.symbol} {best.strategy} ${best.pnl:+.0f}",
            "worst_trade": f"{worst.symbol} {worst.strategy} ${worst.pnl:+.0f}",
        }

        # Add per-strategy win rates
        for strat, data in strategies.items():
            result[f"{strat.lower()}_win_rate"] = f"{data['winners']}/{data['total']}"

        return result

    def load_from_db(self):
        """Restore open positions from database."""
        try:
            rows = database.load_open_positions()
            for row in rows:
                self.open_trades[row["symbol"]] = TradeRecord(
                    symbol=row["symbol"],
                    strategy=row["strategy"],
                    side=row["side"],
                    entry_price=row["entry_price"],
                    entry_time=datetime.fromisoformat(row["entry_time"]),
                    qty=int(row["qty"]),
                    take_profit=row["take_profit"],
                    stop_loss=row["stop_loss"],
                    order_id=row.get("alpaca_order_id", ""),
                    hold_type=row.get("hold_type", "day"),
                    time_stop=datetime.fromisoformat(row["time_stop"]) if row.get("time_stop") else None,
                    max_hold_date=datetime.fromisoformat(row["max_hold_date"]) if row.get("max_hold_date") else None,
                    pair_id=row.get("pair_id", ""),
                    partial_exits=int(row.get("partial_exits", 0)),
                    highest_price_seen=float(row.get("highest_price_seen", 0.0)),
                    lowest_price_seen=float(row.get("lowest_price_seen", 0.0)),
                    entry_atr=float(row.get("entry_atr", 0.0)),
                )
            logger.info(f"Restored {len(self.open_trades)} open trades from database")
        except Exception as e:
            logger.error(f"Failed to load positions from DB: {e}")

    # --- Dynamic Capital Allocation ---

    def update_strategy_weights(self):
        """Recalculate capital allocation weights based on rolling Sharpe.

        Called daily at 9:00 AM. Strategies with higher recent Sharpe
        get proportionally more capital.
        """
        if not config.DYNAMIC_ALLOCATION:
            return

        try:
            sharpes = {}
            strategies = list(config.STRATEGY_ALLOCATIONS.keys())

            for strategy in strategies:
                trades = database.get_recent_trades_by_strategy(
                    strategy, days=config.ALLOCATION_LOOKBACK_DAYS
                )
                if len(trades) < 5:
                    sharpes[strategy] = 0.5  # Default if insufficient data
                else:
                    daily_pnls = self._compute_strategy_daily_returns(trades)
                    sharpes[strategy] = max(self._compute_sharpe(daily_pnls), 0.1)

            total_sharpe = sum(sharpes.values())
            if total_sharpe <= 0:
                return

            new_weights = {
                s: max(sharpe / total_sharpe, config.ALLOCATION_MIN_WEIGHT)
                for s, sharpe in sharpes.items()
            }

            # Normalize so weights sum to 1.0
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                new_weights = {
                    s: w / total_weight for s, w in new_weights.items()
                }

            # T1-004 + BUG-007: Atomically swap immutable dict snapshot under lock
            frozen_weights = dict(new_weights)  # Immutable copy for atomic swap
            with self._lock:
                self._strategy_weights = frozen_weights

            logger.info(f"Capital allocation updated: {new_weights}")
            database.log_allocation_weights(new_weights)

        except Exception as e:
            logger.error(f"Failed to update strategy weights: {e}")

    def get_strategy_weights(self) -> dict:
        """Get current strategy capital weights (thread-safe snapshot)."""
        # BUG-007: Protect _strategy_weights read with lock
        with self._lock:
            return dict(self._strategy_weights)

    def get_weight(self, strategy: str) -> float:
        """T1-004: Get a single strategy's capital weight (thread-safe).

        Returns the weight for the given strategy, or 1.0 if not found.
        Encapsulates locking so callers don't need to manage it.
        """
        with self._lock:
            weights = self._strategy_weights
            if not weights:
                return 1.0
            return weights.get(strategy, 1.0)

    @staticmethod
    def _compute_strategy_daily_returns(trades: list[dict]) -> list[float]:
        """Compute daily return series from a list of trade dicts."""
        from collections import defaultdict
        daily = defaultdict(float)
        for t in trades:
            exit_date = t.get("exit_time", "")[:10]
            if exit_date:
                daily[exit_date] += t.get("pnl_pct", 0.0)
        return list(daily.values()) if daily else []

    @staticmethod
    def _compute_sharpe(daily_returns: list[float], rf_annual: float = 0.045) -> float:
        """Annualized Sharpe ratio. Delegates to analytics.metrics.compute_sharpe."""
        return _compute_sharpe_fn(daily_returns, risk_free_rate=rf_annual)

    # --- Legacy serialization (kept for migration compatibility) ---

    def to_dict(self) -> dict:
        """Serialize state for persistence (legacy)."""
        return {
            "starting_equity": self.starting_equity,
            "day_pnl": self.day_pnl,
            "circuit_breaker_active": self.circuit_breaker_active,
            "signals_today": self.signals_today,
            "open_trades": {
                symbol: {
                    "symbol": t.symbol,
                    "strategy": t.strategy,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "entry_time": t.entry_time.isoformat(),
                    "qty": t.qty,
                    "take_profit": t.take_profit,
                    "stop_loss": t.stop_loss,
                    "order_id": t.order_id,
                    "time_stop": t.time_stop.isoformat() if t.time_stop else None,
                    "hold_type": t.hold_type,
                }
                for symbol, t in self.open_trades.items()
            },
            "closed_trades": [
                {
                    "symbol": t.symbol,
                    "strategy": t.strategy,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "qty": t.qty,
                    "pnl": t.pnl,
                }
                for t in self.closed_trades
            ],
        }

    def load_from_dict(self, d: dict, now: datetime):
        """Restore state from JSON (legacy, for migration)."""
        self.starting_equity = d.get("starting_equity", 0)
        self.day_pnl = d.get("day_pnl", 0)
        self.circuit_breaker_active = d.get("circuit_breaker_active", False)
        self.signals_today = d.get("signals_today", 0)

        for symbol, td in d.get("open_trades", {}).items():
            self.open_trades[symbol] = TradeRecord(
                symbol=td["symbol"],
                strategy=td["strategy"],
                side=td["side"],
                entry_price=td["entry_price"],
                entry_time=datetime.fromisoformat(td["entry_time"]),
                qty=td["qty"],
                take_profit=td["take_profit"],
                stop_loss=td["stop_loss"],
                order_id=td.get("order_id", ""),
                time_stop=datetime.fromisoformat(td["time_stop"]) if td.get("time_stop") else None,
                hold_type=td.get("hold_type", "day"),
            )
        logger.info(f"Restored {len(self.open_trades)} open trades from state")
