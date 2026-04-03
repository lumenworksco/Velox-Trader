"""EXEC-001: Smart Order Router — adaptive order type selection.

Routes orders based on market microstructure conditions:
- Spread width relative to expected alpha
- Order size vs displayed liquidity
- Signal urgency (decay rate)
- Current volatility regime
"""

import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import config

logger = logging.getLogger(__name__)

# WIRE-005: SpreadAnalyzer for enriched spread data (fail-open)
_spread_analyzer = None
try:
    from microstructure.spread_analysis import SpreadAnalyzer as _SA
    _spread_analyzer = _SA()
except ImportError:
    _SA = None


class OrderTypeChoice(Enum):
    """Order type decisions from the router."""
    LIMIT_PASSIVE = "limit_passive"       # At mid — maximum rebate capture
    LIMIT_AGGRESSIVE = "limit_aggressive"  # Inside the spread — balance speed/cost
    LIMIT_JOIN_BBO = "limit_join_bbo"      # At best bid/ask — queue for fill
    MARKET = "market"                      # Immediate execution — speed priority


class UrgencyLevel(Enum):
    """Signal urgency tiers."""
    LOW = "low"          # Mean reversion — signal persists
    MEDIUM = "medium"    # Breakout — moderate decay
    HIGH = "high"        # Momentum — fast decay
    CRITICAL = "critical"  # Event-driven — immediate or nothing


@dataclass
class MarketConditions:
    """Snapshot of market microstructure for routing decisions."""
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0
    bid_size: int = 0         # Displayed size at best bid
    ask_size: int = 0         # Displayed size at best ask
    adv: float = 0.0          # Average daily volume (shares)
    volatility: float = 0.0   # Annualized volatility
    vix: float = 0.0          # VIX level for regime awareness
    last_price: float = 0.0

    @classmethod
    def from_quote(cls, bid: float, ask: float, bid_size: int = 100,
                   ask_size: int = 100, adv: float = 1_000_000,
                   volatility: float = 0.25, vix: float = 18.0) -> "MarketConditions":
        """Build conditions from a live quote."""
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
        spread = ask - bid if ask > bid else 0
        spread_bps = (spread / mid * 10_000) if mid > 0 else 0
        return cls(
            bid=bid, ask=ask, mid=mid, spread=spread,
            spread_bps=spread_bps, bid_size=bid_size, ask_size=ask_size,
            adv=adv, volatility=volatility, vix=vix, last_price=mid,
        )


@dataclass
class OrderParams:
    """Routing decision output — what to submit to the broker."""
    order_type: OrderTypeChoice
    limit_price: float | None = None  # None for market orders
    time_in_force: str = "day"        # "day", "ioc", "gtc"
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM
    reason: str = ""
    chase_enabled: bool = False       # Auto-chase if unfilled
    chase_after_sec: int = 60         # Seconds before chasing
    max_chase_price: float | None = None  # Price cap for chasing

    # Execution schedule (for large orders)
    use_twap: bool = False
    twap_slices: int = 5
    twap_interval_sec: int = 60


# Strategy -> urgency mapping (how fast does the alpha decay?)
_STRATEGY_URGENCY: dict[str, UrgencyLevel] = {
    "STAT_MR": UrgencyLevel.LOW,
    "KALMAN_PAIRS": UrgencyLevel.LOW,
    "VWAP": UrgencyLevel.LOW,
    "ORB": UrgencyLevel.MEDIUM,
    "PEAD": UrgencyLevel.MEDIUM,
    "MICRO_MOM": UrgencyLevel.HIGH,
    "BETA_HEDGE": UrgencyLevel.CRITICAL,
}

# Aggressiveness escalation thresholds (in seconds since order creation)
_ESCALATION_SCHEDULE = [
    (0, OrderTypeChoice.LIMIT_PASSIVE),
    (30, OrderTypeChoice.LIMIT_JOIN_BBO),
    (60, OrderTypeChoice.LIMIT_AGGRESSIVE),
    (120, OrderTypeChoice.MARKET),
]


class SmartOrderRouter:
    """Adaptive order routing engine.

    Selects order type and pricing based on market microstructure,
    signal characteristics, and fill probability estimates.

    Usage:
        router = SmartOrderRouter()
        params = router.route_order(signal, market_data)
        # Submit params.order_type at params.limit_price
    """

    def __init__(self):
        self._spread_threshold_pct: float = getattr(
            config, "SPREAD_THRESHOLD_PCT", 0.0015
        )
        self._chase_after_sec: int = getattr(config, "CHASE_AFTER_SECONDS", 60)
        self._chase_convert_sec: int = getattr(
            config, "CHASE_CONVERT_MARKET_AFTER", 120
        )
        self._twap_enabled: bool = getattr(config, "ADAPTIVE_TWAP_ENABLED", True)

        # Fill probability model parameters (calibrated from historical data)
        self._fill_prob_halflife_sec: float = 45.0  # Time for 50% fill prob decay
        self._size_impact_coeff: float = 0.5  # Impact of size/adv on routing

        # Routing statistics
        self._route_counts: dict[str, int] = {t.value: 0 for t in OrderTypeChoice}
        self._total_routed: int = 0

    def route_order(self, signal, market_data: MarketConditions,
                    now: datetime | None = None) -> OrderParams:
        """Determine optimal order routing for a signal.

        V11.2 enhancements:
        - Avoid first 5 min (9:30-9:35) and last 5 min (15:55-16:00) for limit orders.
        - Mid-quote improvement: for liquid symbols, try limit at mid for 30s before
          falling back to market.

        Args:
            signal: Trading signal with symbol, strategy, side, entry_price.
            market_data: Current market microstructure snapshot.
            now: Current datetime (defaults to datetime.now()).

        Returns:
            OrderParams with order type, price, and execution instructions.
        """
        if now is None:
            now = datetime.now()

        urgency = _STRATEGY_URGENCY.get(signal.strategy, UrgencyLevel.MEDIUM)

        # V11.2: Timing guard — avoid first/last 5 minutes for limit orders
        market_time = now.replace(tzinfo=None).time() if now.tzinfo else now.time()
        from datetime import time as time_cls
        in_open_auction = time_cls(9, 30) <= market_time < time_cls(9, 35)
        in_close_auction = time_cls(15, 55) <= market_time <= time_cls(16, 0)

        if in_open_auction or in_close_auction:
            # Use market orders during volatile open/close periods
            self._route_counts[OrderTypeChoice.MARKET.value] += 1
            self._total_routed += 1
            period = "open" if in_open_auction else "close"
            return OrderParams(
                order_type=OrderTypeChoice.MARKET,
                limit_price=None,
                urgency=urgency,
                reason=f"Market order during {period} auction window",
                chase_enabled=False,
            )

        # Gather decision factors
        spread_bps = market_data.spread_bps

        # WIRE-005: Enrich spread_bps with SpreadAnalyzer effective spread (fail-open)
        try:
            if _spread_analyzer is not None:
                effective_bps = _spread_analyzer.get_effective_spread_bps(signal.symbol)
                if effective_bps is not None and effective_bps > 0:
                    # Use the wider of quote spread and effective spread for safety
                    spread_bps = max(spread_bps, effective_bps)
        except Exception as _e:
            logger.debug("WIRE-005: SpreadAnalyzer failed for %s (fail-open): %s", signal.symbol, _e)
        size_ratio = self._compute_size_ratio(signal, market_data)
        vol_regime = self._classify_vol_regime(market_data)

        logger.debug(
            f"SmartRouter [{signal.symbol}]: spread={spread_bps:.1f}bps "
            f"size_ratio={size_ratio:.3f} vol={vol_regime} urgency={urgency.value}"
        )

        # Decision tree
        order_type, limit_price, reason = self._select_order_type(
            signal, market_data, urgency, spread_bps, size_ratio, vol_regime
        )

        # Determine TWAP eligibility for large orders
        use_twap = False
        twap_slices = 5
        if self._twap_enabled and size_ratio > 0.02:
            # Order is > 2% of ADV: use TWAP
            use_twap = True
            # More slices for larger orders
            twap_slices = min(20, max(3, int(size_ratio * 100)))
            reason += f" | TWAP {twap_slices} slices (size/ADV={size_ratio:.1%})"

        # V11.2: Mid-quote improvement for liquid symbols
        # If spread is tight (< 5 bps) and urgency is not critical, try mid-quote
        # limit for 30s before escalating to market.
        is_liquid = spread_bps < 5 and market_data.adv > 500_000
        if is_liquid and order_type not in (OrderTypeChoice.MARKET,) and urgency != UrgencyLevel.CRITICAL:
            order_type = OrderTypeChoice.LIMIT_PASSIVE
            limit_price = round(market_data.mid, 2)
            reason += " | mid-quote improvement (30s before fallback)"

        # Chase configuration
        chase_enabled = order_type != OrderTypeChoice.MARKET
        chase_after = self._chase_after_sec
        # For mid-quote improvement, chase after 30s instead of default
        if is_liquid and chase_enabled:
            chase_after = 30

        max_chase = None
        if chase_enabled and market_data.mid > 0:
            # Never chase more than 0.3% from mid
            if signal.side == "buy":
                max_chase = round(market_data.mid * 1.003, 2)
            else:
                max_chase = round(market_data.mid * 0.997, 2)

        # Track statistics
        self._route_counts[order_type.value] += 1
        self._total_routed += 1

        params = OrderParams(
            order_type=order_type,
            limit_price=limit_price,
            urgency=urgency,
            reason=reason,
            chase_enabled=chase_enabled,
            chase_after_sec=chase_after,
            max_chase_price=max_chase,
            use_twap=use_twap,
            twap_slices=twap_slices,
        )

        logger.info(
            f"SmartRouter [{signal.symbol}]: {order_type.value} "
            f"@ {limit_price or 'MKT'} ({reason})"
        )
        return params

    def get_escalated_order_type(
        self, elapsed_sec: float, urgency: UrgencyLevel
    ) -> OrderTypeChoice:
        """Get the escalated order type based on time elapsed.

        Used for chase logic: if an order hasn't filled, escalate aggressiveness.

        Args:
            elapsed_sec: Seconds since initial order submission.
            urgency: Original urgency level.

        Returns:
            Escalated OrderTypeChoice.
        """
        # Higher urgency = faster escalation
        urgency_multiplier = {
            UrgencyLevel.LOW: 2.0,       # Slow escalation
            UrgencyLevel.MEDIUM: 1.0,    # Normal
            UrgencyLevel.HIGH: 0.5,      # Fast
            UrgencyLevel.CRITICAL: 0.25, # Immediate
        }
        mult = urgency_multiplier.get(urgency, 1.0)
        effective_elapsed = elapsed_sec / mult

        result = OrderTypeChoice.LIMIT_PASSIVE
        for threshold_sec, order_type in _ESCALATION_SCHEDULE:
            if effective_elapsed >= threshold_sec:
                result = order_type
        return result

    def _select_order_type(
        self,
        signal,
        mkt: MarketConditions,
        urgency: UrgencyLevel,
        spread_bps: float,
        size_ratio: float,
        vol_regime: str,
    ) -> tuple[OrderTypeChoice, float | None, str]:
        """Core routing logic. Returns (order_type, limit_price, reason)."""

        # Rule 0: ORB-specific routing — breakout signals need aggressive limit
        # at the breakout price to ensure fills at the intended level.
        # CRIT-027: Must be checked BEFORE generic conditions (e.g. size ratio)
        # which would otherwise misroute ORB to passive limits.
        if getattr(signal, 'strategy', '') == 'ORB':
            limit_price = round(signal.entry_price, 2)
            return (
                OrderTypeChoice.LIMIT_AGGRESSIVE,
                limit_price,
                "ORB breakout — aggressive limit at breakout price",
            )

        # Rule 1: Critical urgency always goes market
        if urgency == UrgencyLevel.CRITICAL:
            return OrderTypeChoice.MARKET, None, "critical urgency"

        # Rule 2: Very wide spread + not urgent -> passive limit to capture spread
        if spread_bps > 20 and urgency in (UrgencyLevel.LOW, UrgencyLevel.MEDIUM):
            limit_price = self._compute_passive_price(signal, mkt)
            return (
                OrderTypeChoice.LIMIT_PASSIVE,
                limit_price,
                f"wide spread ({spread_bps:.0f}bps) — passive capture",
            )

        # Rule 3: Large order relative to liquidity -> limit to minimize impact
        if size_ratio > 0.01:
            limit_price = self._compute_aggressive_limit(signal, mkt, aggression=0.3)
            return (
                OrderTypeChoice.LIMIT_AGGRESSIVE,
                limit_price,
                f"large size ({size_ratio:.1%} ADV) — limit to reduce impact",
            )

        # Rule 4: High volatility regime + moderate urgency -> aggressive limit
        if vol_regime == "high" and urgency == UrgencyLevel.HIGH:
            limit_price = self._compute_aggressive_limit(signal, mkt, aggression=0.7)
            return (
                OrderTypeChoice.LIMIT_AGGRESSIVE,
                limit_price,
                f"high vol + high urgency — aggressive limit",
            )

        # Rule 5: High urgency + tight spread -> market (cost is low)
        if urgency == UrgencyLevel.HIGH and spread_bps < 5:
            return OrderTypeChoice.MARKET, None, "high urgency + tight spread"

        # Rule 6: Normal conditions — join BBO
        if spread_bps <= self._spread_threshold_pct * 10_000:
            limit_price = self._compute_bbo_price(signal, mkt)
            return (
                OrderTypeChoice.LIMIT_JOIN_BBO,
                limit_price,
                f"normal spread ({spread_bps:.0f}bps) — join BBO",
            )

        # Rule 7: Default — aggressive limit inside spread
        limit_price = self._compute_aggressive_limit(signal, mkt, aggression=0.5)
        return (
            OrderTypeChoice.LIMIT_AGGRESSIVE,
            limit_price,
            f"default — aggressive limit (spread={spread_bps:.0f}bps)",
        )

    def _compute_passive_price(self, signal, mkt: MarketConditions) -> float:
        """Price at the midpoint — maximum spread capture."""
        return round(mkt.mid, 2)

    def _compute_bbo_price(self, signal, mkt: MarketConditions) -> float:
        """Price at best bid (buys) or best ask (sells) — join the queue."""
        if signal.side == "buy":
            return round(mkt.bid, 2)
        return round(mkt.ask, 2)

    def _compute_aggressive_limit(
        self, signal, mkt: MarketConditions, aggression: float = 0.5
    ) -> float:
        """Price inside the spread. aggression=0 is mid, aggression=1 is crossing.

        Args:
            signal: Trade signal (for side).
            mkt: Market conditions.
            aggression: 0.0 (mid) to 1.0 (cross the spread).

        Returns:
            Limit price rounded to 2 decimals.
        """
        if signal.side == "buy":
            # Move from mid toward the ask
            price = mkt.mid + (mkt.ask - mkt.mid) * aggression
        else:
            # Move from mid toward the bid
            price = mkt.mid - (mkt.mid - mkt.bid) * aggression
        return round(price, 2)

    def _compute_size_ratio(self, signal, mkt: MarketConditions) -> float:
        """Order size as fraction of average daily volume."""
        if mkt.adv <= 0:
            return 0.0
        # Estimate qty from signal if not directly available
        qty = getattr(signal, "qty", 0)
        if qty <= 0 and signal.entry_price > 0:
            # Rough estimate: assume $5k notional
            qty = int(5000 / signal.entry_price)
        return qty / mkt.adv if mkt.adv > 0 else 0.0

    def _classify_vol_regime(self, mkt: MarketConditions) -> str:
        """Classify volatility regime for routing adjustment."""
        if mkt.vix >= 30 or mkt.volatility >= 0.40:
            return "high"
        if mkt.vix >= 20 or mkt.volatility >= 0.25:
            return "medium"
        return "low"

    @property
    def stats(self) -> dict:
        """Routing statistics summary."""
        return {
            "total_routed": self._total_routed,
            "by_type": dict(self._route_counts),
        }


class FillMonitor:
    """Monitors pending orders and tracks fill quality.

    Migrated from smart_routing.py (V12 dedup).
    """

    def __init__(self):
        self._pending: dict[str, dict] = {}  # order_id -> {signal, submit_time, qty}
        self._fill_stats: dict[str, list] = {}  # strategy -> [slippage_pcts]
        self._lock = threading.Lock()

    def register_order(
        self, order_id: str, signal, submit_time: datetime, qty: int
    ):
        """Register a new pending order for monitoring."""
        self._pending[order_id] = {
            "signal": signal,
            "submit_time": submit_time,
            "qty": qty,
        }
        logger.debug(f"FillMonitor: registered order {order_id} for {signal.symbol}")

    def check_pending(self, now: datetime) -> list[dict]:
        """Check pending orders and return recommended actions.

        Returns list of actions:
        - After CHASE_AFTER_SECONDS:          {action: "chase", order_id, new_price}
        - After CHASE_CONVERT_MARKET_AFTER:   {action: "convert_market", order_id}

        Fail-open: returns empty list on error.
        """
        try:
            actions: list[dict] = []
            for order_id, info in list(self._pending.items()):
                elapsed = (now - info["submit_time"]).total_seconds()
                signal = info["signal"]

                if elapsed >= config.CHASE_CONVERT_MARKET_AFTER:
                    actions.append({
                        "action": "convert_market",
                        "order_id": order_id,
                    })
                elif elapsed >= config.CHASE_AFTER_SECONDS:
                    # Chase: move limit closer to current price (use entry as proxy)
                    if signal.side == "buy":
                        new_price = round(signal.entry_price * 1.001, 2)
                    else:
                        new_price = round(signal.entry_price * 0.999, 2)
                    actions.append({
                        "action": "chase",
                        "order_id": order_id,
                        "new_price": new_price,
                    })

            return actions
        except Exception:
            logger.exception("FillMonitor.check_pending failed — fail-open")
            return []

    def remove_order(self, order_id: str):
        """Remove an order from pending tracking (after fill or cancel)."""
        self._pending.pop(order_id, None)

    def record_fill(
        self, order_id: str, fill_price: float, expected_price: float, strategy: str
    ):
        """Record fill quality for analytics."""
        if expected_price == 0:
            return
        slippage_pct = (fill_price - expected_price) / expected_price
        with self._lock:
            self._fill_stats.setdefault(strategy, []).append(slippage_pct)
        self.remove_order(order_id)
        logger.info(
            f"Fill recorded: order={order_id} strategy={strategy} "
            f"slippage={slippage_pct:.4%}"
        )

    def get_slippage_stats(self) -> dict[str, float]:
        """Return average slippage per strategy."""
        with self._lock:
            result: dict[str, float] = {}
            for strategy, slippages in self._fill_stats.items():
                if slippages:
                    result[strategy] = sum(slippages) / len(slippages)
            return result

    @property
    def pending_count(self) -> int:
        return len(self._pending)
