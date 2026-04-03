"""V12 shim: smart_routing.py -> execution/smart_router.py.

FillMonitor has been migrated to execution/smart_router.py.
The legacy SmartOrderRouter (with its simpler .route() API) is preserved
here for backward compatibility — it has a different interface than the
production execution.smart_router.SmartOrderRouter.route_order().
"""

import logging
from dataclasses import dataclass
from datetime import datetime

import config
from strategies.base import Signal

# Re-export FillMonitor from its canonical home
from execution.smart_router import FillMonitor  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass
class OrderParams:
    """Parameters decided by the legacy smart router for order submission."""
    order_type: str  # "market", "limit", "ioc"
    limit_price: float | None
    urgency: str  # "high", "medium", "low"
    use_twap: bool
    twap_slices: int
    twap_interval_sec: int


# Strategies considered time-sensitive (speed > price improvement)
_TIME_SENSITIVE = {"ORB", "MICRO_MOM"}

# Strategies considered mean-reversion (patient limit orders)
_MEAN_REVERSION = {"STAT_MR", "VWAP", "KALMAN_PAIRS"}


class SmartOrderRouter:
    """Legacy order router — choose order type and timing based on signal context.

    NOTE: The production router is execution.smart_router.SmartOrderRouter
    which has microstructure-aware routing via route_order(). This class is
    kept for backward compatibility with code that calls .route().
    """

    def route(
        self,
        signal: Signal,
        qty: int,
        spread_pct: float = 0.0,
        equity: float = 100_000.0,
    ) -> OrderParams:
        """Decision tree for order routing."""
        if not config.SMART_ROUTING_ENABLED:
            return OrderParams(
                order_type="market",
                limit_price=None,
                urgency="medium",
                use_twap=False,
                twap_slices=0,
                twap_interval_sec=0,
            )

        # Rule 1: Wide spread -> limit at mid-price
        if spread_pct > config.SPREAD_THRESHOLD_PCT:
            return OrderParams(
                order_type="limit",
                limit_price=round(signal.entry_price, 2),
                urgency="medium",
                use_twap=False,
                twap_slices=0,
                twap_interval_sec=0,
            )

        # Rule 2: Time-sensitive strategies -> IOC market
        if signal.strategy in _TIME_SENSITIVE:
            return OrderParams(
                order_type="ioc",
                limit_price=None,
                urgency="high",
                use_twap=False,
                twap_slices=0,
                twap_interval_sec=0,
            )

        # Rule 3: Large orders -> TWAP
        order_value = qty * signal.entry_price
        if config.ADAPTIVE_TWAP_ENABLED and order_value > equity * 0.03:
            urgency = self._infer_urgency(signal)
            slices, interval = self.compute_adaptive_twap(signal, qty, urgency)
            return OrderParams(
                order_type="market",
                limit_price=None,
                urgency=urgency,
                use_twap=True,
                twap_slices=slices,
                twap_interval_sec=interval,
            )

        # Rule 4: Mean-reversion -> limit at entry price
        if signal.strategy in _MEAN_REVERSION:
            return OrderParams(
                order_type="limit",
                limit_price=round(signal.entry_price, 2),
                urgency="low",
                use_twap=False,
                twap_slices=0,
                twap_interval_sec=0,
            )

        # Rule 5: PEAD -> limit at entry +/- 0.3%
        if signal.strategy == "PEAD":
            offset = 0.003
            if signal.side == "buy":
                price = round(signal.entry_price * (1 + offset), 2)
            else:
                price = round(signal.entry_price * (1 - offset), 2)
            return OrderParams(
                order_type="limit",
                limit_price=price,
                urgency="medium",
                use_twap=False,
                twap_slices=0,
                twap_interval_sec=0,
            )

        # Default: market order
        return OrderParams(
            order_type="market",
            limit_price=None,
            urgency="medium",
            use_twap=False,
            twap_slices=0,
            twap_interval_sec=0,
        )

    def compute_adaptive_twap(
        self, signal: Signal, qty: int, urgency: str
    ) -> tuple[int, int]:
        """Compute (n_slices, interval_sec) based on urgency."""
        table = {
            "high": (3, 15),
            "medium": (5, 30),
            "low": (8, 60),
        }
        return table.get(urgency, (5, 30))

    @staticmethod
    def _infer_urgency(signal: Signal) -> str:
        """Infer urgency from the signal's strategy."""
        if signal.strategy in _TIME_SENSITIVE:
            return "high"
        if signal.strategy in _MEAN_REVERSION:
            return "low"
        return "medium"
