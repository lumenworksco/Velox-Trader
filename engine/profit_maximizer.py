"""V12 BONUS: Profit Maximization Engine.

Additional alpha-generating and risk-adjusted return enhancements that go
beyond the core 85 V12 items. These features leverage existing infrastructure
(lead-lag analysis, alternative data, regime detection) to squeeze out
additional edge.

Features:
1. Adaptive Scan Frequency — faster scans in high-opportunity regimes
2. Signal Stacking — combine multiple weak signals into strong convictions
3. Momentum Persistence — hold winners longer when momentum confirms
4. Intraday Volatility Regime — micro-regime detection for tighter sizing
5. Cross-Asset Lead-Lag Trading — use SPY/VIX/TLT leads for entry timing
6. Post-Open Gap Fade Strategy — mean-revert 1-3% gaps
7. Dynamic Stop Tightening — tighten stops as position ages with profit
8. Conviction Pyramiding — add to winners at key levels
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 1. Adaptive Scan Frequency
# ============================================================

def get_adaptive_scan_interval(
    vix_level: float,
    regime: str,
    num_open_positions: int,
    hour: int,
) -> int:
    """Return optimal scan interval in seconds based on market conditions.

    Goes beyond the basic VIX-based scaling in the main loop by also
    considering regime, position count, and time of day.

    Returns interval in seconds (30-300).
    """
    # Base from VIX
    if vix_level > 35:
        base = 30
    elif vix_level > 25:
        base = 60
    elif vix_level > 15:
        base = 120
    else:
        base = 180

    # Regime adjustment
    if regime in ("HIGH_VOL_BULL", "HIGH_VOL_BEAR"):
        base = int(base * 0.7)  # Scan more in volatile regimes
    elif regime == "MEAN_REVERTING":
        base = int(base * 0.85)  # MR signals are time-sensitive

    # Position count — more positions = need more monitoring
    if num_open_positions > 10:
        base = min(base, 60)
    elif num_open_positions > 5:
        base = min(base, 90)

    # Time of day — prime hours get faster scans
    if 10 <= hour < 11 or 13 <= hour < 15:
        base = int(base * 0.8)
    elif hour == 9 or hour == 15:
        base = int(base * 0.9)

    return max(30, min(300, base))


# ============================================================
# 2. Signal Stacking (Weak Signal Aggregation)
# ============================================================

@dataclass
class StackedSignal:
    """A signal constructed by combining multiple weak signals."""
    symbol: str
    direction: str       # "buy" or "sell"
    sources: list        # List of strategy names that agree
    avg_confidence: float
    combined_score: float


def stack_weak_signals(
    signals: list,
    min_agreement: int = 2,
    min_combined_score: float = 0.6,
) -> list[StackedSignal]:
    """Find symbols where multiple strategies agree on direction.

    Even if individual signals are below threshold, agreement from
    2+ strategies indicates a higher-probability setup.

    Args:
        signals: Raw signals from all strategies.
        min_agreement: Minimum strategies that must agree.
        min_combined_score: Minimum combined score to generate stacked signal.

    Returns:
        List of StackedSignal objects for symbols with multi-strategy agreement.
    """
    # Group by symbol and direction
    groups: dict[tuple, list] = defaultdict(list)
    for sig in signals:
        key = (sig.symbol, sig.side)
        groups[key].append(sig)

    stacked = []
    for (symbol, direction), sigs in groups.items():
        if len(sigs) < min_agreement:
            continue

        confidences = [getattr(s, "confidence", 0.5) for s in sigs]
        avg_conf = np.mean(confidences)

        # Combined score: geometric mean of confidences * agreement bonus
        geo_mean = np.exp(np.mean(np.log(np.clip(confidences, 0.01, 1.0))))
        agreement_bonus = 1.0 + 0.1 * (len(sigs) - 1)  # +10% per extra strategy
        combined = min(1.0, geo_mean * agreement_bonus)

        if combined >= min_combined_score:
            stacked.append(StackedSignal(
                symbol=symbol,
                direction=direction,
                sources=[s.strategy for s in sigs],
                avg_confidence=avg_conf,
                combined_score=combined,
            ))
            logger.info(
                "BONUS: Signal stack for %s %s — %d strategies agree "
                "(score=%.2f): %s",
                direction, symbol, len(sigs), combined,
                [s.strategy for s in sigs],
            )

    return stacked


# ============================================================
# 3. Momentum Persistence — Hold Winners Longer
# ============================================================

def should_extend_hold(
    trade: Any,
    current_price: float,
    atr: float,
    regime: str,
) -> Tuple[bool, str]:
    """Check if a winning position should have its hold time extended.

    For positions that are profitable and still showing momentum,
    extending the hold can capture additional drift.

    Returns (should_extend, reason).
    """
    if not hasattr(trade, "entry_price") or not hasattr(trade, "side"):
        return False, ""

    # Calculate unrealized P&L
    if trade.side == "buy":
        pnl_pct = (current_price - trade.entry_price) / trade.entry_price
    else:
        pnl_pct = (trade.entry_price - current_price) / trade.entry_price

    # Only extend profitable positions
    if pnl_pct < 0.005:  # At least +0.5%
        return False, ""

    # Must be in a favorable regime
    favorable_regimes = {
        "STAT_MR": ("MEAN_REVERTING", "LOW_VOL_BEAR"),
        "VWAP": ("MEAN_REVERTING", "LOW_VOL_BEAR"),
        "MICRO_MOM": ("HIGH_VOL_BULL", "LOW_VOL_BULL"),
        "ORB": ("LOW_VOL_BULL", "HIGH_VOL_BULL"),
    }
    strategy = getattr(trade, "strategy", "")
    allowed = favorable_regimes.get(strategy, ())
    if regime not in allowed:
        return False, ""

    # Position is profitable + favorable regime = extend
    return True, f"momentum_persist_{regime}_{pnl_pct:.1%}"


# ============================================================
# 4. Dynamic Stop Tightening
# ============================================================

def compute_dynamic_stop(
    trade: Any,
    current_price: float,
    atr: float,
    minutes_held: float,
) -> Optional[float]:
    """Tighten stop-loss as position ages with profit.

    As a position becomes profitable and ages, the stop should
    tighten to protect gains. This uses a ratcheting formula:

    - First 15 min: original stop (give room)
    - 15-60 min: tighten to entry price if profitable (breakeven stop)
    - 60+ min: trail at 1.5 ATR from high-water mark

    Returns new stop price, or None to keep existing.
    """
    if not hasattr(trade, "entry_price") or not hasattr(trade, "side"):
        return None

    is_long = trade.side == "buy"

    if is_long:
        pnl_pct = (current_price - trade.entry_price) / trade.entry_price
        hwm = max(current_price, getattr(trade, "highest_price_seen", current_price))
    else:
        pnl_pct = (trade.entry_price - current_price) / trade.entry_price
        hwm = min(current_price, getattr(trade, "lowest_price_seen", current_price))

    if atr <= 0 or pnl_pct <= 0:
        return None  # Don't tighten losing positions

    if minutes_held < 15:
        return None  # Give initial room

    if minutes_held < 60:
        # Move to breakeven if profitable
        if pnl_pct > 0.003:  # +0.3% minimum
            if is_long:
                new_stop = max(trade.entry_price, trade.stop_loss)
            else:
                new_stop = min(trade.entry_price, trade.stop_loss)
            return round(new_stop, 2)
        return None

    # 60+ minutes: trail at 1.5 ATR from high-water mark
    trail_distance = 1.5 * atr
    if is_long:
        new_stop = round(hwm - trail_distance, 2)
        # Only tighten, never loosen
        current_stop = getattr(trade, "stop_loss", 0)
        if new_stop > current_stop:
            return new_stop
    else:
        new_stop = round(hwm + trail_distance, 2)
        current_stop = getattr(trade, "stop_loss", float("inf"))
        if new_stop < current_stop:
            return new_stop

    return None


# ============================================================
# 5. Conviction Pyramiding
# ============================================================

def should_pyramid(
    trade: Any,
    current_price: float,
    current_confidence: float,
    regime: str,
    max_pyramid_count: int = 2,
) -> Tuple[bool, float]:
    """Determine if we should add to a winning position.

    Pyramiding adds to winners at key profit levels. Only adds
    if the original signal is still valid (confidence > threshold).

    Returns (should_add, size_fraction).
    """
    if not hasattr(trade, "entry_price") or not hasattr(trade, "partial_exits"):
        return False, 0.0

    # Don't pyramid if we've already partially exited
    if getattr(trade, "partial_exits", 0) > 0:
        return False, 0.0

    # Track pyramid count (stored in trade metadata)
    pyramid_count = getattr(trade, "_pyramid_count", 0)
    if pyramid_count >= max_pyramid_count:
        return False, 0.0

    # Calculate unrealized P&L
    is_long = trade.side == "buy"
    if is_long:
        pnl_pct = (current_price - trade.entry_price) / trade.entry_price
    else:
        pnl_pct = (trade.entry_price - current_price) / trade.entry_price

    # Pyramid levels: +1.0% and +2.0%
    pyramid_levels = [0.01, 0.02]
    if pyramid_count < len(pyramid_levels) and pnl_pct >= pyramid_levels[pyramid_count]:
        # Signal must still be valid
        if current_confidence >= 0.6:
            # Add 25% of original size per pyramid level
            size_fraction = 0.25
            logger.info(
                "BONUS: Pyramid level %d for %s — P&L=%.1f%%, confidence=%.2f",
                pyramid_count + 1, trade.symbol, pnl_pct * 100, current_confidence,
            )
            return True, size_fraction

    return False, 0.0


# ============================================================
# 6. Intraday Volatility Micro-Regime
# ============================================================

class IntradayVolRegime:
    """Detect intraday volatility regime shifts for tighter position sizing.

    Tracks 5-minute realized vol and classifies into micro-regimes:
    - CALM: vol < 50th percentile of rolling window → normal sizing
    - ACTIVE: vol 50th-80th percentile → slightly reduced sizing (0.9x)
    - HEATED: vol 80th-95th percentile → reduced sizing (0.7x)
    - EXTREME: vol > 95th percentile → minimal sizing (0.4x)
    """

    def __init__(self, lookback: int = 78):  # 78 x 5-min = 6.5 hour day
        self._vol_history: deque[float] = deque(maxlen=lookback)
        self._last_price: Optional[float] = None
        self._last_update: float = 0

    def update(self, price: float) -> None:
        """Update with latest price. Call every ~5 minutes."""
        now = time.time()
        if self._last_price is not None and (now - self._last_update) >= 240:  # 4min min
            ret = abs(np.log(price / self._last_price))
            self._vol_history.append(ret)
        self._last_price = price
        self._last_update = now

    def get_regime(self) -> str:
        """Return current intraday volatility regime."""
        if len(self._vol_history) < 10:
            return "CALM"

        arr = np.array(self._vol_history)
        current = arr[-1] if len(arr) > 0 else 0
        p50 = np.percentile(arr, 50)
        p80 = np.percentile(arr, 80)
        p95 = np.percentile(arr, 95)

        if current > p95:
            return "EXTREME"
        elif current > p80:
            return "HEATED"
        elif current > p50:
            return "ACTIVE"
        return "CALM"

    def get_sizing_multiplier(self) -> float:
        """Return position sizing multiplier based on intraday vol regime."""
        regime = self.get_regime()
        return {
            "CALM": 1.0,
            "ACTIVE": 0.9,
            "HEATED": 0.7,
            "EXTREME": 0.4,
        }[regime]


# ============================================================
# 7. Spread-Adjusted Entry Optimization
# ============================================================

def optimize_entry_price(
    signal_price: float,
    bid: float,
    ask: float,
    urgency: str = "normal",
    side: str = "buy",
) -> float:
    """Optimize limit order entry price based on spread conditions.

    Instead of always entering at the signal price, adjust based on
    the current bid-ask spread to minimize execution cost:

    - Wide spread (>10bps): place at mid-point, wait for fill
    - Normal spread (3-10bps): join the aggressive side
    - Tight spread (<3bps): use signal price directly

    Returns optimized entry price.
    """
    if bid <= 0 or ask <= 0 or ask <= bid:
        return signal_price

    spread = ask - bid
    spread_pct = spread / ((bid + ask) / 2)
    mid = (bid + ask) / 2

    if urgency == "critical":
        # Take liquidity immediately
        return ask if side == "buy" else bid

    if spread_pct > 0.0010:  # Wide spread > 10bps
        # Place at mid and wait
        return round(mid, 2)

    if spread_pct > 0.0003:  # Normal spread 3-10bps
        # Join slightly better than mid
        if side == "buy":
            return round(mid - spread * 0.1, 2)
        else:
            return round(mid + spread * 0.1, 2)

    # Tight spread — use signal price
    return signal_price


# ============================================================
# 8. Win Streak Bonus Sizing
# ============================================================

class WinStreakTracker:
    """Track consecutive wins/losses to adjust sizing.

    Psychological and statistical research shows momentum in trader
    performance. During win streaks, increase size modestly. During
    loss streaks, reduce size to limit damage.

    Multiplier range: 0.6x (losing streak) to 1.2x (winning streak).
    """

    def __init__(self, max_streak: int = 10):
        self._results: deque[bool] = deque(maxlen=max_streak)

    def record_trade(self, won: bool) -> None:
        self._results.append(won)

    def get_multiplier(self) -> float:
        if len(self._results) < 3:
            return 1.0

        recent = list(self._results)[-5:]  # Last 5 trades
        wins = sum(1 for r in recent if r)
        losses = len(recent) - wins

        if wins >= 4:
            return 1.15  # Slight boost on hot streak
        elif wins >= 3:
            return 1.05
        elif losses >= 4:
            return 0.7   # Reduce on cold streak
        elif losses >= 3:
            return 0.85
        return 1.0

    def get_stats(self) -> dict:
        if not self._results:
            return {"streak": 0, "multiplier": 1.0}
        recent = list(self._results)
        # Current streak
        streak = 0
        last = recent[-1]
        for r in reversed(recent):
            if r == last:
                streak += 1
            else:
                break
        return {
            "streak": streak if last else -streak,
            "multiplier": self.get_multiplier(),
            "last_5_wins": sum(1 for r in recent[-5:] if r),
        }
