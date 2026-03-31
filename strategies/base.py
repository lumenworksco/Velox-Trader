"""Shared dataclasses, types, and abstract base for all strategies (ARCH-005).

Signal is the core data structure passed between strategy -> signal_processor -> execution.
Strategy is the abstract interface that all strategies should implement.

Backward compatibility: Signal retains all original fields with their defaults.
New fields (confidence, metadata, timestamp, pair_symbol) are optional and default
to safe values so existing strategy code continues to work without changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================
# Signal — the core output of every strategy scan
# ============================================================

@dataclass
class Signal:
    """A trading signal emitted by a strategy.

    Required fields (positional):
        symbol, strategy, side, entry_price, take_profit, stop_loss

    Optional fields (backward-compatible defaults):
        reason       — human-readable explanation (used in logs/dashboard)
        hold_type    — "day" or "swing" (controls EOD handling, overnight logic)
        pair_id      — links two legs of a pairs trade for atomic execution

    New fields (ARCH-005):
        confidence   — strategy conviction 0.0-1.0 (used by signal ranker)
        metadata     — arbitrary strategy-specific data (indicators, z-scores, etc.)
        timestamp    — when the signal was generated
        pair_symbol  — the other symbol in a pairs trade (informational)
    """
    symbol: str
    strategy: str          # "STAT_MR", "VWAP", "KALMAN_PAIRS", "ORB", "MICRO_MOM", "PEAD"
    side: str              # "buy" or "sell"
    entry_price: float
    take_profit: float
    stop_loss: float
    reason: str = ""
    hold_type: str = "day"  # "day" or "swing" (multi-day)
    pair_id: str = ""       # Links two legs of a pairs trade
    confidence: float = 0.5  # 0.0 to 1.0 — default 0.5 for backward compat
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    pair_symbol: Optional[str] = None

    def __post_init__(self):
        """V11.3: Validate signal fields to catch misconfigured strategies early."""
        self.confidence = max(0.0, min(1.0, self.confidence))
        if self.entry_price <= 0:
            raise ValueError(f"Signal {self.symbol}: entry_price must be > 0, got {self.entry_price}")
        if self.stop_loss <= 0 or self.take_profit <= 0:
            raise ValueError(f"Signal {self.symbol}: stop_loss and take_profit must be > 0")


# ============================================================
# ExitParams — returned by strategy exit checks
# ============================================================

@dataclass
class ExitParams:
    """Parameters for a position exit decision.

    Strategies return this from get_exit_params() to communicate exit
    decisions to the exit processor. Supports full exits, partial exits,
    and stop/target adjustments (trailing stops, breakeven moves).
    """
    should_exit: bool = False
    exit_reason: str = ""
    partial_exit_pct: float = 0.0        # 0.0 = full exit, 0.0-1.0 = partial
    new_stop_loss: Optional[float] = None
    new_take_profit: Optional[float] = None


# ============================================================
# StrategyMetadata — describes a strategy's characteristics
# ============================================================

@dataclass
class StrategyMetadata:
    """Static metadata describing a strategy's characteristics.

    Used by the adaptive allocator, signal ranker, and dashboard
    to understand each strategy's properties without hard-coding.
    """
    name: str                    # e.g. "STAT_MR"
    version: str                 # e.g. "10.0"
    description: str             # Human-readable description
    default_allocation: float    # Default portfolio allocation (0.0-1.0)
    min_bars_required: int       # Minimum bars needed before generating signals
    supported_sides: List[str]   # ["long"], ["short"], or ["long", "short"]
    timeframe: str               # e.g. "2min", "5min", "1d"
    max_positions: int           # Max concurrent positions for this strategy


# ============================================================
# Strategy — abstract base class (ARCH-005)
# ============================================================

class Strategy(ABC):
    """Abstract interface for all trading strategies.

    Lifecycle (called by engine/scanner.py and main.py):
        1. prepare_universe(date) — filter/select tradeable symbols (daily)
        2. generate_signals(bars)  — produce entry signals from market data
        3. get_exit_params(trade, price, bars) — check open positions for exits
        4. reset_daily()           — clear per-day state at start of each day
        5. get_metadata()          — return static strategy characteristics

    Note: Existing strategies do NOT yet inherit from this ABC. This class
    defines the target interface for incremental migration. Strategies can
    be migrated one at a time by inheriting from Strategy and implementing
    the abstract methods while keeping their existing public API.
    """

    @abstractmethod
    def prepare_universe(self, date) -> List[str]:
        """Select the tradeable universe for the given date.

        Called once per day (typically at market open or pre-market).
        Returns a list of symbols that this strategy will scan.
        """
        ...

    @abstractmethod
    def generate_signals(self, bars: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate entry signals from current market data.

        Called every scan cycle (typically every 120 seconds).
        Returns a list of Signal objects for the signal processor.

        Args:
            bars: Dict mapping symbol -> DataFrame of recent OHLCV bars
        """
        ...

    @abstractmethod
    def get_exit_params(
        self, trade, current_price: float, bars: Optional[pd.DataFrame] = None
    ) -> ExitParams:
        """Check if an open position should be exited or adjusted.

        Called every scan cycle for each open position owned by this strategy.

        Args:
            trade: TradeRecord from risk manager (has .symbol, .side, .entry_price, etc.)
            current_price: Latest price of the position's symbol
            bars: Optional recent bars for the symbol (for indicator-based exits)

        Returns:
            ExitParams with exit decision and optional stop/target adjustments
        """
        ...

    @abstractmethod
    def reset_daily(self):
        """Clear all per-day state for a fresh trading day.

        Called at the start of each trading day (before market open).
        Should reset universe, triggered symbols, daily counters, etc.
        Persistent state (e.g. Kalman filter, weekly pairs) should NOT be reset.
        """
        ...

    @abstractmethod
    def get_metadata(self) -> StrategyMetadata:
        """Return static metadata about this strategy.

        Used by adaptive allocator, dashboard, and signal ranker.
        Should return the same value every time (immutable characteristics).
        """
        ...


# ============================================================
# Legacy dataclasses — kept for backward compatibility
# ============================================================

@dataclass
class ORBRange:
    high: float
    low: float
    volume: float          # total volume during ORB period
    prev_close: float      # yesterday's close for gap check


@dataclass
class VWAPState:
    vwap: float = 0.0
    upper_band: float = 0.0
    lower_band: float = 0.0
    cumulative_volume: float = 0.0
    cumulative_vp: float = 0.0    # volume * price
    cumulative_vp2: float = 0.0   # volume * price^2
