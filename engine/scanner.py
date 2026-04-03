"""V10 Engine — Strategy scanning and signal generation.

Extracts the multi-strategy scanning loop from main.py into a reusable module.
Handles scanning all active strategies, ranking signals, and processing them.

PROD-001: Added prefetch_bars() for concurrent bar fetching across all symbols
before strategies run their scans. Strategies can optionally accept a `bars_cache`
parameter to avoid redundant API calls.

ARCH-011: Added StrategyRegistry for data-driven strategy dispatch (replaces
hard-coded if/elif chains).

T5-015: Added ScanScheduler with tabular RL policy for adaptive scan intervals.
"""

import logging
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.failure_modes import FailureMode, handle_failure
from risk.risk_manager import is_strategy_in_time_window

import pandas as pd
from alpaca.data.timeframe import TimeFrame

from strategies.base import Signal
from engine.signal_processor import process_signals
from engine.exit_processor import handle_strategy_exits, get_current_prices

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ARCH-011: Strategy Registry
# ---------------------------------------------------------------------------

class StrategyRegistry:
    """Registry mapping strategy names to their scan/exit callables.

    Instead of hard-coded if/elif chains in scan_all_strategies(), callers
    register strategy instances here.  The scanner iterates the registry
    to collect signals.

    Usage::

        registry = StrategyRegistry()
        registry.register("STAT_MR", stat_mr, required=True)
        registry.register("ORB", orb_strategy, required=False)
        signals = registry.scan_all(current, regime)
    """

    def __init__(self) -> None:
        self._strategies: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        strategy,
        required: bool = False,
        scan_kwargs_fn: Optional[Callable] = None,
    ) -> None:
        """Register a strategy instance.

        Args:
            name: Unique strategy name (e.g. "STAT_MR", "VWAP").
            strategy: Strategy instance with a `scan()` method.
            required: If True, log error on scan failure; otherwise warning.
            scan_kwargs_fn: Optional callable returning extra kwargs for scan().
                            Called with (current, regime) and merged into scan() args.
        """
        if strategy is None:
            return
        self._strategies[name] = {
            "instance": strategy,
            "required": required,
            "scan_kwargs_fn": scan_kwargs_fn,
        }

    def unregister(self, name: str) -> None:
        """Remove a strategy from the registry."""
        self._strategies.pop(name, None)

    @property
    def names(self) -> List[str]:
        """Return registered strategy names."""
        return list(self._strategies.keys())

    def get(self, name: str):
        """Return the strategy instance by name, or None."""
        entry = self._strategies.get(name)
        return entry["instance"] if entry else None

    def scan_all(
        self,
        current: datetime,
        regime: str,
        signal_ranker=None,
        extra_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Signal]:
        """Scan all registered strategies and return combined signals.

        Each strategy is scanned inside its own try/except (fail-open).
        Optional signal_ranker is applied after collection.

        Args:
            current: Current datetime.
            regime: Current market regime string.
            signal_ranker: Optional ranker with .rank(signals, regime=...).
            extra_kwargs: Per-strategy extra kwargs, keyed by strategy name.

        Returns:
            Combined (optionally ranked) list of Signal objects.
        """
        all_signals: List[Signal] = []
        counts: Dict[str, int] = {}
        extra_kwargs = extra_kwargs or {}

        for name, entry in self._strategies.items():
            # V12 4.2: Skip strategies outside their optimal time window
            if not is_strategy_in_time_window(name, current):
                logger.debug("V12 4.2: %s skipped — outside time window at %s", name, current.time())
                counts[name] = 0
                continue

            strategy = entry["instance"]
            try:
                # Build scan kwargs
                kwargs: Dict[str, Any] = {"regime": regime}
                if entry["scan_kwargs_fn"]:
                    kwargs.update(entry["scan_kwargs_fn"](current, regime))
                if name in extra_kwargs:
                    kwargs.update(extra_kwargs[name])

                sigs = strategy.scan(current, **kwargs)
                all_signals.extend(sigs)
                counts[name] = len(sigs)
            except Exception as e:
                level = logging.ERROR if entry["required"] else logging.WARNING
                logger.log(level, "%s scan failed: %s", name, e)
                counts[name] = 0

        # Rank if available
        if signal_ranker and all_signals:
            try:
                all_signals = signal_ranker.rank(all_signals, regime=regime)
            except Exception as exc:
                handle_failure(FailureMode.DEGRADE_GRACEFULLY,
                               "scanner.registry_signal_ranking", exc)

        parts = " ".join(f"{k}={v}" for k, v in counts.items())
        logger.info(
            "Registry scan complete: %d signals (%s) regime=%s",
            len(all_signals), parts, regime,
        )
        return all_signals

    def __len__(self) -> int:
        return len(self._strategies)

    def __contains__(self, name: str) -> bool:
        return name in self._strategies

    def __repr__(self) -> str:
        return f"StrategyRegistry({list(self._strategies.keys())})"

# ---------------------------------------------------------------------------
# PROD-001: Concurrent bar prefetching
# ---------------------------------------------------------------------------

# Module-level cache for prefetched bars (cleared each scan cycle)
_bars_cache: dict[str, pd.DataFrame] = {}
_cache_timestamp: Optional[datetime] = None


def prefetch_bars(
    symbols: list[str],
    timeframe: TimeFrame,
    start: datetime,
    end: Optional[datetime] = None,
    max_workers: int = 8,
    cache_ttl_sec: float = 60.0,
) -> dict[str, pd.DataFrame]:
    """PROD-001: Pre-fetch bars for all symbols concurrently using ThreadPoolExecutor.

    Called at the top of each scan cycle so that individual strategies can
    read from the cache instead of making sequential API calls.

    Args:
        symbols: List of ticker symbols to fetch.
        timeframe: Alpaca TimeFrame (e.g., 1-min, 5-min, daily).
        start: Start datetime for bars.
        end: Optional end datetime.
        max_workers: Max concurrent fetch threads (default 8).
        cache_ttl_sec: How long cached bars remain valid (default 60s).

    Returns:
        Dict mapping symbol -> DataFrame of bars.
    """
    global _bars_cache, _cache_timestamp
    from data.fetcher import get_bars

    # Return cached data if still fresh
    now = datetime.now()
    if (_cache_timestamp and (now - _cache_timestamp).total_seconds() < cache_ttl_sec
            and all(sym in _bars_cache for sym in symbols)):
        logger.debug("PROD-001: Returning cached bars for %d symbols", len(symbols))
        return {sym: _bars_cache.get(sym, pd.DataFrame()) for sym in symbols}

    results: dict[str, pd.DataFrame] = {}

    def _fetch_one(sym: str) -> tuple[str, pd.DataFrame]:
        return sym, get_bars(sym, timeframe, start=start, end=end)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                sym, df = future.result()
                results[sym] = df
            except Exception as e:
                logger.warning("PROD-001: Failed to prefetch bars for %s: %s", sym, e)
                results[sym] = pd.DataFrame()

    # Update module-level cache
    _bars_cache.update(results)
    _cache_timestamp = now
    logger.info(
        "PROD-001: Prefetched bars for %d/%d symbols (max_workers=%d)",
        sum(1 for df in results.values() if not df.empty), len(symbols), max_workers,
    )
    return results


def get_cached_bars(symbol: str) -> Optional[pd.DataFrame]:
    """Retrieve bars from the prefetch cache for a single symbol.

    Returns None if not cached (caller should fall back to direct fetch).
    """
    return _bars_cache.get(symbol)


def clear_bars_cache():
    """Clear the prefetched bars cache (call at end of scan cycle or EOD)."""
    global _bars_cache, _cache_timestamp
    _bars_cache.clear()
    _cache_timestamp = None


def scan_all_strategies(
    current: datetime,
    regime: str,
    stat_mr,
    kalman_pairs,
    micro_mom,
    vwap_strategy,
    orb_strategy=None,
    pead_strategy=None,
    copula_pairs=None,
    cross_sectional_momentum=None,
    sector_momentum=None,
    multi_timeframe=None,
    signal_ranker=None,
    day_pnl_pct: float = 0.0,
) -> list[Signal]:
    """Scan all active strategies and return ranked signals.

    Each strategy scan is individually wrapped in try/except (fail-open).
    Returns a combined, optionally ranked signal list.
    """
    signals: list[Signal] = []
    mr_signals = []
    vwap_signals = []
    pair_signals = []
    orb_signals = []
    micro_signals = []
    pead_signals = []

    # Detect micro momentum events first
    try:
        micro_mom.detect_event(current)
    except Exception as e:
        logger.error(f"Micro event detection failed: {e}")

    # Scan each strategy (V12 4.2: skip if outside optimal time window)
    if is_strategy_in_time_window("STAT_MR", current):
        try:
            mr_signals = stat_mr.scan(current, regime)
            signals.extend(mr_signals)
        except Exception as e:
            logger.error(f"StatMR scan failed: {e}")
    else:
        logger.debug("V12 4.2: STAT_MR skipped — outside time window at %s", current.time())

    if is_strategy_in_time_window("VWAP", current):
        try:
            vwap_signals = vwap_strategy.scan(current, regime)
            signals.extend(vwap_signals)
        except Exception as e:
            logger.error(f"VWAP scan failed: {e}")
    else:
        logger.debug("V12 4.2: VWAP skipped — outside time window at %s", current.time())

    if is_strategy_in_time_window("KALMAN_PAIRS", current):
        try:
            pair_signals = kalman_pairs.scan(current, regime)
            signals.extend(pair_signals)
        except Exception as e:
            logger.error(f"KalmanPairs scan failed: {e}")
    else:
        logger.debug("V12 4.2: KALMAN_PAIRS skipped — outside time window at %s", current.time())

    if orb_strategy and is_strategy_in_time_window("ORB", current):
        try:
            orb_signals = orb_strategy.scan(current, regime)
            signals.extend(orb_signals)
        except Exception as e:
            logger.error(f"ORB scan failed: {e}")
    elif orb_strategy:
        logger.debug("V12 4.2: ORB skipped — outside time window at %s", current.time())

    if is_strategy_in_time_window("MICRO_MOM", current):
        try:
            micro_signals = micro_mom.scan(
                current, day_pnl_pct=day_pnl_pct, regime=regime
            )
            signals.extend(micro_signals)
        except Exception as e:
            logger.error(f"MicroMom scan failed: {e}")
    else:
        logger.debug("V12 4.2: MICRO_MOM skipped — outside time window at %s", current.time())

    if pead_strategy and is_strategy_in_time_window("PEAD", current):
        try:
            pead_signals = pead_strategy.scan(current)
            signals.extend(pead_signals)
        except Exception as e:
            logger.error(f"PEAD scan failed: {e}")
    elif pead_strategy:
        logger.debug("V12 4.2: PEAD skipped — outside time window at %s", current.time())

    # --- New V10 strategies ---
    copula_signals = []
    csm_signals = []
    sector_mom_signals = []
    mtf_signals = []

    if copula_pairs:
        try:
            copula_signals = copula_pairs.scan(current, regime)
            signals.extend(copula_signals)
        except Exception as e:
            logger.error(f"CopulaPairs scan failed: {e}")

    if cross_sectional_momentum:
        try:
            csm_signals = cross_sectional_momentum.scan(current, regime)
            signals.extend(csm_signals)
        except Exception as e:
            logger.error(f"CrossSectionalMomentum scan failed: {e}")

    if sector_momentum:
        try:
            sector_mom_signals = sector_momentum.scan(current, regime)
            signals.extend(sector_mom_signals)
        except Exception as e:
            logger.error(f"SectorMomentum scan failed: {e}")

    if multi_timeframe:
        try:
            mtf_signals = multi_timeframe.scan(current, regime)
            signals.extend(mtf_signals)
        except Exception as e:
            logger.error(f"MultiTimeframe scan failed: {e}")

    # Rank signals by expected value
    if signal_ranker and signals:
        try:
            signals = signal_ranker.rank(signals, regime=regime)
        except Exception as exc:
            handle_failure(FailureMode.DEGRADE_GRACEFULLY,
                           "scanner.signal_ranking", exc)

    # Log counts
    logger.info(
        f"Scan complete: {len(signals)} signals "
        f"(MR={len(mr_signals)} VWAP={len(vwap_signals)} PAIRS={len(pair_signals)} "
        f"ORB={len(orb_signals)} MICRO={len(micro_signals)} PEAD={len(pead_signals)} "
        f"COPULA={len(copula_signals)} CSM={len(csm_signals)} "
        f"SECTMOM={len(sector_mom_signals)} MTF={len(mtf_signals)}) "
        f"regime={regime}"
    )

    return signals


# =============================================================================
# T2-001: Async scan — fan out per-strategy scans concurrently
# =============================================================================

import asyncio


async def _async_scan_strategy(name: str, strategy, current: datetime, regime: str,
                               extra_kwargs: dict | None = None) -> tuple[str, list[Signal]]:
    """Run a single strategy scan in the async event loop.

    The actual strategy.scan() is synchronous, so we run it in the default
    executor (thread pool) to avoid blocking the event loop.
    """
    loop = asyncio.get_running_loop()
    kwargs = {"regime": regime}
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    def _scan():
        return strategy.scan(current, **kwargs)

    try:
        sigs = await loop.run_in_executor(None, _scan)
        return name, sigs
    except Exception as e:
        logger.error(f"Async {name} scan failed: {e}")
        return name, []


async def _async_scan_all_strategies_impl(
    current: datetime,
    regime: str,
    strategy_map: dict,
    signal_ranker=None,
    extra_kwargs: dict | None = None,
) -> list[Signal]:
    """T2-001: Internal async implementation — fans out strategy scans via asyncio.gather().

    Args:
        current: Current datetime.
        regime: Market regime string.
        strategy_map: Dict of {name: strategy_instance} (only non-None strategies).
        signal_ranker: Optional signal ranker.
        extra_kwargs: Per-strategy extra kwargs dict keyed by name.

    Returns:
        Combined (optionally ranked) signal list.
    """
    extra_kwargs = extra_kwargs or {}

    tasks = [
        _async_scan_strategy(name, strat, current, regime, extra_kwargs.get(name))
        for name, strat in strategy_map.items()
        if strat is not None
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_signals: list[Signal] = []
    counts: dict[str, int] = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Async scan task raised: {result}")
            continue
        name, sigs = result
        all_signals.extend(sigs)
        counts[name] = len(sigs)

    if signal_ranker and all_signals:
        try:
            all_signals = signal_ranker.rank(all_signals, regime=regime)
        except Exception:
            pass

    parts = " ".join(f"{k}={v}" for k, v in counts.items())
    logger.info(
        "Async scan complete: %d signals (%s) regime=%s",
        len(all_signals), parts, regime,
    )
    return all_signals


def async_scan_all_strategies(
    current: datetime,
    regime: str,
    strategy_map: dict,
    signal_ranker=None,
    extra_kwargs: dict | None = None,
) -> list[Signal]:
    """T2-001: Public sync-compatible wrapper — runs async scan in a dedicated thread.

    This keeps the main loop synchronous while the scans fan out concurrently.
    Uses asyncio.run() in a dedicated thread to avoid event loop conflicts.

    Args:
        current: Current datetime.
        regime: Market regime string.
        strategy_map: Dict of {name: strategy_instance} (only non-None).
        signal_ranker: Optional signal ranker.
        extra_kwargs: Per-strategy extra kwargs dict.

    Returns:
        Combined signal list.
    """
    import concurrent.futures

    def _run_async():
        return asyncio.run(_async_scan_all_strategies_impl(
            current, regime, strategy_map, signal_ranker, extra_kwargs,
        ))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_async)
        return future.result(timeout=120)  # 2 minute hard timeout


def check_all_exits(
    current: datetime,
    risk,
    stat_mr,
    kalman_pairs,
    micro_mom,
    orb_strategy=None,
    pead_strategy=None,
    ws_monitor=None,
):
    """Check all strategies for exit signals and process them."""
    for name, strategy in [
        ("StatMR", stat_mr),
        ("KalmanPairs", kalman_pairs),
        ("MicroMom", micro_mom),
        ("ORB", orb_strategy),
        ("PEAD", pead_strategy),
    ]:
        if strategy is None:
            continue
        try:
            exits = strategy.check_exits(risk.open_trades, current)
            if exits:
                handle_strategy_exits(exits, risk, current, ws_monitor)
        except Exception as e:
            logger.error(f"{name} exit check failed: {e}")


def run_beta_neutralization(
    current: datetime,
    risk,
    beta_neutral,
    vol_engine,
    pnl_lock,
    ws_monitor=None,
    regime: str = "UNKNOWN",
):
    """Check and apply beta neutralization if needed."""
    if not beta_neutral.should_check_now(current):
        return

    try:
        prices = get_current_prices(risk.open_trades)
        beta_neutral.compute_portfolio_beta(risk.open_trades, prices)

        if beta_neutral.needs_hedge():
            from data import get_snapshot
            try:
                spy_snap = get_snapshot("SPY")
                spy_price = float(spy_snap.latest_trade.price) if spy_snap and spy_snap.latest_trade else 0
            except Exception:
                spy_price = 0

            if spy_price > 0:
                hedge_signal = beta_neutral.compute_hedge_signal(
                    risk.current_equity, spy_price
                )
                if hedge_signal:
                    process_signals(
                        [hedge_signal], risk, regime, current,
                        vol_engine, pnl_lock, ws_monitor,
                    )
    except Exception as e:
        logger.error(f"Beta neutralization failed: {e}")


# ---------------------------------------------------------------------------
# T5-015: Adaptive RL-Informed Scan Scheduling
# ---------------------------------------------------------------------------

class ScanScheduler:
    """Tabular RL-based scan interval scheduler.

    State features: VIX level bucket, time-of-day bucket, avg VPIN,
    open positions count, recent signal quality.

    Action space: scan interval in {15s, 30s, 60s, 120s, 180s}.

    Reward: signal_capture_rate - scan_cost_penalty.

    Gated behind ``ADAPTIVE_SCAN_ENABLED`` config flag.
    Falls back to VIX-based scaling when disabled.

    Usage::

        scheduler = ScanScheduler()
        interval = scheduler.get_interval(vix=22, time_bucket=3,
                                           avg_vpin=0.45, open_positions=3,
                                           signal_quality=0.6)
        # Returns one of [15, 30, 60, 120, 180]
    """

    ACTIONS = [15, 30, 60, 120, 180]  # Scan intervals in seconds

    # State discretization
    VIX_BUCKETS = [0, 15, 20, 25, 30, 40, 100]  # 6 buckets
    TIME_BUCKETS = 8  # 3.25 hours of trading / 8 = ~24 min per bucket
    VPIN_BUCKETS = [0.0, 0.3, 0.5, 0.7, 1.0]  # 4 buckets
    POS_BUCKETS = [0, 1, 3, 5, 10, 50]  # 5 buckets
    QUALITY_BUCKETS = [0.0, 0.3, 0.5, 0.7, 1.0]  # 4 buckets

    # RL parameters
    LEARNING_RATE = 0.1
    DISCOUNT = 0.95
    EPSILON = 0.1  # Exploration rate
    SCAN_COST_PER_SEC = 0.001  # Small penalty for frequent scanning

    def __init__(self):
        import numpy as _np
        self._np = _np

        # Q-table: state_key -> {action_idx: q_value}
        self._q_table: Dict[str, Dict[int, float]] = {}
        self._lock = threading.Lock()

        # Tracking for reward computation
        self._last_state: Optional[str] = None
        self._last_action: Optional[int] = None
        self._signals_before: int = 0
        self._scans_since_signal: int = 0

        self._enabled = False
        try:
            import config as _cfg
            self._enabled = getattr(_cfg, "ADAPTIVE_SCAN_ENABLED", False)
        except Exception:
            pass

        if self._enabled:
            logger.info("T5-015: ScanScheduler initialized with tabular RL policy")

    # ------------------------------------------------------------------
    # State discretization
    # ------------------------------------------------------------------

    @staticmethod
    def _discretize(value: float, buckets: list) -> int:
        """Map a continuous value to a bucket index."""
        for i in range(len(buckets) - 1):
            if value < buckets[i + 1]:
                return i
        return len(buckets) - 2

    def _get_state_key(
        self,
        vix: float,
        time_bucket: int,
        avg_vpin: float,
        open_positions: int,
        signal_quality: float,
    ) -> str:
        """Convert continuous state to a discrete key for Q-table lookup."""
        vix_b = self._discretize(vix, self.VIX_BUCKETS)
        time_b = min(time_bucket, self.TIME_BUCKETS - 1)
        vpin_b = self._discretize(avg_vpin, self.VPIN_BUCKETS)
        pos_b = self._discretize(float(open_positions), self.POS_BUCKETS)
        qual_b = self._discretize(signal_quality, self.QUALITY_BUCKETS)
        return f"{vix_b}_{time_b}_{vpin_b}_{pos_b}_{qual_b}"

    def _get_time_bucket(self, now: Optional[datetime] = None) -> int:
        """Convert current time to a time-of-day bucket (0-7)."""
        if now is None:
            try:
                import config as _cfg
                now = datetime.now(_cfg.ET)
            except Exception:
                return 4  # Default to mid-day
        # Trading hours: 9:30 to 16:00 = 6.5 hours = 390 minutes
        minutes_since_open = (now.hour - 9) * 60 + now.minute - 30
        minutes_since_open = max(0, min(390, minutes_since_open))
        return min(self.TIME_BUCKETS - 1, int(minutes_since_open / (390 / self.TIME_BUCKETS)))

    # ------------------------------------------------------------------
    # Action selection (epsilon-greedy)
    # ------------------------------------------------------------------

    def _select_action(self, state_key: str) -> int:
        """Select action using epsilon-greedy policy."""
        np = self._np

        # Exploration
        if np.random.random() < self.EPSILON:
            return int(np.random.randint(len(self.ACTIONS)))

        # Exploitation
        with self._lock:
            q_values = self._q_table.get(state_key, {})

        if not q_values:
            # Default: action index 2 = 60 seconds (moderate)
            return 2

        best_action = max(q_values, key=lambda a: q_values[a])
        return best_action

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_interval(
        self,
        vix: float = 20.0,
        time_bucket: int | None = None,
        avg_vpin: float = 0.5,
        open_positions: int = 0,
        signal_quality: float = 0.5,
        now: datetime | None = None,
    ) -> int:
        """Get the recommended scan interval in seconds.

        Args:
            vix: Current VIX level.
            time_bucket: Time-of-day bucket (0-7). Auto-computed if None.
            avg_vpin: Average VPIN across tracked symbols.
            open_positions: Number of open positions.
            signal_quality: Recent signal capture quality (0-1).
            now: Current datetime for time bucket computation.

        Returns:
            Scan interval in seconds (15, 30, 60, 120, or 180).
        """
        if not self._enabled:
            return self._vix_fallback(vix)

        try:
            if time_bucket is None:
                time_bucket = self._get_time_bucket(now)

            state_key = self._get_state_key(
                vix, time_bucket, avg_vpin, open_positions, signal_quality
            )

            action_idx = self._select_action(state_key)
            interval = self.ACTIONS[action_idx]

            # Store for reward update
            self._last_state = state_key
            self._last_action = action_idx

            return interval

        except Exception as e:
            logger.debug("T5-015: ScanScheduler failed (fall-back to VIX): %s", e)
            return self._vix_fallback(vix)

    def update_reward(self, signals_found: int, scans_taken: int = 1):
        """Update Q-value based on the outcome of the last scan cycle.

        Args:
            signals_found: Number of actionable signals from this scan.
            scans_taken: Number of scans in this cycle (usually 1).
        """
        if not self._enabled or self._last_state is None or self._last_action is None:
            return

        try:
            # Reward: signal capture - scan cost
            signal_reward = min(1.0, signals_found * 0.5)  # Cap at 1.0
            scan_cost = scans_taken * self.SCAN_COST_PER_SEC * self.ACTIONS[self._last_action]
            reward = signal_reward - scan_cost

            state = self._last_state
            action = self._last_action

            with self._lock:
                if state not in self._q_table:
                    self._q_table[state] = {i: 0.0 for i in range(len(self.ACTIONS))}

                old_q = self._q_table[state].get(action, 0.0)
                # Simple Q-learning update (no next state — immediate reward)
                new_q = old_q + self.LEARNING_RATE * (reward - old_q)
                self._q_table[state][action] = new_q

        except Exception as e:
            logger.debug("T5-015: Reward update failed: %s", e)

    @staticmethod
    def _vix_fallback(vix: float) -> int:
        """VIX-based fallback (same as T4-006)."""
        if vix < 15:
            return 180
        elif vix < 25:
            return 120
        elif vix < 35:
            return 60
        else:
            return 30

    def get_q_table_stats(self) -> Dict:
        """Return Q-table statistics for monitoring."""
        with self._lock:
            n_states = len(self._q_table)
            if n_states == 0:
                return {"states": 0, "avg_q": 0.0}
            all_q = [q for state_qs in self._q_table.values() for q in state_qs.values()]
            return {
                "states": n_states,
                "avg_q": float(self._np.mean(all_q)) if all_q else 0.0,
                "max_q": float(max(all_q)) if all_q else 0.0,
                "min_q": float(min(all_q)) if all_q else 0.0,
            }

    def save_policy(self, path: str):
        """Save the Q-table to a JSON file for persistence."""
        import json
        with self._lock:
            data = {k: v for k, v in self._q_table.items()}
        try:
            with open(path, "w") as f:
                json.dump(data, f)
            logger.info("T5-015: Q-table saved (%d states) to %s", len(data), path)
        except Exception as e:
            logger.error("T5-015: Failed to save Q-table: %s", e)

    def load_policy(self, path: str):
        """Load a Q-table from a JSON file."""
        import json
        try:
            with open(path, "r") as f:
                data = json.load(f)
            with self._lock:
                self._q_table = {k: {int(ak): av for ak, av in v.items()} for k, v in data.items()}
            logger.info("T5-015: Q-table loaded (%d states) from %s", len(self._q_table), path)
        except FileNotFoundError:
            logger.info("T5-015: No saved Q-table found at %s, starting fresh", path)
        except Exception as e:
            logger.error("T5-015: Failed to load Q-table: %s", e)


# Module-level singleton
_scan_scheduler: Optional[ScanScheduler] = None
_scheduler_lock = threading.Lock()


def get_scan_scheduler() -> ScanScheduler:
    """Get or create the global ScanScheduler singleton."""
    global _scan_scheduler
    if _scan_scheduler is None:
        with _scheduler_lock:
            if _scan_scheduler is None:
                _scan_scheduler = ScanScheduler()
    return _scan_scheduler
