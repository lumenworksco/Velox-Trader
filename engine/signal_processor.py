"""V10 Engine — Signal processing pipeline: filtering, sizing, and order submission.

Integrates:
- OMS order tracking (order lifecycle, idempotency, audit trail)
- Event bus (signal.generated, order.submitted, position.opened events)
- Transaction cost model (reject negative-EV trades — PROFIT-GAP-001)
"""

import logging
import threading
import time as _time
from datetime import datetime, timedelta

import config
import database
from strategies.base import Signal
from risk import RiskManager, TradeRecord, VolatilityTargetingRiskEngine, DailyPnLLock
from execution import submit_bracket_order, close_position, can_short
from earnings import has_earnings_soon
from correlation import is_too_correlated

from engine.event_log import log_event, EventType
from engine.failure_modes import FailureMode, handle_failure

logger = logging.getLogger(__name__)

# Lazy-loaded optional modules
try:
    from analytics.intraday_seasonality import IntradaySeasonality
except ImportError:
    IntradaySeasonality = None

# WIRE-001: Feature store integration (fail-open)
try:
    from data.feature_store import get_feature_store as _get_feature_store
    _FEATURE_STORE_AVAILABLE = True
except ImportError:
    _FEATURE_STORE_AVAILABLE = False

# =============================================================================
# T2-004: Unified SignalProcessorState — single RLock replaces 5 fine-grained
# locks (_ml_model_lock, _vpin_lock, _seasonality_lock, _pair_lock,
# _asset_status_lock) to eliminate deadlock risk from inconsistent lock ordering.
# =============================================================================

from dataclasses import dataclass as _dataclass, field as _field


@_dataclass
class SignalProcessorState:
    """T2-004: Consolidated mutable state for the signal processor.

    All mutable state that was previously guarded by separate locks is
    collected here and protected by a single RLock.
    """
    lock: threading.RLock = _field(default_factory=threading.RLock, repr=False)

    # ML model (WIRE-002)
    ml_model: object = None
    ml_model_load_attempted: bool = False

    # VPIN instances (WIRE-003)
    vpin_instances: dict = _field(default_factory=dict)  # symbol -> VPIN

    # Seasonality singleton (BUG-042)
    seasonality_instance: object = None

    # T5-004: FinBERT NLP sentiment (fail-open)
    finbert_sentiment: object = None
    finbert_load_attempted: bool = False

    # Halted / asset status blocklists (BUG-024, RISK-005)
    halted_symbols: dict = _field(default_factory=dict)        # symbol -> expiry_ts
    asset_status_blocklist: dict = _field(default_factory=dict) # symbol -> expiry_ts
    last_asset_status_refresh: float = 0.0
    last_successful_asset_refresh: float = 0.0

    # Stop-loss re-entry cooldown: symbol -> cooldown_expiry_datetime
    stopout_cooldowns: dict = _field(default_factory=dict)


# Module-level singleton
_state = SignalProcessorState()


# WIRE-002 + V11.3 T9: ML model integration via BatchInferenceEngine (fail-open)
_ml_inference_engine = None
_ml_inference_attempted = False


_ml_model_file_mtime = 0.0  # V11.4: Track model file modification time for hot-swap


def reload_ml_model():
    """V11.4: Force reload the ML model (e.g., after retraining).

    Thread-safe — acquires state lock before swapping.
    """
    global _ml_inference_engine, _ml_model_file_mtime
    with _state.lock:
        _state.ml_model = None
        _state.ml_model_load_attempted = False
        _ml_inference_engine = None
        _ml_model_file_mtime = 0.0
    logger.info("V11.4: ML model reload triggered")
    return _get_ml_model()


def check_ml_model_freshness():
    """V11.4: Check if a newer model file exists and hot-swap if so.

    Called periodically (e.g., every 5 minutes) from the main loop.
    Zero-downtime: loads new model, then swaps reference under lock.
    """
    global _ml_model_file_mtime
    try:
        import glob as _glob
        import os
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        pkl_files = sorted(_glob.glob(os.path.join(model_dir, "model_*.pkl")))
        if not pkl_files:
            return
        latest = pkl_files[-1]
        mtime = os.path.getmtime(latest)
        if mtime > _ml_model_file_mtime and _ml_model_file_mtime > 0:
            logger.info("V11.4: New ML model detected: %s (mtime %.0f > %.0f)", latest, mtime, _ml_model_file_mtime)
            reload_ml_model()
        _ml_model_file_mtime = mtime
    except Exception as e:
        logger.debug("V11.4: Model freshness check failed: %s", e)


def _get_ml_model():
    """Lazy-load the most recent trained ML model (singleton, fail-open).

    V11.3 T9: Tries BatchInferenceEngine first (joblib models), then falls
    back to the legacy ModelTrainer.load_model path.
    """
    global _ml_inference_engine, _ml_inference_attempted, _ml_model_file_mtime

    if _state.ml_model_load_attempted:
        return _state.ml_model
    with _state.lock:
        if _state.ml_model_load_attempted:
            return _state.ml_model
        _state.ml_model_load_attempted = True

        # V11.3 T9: Try BatchInferenceEngine with joblib models first
        try:
            import glob
            import os
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

            # Look for joblib models first, then pkl
            joblib_files = sorted(glob.glob(os.path.join(model_dir, "*.joblib")))
            pkl_files = sorted(glob.glob(os.path.join(model_dir, "model_*.pkl")))

            model_path = None
            if joblib_files:
                model_path = joblib_files[-1]
            elif pkl_files:
                model_path = pkl_files[-1]

            if model_path:
                # V11.4: Track model file mtime for hot-swap detection
                _ml_model_file_mtime = os.path.getmtime(model_path)
                try:
                    from ml.inference import BatchInferenceEngine
                    _ml_inference_engine = BatchInferenceEngine(model_path=model_path)
                    if _ml_inference_engine.is_loaded:
                        # Use the BatchInferenceEngine's internal model
                        _state.ml_model = _ml_inference_engine
                        logger.info("V11.3 T9: BatchInferenceEngine loaded: %s", model_path)
                    else:
                        logger.info("V11.3 T9: BatchInferenceEngine failed to load model, trying legacy")
                        _ml_inference_engine = None
                except ImportError:
                    logger.debug("V11.3 T9: BatchInferenceEngine not available, trying legacy")

                # Legacy fallback
                if _state.ml_model is None and pkl_files:
                    try:
                        from ml.training import ModelTrainer
                        _state.ml_model = ModelTrainer.load_model(pkl_files[-1])
                        logger.info("WIRE-002: ML model loaded (legacy): %s", pkl_files[-1])
                    except Exception as e:
                        logger.info("WIRE-002: Legacy ML model load failed: %s", e)
            else:
                logger.info("WIRE-002: No trained ML model found in %s", model_dir)
        except Exception as e:
            logger.info("WIRE-002: ML model load skipped (fail-open): %s", e)
    return _state.ml_model


# T5-004: FinBERT NLP sentiment integration (fail-open)
def _get_finbert_sentiment():
    """Lazy-load FinBERTSentiment (singleton, fail-open)."""
    if _state.finbert_load_attempted:
        return _state.finbert_sentiment
    with _state.lock:
        if _state.finbert_load_attempted:
            return _state.finbert_sentiment
        _state.finbert_load_attempted = True
        try:
            if getattr(config, "NLP_SENTIMENT_ENABLED", False):
                from ml.models.fingpt import FinBERTSentiment
                _state.finbert_sentiment = FinBERTSentiment()
                logger.info("T5-004: FinBERTSentiment loaded successfully")
            else:
                logger.info("T5-004: NLP_SENTIMENT_ENABLED=False, skipping FinBERT")
        except Exception as e:
            logger.info("T5-004: FinBERTSentiment load skipped (fail-open): %s", e)
    return _state.finbert_sentiment


# WIRE-003: VPIN integration (fail-open)
try:
    from microstructure.vpin import VPIN as _VPIN_Class
    _VPIN_AVAILABLE = True
except ImportError:
    _VPIN_AVAILABLE = False

# T5-013: Lead-lag signal integration (fail-open)
try:
    from analytics.lead_lag import get_lead_lag_size_multiplier as _get_lead_lag_mult
    _LEAD_LAG_AVAILABLE = True
except ImportError:
    _LEAD_LAG_AVAILABLE = False

# T5-012: Alpha agent integration (fail-open)
try:
    from ml.alpha_agents import get_alpha_orchestrator as _get_alpha_orch
    _ALPHA_AGENTS_AVAILABLE = True
except ImportError:
    _ALPHA_AGENTS_AVAILABLE = False

# V11.3 T8: Data quality gate (fail-open)
_DATA_QUALITY_AVAILABLE = False
_data_quality_framework = None
try:
    from data.quality import DataQualityFramework
    _DATA_QUALITY_AVAILABLE = True
except ImportError:
    pass


def _check_data_quality(symbol: str, now) -> float | None:
    """Quick data quality check — returns score 0.0-1.0 or None if unavailable."""
    global _data_quality_framework
    if not _DATA_QUALITY_AVAILABLE:
        return None
    try:
        if _data_quality_framework is None:
            _data_quality_framework = DataQualityFramework()
        from data import get_snapshot
        snap = get_snapshot(symbol)
        if snap and snap.latest_trade:
            # Check staleness: how old is the last trade?
            trade_time = getattr(snap.latest_trade, 'timestamp', None)
            if trade_time:
                from datetime import timezone
                if hasattr(trade_time, 'tzinfo') and trade_time.tzinfo:
                    age_sec = (now.astimezone(timezone.utc) - trade_time.astimezone(timezone.utc)).total_seconds()
                else:
                    age_sec = 0
                if age_sec > 300:  # > 5 min old
                    return max(0.0, 1.0 - (age_sec / 600))  # Linear decay, 0 at 10 min
            return 1.0
        return 0.3  # No snapshot → low quality
    except Exception:
        return None


# T7-003: EDGAR 8-K monitor integration (fail-open)
_edgar_monitor = None
try:
    from data.alternative.sec_filings import EdgarMonitor as _EdgarMonitorClass
    _EDGAR_AVAILABLE = True
except ImportError:
    _EDGAR_AVAILABLE = False


def set_edgar_monitor(monitor) -> None:
    """Inject the EdgarMonitor instance from main.py."""
    global _edgar_monitor
    _edgar_monitor = monitor


def register_stopout(symbol: str) -> None:
    """Register a stop-loss event — blocks re-entry for REENTRY_COOLDOWN_MIN minutes."""
    cooldown_min = getattr(config, "REENTRY_COOLDOWN_MIN", 15)
    expiry = datetime.now() + timedelta(minutes=cooldown_min)
    with _state.lock:
        _state.stopout_cooldowns[symbol] = expiry
    logger.info("Cooldown: %s blocked for re-entry until %s (%d min)",
                symbol, expiry.strftime("%H:%M"), cooldown_min)


def _is_in_cooldown(symbol: str, now: datetime) -> bool:
    """Check if a symbol is in post-stop-loss cooldown."""
    with _state.lock:
        expiry = _state.stopout_cooldowns.get(symbol)
        if expiry is None:
            return False
        if now >= expiry:
            del _state.stopout_cooldowns[symbol]
            return False
        return True


def cleanup_stale_vpin_instances(active_symbols: set[str] | None = None) -> int:
    """T1-009: Evict VPIN instances for symbols no longer in the active universe.

    Call at EOD or universe refresh to prevent unbounded memory growth.

    Args:
        active_symbols: Set of currently active symbols. If None, clears all instances.

    Returns:
        Number of evicted instances.
    """
    with _state.lock:
        if active_symbols is None:
            count = len(_state.vpin_instances)
            _state.vpin_instances.clear()
            if count:
                logger.info("T1-009: Cleared all %d VPIN instances (EOD)", count)
            return count

        stale = [sym for sym in _state.vpin_instances if sym not in active_symbols]
        for sym in stale:
            del _state.vpin_instances[sym]
        if stale:
            logger.info("T1-009: Evicted %d stale VPIN instances: %s", len(stale), stale[:10])
        return len(stale)


# V10 BUG-042: Singleton IntradaySeasonality (created once, not per-signal)
def _get_seasonality():
    if _state.seasonality_instance is None and IntradaySeasonality:
        with _state.lock:
            if _state.seasonality_instance is None:
                _state.seasonality_instance = IntradaySeasonality()
    return _state.seasonality_instance


# Legacy aliases for backward compatibility with code referencing module-level dicts
# These are now proxied through _state
_vpin_instances = _state.vpin_instances
_halted_symbols = _state.halted_symbols
_asset_status_blocklist = _state.asset_status_blocklist

# OMS integration (fail-open: if OMS not available, orders still go through)
try:
    from oms.order import Order, OrderState
    from oms.order_manager import OrderManager
    from oms.transaction_cost import is_trade_profitable_after_costs
    _OMS_AVAILABLE = True
except ImportError:
    _OMS_AVAILABLE = False

# Event bus integration (fail-open)
try:
    from engine.events import get_event_bus, Event, EventTypes
    _EVENTS_AVAILABLE = True
except ImportError:
    _EVENTS_AVAILABLE = False

_notifications = None
_order_manager = None

# BUG-024 + RISK-005: Halted/delisted symbol blocklist with Alpaca asset status
_halted_symbols: dict[str, float] = {}  # symbol -> expiry timestamp
_HALT_BLOCKLIST_TTL_SEC = 300  # Block for 5 minutes after detection
# CRIT-009: Pair lock now uses _state.lock (T2-004)
_ASSET_STATUS_REFRESH_SEC = 300  # Refresh every 5 minutes
_ASSET_STALE_THRESHOLD_SEC = 600  # 10 minutes — if stale, block unknown symbols
_last_successful_asset_refresh: float = 0.0

# T1-005: Emergency escape hatch — bypass OMS requirement (paper trading only)
_force_no_oms: bool = False


def _refresh_asset_status_blocklist() -> None:
    """RISK-005: Refresh the halted/delisted blocklist from Alpaca asset status API.

    Called automatically when the blocklist is stale (> 5 minutes old).
    Fail-open: if the API call fails, the stale blocklist is kept.
    """
    # T2-004: asset status refresh uses _state.last_asset_status_refresh

    current_ts = _time.time()
    with _state.lock:
        if (current_ts - _state.last_asset_status_refresh) < _ASSET_STATUS_REFRESH_SEC:
            return  # Still fresh

    try:
        from broker.alpaca_client import get_trading_client

        client = get_trading_client()
        # Fetch the full set of open-position symbols + recently seen symbols
        # to check their tradability status
        symbols_to_check = set(_halted_symbols.keys()) | set(_asset_status_blocklist.keys())

        new_blocklist: dict[str, float] = {}
        expiry = current_ts + _ASSET_STATUS_REFRESH_SEC

        for symbol in symbols_to_check:
            try:
                asset = client.get_asset(symbol)
                if not asset.tradable or asset.status == "inactive":
                    new_blocklist[symbol] = expiry
                    logger.info(f"RISK-005: {symbol} is not tradable (status={asset.status}), blocking")
            except Exception as e:
                # T1-002: Fail-safe — if we can't verify, mark UNKNOWN and block
                new_blocklist[symbol] = expiry
                logger.warning(f"T1-002: asset status unavailable, blocking {symbol}: {e}")

        with _state.lock:
            _asset_status_blocklist.update(new_blocklist)
            _state.last_asset_status_refresh = current_ts

        # T1-002: Track successful refresh for staleness detection
        global _last_successful_asset_refresh
        _last_successful_asset_refresh = current_ts

        logger.debug(f"RISK-005: Asset status blocklist refreshed, {len(new_blocklist)} blocked symbols")

    except Exception as e:
        logger.warning(f"RISK-005: Asset status refresh failed (fail-safe): {e}")
        with _state.lock:
            _state.last_asset_status_refresh = current_ts  # Don't retry immediately


def _is_asset_blocked(symbol: str) -> bool:
    """RISK-005: Check if a symbol is blocked due to Alpaca asset status (halted/delisted).

    Also checks the symbol on-demand if not in the blocklist, to catch newly halted symbols.
    Fail-open: returns False if the check fails.
    """
    current_ts = _time.time()

    # Trigger periodic refresh
    _refresh_asset_status_blocklist()

    # Check blocklist
    with _state.lock:
        if symbol in _asset_status_blocklist:
            if current_ts < _asset_status_blocklist[symbol]:
                return True
            else:
                del _asset_status_blocklist[symbol]

    # T1-002 + V11.4: If asset status data is stale (> 10 min since last successful refresh),
    # only block symbols that were previously seen as halted — don't block unknown symbols
    # because that would prevent ALL trading when the asset status API is temporarily down.
    if _last_successful_asset_refresh > 0 and (current_ts - _last_successful_asset_refresh) > _ASSET_STALE_THRESHOLD_SEC:
        with _state.lock:
            if symbol in _asset_status_blocklist:
                logger.warning(f"T1-002: asset status stale, blocking previously-halted {symbol}")
                return True
        # V11.4: Don't block unknown symbols — fail-open for API outages
        logger.debug(f"V11.4: asset status stale but {symbol} not in blocklist, allowing")

    # On-demand check for symbols not in blocklist
    # Only do on-demand checks if we've had at least one successful refresh
    # (i.e., the broker API is actually available in this environment)
    if _last_successful_asset_refresh > 0:
        try:
            from broker.alpaca_client import get_trading_client

            client = get_trading_client()
            asset = client.get_asset(symbol)

            if not asset.tradable or asset.status == "inactive":
                with _state.lock:
                    _asset_status_blocklist[symbol] = current_ts + _ASSET_STATUS_REFRESH_SEC
                logger.warning(f"RISK-005: {symbol} is not tradable (status={asset.status}), blocking")
                return True

        except Exception as e:
            # T1-002: Fail-safe — if on-demand check fails, block the symbol
            with _state.lock:
                _asset_status_blocklist[symbol] = current_ts + _ASSET_STATUS_REFRESH_SEC
            logger.warning(f"T1-002: asset status unavailable, blocking {symbol}: {e}")
            return True

    return False


def _is_symbol_halted(symbol: str, now: datetime) -> bool:
    """BUG-024 + RISK-005: Check if a symbol is halted or blocked.

    Combines volume-based halt detection (BUG-024) with Alpaca asset
    status checks (RISK-005). Fail-open on all checks.
    """
    # RISK-005: Check Alpaca asset status blocklist first (fast path)
    if _is_asset_blocked(symbol):
        return True

    # BUG-024: Check volume-based blocklist
    current_ts = _time.time()
    if symbol in _halted_symbols:
        if current_ts < _halted_symbols[symbol]:
            return True
        else:
            del _halted_symbols[symbol]

    try:
        from data import get_intraday_bars
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        lookback = now - timedelta(minutes=15)
        bars = get_intraday_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), start=lookback, end=now)
        if bars is None or bars.empty or len(bars) < 5:
            return False

        # Check last 5 bars for zero volume
        last_5_vol = bars["volume"].iloc[-5:]
        if (last_5_vol == 0).all():
            _halted_symbols[symbol] = current_ts + _HALT_BLOCKLIST_TTL_SEC
            logger.warning(f"BUG-024: {symbol} appears halted (zero volume for 5+ bars), blocking for {_HALT_BLOCKLIST_TTL_SEC}s")
            return True
    except Exception as e:
        logger.debug(f"Halt detection check failed for {symbol}: {e}")

    return False


def _get_notifications():
    global _notifications
    if _notifications is None:
        try:
            import notifications as _n
            _notifications = _n
        except ImportError:
            _notifications = False
    return _notifications if _notifications else None


def set_order_manager(mgr):
    """Set the OMS order manager for order tracking."""
    global _order_manager
    _order_manager = mgr


def _emit_event(event_type: str, data: dict, source: str = "signal_processor"):
    """Emit an event on the bus (fail-open)."""
    if _EVENTS_AVAILABLE:
        try:
            bus = get_event_bus()
            bus.publish(Event(event_type, data, source=source))
        except Exception as e:
            logger.debug(f"Event bus publish failed: {e}")


def _pairs_atomic_rollback(leg1_order_id: str, symbol: str) -> bool:
    """T1-001: Roll back leg 1 of a pairs trade if leg 2 fails.

    Submits a market close order for the given symbol with retry logic.
    Returns True if rollback succeeded, False if all attempts exhausted.
    """
    for attempt in range(3):
        try:
            close_position(symbol, reason="pair_atomic_rollback")
            logger.warning(f"T1-001: Pairs rollback succeeded for {symbol} (order {leg1_order_id}) on attempt {attempt + 1}")
            _emit_event("PAIRS_ROLLBACK_TRIGGERED", {
                "symbol": symbol,
                "leg1_order_id": str(leg1_order_id),
                "attempt": attempt + 1,
                "status": "success",
            })
            return True
        except Exception as e:
            logger.error(f"T1-001: Pairs rollback attempt {attempt + 1}/3 failed for {symbol}: {e}")
            if attempt < 2:
                _time.sleep(0.5 * (attempt + 1))  # 500ms backoff

    logger.critical(f"T1-001: PAIRS ROLLBACK FAILED after 3 attempts for {symbol} (order {leg1_order_id}) — manual intervention required")
    _emit_event("PAIRS_ROLLBACK_TRIGGERED", {
        "symbol": symbol,
        "leg1_order_id": str(leg1_order_id),
        "attempt": 3,
        "status": "FAILED",
    })
    return False


def process_signals(
    signals: list[Signal],
    risk: RiskManager,
    regime: str,
    now: datetime,
    vol_engine: VolatilityTargetingRiskEngine,
    pnl_lock: DailyPnLLock,
    ws_monitor=None,
    news_sentiment=None,
    llm_scorer=None,
    regime_detector=None,
    cross_asset_monitor=None,
    var_monitor=None,
    corr_limiter=None,
):
    """Process signals: check filters, risk, size, and submit orders.

    Filters: position conflict, earnings, correlation, short pre-check,
    news sentiment (soft), LLM scoring (optional), VaR budget, concentration.
    """
    if not pnl_lock.is_trading_allowed():
        logger.info("PnL lock LOSS_HALT active -- skipping all signals")
        for signal in signals:
            database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, "pnl_halt")
        return

    # Group pairs signals by pair_id for atomic processing
    pair_groups: dict[str, list[Signal]] = {}
    non_pair_signals: list[Signal] = []
    for signal in signals:
        if signal.pair_id:
            pair_groups.setdefault(signal.pair_id, []).append(signal)
        else:
            non_pair_signals.append(signal)

    # Process non-pair signals
    for signal in non_pair_signals:
        _process_single_signal(signal, risk, regime, now, vol_engine, pnl_lock, ws_monitor,
                               news_sentiment, llm_scorer, regime_detector, cross_asset_monitor,
                               var_monitor, corr_limiter)

    # Process pairs atomically (both legs or neither)
    # CRIT-009: Lock around pair submission to prevent unhedged exposure
    with _state.lock:
        for pair_id, pair_signals in pair_groups.items():
            if len(pair_signals) != 2:
                logger.warning(f"Pair {pair_id} has {len(pair_signals)} signals, skipping")
                continue

            all_ok = True
            for sig in pair_signals:
                if sig.symbol in risk.open_trades:
                    all_ok = False
                    break
                allowed, reason = risk.can_open_trade(strategy=sig.strategy)
                if not allowed:
                    all_ok = False
                    break

            if all_ok:
                # T1-001 + CRIT-009: Submit both legs atomically with rollback if second fails
                first_sig, second_sig = pair_signals[0], pair_signals[1]
                _process_single_signal(first_sig, risk, regime, now, vol_engine, pnl_lock, ws_monitor,
                                       news_sentiment, llm_scorer, regime_detector, cross_asset_monitor,
                                       var_monitor, corr_limiter)

                if first_sig.symbol in risk.open_trades:
                    # T1-001: Track leg 1 order ID before submitting leg 2
                    leg1_trade = risk.open_trades.get(first_sig.symbol)
                    leg1_order_id = leg1_trade.order_id if leg1_trade else ""

                    _process_single_signal(second_sig, risk, regime, now, vol_engine, pnl_lock, ws_monitor,
                                           news_sentiment, llm_scorer, regime_detector, cross_asset_monitor,
                                           var_monitor, corr_limiter)

                    # T1-001: If leg 2 failed (rejected or not in open_trades), roll back leg 1
                    if second_sig.symbol not in risk.open_trades:
                        logger.warning(f"T1-001: Pair {pair_id}: second leg {second_sig.symbol} failed, rolling back first leg {first_sig.symbol}")
                        rollback_ok = _pairs_atomic_rollback(leg1_order_id, first_sig.symbol)
                        # Also clean up risk manager state for leg 1
                        trade = risk.open_trades.get(first_sig.symbol)
                        if trade:
                            risk.close_trade(first_sig.symbol, trade.entry_price, now, exit_reason="pair_atomic_rollback")
                        if not rollback_ok:
                            logger.critical(f"T1-001: PAIR ROLLBACK FAILED for {first_sig.symbol} — manual intervention required")
                else:
                    database.log_signal(now, second_sig.symbol, second_sig.strategy, second_sig.side, False, "pair_first_leg_failed")
            else:
                for sig in pair_signals:
                    database.log_signal(now, sig.symbol, sig.strategy, sig.side, False, "pair_blocked")


def _process_single_signal(
    signal: Signal,
    risk: RiskManager,
    regime: str,
    now: datetime,
    vol_engine: VolatilityTargetingRiskEngine,
    pnl_lock: DailyPnLLock,
    ws_monitor=None,
    news_sentiment=None,
    llm_scorer=None,
    regime_detector=None,
    cross_asset_monitor=None,
    var_monitor=None,
    corr_limiter=None,
):
    """Process a single signal through filters and submit if valid."""
    skip_reason = ""

    # 1. Position conflict
    if signal.symbol in risk.open_trades:
        skip_reason = "already_in_position"
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # 1a. BUG-024: Halted symbol detection
    if _is_symbol_halted(signal.symbol, now):
        skip_reason = "symbol_halted"
        logger.info(f"Signal skipped for {signal.symbol}: appears halted (zero volume)")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # 1a2. Re-entry cooldown after stop-loss
    if _is_in_cooldown(signal.symbol, now):
        skip_reason = "stopout_cooldown"
        logger.info(f"Signal skipped for {signal.symbol}: in post-stop cooldown")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # 1a3. Per-symbol daily loss cap
    max_symbol_loss = getattr(config, "MAX_SYMBOL_DAILY_LOSS", 200.0)
    symbol_pnl = risk._symbol_daily_pnl.get(signal.symbol, 0.0)
    if symbol_pnl <= -max_symbol_loss:
        skip_reason = f"symbol_daily_loss_cap_{symbol_pnl:.0f}"
        logger.info(f"Signal skipped for {signal.symbol}: daily loss ${symbol_pnl:.2f} exceeds cap -${max_symbol_loss}")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # 1b. T5-008: Enhanced PDT rule enforcement
    # Uses both PDTTracker (simple) and PDTCompliance (full) for defense-in-depth.
    if signal.hold_type == "day" and getattr(config, "PDT_PROTECTION_ENABLED", True):
        try:
            # Primary: PDTTracker (simple, already integrated)
            from risk.pdt_tracker import PDTTracker
            if not getattr(risk, '_pdt', None):
                risk._pdt = PDTTracker()
            if not risk._pdt.can_day_trade(risk.current_equity):
                skip_reason = "pdt_limit"
                logger.warning(f"PDT: Blocked day trade for {signal.symbol} — at limit")
                log_event(EventType.PDT_BLOCKED, "signal_processor",
                          symbol=signal.symbol, strategy=signal.strategy,
                          details=f"Day trade blocked — PDT limit reached")
                database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
                return

            # T5-008: Secondary: PDTCompliance (full compliance with persistence)
            try:
                from compliance.pdt import PDTCompliance
                if not getattr(risk, '_pdt_compliance', None):
                    risk._pdt_compliance = PDTCompliance()
                allowed, remaining, reason = risk._pdt_compliance.can_day_trade(risk.current_equity)
                if not allowed:
                    skip_reason = "pdt_compliance_limit"
                    logger.warning(f"PDT Compliance: Blocked day trade for {signal.symbol} — {reason}")
                    log_event(EventType.PDT_BLOCKED, "signal_processor",
                              symbol=signal.symbol, strategy=signal.strategy,
                              details=f"PDT compliance block: {reason}")
                    database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
                    return
                elif remaining == 1:
                    logger.warning(f"PDT: LAST day trade available for {signal.symbol} — use with caution")
            except ImportError:
                pass  # PDTCompliance not available, PDTTracker is sufficient
        except Exception as e:
            logger.warning(f"PDT check failed for {signal.symbol} (fail-open): {e}")

    # 2. Earnings filter
    if has_earnings_soon(signal.symbol):
        skip_reason = "earnings_soon"
        logger.info(f"Signal skipped for {signal.symbol}: earnings soon")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # 2b. V11.3 T8: Data quality gate — skip signals on stale/low-quality data
    _data_quality_mult = 1.0
    if _DATA_QUALITY_AVAILABLE:
        try:
            dq_score = _check_data_quality(signal.symbol, now)
            if dq_score is not None:
                if dq_score < 0.5:
                    skip_reason = f"low_data_quality_{dq_score:.2f}"
                    logger.info(f"Signal skipped for {signal.symbol}: data quality {dq_score:.2f} < 0.5")
                    database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
                    return
                elif dq_score < 0.8:
                    # Reduce position size proportionally
                    _data_quality_mult = dq_score / 0.8
        except Exception:
            pass  # Fail-open

    # 3. Correlation filter (skip for pairs — inherently correlated)
    if signal.strategy != "KALMAN_PAIRS":
        open_symbols = list(risk.open_trades.keys())
        if open_symbols and is_too_correlated(signal.symbol, open_symbols):
            skip_reason = "high_correlation"
            database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
            return

    # 4. Short selling pre-check
    if signal.side == "sell":
        if not config.ALLOW_SHORT:
            skip_reason = "shorting_disabled"
            database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
            return
        shortable, short_reason = can_short(signal.symbol, 1, signal.entry_price)
        if not shortable:
            skip_reason = f"short_blocked_{short_reason}"
            database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
            return

    # 5. Risk limits
    allowed, reason = risk.can_open_trade(strategy=signal.strategy)
    if not allowed:
        skip_reason = reason
        logger.info(f"Trade blocked for {signal.symbol}: {reason}")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # 5a. V10: Correlation-based concentration check (skip for pairs — inherently correlated)
    if corr_limiter and signal.strategy != "KALMAN_PAIRS":
        try:
            open_symbols = list(risk.open_trades.keys())
            if open_symbols:
                conc = corr_limiter.check_new_position(signal.symbol, open_symbols)
                if conc.too_concentrated:
                    skip_reason = f"concentration_{conc.reason}"
                    logger.info(f"Trade blocked for {signal.symbol}: {skip_reason}")
                    database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
                    return
        except Exception as e:
            logger.warning(f"Concentration check failed for {signal.symbol} (fail-open): {e}")

    # 5b. News sentiment size adjustment (soft filter)
    news_mult = 1.0
    if news_sentiment:
        try:
            news_mult, news_reason = news_sentiment.get_sentiment_size_mult(signal.symbol)
            if news_mult == 0.0:
                database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, news_reason)
                return
        except Exception as exc:
            handle_failure(FailureMode.DEGRADE_GRACEFULLY,
                           "signal_processor.news_sentiment", exc,
                           symbol=signal.symbol, strategy=signal.strategy)
            news_mult = 1.0

    # 5c. LLM signal scoring (optional, fail-open)
    llm_mult = 1.0
    if llm_scorer and config.LLM_SCORING_ENABLED:
        try:
            context = {
                'spy_day_return': risk.day_pnl,
                'vix_level': getattr(vol_engine, '_last_vix', 20.0),
            }
            scored = llm_scorer.score_signal(signal, context)
            if scored.score < config.LLM_SCORE_THRESHOLD:
                database.log_signal(now, signal.symbol, signal.strategy, signal.side, False,
                                   f'llm_low_score_{scored.score:.2f}')
                return
            if config.LLM_SCORE_SIZE_MULT:
                llm_mult = scored.size_mult
        except Exception as exc:
            handle_failure(FailureMode.DEGRADE_GRACEFULLY,
                           "signal_processor.llm_scoring", exc,
                           symbol=signal.symbol, strategy=signal.strategy)
            llm_mult = 1.0

    # 6. Position sizing via vol-targeting engine
    vol_scalar = vol_engine.last_scalar
    lock_mult = pnl_lock.get_size_multiplier()
    qty = vol_engine.calculate_position_size(
        equity=risk.current_equity,
        entry_price=signal.entry_price,
        stop_price=signal.stop_loss,
        vol_scalar=vol_scalar,
        strategy=signal.strategy,
        side=signal.side,
        pnl_lock_mult=lock_mult,
    )

    # Regime affinity multiplier (fail-open)
    regime_mult = 1.0
    if regime_detector and getattr(config, "HMM_REGIME_ENABLED", False):
        try:
            regime_mult = regime_detector.get_regime_affinity(signal.strategy)
        except Exception as exc:
            handle_failure(FailureMode.DEGRADE_GRACEFULLY,
                           "signal_processor.regime_affinity", exc,
                           symbol=signal.symbol, strategy=signal.strategy)
            regime_mult = 1.0

    # V10 BUG-042: Intraday seasonality multiplier (singleton, fail-open)
    seasonality_mult = 1.0
    if getattr(config, "INTRADAY_SEASONALITY_ENABLED", False) and IntradaySeasonality:
        try:
            seasonality_mult = _get_seasonality().get_window_score(signal.strategy, now)
        except Exception as exc:
            handle_failure(FailureMode.DEGRADE_GRACEFULLY,
                           "signal_processor.seasonality", exc,
                           symbol=signal.symbol, strategy=signal.strategy)
            seasonality_mult = 1.0

    # Cross-asset bias multiplier (fail-open)
    cross_asset_mult = 1.0
    if getattr(config, "CROSS_ASSET_ENABLED", False) and cross_asset_monitor:
        try:
            bias = cross_asset_monitor.get_equity_bias()
            if bias < -0.5:
                cross_asset_mult = getattr(config, "CROSS_ASSET_FLIGHT_REDUCTION", 0.30)
        except Exception as exc:
            handle_failure(FailureMode.DEGRADE_GRACEFULLY,
                           "signal_processor.cross_asset_bias", exc,
                           symbol=signal.symbol, strategy=signal.strategy)
            cross_asset_mult = 1.0

    # V10: VaR risk budget multiplier (fail-open: 1.0)
    var_mult = 1.0
    if var_monitor:
        try:
            var_mult = var_monitor.size_multiplier
        except Exception as exc:
            handle_failure(FailureMode.DEGRADE_GRACEFULLY,
                           "signal_processor.var_monitor", exc,
                           symbol=signal.symbol, strategy=signal.strategy)
            var_mult = 1.0

    # WIRE-001: Feature store — fetch features for ML/sizing enrichment (fail-open)
    _signal_features = None
    if _FEATURE_STORE_AVAILABLE:
        try:
            fs = _get_feature_store()
            _signal_features = fs.get_all_features(signal.symbol)
            if not _signal_features:
                _signal_features = None
        except Exception as e:
            logger.debug("WIRE-001: Feature store lookup failed for %s (fail-open): %s", signal.symbol, e)

    # WIRE-002 + V11.3 T9: ML model confidence multiplier (fail-open)
    # Uses BatchInferenceEngine when available, falls back to legacy model.predict()
    ml_conf_mult = 1.0
    try:
        model = _get_ml_model()
        if model is not None:
            import pandas as _pd
            confidence = 0.5  # neutral default

            # V11.3 T9: Use BatchInferenceEngine if available
            if _ml_inference_engine is not None and _ml_inference_engine.is_loaded:
                if _signal_features:
                    feat_series = _pd.Series(_signal_features)
                    confidence = _ml_inference_engine.predict_single(signal.symbol, feat_series)
                else:
                    # Even without feature store, try with basic features
                    basic_features = {
                        "price": signal.entry_price,
                        "side": 1.0 if signal.side == "buy" else -1.0,
                        "strategy": hash(signal.strategy) % 100 / 100.0,
                    }
                    feat_series = _pd.Series(basic_features)
                    confidence = _ml_inference_engine.predict_single(signal.symbol, feat_series)
            elif _signal_features:
                # Legacy path: direct model.predict()
                feat_df = _pd.DataFrame([_signal_features])
                prediction = model.predict(feat_df)
                confidence = float(prediction[0]) if len(prediction) > 0 else 0.5

            # V11.3 T9: ML confidence gating
            # < 0.35 → skip signal (strong ML disagreement)
            # 0.35-0.65 → neutral (no adjustment)
            # > 0.65 → boost conviction 10-20%
            if confidence < 0.35:
                logger.info(
                    f"V11.3 T9: ML rejects {signal.symbol} — confidence={confidence:.2f} < 0.35"
                )
                return None
            elif confidence > 0.65:
                ml_conf_mult = 1.0 + (confidence - 0.65) * 0.57  # Maps 0.65→1.0, 1.0→1.2
                ml_conf_mult = min(ml_conf_mult, 1.2)
            else:
                ml_conf_mult = 1.0  # Neutral zone
    except Exception as e:
        logger.debug("WIRE-002: ML prediction failed for %s (fail-open): %s", signal.symbol, e)
        ml_conf_mult = 1.0

    # WIRE-003 + T5-007: VPIN toxicity-adjusted position sizing (fail-open)
    # T5-007: toxicity_multiplier = max(0.3, 1.0 - 2.0 * (vpin - 0.25)) for vpin in [0.25, 0.60]
    # Replaces the simple >0.7 threshold with a continuous adjustment curve.
    vpin_mult = 1.0
    if _VPIN_AVAILABLE:
        try:
            with _state.lock:
                vpin_inst = _vpin_instances.get(signal.symbol)
            if vpin_inst is not None:
                vpin_value = vpin_inst.compute_vpin()
                if vpin_value > 0.25:
                    # T5-007: Continuous toxicity adjustment
                    toxicity_multiplier = max(0.3, 1.0 - 2.0 * (vpin_value - 0.25))
                    vpin_mult = toxicity_multiplier
                    logger.info(
                        "T5-007: VPIN=%.2f for %s — toxicity_mult=%.2f (size reduction %.0f%%)",
                        vpin_value, signal.symbol, toxicity_multiplier,
                        (1.0 - toxicity_multiplier) * 100,
                    )
        except Exception as e:
            logger.debug("WIRE-003: VPIN check failed for %s (fail-open): %s", signal.symbol, e)

    # T5-004: FinBERT NLP sentiment multiplier (fail-open)
    finbert_mult = 1.0
    if getattr(config, "NLP_SENTIMENT_ENABLED", False):
        try:
            finbert = _get_finbert_sentiment()
            if finbert is not None and news_sentiment:
                # Use existing news headlines if available from news_sentiment module
                headlines = []
                try:
                    headlines = news_sentiment.get_recent_headlines(signal.symbol)
                except (AttributeError, Exception):
                    pass
                if headlines:
                    finbert_mult, finbert_reason = finbert.get_sentiment_signal(signal.symbol, headlines)
                    if finbert_mult == 0.0:
                        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, finbert_reason)
                        return
        except Exception as e:
            logger.debug("T5-004: FinBERT check failed for %s (fail-open): %s", signal.symbol, e)

    # T5-013: Lead-lag bias multiplier (fail-open)
    lead_lag_mult = 1.0
    if _LEAD_LAG_AVAILABLE and getattr(config, "LEAD_LAG_ENABLED", False):
        try:
            lead_lag_mult = _get_lead_lag_mult(signal.symbol)
        except Exception as e:
            logger.debug("T5-013: Lead-lag check failed for %s (fail-open): %s", signal.symbol, e)

    # T5-012: Alpha agent sizing multiplier (fail-open)
    alpha_agent_mult = 1.0
    if _ALPHA_AGENTS_AVAILABLE and getattr(config, "ALPHA_AGENTS_ENABLED", False):
        try:
            orch = _get_alpha_orch()
            alpha_result = orch.get_alpha_score(signal.symbol, {
                "vix_level": getattr(vol_engine, '_last_vix', 20.0),
                "spy_return": getattr(risk, 'day_pnl', 0.0),
            })
            alpha_agent_mult = alpha_result.size_multiplier
        except Exception as e:
            logger.debug("T5-012: Alpha agent check failed for %s (fail-open): %s", signal.symbol, e)

    # T7-003: EDGAR 8-K filing bias multiplier (fail-open)
    edgar_mult = 1.0
    if _EDGAR_AVAILABLE and _edgar_monitor and getattr(config, "EDGAR_MONITOR_ENABLED", False):
        try:
            edgar_bias = _edgar_monitor.get_signal_bias(signal.symbol)
            if abs(edgar_bias) > 0.1:
                # Map bias [-1, 1] to multiplier [0.7, 1.3]
                edgar_mult = 1.0 + 0.3 * edgar_bias
                edgar_mult = max(0.7, min(1.3, edgar_mult))
                logger.info(
                    "T7-003: EDGAR bias=%.2f for %s — edgar_mult=%.2f",
                    edgar_bias, signal.symbol, edgar_mult,
                )
        except Exception as e:
            logger.debug("T7-003: EDGAR check failed for %s (fail-open): %s", signal.symbol, e)

    # V11.3 T1: Conviction scoring — weighted average instead of multiplicative cascade.
    # The old multiplicative approach (12 factors multiplied) caused catastrophic size
    # reduction: three factors at 0.8 → 0.51 combined. Now we use weighted averaging
    # with hard vetoes only for critical signals (news=0, VPIN extreme, FinBERT=0).
    #
    # Hard vetoes (already handled above with early returns):
    #   - news_mult == 0.0 → already returned
    #   - finbert_mult == 0.0 → already returned
    #   - LLM score below threshold → already returned
    #
    # Soft multipliers → conviction score via weighted average:
    _conviction_inputs = [
        (regime_mult,       0.20, "regime"),        # Market regime alignment
        (vpin_mult,         0.20, "vpin"),           # Market microstructure toxicity
        (ml_conf_mult,      0.15, "ml"),             # ML model confidence
        (cross_asset_mult,  0.10, "cross_asset"),    # Cross-asset signal
        (llm_mult,          0.10, "llm"),            # LLM scoring
        (seasonality_mult,  0.08, "seasonality"),    # Time-of-day effects
        (lead_lag_mult,     0.07, "lead_lag"),        # Lead-lag relationships
        (edgar_mult,        0.05, "edgar"),           # SEC filing bias
        (alpha_agent_mult,  0.05, "alpha_agent"),     # Alpha orchestrator
    ]
    # var_mult and news_mult applied directly (risk budget and hard veto respectively)
    weighted_sum = sum(mult * weight for mult, weight, _ in _conviction_inputs)
    total_weight = sum(weight for _, weight, _ in _conviction_inputs)
    conviction_score = weighted_sum / total_weight if total_weight > 0 else 1.0

    # Apply VaR budget as a direct multiplier (risk management, not conviction)
    # and news_mult (already filtered for 0.0, so always 1.0 here)
    # T8: data quality multiplier reduces size on stale data
    combined_mult = conviction_score * var_mult * news_mult * _data_quality_mult
    # Bound to [0.4, 1.5] — prevents both over-sizing and excessive reduction
    combined_mult = max(0.4, min(combined_mult, 1.5))
    qty = int(qty * combined_mult)
    logger.debug(
        "V11.3 conviction: %s conviction=%.2f var=%.2f → combined=%.2f qty=%d",
        signal.symbol, conviction_score, var_mult, combined_mult, qty,
    )

    if qty <= 0:
        skip_reason = "position_size_zero"
        logger.info(f"Position size 0 for {signal.symbol}, skipping")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        _emit_event(EventTypes.SIGNAL_FILTERED if _EVENTS_AVAILABLE else "signal.filtered",
                    {"symbol": signal.symbol, "reason": skip_reason})
        return

    # 6b. V10 PROFIT-GAP-001: Transaction cost filter (reject negative-EV trades)
    if _OMS_AVAILABLE and getattr(config, "COST_FILTER_ENABLED", True):
        try:
            # Use strategy-specific win rate if available, else 55%
            strategy_win_rates = getattr(config, "STRATEGY_WIN_RATES", {})
            win_rate = strategy_win_rates.get(signal.strategy, 0.55)

            profitable, cost_details = is_trade_profitable_after_costs(
                entry_price=signal.entry_price,
                take_profit=signal.take_profit,
                stop_loss=signal.stop_loss,
                qty=qty,
                side=signal.side,
                win_rate=win_rate,
                strategy=signal.strategy,  # BUG-026: pass strategy for per-strategy win rate
            )
            if not profitable:
                skip_reason = f"negative_ev_after_costs_{cost_details['cost_bps']:.0f}bps"
                logger.info(
                    f"Trade {signal.symbol} rejected: EV=${cost_details['expected_value']:.2f} "
                    f"after costs (${cost_details['total_cost']:.2f}, {cost_details['cost_bps']:.1f}bps)"
                )
                database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
                _emit_event(EventTypes.SIGNAL_FILTERED if _EVENTS_AVAILABLE else "signal.filtered",
                            {"symbol": signal.symbol, "reason": skip_reason, **cost_details})
                return
        except Exception as e:
            logger.debug(f"Cost filter failed (proceeding): {e}")

    # 7. Submit bracket order (with OMS tracking as pre-condition — T1-005)
    oms_order = None
    if _OMS_AVAILABLE and _order_manager:
        try:
            oms_order = _order_manager.create_order(
                symbol=signal.symbol,
                strategy=signal.strategy,
                side=signal.side,
                order_type="bracket",
                qty=qty,
                limit_price=signal.entry_price,
                take_profit=signal.take_profit,
                stop_loss=signal.stop_loss,
                pair_id=getattr(signal, "pair_id", ""),
                idempotency_key=f"{signal.symbol}_{signal.strategy}_{now.strftime('%Y%m%d%H%M%S')}",
            )
        except Exception as e:
            # T1-005: OMS registration failed — abort order to prevent orphaned orders
            if not _force_no_oms:
                logger.critical(f"T1-005: OMS registration failed for {signal.symbol}, aborting order submission: {e}")
                database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, "oms_registration_failed")
                _emit_event("order.oms_failed", {"symbol": signal.symbol, "strategy": signal.strategy, "error": str(e)})
                return
            else:
                logger.warning(f"T1-005: OMS registration failed but _force_no_oms=True, proceeding: {e}")
    elif _OMS_AVAILABLE and not _order_manager:
        # OMS module importable but order_manager never initialized — proceed without OMS tracking
        logger.debug(f"T1-005: OMS module available but order_manager not set for {signal.symbol}, proceeding without OMS tracking")

    order_id = submit_bracket_order(signal, qty)

    if order_id is None:
        skip_reason = "order_failed"
        logger.error(f"Failed to submit order for {signal.symbol}, skipping (no naked entry)")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        if oms_order and _order_manager:
            _order_manager.update_state(oms_order.oms_id, OrderState.FAILED)
        _emit_event(EventTypes.ORDER_FAILED if _EVENTS_AVAILABLE else "order.failed",
                    {"symbol": signal.symbol, "strategy": signal.strategy})
        return

    # Update OMS with broker order ID
    if oms_order and _order_manager:
        broker_id = order_id if isinstance(order_id, str) else str(order_id[0]) if order_id else ""
        _order_manager.update_state(oms_order.oms_id, OrderState.SUBMITTED, broker_order_id=broker_id)
    _emit_event(EventTypes.ORDER_SUBMITTED if _EVENTS_AVAILABLE else "order.submitted",
                {"symbol": signal.symbol, "strategy": signal.strategy, "qty": qty, "order_id": str(order_id)})
    log_event(EventType.ORDER_SUBMITTED, "signal_processor",
              symbol=signal.symbol, strategy=signal.strategy,
              details=f"qty={qty} side={signal.side} order_id={order_id}")

    # 8. Register trade with time stops / max hold
    time_stop = None
    max_hold_date = None
    hold_type = getattr(signal, "hold_type", "day")

    if signal.strategy == "STAT_MR":
        pass  # z-score exits handle it
    elif signal.strategy == "KALMAN_PAIRS":
        max_hold_date = now + timedelta(days=config.PAIRS_MAX_HOLD_DAYS)
    elif signal.strategy == "MICRO_MOM":
        time_stop = now + timedelta(minutes=config.MICRO_MAX_HOLD_MINUTES)
    elif signal.strategy == "BETA_HEDGE":
        hold_type = "day"

    trade = TradeRecord(
        symbol=signal.symbol,
        strategy=signal.strategy,
        side=signal.side,
        entry_price=signal.entry_price,
        entry_time=now,
        qty=qty,
        take_profit=signal.take_profit,
        stop_loss=signal.stop_loss,
        order_id=order_id,
        time_stop=time_stop,
        hold_type=hold_type,
        max_hold_date=max_hold_date,
        pair_id=getattr(signal, "pair_id", ""),
        highest_price_seen=signal.entry_price,
    )
    risk.register_trade(trade)

    if ws_monitor:
        ws_monitor.subscribe(signal.symbol)

    notif = _get_notifications()
    if notif and config.TELEGRAM_ENABLED:
        try:
            notif.notify_trade_opened(trade)
        except Exception as e:
            logger.warning(f"Notification failed: {e}")

    database.log_signal(now, signal.symbol, signal.strategy, signal.side, True, "")

    # Emit position opened event
    _emit_event(EventTypes.POSITION_OPENED if _EVENTS_AVAILABLE else "position.opened", {
        "symbol": signal.symbol,
        "strategy": signal.strategy,
        "side": signal.side,
        "qty": qty,
        "entry_price": signal.entry_price,
        "take_profit": signal.take_profit,
        "stop_loss": signal.stop_loss,
    })
