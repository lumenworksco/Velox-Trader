"""Velox V11.3 — Institutional-Grade Quantitative Trading System.

Six-strategy portfolio with HMM regime detection, ML-enhanced alpha,
volatility-targeted sizing, Kelly criterion, conviction-scored signal pipeline,
intraday risk controls, daily P&L locks, ATR-based trailing stops, data quality
gating, pre-trade slippage prediction, and operational hardening.

Strategies: StatMR (35%), VWAP (20%), KalmanPairs (20%), ORB (10%),
            MicroMom (10%), PEAD (5%).

V11.3 Upgrades:
- Conviction scoring replaces multiplicative multiplier stack (T1)
- Intraday risk controls with rolling P&L limits (T2)
- ATR-based trailing stops for MicroMomentum and ORB (T3/T12)
- Pre-trade slippage model integration (T4)
- VIX rate-of-change scaling (T6)
- ML inference via BatchInferenceEngine (T9)
- Capital reallocation from dead strategies (T10)
- Scaled StatMR exits with overshoot capture (T11)
- Production hardening: thread safety, dead code removal
"""

import argparse
import collections
import logging
import signal
import sys
import threading
import time as time_mod
from datetime import datetime, time, timedelta
from pathlib import Path

from rich.live import Live

import config
import database
import analytics as analytics_mod
from data import (
    verify_connectivity, verify_data_feed, get_account, get_clock,
    get_positions, get_snapshot, get_snapshots,
    get_filled_exit_price, get_filled_exit_info,
    get_feed_monitor,
)
# V10: Decomposed engine modules
from engine.broker_sync import sync_positions_with_broker, check_shadow_exits
from engine.signal_processor import process_signals
from engine.exit_processor import (
    handle_strategy_exits, handle_ws_close, get_current_prices,
)
# V12: Import advanced exit checks (profit tiers, dead signal, scale-out)
try:
    from engine.exit_processor import check_advanced_exits as _check_advanced_exits
except ImportError:
    _check_advanced_exits = None
# V12 AUDIT: Exit orchestrator as primary exit system (fail-open)
try:
    from engine.exit_orchestrator import get_exit_orchestrator
    _exit_orchestrator = get_exit_orchestrator()
except ImportError:
    _exit_orchestrator = None
from engine.scanner import scan_all_strategies, check_all_exits, run_beta_neutralization
from engine.daily_tasks import daily_reset, weekly_tasks, eod_close, backup_database
# V12 BONUS: Profit maximization engine
try:
    from engine.profit_maximizer import (
        IntradayVolRegime, WinStreakTracker,
        compute_dynamic_stop, get_adaptive_scan_interval,
    )
except ImportError:
    IntradayVolRegime = None
    WinStreakTracker = None
    compute_dynamic_stop = None
    get_adaptive_scan_interval = None
from strategies.base import Signal
from strategies.regime import MarketRegime
from strategies.stat_mean_reversion import StatMeanReversion
from strategies.kalman_pairs import KalmanPairsTrader
from strategies.micro_momentum import IntradayMicroMomentum
from strategies.vwap import VWAPStrategy

try:
    from strategies.orb_v2 import ORBStrategyV2
except ImportError:
    ORBStrategyV2 = None
from risk import (
    RiskManager, TradeRecord,
    VolatilityTargetingRiskEngine, DailyPnLLock, BetaNeutralizer,
)
from execution import (
    submit_bracket_order,
    close_position,
    close_partial_position,
    can_short,
)
from dashboard import (
    build_dashboard,
    print_day_summary,
    print_startup_info,
    console,
)
from earnings import load_earnings_cache, has_earnings_soon
from correlation import load_correlation_cache, is_too_correlated
from analytics.consistency_score import compute_consistency_score

try:
    import numpy as np
except ImportError:
    np = None

try:
    from position_monitor import PositionMonitor
except ImportError:
    PositionMonitor = None

try:
    import notifications
except ImportError:
    notifications = None

try:
    from news_sentiment import AlpacaNewsSentiment
except ImportError:
    AlpacaNewsSentiment = None

try:
    from llm_signal_scorer import LLMSignalScorer
except ImportError:
    LLMSignalScorer = None

try:
    from adaptive_exit_manager import AdaptiveExitManager
except ImportError:
    AdaptiveExitManager = None

try:
    from walk_forward import WalkForwardValidator
except ImportError:
    WalkForwardValidator = None

# --- V10 module imports (all fail-open) ---
try:
    from strategies.pead import PEADStrategy
except ImportError:
    PEADStrategy = None

try:
    from strategies.overnight import OvernightManager
except ImportError:
    OvernightManager = None

try:
    from analytics.cross_asset import CrossAssetMonitor
except ImportError:
    CrossAssetMonitor = None

try:
    from analytics.signal_ranker import SignalRanker
except ImportError:
    SignalRanker = None

try:
    from analytics.alpha_decay import AlphaDecayMonitor
except ImportError:
    AlphaDecayMonitor = None

try:
    from analytics.intraday_seasonality import IntradaySeasonality
except ImportError:
    IntradaySeasonality = None

try:
    from risk.adaptive_allocation import AdaptiveAllocator
except ImportError:
    AdaptiveAllocator = None

try:
    from analytics.param_optimizer import BayesianOptimizer
except ImportError:
    BayesianOptimizer = None

try:
    from watchdog import Watchdog, PositionReconciler, AuditTrail
except ImportError:
    Watchdog = None
    PositionReconciler = None
    AuditTrail = None

try:
    from smart_routing import SmartOrderRouter, FillMonitor
except ImportError:
    SmartOrderRouter = None
    FillMonitor = None

# --- V11 module imports (all fail-open for graceful degradation) ---
# V11.3: Removed unused strategy imports (SectorMomentum, CrossSectionalMomentum,
# MultiTimeframe, CopulaPairs) — these strategies are no longer in STRATEGY_ALLOCATIONS.

try:
    from risk.factor_model import FactorRiskModel
except ImportError:
    FactorRiskModel = None

try:
    from risk.hrp import HierarchicalRiskParity
except ImportError:
    HierarchicalRiskParity = None

try:
    from risk.stress_testing import StressTestFramework
except ImportError:
    StressTestFramework = None

try:
    from risk.intraday_controls import IntradayRiskControls
except ImportError:
    IntradayRiskControls = None

try:
    from risk.gap_risk import GapRiskManager
except ImportError:
    GapRiskManager = None

try:
    from risk.correlation import DynamicCorrelation
except ImportError:
    DynamicCorrelation = None

try:
    from execution.smart_router import SmartOrderRouter as V11SmartRouter
except ImportError:
    V11SmartRouter = None

try:
    from execution.optimal_execution import AlmgrenChriss
except ImportError:
    AlmgrenChriss = None

try:
    from execution.slippage_model import SlippageModel
except ImportError:
    SlippageModel = None

try:
    from execution.fill_analytics import FillAnalytics
except ImportError:
    FillAnalytics = None

try:
    from microstructure.vpin import VPIN
except ImportError:
    VPIN = None

try:
    from microstructure.order_book import OrderBookAnalyzer
except ImportError:
    OrderBookAnalyzer = None

try:
    from microstructure.trade_classifier import TradeClassifier
except ImportError:
    TradeClassifier = None

try:
    from ml.features import FeatureEngine
except ImportError:
    FeatureEngine = None

try:
    from ml.change_point import BayesianChangePointDetector
except ImportError:
    BayesianChangePointDetector = None

try:
    from data.feature_store import FeatureStore
except ImportError:
    FeatureStore = None

try:
    from data.quality import DataQualityFramework
except ImportError:
    DataQualityFramework = None

try:
    from monitoring.alerting import AlertManager
except ImportError:
    AlertManager = None

try:
    from monitoring.latency import LatencyTracker
except ImportError:
    LatencyTracker = None

try:
    from monitoring.reconciliation import PositionReconciler as V11Reconciler
except ImportError:
    V11Reconciler = None

try:
    from monitoring.metrics import MetricsPipeline
except ImportError:
    MetricsPipeline = None

try:
    from compliance.audit_trail import AuditTrail as V11AuditTrail
except ImportError:
    V11AuditTrail = None

try:
    from compliance.pdt import PDTCompliance
except ImportError:
    PDTCompliance = None

try:
    from compliance.surveillance import SelfSurveillance
except ImportError:
    SelfSurveillance = None

try:
    from risk.corporate_actions import CorporateActionDetector
except ImportError:
    CorporateActionDetector = None

try:
    from ops.drawdown_risk import DrawdownRiskManager
except ImportError:
    DrawdownRiskManager = None

# V11.4: Black-Litterman portfolio optimization
try:
    from risk.black_litterman import BlackLittermanOptimizer
except ImportError:
    BlackLittermanOptimizer = None

try:
    from ops.disaster_recovery import DisasterRecovery
except ImportError:
    DisasterRecovery = None

try:
    from alpha.cross_asset import CrossAssetSignalGenerator
except ImportError:
    CrossAssetSignalGenerator = None

try:
    from alpha.seasonality import EnhancedSeasonality
except ImportError:
    EnhancedSeasonality = None

try:
    from engine.event_bus import EventBus
except ImportError:
    EventBus = None

# --- Logging setup ---
_file_handler = logging.FileHandler(config.LOG_FILE)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    handlers=[_file_handler, _stream_handler],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def now_et() -> datetime:
    return datetime.now(config.ET)


_market_open_cache: tuple[float, bool] | None = None  # (timestamp, is_open)
_market_open_cache_lock = threading.Lock()  # MED-002: protect concurrent market-open cache access

def is_market_hours(t: time) -> bool:
    """Check if market is open, using Alpaca clock API for holiday detection."""
    global _market_open_cache
    import time as _time

    # V10 BUG-025: Check actual market status (holidays, early close) via API
    # Cache for 60 seconds to avoid API spam
    now_ts = _time.time()
    with _market_open_cache_lock:
        if _market_open_cache and (now_ts - _market_open_cache[0]) < 60:
            return _market_open_cache[1]

    # Wall-clock check first (fast path for obviously closed times)
    if not (config.MARKET_OPEN <= t <= config.MARKET_CLOSE):
        with _market_open_cache_lock:
            _market_open_cache = (now_ts, False)
        return False

    # API check for holidays and early closes
    try:
        clock = get_clock()
        is_open = clock.is_open
        with _market_open_cache_lock:
            _market_open_cache = (now_ts, is_open)
        return is_open
    except Exception:
        # Fallback to wall-clock if API fails
        with _market_open_cache_lock:
            _market_open_cache = (now_ts, True)
        return True


def is_trading_hours(t: time) -> bool:
    return config.TRADING_START <= t <= config.ORB_EXIT_TIME


# ---------------------------------------------------------------------------
# T4-006: Dynamic scan interval tied to VIX regime
# ---------------------------------------------------------------------------
_last_dynamic_interval: int | None = None


def compute_dynamic_scan_interval(base_interval: int = 120,
                                   open_positions: int = 0,
                                   signal_quality: float = 0.5) -> int:
    """T4-006 + T5-015: Compute scan interval.

    If ADAPTIVE_SCAN_ENABLED, uses RL-based ScanScheduler.
    Otherwise falls back to VIX-based scaling:
        VIX < 15  -> 180s (calm market, scan less frequently)
        VIX 15-25 -> 120s (normal)
        VIX 25-35 -> 60s  (elevated vol, scan more frequently)
        VIX > 35  -> 30s  (crisis, maximum scan frequency)

    Returns:
        Scan interval in seconds.
    """
    global _last_dynamic_interval
    try:
        from risk import get_vix_level
        vix = get_vix_level()

        if vix <= 0:
            vix = 20.0

        # T5-015: Try RL-based scheduler first
        if getattr(config, "ADAPTIVE_SCAN_ENABLED", False):
            try:
                from engine.scanner import get_scan_scheduler
                scheduler = get_scan_scheduler()
                interval = scheduler.get_interval(
                    vix=vix,
                    open_positions=open_positions,
                    signal_quality=signal_quality,
                )
                if _last_dynamic_interval is not None and interval != _last_dynamic_interval:
                    logger.info(
                        "T5-015: RL scan interval changed %ds -> %ds (VIX=%.1f, pos=%d)",
                        _last_dynamic_interval, interval, vix, open_positions,
                    )
                _last_dynamic_interval = interval
                return interval
            except Exception as e:
                logger.debug("T5-015: RL scheduler failed, falling back to VIX: %s", e)

        # VIX-based fallback (original T4-006 logic)
        if vix < 15:
            interval = 180
        elif vix < 25:
            interval = 120
        elif vix < 35:
            interval = 60
        else:
            interval = 30

        if _last_dynamic_interval is not None and interval != _last_dynamic_interval:
            logger.info(
                "T4-006: Scan interval changed %ds -> %ds (VIX=%.1f)",
                _last_dynamic_interval, interval, vix,
            )
        _last_dynamic_interval = interval
        return interval

    except Exception as e:
        logger.debug("T4-006: VIX fetch failed, using base interval %ds: %s", base_interval, e)
        return base_interval


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def startup_checks() -> dict:
    """Run all startup checks. Exit on failure."""
    console.print("\n[bold]Running startup checks...[/bold]\n")

    # 1. Verify API connectivity
    try:
        info = verify_connectivity()
        print_startup_info(info)
    except Exception as e:
        console.print(f"[bold red]FATAL: Cannot connect to Alpaca API: {e}[/bold red]")
        console.print("Check ALPACA_API_KEY and ALPACA_API_SECRET environment variables.")
        sys.exit(1)

    # 2. Check market status
    if not info["market_open"]:
        next_open = info.get("next_open", "unknown")
        console.print(f"[yellow]Market is closed. Next open: {next_open}[/yellow]")
        console.print("[yellow]Bot will wait for market open...[/yellow]")

    # 3. Verify data feed
    console.print("Verifying data feed...")
    try:
        if not verify_data_feed("SPY"):
            console.print("[bold red]FATAL: Cannot fetch market data. Check API permissions.[/bold red]")
            sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]FATAL: Data feed error: {e}[/bold red]")
        sys.exit(1)
    console.print("[green]Data feed verified.[/green]")

    # 4. Print symbol list
    console.print(
        f"\n[bold]Symbol universe:[/bold] {len(config.SYMBOLS)} symbols "
        f"({len(config.CORE_SYMBOLS)} core + {len(config.SYMBOLS) - len(config.CORE_SYMBOLS)} extended)"
    )
    console.print(", ".join(config.SYMBOLS[:10]) + f"... and {len(config.SYMBOLS) - 10} more\n")

    return info


# ---------------------------------------------------------------------------
# V10: Functions extracted to engine/ package:
#   sync_positions_with_broker, check_shadow_exits -> engine.broker_sync
#   process_signals, _process_single_signal -> engine.signal_processor
#   handle_strategy_exits, handle_ws_close, get_current_prices -> engine.exit_processor
# ---------------------------------------------------------------------------

# ===========================================================================
# MAIN (synchronous mode)
# ===========================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Velox V11 — Institutional-Grade Quant System")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting engine")
    parser.add_argument("--walkforward", action="store_true", help="Run walk-forward test")
    parser.add_argument("--live", action="store_true", help="Alias for ALPACA_LIVE=true")
    args = parser.parse_args()

    if args.live:
        import os
        os.environ["ALPACA_LIVE"] = "true"

    # V12: Pre-flight check — verify critical systems before trading
    def _preflight_check():
        """Verify critical subsystems are functional before entering main loop."""
        issues = []

        # 1. Database writable
        try:
            database.init_db()
        except Exception as e:
            issues.append(f"Database init failed: {e}")

        # 2. ML model loaded
        try:
            import glob as _g
            import os as _os
            model_dir = _os.path.join(_os.path.dirname(__file__), "models")
            ml_models = _g.glob(_os.path.join(model_dir, "model_*.pkl"))
            if not ml_models:
                issues.append("No trained ML model found in models/ — ML will return neutral (0.5)")
            else:
                logger.info(f"V12 preflight: ML model found: {ml_models[-1]}")
        except Exception:
            pass

        # 3. Strategy allocations sum to 1.0
        alloc_sum = sum(config.STRATEGY_ALLOCATIONS.values())
        if abs(alloc_sum - 1.0) > 0.01:
            issues.append(f"Strategy allocations sum to {alloc_sum:.3f}, not 1.0")

        # 4. Config sanity
        if config.RISK_PER_TRADE_PCT > 0.05:
            issues.append(f"RISK_PER_TRADE_PCT={config.RISK_PER_TRADE_PCT} is dangerously high (>5%)")
        if config.MAX_POSITIONS < 1:
            issues.append(f"MAX_POSITIONS={config.MAX_POSITIONS} — no trades possible")

        # V12 11.4: Startup Self-Test — run synthetic signal through pipeline
        try:
            from strategies.base import Signal
            test_signal = Signal(
                symbol="AAPL",
                side="buy",
                strategy="STAT_MR",
                entry_price=150.0,
                take_profit=153.0,
                stop_loss=148.0,
                confidence=0.7,
            )
            # Test risk manager sizing (dry-run)
            from risk.risk_manager import RiskManager
            test_risk = RiskManager(equity=100_000.0, cash=100_000.0)
            test_qty = test_risk.calculate_position_size(
                entry_price=test_signal.entry_price,
                stop_price=test_signal.stop_loss,
                regime="MEAN_REVERTING",
                strategy="STAT_MR",
            )
            if test_qty > 0:
                logger.info(
                    "V12 11.4: Self-test PASSED — synthetic AAPL STAT_MR signal "
                    "sized to %d shares", test_qty,
                )
            else:
                issues.append("Self-test: RiskManager returned qty=0 for synthetic signal")
        except Exception as e:
            issues.append(f"Self-test pipeline failed: {e}")
            logger.warning("V12 11.4: Self-test failed (non-fatal): %s", e)

        if issues:
            for issue in issues:
                logger.warning(f"V12 PREFLIGHT WARNING: {issue}")
                console.print(f"[yellow]PREFLIGHT: {issue}[/yellow]")
        else:
            logger.info("V12 preflight: All checks passed")
            console.print("[green]V12 preflight: All systems go[/green]")

        return len(issues) == 0

    _preflight_check()

    # Backtest mode
    if args.backtest:
        from backtester import run_backtest
        database.init_db()
        run_backtest()
        return

    # Walk-forward mode
    if args.walkforward:
        from backtester import walk_forward_test
        database.init_db()
        walk_forward_test()
        return

    # IMPL-006: Graceful shutdown sequence with proper signal handling
    _shutdown_requested = False

    def _graceful_shutdown(signame: str = "unknown"):
        """Execute graceful shutdown sequence.

        IMPL-006 shutdown steps:
        1. Set BLACK circuit breaker level (halt all new trading)
        2. Cancel all open orders at broker
        3. Wait for pending fills (up to 10 seconds)
        4. Save full state (positions, analytics, config snapshot)
        5. Flush all log handlers
        6. Exit cleanly
        """
        nonlocal _shutdown_requested
        if _shutdown_requested:
            # Already shutting down, don't re-enter
            return
        _shutdown_requested = True

        console.print(f"\n[yellow]Signal {signame} received — starting graceful shutdown...[/yellow]")
        logger.info(f"Graceful shutdown initiated (signal={signame})")

        # Step 1: Activate BLACK circuit breaker (halt all trading)
        try:
            if kill_switch and hasattr(kill_switch, 'activate'):
                kill_switch.activate(
                    reason=f"graceful_shutdown_{signame}",
                    risk_manager=risk,
                    order_manager=order_manager,
                )
                console.print("[yellow]  1/6 Kill switch activated (BLACK breaker)[/yellow]")
            elif tiered_cb and hasattr(tiered_cb, 'trip'):
                tiered_cb.trip("BLACK", reason=f"graceful_shutdown_{signame}")
                console.print("[yellow]  1/6 Circuit breaker tripped to BLACK[/yellow]")
            else:
                console.print("[yellow]  1/6 No circuit breaker available — skipping[/yellow]")
        except Exception as e:
            logger.warning(f"Shutdown step 1 (kill switch) failed: {e}")

        # Step 2: Cancel all open orders
        try:
            if order_manager and hasattr(order_manager, 'cancel_all'):
                cancelled = order_manager.cancel_all(reason="graceful_shutdown")
                console.print(f"[yellow]  2/6 Cancelled open orders: {cancelled}[/yellow]")
            else:
                from execution import cancel_all_orders
                try:
                    cancel_all_orders()
                    console.print("[yellow]  2/6 Cancelled all open orders via broker[/yellow]")
                except Exception:
                    console.print("[yellow]  2/6 Order cancellation skipped (no OMS)[/yellow]")
        except Exception as e:
            logger.warning(f"Shutdown step 2 (cancel orders) failed: {e}")

        # Step 3: Wait for pending fills (up to 10 seconds)
        try:
            console.print("[yellow]  3/6 Waiting for pending fills (up to 10s)...[/yellow]")
            for _ in range(10):
                time_mod.sleep(1)
                # Check if there are any open orders remaining
                try:
                    if order_manager and hasattr(order_manager, 'get_active_orders'):
                        active = order_manager.get_active_orders()
                        if not active:
                            break
                except Exception:
                    break
            console.print("[yellow]  3/6 Fill wait complete[/yellow]")
        except Exception as e:
            logger.warning(f"Shutdown step 3 (wait fills) failed: {e}")

        # Step 4: Save state
        try:
            database.save_open_positions(risk.open_trades)
            console.print("[yellow]  4/6 State saved to database[/yellow]")
        except Exception as e:
            logger.error(f"Shutdown step 4 (save state) failed: {e}")
            console.print(f"[red]  4/6 State save failed: {e}[/red]")

        # Also save a config snapshot and DB backup
        try:
            from ops.disaster_recovery import DisasterRecovery
            dr = DisasterRecovery(data_dir=str(Path(__file__).resolve().parent))
            dr.save_config_snapshot()
            dr.create_backup("bot.db")
            console.print("[yellow]  4b/6 Config snapshot and DB backup saved[/yellow]")
        except Exception as e:
            logger.debug(f"Shutdown backup failed (non-critical): {e}")

        # Step 5: Stop websocket monitor and EDGAR monitor
        try:
            if ws_monitor:
                ws_monitor.stop()
            console.print("[yellow]  5/6 WebSocket monitor stopped[/yellow]")
        except Exception as e:
            logger.warning(f"Shutdown step 5 (websocket) failed: {e}")
        try:
            if _edgar_monitor:
                _edgar_monitor.stop()
                console.print("[yellow]  5b/6 EDGAR monitor stopped[/yellow]")
        except Exception as e:
            logger.debug(f"Shutdown: EDGAR monitor stop failed: {e}")

        # Step 6: Flush all log handlers
        try:
            for handler in logging.root.handlers:
                handler.flush()
            console.print("[yellow]  6/6 Logs flushed[/yellow]")
        except Exception as e:
            logger.debug(f"Shutdown step 6 (flush logs) failed: {e}")

        console.print("[green]Graceful shutdown complete. Bot stopped.[/green]")

    def _sigterm_handler(signum, frame):
        _graceful_shutdown("SIGTERM")
        sys.exit(0)

    def _sigint_handler(signum, frame):
        _graceful_shutdown("SIGINT")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigint_handler)

    console.print("[bold cyan]Starting Velox V10 Trading Bot...[/bold cyan]\n")

    # Validate configuration
    config.validate()

    # Initialize database
    database.init_db()
    database.run_migrations()
    database.assert_schema_version()  # T2-006: Fail fast if schema is stale
    database.migrate_from_json()

    # Startup checks
    info = startup_checks()

    # V12 14.1: Disaster recovery — check if recovery needed + full state reconciliation
    _dr_instance = None
    try:
        if DisasterRecovery is not None:
            _dr_instance = DisasterRecovery(data_dir=str(Path(__file__).resolve().parent))
            _dr_status = _dr_instance.get_status()
            if _dr_status:
                logger.info("V12 14.1: Disaster recovery status: %s", _dr_status)
            # Check heartbeat staleness — if stale, a crash likely occurred
            heartbeat_ok = _dr_instance.check_heartbeat()
            if not heartbeat_ok:
                logger.warning("V12 14.1: Stale heartbeat detected — previous session may have crashed")
                console.print("[yellow]V12 14.1: Stale heartbeat — running state recovery...[/yellow]")
            _dr_instance.update_heartbeat()
    except Exception as e:
        logger.info("V12 14.1: Disaster recovery boot check skipped (fail-open): %s", e)

    # V10: Consolidated initialization via engine/startup.py
    from engine.startup import (
        initialize_strategies, initialize_risk_engines,
        initialize_optional_modules, initialize_v10_components,
        initialize_risk_manager, initialize_websocket, initialize_dashboard,
    )

    strats = initialize_strategies()
    stat_mr = strats["stat_mr"]
    kalman_pairs = strats["kalman_pairs"]
    micro_mom = strats["micro_mom"]
    vwap_strategy = strats["vwap_strategy"]
    orb_strategy = strats["orb_strategy"]
    pead_strategy = strats.get("pead_strategy")

    engines = initialize_risk_engines()
    vol_engine = engines["vol_engine"]
    pnl_lock = engines["pnl_lock"]
    beta_neutral = engines["beta_neutral"]
    regime_detector = engines["regime_detector"]

    # V10: PDT rule enforcement
    from risk.pdt_tracker import PDTTracker
    pdt = PDTTracker()

    mods = initialize_optional_modules()
    news_sentiment = mods["news_sentiment"]
    llm_scorer = mods["llm_scorer"]
    walk_forward = mods.get("walk_forward")
    overnight_manager = mods.get("overnight_manager")
    cross_asset_monitor = mods.get("cross_asset_monitor")
    signal_ranker = mods.get("signal_ranker")
    alpha_decay_monitor = mods.get("alpha_decay_monitor")
    adaptive_allocator = mods.get("adaptive_allocator")
    param_optimizer = mods.get("param_optimizer")
    watchdog = mods.get("watchdog")
    reconciler = mods.get("reconciler")

    last_cross_asset_update = now_et()
    last_correlation_refresh = now_et()  # V12 AUDIT: Track intraday correlation recompute
    last_watchdog_check = now_et()
    last_reconciliation = now_et()

    risk = initialize_risk_manager(info["equity"], info["cash"])

    # V12 6.5: Recover positions from broker on startup (mid-day restart safety)
    try:
        from data import get_trading_client as _get_tc
        _tc = _get_tc()
        _broker_positions = _tc.get_all_positions()
        _db_symbols = set(risk.open_trades.keys())
        _broker_symbols = {p.symbol for p in _broker_positions}
        _adopted = 0
        _orphaned = 0

        # Adopt broker positions not in DB
        for bp in _broker_positions:
            if bp.symbol not in _db_symbols:
                try:
                    from risk.risk_manager import TradeRecord
                    _side = "buy" if float(bp.qty) > 0 else "sell"
                    _trade = TradeRecord(
                        symbol=bp.symbol,
                        strategy="UNKNOWN_ADOPTED",
                        side=_side,
                        entry_price=float(bp.avg_entry_price),
                        entry_time=now_et(),
                        qty=abs(int(float(bp.qty))),
                        take_profit=0.0,
                        stop_loss=0.0,
                        pnl=float(bp.unrealized_pl),
                        status="open",
                        order_id="adopted",
                        hold_type="day",
                    )
                    risk.register_trade(_trade)
                    _adopted += 1
                    logger.warning(
                        "V12 6.5: ADOPTED broker position %s (%s %s @ $%.2f) — "
                        "not in DB, likely from pre-crash state",
                        bp.symbol, _side, bp.qty, float(bp.avg_entry_price),
                    )
                except Exception as _ae:
                    logger.error("V12 6.5: Failed to adopt %s: %s", bp.symbol, _ae)

        # Flag DB positions not at broker (orphans)
        for sym in _db_symbols:
            if sym not in _broker_symbols:
                _orphaned += 1
                logger.warning(
                    "V12 6.5: ORPHAN position %s — in DB but not at broker. "
                    "Marking as closed.",
                    sym,
                )
                try:
                    risk.close_trade(sym, 0.0, now_et(), exit_reason="orphan_recovery")
                except Exception:
                    pass

        if _adopted or _orphaned:
            logger.warning(
                "V12 6.5: Position recovery complete — %d adopted, %d orphaned",
                _adopted, _orphaned,
            )
            if notifications and config.TELEGRAM_ENABLED:
                try:
                    notifications.send_alert(
                        f"Position Recovery: {_adopted} adopted, {_orphaned} orphaned",
                        severity="WARNING",
                    )
                except Exception:
                    pass
        else:
            logger.info("V12 6.5: Position recovery — all positions in sync")
    except Exception as _e:
        logger.warning("V12 6.5: Position recovery failed (non-fatal): %s", _e)

    v10 = initialize_v10_components()
    order_manager = v10["order_manager"]
    tiered_cb = v10["tiered_cb"]
    kill_switch = v10["kill_switch"]
    var_monitor = v10["var_monitor"]
    corr_limiter = v10["corr_limiter"]

    # V12 2.9: Track VIX values over a rolling 15-minute window for spike detection.
    # Each entry is (timestamp_epoch, vix_value).  Checked every scan cycle.
    _vix_history: collections.deque = collections.deque()
    VIX_SPIKE_WINDOW_SEC = getattr(config, 'VIX_SPIKE_WINDOW_SEC', 15 * 60)
    VIX_SPIKE_THRESHOLD_PCT = getattr(config, 'VIX_SPIKE_THRESHOLD_PCT', 0.20)

    ws_monitor = initialize_websocket(risk)
    initialize_dashboard(order_manager, kill_switch, tiered_cb)

    # V12 BONUS: Initialize profit maximization components
    _intraday_vol_regime = IntradayVolRegime() if IntradayVolRegime else None
    _win_streak_tracker = WinStreakTracker() if WinStreakTracker else None
    if _intraday_vol_regime:
        logger.info("V12 BONUS: IntradayVolRegime + WinStreakTracker initialized")

    # V11.3 T2: Initialize intraday risk controls (velocity + rolling P&L limits)
    intraday_controls = None
    if IntradayRiskControls:
        try:
            intraday_controls = IntradayRiskControls()
            # Share instance with broker_sync and exit_processor
            from engine.broker_sync import set_intraday_controls as _set_irc_broker
            _set_irc_broker(intraday_controls)
            from engine.exit_processor import set_intraday_controls as _set_irc_exit
            _set_irc_exit(intraday_controls)
            logger.info("V11.3: Intraday risk controls initialized (5m/30m/1h windows + velocity)")
        except Exception as e:
            logger.warning(f"V11.3: Intraday risk controls init failed (non-fatal): {e}")

    # V11.4: Black-Litterman portfolio optimization
    bl_optimizer = None
    if BlackLittermanOptimizer and getattr(config, "BLACK_LITTERMAN_ENABLED", False):
        try:
            bl_optimizer = BlackLittermanOptimizer()
            logger.info("V11.4: Black-Litterman optimizer initialized")
        except Exception as e:
            logger.warning(f"V11.4: Black-Litterman init failed (non-fatal): {e}")

    # V12 Item 2.3: Initialize data feed monitor for outage detection
    feed_monitor = get_feed_monitor()
    logger.info("V12-2.3: DataFeedMonitor initialized")

    # Load filters
    try:
        load_earnings_cache(config.SYMBOLS)
        load_correlation_cache(config.SYMBOLS)
    except Exception as e:
        logger.warning(f"Filter cache load failed: {e}")

    # --- Loop state ---
    start_time = now_et()
    last_scan = None
    last_state_save = now_et()
    last_analytics_update = now_et()
    last_day = now_et().date()
    eod_summary_printed = False
    current_analytics = None
    universe_prepared_today = False
    universe_refreshed_today = False  # V12-5.4: track 8:30 AM universe refresh
    last_sunday_task = None
    latest_consistency_score = 0.0
    last_corp_action_check = now_et()
    reconciled_at_open_today = False
    reconciled_at_close_today = False

    # --- V12 2.5: Corporate action detector (fail-open) ---
    corp_action_detector = None
    if CorporateActionDetector:
        try:
            corp_action_detector = CorporateActionDetector()
            logger.info("V12 2.5: CorporateActionDetector initialized")
        except Exception as e:
            logger.warning(f"V12 2.5: CorporateActionDetector init failed (fail-open): {e}")

    # Feature flags
    features = ["MR40%", "VWAP20%", "PAIRS20%", "PEAD10%", "ORB5%", "MICRO5%"]
    if config.ALLOW_SHORT:
        features.append("Short")
    if config.TELEGRAM_ENABLED:
        features.append("Notify")
    if config.WEB_DASHBOARD_ENABLED:
        features.append("Web")
    if config.WEBSOCKET_MONITORING:
        features.append("WS")

    # --- V11: Initialize modules via Container (ARCH-011) ---
    # All V11 modules are registered as lazy factories in Container.
    # We eagerly resolve each one here to detect import/init errors at startup.
    # The fail-open pattern is preserved: each module is individually wrapped.
    from container import Container as _Container
    _ctr = _Container.instance()

    # --- V12 3.1: Enhanced reconciler from Container (fail-open) ---
    # V12 AUDIT: Moved after _ctr initialization to avoid NameError
    v11_reconciler = None
    if V11Reconciler:
        try:
            v11_reconciler = _ctr.get("v11_reconciler")
            logger.info("V12 3.1: V11 PositionReconciler initialized")
        except Exception as e:
            logger.warning(f"V12 3.1: V11 PositionReconciler init failed (fail-open): {e}")

    _V11_MODULE_FEATURES = [
        # (container_key, feature_tag)
        ("intraday_controls", "IntradayRisk"),
        ("factor_model", "FactorRisk"),
        ("stress_test", "StressTest"),
        ("gap_risk", "GapRisk"),
        ("drawdown_risk", "DrawdownRisk"),
        ("fill_analytics", "FillAnalytics"),
        ("slippage_model", "SlippageModel"),
        ("vpin", "VPIN"),
        ("feature_store", "FeatureStore"),
        ("feature_engine", "ML-Features"),
        ("alert_manager", "Alerting"),
        ("latency_tracker", "LatencyMon"),
        ("metrics_pipeline", "MetricsPipe"),
        ("audit_trail", "AuditTrail"),
        ("pdt_compliance", "PDTv11"),
        ("surveillance", "Surveillance"),
        ("bocpd", "BOCPD"),
        ("enhanced_seasonality", "SeasonV11"),
        ("data_quality", "DQv11"),
        ("v11_reconciler", "ReconV11"),
        # V11.1 new modules
        ("batch_inference", "BatchInference"),
        ("model_registry", "ModelRegistry"),
        ("watchdog", "Watchdog"),
        ("shadow_trader", "ShadowTrader"),
        ("dynamic_hedger", "DynHedge"),
        ("margin_monitor", "MarginMon"),
        ("corporate_actions", "CorpActions"),
        ("conformal_stops", "ConfStops"),
    ]

    v11_active = 0
    for _key, _feat in _V11_MODULE_FEATURES:
        try:
            _ctr.get(_key)
            features.append(_feat)
            v11_active += 1
        except Exception as e:
            logger.warning(f"V11 {_key} init failed: {e}")

    # Disaster recovery gets special treatment — run state recovery on startup
    try:
        dr = _ctr.get("disaster_recovery")
        try:
            dr.recover_state()
        except Exception as e:
            logger.warning(f"V11 state recovery incomplete: {e}")
        features.append("DR")
        v11_active += 1
    except Exception as e:
        logger.warning(f"V11 disaster_recovery init failed: {e}")

    logger.info(f"V11 modules initialized: {v11_active} active (via Container)")

    # --- T2-005: Register live instances into the Container ---
    # This ensures that V11 modules resolved through the container can access
    # the same risk_manager, vol_engine, etc. that the main loop uses.
    _ctr.register_instance("risk_manager", risk)
    _ctr.register_instance("vol_engine", vol_engine)
    _ctr.register_instance("pnl_lock", pnl_lock)
    if order_manager:
        _ctr.register_instance("oms", order_manager)
    if tiered_cb:
        _ctr.register_instance("circuit_breaker", tiered_cb)
    logger.info("T2-005: Live instances registered into Container (risk_manager, vol_engine, pnl_lock, oms, circuit_breaker)")

    # --- T7-003: Start EDGAR 8-K monitor ---
    _edgar_monitor = None
    if getattr(config, "EDGAR_MONITOR_ENABLED", False):
        try:
            from data.alternative.sec_filings import EdgarMonitor
            from engine.signal_processor import set_edgar_monitor
            _edgar_monitor = EdgarMonitor(universe=list(config.UNIVERSE))
            set_edgar_monitor(_edgar_monitor)
            _edgar_monitor.start()
            features.append("EDGAR-8K")
            v11_active += 1
            logger.info("T7-003: EDGAR 8-K monitor started")
        except Exception as e:
            logger.warning(f"T7-003: EDGAR monitor init failed (fail-open): {e}")

    features_str = ", ".join(features)
    console.print(f"\n[bold green]Velox V11 is running. Press Ctrl+C to stop.[/bold green]")
    console.print(f"[dim]Strategies: STAT_MR + VWAP + KALMAN_PAIRS + ORB + MICRO_MOM + PEAD[/dim]")
    console.print(f"[dim]V11 Modules: {v11_active} active[/dim]")
    console.print(f"[dim]Features: {features_str}[/dim]\n")

    # -----------------------------------------------------------------------
    # Initialize strategies at startup (don't wait for scheduled times)
    # -----------------------------------------------------------------------
    startup_now = now_et()
    if is_market_hours(startup_now.time()) or startup_now.time() >= config.MR_UNIVERSE_PREP_TIME:
        console.print("[cyan]Preparing StatMR universe at startup...[/cyan]")
        try:
            stat_mr.prepare_universe(config.STANDARD_SYMBOLS, startup_now)
            universe_prepared_today = True
            console.print(f"[green]StatMR universe ready: {len(stat_mr.universe)} symbols[/green]")
            # Save OU params to database
            for sym, ou in stat_mr.ou_params.items():
                try:
                    database.save_ou_parameters(sym, ou)
                except Exception as e:
                    logger.debug(f"Failed to save OU params for {sym}: {e}")
        except Exception as e:
            console.print(f"[yellow]StatMR universe prep failed: {e}[/yellow]")

    # Initialize Kalman pairs if empty or stale
    try:
        active_pairs_count = len(database.get_active_kalman_pairs())
    except Exception as e:
        logger.warning(f"Failed to check active Kalman pairs count: {e}")
        active_pairs_count = 0

    if active_pairs_count == 0:
        console.print("[cyan]Initializing Kalman pairs at startup (none active)...[/cyan]")
        try:
            kalman_pairs.select_pairs_weekly(startup_now)
            if kalman_pairs.active_pairs:
                console.print(f"[green]Kalman pairs ready: {len(kalman_pairs.active_pairs)} pairs[/green]")
            else:
                console.print("[yellow]Kalman pairs: no cointegrated pairs found — will retry at next weekly selection[/yellow]")
                logger.warning("No cointegrated pairs found at startup — KalmanPairs strategy will be idle until pairs are found")
        except Exception as e:
            console.print(f"[yellow]Kalman pairs init failed: {e}[/yellow]")
            logger.error(f"Kalman pairs init error: {e}", exc_info=True)
    else:
        # Load existing pairs from DB into memory
        try:
            db_pairs = database.get_active_kalman_pairs()
            # CRIT-011: Keep as dicts (scan() expects dict with 'symbol1'/'symbol2' keys)
            kalman_pairs.active_pairs = db_pairs
            # Restore Kalman state from DB values
            import numpy as _np
            for p in db_pairs:
                pair_key = f"{p['symbol1']}_{p['symbol2']}"
                hr = float(p.get('hedge_ratio', 1.0))
                kalman_pairs.kalman_state[pair_key] = {
                    'theta': _np.array([hr, 0.0]),
                    'P': _np.eye(2) * 1.0,
                    'spread_mean': float(p.get('spread_mean', 0.0)),
                    'spread_std': max(0.001, float(p.get('spread_std', 1.0))),
                }
            kalman_pairs._pairs_ready = True
            console.print(f"[green]Kalman pairs loaded from DB: {len(kalman_pairs.active_pairs)} pairs[/green]")
        except Exception as e:
            console.print(f"[yellow]Failed to load Kalman pairs from DB: {e}[/yellow]")

    # Stop logging to terminal -- dashboard takes over
    logging.getLogger().removeHandler(_stream_handler)

    try:
        with Live(
            build_dashboard(
                risk, regime_detector.regime, start_time, now_et(), last_scan,
                len(config.SYMBOLS), current_analytics,
                pnl_lock_state=pnl_lock.state.value,
                vol_scalar=vol_engine.last_scalar,
                portfolio_beta=beta_neutral.portfolio_beta,
                consistency_score=latest_consistency_score,
            ),
            console=console,
            refresh_per_second=0.2,
            transient=False,
        ) as live:
            while True:
                current = now_et()
                current_time = current.time()

                # -------------------------------------------------------
                # Daily reset
                # -------------------------------------------------------
                if current.date() != last_day:
                    daily_reset(
                        risk, stat_mr, kalman_pairs, micro_mom, vwap_strategy,
                        pnl_lock, beta_neutral,
                        orb_strategy=orb_strategy, pead_strategy=pead_strategy,
                        overnight_manager=overnight_manager, news_sentiment=news_sentiment,
                        llm_scorer=llm_scorer, tiered_cb=tiered_cb,
                    )
                    # V11.3 T2: Reset intraday controls for new day
                    if intraday_controls:
                        intraday_controls.reset_daily()
                    # V12 2.9: Clear VIX spike history at day boundary
                    _vix_history.clear()
                    universe_prepared_today = False
                    reconciled_at_open_today = False
                    reconciled_at_close_today = False
                    universe_refreshed_today = False  # V12-5.4: reset for new day
                    last_day = current.date()
                    eod_summary_printed = False

                # -------------------------------------------------------
                # V12-5.4: Universe refresh at 8:30 AM ET
                # -------------------------------------------------------
                if not universe_refreshed_today and current_time >= time(8, 30):
                    universe_refreshed_today = True
                    try:
                        from data.universe import get_dynamic_universe
                        _universe = get_dynamic_universe()
                        count = _universe.refresh_daily()
                        logger.info("V12-5.4: Universe refreshed at 8:30 AM — %d symbols", count)
                    except Exception as e:
                        logger.error("V12-5.4: Universe refresh failed: %s", e)

                # -------------------------------------------------------
                # Sunday tasks — weekly pair selection, optimization, validation
                # -------------------------------------------------------
                if (current.weekday() == 6 and current_time >= time(0, 0)
                        and last_sunday_task != current.date()):
                    last_sunday_task = current.date()
                    weekly_tasks(current, kalman_pairs, param_optimizer, walk_forward,
                                hmm_detector=regime_detector)

                # -------------------------------------------------------
                # Update regime
                # -------------------------------------------------------
                regime = regime_detector.update(current)

                # -------------------------------------------------------
                # Market hours logic
                # -------------------------------------------------------
                if is_market_hours(current_time):

                    # 9:00 AM tasks: universe prep + Monday pair selection
                    if not universe_prepared_today and current_time >= config.MR_UNIVERSE_PREP_TIME:
                        # V10: Dynamic universe selection based on regime
                        scan_symbols = config.STANDARD_SYMBOLS
                        try:
                            from strategies.dynamic_universe import DynamicUniverse
                            _dyn_universe = DynamicUniverse()
                            selection = _dyn_universe.select(regime=regime)
                            if selection.symbols:
                                scan_symbols = selection.symbols
                                logger.info(
                                    f"Dynamic universe: {len(scan_symbols)} symbols "
                                    f"(+{len(selection.added)} -{len(selection.removed)}) regime={regime}"
                                )
                        except Exception as e:
                            logger.debug(f"Dynamic universe failed (using static): {e}")

                        # T4-001: Pre-warm bar cache for all universe symbols
                        try:
                            from data.bar_cache import bar_cache
                            bar_cache.pre_warm(scan_symbols)
                        except Exception as e:
                            logger.debug(f"T4-001: Bar cache pre-warm failed (non-critical): {e}")

                        try:
                            stat_mr.prepare_universe(scan_symbols, current)
                            universe_prepared_today = True
                            logger.info(f"MR universe prepared: {len(stat_mr.universe)} symbols")
                            for sym, ou in stat_mr.ou_params.items():
                                try:
                                    database.save_ou_parameters(sym, ou)
                                except Exception as e:
                                    logger.debug(f"Failed to save OU params for {sym}: {e}")
                        except Exception as e:
                            logger.error(f"MR universe prep failed: {e}")

                        # Monday: refresh pairs
                        if current.weekday() == 0:
                            try:
                                kalman_pairs.select_pairs_weekly(current)
                                logger.info(f"Monday pair selection: {len(kalman_pairs.active_pairs)} pairs")
                            except Exception as e:
                                logger.error(f"Monday pair selection failed: {e}")

                    # V7: ORB opening range recording at 10:00 AM
                    if orb_strategy and current_time >= time(10, 0) and not getattr(orb_strategy, '_ranges_recorded_today', False):
                        try:
                            from data import get_intraday_bars
                            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
                            market_open = datetime(current.year, current.month, current.day, 9, 30, tzinfo=config.ET)
                            orb_10am = datetime(current.year, current.month, current.day, 10, 0, tzinfo=config.ET)
                            for symbol in config.STANDARD_SYMBOLS[:config.ORB_SCAN_SYMBOLS]:
                                try:
                                    bars_930_1000 = get_intraday_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), start=market_open, end=orb_10am)
                                    if bars_930_1000 is not None and not bars_930_1000.empty:
                                        orb_strategy.record_opening_range(symbol, bars_930_1000)
                                except Exception as e:
                                    logger.debug(f"ORB range recording failed for {symbol}: {e}")
                            orb_strategy._ranges_recorded_today = True
                            logger.info(f"ORB opening ranges recorded: {len(orb_strategy.opening_ranges)} symbols")
                        except Exception as e:
                            logger.error(f"ORB opening range recording failed: {e}")

                    # V12 3.1: Reconciliation at market open (once per day)
                    if not reconciled_at_open_today:
                        reconciled_at_open_today = True
                        if v11_reconciler:
                            try:
                                report = v11_reconciler.reconcile()
                                if report.has_issues:
                                    logger.warning(
                                        f"V12 3.1: Open reconciliation found {len(report.discrepancies)} "
                                        f"discrepancies ({report.auto_healed_count} auto-healed)"
                                    )
                                else:
                                    logger.info("V12 3.1: Open reconciliation — all positions reconciled")
                            except Exception as e:
                                logger.error(f"V12 3.1: Open reconciliation failed (fail-open): {e}")
                        if reconciler:
                            try:
                                reconciler.reconcile()
                                last_reconciliation = current
                            except Exception as e:
                                logger.error(f"V12 3.1: V9 open reconciliation failed: {e}")

                    # Trading hours scan
                    if is_trading_hours(current_time):
                        # 1. Update PnL lock state
                        pnl_lock.update(risk.day_pnl)

                        # V10: Tiered circuit breaker (replaces single-threshold)
                        skip_scan = False
                        # V12 FINAL: Initialize to 0.0 as safe default. When tiered_cb is None
                        # (e.g. during tests or if circuit breaker init fails), strategies receive
                        # day_pnl_pct=0.0 which is conservative (assumes no P&L movement).
                        day_pnl_pct = 0.0
                        if tiered_cb:
                            day_pnl_pct = risk.day_pnl / max(risk.current_equity, 1)
                            tier = tiered_cb.update(day_pnl_pct, current)

                            # V12 2.9: VIX spike circuit breaker — escalate to
                            # ORANGE if VIX rises >20 % within 15 minutes, even
                            # when realized P&L hasn't triggered a tier yet.
                            try:
                                from risk import get_vix_level as _get_vix
                                from risk.circuit_breaker import CircuitTier
                                _cur_vix = _get_vix()
                                _now_ts = time_mod.time()
                                if _cur_vix > 0:
                                    _vix_history.append((_now_ts, _cur_vix))
                                    # Purge entries older than the 15-min window
                                    _cutoff = _now_ts - VIX_SPIKE_WINDOW_SEC
                                    while _vix_history and _vix_history[0][0] < _cutoff:
                                        _vix_history.popleft()
                                    # Check for spike: compare oldest value in window to current
                                    if len(_vix_history) >= 2:
                                        _oldest_vix = _vix_history[0][1]
                                        if _oldest_vix > 0:
                                            _vix_change_pct = (_cur_vix - _oldest_vix) / _oldest_vix
                                            if _vix_change_pct >= VIX_SPIKE_THRESHOLD_PCT:
                                                tiered_cb.escalate_to(
                                                    CircuitTier.ORANGE,
                                                    reason=f"VIX spike {_vix_change_pct:.1%} in 15min "
                                                           f"({_oldest_vix:.1f} -> {_cur_vix:.1f})",
                                                )
                                                tier = tiered_cb.current_tier
                                                logger.warning(
                                                    "V12 2.9: VIX spike detected — %.1f -> %.1f "
                                                    "(+%.1f%%) in 15min — circuit breaker escalated to %s",
                                                    _oldest_vix, _cur_vix,
                                                    _vix_change_pct * 100, tier.name,
                                                )
                            except Exception as _vix_err:
                                logger.debug("V12 2.9: VIX spike check failed (non-fatal): %s", _vix_err)

                            if tiered_cb.should_close_all and kill_switch:
                                try:
                                    kill_switch.activate("tiered_cb_black", risk_manager=risk, order_manager=order_manager)
                                except Exception as ks_err:
                                    # V12 2.8: Kill switch failed — fall back to direct
                                    # synchronous market-order close of all positions via
                                    # the broker API so we never hold unmanaged positions.
                                    logger.critical(
                                        "Kill switch activation FAILED (%s) — "
                                        "falling back to direct position closure",
                                        ks_err, exc_info=True,
                                    )
                                    try:
                                        from execution import close_position as _emergency_close
                                        for _sym in list(risk.open_trades.keys()):
                                            try:
                                                _emergency_close(_sym, reason="kill_switch_fallback")
                                                _t = risk.open_trades.get(_sym)
                                                if _t:
                                                    risk.close_trade(
                                                        _sym, _t.entry_price, current,
                                                        exit_reason="kill_switch_fallback",
                                                    )
                                                logger.info("Kill switch fallback: closed %s", _sym)
                                            except Exception as close_err:
                                                logger.critical(
                                                    "Kill switch fallback: FAILED to close %s: %s "
                                                    "— MANUAL INTERVENTION REQUIRED",
                                                    _sym, close_err,
                                                )
                                    except Exception as import_err:
                                        logger.critical(
                                            "Kill switch fallback: cannot import execution module: %s "
                                            "— MANUAL INTERVENTION REQUIRED",
                                            import_err,
                                        )
                                skip_scan = True
                            elif tiered_cb.should_close_day_trades:
                                # Red tier: close day-hold positions
                                for sym in list(risk.open_trades.keys()):
                                    t = risk.open_trades[sym]
                                    if t.hold_type == "day":
                                        try:
                                            from execution import close_position as _close_pos
                                            _close_pos(sym, reason="circuit_breaker_red")
                                            risk.close_trade(sym, t.entry_price, current, exit_reason="circuit_breaker_red")
                                        except Exception as e:
                                            logger.error(f"Circuit breaker close failed for {sym}: {e}", exc_info=True)
                                skip_scan = True
                            elif not tiered_cb.allow_new_entries:
                                skip_scan = True
                        elif risk.check_circuit_breaker():
                            # Fallback to legacy single-threshold
                            skip_scan = True

                        if skip_scan:
                            # V12 FINAL: Define tier_name before any conditional block
                            tier_name = tiered_cb.current_tier.name if tiered_cb else "ACTIVE"
                            if notifications and config.TELEGRAM_ENABLED:
                                try:
                                    notifications.notify_circuit_breaker(risk.day_pnl)
                                except Exception as e:
                                    logger.warning(f"Circuit breaker notification failed: {e}")
                            logger.warning(f"Circuit breaker {tier_name} — skipping scan cycle (day P&L: {risk.day_pnl:.2f})")
                            last_scan = current
                            continue

                        # V12-2.3: Data feed health check — probe snapshot API
                        # for open positions (or core symbols) to detect outages.
                        try:
                            _probe_symbols = list(risk.open_trades.keys()) or config.CORE_SYMBOLS[:5]
                            _probe_succeeded = []
                            _probe_failed = []
                            _probe_prices: dict[str, float] = {}
                            try:
                                _probe_snaps = get_snapshots(_probe_symbols)
                                for _ps in _probe_symbols:
                                    _snap = _probe_snaps.get(_ps)
                                    if _snap and _snap.latest_trade:
                                        _probe_succeeded.append(_ps)
                                        _probe_prices[_ps] = float(_snap.latest_trade.price)
                                    else:
                                        _probe_failed.append(_ps)
                            except Exception as _probe_err:
                                logger.warning("V12-2.3: Feed probe failed entirely: %s", _probe_err)
                                _probe_failed = _probe_symbols
                            feed_monitor.report_cycle(
                                succeeded=_probe_succeeded,
                                failed=_probe_failed,
                                prices=_probe_prices,
                            )
                        except Exception as _fm_err:
                            logger.debug("V12-2.3: Feed monitor update failed (non-fatal): %s", _fm_err)

                        # 2-3. Scan all strategies + detect events
                        # V12-2.3: Block new signal generation when feed is down
                        if feed_monitor.is_feed_down:
                            signals = []
                            logger.warning(
                                "V12-2.3: Feed DOWN (cycle %d) — skipping signal scan, "
                                "exit checks use cached prices",
                                feed_monitor.consecutive_down_cycles,
                            )
                        else:
                            signals = scan_all_strategies(
                                current, regime, stat_mr, kalman_pairs, micro_mom,
                                vwap_strategy, orb_strategy, pead_strategy, signal_ranker,
                                day_pnl_pct=day_pnl_pct,
                            )

                        # V12-2.3: Apply stale-param confidence reduction after recovery
                        if signals and feed_monitor.are_params_stale():
                            _conf_mult = feed_monitor.confidence_multiplier
                            for _sig in signals:
                                if hasattr(_sig, 'confidence') and _sig.confidence is not None:
                                    _sig.confidence *= _conf_mult
                            logger.info(
                                "V12-2.3: OU params stale — signal confidence reduced "
                                "by %.0f%% (%d signals)",
                                (1 - _conf_mult) * 100, len(signals),
                            )

                        # 4. Process signals (with intraday risk control gate)
                        if signals and intraday_controls:
                            allowed, irc_reason = intraday_controls.should_allow_trade(current)
                            if not allowed:
                                logger.warning(f"V11.3 intraday controls blocking signals: {irc_reason}")
                                signals = []  # Block all new entries
                        if signals:
                            process_signals(
                                signals, risk, regime, current,
                                vol_engine, pnl_lock, ws_monitor,
                                news_sentiment, llm_scorer,
                                regime_detector, cross_asset_monitor,
                                var_monitor, corr_limiter,
                            )

                        # 5. Check strategy exits (always runs — uses cached prices if feed down)
                        # V12 FINAL: ExitOrchestrator is primary; legacy is fallback
                        if _exit_orchestrator is not None:
                            try:
                                # Build strategies dict for orchestrator
                                _strat_dict = {
                                    "STAT_MR": stat_mr,
                                    "KALMAN_PAIRS": kalman_pairs,
                                    "MICRO_MOM": micro_mom,
                                }
                                if orb_strategy:
                                    _strat_dict["ORB"] = orb_strategy
                                if pead_strategy:
                                    _strat_dict["PEAD"] = pead_strategy

                                exit_actions = _exit_orchestrator.check_exits(
                                    risk_manager=risk,
                                    now=current,
                                    strategies=_strat_dict,
                                    ws_monitor=ws_monitor,
                                )
                                if exit_actions:
                                    _exit_orchestrator.execute_exits(
                                        exit_actions, risk, current, ws_monitor,
                                    )
                                    logger.info(
                                        "V12 FINAL: ExitOrchestrator processed %d exit actions",
                                        len(exit_actions),
                                    )
                                elif risk.open_trades:
                                    logger.debug(
                                        "V12 FINAL: ExitOrchestrator returned 0 actions "
                                        "(%d open positions checked)",
                                        len(risk.open_trades),
                                    )
                            except Exception as oe:
                                logger.warning(
                                    "V12 FINAL: ExitOrchestrator failed (%s), "
                                    "falling back to legacy exits", oe,
                                )
                                check_all_exits(
                                    current, risk, stat_mr, kalman_pairs, micro_mom,
                                    orb_strategy, pead_strategy, ws_monitor,
                                )
                                if _check_advanced_exits is not None:
                                    try:
                                        adv_exits = _check_advanced_exits(risk, current)
                                        if adv_exits:
                                            handle_strategy_exits(adv_exits, risk, current, ws_monitor)
                                    except Exception as e:
                                        logger.error("Advanced exits failed: %s", e)
                        else:
                            # No orchestrator — use legacy exits
                            check_all_exits(
                                current, risk, stat_mr, kalman_pairs, micro_mom,
                                orb_strategy, pead_strategy, ws_monitor,
                            )
                            if _check_advanced_exits is not None:
                                try:
                                    adv_exits = _check_advanced_exits(risk, current)
                                    if adv_exits:
                                        handle_strategy_exits(adv_exits, risk, current, ws_monitor)
                                except Exception as e:
                                    logger.error("Advanced exits failed: %s", e)

                        # 6. Beta neutralization
                        run_beta_neutralization(
                            current, risk, beta_neutral, vol_engine, pnl_lock,
                            ws_monitor, regime,
                        )

                        # V9: Cross-asset monitor update (every 15 min)
                        if cross_asset_monitor:
                            if (current - last_cross_asset_update).total_seconds() >= config.CROSS_ASSET_UPDATE_INTERVAL:
                                try:
                                    cross_asset_monitor.update(current)
                                    last_cross_asset_update = current
                                except Exception as e:
                                    logger.error(f"Cross-asset update failed: {e}")

                        # V12 AUDIT: Recompute correlation matrix every 60 minutes (not just at open)
                        if (current - last_correlation_refresh).total_seconds() >= getattr(config, 'CORRELATION_REFRESH_INTERVAL_SEC', 3600):
                            try:
                                load_correlation_cache(config.SYMBOLS)
                                last_correlation_refresh = current
                                logger.info("V12 AUDIT: Intraday correlation matrix refreshed")
                            except Exception as e:
                                logger.debug("V12 AUDIT: Correlation refresh failed: %s", e)

                        # V9: Watchdog health check
                        if watchdog:
                            if (current - last_watchdog_check).total_seconds() >= config.WATCHDOG_CHECK_INTERVAL:
                                try:
                                    health = watchdog.check_health()
                                    if not health.overall_healthy:
                                        for issue in health.issues:
                                            watchdog.recover(issue)
                                    last_watchdog_check = current
                                except Exception as e:
                                    logger.error(f"Watchdog check failed: {e}")

                        # V9 + V12 3.1: Position reconciliation (every 30 min)
                        if (current - last_reconciliation).total_seconds() >= config.RECONCILIATION_INTERVAL:
                            if reconciler:
                                try:
                                    reconciler.reconcile()
                                except Exception as e:
                                    logger.error(f"Reconciliation failed: {e}")
                            if v11_reconciler:
                                try:
                                    _recon_report = v11_reconciler.reconcile()
                                    if _recon_report.has_issues:
                                        logger.warning(
                                            f"V12 3.1: Reconciliation found {len(_recon_report.discrepancies)} "
                                            f"discrepancies ({_recon_report.auto_healed_count} auto-healed)"
                                        )
                                except Exception as e:
                                    logger.error(f"V12 3.1: V11 reconciliation failed (fail-open): {e}")
                            last_reconciliation = current

                        # V12 2.5: Corporate action check (every 30 min)
                        if corp_action_detector:
                            if (current - last_corp_action_check).total_seconds() >= getattr(config, 'CORP_ACTION_CHECK_INTERVAL_SEC', 1800):
                                try:
                                    open_symbols = list(risk.open_trades.keys())
                                    if open_symbols:
                                        actions = corp_action_detector.check_actions(open_symbols)
                                        if actions:
                                            for action in actions:
                                                logger.warning(
                                                    f"V12 2.5: Corporate action detected — "
                                                    f"{action.symbol} {action.action_type.value} "
                                                    f"effective {action.effective_date}"
                                                )
                                            adjustments = corp_action_detector.apply_adjustments(
                                                actions, risk.open_trades,
                                            )
                                            if adjustments:
                                                for adj in adjustments:
                                                    logger.info(
                                                        f"V12 2.5: Position adjusted — "
                                                        f"{adj.symbol} {adj.reason}"
                                                    )
                                except Exception as e:
                                    logger.error(f"V12 2.5: Corporate action check failed (fail-open): {e}")
                                last_corp_action_check = current

                        # Check shadow exits
                        check_shadow_exits(current)

                    # EOD: close day-hold positions + respect overnight holds (BUG-034, BUG-039)
                    if current_time >= config.ORB_EXIT_TIME:
                        eod_close(current, risk, ws_monitor, overnight_manager, regime)

                        # V12 3.1: Reconciliation at market close (once per day)
                        if not reconciled_at_close_today:
                            reconciled_at_close_today = True
                            if v11_reconciler:
                                try:
                                    _close_report = v11_reconciler.reconcile()
                                    if _close_report.has_issues:
                                        logger.warning(
                                            f"V12 3.1: Close reconciliation found "
                                            f"{len(_close_report.discrepancies)} discrepancies "
                                            f"({_close_report.auto_healed_count} auto-healed)"
                                        )
                                    else:
                                        logger.info("V12 3.1: Close reconciliation — all positions reconciled")
                                except Exception as e:
                                    logger.error(f"V12 3.1: Close reconciliation failed (fail-open): {e}")
                            if reconciler:
                                try:
                                    reconciler.reconcile()
                                    last_reconciliation = current
                                except Exception as e:
                                    logger.error(f"V12 3.1: V9 close reconciliation failed: {e}")

                    # Sync with broker
                    # V10 BUG-026: Always sync with broker regardless of WebSocket state
                    # WebSocket handles real-time updates; sync handles drift correction
                    sync_positions_with_broker(risk, current, ws_monitor)
                    try:
                        account = get_account()
                        risk.update_equity(float(account.equity), float(account.cash))
                    except Exception as e:
                        logger.error(f"Failed to update account: {e}")

                    last_scan = current

                # -------------------------------------------------------
                # EOD summary at EOD_SUMMARY_TIME
                # -------------------------------------------------------
                if current_time >= config.EOD_SUMMARY_TIME and not eod_summary_printed:
                    summary = risk.get_day_summary()
                    print_day_summary(summary, consistency_score=latest_consistency_score)
                    logger.info(f"Day summary: {summary}")

                    if notifications and config.TELEGRAM_ENABLED:
                        try:
                            notifications.notify_daily_summary(summary, risk.current_equity)
                        except Exception as e:
                            logger.warning(f"Daily summary notification failed: {e}")

                    # Save daily snapshot
                    try:
                        wr = summary.get("win_rate", 0) if summary.get("trades", 0) > 0 else 0
                        database.save_daily_snapshot(
                            date=current.strftime("%Y-%m-%d"),
                            portfolio_value=risk.current_equity,
                            cash=risk.current_cash,
                            day_pnl=risk.day_pnl * risk.starting_equity,
                            day_pnl_pct=risk.day_pnl,
                            total_trades=summary.get("trades", 0),
                            win_rate=wr,
                            sharpe_rolling=current_analytics.get("sharpe_7d", 0) if current_analytics else 0,
                        )
                    except Exception as e:
                        logger.error(f"Failed to save daily snapshot: {e}")

                    # Compute and save consistency score
                    try:
                        daily_pnls = database.get_daily_pnl_series(days=30)
                        sharpe = current_analytics.get("sharpe_7d", 0) if current_analytics else 0
                        max_dd = current_analytics.get("max_drawdown", 0) if current_analytics else 0
                        latest_consistency_score = compute_consistency_score(daily_pnls, sharpe, max_dd)
                        pct_positive = sum(1 for p in daily_pnls if p > 0) / max(len(daily_pnls), 1)
                        database.save_consistency_log(
                            date=current.strftime("%Y-%m-%d"),
                            consistency_score=latest_consistency_score,
                            pct_positive=pct_positive,
                            sharpe=sharpe,
                            max_drawdown=max_dd,
                            vol_scalar_avg=vol_engine.last_scalar,
                            beta_avg=beta_neutral.portfolio_beta,
                        )
                        logger.info(f"Consistency score: {latest_consistency_score:.1f}")
                    except Exception as e:
                        logger.error(f"Consistency score computation failed: {e}")

                    # V9: Alpha decay check at EOD
                    if alpha_decay_monitor:
                        try:
                            report = alpha_decay_monitor.get_strategy_health_report()
                            for strat, health in report.items():
                                status = health.get("status", "unknown")
                                if status == "critical":
                                    logger.warning(f"Alpha decay CRITICAL: {strat} — consider demoting")
                                elif status == "warning":
                                    logger.info(f"Alpha decay warning: {strat} — Sharpe declining")
                        except Exception as e:
                            logger.error(f"Alpha decay check failed: {e}")

                    # V9: Adaptive allocation rebalance at EOD
                    if adaptive_allocator:
                        try:
                            # HIGH-015: Use eod_rebalance which fetches trade history
                            # and converts regime detector probs automatically
                            new_weights = adaptive_allocator.eod_rebalance(
                                regime=regime,
                                regime_detector=regime_detector,
                            )
                            if new_weights:
                                vol_engine.set_adaptive_weights(new_weights)
                                logger.info(f"Adaptive weights updated: {new_weights}")
                        except Exception as e:
                            logger.error(f"Adaptive allocation failed: {e}")

                    # V11.4: Black-Litterman portfolio optimization at EOD
                    if bl_optimizer:
                        try:
                            active_syms = list(risk.open_trades.keys())
                            if len(active_syms) >= 2:
                                bl_optimizer.set_universe(active_syms)
                                # Use correlation limiter's matrix if available
                                if corr_limiter and hasattr(corr_limiter, '_last_good_matrix'):
                                    cov = getattr(corr_limiter, '_last_good_matrix', None)
                                    if cov is not None and cov.shape[0] == len(active_syms):
                                        bl_optimizer.set_market_data(covariance=cov)
                                        result = bl_optimizer.optimize()
                                        if result and result.weights is not None:
                                            bl_weights = {s: float(w) for s, w in zip(result.symbols, result.weights)}
                                            logger.info(f"V11.4: BL weights: {bl_weights}")
                        except Exception as e:
                            logger.debug(f"V11.4: Black-Litterman EOD optimization skipped: {e}")

                    # V12 14.3: Tax-loss harvesting scan on Fridays at EOD
                    if current.weekday() == 4:  # Friday
                        try:
                            from ops.tax_harvesting import TaxLossHarvester
                            _tax_harvester = TaxLossHarvester()
                            _opportunities = _tax_harvester.scan_opportunities(
                                list(risk.open_trades.values())
                            )
                            if _opportunities:
                                logger.info(
                                    "V12 14.3: Found %d tax-loss harvesting opportunities "
                                    "(total estimated savings: $%.2f)",
                                    len(_opportunities),
                                    sum(o.estimated_loss for o in _opportunities),
                                )
                                for opp in _opportunities:
                                    logger.info(
                                        "  TLH: %s — sell %d shares, est loss $%.2f, "
                                        "substitutes: %s",
                                        opp.symbol, opp.shares_to_sell,
                                        opp.estimated_loss, opp.substitute_symbols,
                                    )
                        except Exception as e:
                            logger.debug("V12 14.3: Tax harvesting scan failed (non-fatal): %s", e)

                    eod_summary_printed = True

                # -------------------------------------------------------
                # Save state periodically
                # -------------------------------------------------------
                if (current - last_state_save).total_seconds() >= config.STATE_SAVE_INTERVAL_SEC:
                    try:
                        database.save_open_positions(risk.open_trades)
                    except Exception as e:
                        logger.error(f"Failed to save state: {e}")
                    last_state_save = current

                # -------------------------------------------------------
                # Update analytics every 5 minutes
                # -------------------------------------------------------
                if (current - last_analytics_update).total_seconds() >= 300:
                    try:
                        current_analytics = analytics_mod.compute_analytics()
                        if current_analytics and notifications and config.TELEGRAM_ENABLED:
                            dd = current_analytics.get("max_drawdown", 0)
                            if dd > 0.05:
                                try:
                                    notifications.notify_drawdown_warning(dd)
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.error(f"Failed to compute analytics: {e}")

                    # V10: Update VaR monitor
                    if var_monitor:
                        try:
                            pnl_series_var = database.get_daily_pnl_series(days=60)
                            if pnl_series_var and len(pnl_series_var) >= 10:
                                var_result = var_monitor.update(pnl_series_var, risk.current_equity)
                                # Log if VaR budget is running low
                                if var_monitor.risk_budget_remaining < 0.3:
                                    logger.warning(
                                        f"VaR budget low: {var_monitor.risk_budget_remaining:.0%} remaining "
                                        f"(VaR95=${var_result.var_95:.0f}, {var_result.var_95_pct:.2%})"
                                    )
                        except Exception as e:
                            logger.debug(f"VaR update failed: {e}")

                    # V11.4: Check for new ML model (hot-swap)
                    try:
                        from engine.signal_processor import check_ml_model_freshness
                        check_ml_model_freshness()
                    except Exception as e:
                        logger.debug(f"V11.4: ML model freshness check failed: {e}")

                    last_analytics_update = current

                # -------------------------------------------------------
                # Update vol scalar periodically (every analytics cycle)
                # -------------------------------------------------------
                try:
                    from risk import get_vix_level
                    vix = get_vix_level()
                    pnl_series = database.get_daily_pnl_series(days=20)
                    if np is not None and len(pnl_series) > 2:
                        rolling_std = float(np.std(pnl_series))
                    else:
                        rolling_std = 0.01
                    vol_engine.compute_vol_scalar(
                        vix=vix,
                        portfolio_atr_vol=0.01,  # Simplified; could compute from positions
                        rolling_pnl_std=rolling_std,
                    )
                except Exception as e:
                    logger.error(f"Vol scalar computation failed (using last value): {e}", exc_info=True)

                # -------------------------------------------------------
                # Update dashboard
                # -------------------------------------------------------
                # V9: Update web dashboard shared state
                if config.WEB_DASHBOARD_ENABLED:
                    try:
                        from web_dashboard import update_v9_state
                        v9_update = {}
                        if regime_detector and hasattr(regime_detector, 'hmm_regime'):
                            v9_update['hmm_regime'] = str(regime_detector.hmm_regime or "UNKNOWN")
                            v9_update['hmm_probabilities'] = getattr(regime_detector, 'hmm_probabilities', {})
                        if cross_asset_monitor:
                            v9_update['cross_asset_bias'] = cross_asset_monitor.get_equity_bias()
                            try:
                                v9_update['cross_asset_signals'] = cross_asset_monitor.compute_signals()
                            except Exception:
                                pass
                        if overnight_manager:
                            v9_update['overnight_count'] = len(overnight_manager.get_overnight_positions())
                        if adaptive_allocator and hasattr(vol_engine, '_adaptive_weights'):
                            v9_update['adaptive_weights'] = getattr(vol_engine, '_adaptive_weights', {})
                        update_v9_state(**v9_update)
                    except Exception:
                        pass  # Web dashboard update is non-critical

                # Build V9 dashboard kwargs
                _v9_dash = {}
                try:
                    if regime_detector and hasattr(regime_detector, 'hmm_regime'):
                        _v9_dash['hmm_regime_state'] = str(regime_detector.hmm_regime or "")
                        _v9_dash['hmm_probabilities'] = getattr(regime_detector, 'hmm_probabilities', None)
                    if cross_asset_monitor:
                        _v9_dash['cross_asset_bias'] = cross_asset_monitor.get_equity_bias()
                    if overnight_manager:
                        _v9_dash['overnight_count'] = len(overnight_manager.get_overnight_positions())
                    if alpha_decay_monitor:
                        try:
                            report = alpha_decay_monitor.get_strategy_health_report()
                            warnings = [f"{s}: {d.get('status','')}" for s, d in report.items()
                                       if d.get('status') in ('warning', 'critical')]
                            _v9_dash['alpha_warnings'] = warnings
                        except Exception:
                            pass
                except Exception:
                    pass

                live.update(
                    build_dashboard(
                        risk, regime, start_time, current, last_scan,
                        len(config.SYMBOLS), current_analytics,
                        pnl_lock_state=pnl_lock.state.value,
                        vol_scalar=vol_engine.last_scalar,
                        portfolio_beta=beta_neutral.portfolio_beta,
                        consistency_score=latest_consistency_score,
                        **_v9_dash,
                    )
                )

                # T4-006 + V12 BONUS: Dynamic scan interval
                scan_interval = compute_dynamic_scan_interval(config.SCAN_INTERVAL_SEC)
                # BONUS: Further optimize with regime + position count + time-of-day
                if get_adaptive_scan_interval and regime:
                    try:
                        vix = get_vix_level()
                        bonus_interval = get_adaptive_scan_interval(
                            vix_level=vix,
                            regime=regime,
                            num_open_positions=len(risk.open_trades),
                            hour=current.hour,
                        )
                        scan_interval = min(scan_interval, bonus_interval)
                    except Exception:
                        pass
                # V12 BONUS: Update intraday vol regime
                if _intraday_vol_regime:
                    try:
                        spy_snap = get_snapshot("SPY")
                        if spy_snap and hasattr(spy_snap, "latest_trade"):
                            _intraday_vol_regime.update(float(spy_snap.latest_trade.price))
                    except Exception:
                        pass
                for _ in range(scan_interval):
                    time_mod.sleep(1)

    except KeyboardInterrupt:
        # IMPL-006: Signal handlers call _graceful_shutdown, but in case
        # KeyboardInterrupt propagates from time_mod.sleep or other blocking
        # calls before the signal handler runs, catch it here as fallback.
        if not _shutdown_requested:
            _graceful_shutdown("KeyboardInterrupt")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        console.print(f"[bold red]Bot crashed: {e}[/bold red]")
        # Emergency state save on crash
        try:
            _graceful_shutdown("CRASH")
        except Exception:
            # Last resort: just save positions
            try:
                database.save_open_positions(risk.open_trades)
            except Exception:
                pass
        console.print("[yellow]State saved. Restart with: python main.py[/yellow]")
        sys.exit(1)



# ===========================================================================
# (Legacy async mode removed in V10 — synchronous main loop has all features)
# ===========================================================================


# ===========================================================================
# DIAGNOSTIC MODE
# ===========================================================================

def run_diagnostic():
    """Run one diagnostic scan cycle and print what's blocking trades."""
    print("\n=== VELOX V10 SIGNAL DIAGNOSTIC MODE ===\n")
    print("Initializing strategies and filters...\n")

    from earnings import has_earnings_soon, load_earnings_cache
    from correlation import is_too_correlated, load_correlation_cache
    from data import get_clock, get_account

    stat_mr = StatMeanReversion()
    kalman_pairs = KalmanPairsTrader()
    micro_mom = IntradayMicroMomentum()

    clock = get_clock()
    now = clock.timestamp

    account = get_account()
    print(f"Account equity: ${float(account.equity):,.2f}")
    print(f"Market status: {'OPEN' if clock.is_open else 'CLOSED'}")
    print(f"Time: {now}\n")

    # Load filters
    try:
        load_earnings_cache(config.SYMBOLS)
    except Exception as e:
        print(f"  Earnings cache: {e}")
    try:
        load_correlation_cache(config.SYMBOLS)
    except Exception as e:
        print(f"  Correlation cache: {e}")

    # Detect regime
    regime_detector = MarketRegime()
    regime = regime_detector.update(now)
    print(f"Market regime: {regime}\n")

    # Track rejections
    rejection_counts: dict[str, int] = {}
    signals_generated: list[str] = []

    def count(reason: str):
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    mr_signals = []
    pair_signals = []
    micro_signals = []

    if clock.is_open:
        # Prepare MR universe
        print("Preparing MR universe...")
        stat_mr.prepare_universe(config.STANDARD_SYMBOLS, now)
        print(f"  MR universe: {len(stat_mr.universe)} symbols")

        # Scan strategies
        try:
            mr_signals = stat_mr.scan(now, regime)
            for sig in mr_signals:
                signals_generated.append(f"{sig.symbol} STAT_MR {sig.side} @ ${sig.entry_price:.2f}")
        except Exception as e:
            print(f"  StatMR scan error: {e}")

        try:
            pair_signals = kalman_pairs.scan(now, regime)
            for sig in pair_signals:
                signals_generated.append(f"{sig.symbol} KALMAN_PAIRS {sig.side} @ ${sig.entry_price:.2f}")
        except Exception as e:
            print(f"  KalmanPairs scan error: {e}")

        try:
            micro_mom.detect_event(now)
            micro_signals = micro_mom.scan(now, day_pnl_pct=0.0, regime=regime)
            for sig in micro_signals:
                signals_generated.append(f"{sig.symbol} MICRO_MOM {sig.side} @ ${sig.entry_price:.2f}")
        except Exception as e:
            print(f"  MicroMom scan error: {e}")

    # Filter diagnostic
    all_signals = mr_signals + pair_signals + micro_signals
    blocked_signals = []
    passed_signals = []

    for sig in all_signals:
        try:
            if has_earnings_soon(sig.symbol):
                count("filter_earnings")
                blocked_signals.append(f"{sig.symbol} {sig.strategy}: BLOCKED by earnings")
                continue
        except Exception:
            pass

        if sig.strategy != "KALMAN_PAIRS":
            try:
                if is_too_correlated(sig.symbol, []):
                    count("filter_correlation")
                    blocked_signals.append(f"{sig.symbol} {sig.strategy}: BLOCKED by correlation")
                    continue
            except Exception:
                pass

        passed_signals.append(f"{sig.symbol} {sig.strategy} {sig.side} @ ${sig.entry_price:.2f}")

    # Print results
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SCAN RESULTS")
    print("=" * 60)
    print(f"\n  Total symbols scanned:    {len(config.SYMBOLS)}")
    print(f"  MR universe size:         {len(stat_mr.universe)}")
    print(f"  Active pairs:             {len(kalman_pairs.active_pairs)}")
    print(f"  Signals generated:        {len(all_signals)}")
    print(f"  Signals blocked:          {len(blocked_signals)}")
    print(f"  Signals passed:           {len(passed_signals)}")

    if rejection_counts:
        print(f"\n  REJECTION BREAKDOWN:")
        for reason, cnt in sorted(rejection_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason:<30} {cnt:>4}")

    if blocked_signals:
        print(f"\n  BLOCKED SIGNALS:")
        for b in blocked_signals:
            print(f"    {b}")

    if passed_signals:
        print(f"\n  PASSED SIGNALS (would trade):")
        for p in passed_signals:
            print(f"    {p}")
    else:
        print(f"\n  NO SIGNALS PASSED -- all blocked or no setups found")

    print()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse as _argparse

    _parser = _argparse.ArgumentParser(description="Velox V10 Trading Bot")
    _parser.add_argument("--diagnose", action="store_true", help="Run one diagnostic scan cycle (read-only)")
    _args = _parser.parse_args()

    if _args.diagnose:
        run_diagnostic()
    else:
        main()
