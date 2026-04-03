"""V10 Engine — Daily reset, weekly tasks, and EOD close logic.

Extracts the daily/weekly/EOD operations from main.py into reusable functions.
"""

import logging
import os
import shutil
from datetime import datetime, time, timedelta
from pathlib import Path

import config
import database
from data import get_account, get_snapshot
from execution import close_position
from earnings import load_earnings_cache
from correlation import load_correlation_cache

logger = logging.getLogger(__name__)

# V12 11.3: Database backup settings
_BOT_DIR = Path(__file__).resolve().parent.parent
_BACKUP_DIR = _BOT_DIR / "backups"
_BACKUP_RETENTION_DAYS = 30

# WIRE-007: Pre-market stress testing (fail-open)
_stress_framework = None
try:
    from risk.stress_testing import StressTestFramework as _STF
    _stress_framework = _STF()
except ImportError:
    _STF = None

# WIRE-012: Tax-loss harvesting (fail-open)
_tax_harvester = None
try:
    from ops.tax_harvesting import TaxLossHarvester as _TLH
    _tax_harvester = _TLH()
except ImportError:
    _TLH = None


def backup_database(db_name: str = "bot.db") -> bool:
    """V12 11.3: Copy bot.db to backups/bot_YYYYMMDD.db. Keep last 30 days.

    Returns True if backup succeeded, False otherwise.
    """
    src = _BOT_DIR / db_name
    if not src.exists():
        logger.warning(f"V12 11.3: Database file {src} not found — skipping backup")
        return False

    try:
        _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(config.ET).strftime("%Y%m%d")
        dst = _BACKUP_DIR / f"bot_{date_str}.db"

        shutil.copy2(str(src), str(dst))
        logger.info(f"V12 11.3: Database backed up to {dst}")

        # Prune backups older than retention period
        cutoff = datetime.now(config.ET) - timedelta(days=_BACKUP_RETENTION_DAYS)
        pruned = 0
        for backup_file in _BACKUP_DIR.glob("bot_*.db"):
            try:
                # Extract date from filename: bot_YYYYMMDD.db
                date_part = backup_file.stem.split("_", 1)[1]
                file_date = datetime.strptime(date_part, "%Y%m%d")
                if file_date < cutoff.replace(tzinfo=None):
                    backup_file.unlink()
                    pruned += 1
            except (ValueError, IndexError):
                continue  # Skip files that don't match the naming pattern

        if pruned:
            logger.info(f"V12 11.3: Pruned {pruned} old backup(s) (>{_BACKUP_RETENTION_DAYS} days)")
        return True
    except Exception as e:
        logger.error(f"V12 11.3: Database backup failed: {e}")
        return False


def daily_reset(
    risk,
    stat_mr, kalman_pairs, micro_mom, vwap_strategy,
    pnl_lock, beta_neutral,
    orb_strategy=None, pead_strategy=None,
    overnight_manager=None, news_sentiment=None, llm_scorer=None,
    tiered_cb=None,
):
    """Reset all strategies and risk engines for a new trading day.

    BUG-022: Each strategy's reset_daily() is wrapped in try/except so that
    a failure in one strategy does not prevent others from resetting.
    All failures are collected and reported in a summary log.
    """
    logger.info("New trading day -- resetting state")

    # BUG-022: Track all reset failures
    reset_failures: list[str] = []

    for name, obj in [
        ("stat_mr", stat_mr),
        ("micro_mom", micro_mom),
        ("vwap_strategy", vwap_strategy),
        ("pnl_lock", pnl_lock),
        ("beta_neutral", beta_neutral),
    ]:
        try:
            obj.reset_daily()
        except Exception as e:
            logger.error(f"Daily reset failed for {name}: {e}", exc_info=True)
            reset_failures.append(name)

    if tiered_cb:
        try:
            tiered_cb.reset_daily()
        except Exception as e:
            logger.error(f"Daily reset failed for tiered_cb: {e}", exc_info=True)
            reset_failures.append("tiered_cb")

    if orb_strategy:
        try:
            orb_strategy.reset_daily()
            orb_strategy._ranges_recorded_today = False
        except Exception as e:
            logger.error(f"Daily reset failed for orb_strategy: {e}", exc_info=True)
            reset_failures.append("orb_strategy")

    if news_sentiment:
        try:
            news_sentiment.clear_daily_cache()
        except Exception as e:
            logger.error(f"Daily reset failed for news_sentiment: {e}", exc_info=True)
            reset_failures.append("news_sentiment")

    if llm_scorer:
        try:
            llm_scorer.reset_daily()
        except Exception as e:
            logger.error(f"Daily reset failed for llm_scorer: {e}", exc_info=True)
            reset_failures.append("llm_scorer")

    if pead_strategy:
        try:
            pead_strategy.reset_daily()
        except Exception as e:
            logger.error(f"Daily reset failed for pead_strategy: {e}", exc_info=True)
            reset_failures.append("pead_strategy")

    if overnight_manager:
        try:
            overnight_manager.reset_daily()
        except Exception as e:
            logger.error(f"Daily reset failed for overnight_manager: {e}", exc_info=True)
            reset_failures.append("overnight_manager")

    # BUG-022: Report summary of all failures
    if reset_failures:
        logger.warning(f"Daily reset completed with {len(reset_failures)} failure(s): {reset_failures}")

    # Reset risk manager with fresh account data
    try:
        account = get_account()
        risk.reset_daily(float(account.equity), float(account.cash))
    except Exception as e:
        logger.error(f"Failed to reset daily account: {e}")

    # Refresh daily caches
    try:
        load_earnings_cache(config.SYMBOLS)
        load_correlation_cache(config.SYMBOLS)
    except Exception as e:
        logger.error(f"Failed to refresh daily caches: {e}")

    # WIRE-007: Run pre-market stress test (fail-open)
    try:
        if _stress_framework is not None:
            equity = float(getattr(risk, 'current_equity', 0) or 0)
            positions = getattr(risk, 'open_trades', {})
            if equity > 0:
                stress_result = _stress_framework.run_stress_tests(positions, equity)
                if _stress_framework.should_block_new_positions():
                    logger.warning("WIRE-007: Stress test recommends blocking new positions")
                else:
                    logger.info("WIRE-007: Pre-market stress test passed")
    except Exception as e:
        logger.debug("WIRE-007: Stress test failed (fail-open): %s", e)


def weekly_tasks(current: datetime, kalman_pairs, param_optimizer=None,
                 walk_forward=None, hmm_detector=None):
    """Run weekly tasks (Sunday): HMM retrain, pair selection, parameter optimization, walk-forward.

    IMPL-008: Added HMM retrain as the first weekly task. The HMM should
    be retrained before pair selection so that regime-aware decisions use
    the freshest model.
    """
    # IMPL-008: Retrain HMM regime detector on recent SPY data
    if hmm_detector is not None:
        try:
            logger.info("Sunday: retraining HMM regime detector...")
            success = hmm_detector.retrain_weekly(lookback_days=252)
            if success:
                logger.info(
                    f"HMM retrain complete: current regime = "
                    f"{hmm_detector.current_regime.value}"
                )
            else:
                logger.warning("HMM retrain returned False — using existing model")
        except Exception as e:
            logger.error(f"HMM weekly retrain failed: {e}")
    else:
        # Try to import and create detector if not provided
        try:
            from analytics.hmm_regime import HMMRegimeDetector
            detector = HMMRegimeDetector()
            if detector.retrain_weekly():
                logger.info("HMM retrain completed (ad-hoc detector instance)")
        except Exception as e:
            logger.debug(f"HMM retrain skipped (no detector available): {e}")

    try:
        logger.info("Sunday: selecting cointegrated pairs...")
        kalman_pairs.select_pairs_weekly(current)
    except Exception as e:
        logger.error(f"Weekly pair selection failed: {e}")

    if param_optimizer:
        try:
            results = param_optimizer.optimize_all()
            if results:
                logger.info(f"Parameter optimization: {len(results)} strategies updated")
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")

    if walk_forward:
        try:
            results = walk_forward.run_weekly_validation()
            if results:
                logger.info(f"Walk-forward validation: {len(results)} strategies tested")
        except Exception as e:
            logger.error(f"Walk-forward validation failed: {e}")


def eod_close(
    current: datetime,
    risk,
    ws_monitor=None,
    overnight_manager=None,
    regime: str = "UNKNOWN",
):
    """Close all day-hold positions at EOD. Respects overnight holds (BUG-034).

    T1-008: Overnight holds are persisted to DB so they survive bot restarts.
    On startup, load_overnight_holds() restores the registry.
    """
    # V10 BUG-034: Track overnight hold symbols to skip in EOD close
    # T1-008: First reload any persisted overnight holds (survives restart)
    overnight_hold_symbols: set[str] = database.load_overnight_holds()

    if overnight_manager:
        try:
            holds = overnight_manager.select_overnight_holds(risk.open_trades, regime)
            if holds:
                overnight_hold_symbols |= {h.symbol for h in holds}
                logger.info(f"Overnight holds selected: {list(overnight_hold_symbols)}")
                # T1-008: Persist to DB at registration time
                database.save_overnight_holds(overnight_hold_symbols)
        except Exception as e:
            logger.error(f"Overnight hold selection failed: {e}")

    day_strategies = ("MICRO_MOM", "BETA_HEDGE", "ORB", "VWAP", "STAT_MR", "KALMAN_PAIRS")
    closed_count = 0

    for symbol in list(risk.open_trades.keys()):
        trade = risk.open_trades[symbol]

        if symbol in overnight_hold_symbols:
            continue

        if trade.hold_type == "day" and trade.strategy in day_strategies:
            try:
                close_position(symbol, reason="eod_close")
                try:
                    snap = get_snapshot(symbol)
                    ep = float(snap.latest_trade.price) if snap and snap.latest_trade else trade.entry_price
                except Exception:
                    ep = trade.entry_price
                risk.close_trade(symbol, ep, current, exit_reason="eod_close")
                if ws_monitor:
                    ws_monitor.unsubscribe(symbol)
                closed_count += 1
            except Exception as e:
                logger.error(f"EOD close failed for {symbol}: {e}")

    if closed_count:
        logger.info(f"EOD: closed {closed_count} day-hold positions")

    # WIRE-012: Tax-loss harvesting scan after market close (fail-open)
    try:
        if _tax_harvester is not None:
            positions = getattr(risk, 'open_trades', {})
            if positions:
                harvest_candidates = _tax_harvester.scan_for_harvesting(
                    positions=positions,
                    threshold=getattr(config, 'TAX_HARVEST_THRESHOLD', -0.03),
                )
                if harvest_candidates:
                    logger.info("WIRE-012: Tax harvest candidates: %d positions", len(harvest_candidates))
    except Exception as e:
        logger.debug("WIRE-012: Tax harvesting scan failed (fail-open): %s", e)

    # T1-009: Clean up stale VPIN instances at EOD to prevent memory leak
    try:
        from engine.signal_processor import cleanup_stale_vpin_instances
        cleanup_stale_vpin_instances(active_symbols=None)  # Clear all at EOD
    except Exception as e:
        logger.debug("T1-009: VPIN cleanup failed (fail-open): %s", e)

    return closed_count
