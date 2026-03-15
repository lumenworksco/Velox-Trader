"""Entry point V7 — Velox V7 Autonomous Trading System.

Five-strategy portfolio with volatility-targeted sizing, daily P&L locks,
beta neutralization, news sentiment, optional LLM scoring, and adaptive exits.

Strategies: StatMR (50%), VWAP (20%), KalmanPairs (20%), ORB (5%), MicroMom (5%).
"""

import argparse
import asyncio
import logging
import sys
import time as time_mod
from datetime import datetime, time, timedelta
from pathlib import Path

from rich.console import Console
from rich.live import Live

import config
import database
import analytics as analytics_mod
from data import (
    verify_connectivity, verify_data_feed, get_account, get_clock,
    get_positions, get_snapshot, get_snapshots,
)
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
from risk.daily_pnl_lock import LockState
from execution import (
    submit_bracket_order,
    close_position,
    close_partial_position,
    cancel_all_open_orders,
    can_short,
)
from dashboard import (
    build_dashboard,
    print_day_summary,
    print_startup_info,
    console,
)
from earnings import load_earnings_cache, has_earnings_soon, get_excluded_count
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


def is_market_hours(t: time) -> bool:
    return config.MARKET_OPEN <= t <= config.MARKET_CLOSE


def is_trading_hours(t: time) -> bool:
    return config.TRADING_START <= t <= config.ORB_EXIT_TIME


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
# Position / broker sync
# ---------------------------------------------------------------------------

def sync_positions_with_broker(risk: RiskManager, now: datetime, ws_monitor=None):
    """Sync open trades with actual broker positions.

    FIXED in V7: When a position disappears from broker, fetch the last
    known market price for accurate P&L instead of using entry_price (which
    produced 0 P&L for all broker_sync exits).

    Unknown broker positions (ones we never opened) are logged as warnings
    and optionally auto-closed if CLOSE_UNKNOWN_POSITIONS=True.
    """
    try:
        broker_positions = {p.symbol: p for p in get_positions()}
    except Exception as e:
        logger.error(f"Failed to fetch positions: {e}")
        return

    # Close our DB records for positions the broker no longer has
    for symbol in list(risk.open_trades.keys()):
        if symbol not in broker_positions:
            trade = risk.open_trades[symbol]

            # Fetch last known price for accurate P&L
            try:
                snap = get_snapshot(symbol)
                exit_price = float(snap.latest_trade.price) if snap and snap.latest_trade else trade.entry_price
            except Exception:
                exit_price = trade.entry_price

            risk.close_trade(symbol, exit_price, now, exit_reason="broker_sync")
            logger.info(f"Position {symbol} no longer at broker — closed at ${exit_price:.2f} (entry ${trade.entry_price:.2f})")

            if ws_monitor:
                ws_monitor.unsubscribe(symbol)
            if notifications and config.TELEGRAM_ENABLED:
                try:
                    notifications.notify_trade_closed(trade)
                except Exception:
                    pass

    # Warn about unknown broker positions (don't create fake records)
    our_symbols = set(risk.open_trades.keys())
    for symbol in broker_positions:
        if symbol not in our_symbols and symbol != "SPY":  # SPY may be beta hedge
            bp = broker_positions[symbol]
            logger.warning(f"Unknown broker position: {symbol} qty={bp.qty} — not in our records")
            if getattr(config, "CLOSE_UNKNOWN_POSITIONS", False):
                try:
                    close_position(symbol, reason="unknown_position_cleanup")
                    logger.info(f"Auto-closed unknown position: {symbol}")
                except Exception as e:
                    logger.error(f"Failed to auto-close unknown position {symbol}: {e}")

    try:
        account = get_account()
        risk.update_equity(float(account.equity), float(account.cash))
    except Exception as e:
        logger.error(f"Failed to update account: {e}")


def check_shadow_exits(now: datetime):
    """Check shadow trades for TP/SL/time_stop hits and close them."""
    try:
        open_shadows = database.get_open_shadow_trades()
    except Exception as e:
        logger.warning(f"Failed to fetch shadow trades: {e}")
        return

    for shadow in open_shadows:
        try:
            snap = get_snapshot(shadow["symbol"])
            if not snap or not snap.latest_trade:
                continue
            price = float(snap.latest_trade.price)
            side = shadow["side"]
            tp = shadow["take_profit"]
            sl = shadow["stop_loss"]
            exit_reason = None

            if side == "buy":
                if price >= tp:
                    exit_reason = "take_profit"
                elif price <= sl:
                    exit_reason = "stop_loss"
            else:
                if price <= tp:
                    exit_reason = "take_profit"
                elif price >= sl:
                    exit_reason = "stop_loss"

            if exit_reason:
                database.close_shadow_trade(
                    shadow["id"], price, now.isoformat(), exit_reason
                )
                logger.info(
                    f"[SHADOW] Closed {shadow['symbol']} ({shadow['strategy']}) "
                    f"@ {price:.2f} reason={exit_reason}"
                )
        except Exception as e:
            logger.warning(f"Shadow exit check failed for {shadow.get('symbol', '?')}: {e}")


# ---------------------------------------------------------------------------
# Signal processing (V6 — simplified)
# ---------------------------------------------------------------------------

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
):
    """Process signals: check filters, risk, size, and submit orders.

    V7 filters: position conflict, earnings, correlation, short pre-check,
    news sentiment (soft), LLM scoring (optional).
    """

    # PnL lock check
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
                               news_sentiment, llm_scorer)

    # Process pairs atomically (both legs or neither)
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
            for sig in pair_signals:
                _process_single_signal(sig, risk, regime, now, vol_engine, pnl_lock, ws_monitor,
                                       news_sentiment, llm_scorer)
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
):
    """Process a single signal through V7 filters and submit if valid."""
    skip_reason = ""

    # 1. Position conflict
    if signal.symbol in risk.open_trades:
        skip_reason = "already_in_position"
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # 2. Earnings filter
    if has_earnings_soon(signal.symbol):
        skip_reason = "earnings_soon"
        logger.info(f"Signal skipped for {signal.symbol}: earnings soon")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # 3. Correlation filter (skip for KALMAN_PAIRS — they're inherently correlated)
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

    # 5. Check risk limits
    allowed, reason = risk.can_open_trade(strategy=signal.strategy)
    if not allowed:
        skip_reason = reason
        logger.info(f"Trade blocked for {signal.symbol}: {reason}")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # 5b. V7: News sentiment size adjustment (soft filter)
    news_mult = 1.0
    if news_sentiment:
        try:
            news_mult, news_reason = news_sentiment.get_sentiment_size_mult(signal.symbol)
            if news_mult == 0.0:
                database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, news_reason)
                return
        except Exception:
            news_mult = 1.0

    # 5c. V7: LLM signal scoring (optional, fail-open)
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
        except Exception:
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

    # Apply news + LLM multipliers
    qty = int(qty * news_mult * llm_mult)

    if qty <= 0:
        skip_reason = "position_size_zero"
        logger.info(f"Position size 0 for {signal.symbol}, skipping")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # 7. Submit bracket order
    order_id = submit_bracket_order(signal, qty)
    if order_id is None:
        skip_reason = "order_failed"
        logger.error(f"Failed to submit order for {signal.symbol}, skipping (no naked entry)")
        database.log_signal(now, signal.symbol, signal.strategy, signal.side, False, skip_reason)
        return

    # 8. Register the trade with V6 time stops / max hold
    time_stop = None
    max_hold_date = None
    hold_type = getattr(signal, "hold_type", "day")

    if signal.strategy == "STAT_MR":
        # No fixed time stop — z-score exits handle it
        pass
    elif signal.strategy == "KALMAN_PAIRS":
        max_hold_date = now + timedelta(days=config.PAIRS_MAX_HOLD_DAYS)
    elif signal.strategy == "MICRO_MOM":
        time_stop = now + timedelta(minutes=config.MICRO_MAX_HOLD_MINUTES)
    elif signal.strategy == "BETA_HEDGE":
        hold_type = "day"
        # No time stop — held until EOD or beta re-balances

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

    # Subscribe to WebSocket monitoring
    if ws_monitor:
        ws_monitor.subscribe(signal.symbol)

    # Notification
    if notifications and config.TELEGRAM_ENABLED:
        try:
            notifications.notify_trade_opened(trade)
        except Exception as e:
            logger.warning(f"Notification failed: {e}")

    # Log signal as acted on
    database.log_signal(now, signal.symbol, signal.strategy, signal.side, True, "")


# ---------------------------------------------------------------------------
# V6 exit processing helper
# ---------------------------------------------------------------------------

def _handle_strategy_exits(exit_actions: list[dict], risk: RiskManager, now: datetime, ws_monitor=None):
    """Process exit actions returned by strategy check_exits() methods.

    Each action dict: {symbol, action, reason, ...}
    action = "full" -> close_position, "partial" -> close_partial_position
    """
    for action in exit_actions:
        symbol = action["symbol"]
        if symbol not in risk.open_trades:
            continue
        trade = risk.open_trades[symbol]
        reason = action.get("reason", "strategy_exit")

        try:
            snap = get_snapshot(symbol)
            exit_price = float(snap.latest_trade.price) if snap and snap.latest_trade else trade.entry_price
        except Exception:
            exit_price = trade.entry_price

        if action.get("action") == "partial":
            partial_qty = action.get("qty", max(1, trade.qty // 2))
            try:
                close_partial_position(symbol, partial_qty)
                logger.info(f"Partial exit {symbol}: {partial_qty} shares, reason={reason}")
            except Exception as e:
                logger.error(f"Partial close failed for {symbol}: {e}")
        else:
            # Full close
            try:
                close_position(symbol, reason=reason)
            except Exception as e:
                logger.error(f"Close failed for {symbol}: {e}")
                continue
            risk.close_trade(symbol, exit_price, now, exit_reason=reason)
            if ws_monitor:
                ws_monitor.unsubscribe(symbol)
            if notifications and config.TELEGRAM_ENABLED:
                try:
                    notifications.notify_trade_closed(trade)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# WebSocket close callback
# ---------------------------------------------------------------------------

def _handle_ws_close(symbol: str, reason: str, risk: RiskManager, ws_monitor):
    """Callback for WebSocket-triggered position closes."""
    if symbol in risk.open_trades:
        trade = risk.open_trades[symbol]
        try:
            close_position(symbol)
        except Exception as e:
            logger.error(f"WS close failed for {symbol}: {e}")
            return
        risk.close_trade(symbol, trade.entry_price, now_et(), exit_reason=reason)
        ws_monitor.unsubscribe(symbol)

        if notifications and config.TELEGRAM_ENABLED:
            try:
                notifications.notify_trade_closed(trade)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Fetch current prices for open positions
# ---------------------------------------------------------------------------

def _get_current_prices(open_trades: dict) -> dict[str, float]:
    """Fetch current prices for open trades (for beta calculation etc.)."""
    symbols = list(open_trades.keys())
    if not symbols:
        return {}
    prices: dict[str, float] = {}
    try:
        snapshots = get_snapshots(symbols)
        for sym, snap in snapshots.items():
            if snap and snap.latest_trade:
                prices[sym] = float(snap.latest_trade.price)
    except Exception as e:
        logger.warning(f"Price fetch failed: {e}")
    # Fallback to entry price for any missing
    for sym, trade in open_trades.items():
        if sym not in prices:
            prices[sym] = trade.entry_price
    return prices


# ===========================================================================
# MAIN (synchronous mode)
# ===========================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Velox V6 Trading Bot")
    parser.add_argument("--backtest", action="store_true", help="Run backtesting engine")
    parser.add_argument("--walkforward", action="store_true", help="Run walk-forward test")
    parser.add_argument("--live", action="store_true", help="Alias for ALPACA_LIVE=true")
    args = parser.parse_args()

    if args.live:
        import os
        os.environ["ALPACA_LIVE"] = "true"

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

    console.print("[bold cyan]Starting Velox V6 Trading Bot...[/bold cyan]\n")

    # Initialize database
    database.init_db()
    database.migrate_from_json()

    # Startup checks
    info = startup_checks()

    # --- V7 strategy initialization ---
    stat_mr = StatMeanReversion()
    kalman_pairs = KalmanPairsTrader()
    micro_mom = IntradayMicroMomentum()
    vwap_strategy = VWAPStrategy()
    orb_strategy = ORBStrategyV2() if ORBStrategyV2 and config.ORB_ENABLED else None

    # V7 risk engine
    vol_engine = VolatilityTargetingRiskEngine()
    pnl_lock = DailyPnLLock()
    beta_neutral = BetaNeutralizer()

    # V7 optional modules
    news_sentiment = None
    if config.NEWS_SENTIMENT_ENABLED and AlpacaNewsSentiment:
        try:
            news_sentiment = AlpacaNewsSentiment()
            logger.info("News sentiment filter enabled")
        except Exception as e:
            logger.warning(f"News sentiment init failed: {e}")

    llm_scorer = None
    if config.LLM_SCORING_ENABLED and LLMSignalScorer:
        try:
            llm_scorer = LLMSignalScorer()
            logger.info("LLM signal scoring enabled")
        except Exception as e:
            logger.warning(f"LLM scorer init failed: {e}")

    adaptive_exits = AdaptiveExitManager() if AdaptiveExitManager and config.ADAPTIVE_EXITS_ENABLED else None
    walk_forward = WalkForwardValidator() if WalkForwardValidator and config.WALK_FORWARD_ENABLED else None

    # Regime detector (kept)
    regime_detector = MarketRegime()

    # Risk manager (kept)
    risk = RiskManager()
    risk.reset_daily(info["equity"], info["cash"])
    risk.load_from_db()

    # WebSocket position monitor
    ws_monitor = None
    if config.WEBSOCKET_MONITORING and PositionMonitor:
        ws_monitor = PositionMonitor(risk)
        ws_monitor.set_close_callback(
            lambda symbol, reason: _handle_ws_close(symbol, reason, risk, ws_monitor)
        )
        for symbol in risk.open_trades:
            ws_monitor.subscribe(symbol)
        ws_monitor.start()
        console.print("[green]WebSocket position monitor started.[/green]")

    # Web dashboard
    if config.WEB_DASHBOARD_ENABLED:
        try:
            from web_dashboard import start_web_dashboard
            start_web_dashboard()
            console.print(f"[green]Web dashboard: http://localhost:{config.WEB_DASHBOARD_PORT}[/green]")
        except Exception as e:
            console.print(f"[yellow]Web dashboard failed to start: {e}[/yellow]")

    # Notifications setup
    if notifications and config.TELEGRAM_ENABLED:
        console.print("[green]Notifications enabled.[/green]")

    # Load filters
    console.print("Loading earnings calendar...")
    try:
        load_earnings_cache(config.SYMBOLS)
    except Exception as e:
        console.print(f"[yellow]Earnings filter load failed: {e} (continuing without)[/yellow]")

    console.print("Loading correlation data...")
    try:
        load_correlation_cache(config.SYMBOLS)
    except Exception as e:
        console.print(f"[yellow]Correlation filter load failed: {e} (continuing without)[/yellow]")

    # --- Loop state ---
    start_time = now_et()
    last_scan = None
    last_state_save = now_et()
    last_analytics_update = now_et()
    last_day = now_et().date()
    eod_summary_printed = False
    current_analytics = None
    universe_prepared_today = False
    last_sunday_task = None
    latest_consistency_score = 0.0

    # Feature flags (V6)
    features = ["MR60%", "PAIRS25%", "MICRO15%"]
    if config.ALLOW_SHORT:
        features.append("Short")
    if config.TELEGRAM_ENABLED:
        features.append("Notify")
    if config.WEB_DASHBOARD_ENABLED:
        features.append("Web")
    if config.WEBSOCKET_MONITORING:
        features.append("WS")

    features_str = ", ".join(features)
    console.print(f"\n[bold green]Velox V7 is running. Press Ctrl+C to stop.[/bold green]")
    console.print(f"[dim]Strategies: STAT_MR + KALMAN_PAIRS + MICRO_MOM[/dim]")
    console.print(f"[dim]Features: {features_str}[/dim]\n")

    # -----------------------------------------------------------------------
    # V7 FIX: Initialize strategies at startup (don't wait for scheduled times)
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
                except Exception:
                    pass
        except Exception as e:
            console.print(f"[yellow]StatMR universe prep failed: {e}[/yellow]")

    # Initialize Kalman pairs if empty or stale
    try:
        active_pairs_count = len(database.get_active_kalman_pairs())
    except Exception:
        active_pairs_count = 0

    if active_pairs_count == 0:
        console.print("[cyan]Initializing Kalman pairs at startup (none active)...[/cyan]")
        try:
            kalman_pairs.select_pairs_weekly(startup_now)
            console.print(f"[green]Kalman pairs ready: {len(kalman_pairs.active_pairs)} pairs[/green]")
        except Exception as e:
            console.print(f"[yellow]Kalman pairs init failed: {e}[/yellow]")
    else:
        # Load existing pairs from DB into memory
        try:
            db_pairs = database.get_active_kalman_pairs()
            kalman_pairs.active_pairs = [
                (p['symbol1'], p['symbol2']) for p in db_pairs
            ]
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
                    logger.info("New trading day -- resetting state")
                    stat_mr.reset_daily()
                    micro_mom.reset_daily()
                    vwap_strategy.reset_daily()
                    pnl_lock.reset_daily()
                    beta_neutral.reset_daily()
                    # kalman_pairs.reset_daily() -- does nothing (pairs persist weekly)
                    if orb_strategy:
                        orb_strategy.reset_daily()
                        orb_strategy._ranges_recorded_today = False
                    if news_sentiment:
                        news_sentiment.clear_daily_cache()
                    if llm_scorer:
                        llm_scorer.reset_daily()
                    universe_prepared_today = False

                    try:
                        account = get_account()
                        risk.reset_daily(float(account.equity), float(account.cash))
                    except Exception as e:
                        logger.error(f"Failed to reset daily: {e}")
                    last_day = current.date()
                    eod_summary_printed = False

                    # Refresh daily caches
                    try:
                        load_earnings_cache(config.SYMBOLS)
                        load_correlation_cache(config.SYMBOLS)
                    except Exception as e:
                        logger.error(f"Failed to refresh daily caches: {e}")

                # -------------------------------------------------------
                # Sunday tasks — weekly pair selection
                # -------------------------------------------------------
                if (current.weekday() == 6 and current_time >= time(0, 0)
                        and last_sunday_task != current.date()):
                    last_sunday_task = current.date()
                    try:
                        logger.info("Sunday: selecting cointegrated pairs...")
                        kalman_pairs.select_pairs_weekly(current)
                    except Exception as e:
                        logger.error(f"Weekly pair selection failed: {e}")

                    # Walk-forward validation (weekly)
                    if walk_forward:
                        try:
                            logger.info("Sunday: running walk-forward validation...")
                            wf_results = walk_forward.run_weekly_validation()
                            for strat, result in wf_results.items():
                                rec = result.get('recommendation', 'unknown')
                                sharpe = result.get('sharpe', 0.0)
                                logger.info(f"WF {strat}: Sharpe={sharpe:.2f} -> {rec}")
                                if rec == 'demote':
                                    logger.warning(f"Walk-forward: {strat} demoted (OOS Sharpe {sharpe:.2f} < {config.WALK_FORWARD_MIN_SHARPE})")
                        except Exception as e:
                            logger.error(f"Walk-forward validation failed: {e}")

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
                        try:
                            stat_mr.prepare_universe(config.STANDARD_SYMBOLS, current)
                            universe_prepared_today = True
                            logger.info(f"MR universe prepared: {len(stat_mr.universe)} symbols")
                            # Save OU params to database for diagnostics
                            for sym, ou in stat_mr.ou_params.items():
                                try:
                                    database.save_ou_parameters(sym, ou)
                                except Exception:
                                    pass
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
                            from alpaca.data.timeframe import TimeFrame
                            market_open = datetime(current.year, current.month, current.day, 9, 30, tzinfo=config.ET)
                            orb_10am = datetime(current.year, current.month, current.day, 10, 0, tzinfo=config.ET)
                            for symbol in config.STANDARD_SYMBOLS[:config.ORB_SCAN_SYMBOLS]:
                                try:
                                    bars_930_1000 = get_intraday_bars(symbol, TimeFrame.Minute, start=market_open, end=orb_10am)
                                    if bars_930_1000 is not None and not bars_930_1000.empty:
                                        orb_strategy.record_opening_range(symbol, bars_930_1000)
                                except Exception:
                                    pass
                            orb_strategy._ranges_recorded_today = True
                            logger.info(f"ORB opening ranges recorded: {len(orb_strategy.opening_ranges)} symbols")
                        except Exception as e:
                            logger.error(f"ORB opening range recording failed: {e}")

                    # Trading hours scan
                    if is_trading_hours(current_time):
                        # 1. Update PnL lock state
                        pnl_lock.update(risk.day_pnl)

                        # 2. Detect micro momentum events
                        try:
                            micro_mom.detect_event(current)
                        except Exception as e:
                            logger.error(f"Micro event detection failed: {e}")

                        # 3. Scan all five strategies
                        signals: list[Signal] = []

                        try:
                            mr_signals = stat_mr.scan(current, regime)
                            signals.extend(mr_signals)
                        except Exception as e:
                            logger.error(f"StatMR scan failed: {e}")

                        try:
                            vwap_signals = vwap_strategy.scan(
                                config.STANDARD_SYMBOLS, current, regime
                            )
                            signals.extend(vwap_signals)
                        except Exception as e:
                            logger.error(f"VWAP scan failed: {e}")

                        try:
                            pair_signals = kalman_pairs.scan(current, regime)
                            signals.extend(pair_signals)
                        except Exception as e:
                            logger.error(f"KalmanPairs scan failed: {e}")

                        if orb_strategy:
                            try:
                                orb_signals = orb_strategy.scan(current, regime)
                                signals.extend(orb_signals)
                            except Exception as e:
                                logger.error(f"ORB scan failed: {e}")

                        try:
                            micro_signals = micro_mom.scan(
                                current, day_pnl_pct=risk.day_pnl, regime=regime
                            )
                            signals.extend(micro_signals)
                        except Exception as e:
                            logger.error(f"MicroMom scan failed: {e}")

                        # 4. Process signals
                        if signals:
                            process_signals(
                                signals, risk, regime, current,
                                vol_engine, pnl_lock, ws_monitor,
                                news_sentiment, llm_scorer,
                            )

                        # 5. Check strategy exits
                        try:
                            mr_exits = stat_mr.check_exits(risk.open_trades, current)
                            if mr_exits:
                                _handle_strategy_exits(mr_exits, risk, current, ws_monitor)
                        except Exception as e:
                            logger.error(f"StatMR exit check failed: {e}")

                        try:
                            pair_exits = kalman_pairs.check_exits(risk.open_trades, current)
                            if pair_exits:
                                _handle_strategy_exits(pair_exits, risk, current, ws_monitor)
                        except Exception as e:
                            logger.error(f"KalmanPairs exit check failed: {e}")

                        try:
                            micro_exits = micro_mom.check_exits(risk.open_trades, current)
                            if micro_exits:
                                _handle_strategy_exits(micro_exits, risk, current, ws_monitor)
                        except Exception as e:
                            logger.error(f"MicroMom exit check failed: {e}")

                        if orb_strategy:
                            try:
                                orb_exits = orb_strategy.check_exits(risk.open_trades, current)
                                if orb_exits:
                                    _handle_strategy_exits(orb_exits, risk, current, ws_monitor)
                            except Exception as e:
                                logger.error(f"ORB exit check failed: {e}")

                        # 6. Beta neutralization (every BETA_CHECK_INTERVAL_MIN)
                        if beta_neutral.should_check_now(current):
                            if not beta_neutral.should_skip(current):
                                try:
                                    prices = _get_current_prices(risk.open_trades)
                                    beta = beta_neutral.compute_portfolio_beta(risk.open_trades, prices)
                                    if beta_neutral.needs_hedge():
                                        spy_price = prices.get("SPY")
                                        if not spy_price:
                                            try:
                                                snap = get_snapshot("SPY")
                                                spy_price = float(snap.latest_trade.price) if snap and snap.latest_trade else None
                                            except Exception:
                                                spy_price = None
                                        if spy_price:
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

                        # Check shadow exits
                        check_shadow_exits(current)

                    # EOD close for day-hold positions at ORB_EXIT_TIME
                    if current_time >= config.ORB_EXIT_TIME:
                        for symbol in list(risk.open_trades.keys()):
                            trade = risk.open_trades[symbol]
                            if trade.hold_type == "day" and trade.strategy in ("MICRO_MOM", "BETA_HEDGE", "ORB", "VWAP"):
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
                                except Exception as e:
                                    logger.error(f"EOD close failed for {symbol}: {e}")

                    # Sync with broker
                    if not (ws_monitor and ws_monitor.is_connected):
                        sync_positions_with_broker(risk, current, ws_monitor)
                    else:
                        try:
                            account = get_account()
                            risk.update_equity(float(account.equity), float(account.cash))
                        except Exception as e:
                            logger.error(f"Failed to update account: {e}")

                    # Check circuit breaker
                    if risk.check_circuit_breaker():
                        if notifications and config.TELEGRAM_ENABLED:
                            try:
                                notifications.notify_circuit_breaker(risk.day_pnl)
                            except Exception:
                                pass

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
                except Exception:
                    pass  # Vol scalar stays at last value

                # -------------------------------------------------------
                # Update dashboard
                # -------------------------------------------------------
                live.update(
                    build_dashboard(
                        risk, regime, start_time, current, last_scan,
                        len(config.SYMBOLS), current_analytics,
                        pnl_lock_state=pnl_lock.state.value,
                        vol_scalar=vol_engine.last_scalar,
                        portfolio_beta=beta_neutral.portfolio_beta,
                        consistency_score=latest_consistency_score,
                    )
                )

                # Sleep until next scan
                time_mod.sleep(config.SCAN_INTERVAL_SEC)

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")
        if ws_monitor:
            ws_monitor.stop()
        try:
            database.save_open_positions(risk.open_trades)
        except Exception:
            pass
        console.print("[green]State saved. Bot stopped.[/green]")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        if ws_monitor:
            ws_monitor.stop()
        try:
            database.save_open_positions(risk.open_trades)
        except Exception:
            pass
        console.print(f"[bold red]Bot crashed: {e}[/bold red]")
        console.print("[yellow]State saved. Restart with: python main.py[/yellow]")
        sys.exit(1)


# ===========================================================================
# ASYNC MODE
# ===========================================================================

async def _async_scanner(
    risk, regime_detector, stat_mr, kalman_pairs, micro_mom,
    vol_engine, pnl_lock, beta_neutral, ws_monitor,
):
    """Async task: scan for V6 signals and process them every SCAN_INTERVAL_SEC."""
    universe_prepared_today = False

    while True:
        try:
            current = now_et()
            ct = current.time()
            regime = regime_detector.regime

            if is_market_hours(ct):
                # 9:00 AM universe prep
                if not universe_prepared_today and ct >= config.MR_UNIVERSE_PREP_TIME:
                    try:
                        stat_mr.prepare_universe(config.STANDARD_SYMBOLS, current)
                        universe_prepared_today = True
                    except Exception as e:
                        logger.error(f"[async] MR universe prep failed: {e}")

                    if current.weekday() == 0:
                        try:
                            kalman_pairs.select_pairs_weekly(current)
                        except Exception as e:
                            logger.error(f"[async] Monday pair selection failed: {e}")

                # Trading hours
                if is_trading_hours(ct):
                    pnl_lock.update(risk.day_pnl)

                    try:
                        micro_mom.detect_event(current)
                    except Exception as e:
                        logger.error(f"[async] Micro event detection failed: {e}")

                    signals: list[Signal] = []
                    try:
                        signals.extend(stat_mr.scan(current, regime))
                    except Exception as e:
                        logger.error(f"[async] StatMR scan failed: {e}")
                    try:
                        signals.extend(kalman_pairs.scan(current, regime))
                    except Exception as e:
                        logger.error(f"[async] KalmanPairs scan failed: {e}")
                    try:
                        signals.extend(micro_mom.scan(current, day_pnl_pct=risk.day_pnl, regime=regime))
                    except Exception as e:
                        logger.error(f"[async] MicroMom scan failed: {e}")

                    if signals:
                        process_signals(
                            signals, risk, regime, current,
                            vol_engine, pnl_lock, ws_monitor,
                        )

            # Reset universe tracking at day boundary
            if ct < config.MARKET_OPEN:
                universe_prepared_today = False

        except Exception as e:
            logger.error(f"[async] Scanner error: {e}", exc_info=True)

        await asyncio.sleep(config.SCAN_INTERVAL_SEC)


async def _async_exit_checker(
    risk, stat_mr, kalman_pairs, micro_mom, beta_neutral,
    vol_engine, pnl_lock, ws_monitor,
):
    """Async task: check V6 exit conditions every 30 seconds."""
    while True:
        try:
            current = now_et()
            ct = current.time()

            if is_trading_hours(ct):
                # Strategy exits
                try:
                    mr_exits = stat_mr.check_exits(risk.open_trades, current)
                    if mr_exits:
                        _handle_strategy_exits(mr_exits, risk, current, ws_monitor)
                except Exception as e:
                    logger.error(f"[async] StatMR exit check failed: {e}")

                try:
                    pair_exits = kalman_pairs.check_exits(risk.open_trades, current)
                    if pair_exits:
                        _handle_strategy_exits(pair_exits, risk, current, ws_monitor)
                except Exception as e:
                    logger.error(f"[async] KalmanPairs exit check failed: {e}")

                try:
                    micro_exits = micro_mom.check_exits(risk.open_trades, current)
                    if micro_exits:
                        _handle_strategy_exits(micro_exits, risk, current, ws_monitor)
                except Exception as e:
                    logger.error(f"[async] MicroMom exit check failed: {e}")

                # Beta neutralization
                if beta_neutral.should_check_now(current):
                    if not beta_neutral.should_skip(current):
                        try:
                            prices = _get_current_prices(risk.open_trades)
                            beta_neutral.compute_portfolio_beta(risk.open_trades, prices)
                            if beta_neutral.needs_hedge():
                                spy_price = prices.get("SPY")
                                if not spy_price:
                                    try:
                                        snap = get_snapshot("SPY")
                                        spy_price = float(snap.latest_trade.price) if snap and snap.latest_trade else None
                                    except Exception:
                                        spy_price = None
                                if spy_price:
                                    hedge_signal = beta_neutral.compute_hedge_signal(
                                        risk.current_equity, spy_price
                                    )
                                    if hedge_signal:
                                        regime = "UNKNOWN"  # Hedge is regime-agnostic
                                        process_signals(
                                            [hedge_signal], risk, regime, current,
                                            vol_engine, pnl_lock, ws_monitor,
                                        )
                        except Exception as e:
                            logger.error(f"[async] Beta neutralization failed: {e}")

                # EOD close for day-hold positions
                if ct >= config.ORB_EXIT_TIME:
                    for symbol in list(risk.open_trades.keys()):
                        trade = risk.open_trades[symbol]
                        if trade.hold_type == "day" and trade.strategy in ("MICRO_MOM", "BETA_HEDGE"):
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
                            except Exception as e:
                                logger.error(f"[async] EOD close failed for {symbol}: {e}")

                # Shadow exits
                check_shadow_exits(current)

        except Exception as e:
            logger.error(f"[async] Exit checker error: {e}", exc_info=True)

        await asyncio.sleep(30)


async def _async_broker_sync(risk, ws_monitor):
    """Async task: sync positions with broker every 60 seconds."""
    while True:
        try:
            current = now_et()
            if is_market_hours(current.time()):
                if not (ws_monitor and ws_monitor.is_connected):
                    sync_positions_with_broker(risk, current, ws_monitor)
                else:
                    try:
                        account = get_account()
                        risk.update_equity(float(account.equity), float(account.cash))
                    except Exception as e:
                        logger.error(f"[async] Account update failed: {e}")

                if risk.check_circuit_breaker():
                    if notifications and config.TELEGRAM_ENABLED:
                        try:
                            notifications.notify_circuit_breaker(risk.day_pnl)
                        except Exception:
                            pass
        except Exception as e:
            logger.error(f"[async] Broker sync error: {e}", exc_info=True)

        await asyncio.sleep(60)


async def _async_state_saver(risk):
    """Async task: save open positions to DB every STATE_SAVE_INTERVAL_SEC."""
    while True:
        await asyncio.sleep(config.STATE_SAVE_INTERVAL_SEC)
        try:
            database.save_open_positions(risk.open_trades)
        except Exception as e:
            logger.error(f"[async] State save failed: {e}")


async def _async_daily_tasks(
    risk, regime_detector, stat_mr, kalman_pairs, micro_mom,
    pnl_lock, beta_neutral, vol_engine,
):
    """Async task: handle daily resets, regime updates, analytics, consistency score."""
    last_day = now_et().date()
    last_sunday_task = None
    eod_summary_printed = False
    current_analytics = None
    latest_consistency_score = 0.0

    while True:
        try:
            current = now_et()
            ct = current.time()

            # Daily reset
            if current.date() != last_day:
                logger.info("[async] New trading day -- resetting state")
                stat_mr.reset_daily()
                micro_mom.reset_daily()
                pnl_lock.reset_daily()
                beta_neutral.reset_daily()

                try:
                    account = get_account()
                    risk.reset_daily(float(account.equity), float(account.cash))
                except Exception as e:
                    logger.error(f"[async] Daily reset failed: {e}")
                last_day = current.date()
                eod_summary_printed = False

                try:
                    load_earnings_cache(config.SYMBOLS)
                    load_correlation_cache(config.SYMBOLS)
                except Exception as e:
                    logger.error(f"[async] Cache refresh failed: {e}")

            # Sunday tasks
            if (current.weekday() == 6 and ct >= time(0, 0)
                    and last_sunday_task != current.date()):
                last_sunday_task = current.date()
                try:
                    kalman_pairs.select_pairs_weekly(current)
                except Exception as e:
                    logger.error(f"[async] Weekly pair selection failed: {e}")

            # Regime update
            regime_detector.update(current)

            # Vol scalar update
            try:
                from risk import get_vix_level
                vix = get_vix_level()
                pnl_series = database.get_daily_pnl_series(days=20)
                rolling_std = float(np.std(pnl_series)) if np is not None and len(pnl_series) > 2 else 0.01
                vol_engine.compute_vol_scalar(vix=vix, rolling_pnl_std=rolling_std)
            except Exception:
                pass

            # EOD summary
            if ct >= config.EOD_SUMMARY_TIME and not eod_summary_printed:
                summary = risk.get_day_summary()
                print_day_summary(summary, consistency_score=latest_consistency_score)
                logger.info(f"Day summary: {summary}")
                if notifications and config.TELEGRAM_ENABLED:
                    try:
                        notifications.notify_daily_summary(summary, risk.current_equity)
                    except Exception:
                        pass
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
                    logger.error(f"[async] Daily snapshot failed: {e}")

                # Consistency score
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
                except Exception as e:
                    logger.error(f"[async] Consistency score failed: {e}")

                eod_summary_printed = True

            # Analytics
            try:
                current_analytics = analytics_mod.compute_analytics()
                if current_analytics and notifications and config.TELEGRAM_ENABLED:
                    dd = current_analytics.get("max_drawdown", 0)
                    if dd > 0.05:
                        try:
                            notifications.notify_drawdown_warning(dd)
                        except Exception:
                            pass
            except Exception:
                pass

        except Exception as e:
            logger.error(f"[async] Daily tasks error: {e}", exc_info=True)

        await asyncio.sleep(60)


async def _async_dashboard_updater(
    risk, regime_detector, start_time, live,
    pnl_lock, vol_engine, beta_neutral,
):
    """Async task: refresh the Rich dashboard."""
    latest_consistency_score = 0.0
    while True:
        try:
            current = now_et()
            live.update(
                build_dashboard(
                    risk, regime_detector.regime, start_time, current, current,
                    len(config.SYMBOLS), None,
                    pnl_lock_state=pnl_lock.state.value,
                    vol_scalar=vol_engine.last_scalar,
                    portfolio_beta=beta_neutral.portfolio_beta,
                    consistency_score=latest_consistency_score,
                )
            )
        except Exception as e:
            logger.error(f"[async] Dashboard update failed: {e}")
        await asyncio.sleep(5)


async def async_main():
    """Async entry point -- runs all tasks concurrently under supervision."""
    import asyncio as _asyncio
    from supervisor import TaskSupervisor

    console.print("[bold cyan]Starting Velox V6 Trading Bot (ASYNC MODE)...[/bold cyan]\n")

    # Initialize database
    database.init_db()
    database.migrate_from_json()

    # Startup checks
    info = startup_checks()

    # V6 strategy initialization
    stat_mr = StatMeanReversion()
    kalman_pairs = KalmanPairsTrader()
    micro_mom = IntradayMicroMomentum()

    # V6 risk engine
    vol_engine = VolatilityTargetingRiskEngine()
    pnl_lock = DailyPnLLock()
    beta_neutral = BetaNeutralizer()

    regime_detector = MarketRegime()
    risk = RiskManager()
    risk.reset_daily(info["equity"], info["cash"])
    risk.load_from_db()

    # WS monitor
    ws_monitor = None
    if config.WEBSOCKET_MONITORING and PositionMonitor:
        ws_monitor = PositionMonitor(risk)
        ws_monitor.set_close_callback(
            lambda symbol, reason: _handle_ws_close(symbol, reason, risk, ws_monitor)
        )
        for symbol in risk.open_trades:
            ws_monitor.subscribe(symbol)
        ws_monitor.start()

    # Web dashboard
    if config.WEB_DASHBOARD_ENABLED:
        try:
            from web_dashboard import start_web_dashboard
            start_web_dashboard()
        except Exception:
            pass

    # Notifications
    if notifications and config.TELEGRAM_ENABLED:
        console.print("[green]Notifications enabled.[/green]")

    # Load filters
    try:
        load_earnings_cache(config.SYMBOLS)
    except Exception:
        pass
    try:
        load_correlation_cache(config.SYMBOLS)
    except Exception:
        pass

    start_time = now_et()
    logging.getLogger().removeHandler(_stream_handler)

    console.print(f"\n[bold green]Velox V6 (ASYNC) is running. Press Ctrl+C to stop.[/bold green]\n")

    supervisor = TaskSupervisor()

    if notifications and config.TELEGRAM_ENABLED:
        async def _crash_notify(name, exc, count):
            try:
                notifications.send_message(f"Task '{name}' crashed (#{count}): {exc}")
            except Exception:
                pass
        supervisor.set_notify(_crash_notify)

    try:
        with Live(
            build_dashboard(
                risk, regime_detector.regime, start_time, now_et(), None,
                len(config.SYMBOLS), None,
                pnl_lock_state=pnl_lock.state.value,
                vol_scalar=vol_engine.last_scalar,
                portfolio_beta=beta_neutral.portfolio_beta,
            ),
            console=console, refresh_per_second=0.2, transient=False,
        ) as live:
            await supervisor.launch(
                "scanner", _async_scanner,
                risk, regime_detector, stat_mr, kalman_pairs, micro_mom,
                vol_engine, pnl_lock, beta_neutral, ws_monitor,
            )
            await supervisor.launch(
                "exit_checker", _async_exit_checker,
                risk, stat_mr, kalman_pairs, micro_mom, beta_neutral,
                vol_engine, pnl_lock, ws_monitor,
            )
            await supervisor.launch(
                "broker_sync", _async_broker_sync,
                risk, ws_monitor,
            )
            await supervisor.launch(
                "state_saver", _async_state_saver,
                risk,
            )
            await supervisor.launch(
                "daily_tasks", _async_daily_tasks,
                risk, regime_detector, stat_mr, kalman_pairs, micro_mom,
                pnl_lock, beta_neutral, vol_engine,
            )
            await supervisor.launch(
                "dashboard", _async_dashboard_updater,
                risk, regime_detector, start_time, live,
                pnl_lock, vol_engine, beta_neutral,
            )

            try:
                await _asyncio.Event().wait()
            except _asyncio.CancelledError:
                pass

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down async tasks...[/yellow]")
    finally:
        await supervisor.stop_all()
        if ws_monitor:
            ws_monitor.stop()
        try:
            database.save_open_positions(risk.open_trades)
        except Exception:
            pass
        console.print("[green]State saved. Bot stopped.[/green]")


# ===========================================================================
# DIAGNOSTIC MODE
# ===========================================================================

def run_diagnostic():
    """Run one diagnostic scan cycle and print what's blocking trades."""
    print("\n=== VELOX V6 SIGNAL DIAGNOSTIC MODE ===\n")
    print("Initializing V6 strategies and filters...\n")

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
    from strategies.regime import detect_regime
    regime = detect_regime(now)
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

    _parser = _argparse.ArgumentParser(description="Velox V6 Trading Bot")
    _parser.add_argument("--diagnose", action="store_true", help="Run one diagnostic scan cycle (read-only)")
    _args = _parser.parse_args()

    if _args.diagnose:
        run_diagnostic()
    elif config.ASYNC_MODE:
        import asyncio
        asyncio.run(async_main())
    else:
        main()
