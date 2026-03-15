"""Backtesting engine — simulate strategies on historical data."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

import config
import database

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class BacktestTrade:
    symbol: str
    strategy: str
    side: str
    entry_price: float
    entry_time: datetime
    qty: int
    take_profit: float = 0.0
    stop_loss: float = 0.0
    exit_price: float = 0.0
    exit_time: datetime | None = None
    pnl: float = 0.0
    commission: float = 0.0


@dataclass
class BacktestResult:
    strategy: str
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    avg_hold_minutes: float = 0.0


def download_data(symbols: list[str], months: int = 6) -> dict[str, pd.DataFrame]:
    """Download historical data via yfinance."""
    import yfinance as yf

    console.print(f"Downloading {months}M of data for {len(symbols)} symbols...")
    end = datetime.now()
    start = end - timedelta(days=months * 30)

    all_data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            # Get hourly data (max 730 days for 1h)
            df = ticker.history(start=start, end=end, interval="1h", auto_adjust=True)
            if df.empty or len(df) < 100:
                continue
            df.columns = [c.lower() for c in df.columns]
            all_data[symbol] = df
            console.print(f"  {symbol}: {len(df)} bars", style="dim")
        except Exception as e:
            console.print(f"  {symbol}: failed ({e})", style="dim red")

    console.print(f"Downloaded data for {len(all_data)}/{len(symbols)} symbols\n")
    return all_data


def simulate_orb(data: dict[str, pd.DataFrame], initial_capital: float = 100000) -> BacktestResult:
    """Simulate ORB strategy on historical hourly data."""
    trades = []
    portfolio = initial_capital
    portfolio_history = [portfolio]

    for symbol, df in data.items():
        # Group by trading day
        df["date"] = df.index.date
        days = df.groupby("date")

        for date, day_bars in days:
            if len(day_bars) < 4:
                continue

            # Simulate ORB: first 1-hour bar = "opening range" (approximation)
            first_bar = day_bars.iloc[0]
            orb_high = first_bar["high"]
            orb_low = first_bar["low"]
            orb_range = orb_high - orb_low

            if orb_range <= 0:
                continue

            # Range quality filter
            range_pct = orb_range / ((orb_high + orb_low) / 2)
            if range_pct > config.ORB_MAX_RANGE_PCT:
                continue

            # Look for breakout in subsequent bars
            for i in range(1, len(day_bars)):
                bar = day_bars.iloc[i]

                if bar["close"] > orb_high and bar["volume"] > 0:
                    # Breakout signal
                    entry = orb_high * (1 + config.BACKTEST_SLIPPAGE)
                    tp = entry + config.ORB_TP_MULT * orb_range
                    sl = entry - config.ORB_SL_MULT * orb_range

                    qty = max(1, int((portfolio * config.RISK_PER_TRADE_PCT) / abs(entry - sl)))
                    commission = qty * config.BACKTEST_COMMISSION * 2

                    # Check exit on remaining bars
                    exit_price = entry  # default: close at EOD
                    exit_time = day_bars.index[-1]

                    for j in range(i + 1, len(day_bars)):
                        check = day_bars.iloc[j]
                        if check["high"] >= tp:
                            exit_price = tp * (1 - config.BACKTEST_SLIPPAGE)
                            exit_time = day_bars.index[j]
                            break
                        if check["low"] <= sl:
                            exit_price = sl * (1 - config.BACKTEST_SLIPPAGE)
                            exit_time = day_bars.index[j]
                            break
                    else:
                        exit_price = day_bars.iloc[-1]["close"] * (1 - config.BACKTEST_SLIPPAGE)

                    pnl = (exit_price - entry) * qty - commission
                    portfolio += pnl
                    portfolio_history.append(portfolio)

                    trades.append(BacktestTrade(
                        symbol=symbol, strategy="ORB", side="buy",
                        entry_price=entry, entry_time=day_bars.index[i],
                        qty=qty, take_profit=tp, stop_loss=sl,
                        exit_price=exit_price, exit_time=exit_time,
                        pnl=pnl, commission=commission,
                    ))
                    break  # One trade per symbol per day

    return _compute_result("ORB", trades, portfolio_history, initial_capital)


def simulate_vwap(data: dict[str, pd.DataFrame], initial_capital: float = 100000) -> BacktestResult:
    """Simulate VWAP strategy on historical hourly data."""
    trades = []
    portfolio = initial_capital
    portfolio_history = [portfolio]

    for symbol, df in data.items():
        df["date"] = df.index.date
        days = df.groupby("date")

        for date, day_bars in days:
            if len(day_bars) < 6:
                continue

            # Compute running VWAP
            typical = (day_bars["high"] + day_bars["low"] + day_bars["close"]) / 3
            cum_vol = day_bars["volume"].cumsum()
            cum_vp = (typical * day_bars["volume"]).cumsum()

            for i in range(4, len(day_bars) - 1):
                if cum_vol.iloc[i] == 0:
                    continue

                vwap = cum_vp.iloc[i] / cum_vol.iloc[i]
                cum_vp2 = (typical[:i+1]**2 * day_bars["volume"][:i+1]).cumsum()
                var = cum_vp2.iloc[i] / cum_vol.iloc[i] - vwap**2
                std = np.sqrt(max(var, 0))
                lower = vwap - config.VWAP_BAND_STD * std

                bar = day_bars.iloc[i]
                prev = day_bars.iloc[i - 1]

                # Buy signal: touched lower band and bounced
                if prev["low"] <= lower and bar["close"] > lower:
                    entry = bar["close"] * (1 + config.BACKTEST_SLIPPAGE)
                    sl = lower - config.VWAP_STOP_EXTENSION * std
                    tp = vwap

                    if abs(entry - sl) < 0.01:
                        continue

                    qty = max(1, int((portfolio * config.RISK_PER_TRADE_PCT) / abs(entry - sl)))
                    commission = qty * config.BACKTEST_COMMISSION * 2

                    # Check exit within next few bars (time stop ~3 bars for hourly)
                    exit_price = entry
                    exit_time = day_bars.index[min(i + 3, len(day_bars) - 1)]

                    for j in range(i + 1, min(i + 4, len(day_bars))):
                        check = day_bars.iloc[j]
                        if check["high"] >= tp:
                            exit_price = tp * (1 - config.BACKTEST_SLIPPAGE)
                            exit_time = day_bars.index[j]
                            break
                        if check["low"] <= sl:
                            exit_price = sl * (1 - config.BACKTEST_SLIPPAGE)
                            exit_time = day_bars.index[j]
                            break
                    else:
                        k = min(i + 3, len(day_bars) - 1)
                        exit_price = day_bars.iloc[k]["close"] * (1 - config.BACKTEST_SLIPPAGE)

                    pnl = (exit_price - entry) * qty - commission
                    portfolio += pnl
                    portfolio_history.append(portfolio)

                    trades.append(BacktestTrade(
                        symbol=symbol, strategy="VWAP", side="buy",
                        entry_price=entry, entry_time=day_bars.index[i],
                        qty=qty, take_profit=tp, stop_loss=sl,
                        exit_price=exit_price, exit_time=exit_time,
                        pnl=pnl, commission=commission,
                    ))
                    break  # One VWAP trade per symbol per day

    return _compute_result("VWAP", trades, portfolio_history, initial_capital)


def _compute_result(strategy: str, trades: list[BacktestTrade],
                    portfolio_history: list[float],
                    initial_capital: float) -> BacktestResult:
    """Compute backtest metrics from trades and portfolio history."""
    if not trades:
        return BacktestResult(strategy=strategy)

    total_return = (portfolio_history[-1] - initial_capital) / initial_capital
    days = 126  # ~6 months of trading days
    annualized = (1 + total_return) ** (252 / max(days, 1)) - 1

    # Daily returns for Sharpe
    arr = np.array(portfolio_history)
    daily_returns = np.diff(arr) / arr[:-1]
    daily_rf = config.BACKTEST_RISK_FREE_RATE / 252
    excess = daily_returns - daily_rf
    sharpe = 0.0
    if len(excess) > 1 and np.std(excess) > 0:
        sharpe = float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(252))

    # Win rate
    winners = [t for t in trades if t.pnl > 0]
    wr = len(winners) / len(trades)

    # Profit factor
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    peak = arr[0]
    max_dd = 0.0
    for val in arr:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Average hold time
    hold_times = []
    for t in trades:
        if t.exit_time and t.entry_time:
            delta = t.exit_time - t.entry_time
            hold_times.append(delta.total_seconds() / 60)
    avg_hold = np.mean(hold_times) if hold_times else 0

    return BacktestResult(
        strategy=strategy,
        total_return=total_return,
        annualized_return=annualized,
        sharpe_ratio=sharpe,
        win_rate=wr,
        profit_factor=pf,
        max_drawdown=max_dd,
        total_trades=len(trades),
        avg_hold_minutes=avg_hold,
    )


def run_backtest():
    """Main backtest entry point. Downloads data, runs strategies, prints results."""
    console.print("\n[bold cyan]BACKTESTING ENGINE[/bold cyan]\n")

    # Use top N most liquid symbols
    symbols = config.CORE_SYMBOLS[:config.BACKTEST_TOP_N]
    console.print(f"Symbols: {', '.join(symbols)}")
    console.print(f"Period: 6 months | Slippage: {config.BACKTEST_SLIPPAGE:.2%} | Commission: ${config.BACKTEST_COMMISSION}/share\n")

    # Download data
    data = download_data(symbols, months=6)
    if not data:
        console.print("[bold red]No data downloaded. Cannot backtest.[/bold red]")
        return

    # Run strategies
    console.print("[bold]Running ORB strategy...[/bold]")
    orb_result = simulate_orb(data)

    console.print("[bold]Running VWAP strategy...[/bold]")
    vwap_result = simulate_vwap(data)

    # Print results table
    console.print()
    table = Table(title="Backtest Results (6 Months)", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("ORB", justify="right")
    table.add_column("VWAP", justify="right")

    for label, orb_val, vwap_val, fmt in [
        ("Total Return", orb_result.total_return, vwap_result.total_return, "{:+.1%}"),
        ("Annualized Return", orb_result.annualized_return, vwap_result.annualized_return, "{:+.1%}"),
        ("Sharpe Ratio", orb_result.sharpe_ratio, vwap_result.sharpe_ratio, "{:.2f}"),
        ("Win Rate", orb_result.win_rate, vwap_result.win_rate, "{:.0%}"),
        ("Profit Factor", orb_result.profit_factor, vwap_result.profit_factor, "{:.2f}"),
        ("Max Drawdown", orb_result.max_drawdown, vwap_result.max_drawdown, "{:.1%}"),
        ("Total Trades", orb_result.total_trades, vwap_result.total_trades, "{}"),
        ("Avg Hold (min)", orb_result.avg_hold_minutes, vwap_result.avg_hold_minutes, "{:.0f}"),
    ]:
        table.add_row(label, fmt.format(orb_val), fmt.format(vwap_val))

    console.print(table)

    # Warnings
    for result in [orb_result, vwap_result]:
        if result.total_trades > 0:
            if result.sharpe_ratio < 0.5:
                console.print(f"[yellow]WARNING: {result.strategy} Sharpe ({result.sharpe_ratio:.2f}) < 0.5[/yellow]")
            if result.win_rate < 0.40:
                console.print(f"[yellow]WARNING: {result.strategy} Win Rate ({result.win_rate:.0%}) < 40%[/yellow]")

    # Save to database
    try:
        database.init_db()
        run_date = datetime.now().isoformat()
        for result in [orb_result, vwap_result]:
            database.save_backtest_result(
                run_date=run_date,
                strategy=result.strategy,
                total_return=result.total_return,
                annualized_return=result.annualized_return,
                sharpe_ratio=result.sharpe_ratio,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
                max_drawdown=result.max_drawdown,
                total_trades=result.total_trades,
                avg_hold_minutes=result.avg_hold_minutes,
            )
        console.print("\n[green]Results saved to bot.db (backtest_results table)[/green]")
    except Exception as e:
        console.print(f"[red]Failed to save results: {e}[/red]")

    console.print()


# =============================================================================
# V3: Walk-Forward Backtesting Validation
# =============================================================================

def quick_recent_backtest(strategy: str, days: int = 30) -> dict:
    """Run a fast recent backtest for strategy health check.

    Returns dict with: sharpe_ratio, win_rate, total_trades, total_return
    """
    try:
        symbols = config.STANDARD_SYMBOLS[:10]  # Small set for speed

        import yfinance as yf
        end = datetime.now()
        start = end - timedelta(days=days + 10)  # buffer for weekends

        all_data = {}
        for symbol in symbols:
            try:
                data = yf.download(symbol, start=start, end=end, interval="1h", progress=False)
                if data.empty or len(data) < 20:
                    continue
                data.columns = [c.lower() for c in data.columns]
                all_data[symbol] = data
            except Exception as e:
                logger.debug(f"yfinance download failed for {symbol}: {e}")
                continue

        if not all_data:
            return {"sharpe_ratio": 0.0, "win_rate": 0.0, "total_trades": 0, "total_return": 0.0}

        if strategy == "ORB":
            result = simulate_orb(all_data)
        elif strategy == "VWAP":
            result = simulate_vwap(all_data)
        else:
            return {"sharpe_ratio": 0.0, "win_rate": 0.0, "total_trades": 0, "total_return": 0.0}

        return {
            "sharpe_ratio": round(result.sharpe_ratio, 2),
            "win_rate": round(result.win_rate, 2),
            "total_trades": result.total_trades,
            "total_return": round(result.total_return, 4),
        }
    except Exception as e:
        logger.error(f"Quick backtest failed for {strategy}: {e}")
        return {"sharpe_ratio": 0.0, "win_rate": 0.0, "total_trades": 0, "total_return": 0.0}


def walk_forward_test(n_splits: int = 4):
    """Walk-forward out-of-sample validation.

    Downloads 12 months of data and splits into n_splits folds.
    For each fold: test on fold using default params. Reports OOS metrics.
    """
    console.print("\n[bold cyan]WALK-FORWARD VALIDATION[/bold cyan]\n")

    symbols = config.CORE_SYMBOLS[:config.BACKTEST_TOP_N]
    console.print(f"Symbols: {', '.join(symbols)}")
    console.print(f"Period: 12 months | Folds: {n_splits}\n")

    data = download_data(symbols, months=12)
    if not data:
        console.print("[bold red]No data downloaded. Cannot run walk-forward.[/bold red]")
        return

    # Collect all unique dates
    all_dates = set()
    for sym, df in data.items():
        all_dates.update(df.index.date)
    sorted_dates = sorted(all_dates)

    if len(sorted_dates) < n_splits * 20:
        console.print("[bold red]Not enough data for walk-forward validation.[/bold red]")
        return

    fold_size = len(sorted_dates) // (n_splits + 1)
    orb_results = []
    vwap_results = []

    for fold in range(1, n_splits + 1):
        test_start_idx = fold * fold_size
        test_end_idx = test_start_idx + fold_size
        if test_end_idx > len(sorted_dates):
            break

        test_start_date = sorted_dates[test_start_idx]
        test_end_date = sorted_dates[min(test_end_idx, len(sorted_dates) - 1)]

        test_data = {}
        for sym, df in data.items():
            mask = (df.index.date >= test_start_date) & (df.index.date <= test_end_date)
            test_df = df[mask]
            if len(test_df) >= 20:
                test_data[sym] = test_df

        if not test_data:
            continue

        orb_r = simulate_orb(test_data)
        vwap_r = simulate_vwap(test_data)
        orb_results.append(orb_r)
        vwap_results.append(vwap_r)

        console.print(
            f"  Fold {fold}: {test_start_date} to {test_end_date} | "
            f"ORB: Sharpe {orb_r.sharpe_ratio:.2f}, WR {orb_r.win_rate:.0%} ({orb_r.total_trades}t) | "
            f"VWAP: Sharpe {vwap_r.sharpe_ratio:.2f}, WR {vwap_r.win_rate:.0%} ({vwap_r.total_trades}t)"
        )

    if not orb_results:
        console.print("[bold red]No walk-forward results generated.[/bold red]")
        return

    console.print()
    avg_orb_sharpe = np.mean([r.sharpe_ratio for r in orb_results])
    avg_vwap_sharpe = np.mean([r.sharpe_ratio for r in vwap_results])

    table = Table(title="Walk-Forward Summary (Out-of-Sample)", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("ORB", justify="right")
    table.add_column("VWAP", justify="right")
    table.add_row("Avg OOS Sharpe", f"{avg_orb_sharpe:.2f}", f"{avg_vwap_sharpe:.2f}")
    table.add_row("Avg OOS Win Rate",
                   f"{np.mean([r.win_rate for r in orb_results]):.0%}",
                   f"{np.mean([r.win_rate for r in vwap_results]):.0%}")
    table.add_row("Total OOS Trades",
                   str(sum(r.total_trades for r in orb_results)),
                   str(sum(r.total_trades for r in vwap_results)))
    console.print(table)

    for name, avg in [("ORB", avg_orb_sharpe), ("VWAP", avg_vwap_sharpe)]:
        if avg < 0.5:
            console.print(f"[yellow]WARNING: {name} avg OOS Sharpe ({avg:.2f}) < 0.5[/yellow]")

    try:
        run_date = datetime.now().isoformat()
        for results, strategy in [(orb_results, "ORB_WF"), (vwap_results, "VWAP_WF")]:
            for r in results:
                database.save_backtest_result(
                    run_date=run_date, strategy=strategy,
                    total_return=r.total_return, annualized_return=r.annualized_return,
                    sharpe_ratio=r.sharpe_ratio, win_rate=r.win_rate,
                    profit_factor=r.profit_factor, max_drawdown=r.max_drawdown,
                    total_trades=r.total_trades, avg_hold_minutes=r.avg_hold_minutes,
                )
        console.print("\n[green]Walk-forward results saved to bot.db[/green]")
    except Exception as e:
        console.print(f"[red]Failed to save results: {e}[/red]")

    console.print()


# =============================================================================
# V8: Additional strategy backtests
# =============================================================================

def simulate_stat_mr(data: dict[str, pd.DataFrame], initial_capital: float = 100000,
                     slippage: float = 0.0005, commission_per_share: float = 0.0035) -> BacktestResult | None:
    """Simulate Statistical Mean Reversion strategy.

    At each bar: compute OU params (kappa, mu, sigma), z-score.
    Entry: |z-score| > 1.5, Hurst < 0.52
    Exit: z-score < 0.2 (full), < 0.5 (partial), > 2.5 (stop)
    """
    trades = []
    portfolio = initial_capital
    portfolio_history = [initial_capital]

    for symbol, bars in data.items():
        if bars is None or len(bars) < 50:
            continue

        try:
            from analytics.ou_tools import fit_ou_params, compute_zscore
            from analytics.hurst import hurst_exponent

            closes = bars["close"].values
            position = None

            for i in range(50, len(closes)):
                window = closes[max(0, i - 200):i]

                if len(window) < 30:
                    continue

                params = fit_ou_params(pd.Series(window))
                if not params:
                    continue

                current = closes[i]
                z = compute_zscore(current, params["mu"], params["sigma"])
                h = hurst_exponent(pd.Series(window))

                if position is None:
                    # Entry conditions
                    if h < 0.52 and abs(z) > 1.5:
                        side = "buy" if z < -1.5 else "sell"
                        entry_price = current * (1 + slippage if side == "buy" else 1 - slippage)
                        position = {
                            "symbol": symbol, "side": side, "entry_price": entry_price,
                            "entry_idx": i, "entry_time": bars.index[i],
                            "mu": params["mu"], "sigma": params["sigma"],
                        }
                else:
                    # Exit conditions
                    exit_price = None
                    exit_reason = None

                    if abs(z) < 0.2:
                        exit_price = current * (1 - slippage if position["side"] == "buy" else 1 + slippage)
                        exit_reason = "z_revert"
                    elif abs(z) > 2.5:
                        exit_price = current * (1 - slippage if position["side"] == "buy" else 1 + slippage)
                        exit_reason = "z_stop"
                    elif i - position["entry_idx"] > 500:  # Time stop
                        exit_price = current * (1 - slippage if position["side"] == "buy" else 1 + slippage)
                        exit_reason = "time_stop"

                    if exit_price is not None:
                        qty = max(1, int(portfolio * 0.02 / position["entry_price"]))
                        if position["side"] == "buy":
                            pnl = (exit_price - position["entry_price"]) * qty
                        else:
                            pnl = (position["entry_price"] - exit_price) * qty
                        pnl -= commission_per_share * qty * 2

                        portfolio += pnl
                        portfolio_history.append(portfolio)

                        trades.append(BacktestTrade(
                            symbol=symbol, strategy="STAT_MR", side=position["side"],
                            entry_price=position["entry_price"], exit_price=exit_price,
                            entry_time=position["entry_time"], exit_time=bars.index[i],
                            qty=qty, pnl=pnl,
                            commission=commission_per_share * qty * 2,
                        ))
                        position = None
        except Exception as e:
            logger.debug(f"StatMR backtest failed for {symbol}: {e}")
            continue

    if not trades:
        return None
    return _compute_result("STAT_MR", trades, portfolio_history, initial_capital)


def simulate_kalman_pairs(data: dict[str, pd.DataFrame], initial_capital: float = 100000,
                          slippage: float = 0.0005, commission_per_share: float = 0.0035) -> BacktestResult | None:
    """Simulate Kalman Pairs Trading strategy.

    Weekly pair selection via correlation, Kalman filter hedge ratio,
    entry at |spread_z| > 2.0, exit at |spread_z| < 0.2.
    """
    import config
    trades = []
    portfolio = initial_capital
    portfolio_history = [initial_capital]

    # Test all pairs from sector groups
    pair_candidates = []
    for group_name, symbols in config.SECTOR_GROUPS.items():
        if isinstance(symbols[0], tuple):
            pair_candidates.extend(symbols)
        else:
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    if symbols[i] in data and symbols[j] in data:
                        pair_candidates.append((symbols[i], symbols[j]))

    for sym1, sym2 in pair_candidates[:20]:  # Limit to first 20 pairs
        if sym1 not in data or sym2 not in data:
            continue

        try:
            bars1 = data[sym1]["close"]
            bars2 = data[sym2]["close"]

            # Align dates
            combined = pd.DataFrame({"s1": bars1, "s2": bars2}).dropna()
            if len(combined) < 60:
                continue

            # Check correlation
            corr = combined["s1"].corr(combined["s2"])
            if abs(corr) < 0.80:
                continue

            # Simple hedge ratio (OLS)
            from numpy.polynomial.polynomial import polyfit
            coeffs = np.polyfit(combined["s2"].values, combined["s1"].values, 1)
            hedge_ratio = coeffs[0]

            # Compute spread
            spread = combined["s1"] - hedge_ratio * combined["s2"]
            spread_mean = spread.rolling(20).mean()
            spread_std = spread.rolling(20).std()

            position = None

            for i in range(20, len(spread)):
                if spread_std.iloc[i] < 1e-6:
                    continue
                z = (spread.iloc[i] - spread_mean.iloc[i]) / spread_std.iloc[i]

                if position is None:
                    if abs(z) > 2.0:
                        side = "sell" if z > 2.0 else "buy"  # Sell spread if wide
                        position = {
                            "entry_idx": i, "side": side,
                            "entry_spread": spread.iloc[i],
                            "entry_time": combined.index[i],
                        }
                else:
                    exit_trade = False
                    if abs(z) < 0.2:
                        exit_trade = True
                        reason = "convergence"
                    elif abs(z) > 3.0:
                        exit_trade = True
                        reason = "divergence_stop"
                    elif i - position["entry_idx"] > 50:
                        exit_trade = True
                        reason = "time_stop"

                    if exit_trade:
                        entry_s = position["entry_spread"]
                        exit_s = spread.iloc[i]

                        if position["side"] == "sell":
                            pnl = (entry_s - exit_s) * 10  # Notional
                        else:
                            pnl = (exit_s - entry_s) * 10

                        pnl *= (1 - slippage * 2)
                        portfolio += pnl
                        portfolio_history.append(portfolio)

                        trades.append(BacktestTrade(
                            symbol=f"{sym1}/{sym2}", strategy="KALMAN_PAIRS",
                            side=position["side"],
                            entry_price=entry_s, exit_price=exit_s,
                            entry_time=position["entry_time"],
                            exit_time=combined.index[i],
                            qty=10, pnl=pnl, commission=0.07,
                        ))
                        position = None
        except Exception as e:
            logger.debug(f"Pairs backtest failed for {sym1}/{sym2}: {e}")
            continue

    if not trades:
        return None
    return _compute_result("KALMAN_PAIRS", trades, portfolio_history, initial_capital)


def simulate_micro_momentum(data: dict[str, pd.DataFrame], initial_capital: float = 100000,
                            slippage: float = 0.001) -> BacktestResult | None:
    """Simulate Intraday Micro-Momentum strategy.

    Detect volume spikes in SPY (>3x 20-bar avg) with price move >0.15%.
    On event: 8-minute hold with 0.6% TP / 0.3% SL.
    """
    trades = []
    portfolio = initial_capital
    portfolio_history = [initial_capital]

    if "SPY" not in data:
        return None

    spy = data["SPY"]
    if len(spy) < 30:
        return None

    try:
        spy_close = spy["close"].values
        spy_volume = spy["volume"].values

        for i in range(20, len(spy_close) - 8):
            avg_vol = np.mean(spy_volume[i - 20:i])
            if avg_vol < 1:
                continue

            vol_ratio = spy_volume[i] / avg_vol
            price_move = abs(spy_close[i] - spy_close[i - 1]) / spy_close[i - 1]

            if vol_ratio > 3.0 and price_move > 0.0015:
                direction = "buy" if spy_close[i] > spy_close[i - 1] else "sell"
                entry_price = spy_close[i] * (1 + slippage if direction == "buy" else 1 - slippage)

                # 8-bar hold (simulating 8-minute hold with whatever timeframe we have)
                exit_idx = min(i + 8, len(spy_close) - 1)
                exit_price = spy_close[exit_idx]

                # Apply TP/SL
                tp_price = entry_price * (1 + 0.006 if direction == "buy" else 1 - 0.006)
                sl_price = entry_price * (1 - 0.003 if direction == "buy" else 1 + 0.003)

                for j in range(i + 1, exit_idx + 1):
                    if direction == "buy":
                        if spy_close[j] >= tp_price:
                            exit_price = tp_price * (1 - slippage)
                            break
                        if spy_close[j] <= sl_price:
                            exit_price = sl_price * (1 - slippage)
                            break
                    else:
                        if spy_close[j] <= tp_price:
                            exit_price = tp_price * (1 + slippage)
                            break
                        if spy_close[j] >= sl_price:
                            exit_price = sl_price * (1 + slippage)
                            break

                qty = max(1, int(portfolio * 0.01 / entry_price))
                if direction == "buy":
                    pnl = (exit_price - entry_price) * qty
                else:
                    pnl = (entry_price - exit_price) * qty

                portfolio += pnl
                portfolio_history.append(portfolio)

                trades.append(BacktestTrade(
                    symbol="SPY", strategy="MICRO_MOM", side=direction,
                    entry_price=entry_price, exit_price=exit_price,
                    entry_time=spy.index[i], exit_time=spy.index[exit_idx],
                    qty=qty, pnl=pnl, commission=0.0,
                ))
    except Exception as e:
        logger.debug(f"MicroMom backtest failed: {e}")

    if not trades:
        return None
    return _compute_result("MICRO_MOM", trades, portfolio_history, initial_capital)
