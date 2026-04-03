"""Performance analytics — Sharpe, Sortino, drawdown, attribution."""

import logging
import math
from datetime import datetime, timedelta

import numpy as np

import config
import database

logger = logging.getLogger(__name__)


def sharpe_ratio(daily_returns: list[float], risk_free_rate: float = 0.045) -> float:
    """Annualized Sharpe ratio from daily P&L percentages.

    Args:
        daily_returns: List of daily return percentages (e.g., 0.01 = +1%)
        risk_free_rate: Annual risk-free rate (default 4.5%)
    """
    if len(daily_returns) < 2:
        return 0.0

    arr = np.array(daily_returns)
    daily_rf = risk_free_rate / 252
    excess = arr - daily_rf
    mean_excess = np.mean(excess)
    std = np.std(excess, ddof=1)

    if std == 0:
        return 0.0

    return float(mean_excess / std * np.sqrt(252))


def sortino_ratio(daily_returns: list[float], risk_free_rate: float = 0.045) -> float:
    """Sortino ratio — like Sharpe but only penalizes downside volatility."""
    if len(daily_returns) < 2:
        return 0.0

    arr = np.array(daily_returns)
    daily_rf = risk_free_rate / 252
    excess = arr - daily_rf
    mean_excess = np.mean(excess)

    # Downside deviation: only negative returns
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 99.9 if mean_excess > 0 else 0.0

    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0

    return float(mean_excess / downside_std * np.sqrt(252))


def profit_factor(trades: list[dict]) -> float:
    """Gross profit / gross loss."""
    gross_profit = sum(t["pnl"] for t in trades if t.get("pnl", 0) > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t.get("pnl", 0) < 0))

    if gross_loss == 0:
        return 99.9 if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def max_drawdown(portfolio_values: list[float]) -> float:
    """Largest peak-to-trough decline as a percentage."""
    if len(portfolio_values) < 2:
        return 0.0

    arr = np.array(portfolio_values)
    peak = arr[0]
    max_dd = 0.0

    for val in arr:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    return max_dd


def win_rate(trades: list[dict]) -> float:
    """Percentage of winning trades."""
    if not trades:
        return 0.0
    winners = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return winners / len(trades)


def benchmark_comparison(daily_returns: list[float], spy_returns: list[float]) -> float:
    """Bot return minus SPY return over the same period."""
    if not daily_returns or not spy_returns:
        return 0.0

    bot_total = np.prod([1 + r for r in daily_returns]) - 1
    spy_total = np.prod([1 + r for r in spy_returns]) - 1
    return float(bot_total - spy_total)


def strategy_attribution(trades: list[dict]) -> dict:
    """P&L attribution per strategy.

    Returns: {'ORB': {'trades': N, 'pnl': X, 'win_rate': Y}, ...}
    """
    result = {}
    for t in trades:
        strat = t.get("strategy", "UNKNOWN")
        if strat not in result:
            result[strat] = {"trades": 0, "winners": 0, "pnl": 0.0}
        result[strat]["trades"] += 1
        result[strat]["pnl"] += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            result[strat]["winners"] += 1

    for strat, data in result.items():
        data["win_rate"] = data["winners"] / data["trades"] if data["trades"] > 0 else 0.0

    return result


def _derive_daily_returns_from_trades(trades: list[dict], portfolio_value: float = 1_000_000) -> list[float]:
    """Derive daily returns from trade P&L when snapshot data is unreliable.

    Groups trades by exit date and computes daily return as sum(pnl) / portfolio_value.
    """
    from collections import defaultdict

    daily_pnl = defaultdict(float)
    for t in trades:
        exit_time = t.get("exit_time", "")
        pnl = t.get("pnl", 0) or 0
        if exit_time and pnl != 0:
            day = exit_time[:10]  # YYYY-MM-DD
            daily_pnl[day] += pnl

    if not daily_pnl:
        return []

    # Sort by date and convert to returns
    sorted_days = sorted(daily_pnl.keys())
    return [daily_pnl[d] / portfolio_value for d in sorted_days]


def compute_analytics() -> dict:
    """Compute all analytics from DB data. Returns dict for dashboard."""
    # Get recent trades and daily data
    trades_7d = database.get_recent_trades(days=7)
    trades_all = database.get_all_trades()
    daily_returns = database.get_daily_pnl_series(days=30)
    portfolio_vals = database.get_portfolio_values(days=30)

    # If snapshot daily returns are all near-zero but we have trades with real P&L,
    # derive daily returns from trade data instead
    has_real_snapshots = any(abs(r) > 1e-5 for r in daily_returns)
    if not has_real_snapshots and trades_7d:
        portfolio_val = portfolio_vals[-1] if portfolio_vals else 1_000_000
        daily_returns = _derive_daily_returns_from_trades(trades_all, portfolio_val)

    # Weekly P&L (already net of commissions — V12 6.1)
    week_pnl = sum(t.get("pnl", 0) for t in trades_7d)
    week_pnl_pct = (week_pnl / portfolio_vals[-1] * 100) if portfolio_vals else 0.0

    # V12 6.1: Total commissions tracked
    total_commissions_7d = sum(t.get("commission", 0) or 0 for t in trades_7d)
    total_commissions_all = sum(t.get("commission", 0) or 0 for t in trades_all)

    # Clamp Sharpe/Sortino to reasonable range for display
    def _clamp_ratio(val: float, lo: float = -10.0, hi: float = 10.0) -> float:
        if math.isinf(val) or math.isnan(val):
            return 0.0
        return max(lo, min(hi, val))

    result = {
        "sharpe_7d": _clamp_ratio(sharpe_ratio(daily_returns[-7:])) if len(daily_returns) >= 3 else 0.0,
        "sharpe_30d": _clamp_ratio(sharpe_ratio(daily_returns)) if len(daily_returns) >= 5 else 0.0,
        "sortino_7d": _clamp_ratio(sortino_ratio(daily_returns[-7:])) if len(daily_returns) >= 3 else 0.0,
        "win_rate": win_rate(trades_7d),
        "profit_factor": profit_factor(trades_7d),
        "max_drawdown": max_drawdown(portfolio_vals) if portfolio_vals else 0.0,
        "week_pnl": round(week_pnl, 2),
        "week_pnl_pct": round(week_pnl_pct, 4),
        "strategy_breakdown": strategy_attribution(trades_7d),
        "total_trades_7d": len(trades_7d),
        "total_trades_all": len(trades_all),
        "total_commissions_7d": round(total_commissions_7d, 2),
        "total_commissions_all": round(total_commissions_all, 2),
    }
    return result
