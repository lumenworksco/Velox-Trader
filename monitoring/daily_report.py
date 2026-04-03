"""V12 10.2: Daily P&L Telegram Report — comprehensive EOD performance summary.

Scheduled at 4:05 PM ET daily. Sends a Telegram message with:
  - Daily P&L ($ and %)
  - Win rate
  - Trade count
  - Best/worst trade
  - Per-strategy breakdown
  - Current drawdown from peak

Usage:
    from monitoring.daily_report import DailyReportGenerator

    generator = DailyReportGenerator()
    generator.send_daily_report(risk_manager, analytics)
"""

import logging
from datetime import datetime
from typing import Optional

import config

logger = logging.getLogger(__name__)


class DailyReportGenerator:
    """Generates and sends comprehensive daily P&L reports via Telegram.

    Pulls data from the RiskManager's day summary and the database's
    daily snapshot history to compute drawdown and per-strategy metrics.
    """

    def __init__(self):
        self._last_report_date: Optional[str] = None
        logger.info("V12 10.2: DailyReportGenerator initialized")

    def has_sent_today(self) -> bool:
        """Check if report was already sent for the current date."""
        today = datetime.now(config.ET).strftime("%Y-%m-%d")
        return self._last_report_date == today

    def send_daily_report(self, risk_manager, analytics: Optional[dict] = None) -> bool:
        """Generate and send the daily P&L report.

        Args:
            risk_manager: The RiskManager instance with day summary.
            analytics: Optional analytics dict with max_drawdown, sharpe, etc.

        Returns:
            True if report was sent successfully, False otherwise.
        """
        if not config.TELEGRAM_ENABLED:
            logger.debug("V12 10.2: Telegram disabled, skipping daily report")
            return False

        today = datetime.now(config.ET).strftime("%Y-%m-%d")
        if self._last_report_date == today:
            logger.debug("V12 10.2: Daily report already sent for %s", today)
            return False

        try:
            report_text = self._build_report(risk_manager, analytics)
            self._send(report_text)
            self._last_report_date = today
            logger.info("V12 10.2: Daily P&L report sent for %s", today)
            return True
        except Exception as e:
            logger.error("V12 10.2: Failed to send daily report: %s", e)
            return False

    def _build_report(self, risk_manager, analytics: Optional[dict] = None) -> str:
        """Build the formatted daily report message."""
        now = datetime.now(config.ET)
        summary = risk_manager.get_day_summary()

        # -- Header --
        lines = [
            "\U0001f4ca *DAILY P&L REPORT*",
            f"Date: {now.strftime('%A, %B %d, %Y')}",
            "",
        ]

        # -- P&L --
        n_trades = summary.get("trades", 0)
        if n_trades == 0:
            lines.append("_No trades today._")
            lines.append("")
        else:
            total_pnl = summary.get("total_pnl", 0)
            pnl_pct = summary.get("pnl_pct", 0)
            win_rate = summary.get("win_rate", 0)
            winners = summary.get("winners", 0)
            losers = summary.get("losers", 0)

            pnl_emoji = "\u2705" if total_pnl >= 0 else "\u274c"

            lines.extend([
                f"{pnl_emoji} *P&L: ${total_pnl:+,.2f} ({pnl_pct:+.2%})*",
                "",
                "\U0001f4c8 *Performance*",
                f"  Trades: {n_trades} ({winners}W / {losers}L)",
                f"  Win Rate: {win_rate:.1%}",
                f"  Best: {summary.get('best_trade', 'N/A')}",
                f"  Worst: {summary.get('worst_trade', 'N/A')}",
                "",
            ])

            # -- Per-strategy breakdown --
            strategy_lines = self._build_strategy_breakdown(summary)
            if strategy_lines:
                lines.append("\U0001f3af *Strategy Breakdown*")
                lines.extend(strategy_lines)
                lines.append("")

        # -- Drawdown --
        drawdown_pct = self._compute_drawdown()
        if drawdown_pct > 0:
            dd_emoji = "\u26a0\ufe0f" if drawdown_pct > 0.03 else "\U0001f4c9"
            lines.append(f"{dd_emoji} *Drawdown from peak: {drawdown_pct:.2%}*")
        else:
            lines.append("\U0001f3c6 *At equity peak*")

        # -- Portfolio info --
        equity = getattr(risk_manager, 'current_equity', 0)
        open_positions = len(getattr(risk_manager, 'open_trades', {}))
        if equity:
            lines.append(f"\n\U0001f4b0 Equity: ${equity:,.2f} | Open: {open_positions}")

        # -- Analytics extras --
        if analytics:
            sharpe = analytics.get("sharpe_7d", None)
            if sharpe is not None:
                lines.append(f"\U0001f4ca 7d Sharpe: {sharpe:.2f}")

        return "\n".join(lines)

    def _build_strategy_breakdown(self, summary: dict) -> list[str]:
        """Extract per-strategy stats from the summary dict."""
        lines = []
        for strategy in config.STRATEGY_ALLOCATIONS:
            key = f"{strategy.lower()}_win_rate"
            if key in summary:
                lines.append(f"  {strategy}: {summary[key]}")

        # Also try to get per-strategy P&L from database
        try:
            import database
            today_trades = database.get_recent_trades(days=1)
            if today_trades:
                strat_pnl: dict[str, float] = {}
                for t in today_trades:
                    s = t.get("strategy", "UNKNOWN")
                    strat_pnl[s] = strat_pnl.get(s, 0) + (t.get("pnl", 0) or 0)

                if strat_pnl:
                    lines = []  # Replace with richer breakdown
                    for strat, pnl in sorted(strat_pnl.items(), key=lambda x: -x[1]):
                        emoji = "\u2705" if pnl >= 0 else "\u274c"
                        # Count trades for this strategy
                        strat_trades = [t for t in today_trades if t.get("strategy") == strat]
                        wins = sum(1 for t in strat_trades if (t.get("pnl") or 0) > 0)
                        lines.append(
                            f"  {emoji} {strat}: ${pnl:+,.2f} ({wins}/{len(strat_trades)} W/L)"
                        )
        except Exception as e:
            logger.debug("Strategy P&L breakdown from DB failed: %s", e)

        return lines

    def _compute_drawdown(self) -> float:
        """Compute current drawdown from portfolio peak using daily snapshots."""
        try:
            import database
            values = database.get_portfolio_values(days=90)
            if not values or len(values) < 2:
                return 0.0

            peak = max(values)
            current = values[-1]  # Most recent
            if peak <= 0:
                return 0.0
            return max(0.0, (peak - current) / peak)
        except Exception as e:
            logger.debug("Drawdown computation failed: %s", e)
            return 0.0

    def _send(self, message: str):
        """Send the report via Telegram."""
        try:
            import notifications
            notifications._send_telegram(message)
        except Exception as e:
            logger.error("V12 10.2: Telegram send failed: %s", e)
            raise
