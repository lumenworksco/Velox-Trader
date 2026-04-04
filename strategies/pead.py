"""Post-Earnings Announcement Drift (PEAD) — event-driven swing strategy.

Trades the well-documented drift after earnings surprises:
- LONG: surprise > +5%, volume > 2x, gap up
- SHORT: surprise < -5%, volume > 2x, gap down
- Entry: Day AFTER earnings at market open
- Exit: 10-20 day hold with 5% TP / 3% SL / ATR trailing after +2%
- Max 5 concurrent PEAD positions, 2% of portfolio per position
"""

import logging
from datetime import datetime, timedelta

import config
from strategies.base import Signal

logger = logging.getLogger(__name__)


class PEADStrategy:
    """Post-Earnings Announcement Drift strategy."""

    def __init__(self):
        self.triggered: dict[str, datetime] = {}  # symbol -> last trigger time
        self._candidates: list[dict] = []  # Pending earnings candidates
        self._scanned_today = False

    def scan(self, now: datetime, regime: str = "UNKNOWN") -> list[Signal]:
        """Daily scan -- check for recent earnings surprises.

        Uses yfinance to get earnings surprise data for symbols that reported
        in the last 24 hours. Filters by surprise %, volume, and gap direction.

        Since this is swing, only scan once per day (at or after 9:00 AM).
        Must be fail-open: if yfinance fails, return empty list.
        """
        if not config.PEAD_ENABLED:
            return []

        # Only scan once per day
        if self._scanned_today:
            return []

        # Only scan at or after 9:00 AM ET
        scan_time = now.time()
        from datetime import time as dt_time
        if scan_time < dt_time(9, 0):
            return []

        self._scanned_today = True

        # Check how many PEAD positions we already have (via triggered)
        active_count = len(self.triggered)
        if active_count >= config.PEAD_MAX_POSITIONS:
            logger.debug(f"PEAD: max positions ({config.PEAD_MAX_POSITIONS}) reached")
            return []

        # Reduce in HIGH_VOL_BEAR regime
        if regime == "HIGH_VOL_BEAR":
            logger.info("PEAD: skipping scan in HIGH_VOL_BEAR regime")
            return []

        # Get earnings surprises
        symbols = config.STANDARD_SYMBOLS
        surprises = self._get_earnings_surprises(symbols)
        if not surprises:
            return []

        signals = []
        slots_remaining = config.PEAD_MAX_POSITIONS - active_count

        for data in surprises:
            if slots_remaining <= 0:
                break

            symbol = data["symbol"]
            surprise_pct = data["surprise_pct"]
            volume_ratio = data["volume_ratio"]
            gap_pct = data["gap_pct"]

            # Skip if already triggered recently (within 30 days)
            if symbol in self.triggered:
                last = self.triggered[symbol]
                if (now - last).days < 30:
                    continue

            # Filter: minimum surprise magnitude
            if abs(surprise_pct) < config.PEAD_MIN_SURPRISE_PCT:
                continue

            # Filter: minimum volume ratio
            if volume_ratio < config.PEAD_MIN_VOLUME_RATIO:
                continue

            # Determine direction
            if surprise_pct > 0 and gap_pct > 0:
                side = "buy"
                entry_price = data.get("current_price", 0)
                if entry_price <= 0:
                    continue
                take_profit = round(entry_price * (1 + config.PEAD_TAKE_PROFIT), 2)
                stop_loss = round(entry_price * (1 - config.PEAD_STOP_LOSS), 2)
                reason = (f"PEAD long: surprise={surprise_pct:+.1f}% "
                          f"vol={volume_ratio:.1f}x gap={gap_pct:+.1f}%")
            elif surprise_pct < 0 and gap_pct < 0:
                if not config.ALLOW_SHORT:
                    continue
                if symbol in config.NO_SHORT_SYMBOLS:
                    continue
                side = "sell"
                entry_price = data.get("current_price", 0)
                if entry_price <= 0:
                    continue
                take_profit = round(entry_price * (1 - config.PEAD_TAKE_PROFIT), 2)
                stop_loss = round(entry_price * (1 + config.PEAD_STOP_LOSS), 2)
                reason = (f"PEAD short: surprise={surprise_pct:+.1f}% "
                          f"vol={volume_ratio:.1f}x gap={gap_pct:+.1f}%")
            else:
                # Surprise and gap direction mismatch -- skip
                continue

            signals.append(Signal(
                symbol=symbol,
                strategy="PEAD",
                side=side,
                entry_price=entry_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                reason=reason,
                hold_type="swing",
            ))
            self.triggered[symbol] = now
            slots_remaining -= 1

        logger.info(f"PEAD scan: {len(signals)} signals from {len(surprises)} surprises")
        return signals

    def check_exits(self, open_trades: dict, now: datetime) -> list[dict]:
        """Check time stop (max 20 days), profit target (5%), stop loss (3%).

        Returns list of exit action dicts: {symbol, action: "full", reason}.
        """
        exits = []

        for symbol, trade in open_trades.items():
            if trade.strategy != "PEAD":
                continue

            try:
                # Time stop: max hold days (5), with extension to 10 if profit > 2%
                if hasattr(trade, "entry_time") and trade.entry_time:
                    hold_days = (now - trade.entry_time).total_seconds() / 86400.0
                    # Check if trade is profitable enough to extend hold
                    max_hold = config.PEAD_HOLD_DAYS_MAX  # 5 days default
                    try:
                        from data import get_snapshot
                        snap = get_snapshot(symbol)
                        cur_price = float(snap.latest_trade.price) if snap and snap.latest_trade else None
                        if cur_price and trade.entry_price > 0:
                            if trade.side == "buy":
                                cur_pnl = (cur_price - trade.entry_price) / trade.entry_price
                            else:
                                cur_pnl = (trade.entry_price - cur_price) / trade.entry_price
                            # Extend to 10 days if profit > 2%
                            if cur_pnl > 0.02:
                                max_hold = 10
                    except Exception:
                        pass
                    if hold_days >= max_hold:
                        exits.append({
                            "symbol": symbol,
                            "action": "full",
                            "reason": f"PEAD time stop ({hold_days}d >= {max_hold}d)",
                        })
                        continue

                # P&L based exits using current market price
                try:
                    from data import get_snapshot
                    snap = get_snapshot(symbol)
                    current_price = float(snap.latest_trade.price) if snap and snap.latest_trade else trade.entry_price
                except Exception:
                    current_price = trade.entry_price
                if trade.entry_price <= 0:
                    continue
                if trade.side == "buy":
                    pnl_pct = (current_price - trade.entry_price) / trade.entry_price
                    if pnl_pct >= config.PEAD_TAKE_PROFIT:
                        exits.append({
                            "symbol": symbol,
                            "action": "full",
                            "reason": f"PEAD take profit ({pnl_pct:.1%})",
                        })
                    elif pnl_pct <= -config.PEAD_STOP_LOSS:
                        exits.append({
                            "symbol": symbol,
                            "action": "full",
                            "reason": f"PEAD stop loss ({pnl_pct:.1%})",
                        })
                elif trade.side == "sell":
                    pnl_pct = (trade.entry_price - current_price) / trade.entry_price
                    if pnl_pct >= config.PEAD_TAKE_PROFIT:
                        exits.append({
                            "symbol": symbol,
                            "action": "full",
                            "reason": f"PEAD take profit ({pnl_pct:.1%})",
                        })
                    elif pnl_pct <= -config.PEAD_STOP_LOSS:
                        exits.append({
                            "symbol": symbol,
                            "action": "full",
                            "reason": f"PEAD stop loss ({pnl_pct:.1%})",
                        })

            except Exception as e:
                logger.debug(f"PEAD exit check error for {symbol}: {e}")

        return exits

    # ------------------------------------------------------------------
    # T5-001: Pre-Earnings Implied-Move Exploitation
    # ------------------------------------------------------------------

    def scan_pre_earnings(self, now: datetime, regime: str = "UNKNOWN") -> list[Signal]:
        """Scan for pre-earnings implied-move exploitation opportunities.

        Screens for stocks where historical post-earnings realized move > 1.5x
        the implied move (ATR proxy). Enter 2 days before earnings, sized at
        50% of normal. Exit within 2 days post-earnings or at 1.5x ATR stop.
        """
        if not getattr(config, "PEAD_PRE_EARNINGS_ENABLED", False):
            return []

        if regime == "HIGH_VOL_BEAR":
            return []

        active_count = len(self.triggered)
        if active_count >= config.PEAD_MAX_POSITIONS:
            return []

        symbols = config.STANDARD_SYMBOLS
        upcoming = self._get_upcoming_earnings(symbols, days_ahead=config.PEAD_PRE_ENTRY_DAYS)
        if not upcoming:
            return []

        signals = []
        slots_remaining = config.PEAD_MAX_POSITIONS - active_count

        for data in upcoming:
            if slots_remaining <= 0:
                break

            symbol = data["symbol"]

            # Skip if already triggered recently
            if symbol in self.triggered:
                last = self.triggered[symbol]
                if (now - last).days < 30:
                    continue

            # Compute implied move (ATR proxy)
            implied_move = self._compute_implied_move(symbol)
            if implied_move <= 0:
                continue

            # Check historical realized move vs implied
            realized_move = data.get("avg_realized_move", 0.0)
            if realized_move <= 0:
                continue

            ratio = realized_move / implied_move
            if ratio < config.PEAD_IMPLIED_MOVE_RATIO_THRESHOLD:
                continue

            # Determine direction from historical earnings drift bias
            drift_bias = data.get("drift_bias", 0.0)
            entry_price = data.get("current_price", 0.0)
            if entry_price <= 0:
                continue

            atr_stop = implied_move * config.PEAD_PRE_ATR_STOP_MULT

            if drift_bias >= 0:
                side = "buy"
                stop_loss = round(entry_price - atr_stop, 2)
                take_profit = round(entry_price + atr_stop, 2)
            else:
                if not config.ALLOW_SHORT:
                    continue
                if symbol in config.NO_SHORT_SYMBOLS:
                    continue
                side = "sell"
                stop_loss = round(entry_price + atr_stop, 2)
                take_profit = round(entry_price - atr_stop, 2)

            reason = (
                f"PEAD pre-earnings: implied={implied_move:.2f} "
                f"realized={realized_move:.2f} ratio={ratio:.1f}x "
                f"bias={'long' if drift_bias >= 0 else 'short'}"
            )

            signals.append(Signal(
                symbol=symbol,
                strategy="PEAD",
                side=side,
                entry_price=entry_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                reason=reason,
                hold_type="swing",
            ))
            self.triggered[symbol] = now
            slots_remaining -= 1

        logger.info(f"PEAD pre-earnings scan: {len(signals)} signals from {len(upcoming)} upcoming")
        return signals

    def _compute_implied_move(self, symbol: str) -> float:
        """Estimate implied move from ATR as proxy for options implied volatility.

        Uses 14-day ATR on daily bars as a proxy for the expected move.
        If options data were available, we'd use at-the-money straddle pricing.

        Returns:
            Estimated implied move in dollar terms, or 0.0 on failure.
        """
        try:
            import yfinance as yf
            import pandas as pd

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d")
            if hist is None or len(hist) < 14:
                return 0.0

            # ATR-14 calculation
            high = hist["High"].values
            low = hist["Low"].values
            close = hist["Close"].values

            tr = []
            for i in range(1, len(high)):
                tr.append(max(
                    high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]),
                ))

            if len(tr) < 14:
                return 0.0

            atr = float(pd.Series(tr).rolling(14).mean().iloc[-1])
            return atr if not pd.isna(atr) else 0.0

        except Exception as e:
            logger.debug(f"PEAD implied move computation failed for {symbol}: {e}")
            return 0.0

    def _get_upcoming_earnings(self, symbols: list[str], days_ahead: int = 2) -> list[dict]:
        """Get symbols with earnings in the next `days_ahead` days.

        Returns list of dicts with: symbol, earnings_date, avg_realized_move,
        drift_bias, current_price.
        """
        results = []
        try:
            import yfinance as yf
            import pandas as pd

            now = pd.Timestamp.now(tz="America/New_York")
            cutoff = now + pd.Timedelta(days=days_ahead)

            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    earnings = getattr(ticker, "earnings_dates", None)
                    if earnings is None or (hasattr(earnings, "empty") and earnings.empty):
                        continue

                    # Find next upcoming earnings
                    future_earnings = earnings[earnings.index > now]
                    if future_earnings.empty:
                        continue

                    next_earnings = future_earnings.index[-1]  # Earliest future date
                    if next_earnings > cutoff:
                        continue

                    # Get historical realized moves around past earnings
                    past_earnings = earnings[earnings.index <= now]
                    realized_moves = []
                    drift_biases = []

                    if not past_earnings.empty and "Surprise(%)" in past_earnings.columns:
                        for idx, row in past_earnings.head(8).iterrows():
                            surprise = row.get("Surprise(%)")
                            if pd.notna(surprise):
                                realized_moves.append(abs(float(surprise)))
                                drift_biases.append(float(surprise))

                    avg_realized = float(pd.Series(realized_moves).mean()) if realized_moves else 0.0
                    avg_drift = float(pd.Series(drift_biases).mean()) if drift_biases else 0.0

                    # Convert surprise % to dollar move using current price
                    hist = ticker.history(period="5d")
                    if hist is None or len(hist) < 1:
                        continue
                    current_price = float(hist["Close"].iloc[-1])
                    dollar_realized = current_price * (avg_realized / 100.0)

                    if current_price > 0:
                        results.append({
                            "symbol": symbol,
                            "earnings_date": str(next_earnings.date()),
                            "avg_realized_move": dollar_realized,
                            "drift_bias": avg_drift,
                            "current_price": current_price,
                        })

                except Exception as e:
                    logger.debug(f"PEAD upcoming earnings check failed for {symbol}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"PEAD upcoming earnings scan failed: {e}")

        return results

    def check_pre_earnings_exits(self, open_trades: dict, now: datetime) -> list[dict]:
        """Check exits for pre-earnings positions.

        Exit within 2 days post-earnings or at 1.5x ATR stop.
        """
        exits = []
        for symbol, trade in open_trades.items():
            if trade.strategy != "PEAD":
                continue
            if "pre-earnings" not in getattr(trade, 'exit_reason', ''):
                # Check if this is a pre-earnings trade by reason
                if not hasattr(trade, 'reason'):
                    continue

            try:
                if hasattr(trade, "entry_time") and trade.entry_time:
                    hold_days = (now - trade.entry_time).total_seconds() / 86400.0
                    # Pre-earnings trades exit faster: entry_days + post_days
                    max_hold = config.PEAD_PRE_ENTRY_DAYS + config.PEAD_POST_EXIT_DAYS
                    if hold_days >= max_hold:
                        exits.append({
                            "symbol": symbol,
                            "action": "full",
                            "reason": f"PEAD pre-earnings time stop ({hold_days}d >= {max_hold}d)",
                        })
            except Exception as e:
                logger.debug(f"PEAD pre-earnings exit check error for {symbol}: {e}")

        return exits

    def reset_daily(self):
        """Clear daily state (allow re-scan next day)."""
        self._scanned_today = False
        self._candidates = []

    # IMPL-010: Earnings data cache (per-day, shared across scans)
    _earnings_cache: dict[str, list[dict]] = {}
    _earnings_cache_date: str = ""

    def _get_earnings_surprises(self, symbols: list[str]) -> list[dict]:
        """Fetch recent earnings surprises with fallback data sources and caching.

        IMPL-010: Enhanced pipeline with:
        1. Primary: Financial Modeling Prep (FMP) API (if API key configured)
        2. Fallback: yfinance earnings_dates
        3. Per-day caching to avoid redundant API calls

        Returns list of {symbol, surprise_pct, volume_ratio, gap_pct, current_price}.
        Fail-open: returns empty list on error.
        """
        import pandas as pd

        # Check daily cache
        today_str = datetime.now().strftime("%Y-%m-%d")
        if self._earnings_cache_date == today_str and self._earnings_cache.get(today_str):
            logger.debug("PEAD: returning cached earnings data")
            return self._earnings_cache[today_str]

        results = []

        # Try FMP API first (requires FMP_API_KEY in config)
        fmp_key = getattr(config, "FMP_API_KEY", None)
        if fmp_key:
            fmp_results = self._get_earnings_fmp(symbols, fmp_key)
            if fmp_results:
                results = fmp_results
                logger.info(f"PEAD: {len(results)} earnings from FMP API")

        # Fallback to yfinance if FMP returned nothing
        if not results:
            yf_results = self._get_earnings_yfinance(symbols)
            if yf_results:
                results = yf_results
                logger.info(f"PEAD: {len(results)} earnings from yfinance (fallback)")

        # Cache results for the day
        self._earnings_cache_date = today_str
        self._earnings_cache[today_str] = results

        return results

    def _get_earnings_fmp(self, symbols: list[str], api_key: str) -> list[dict]:
        """Fetch earnings surprises from Financial Modeling Prep API.

        IMPL-010: Uses FMP's earnings calendar and earnings surprise endpoints.
        Requires a valid FMP_API_KEY in config.

        Returns list of {symbol, surprise_pct, volume_ratio, gap_pct, current_price}.
        """
        results = []

        try:
            import requests
        except ImportError:
            logger.debug("PEAD FMP: requests not installed")
            return []

        try:
            from datetime import date as date_cls
            today = date_cls.today()
            yesterday = today - timedelta(days=1)

            # FMP earnings calendar endpoint
            url = (
                f"https://financialmodelingprep.com/api/v3/earning_calendar"
                f"?from={yesterday.isoformat()}&to={today.isoformat()}"
                f"&apikey={api_key}"
            )

            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                logger.debug(f"PEAD FMP: HTTP {resp.status_code}")
                return []

            calendar = resp.json()
            if not isinstance(calendar, list):
                return []

            # Filter to our symbols that reported earnings
            symbol_set = set(symbols)
            for entry in calendar:
                symbol = entry.get("symbol", "")
                if symbol not in symbol_set:
                    continue

                eps_actual = entry.get("eps", None)
                eps_estimate = entry.get("epsEstimated", None)

                if eps_actual is None or eps_estimate is None:
                    continue
                if eps_estimate == 0:
                    continue

                surprise_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100

                # Get price data for volume ratio and gap
                try:
                    price_url = (
                        f"https://financialmodelingprep.com/api/v3/historical-price-full/"
                        f"{symbol}?timeseries=5&apikey={api_key}"
                    )
                    price_resp = requests.get(price_url, timeout=10)
                    if price_resp.status_code != 200:
                        continue
                    price_data = price_resp.json().get("historical", [])
                    if len(price_data) < 2:
                        continue

                    # Most recent first in FMP
                    latest = price_data[0]
                    prior = price_data[1]

                    volume_ratio = (
                        latest.get("volume", 0) / prior.get("volume", 1)
                        if prior.get("volume", 0) > 0 else 0
                    )
                    gap_pct = (
                        (latest.get("open", 0) - prior.get("close", 0))
                        / prior.get("close", 1) * 100
                        if prior.get("close", 0) > 0 else 0
                    )
                    current_price = float(latest.get("close", 0))

                    if current_price > 0:
                        results.append({
                            "symbol": symbol,
                            "surprise_pct": round(surprise_pct, 2),
                            "volume_ratio": round(volume_ratio, 2),
                            "gap_pct": round(gap_pct, 2),
                            "current_price": current_price,
                            "source": "fmp",
                        })
                except Exception as e:
                    logger.debug(f"PEAD FMP price fetch failed for {symbol}: {e}")
                    continue

        except Exception as e:
            logger.debug(f"PEAD FMP earnings fetch failed: {e}")

        return results

    def _get_earnings_yfinance(self, symbols: list[str]) -> list[dict]:
        """Fetch recent earnings surprises via yfinance (fallback).

        Returns list of {symbol, surprise_pct, volume_ratio, gap_pct, current_price}.
        Fail-open: returns empty list on error.
        """
        results = []
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("PEAD: yfinance not installed, skipping")
            return []

        import pandas as pd

        # Suppress yfinance logging noise
        yf_logger = logging.getLogger("yfinance")
        prev_level = yf_logger.level
        yf_logger.setLevel(logging.CRITICAL)

        try:
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)

                    # Get earnings history for surprise data
                    earnings = getattr(ticker, "earnings_dates", None)
                    if earnings is None or (hasattr(earnings, "empty") and earnings.empty):
                        continue

                    now = pd.Timestamp.now(tz="America/New_York")
                    past_earnings = earnings[earnings.index <= now]
                    if past_earnings.empty:
                        continue

                    last_earnings = past_earnings.index[0]
                    days_since = (now - last_earnings).days
                    if days_since > 1:
                        continue  # Only interested in earnings from last 24h

                    # Get surprise percentage
                    row = past_earnings.iloc[0]
                    surprise_pct = 0.0
                    if "Surprise(%)" in past_earnings.columns:
                        val = row.get("Surprise(%)")
                        if pd.notna(val):
                            surprise_pct = float(val)
                    if surprise_pct == 0.0:
                        continue

                    # Get volume ratio and gap from recent price data
                    hist = ticker.history(period="5d")
                    if hist is None or len(hist) < 2:
                        continue

                    last_vol = hist["Volume"].iloc[-1]
                    avg_vol = hist["Volume"].iloc[:-1].mean()
                    volume_ratio = last_vol / avg_vol if avg_vol > 0 else 0

                    gap_pct = ((hist["Open"].iloc[-1] - hist["Close"].iloc[-2])
                               / hist["Close"].iloc[-2]) * 100

                    current_price = float(hist["Close"].iloc[-1])

                    results.append({
                        "symbol": symbol,
                        "surprise_pct": surprise_pct,
                        "volume_ratio": volume_ratio,
                        "gap_pct": gap_pct,
                        "current_price": current_price,
                        "source": "yfinance",
                    })

                except Exception as e:
                    logger.debug(f"PEAD yfinance check failed for {symbol}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"PEAD yfinance scan failed: {e}")
        finally:
            yf_logger.setLevel(prev_level)

        return results
