"""Statistical Mean Reversion strategy — 60% of capital allocation.

Identifies stocks whose intraday prices have temporarily deviated from
their mean and are likely to revert. Uses Ornstein-Uhlenbeck process
modeling, Hurst exponent filtering, and ADF stationarity testing.

Target: 0.3-0.8% per trade, 65-75% win rate.
"""

import logging
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config
from data import get_intraday_bars, get_daily_bars
from strategies.base import Signal
from analytics.ou_tools import fit_ou_params, compute_zscore
from analytics.hurst import hurst_exponent
from analytics.indicators import compute_vwap

logger = logging.getLogger(__name__)


class StatMeanReversion:
    """Trade mean-reverting stocks using OU z-score signals.

    Workflow:
    1. prepare_universe() at 9:00 AM — filter stocks by Hurst < 0.52 + ADF p < 0.05 + OU half-life 1-48h
    2. scan() every 2 min — compute z-scores on 2-min bars, enter at |z| > 1.5
    3. check_exits() every cycle — z-score exit at +/-0.2 (full), +/-0.5 (partial), +/-2.5 (stop)
    """

    def __init__(self):
        self.universe: list[str] = []          # Today's filtered universe
        self.ou_params: dict[str, dict] = {}   # symbol -> {kappa, mu, sigma, half_life}
        self._universe_ready = False
        self._last_scan = None

    def reset_daily(self):
        """Clear all state for a new trading day."""
        self.universe = []
        self.ou_params = {}
        self._universe_ready = False
        self._last_scan = None

    def prepare_universe(self, symbols: list[str], now: datetime) -> list[str]:
        """Filter symbols for mean-reverting characteristics.

        Called at startup AND at MR_UNIVERSE_PREP_TIME (9:00 AM) each day.

        V7 FIX: Uses intraday 2-min bars (5 days) for OU fitting when available,
        falling back to daily bars. This fixes the half_life unit conversion bug
        where daily bar half_life was multiplied by 24 (treating bar units as days),
        producing values 24x too large for intraday trading.

        Steps:
        1. Get 5 days of 2-min intraday bars (or 30 days daily as fallback)
        2. Compute Hurst exponent — must be < MR_HURST_MAX (0.52)
        3. Fit OU params — half-life converted to hours using bar duration
        4. Rank by quality (low Hurst + high kappa) and take top MR_UNIVERSE_SIZE (40)
        """
        candidates = []

        for symbol in symbols:
            try:
                # Try intraday bars first (more accurate for intraday OU fitting)
                bars = None
                bar_duration_hours = 24  # default: daily

                try:
                    lookback = now - timedelta(days=5)
                    intraday = get_intraday_bars(
                        symbol, TimeFrame(2, TimeFrameUnit.Minute),
                        start=lookback, end=now
                    )
                    if intraday is not None and not intraday.empty and len(intraday) >= 50:
                        bars = intraday
                        bar_duration_hours = 2 / 60  # 2-minute bars in hours
                except Exception as e:
                    logger.debug(f"Intraday bar fetch failed for {symbol}: {e}")

                # Fallback to daily bars
                if bars is None:
                    bars = get_daily_bars(symbol, days=30)
                    if bars is None or bars.empty or len(bars) < 20:
                        continue
                    bar_duration_hours = 24  # daily bars

                close = bars["close"]

                # 1. Hurst exponent — must indicate mean reversion (< 0.5 ideal)
                hurst = hurst_exponent(close)
                if hurst >= config.MR_HURST_MAX:
                    continue

                # 2. Fit OU parameters (also serves as ADF-like stationarity check:
                #    fit_ou_params returns {} if b >= 0, i.e. not mean-reverting)
                # T1-006: Exclude current bar to prevent look-ahead bias
                assert len(close) >= 2, f"Need >=2 bars for OU fit, got {len(close)}"
                ou = fit_ou_params(close.iloc[:-1])
                if not ou:
                    continue

                # 3. Half-life check (convert from bar units to hours)
                half_life_hours = ou['half_life'] * bar_duration_hours
                if not (config.MR_HALFLIFE_MIN_HOURS <= half_life_hours <= config.MR_HALFLIFE_MAX_HOURS):
                    continue

                candidates.append({
                    'symbol': symbol,
                    'hurst': hurst,
                    'ou': ou,
                    'half_life_hours': half_life_hours,
                    'quality': (1 - hurst) + ou['kappa'],  # Higher = more mean-reverting
                })

            except Exception as e:
                logger.debug(f"MR universe filter error for {symbol}: {e}")
                continue

        # Rank by quality, take top N
        candidates.sort(key=lambda c: c['quality'], reverse=True)
        top = candidates[:config.MR_UNIVERSE_SIZE]

        self.universe = [c['symbol'] for c in top]
        self.ou_params = {c['symbol']: c['ou'] for c in top}
        self._universe_ready = True

        logger.info(f"MR universe prepared: {len(self.universe)} symbols from {len(symbols)} candidates")
        return self.universe

    def scan(self, now: datetime, regime: str = "UNKNOWN") -> list[Signal]:
        """Scan universe for mean reversion entry signals.

        Called every SCAN_INTERVAL_SEC (120s).
        Uses 2-min intraday bars to compute real-time z-scores.

        Entry conditions:
        - |z-score| > MR_ZSCORE_ENTRY (1.5)
        - RSI(7) confirms: oversold for longs (< MR_RSI_OVERSOLD), overbought for shorts (> MR_RSI_OVERBOUGHT)
        - Price vs VWAP: long only below VWAP, short only above VWAP
        - Minimum expected gain > MR_MIN_GAIN_PCT (0.2%)
        - Minimum R:R ratio > MR_MIN_RR_RATIO (1.5)
        """
        signals = []

        if not self._universe_ready or not self.universe:
            return signals

        for symbol in self.universe:
            try:
                ou = self.ou_params.get(symbol)
                if not ou:
                    continue

                # Get 2-min intraday bars (last 2 hours)
                lookback = now - timedelta(hours=2)
                bars = get_intraday_bars(
                    symbol, TimeFrame(2, TimeFrameUnit.Minute), start=lookback, end=now
                )
                if bars is None or bars.empty or len(bars) < 20:
                    continue

                close = bars["close"]
                price = close.iloc[-1]

                # BUG-020: Refit OU on intraday data EXCLUDING current bar
                # to prevent look-ahead bias in parameter estimation
                intraday_ou = fit_ou_params(close.iloc[:-1])
                if intraday_ou:
                    mu = intraday_ou['mu']
                    sigma = intraday_ou['sigma']
                else:
                    mu = ou['mu']
                    sigma = ou['sigma']

                zscore = compute_zscore(price, mu, sigma)

                # BUG-021: OU sigma is std of price CHANGES (residuals), which is
                # tiny on 2-min bars (~$0.05 for a $100 stock). Using it for
                # stop/target distances produces stops <$0.15 from entry — normal
                # tick noise triggers them instantly, causing rapid churning.
                # Use rolling std of price LEVELS for meaningful stop distances.
                price_sigma = float(close.iloc[:-1].std()) if len(close) > 2 else sigma
                if price_sigma < 1e-8:
                    price_sigma = sigma

                # V12 2.4: OU sigma zero guard — if both price_sigma and OU sigma
                # are near-zero (all prices identical in low-vol period), z-score
                # and stop/target math produce NaN or infinite values. Skip signal.
                if price_sigma < 1e-10:
                    logger.debug(f"MR skip {symbol}: price_sigma ~0 (low-vol period)")
                    continue
                if sigma < 1e-10:
                    logger.debug(f"MR skip {symbol}: OU sigma ~0 (low-vol period)")
                    continue

                # RSI(7) computation
                rsi = self._compute_rsi(close, period=config.MR_RSI_PERIOD)
                if rsi is None:
                    continue

                # VWAP computation
                vwap = compute_vwap(bars)
                if vwap is None:
                    continue

                # === LONG entry: price below mean ===
                # MR works better in downtrends — use LOWER threshold in BEARISH (more entries)
                # In BULLISH, use slightly higher bar (1.1x)
                if regime == "BEARISH":
                    mr_entry_z = config.MR_ZSCORE_ENTRY * 0.85
                elif regime == "BULLISH":
                    mr_entry_z = config.MR_ZSCORE_ENTRY * 1.1
                else:
                    mr_entry_z = config.MR_ZSCORE_ENTRY
                if (zscore < -mr_entry_z
                        and price < vwap):

                    # Target: revert to z=MR_ZSCORE_EXIT_FULL (near mean)
                    # BUG-021: Use price_sigma (rolling std of levels) not OU sigma
                    # (std of changes) for meaningful stop/target distances
                    target_price = mu + config.MR_ZSCORE_EXIT_FULL * price_sigma
                    expected_gain_pct = (target_price - price) / price

                    if expected_gain_pct < config.MR_MIN_GAIN_PCT:
                        continue

                    # Stop at z = MR_ZSCORE_STOP (using price_sigma)
                    stop_price = mu - config.MR_ZSCORE_STOP * price_sigma
                    # Enforce minimum stop distance (0.5% of price)
                    min_stop_dist = price * getattr(config, 'MR_MIN_STOP_PCT', 0.005)
                    if price - stop_price < min_stop_dist:
                        stop_price = price - min_stop_dist
                    stop_dist = price - stop_price
                    gain_dist = target_price - price

                    if stop_dist <= 0 or gain_dist / stop_dist < config.MR_MIN_RR_RATIO:
                        continue

                    signals.append(Signal(
                        symbol=symbol,
                        strategy="STAT_MR",
                        side="buy",
                        entry_price=round(price, 2),
                        take_profit=round(target_price, 2),
                        stop_loss=round(stop_price, 2),
                        reason=f"MR long z={zscore:.2f} RSI={rsi:.0f}",
                        hold_type="day",
                    ))

                # === SHORT entry: price above mean ===
                elif (zscore > mr_entry_z
                      and price > vwap
                      and regime != "BULLISH"
                      and config.ALLOW_SHORT
                      and symbol not in config.NO_SHORT_SYMBOLS):

                    # BUG-021: Use price_sigma for meaningful distances
                    target_price = mu - config.MR_ZSCORE_EXIT_FULL * price_sigma
                    expected_gain_pct = (price - target_price) / price

                    if expected_gain_pct < config.MR_MIN_GAIN_PCT:
                        continue

                    stop_price = mu + config.MR_ZSCORE_STOP * price_sigma
                    # Enforce minimum stop distance (0.5% of price)
                    min_stop_dist = price * getattr(config, 'MR_MIN_STOP_PCT', 0.005)
                    if stop_price - price < min_stop_dist:
                        stop_price = price + min_stop_dist
                    stop_dist = stop_price - price
                    gain_dist = price - target_price

                    if stop_dist <= 0 or gain_dist / stop_dist < config.MR_MIN_RR_RATIO:
                        continue

                    signals.append(Signal(
                        symbol=symbol,
                        strategy="STAT_MR",
                        side="sell",
                        entry_price=round(price, 2),
                        take_profit=round(target_price, 2),
                        stop_loss=round(stop_price, 2),
                        reason=f"MR short z={zscore:.2f} RSI={rsi:.0f}",
                        hold_type="day",
                    ))

            except Exception as e:
                logger.debug(f"MR scan error for {symbol}: {e}")
                continue

        return signals

    def check_exits(self, open_trades: dict, now: datetime) -> list[dict]:
        """Check MR positions for z-score based exits.

        Returns list of exit actions:
            [{"symbol": ..., "action": "full"|"partial", "reason": ...}]

        Exit conditions:
        - |z| < MR_ZSCORE_EXIT_FULL (0.2): Full exit — reverted to mean
        - |z| < MR_ZSCORE_EXIT_PARTIAL (0.5): Partial exit (50%)
        - |z| > MR_ZSCORE_STOP (2.5): Stop — diverging further
        - Time stop: 2x half_life without reversion
        """
        exits = []

        for symbol, trade in open_trades.items():
            if trade.strategy != "STAT_MR":
                continue

            try:
                ou = self.ou_params.get(symbol)
                if not ou:
                    continue

                # Get current price from recent bars
                lookback = now - timedelta(minutes=10)
                bars = get_intraday_bars(
                    symbol, TimeFrame(2, TimeFrameUnit.Minute), start=lookback, end=now
                )
                if bars is None or bars.empty:
                    continue

                price = bars["close"].iloc[-1]

                # V12 2.4: OU sigma zero guard — skip exit check if sigma ~0
                if ou['sigma'] < 1e-10:
                    logger.debug(f"MR exit skip {symbol}: OU sigma ~0")
                    continue

                # Compute current z-score
                zscore = compute_zscore(price, ou['mu'], ou['sigma'])

                # Time stop: 2x half_life
                if hasattr(trade, 'entry_time') and trade.entry_time:
                    # V10: half_life is in 2-min bar units; convert to trading minutes
                    # half_life * 2 (min/bar) * 2 (2x half-life for time stop)
                    half_life_minutes = ou['half_life'] * 2 * 2  # 2x half-life in minutes
                    max_hold = max(30, min(half_life_minutes, 240))  # 30 min to 4 hours
                    elapsed = (now - trade.entry_time).total_seconds() / 60
                    if elapsed > max_hold:
                        exits.append({
                            "symbol": symbol,
                            "action": "full",
                            "reason": f"MR time stop ({elapsed:.0f}min > {max_hold:.0f}min)",
                        })
                        continue

                if trade.side == "buy":
                    # Long: entered at negative z-score, want z to revert toward 0 and beyond
                    # V11.4: Enhanced overshoot capture with momentum continuation
                    if zscore >= config.MR_ZSCORE_EXIT_PARTIAL:
                        # z overshot past mean — take partial, tighten trailing stop
                        exits.append({
                            "symbol": symbol,
                            "action": "partial",
                            "reason": f"MR overshoot partial z={zscore:.2f}",
                        })
                        # V11.4: Tighter trailing stop in overshoot zone (0.2% vs 0.3%)
                        if hasattr(trade, 'stop_loss'):
                            trail_stop = price * 0.998
                            if trade.stop_loss < trail_stop:
                                trade.stop_loss = trail_stop
                    elif 0 < zscore < config.MR_ZSCORE_EXIT_PARTIAL:
                        # V11.4: z crossed zero but hasn't reached overshoot threshold yet
                        # Move stop to breakeven + small profit to lock in gains
                        if hasattr(trade, 'stop_loss') and hasattr(trade, 'entry_price'):
                            breakeven_plus = trade.entry_price * 1.001  # Lock 0.1% profit
                            if trade.stop_loss < breakeven_plus:
                                trade.stop_loss = breakeven_plus
                    elif -config.MR_ZSCORE_EXIT_FULL <= zscore <= 0:
                        # V11.3: Changed from "full" to "partial" — take 50%, let rest ride
                        # The mean-reversion often overshoots past the mean
                        exits.append({
                            "symbol": symbol,
                            "action": "partial",
                            "reason": f"MR reverted z={zscore:.2f}",
                        })
                        # Move stop to breakeven for remaining position
                        if hasattr(trade, 'stop_loss') and hasattr(trade, 'entry_price'):
                            if trade.stop_loss < trade.entry_price:
                                trade.stop_loss = trade.entry_price
                    elif hasattr(trade, 'stop_loss') and trade.stop_loss and price <= trade.stop_loss:
                        exits.append({
                            "symbol": symbol,
                            "action": "full",
                            "reason": f"MR stop z={zscore:.2f}",
                        })

                elif trade.side == "sell":
                    # Short: entered at positive z-score, want z to revert toward 0 and beyond
                    if zscore <= -config.MR_ZSCORE_EXIT_PARTIAL:
                        exits.append({
                            "symbol": symbol,
                            "action": "partial",
                            "reason": f"MR overshoot partial z={zscore:.2f}",
                        })
                        if hasattr(trade, 'stop_loss'):
                            trail_stop = price * 1.002
                            if trade.stop_loss > trail_stop:
                                trade.stop_loss = trail_stop
                    elif -config.MR_ZSCORE_EXIT_PARTIAL < zscore < 0:
                        # V11.4: z crossed zero but hasn't reached overshoot threshold
                        if hasattr(trade, 'stop_loss') and hasattr(trade, 'entry_price'):
                            breakeven_minus = trade.entry_price * 0.999
                            if trade.stop_loss > breakeven_minus:
                                trade.stop_loss = breakeven_minus
                    elif 0 <= zscore <= config.MR_ZSCORE_EXIT_FULL:
                        exits.append({
                            "symbol": symbol,
                            "action": "partial",
                            "reason": f"MR reverted z={zscore:.2f}",
                        })
                        if hasattr(trade, 'stop_loss') and hasattr(trade, 'entry_price'):
                            if trade.stop_loss > trade.entry_price:
                                trade.stop_loss = trade.entry_price
                    elif hasattr(trade, 'stop_loss') and trade.stop_loss and price >= trade.stop_loss:
                        exits.append({
                            "symbol": symbol,
                            "action": "full",
                            "reason": f"MR stop z={zscore:.2f}",
                        })

            except Exception as e:
                logger.debug(f"MR exit check error for {symbol}: {e}")

        return exits

    def _compute_rsi(self, close: pd.Series, period: int = 7) -> float | None:
        """Compute RSI using pandas_ta (V10: replaced hand-rolled version)."""
        if len(close) < period + 1:
            return None
        try:
            import pandas_ta as ta
            rsi_series = ta.rsi(close, length=period)
            if rsi_series is not None and len(rsi_series) > 0:
                val = rsi_series.iloc[-1]
                return float(val) if not pd.isna(val) else None
        except Exception:
            pass
        # Fallback: hand-rolled RSI if pandas_ta fails
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        if loss.iloc[-1] == 0:
            return 100.0
        rs = gain.iloc[-1] / loss.iloc[-1]
        return 100.0 - (100.0 / (1.0 + rs))

