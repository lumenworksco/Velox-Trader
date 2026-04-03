"""VWAP Mean Reversion strategy — 20% of capital allocation.

Combines VWAP band deviation with OU z-score confirmation and bid-ask
spread filtering.

Entry: VWAP deviation AND OU z-score both agree price is cheap/expensive
Exit:  Handled by ExitManager (time stops, trailing stops, scaled TP)
"""

import logging
import math
from datetime import datetime

import pandas_ta as ta
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config
from data import get_intraday_bars, get_snapshot
from strategies.base import Signal, VWAPState
from analytics.ou_tools import fit_ou_params, compute_zscore
from analytics.indicators import compute_vwap_bands

logger = logging.getLogger(__name__)


class VWAPStrategy:
    """VWAP Mean Reversion V2 — hybrid VWAP + OU z-score."""

    def __init__(self, symbols: list[str] | None = None):
        self.symbols: list[str] = symbols or config.STANDARD_SYMBOLS
        self.states: dict[str, VWAPState] = {}
        self.triggered: dict[str, datetime] = {}  # cooldown tracking
        self.daily_moves: dict[str, float] = {}

    def reset_daily(self):
        self.states.clear()
        self.triggered.clear()
        self.daily_moves.clear()

    def scan(self, now: datetime, regime: str, symbols: list[str] | None = None) -> list[Signal]:
        """Scan for VWAP mean reversion signals with OU z-score confirmation."""
        scan_symbols = symbols if symbols is not None else self.symbols
        signals = []
        today = now.date()
        market_open = datetime(today.year, today.month, today.day, 9, 30, tzinfo=config.ET)

        for symbol in scan_symbols:
            # Cooldown: don't re-trigger within 5 minutes
            if symbol in self.triggered:
                if (now - self.triggered[symbol]).total_seconds() < 300:
                    continue

            try:
                # Get all 1-min bars from open until now
                bars = get_intraday_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), start=market_open, end=now)
                if bars.empty or len(bars) < 15:
                    continue

                # Trend filter: skip if stock moved > 3% today
                first_price = bars["open"].iloc[0]
                last_price = bars["close"].iloc[-1]
                day_move = abs(last_price - first_price) / first_price
                self.daily_moves[symbol] = day_move

                if day_move > config.MAX_INTRADAY_MOVE_PCT:
                    continue

                # Compute VWAP + bands
                result = compute_vwap_bands(bars, getattr(config, "VWAP_BAND_STD", 2.0))
                if result is None:
                    continue
                vwap, upper, lower = result

                # Compute RSI(14)
                rsi_series = ta.rsi(bars["close"], length=14)
                if rsi_series is None or rsi_series.empty:
                    continue
                rsi = rsi_series.iloc[-1]

                # OU z-score confirmation on intraday bars
                # T1-010: Use NaN instead of None to distinguish fit failure from neutral z=0.0
                # (None is falsy and downstream code could conflate it with 0.0)
                ou_zscore = float('nan')
                ou = None
                try:
                    close = bars["close"]
                    # T1-006: Exclude current bar to prevent look-ahead bias in OU fitting
                    assert len(close) >= 2, f"Need >=2 bars for OU fit, got {len(close)}"
                    ou = fit_ou_params(close.iloc[:-1])
                    if ou:
                        # V12 2.4: OU sigma zero guard — skip if sigma ~0 (low-vol period)
                        if ou['sigma'] < 1e-10:
                            logger.debug(f"VWAP skip {symbol}: OU sigma ~0 (low-vol period)")
                            continue
                        ou_zscore = compute_zscore(last_price, ou['mu'], ou['sigma'])
                    else:
                        logger.debug("T1-010: OU fit returned empty for %s, z-score=NaN", symbol)
                except Exception as e:
                    logger.warning(f"OU fit failed for {symbol}: {e}")  # V10: WARNING not debug

                # Bid-ask spread check (skip wide-spread stocks)
                try:
                    snap = get_snapshot(symbol)
                    if snap and snap.latest_quote:
                        bid = float(snap.latest_quote.bid_price)
                        ask = float(snap.latest_quote.ask_price)
                        if bid > 0 and ask > 0:
                            spread_pct = (ask - bid) / last_price
                            if spread_pct > config.VWAP_MAX_SPREAD_PCT:
                                continue
                except Exception as e:
                    logger.debug(f"Snapshot fetch failed for {symbol}: {e}")
                    pass  # Snapshot failure is non-fatal

                prev_bar = bars.iloc[-2]
                curr_bar = bars.iloc[-1]

                # Volume ratio check
                vol_ratio = 1.0
                if len(bars) >= 20:
                    avg_vol = bars["volume"].iloc[-20:].mean()
                    if avg_vol > 0:
                        vol_ratio = bars["volume"].iloc[-1] / avg_vol

                # BUY signal: price touched lower band and bounced back above
                # T1-010: Require valid OU z-score — skip symbol if OU fit failed (NaN)
                if math.isnan(ou_zscore):
                    continue
                ou_buy_ok = ou_zscore < -config.VWAP_OU_ZSCORE_MIN
                if (prev_bar["low"] <= lower
                        and curr_bar["close"] > lower
                        and rsi < config.VWAP_RSI_OVERSOLD
                        and ou_buy_ok
                        and vol_ratio > config.VWAP_VOLUME_RATIO):

                    # Confirmation bar check
                    if config.VWAP_CONFIRMATION_BARS >= 2 and len(bars) >= 3:
                        bar_2ago = bars.iloc[-3]
                        if not (bar_2ago["low"] <= lower and prev_bar["close"] > prev_bar["open"]):
                            continue

                    std_dev = (upper - vwap) / config.VWAP_BAND_STD
                    stop_loss = lower - config.VWAP_STOP_EXTENSION * std_dev

                    entry = curr_bar["close"]
                    min_stop = entry * config.VWAP_MIN_STOP_PCT
                    if abs(entry - stop_loss) < min_stop:
                        stop_loss = entry - min_stop

                    # Use tighter of VWAP and OU targets
                    vwap_target = vwap
                    if ou and not math.isnan(ou_zscore) and ou_zscore < -1.5:
                        ou_target = ou['mu']
                        target = min(vwap_target, ou_target)  # More conservative
                    else:
                        target = vwap_target

                    # Skip if risk/reward is worse than 1:1
                    reward = abs(target - entry)
                    risk = abs(entry - stop_loss)
                    if risk > 0 and reward / risk < getattr(config, 'MR_MIN_RR_RATIO', 1.0):
                        continue

                    signals.append(Signal(
                        symbol=symbol,
                        strategy="VWAP",
                        side="buy",
                        entry_price=round(entry, 2),
                        take_profit=round(target, 2),
                        stop_loss=round(stop_loss, 2),
                        reason=f"VWAP+OU bounce z={ou_zscore if not math.isnan(ou_zscore) else 'N/A'}, RSI={rsi:.1f}",
                        hold_type="day",
                    ))
                    self.triggered[symbol] = now

                # SHORT signal: upper band rejection
                elif (
                    config.ALLOW_SHORT
                    and regime in ("BEARISH", "UNKNOWN")
                    and symbol not in config.NO_SHORT_SYMBOLS
                    and prev_bar["high"] >= upper
                    and curr_bar["close"] < upper
                    and rsi > config.VWAP_RSI_OVERBOUGHT
                    and ou_zscore > config.VWAP_OU_ZSCORE_MIN
                    and day_move > 0.01
                    and vol_ratio > config.VWAP_VOLUME_RATIO
                ):
                    if config.VWAP_CONFIRMATION_BARS >= 2 and len(bars) >= 3:
                        bar_2ago = bars.iloc[-3]
                        if not (bar_2ago["high"] >= upper and prev_bar["close"] < prev_bar["open"]):
                            continue

                    std_dev = (upper - vwap) / config.VWAP_BAND_STD
                    stop_loss = upper + config.VWAP_STOP_EXTENSION * std_dev

                    entry = curr_bar["close"]
                    min_stop = entry * config.VWAP_MIN_STOP_PCT
                    if abs(stop_loss - entry) < min_stop:
                        stop_loss = entry + min_stop

                    # Use tighter of VWAP and OU targets
                    vwap_target = vwap
                    if ou and not math.isnan(ou_zscore) and ou_zscore > 1.5:
                        ou_target = ou['mu']
                        target = max(vwap_target, ou_target)
                    else:
                        target = vwap_target

                    # Skip if risk/reward is worse than 1:1
                    reward = abs(entry - target)
                    risk = abs(stop_loss - entry)
                    if risk > 0 and reward / risk < getattr(config, 'MR_MIN_RR_RATIO', 1.0):
                        continue

                    signals.append(Signal(
                        symbol=symbol,
                        strategy="VWAP",
                        side="sell",
                        entry_price=round(entry, 2),
                        take_profit=round(target, 2),
                        stop_loss=round(stop_loss, 2),
                        reason=f"VWAP+OU rejection z={ou_zscore if not math.isnan(ou_zscore) else 'N/A'}, RSI={rsi:.1f}",
                        hold_type="day",
                    ))
                    self.triggered[symbol] = now

            except Exception as e:
                logger.warning(f"VWAP scan error for {symbol}: {e}")

        return signals

    def check_exits(self, open_trades: list, now: datetime) -> list[dict]:
        """Check VWAP positions for exit signals.

        VWAP exits are handled entirely by ExitManager (time stops, trailing
        stops, scaled TP). This passthrough satisfies the common strategy
        interface.
        """
        return []
