"""Intraday Micro-Momentum — 15% of capital allocation.

Capitalizes on quick momentum moves following economic data releases
(detected via SPY volume spikes). Trades the highest-beta stocks
in the direction of the SPY move. Very short hold time (max 8 min).

Target: 0.3-0.6% per trade, high frequency on event days.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config
from data import get_intraday_bars
from strategies.base import Signal

logger = logging.getLogger(__name__)

# WIRE-004: Order book imbalance confirmation (fail-open)
_order_book_analyzer = None
try:
    from microstructure.order_book import OrderBookAnalyzer as _OBA
    _order_book_analyzer = _OBA()
except ImportError:
    _OBA = None


# Beta table — sourced from config for tunability
STOCK_BETAS = config.MICRO_BETA_TABLE

# V12 AUDIT: Dynamic beta estimation cache {symbol: (date_str, beta)}
_dynamic_beta_cache: dict[str, tuple[str, float]] = {}


def _get_dynamic_beta(symbol: str, fallback_beta: float) -> float:
    """Compute 60-day rolling beta vs SPY. Fall back to config if unavailable.

    Results are cached per trading day to avoid repeated API calls.
    """
    from datetime import date as _date
    today_str = _date.today().isoformat()

    # Return cached value if computed today
    if symbol in _dynamic_beta_cache:
        cached_date, cached_beta = _dynamic_beta_cache[symbol]
        if cached_date == today_str:
            return cached_beta

    try:
        from data import get_bars
        spy_bars = get_bars("SPY", timeframe="1Day", limit=60)
        sym_bars = get_bars(symbol, timeframe="1Day", limit=60)
        if spy_bars is not None and sym_bars is not None and len(spy_bars) >= 20 and len(sym_bars) >= 20:
            spy_ret = spy_bars['close'].pct_change().dropna()
            sym_ret = sym_bars['close'].pct_change().dropna()
            # Align indices
            aligned = spy_ret.align(sym_ret, join='inner')
            if len(aligned[0]) >= 20:
                cov = np.cov(aligned[1], aligned[0])
                beta = cov[0, 1] / max(cov[1, 1], 1e-10)
                beta = max(0.5, min(3.0, beta))  # Clamp to reasonable range
                _dynamic_beta_cache[symbol] = (today_str, beta)
                return beta
    except Exception:
        pass

    _dynamic_beta_cache[symbol] = (today_str, fallback_beta)
    return fallback_beta


class IntradayMicroMomentum:
    """Trade high-beta stocks during economic data release momentum.

    Workflow:
    1. detect_event() — check if SPY has a volume spike + price move
    2. scan() — on event detection, select top-beta stocks in SPY direction
    3. check_exits() — 8-minute hard time stop, or target/stop hit
    4. Disabled if daily P&L > MICRO_MAX_DAILY_GAIN_DISABLE (+1.5%)
    """

    def __init__(self):
        self._event_active = False
        self._event_direction: str = ""  # "up" or "down"
        self._event_time: datetime | None = None
        self._daily_trade_count = 0
        self._trades_this_event = 0
        self._triggered_symbols: set = set()

    def reset_daily(self):
        self._event_active = False
        self._event_direction = ""
        self._event_time = None
        self._daily_trade_count = 0
        self._trades_this_event = 0
        self._triggered_symbols = set()

    def detect_event(self, now: datetime) -> bool:
        """Detect a potential economic data release via SPY volume spike.

        Checks if:
        1. SPY 1-min volume > MICRO_SPY_VOL_SPIKE_MULT (3.0) × 20-bar avg volume
        2. SPY |price move| > MICRO_SPY_MIN_MOVE_PCT (0.15%) in last bar

        Returns True if event detected.
        """
        # Cooldown: don't detect new events within cooldown period
        if self._event_time and (now - self._event_time).total_seconds() < config.MICRO_EVENT_COOLDOWN_SEC:
            return self._event_active

        try:
            lookback = now - timedelta(minutes=30)
            bars = get_intraday_bars("SPY", TimeFrame(1, TimeFrameUnit.Minute), start=lookback, end=now)

            if bars is None or bars.empty or len(bars) < 20:
                return False

            volume = bars["volume"]
            close = bars["close"]

            # V10 BUG-032: Exclude current bar from rolling average to avoid self-inflation
            avg_vol = volume.iloc[:-1].rolling(20).mean().iloc[-1]
            if avg_vol <= 0:
                return False

            vol_ratio = volume.iloc[-1] / avg_vol

            # Price move check
            if len(close) < 2:
                return False
            price_move_pct = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]

            if (vol_ratio >= config.MICRO_SPY_VOL_SPIKE_MULT
                    and abs(price_move_pct) >= config.MICRO_SPY_MIN_MOVE_PCT):

                self._event_active = True
                self._event_direction = "up" if price_move_pct > 0 else "down"
                self._event_time = now
                self._trades_this_event = 0

                logger.info(
                    f"MICRO event detected: SPY {self._event_direction} "
                    f"vol_ratio={vol_ratio:.1f}x move={price_move_pct:.3%}"
                )
                return True

            return False

        except Exception as e:
            logger.debug(f"Micro event detection error: {e}")
            return False

    def scan(self, now: datetime, day_pnl_pct: float = 0.0, regime: str = "UNKNOWN") -> list[Signal]:
        """Generate signals for highest-beta stocks in SPY direction.

        Called every scan cycle. Only generates signals during active events.

        Constraints:
        - Max MICRO_MAX_TRADES_PER_EVENT (3) per event
        - Disabled if day P&L > MICRO_MAX_DAILY_GAIN_DISABLE (+1.5%)
        - Top MICRO_TOP_BETA_STOCKS (5) highest-beta from standard symbols
        """
        signals = []

        if not self._event_active:
            return signals

        # Disable if day P&L is already good enough
        if day_pnl_pct >= config.MICRO_MAX_DAILY_GAIN_DISABLE:
            return signals

        # Max trades per event
        if self._trades_this_event >= config.MICRO_MAX_TRADES_PER_EVENT:
            return signals

        # Event window: only trade within event window
        if self._event_time and (now - self._event_time).total_seconds() > config.MICRO_EVENT_WINDOW_SEC:
            self._event_active = False
            return signals

        # Select top-beta stocks not already triggered
        # V12 AUDIT: Dynamic beta estimation with hardcoded fallback
        candidates = [
            (sym, _get_dynamic_beta(sym, beta))
            for sym, beta in STOCK_BETAS.items()
            if sym not in self._triggered_symbols
            and sym not in config.LEVERAGED_ETFS
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:config.MICRO_TOP_BETA_STOCKS]

        for symbol, beta in top:
            if self._trades_this_event >= config.MICRO_MAX_TRADES_PER_EVENT:
                break

            try:
                # Get current price
                lookback = now - timedelta(minutes=5)
                bars = get_intraday_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), start=lookback, end=now)
                if bars is None or bars.empty:
                    continue

                price = bars["close"].iloc[-1]

                # V11.3 T12: ATR-based stops — adapt to current volatility
                # instead of fixed 1% stop / 2% target
                atr_stop_mult = 1.5  # Stop at 1.5x ATR
                atr_target_mult = 3.0  # Target at 3x ATR (2:1 R:R)
                try:
                    if len(bars) >= 5:
                        high = bars["high"].iloc[-5:]
                        low = bars["low"].iloc[-5:]
                        close_prev = bars["close"].iloc[-6:-1] if len(bars) > 5 else bars["close"].iloc[-5:]
                        tr = pd.concat([
                            high - low,
                            (high - close_prev).abs(),
                            (low - close_prev).abs(),
                        ], axis=1).max(axis=1)
                        atr_val = float(tr.mean())
                    else:
                        atr_val = price * config.MICRO_STOP_PCT  # Fallback
                except Exception:
                    atr_val = price * config.MICRO_STOP_PCT

                # Floor and ceiling for stop distance
                stop_dist = atr_val * atr_stop_mult
                min_stop = price * 0.003  # Floor: 0.3%
                max_stop = price * 0.020  # Ceiling: 2.0%
                stop_dist = max(min_stop, min(stop_dist, max_stop))
                target_dist = stop_dist * (atr_target_mult / atr_stop_mult)  # Maintain R:R

                if self._event_direction == "up":
                    # Buy high-beta stocks
                    stop_loss = price - stop_dist
                    take_profit = price + target_dist
                    side = "buy"
                else:
                    # Short high-beta stocks
                    if not config.ALLOW_SHORT or symbol in config.NO_SHORT_SYMBOLS:
                        continue
                    stop_loss = price + stop_dist
                    take_profit = price - target_dist
                    side = "sell"

                # WIRE-004: Order book imbalance confirmation (fail-open)
                # Skip signal if imbalance contradicts trade direction
                _imbalance_ok = True
                try:
                    if _order_book_analyzer is not None:
                        imb = _order_book_analyzer.get_rolling_imbalance(symbol)
                        # For buys, require non-negative imbalance; for sells, non-positive
                        if side == "buy" and imb < -0.3:
                            _imbalance_ok = False
                            logger.debug("WIRE-004: %s buy skipped — order book imbalance %.2f", symbol, imb)
                        elif side == "sell" and imb > 0.3:
                            _imbalance_ok = False
                            logger.debug("WIRE-004: %s sell skipped — order book imbalance %.2f", symbol, imb)
                except Exception as _e:
                    logger.debug("WIRE-004: Order book check failed for %s (fail-open): %s", symbol, _e)

                if not _imbalance_ok:
                    continue

                signals.append(Signal(
                    symbol=symbol,
                    strategy="MICRO_MOM",
                    side=side,
                    entry_price=round(price, 2),
                    take_profit=round(take_profit, 2),
                    stop_loss=round(stop_loss, 2),
                    reason=f"Micro {self._event_direction} beta={beta:.1f}",
                    hold_type="day",
                ))

                self._triggered_symbols.add(symbol)
                self._trades_this_event += 1

            except Exception as e:
                logger.debug(f"Micro scan error for {symbol}: {e}")

        return signals

    def check_exits(self, open_trades: dict, now: datetime) -> list[dict]:
        """Check micro-momentum positions for trailing stop and time stop.

        V11.3 T3+T12: ATR-based trailing stop + extended time stop (30 min).
        - Once trade reaches +0.5% profit, trail stop at entry + 0.25%
        - Once trade reaches +1.0% profit, trail stop at highest profit - 0.3%
        - Hard time stop at MICRO_MAX_HOLD_MINUTES (30 min)

        Returns list of exit actions.
        """
        exits = []

        for symbol, trade in open_trades.items():
            if trade.strategy != "MICRO_MOM":
                continue

            try:
                # Get current price
                lookback = now - timedelta(minutes=2)
                bars = get_intraday_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), start=lookback, end=now)
                if bars is not None and not bars.empty:
                    price = float(bars["close"].iloc[-1])
                else:
                    continue

                # Calculate unrealized P&L %
                if trade.side == "buy":
                    pnl_pct = (price - trade.entry_price) / trade.entry_price
                else:
                    pnl_pct = (trade.entry_price - price) / trade.entry_price

                # V11.3 T3: Trailing stop logic
                # Once profitable by 0.5%, move stop to breakeven + 0.25%
                if pnl_pct >= 0.005:
                    if trade.side == "buy":
                        breakeven_trail = trade.entry_price * 1.0025
                        if hasattr(trade, 'stop_loss') and trade.stop_loss < breakeven_trail:
                            trade.stop_loss = breakeven_trail
                            logger.debug(f"MICRO trailing: {symbol} stop moved to breakeven+0.25% = {breakeven_trail:.2f}")
                    else:
                        breakeven_trail = trade.entry_price * 0.9975
                        if hasattr(trade, 'stop_loss') and trade.stop_loss > breakeven_trail:
                            trade.stop_loss = breakeven_trail

                # Once profitable by 1.0%, use tighter trailing stop (0.3% from current)
                if pnl_pct >= 0.01:
                    if trade.side == "buy":
                        tight_trail = price * 0.997
                        if hasattr(trade, 'stop_loss') and trade.stop_loss < tight_trail:
                            trade.stop_loss = tight_trail
                            logger.debug(f"MICRO tight trail: {symbol} stop at {tight_trail:.2f} (price={price:.2f})")
                    else:
                        tight_trail = price * 1.003
                        if hasattr(trade, 'stop_loss') and trade.stop_loss > tight_trail:
                            trade.stop_loss = tight_trail

                # Check if current price hit trailing stop
                if hasattr(trade, 'stop_loss') and trade.stop_loss:
                    if trade.side == "buy" and price <= trade.stop_loss:
                        exits.append({
                            "symbol": symbol, "action": "full",
                            "reason": f"Micro trailing stop (price={price:.2f} <= stop={trade.stop_loss:.2f})",
                        })
                        continue
                    elif trade.side == "sell" and price >= trade.stop_loss:
                        exits.append({
                            "symbol": symbol, "action": "full",
                            "reason": f"Micro trailing stop (price={price:.2f} >= stop={trade.stop_loss:.2f})",
                        })
                        continue

            except Exception as e:
                logger.debug(f"Micro exit check error for {symbol}: {e}")

            # Time stop (V11.3: extended from 8 to 30 min via config)
            if hasattr(trade, 'entry_time') and trade.entry_time:
                elapsed_min = (now - trade.entry_time).total_seconds() / 60
                if elapsed_min >= config.MICRO_MAX_HOLD_MINUTES:
                    exits.append({
                        "symbol": symbol,
                        "action": "full",
                        "reason": f"Micro time stop ({elapsed_min:.0f}min)",
                    })

        return exits
