"""ML-001: Feature Engineering Framework.

Computes 200+ features across categories: price-based, volume-based,
cross-asset, calendar, and sentiment.  All features are returned as a
flat Dict[str, float] with naming convention ``{category}_{name}_{param}``.

Dependencies (numpy, pandas) are always available in this project.
Optional cross-asset data is passed via *market_data* dict.
"""

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(val: Any) -> float:
    """Coerce to float, mapping NaN/Inf/None to 0.0."""
    if val is None:
        return 0.0
    f = float(val)
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


def _returns(prices: pd.Series, period: int) -> float:
    """Log return over *period* bars."""
    if len(prices) < period + 1:
        return 0.0
    return _safe(np.log(prices.iloc[-1] / prices.iloc[-1 - period]))


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, min_periods=max(1, span // 2)).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=window, min_periods=max(1, window // 2)).mean()


def _rsi(prices: pd.Series, period: int = 14) -> float:
    """Compute RSI from price series."""
    if len(prices) < period + 1:
        return 50.0
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(span=period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, min_periods=period).mean()
    if loss.iloc[-1] == 0:
        return 100.0
    rs = gain.iloc[-1] / loss.iloc[-1]
    return _safe(100.0 - 100.0 / (1.0 + rs))


def _atr(highs: pd.Series, lows: pd.Series, closes: pd.Series,
         period: int = 14) -> float:
    """Average True Range."""
    if len(closes) < period + 1:
        return 0.0
    high = highs.values[-period:]
    low = lows.values[-period:]
    prev_close = closes.values[-period - 1:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close),
                                            np.abs(low - prev_close)))
    return _safe(np.mean(tr))


# ---------------------------------------------------------------------------
# Feature Engine
# ---------------------------------------------------------------------------

class FeatureEngine:
    """Compute 200+ features for a single symbol from OHLCV bars.

    Usage::

        engine = FeatureEngine()
        feats = engine.compute_all_features("AAPL", bars_df, market_data=md)
    """

    # Feature categories and their methods
    _CATEGORIES = [
        "_price_features",
        "_volume_features",
        "_cross_asset_features",
        "_calendar_features",
        "_sentiment_features",
    ]

    def compute_all_features(
        self,
        symbol: str,
        bars: pd.DataFrame,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Compute all features for *symbol*.

        Args:
            symbol: Ticker symbol (e.g. ``"AAPL"``).
            bars: DataFrame with columns ``open, high, low, close, volume``
                  indexed by timestamp.  Must have at least 50 rows for
                  meaningful output.
            market_data: Optional dict with cross-asset data such as
                ``spy_bars``, ``vix_level``, ``vix_change``, ``sector_etf_bars``,
                ``credit_spread``.

        Returns:
            Dict mapping feature name to float value.
        """
        if bars is None or len(bars) < 5:
            logger.warning("Insufficient bars for %s (%d) — returning empty features",
                           symbol, 0 if bars is None else len(bars))
            return {}

        if market_data is None:
            market_data = {}

        features: Dict[str, float] = {}

        for method_name in self._CATEGORIES:
            try:
                method = getattr(self, method_name)
                feats = method(symbol, bars, market_data)
                features.update(feats)
            except Exception:
                logger.exception("Error computing %s for %s", method_name, symbol)

        return features

    # ------------------------------------------------------------------
    # Price-based features (30+)
    # ------------------------------------------------------------------

    def _price_features(
        self, symbol: str, bars: pd.DataFrame, market_data: Dict
    ) -> Dict[str, float]:
        f: Dict[str, float] = {}
        closes = bars["close"]
        highs = bars["high"]
        lows = bars["low"]
        opens = bars["open"]

        # --- Returns at multiple horizons ---
        for period in [1, 2, 3, 5, 10, 20, 60]:
            f[f"price_return_{period}"] = _returns(closes, period)

        # --- Realized volatility (close-to-close) ---
        for window in [5, 10, 20, 60]:
            log_ret = np.log(closes / closes.shift(1)).dropna()
            if len(log_ret) >= window:
                f[f"price_realized_vol_{window}"] = _safe(
                    log_ret.iloc[-window:].std() * np.sqrt(252)
                )
            else:
                f[f"price_realized_vol_{window}"] = 0.0

        # --- Parkinson volatility ---
        for window in [10, 20]:
            if len(highs) >= window:
                hl = np.log(highs.iloc[-window:] / lows.iloc[-window:])
                f[f"price_parkinson_vol_{window}"] = _safe(
                    np.sqrt(hl.pow(2).mean() / (4.0 * np.log(2))) * np.sqrt(252)
                )
            else:
                f[f"price_parkinson_vol_{window}"] = 0.0

        # --- Garman-Klass volatility ---
        for window in [10, 20]:
            if len(highs) >= window:
                h = highs.iloc[-window:].values
                l = lows.iloc[-window:].values
                o = opens.iloc[-window:].values
                c = closes.iloc[-window:].values
                hl2 = 0.5 * np.log(h / l) ** 2
                co2 = -(2.0 * np.log(2) - 1.0) * np.log(c / o) ** 2
                gk = np.mean(hl2 + co2)
                f[f"price_garman_klass_vol_{window}"] = _safe(np.sqrt(max(gk, 0)) * np.sqrt(252))
            else:
                f[f"price_garman_klass_vol_{window}"] = 0.0

        # --- Price vs Moving Averages ---
        for ma_period in [5, 10, 20, 50, 200]:
            ma = _sma(closes, ma_period)
            if len(ma.dropna()) > 0 and ma.iloc[-1] != 0:
                f[f"price_vs_sma_{ma_period}"] = _safe(
                    (closes.iloc[-1] - ma.iloc[-1]) / ma.iloc[-1]
                )
            else:
                f[f"price_vs_sma_{ma_period}"] = 0.0

        # --- EMA ratios ---
        for ma_period in [5, 10, 20, 50]:
            ema = _ema(closes, ma_period)
            if len(ema.dropna()) > 0 and ema.iloc[-1] != 0:
                f[f"price_vs_ema_{ma_period}"] = _safe(
                    (closes.iloc[-1] - ema.iloc[-1]) / ema.iloc[-1]
                )
            else:
                f[f"price_vs_ema_{ma_period}"] = 0.0

        # --- Bollinger Bands ---
        for window in [20]:
            sma = _sma(closes, window)
            std = closes.rolling(window=window, min_periods=max(1, window // 2)).std()
            if len(sma.dropna()) > 0 and std.iloc[-1] > 0:
                upper = sma.iloc[-1] + 2.0 * std.iloc[-1]
                lower = sma.iloc[-1] - 2.0 * std.iloc[-1]
                width = upper - lower
                f[f"price_bb_width_{window}"] = _safe(width / sma.iloc[-1])
                f[f"price_bb_pctb_{window}"] = _safe(
                    (closes.iloc[-1] - lower) / width if width > 0 else 0.5
                )
            else:
                f[f"price_bb_width_{window}"] = 0.0
                f[f"price_bb_pctb_{window}"] = 0.5

        # --- ATR at multiple periods ---
        for period in [5, 10, 14, 20]:
            f[f"price_atr_{period}"] = _atr(highs, lows, closes, period)

        # ATR as % of price
        if closes.iloc[-1] > 0:
            for period in [14]:
                f[f"price_atr_pct_{period}"] = _safe(
                    _atr(highs, lows, closes, period) / closes.iloc[-1]
                )
        else:
            f["price_atr_pct_14"] = 0.0

        # --- Price acceleration (second derivative of returns) ---
        if len(closes) >= 12:
            period = 5
            ret_5 = _returns(closes, period)
            ret_5_prev = _returns(closes.iloc[-(period * 2):-period], period) if len(closes) >= period * 2 + 1 else 0.0
            f["price_acceleration_5"] = _safe(ret_5 - ret_5_prev)
        else:
            f["price_acceleration_5"] = 0.0

        # --- RSI at multiple periods ---
        for period in [7, 14, 21]:
            f[f"price_rsi_{period}"] = _rsi(closes, period)

        # --- MACD ---
        if len(closes) >= 26:
            ema_12 = _ema(closes, 12)
            ema_26 = _ema(closes, 26)
            macd_line = ema_12 - ema_26
            signal_line = _ema(macd_line.dropna(), 9)
            if len(signal_line.dropna()) > 0:
                f["price_macd_line"] = _safe(macd_line.iloc[-1])
                f["price_macd_signal"] = _safe(signal_line.iloc[-1])
                f["price_macd_hist"] = _safe(macd_line.iloc[-1] - signal_line.iloc[-1])
            else:
                f["price_macd_line"] = 0.0
                f["price_macd_signal"] = 0.0
                f["price_macd_hist"] = 0.0
        else:
            f["price_macd_line"] = 0.0
            f["price_macd_signal"] = 0.0
            f["price_macd_hist"] = 0.0

        # --- High/Low position ---
        for window in [20, 60]:
            if len(closes) >= window:
                hi = closes.iloc[-window:].max()
                lo = closes.iloc[-window:].min()
                rng = hi - lo
                f[f"price_position_{window}"] = _safe(
                    (closes.iloc[-1] - lo) / rng if rng > 0 else 0.5
                )
            else:
                f[f"price_position_{window}"] = 0.5

        # --- Intraday range ratio ---
        if len(bars) >= 2:
            today_range = highs.iloc[-1] - lows.iloc[-1]
            avg_range = (highs - lows).iloc[-20:].mean() if len(bars) >= 20 else today_range
            f["price_range_ratio"] = _safe(today_range / avg_range if avg_range > 0 else 1.0)
        else:
            f["price_range_ratio"] = 1.0

        # --- Gap ---
        if len(bars) >= 2:
            f["price_gap"] = _safe(
                (opens.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]
                if closes.iloc[-2] > 0 else 0.0
            )
        else:
            f["price_gap"] = 0.0

        # --- Close vs Open (body direction) ---
        f["price_body_ratio"] = _safe(
            (closes.iloc[-1] - opens.iloc[-1]) / (highs.iloc[-1] - lows.iloc[-1])
            if (highs.iloc[-1] - lows.iloc[-1]) > 0 else 0.0
        )

        # --- Fractionally differenced features (V12) ---
        # fracdiff with d=0.7 balances stationarity vs memory preservation.
        # d=0.4 produced 1458-lag filters (too long for 252-bar daily series).
        # d=0.7 gives ~30-50 lags which fits typical bar windows.
        # Fail-open: skip entirely if fracdiff fails (e.g., series too short).
        try:
            from ml.fracdiff import FractionalDifferentiator
            _fd = FractionalDifferentiator()
            _d = 0.7

            # fracdiff of close prices
            if len(closes) >= 50:
                fd_close = _fd.frac_diff(closes, _d)
                last_val = fd_close.dropna()
                if len(last_val) > 0:
                    f["price_fracdiff_close"] = _safe(last_val.iloc[-1])
                else:
                    f["price_fracdiff_close"] = 0.0
            else:
                f["price_fracdiff_close"] = 0.0

            # fracdiff of volume
            volumes = bars["volume"]
            if len(volumes) >= 50:
                fd_vol = _fd.frac_diff(volumes.astype(float), _d)
                last_val = fd_vol.dropna()
                if len(last_val) > 0:
                    f["price_fracdiff_volume"] = _safe(last_val.iloc[-1])
                else:
                    f["price_fracdiff_volume"] = 0.0
            else:
                f["price_fracdiff_volume"] = 0.0

            # fracdiff of OBV
            if len(closes) >= 50:
                price_diff = closes.diff()
                obv = (np.sign(price_diff) * bars["volume"]).fillna(0).cumsum()
                fd_obv = _fd.frac_diff(obv.astype(float), _d)
                last_val = fd_obv.dropna()
                if len(last_val) > 0:
                    f["price_fracdiff_obv"] = _safe(last_val.iloc[-1])
                else:
                    f["price_fracdiff_obv"] = 0.0
            else:
                f["price_fracdiff_obv"] = 0.0
        except Exception:
            logger.debug("Fractional differencing failed for %s — skipping fracdiff features", symbol)
            f.setdefault("price_fracdiff_close", 0.0)
            f.setdefault("price_fracdiff_volume", 0.0)
            f.setdefault("price_fracdiff_obv", 0.0)

        return f

    # ------------------------------------------------------------------
    # Volume-based features (15+)
    # ------------------------------------------------------------------

    def _volume_features(
        self, symbol: str, bars: pd.DataFrame, market_data: Dict
    ) -> Dict[str, float]:
        f: Dict[str, float] = {}
        closes = bars["close"]
        volumes = bars["volume"]
        highs = bars["high"]
        lows = bars["low"]

        # --- Volume ratio vs averages ---
        for window in [5, 10, 20]:
            avg_vol = volumes.iloc[-window:].mean() if len(volumes) >= window else volumes.mean()
            f[f"volume_ratio_{window}"] = _safe(
                volumes.iloc[-1] / avg_vol if avg_vol > 0 else 1.0
            )

        # --- Volume trend (slope of volume over window) ---
        if len(volumes) >= 10:
            x = np.arange(10, dtype=float)
            y = volumes.iloc[-10:].values.astype(float)
            if np.std(y) > 0:
                slope = np.polyfit(x, y, 1)[0]
                f["volume_trend_10"] = _safe(slope / np.mean(y) if np.mean(y) > 0 else 0.0)
            else:
                f["volume_trend_10"] = 0.0
        else:
            f["volume_trend_10"] = 0.0

        # --- OBV (On-Balance Volume) ---
        if len(closes) >= 2:
            price_diff = closes.diff()
            obv = (np.sign(price_diff) * volumes).fillna(0).cumsum()
            f["volume_obv_current"] = _safe(obv.iloc[-1])

            # OBV rate of change
            if len(obv) >= 10 and obv.iloc[-10] != 0:
                f["volume_obv_roc_10"] = _safe(
                    (obv.iloc[-1] - obv.iloc[-10]) / abs(obv.iloc[-10])
                )
            else:
                f["volume_obv_roc_10"] = 0.0

            # OBV divergence (price up, OBV down or vice versa)
            if len(closes) >= 20:
                price_chg = closes.iloc[-1] - closes.iloc[-20]
                obv_chg = obv.iloc[-1] - obv.iloc[-20]
                # +1 if both agree, -1 if diverge, 0 if ambiguous
                if price_chg > 0 and obv_chg < 0:
                    f["volume_obv_divergence"] = -1.0  # bearish divergence
                elif price_chg < 0 and obv_chg > 0:
                    f["volume_obv_divergence"] = 1.0   # bullish divergence
                else:
                    f["volume_obv_divergence"] = 0.0
            else:
                f["volume_obv_divergence"] = 0.0
        else:
            f["volume_obv_current"] = 0.0
            f["volume_obv_roc_10"] = 0.0
            f["volume_obv_divergence"] = 0.0

        # --- Volume-weighted return ---
        for window in [5, 10, 20]:
            if len(closes) >= window and len(volumes) >= window:
                rets = closes.pct_change().iloc[-window:]
                vols = volumes.iloc[-window:]
                total_vol = vols.sum()
                if total_vol > 0:
                    f[f"volume_weighted_return_{window}"] = _safe(
                        (rets * vols).sum() / total_vol
                    )
                else:
                    f[f"volume_weighted_return_{window}"] = 0.0
            else:
                f[f"volume_weighted_return_{window}"] = 0.0

        # --- Accumulation/Distribution Line ---
        if len(closes) >= 2:
            high_low = highs - lows
            clv = ((closes - lows) - (highs - closes)) / high_low.replace(0, np.nan)
            clv = clv.fillna(0)
            ad = (clv * volumes).cumsum()
            f["volume_ad_line"] = _safe(ad.iloc[-1])
            # AD rate of change
            if len(ad) >= 10:
                f["volume_ad_roc_10"] = _safe(
                    (ad.iloc[-1] - ad.iloc[-10]) / abs(ad.iloc[-10])
                    if ad.iloc[-10] != 0 else 0.0
                )
            else:
                f["volume_ad_roc_10"] = 0.0
        else:
            f["volume_ad_line"] = 0.0
            f["volume_ad_roc_10"] = 0.0

        # --- Money Flow Index ---
        if len(closes) >= 15:
            typical_price = (highs + lows + closes) / 3.0
            money_flow = typical_price * volumes
            tp_diff = typical_price.diff()
            pos_flow = money_flow.where(tp_diff > 0, 0.0)
            neg_flow = money_flow.where(tp_diff < 0, 0.0)
            pos_sum = pos_flow.iloc[-14:].sum()
            neg_sum = neg_flow.iloc[-14:].sum()
            if neg_sum > 0:
                mfr = pos_sum / neg_sum
                f["volume_mfi_14"] = _safe(100.0 - 100.0 / (1.0 + mfr))
            else:
                f["volume_mfi_14"] = 100.0
        else:
            f["volume_mfi_14"] = 50.0

        # --- Volume spike flag ---
        if len(volumes) >= 20:
            vol_mean = volumes.iloc[-20:].mean()
            vol_std = volumes.iloc[-20:].std()
            if vol_std > 0:
                f["volume_zscore"] = _safe((volumes.iloc[-1] - vol_mean) / vol_std)
            else:
                f["volume_zscore"] = 0.0
        else:
            f["volume_zscore"] = 0.0

        # --- Relative volume by time of day (placeholder — needs intraday bucketing) ---
        f["volume_relative_tod"] = _safe(
            volumes.iloc[-1] / volumes.mean() if volumes.mean() > 0 else 1.0
        )

        return f

    # ------------------------------------------------------------------
    # Cross-asset features (15+)
    # ------------------------------------------------------------------

    def _cross_asset_features(
        self, symbol: str, bars: pd.DataFrame, market_data: Dict
    ) -> Dict[str, float]:
        f: Dict[str, float] = {}

        # --- SPY returns ---
        spy_bars = market_data.get("spy_bars")
        if spy_bars is not None and len(spy_bars) >= 2:
            spy_close = spy_bars["close"] if isinstance(spy_bars, pd.DataFrame) else spy_bars
            for period in [1, 5, 20]:
                f[f"cross_spy_return_{period}"] = _returns(spy_close, period)
            # Beta to SPY (rolling)
            if len(bars) >= 20 and len(spy_close) >= 20:
                stock_ret = bars["close"].pct_change().iloc[-20:]
                spy_ret = spy_close.pct_change().iloc[-20:]
                # Align lengths
                min_len = min(len(stock_ret), len(spy_ret))
                stock_ret = stock_ret.iloc[-min_len:]
                spy_ret = spy_ret.iloc[-min_len:]
                spy_var = spy_ret.var()
                if spy_var > 0:
                    f["cross_beta_spy_20"] = _safe(
                        stock_ret.cov(spy_ret) / spy_var
                    )
                else:
                    f["cross_beta_spy_20"] = 1.0
            else:
                f["cross_beta_spy_20"] = 1.0
            # Relative strength vs SPY
            if len(bars["close"]) >= 20 and len(spy_close) >= 20:
                stock_perf = bars["close"].iloc[-1] / bars["close"].iloc[-20] - 1
                spy_perf = spy_close.iloc[-1] / spy_close.iloc[-20] - 1
                f["cross_rel_strength_spy_20"] = _safe(stock_perf - spy_perf)
            else:
                f["cross_rel_strength_spy_20"] = 0.0
        else:
            for period in [1, 5, 20]:
                f[f"cross_spy_return_{period}"] = 0.0
            f["cross_beta_spy_20"] = 1.0
            f["cross_rel_strength_spy_20"] = 0.0

        # --- VIX level and change ---
        vix = market_data.get("vix_level")
        f["cross_vix_level"] = _safe(vix) if vix is not None else 20.0
        vix_chg = market_data.get("vix_change")
        f["cross_vix_change"] = _safe(vix_chg) if vix_chg is not None else 0.0
        # VIX regime bins
        vix_val = f["cross_vix_level"]
        f["cross_vix_regime_low"] = 1.0 if vix_val < 15 else 0.0
        f["cross_vix_regime_med"] = 1.0 if 15 <= vix_val < 25 else 0.0
        f["cross_vix_regime_high"] = 1.0 if 25 <= vix_val < 35 else 0.0
        f["cross_vix_regime_extreme"] = 1.0 if vix_val >= 35 else 0.0

        # --- Sector ETF relative strength ---
        sector_bars = market_data.get("sector_etf_bars")
        if sector_bars is not None and isinstance(sector_bars, pd.DataFrame) and len(sector_bars) >= 2:
            sector_close = sector_bars["close"]
            for period in [1, 5, 20]:
                f[f"cross_sector_return_{period}"] = _returns(sector_close, period)
            if len(bars["close"]) >= 20 and len(sector_close) >= 20:
                stock_perf = bars["close"].iloc[-1] / bars["close"].iloc[-20] - 1
                sector_perf = sector_close.iloc[-1] / sector_close.iloc[-20] - 1
                f["cross_rel_strength_sector_20"] = _safe(stock_perf - sector_perf)
            else:
                f["cross_rel_strength_sector_20"] = 0.0
        else:
            for period in [1, 5, 20]:
                f[f"cross_sector_return_{period}"] = 0.0
            f["cross_rel_strength_sector_20"] = 0.0

        # --- Credit spread ---
        credit_spread = market_data.get("credit_spread")
        f["cross_credit_spread"] = _safe(credit_spread) if credit_spread is not None else 0.0

        # --- Dollar index proxy ---
        uup = market_data.get("uup_change")
        f["cross_dollar_change"] = _safe(uup) if uup is not None else 0.0

        # --- Gold ---
        gld = market_data.get("gld_change")
        f["cross_gold_change"] = _safe(gld) if gld is not None else 0.0

        # --- Treasury (TLT) ---
        tlt = market_data.get("tlt_change")
        f["cross_treasury_change"] = _safe(tlt) if tlt is not None else 0.0

        return f

    # ------------------------------------------------------------------
    # Calendar features (10+)
    # ------------------------------------------------------------------

    def _calendar_features(
        self, symbol: str, bars: pd.DataFrame, market_data: Dict
    ) -> Dict[str, float]:
        f: Dict[str, float] = {}

        # Determine current timestamp from bars index or market_data
        if hasattr(bars.index, 'to_pydatetime') and len(bars) > 0:
            try:
                ts = pd.Timestamp(bars.index[-1])
                now = ts.to_pydatetime()
            except Exception:
                now = datetime.now()
        else:
            now = datetime.now()

        # --- Day of week (Mon=0..Fri=4) one-hot ---
        dow = now.weekday()
        for d in range(5):
            f[f"cal_dow_{d}"] = 1.0 if dow == d else 0.0

        # --- Month, Quarter, Day-of-month REMOVED (V12) ---
        # cal_month, cal_quarter, cal_day_of_month_norm create spurious
        # correlations: the training-period month/quarter/day is not
        # predictive in production.  Kept: day-of-week (Monday effect),
        # minutes-since-open (intraday patterns), and catalyst-proximity
        # features (earnings, FOMC, opex).

        import calendar
        days_in_month = calendar.monthrange(now.year, now.month)[1]

        # --- End of month flag (last 3 trading days) ---
        f["cal_end_of_month"] = 1.0 if now.day >= days_in_month - 3 else 0.0

        # --- End of quarter flag ---
        f["cal_end_of_quarter"] = 1.0 if (
            now.month in (3, 6, 9, 12) and now.day >= days_in_month - 5
        ) else 0.0

        # --- Days to/from earnings (from market_data) ---
        days_to_earnings = market_data.get("days_to_earnings")
        f["cal_days_to_earnings"] = _safe(days_to_earnings) if days_to_earnings is not None else 30.0
        days_from_earnings = market_data.get("days_from_earnings")
        f["cal_days_from_earnings"] = _safe(days_from_earnings) if days_from_earnings is not None else 30.0

        # --- FOMC proximity ---
        days_to_fomc = market_data.get("days_to_fomc")
        f["cal_days_to_fomc"] = _safe(days_to_fomc) if days_to_fomc is not None else 30.0
        f["cal_fomc_week"] = 1.0 if (days_to_fomc is not None and abs(days_to_fomc) <= 3) else 0.0

        # --- Options expiration proximity (3rd Friday) ---
        # Approximate: 3rd Friday is between day 15-21
        third_friday_day = 15 + (4 - calendar.weekday(now.year, now.month, 1)) % 7
        days_to_opex = third_friday_day - now.day
        f["cal_days_to_opex"] = float(days_to_opex)
        f["cal_opex_week"] = 1.0 if abs(days_to_opex) <= 2 else 0.0

        # --- Hour of day (for intraday) ---
        f["cal_hour"] = float(now.hour) if hasattr(now, 'hour') else 12.0

        # --- Minutes since market open (9:30 ET) ---
        mkt_open_min = 9 * 60 + 30
        current_min = now.hour * 60 + now.minute if hasattr(now, 'minute') else mkt_open_min
        f["cal_minutes_since_open"] = float(max(0, current_min - mkt_open_min))

        return f

    # ------------------------------------------------------------------
    # Sentiment features (10+) — placeholder for external data
    # ------------------------------------------------------------------

    def _sentiment_features(
        self, symbol: str, bars: pd.DataFrame, market_data: Dict
    ) -> Dict[str, float]:
        """Sentiment features from news/social data.

        These rely on external data passed via *market_data*.  If not
        available, neutral defaults are returned.
        """
        f: Dict[str, float] = {}

        # --- News sentiment (rolling average) ---
        news_sent = market_data.get("news_sentiment")
        if news_sent is not None and isinstance(news_sent, (list, np.ndarray)) and len(news_sent) > 0:
            arr = np.array(news_sent, dtype=float)
            f["sent_news_avg_1d"] = _safe(np.mean(arr[-1:])) if len(arr) >= 1 else 0.0
            f["sent_news_avg_3d"] = _safe(np.mean(arr[-3:])) if len(arr) >= 3 else _safe(np.mean(arr))
            f["sent_news_avg_7d"] = _safe(np.mean(arr[-7:])) if len(arr) >= 7 else _safe(np.mean(arr))
            # Sentiment velocity (change in avg sentiment)
            if len(arr) >= 4:
                recent = np.mean(arr[-2:])
                prior = np.mean(arr[-4:-2])
                f["sent_news_velocity"] = _safe(recent - prior)
            else:
                f["sent_news_velocity"] = 0.0
            # Sentiment dispersion
            if len(arr) >= 3:
                f["sent_news_dispersion"] = _safe(np.std(arr[-7:]))
            else:
                f["sent_news_dispersion"] = 0.0
        else:
            f["sent_news_avg_1d"] = 0.0
            f["sent_news_avg_3d"] = 0.0
            f["sent_news_avg_7d"] = 0.0
            f["sent_news_velocity"] = 0.0
            f["sent_news_dispersion"] = 0.0

        # --- Social/alternative sentiment score ---
        social = market_data.get("social_sentiment")
        f["sent_social_score"] = _safe(social) if social is not None else 0.0

        # --- Analyst consensus ---
        analyst = market_data.get("analyst_consensus")
        f["sent_analyst_consensus"] = _safe(analyst) if analyst is not None else 0.0

        # --- Short interest ratio ---
        short_interest = market_data.get("short_interest_ratio")
        f["sent_short_interest"] = _safe(short_interest) if short_interest is not None else 0.0

        # --- Put/Call ratio ---
        pc_ratio = market_data.get("put_call_ratio")
        f["sent_put_call_ratio"] = _safe(pc_ratio) if pc_ratio is not None else 1.0

        # --- Institutional flow ---
        inst_flow = market_data.get("institutional_flow")
        f["sent_institutional_flow"] = _safe(inst_flow) if inst_flow is not None else 0.0

        return f

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_feature_names(self) -> List[str]:
        """Return a sorted list of all feature names produced by this engine.

        Useful for ensuring feature alignment across train/predict.
        Generates names from a minimal synthetic dataset.
        """
        dummy_bars = pd.DataFrame({
            "open": np.random.uniform(99, 101, 250),
            "high": np.random.uniform(100, 102, 250),
            "low": np.random.uniform(98, 100, 250),
            "close": np.random.uniform(99, 101, 250),
            "volume": np.random.randint(1000, 10000, 250),
        }, index=pd.date_range("2024-01-01", periods=250, freq="h"))

        feats = self.compute_all_features("DUMMY", dummy_bars)
        return sorted(feats.keys())
