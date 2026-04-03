"""Tests for the StatMeanReversion strategy.

All external data calls (get_daily_bars, get_intraday_bars) are mocked.
OU/Hurst analytics are mocked to control filtering and signal logic.
"""

import sys
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_bars(n: int = 25, base_price: float = 100.0,
                     noise: float = 0.5) -> pd.DataFrame:
    """Create a DataFrame of daily bars with OHLCV columns."""
    rng = np.random.RandomState(42)
    dates = pd.date_range(end="2026-03-12", periods=n, freq="B", tz=ET)
    close = base_price + rng.randn(n).cumsum() * noise
    return pd.DataFrame({
        "open": close - rng.rand(n) * 0.5,
        "high": close + rng.rand(n) * 1.0,
        "low": close - rng.rand(n) * 1.0,
        "close": close,
        "volume": (rng.rand(n) * 1_000_000 + 500_000).astype(int),
    }, index=dates)


def _make_intraday_bars(n: int = 30, base_price: float = 100.0,
                        noise: float = 0.3) -> pd.DataFrame:
    """Create a DataFrame of 2-min intraday bars."""
    rng = np.random.RandomState(99)
    now = datetime(2026, 3, 13, 11, 0, tzinfo=ET)
    dates = [now - timedelta(minutes=2 * (n - i)) for i in range(n)]
    close = base_price + rng.randn(n).cumsum() * noise
    high = close + rng.rand(n) * 0.5
    low = close - rng.rand(n) * 0.5
    return pd.DataFrame({
        "open": close - rng.rand(n) * 0.2,
        "high": high,
        "low": low,
        "close": close,
        "volume": (rng.rand(n) * 100_000 + 10_000).astype(int),
    }, index=pd.DatetimeIndex(dates))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def strategy():
    from strategies.stat_mean_reversion import StatMeanReversion
    return StatMeanReversion()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPrepareUniverse:

    @patch("strategies.stat_mean_reversion.get_daily_bars")
    @patch("strategies.stat_mean_reversion.fit_ou_params")
    @patch("strategies.stat_mean_reversion.hurst_exponent")
    def test_prepare_universe_filters_correctly(
        self, mock_hurst, mock_ou, mock_bars, strategy
    ):
        """Symbols with low Hurst + valid OU pass; trending symbols are rejected."""
        bars_mr = _make_daily_bars(25, base_price=100.0)
        bars_trend = _make_daily_bars(25, base_price=200.0)

        mock_bars.side_effect = lambda sym, **kw: (
            bars_mr if sym in ("AAPL", "MSFT") else bars_trend
        )

        # AAPL and MSFT are mean-reverting; TSLA is trending
        def hurst_side(prices):
            # Use the base price to distinguish
            if prices.iloc[0] < 150:
                return 0.40  # mean-reverting
            return 0.65  # trending

        mock_hurst.side_effect = hurst_side

        def ou_side(prices):
            if prices.iloc[0] < 150:
                return {
                    'kappa': 0.15,
                    'mu': 100.0,
                    'sigma': 1.5,
                    'half_life': 0.5,  # 0.5 days = 12 hours
                }
            return {}  # trending -> OU fit fails

        mock_ou.side_effect = ou_side

        now = datetime(2026, 3, 13, 9, 0, tzinfo=ET)
        result = strategy.prepare_universe(["AAPL", "MSFT", "TSLA"], now)

        assert "AAPL" in result
        assert "MSFT" in result
        assert "TSLA" not in result
        assert strategy._universe_ready is True
        assert len(strategy.ou_params) == 2

    @patch("strategies.stat_mean_reversion.get_daily_bars")
    @patch("strategies.stat_mean_reversion.fit_ou_params")
    @patch("strategies.stat_mean_reversion.hurst_exponent")
    def test_prepare_universe_rejects_bad_halflife(
        self, mock_hurst, mock_ou, mock_bars, strategy
    ):
        """Symbols with half-life outside 1-48 hours are rejected."""
        mock_bars.return_value = _make_daily_bars(25)
        mock_hurst.return_value = 0.40

        # half_life = 5 days = 120 hours -> exceeds MR_HALFLIFE_MAX_HOURS (48)
        mock_ou.return_value = {
            'kappa': 0.01,
            'mu': 100.0,
            'sigma': 1.5,
            'half_life': 5.0,
        }

        now = datetime(2026, 3, 13, 9, 0, tzinfo=ET)
        result = strategy.prepare_universe(["AAPL"], now)

        assert result == []
        assert strategy._universe_ready is True


class TestScan:

    @patch("strategies.stat_mean_reversion.get_intraday_bars")
    @patch("strategies.stat_mean_reversion.fit_ou_params")
    @patch("strategies.stat_mean_reversion.compute_zscore")
    def test_scan_generates_long_signal(
        self, mock_zscore, mock_ou, mock_bars, strategy
    ):
        """Negative z-score with low RSI and price < VWAP triggers a buy signal."""
        # Set up universe manually
        strategy.universe = ["AAPL"]
        strategy.ou_params = {
            "AAPL": {'kappa': 0.15, 'mu': 100.0, 'sigma': 1.5, 'half_life': 0.5}
        }
        strategy._universe_ready = True

        # Create bars where price ends at 95.2 (= mu - 1.6*sigma with mu=100, sigma=3)
        # This gives: stop at z=-2.5 → 92.5 (below price), target at z=0.2 → 100.6
        # R:R = (100.6-95.2)/(95.2-92.5) = 5.4/2.7 = 2.0 ✓
        bars = _make_intraday_bars(30, base_price=100.0)
        # Force the last 10 closes to drop to 95.2 (oversold RSI + below VWAP)
        bars.iloc[-10:, bars.columns.get_loc("close")] = np.linspace(100, 95.2, 10)
        bars.iloc[-10:, bars.columns.get_loc("low")] = np.linspace(99.5, 94.7, 10)
        bars.iloc[-10:, bars.columns.get_loc("high")] = np.linspace(100.5, 95.7, 10)

        mock_bars.return_value = bars

        # Intraday OU fit → wide sigma for valid R:R
        mock_ou.return_value = {
            'kappa': 0.2,
            'mu': 100.0,
            'sigma': 3.0,
            'half_life': 0.3,
        }

        # z-score = -1.6 (passes the -1.5 entry threshold)
        mock_zscore.return_value = -1.6

        now = datetime(2026, 3, 13, 11, 0, tzinfo=ET)
        signals = strategy.scan(now, regime="NEUTRAL")

        assert len(signals) == 1
        sig = signals[0]
        assert sig.symbol == "AAPL"
        assert sig.strategy == "STAT_MR"
        assert sig.side == "buy"
        assert "MR long" in sig.reason

    @patch("strategies.stat_mean_reversion.get_intraday_bars")
    @patch("strategies.stat_mean_reversion.fit_ou_params")
    @patch("strategies.stat_mean_reversion.compute_zscore")
    def test_scan_no_signal_when_zscore_small(
        self, mock_zscore, mock_ou, mock_bars, strategy
    ):
        """No signal when z-score is near zero (price near mean)."""
        strategy.universe = ["AAPL"]
        strategy.ou_params = {
            "AAPL": {'kappa': 0.15, 'mu': 100.0, 'sigma': 1.5, 'half_life': 0.5}
        }
        strategy._universe_ready = True

        mock_bars.return_value = _make_intraday_bars(30, base_price=100.0)
        mock_ou.return_value = {
            'kappa': 0.2, 'mu': 100.0, 'sigma': 1.5, 'half_life': 0.3,
        }
        mock_zscore.return_value = 0.3  # Near mean, no signal

        now = datetime(2026, 3, 13, 11, 0, tzinfo=ET)
        signals = strategy.scan(now, regime="NEUTRAL")

        assert len(signals) == 0

    def test_scan_empty_when_universe_not_ready(self, strategy):
        """No signals before prepare_universe is called."""
        now = datetime(2026, 3, 13, 11, 0, tzinfo=ET)
        signals = strategy.scan(now)
        assert signals == []


class TestCheckExits:

    @patch("strategies.stat_mean_reversion.get_intraday_bars")
    @patch("strategies.stat_mean_reversion.compute_zscore")
    def test_check_exits_full_reversion(
        self, mock_zscore, mock_bars, strategy
    ):
        """V11.4: z=0.1 (slightly above mean for long) only moves stop, no exit.
        Full reversion exit triggers at z in [-0.2, 0] for longs."""
        strategy.ou_params = {
            "AAPL": {'kappa': 0.15, 'mu': 100.0, 'sigma': 1.5, 'half_life': 0.5}
        }

        bars = _make_intraday_bars(5, base_price=100.0)
        mock_bars.return_value = bars
        # V11.4: Use z=-0.1 (negative side of mean for long) to trigger partial exit
        mock_zscore.return_value = -0.1

        trade = SimpleNamespace(
            strategy="STAT_MR", side="buy",
            entry_time=datetime(2026, 3, 13, 10, 0, tzinfo=ET),
            stop_loss=99.0, entry_price=100.0,
        )

        now = datetime(2026, 3, 13, 10, 30, tzinfo=ET)
        exits = strategy.check_exits({"AAPL": trade}, now)

        assert len(exits) == 1
        assert exits[0]["symbol"] == "AAPL"
        assert exits[0]["action"] == "partial"
        assert "reverted" in exits[0]["reason"].lower() or "MR" in exits[0]["reason"]

    @patch("strategies.stat_mean_reversion.get_intraday_bars")
    @patch("strategies.stat_mean_reversion.compute_zscore")
    def test_check_exits_partial(
        self, mock_zscore, mock_bars, strategy
    ):
        """Partial exit when z-score partially reverts (between exit_partial and exit_full)."""
        strategy.ou_params = {
            "AAPL": {'kappa': 0.15, 'mu': 100.0, 'sigma': 1.5, 'half_life': 0.5}
        }

        bars = _make_intraday_bars(5, base_price=100.0)
        mock_bars.return_value = bars
        # V10: For a long trade, partial exit triggers when z crosses ABOVE
        # +MR_ZSCORE_EXIT_PARTIAL (0.5), meaning price overshot above the mean.
        mock_zscore.return_value = 0.6

        trade = SimpleNamespace(
            strategy="STAT_MR", side="buy",
            entry_time=datetime(2026, 3, 13, 10, 0, tzinfo=ET),
        )

        now = datetime(2026, 3, 13, 10, 30, tzinfo=ET)
        exits = strategy.check_exits({"AAPL": trade}, now)

        assert len(exits) == 1
        assert exits[0]["action"] == "partial"
        assert "partial" in exits[0]["reason"]

    @patch("strategies.stat_mean_reversion.get_intraday_bars")
    def test_check_exits_stop(
        self, mock_bars, strategy
    ):
        """Stop exit when price diverges far from mean (z-score > MR_ZSCORE_STOP).

        V12 AUDIT: check_exits now computes z-score using price_sigma
        (std of price levels) consistently with scan(). The mock bars must
        produce a z-score beyond the stop threshold (2.5).
        """
        # Set up OU params with mu=100.0
        strategy.ou_params = {
            "AAPL": {'kappa': 0.15, 'mu': 100.0, 'sigma': 1.5, 'half_life': 0.5}
        }

        # Create bars where price has drifted far below mean
        # Price std ~1.0, so z-score = (price - mu) / price_sigma
        # For z < -2.5 (stop threshold): need price < 100 - 2.5 * price_sigma
        # With bars around 95.0 and std ~1.0: z = (95 - 100) / 1.0 = -5.0
        bars = _make_intraday_bars(5, base_price=95.0)
        mock_bars.return_value = bars

        trade = SimpleNamespace(
            strategy="STAT_MR", side="buy",
            entry_time=datetime(2026, 3, 13, 10, 0, tzinfo=ET),
            stop_loss=96.0,
        )

        now = datetime(2026, 3, 13, 10, 30, tzinfo=ET)
        exits = strategy.check_exits({"AAPL": trade}, now)

        assert len(exits) == 1
        assert exits[0]["action"] == "full"
        assert "stop" in exits[0]["reason"]

    @patch("strategies.stat_mean_reversion.get_intraday_bars")
    @patch("strategies.stat_mean_reversion.compute_zscore")
    def test_check_exits_short_stop(
        self, mock_zscore, mock_bars, strategy
    ):
        """Stop exit for a short trade when z-score goes further positive."""
        strategy.ou_params = {
            "MSFT": {'kappa': 0.15, 'mu': 400.0, 'sigma': 3.0, 'half_life': 0.5}
        }

        bars = _make_intraday_bars(5, base_price=400.0)
        mock_bars.return_value = bars
        mock_zscore.return_value = 3.0  # Far above mean -> stop for short

        trade = SimpleNamespace(
            strategy="STAT_MR", side="sell",
            entry_time=datetime(2026, 3, 13, 10, 0, tzinfo=ET),
            stop_loss=399.0,  # Price ~400 will be >= 399 → triggers stop
        )

        now = datetime(2026, 3, 13, 10, 30, tzinfo=ET)
        exits = strategy.check_exits({"MSFT": trade}, now)

        assert len(exits) == 1
        assert exits[0]["action"] == "full"
        assert "stop" in exits[0]["reason"]

    @patch("strategies.stat_mean_reversion.get_intraday_bars")
    @patch("strategies.stat_mean_reversion.compute_zscore")
    def test_check_exits_time_stop(
        self, mock_zscore, mock_bars, strategy
    ):
        """Time stop when trade exceeds max hold duration (capped at 240 min)."""
        # V10: half_life * 2 * 390 = time stop in minutes, capped at 240
        # Use half_life=0.1 so time stop = 0.1 * 2 * 390 = 78 min (within 30-240 clamp)
        strategy.ou_params = {
            "AAPL": {'kappa': 0.15, 'mu': 100.0, 'sigma': 1.5, 'half_life': 0.1}
        }

        bars = _make_intraday_bars(5, base_price=100.0)
        mock_bars.return_value = bars
        mock_zscore.return_value = -1.0  # Still displaced but not at stop

        trade = SimpleNamespace(
            strategy="STAT_MR", side="buy",
            entry_time=datetime(2026, 3, 13, 9, 0, tzinfo=ET),  # ~2 hours ago
        )

        now = datetime(2026, 3, 13, 11, 0, tzinfo=ET)
        exits = strategy.check_exits({"AAPL": trade}, now)

        assert len(exits) == 1
        assert exits[0]["action"] == "full"
        assert "time stop" in exits[0]["reason"]

    def test_check_exits_skips_non_mr_trades(self, strategy):
        """Trades from other strategies are ignored."""
        strategy.ou_params = {"AAPL": {'kappa': 0.15, 'mu': 100.0, 'sigma': 1.5, 'half_life': 0.5}}

        trade = SimpleNamespace(strategy="ORB", side="buy", entry_time=None)
        now = datetime(2026, 3, 13, 11, 0, tzinfo=ET)
        exits = strategy.check_exits({"AAPL": trade}, now)

        assert exits == []


class TestResetDaily:

    def test_reset_daily(self, strategy):
        """Verify all state is cleared on reset."""
        strategy.universe = ["AAPL", "MSFT"]
        strategy.ou_params = {"AAPL": {"kappa": 0.1}}
        strategy._universe_ready = True
        strategy._last_scan = datetime.now()

        strategy.reset_daily()

        assert strategy.universe == []
        assert strategy.ou_params == {}
        assert strategy._universe_ready is False
        assert strategy._last_scan is None


class TestHelpers:

    def test_compute_rsi(self, strategy):
        """RSI returns a value between 0 and 100."""
        close = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 96, 95])
        rsi = strategy._compute_rsi(close, period=7)
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_compute_rsi_insufficient_data(self, strategy):
        """RSI returns None with too few bars."""
        close = pd.Series([100, 101, 102])
        rsi = strategy._compute_rsi(close, period=7)
        assert rsi is None

    def test_compute_vwap(self, strategy):
        """VWAP returns a reasonable price (now via analytics.indicators)."""
        from analytics.indicators import compute_vwap
        bars = _make_intraday_bars(10, base_price=100.0)
        vwap = compute_vwap(bars)
        assert vwap is not None
        assert 90 < vwap < 110

    def test_compute_vwap_empty(self, strategy):
        """VWAP returns None for empty bars (now via analytics.indicators)."""
        from analytics.indicators import compute_vwap
        bars = pd.DataFrame()
        assert compute_vwap(bars) is None
