"""Tests for KalmanPairsTrader strategy."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(prices: list[float], col: str = "close") -> pd.DataFrame:
    """Build a minimal DataFrame that looks like get_daily_bars / get_intraday_bars output."""
    n = len(prices)
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1_000_000] * n,
    }, index=idx)


def _cointegrated_series(n: int = 60, hedge_ratio: float = 1.2, noise_std: float = 0.5):
    """Return two price series that are cointegrated."""
    np.random.seed(42)
    # Random walk for series 2
    s2 = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    # Series 1 = hedge_ratio * s2 + stationary noise
    stationary = np.cumsum(np.random.randn(n) * noise_std)
    # Make stationary mean-reverting by adding pull-back
    for i in range(1, n):
        stationary[i] = 0.8 * stationary[i - 1] + np.random.randn() * noise_std
    s1 = hedge_ratio * s2 + 10.0 + stationary
    return s1.tolist(), s2.tolist()


def _uncorrelated_series(n: int = 60):
    """Return two price series that are NOT correlated."""
    np.random.seed(99)
    s1 = 100.0 + np.cumsum(np.random.randn(n) * 2.0)
    s2 = 50.0 + np.cumsum(np.random.randn(n) * 0.3)
    return s1.tolist(), s2.tolist()


def _make_trader_with_kalman_state(sym1="AAPL", sym2="MSFT",
                                    hedge_ratio=1.0, spread_mean=0.0,
                                    spread_std=1.0):
    """Create a KalmanPairsTrader with pre-loaded Kalman state and active pair."""
    from strategies.kalman_pairs import KalmanPairsTrader
    trader = KalmanPairsTrader()
    pair_key = f"{sym1}_{sym2}"
    trader.kalman_state[pair_key] = {
        'theta': np.array([hedge_ratio, 0.0]),
        'P': np.eye(2) * 1.0,
        'spread_mean': spread_mean,
        'spread_std': spread_std,
    }
    trader.active_pairs = [{
        'symbol1': sym1,
        'symbol2': sym2,
        'hedge_ratio': hedge_ratio,
        'correlation': 0.95,
        'spread_mean': spread_mean,
        'spread_std': spread_std,
        'coint_pvalue': 0.01,
        'sector_group': 'mega_tech',
    }]
    trader._pairs_ready = True
    return trader


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKalmanUpdate:
    """Test _update_kalman returns z-score and evolves state."""

    def test_kalman_update_returns_zscore(self):
        trader = _make_trader_with_kalman_state(
            spread_mean=0.0, spread_std=1.0, hedge_ratio=1.0,
        )
        zscore = trader._update_kalman("AAPL_MSFT", 150.0, 150.0)
        assert isinstance(zscore, float)

    def test_kalman_state_persists(self):
        trader = _make_trader_with_kalman_state(hedge_ratio=1.0)
        theta_before = trader.kalman_state["AAPL_MSFT"]["theta"].copy()
        P_before = trader.kalman_state["AAPL_MSFT"]["P"].copy()

        # Feed several observations where price1 drifts above price2
        for i in range(10):
            trader._update_kalman("AAPL_MSFT", 155.0 + i, 150.0)

        theta_after = trader.kalman_state["AAPL_MSFT"]["theta"]
        P_after = trader.kalman_state["AAPL_MSFT"]["P"]

        # theta should have evolved
        assert not np.allclose(theta_before, theta_after), \
            "theta should evolve after multiple updates"
        # P may have been regularized back to identity by V12 2.7 if
        # condition number exceeds 1e6, which is correct behavior.
        # Just verify it is a valid 2x2 positive-definite matrix.
        assert P_after.shape == (2, 2), "P should remain 2x2"
        assert np.all(np.linalg.eigvalsh(P_after) > 0), \
            "P should remain positive-definite after updates"

    def test_kalman_update_missing_state_returns_zero(self):
        from strategies.kalman_pairs import KalmanPairsTrader
        trader = KalmanPairsTrader()
        zscore = trader._update_kalman("NONEXIST_PAIR", 100.0, 100.0)
        assert zscore == 0.0


class TestTestPair:
    """Test _test_pair cointegration logic."""

    @patch("strategies.kalman_pairs.get_daily_bars")
    def test_test_pair_rejects_uncorrelated(self, mock_bars):
        from strategies.kalman_pairs import KalmanPairsTrader
        s1, s2 = _uncorrelated_series(60)
        mock_bars.side_effect = lambda sym, days=60: (
            _make_bars(s1) if sym == "AAA" else _make_bars(s2)
        )

        trader = KalmanPairsTrader()
        result = trader._test_pair("AAA", "BBB", "test_group")
        assert result is None, "Uncorrelated series should be rejected"

    @patch("strategies.kalman_pairs.get_daily_bars")
    def test_test_pair_accepts_cointegrated(self, mock_bars):
        from strategies.kalman_pairs import KalmanPairsTrader
        s1, s2 = _cointegrated_series(60, hedge_ratio=1.2)
        mock_bars.side_effect = lambda sym, days=60: (
            _make_bars(s1) if sym == "AAA" else _make_bars(s2)
        )

        trader = KalmanPairsTrader()
        result = trader._test_pair("AAA", "BBB", "test_group")
        assert result is not None, "Cointegrated series should be accepted"
        assert "hedge_ratio" in result
        assert "coint_pvalue" in result
        assert result["coint_pvalue"] <= 0.05

    @patch("strategies.kalman_pairs.get_daily_bars")
    def test_test_pair_returns_none_for_insufficient_data(self, mock_bars):
        from strategies.kalman_pairs import KalmanPairsTrader
        mock_bars.return_value = _make_bars([100.0] * 10)  # Too few bars

        trader = KalmanPairsTrader()
        result = trader._test_pair("AAA", "BBB")
        assert result is None


class TestScan:
    """Test scan() signal generation."""

    @patch("strategies.kalman_pairs.get_intraday_bars")
    def test_scan_generates_pair_signals(self, mock_intraday):
        """When spread z-score exceeds entry threshold, two linked signals are generated."""
        trader = _make_trader_with_kalman_state(
            sym1="AAPL", sym2="MSFT",
            hedge_ratio=1.0, spread_mean=0.0, spread_std=0.5,
        )

        # Make price1 much higher than hedge_ratio * price2 to produce large positive z
        mock_intraday.side_effect = lambda sym, tf, start, end: (
            _make_bars([200.0]) if sym == "AAPL" else _make_bars([150.0])
        )

        now = datetime(2026, 3, 13, 11, 0, tzinfo=ET)
        with patch.object(type(trader), '_update_kalman', return_value=2.5):
            with patch("strategies.kalman_pairs.config") as mock_cfg:
                mock_cfg.PAIRS_ZSCORE_ENTRY = 2.0
                mock_cfg.PAIRS_TP_PCT = 0.015
                mock_cfg.PAIRS_SL_PCT = 0.010
                mock_cfg.ALLOW_SHORT = True
                mock_cfg.NO_SHORT_SYMBOLS = set()
                signals = trader.scan(now)

        assert len(signals) == 2, f"Expected 2 signals (pair legs), got {len(signals)}"
        # Both should share the same pair_id
        pair_ids = {s.pair_id for s in signals}
        assert len(pair_ids) == 1, "Both legs must share the same pair_id"
        # One buy, one sell
        sides = {s.side for s in signals}
        assert sides == {"buy", "sell"}, "Pair should have one buy and one sell leg"

    def test_scan_returns_empty_when_not_ready(self):
        from strategies.kalman_pairs import KalmanPairsTrader
        trader = KalmanPairsTrader()
        now = datetime(2026, 3, 13, 11, 0, tzinfo=ET)
        signals = trader.scan(now)
        assert signals == []


class TestCheckExits:
    """Test check_exits() for convergence and divergence stops."""

    def _make_mock_trade(self, symbol, pair_id, entry_time=None):
        trade = SimpleNamespace(
            symbol=symbol,
            strategy="KALMAN_PAIRS",
            pair_id=pair_id,
            entry_time=entry_time or datetime(2026, 3, 13, 10, 0, tzinfo=ET),
        )
        return trade

    @patch("strategies.kalman_pairs.get_intraday_bars")
    def test_check_exits_convergence(self, mock_intraday):
        """When z-score drops below PAIRS_ZSCORE_EXIT, exit is triggered."""
        trader = _make_trader_with_kalman_state(
            sym1="AAPL", sym2="MSFT",
            hedge_ratio=1.0, spread_mean=0.0, spread_std=1.0,
        )

        mock_intraday.side_effect = lambda sym, tf, start, end: _make_bars([150.0])

        trade = self._make_mock_trade("AAPL", "PAIR_AAPL_MSFT_1100")
        open_trades = {"AAPL": trade}

        now = datetime(2026, 3, 13, 14, 0, tzinfo=ET)
        with patch.object(type(trader), '_update_kalman', return_value=0.1):
            exits = trader.check_exits(open_trades, now)

        assert len(exits) == 1
        assert "converged" in exits[0]["reason"]
        assert exits[0]["pair_id"] == "PAIR_AAPL_MSFT_1100"

    @patch("strategies.kalman_pairs.get_intraday_bars")
    def test_check_exits_divergence_stop(self, mock_intraday):
        """When z-score exceeds PAIRS_ZSCORE_STOP, divergence stop is triggered."""
        trader = _make_trader_with_kalman_state(
            sym1="AAPL", sym2="MSFT",
            hedge_ratio=1.0, spread_mean=0.0, spread_std=1.0,
        )

        mock_intraday.side_effect = lambda sym, tf, start, end: _make_bars([150.0])

        trade = self._make_mock_trade("AAPL", "PAIR_AAPL_MSFT_1100")
        open_trades = {"AAPL": trade}

        now = datetime(2026, 3, 13, 14, 0, tzinfo=ET)
        with patch.object(type(trader), '_update_kalman', return_value=3.5):
            exits = trader.check_exits(open_trades, now)

        assert len(exits) == 1
        assert "diverged" in exits[0]["reason"]

    @patch("strategies.kalman_pairs.get_intraday_bars")
    def test_check_exits_max_hold(self, mock_intraday):
        """Trades held beyond PAIRS_MAX_HOLD_DAYS are exited."""
        trader = _make_trader_with_kalman_state(
            sym1="AAPL", sym2="MSFT",
            hedge_ratio=1.0, spread_mean=0.0, spread_std=1.0,
        )

        mock_intraday.side_effect = lambda sym, tf, start, end: _make_bars([150.0])

        entry_time = datetime(2026, 3, 1, 10, 0, tzinfo=ET)  # 12 days ago
        trade = self._make_mock_trade("AAPL", "PAIR_AAPL_MSFT_1100", entry_time=entry_time)
        open_trades = {"AAPL": trade}

        now = datetime(2026, 3, 13, 14, 0, tzinfo=ET)
        # z-score doesn't matter — max hold triggers first
        with patch.object(type(trader), '_update_kalman', return_value=1.0):
            exits = trader.check_exits(open_trades, now)

        assert len(exits) == 1
        assert "max hold" in exits[0]["reason"]

    def test_check_exits_skips_non_kalman_trades(self):
        from strategies.kalman_pairs import KalmanPairsTrader
        trader = KalmanPairsTrader()
        trade = SimpleNamespace(
            symbol="AAPL", strategy="ORB", pair_id="", entry_time=None,
        )
        exits = trader.check_exits({"AAPL": trade}, datetime.now(ET))
        assert exits == []


class TestResetDaily:
    """Test reset_daily preserves pairs."""

    def test_reset_daily_keeps_pairs(self):
        trader = _make_trader_with_kalman_state()
        assert len(trader.active_pairs) == 1
        assert len(trader.kalman_state) == 1

        trader.reset_daily()

        assert len(trader.active_pairs) == 1, "reset_daily must NOT clear active_pairs"
        assert len(trader.kalman_state) == 1, "reset_daily must NOT clear kalman_state"
        assert trader._pairs_ready is True


class TestSelectPairsWeekly:
    """Test weekly pair selection end-to-end."""

    @patch("strategies.kalman_pairs.save_kalman_pair")
    @patch("strategies.kalman_pairs.deactivate_all_kalman_pairs")
    @patch("strategies.kalman_pairs.get_daily_bars")
    def test_select_pairs_weekly_finds_cointegrated(self, mock_bars, mock_deact, mock_save):
        from strategies.kalman_pairs import KalmanPairsTrader

        s1, s2 = _cointegrated_series(60, hedge_ratio=1.2)
        mock_bars.side_effect = lambda sym, days=60: (
            _make_bars(s1) if sym in config_sector_first() else _make_bars(s2)
        )

        # Patch SECTOR_GROUPS to have one small group
        import config as cfg
        original = cfg.SECTOR_GROUPS
        cfg.SECTOR_GROUPS = {'test': ['AAA', 'BBB']}
        try:
            mock_bars.side_effect = lambda sym, days=60: (
                _make_bars(s1) if sym == "AAA" else _make_bars(s2)
            )
            trader = KalmanPairsTrader()
            now = datetime(2026, 3, 9, 18, 0, tzinfo=ET)
            pairs = trader.select_pairs_weekly(now)
        finally:
            cfg.SECTOR_GROUPS = original

        assert len(pairs) >= 1
        assert mock_deact.called
        assert mock_save.called
        assert trader._pairs_ready is True


def config_sector_first():
    """Helper — not really used, just a placeholder for the lambda."""
    return set()
