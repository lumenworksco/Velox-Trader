"""Tests for V8 Kelly Criterion position sizing."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


class TestKellyEngine:
    """Tests for KellyEngine."""

    def test_get_fraction_disabled(self, override_config):
        """When KELLY_ENABLED=False, always returns flat rate."""
        from risk.kelly import KellyEngine
        import config
        engine = KellyEngine()
        with override_config(KELLY_ENABLED=False):
            assert engine.get_fraction("STAT_MR") == config.RISK_PER_TRADE_PCT

    def test_get_fraction_no_data(self):
        """Without compute_fractions, returns flat rate."""
        from risk.kelly import KellyEngine
        import config
        engine = KellyEngine()
        assert engine.get_fraction("STAT_MR") == config.RISK_PER_TRADE_PCT

    def test_compute_fractions_insufficient_trades(self, override_config):
        """With < KELLY_MIN_TRADES, falls back to flat rate."""
        from risk.kelly import KellyEngine
        import config

        # Only 5 trades (less than KELLY_MIN_TRADES=30)
        mock_trades = [{"pnl": 10.0, "pnl_pct": 0.01}] * 5

        with override_config(KELLY_MIN_TRADES=30, KELLY_LOOKBACK=100,
                             KELLY_ENABLED=True):
            with patch("database.get_recent_trades_by_strategy", return_value=mock_trades):
                with patch("database.save_kelly_params"):
                    engine = KellyEngine()
                    engine.compute_fractions()
                    assert engine.get_fraction("STAT_MR") == config.RISK_PER_TRADE_PCT

    def test_compute_fractions_positive_kelly(self, override_config):
        """With profitable history, Kelly fraction is > min risk."""
        from risk.kelly import KellyEngine

        # 60% win rate, 2:1 win/loss ratio
        wins = [{"pnl": 20.0, "pnl_pct": 0.02, "strategy": "STAT_MR"}] * 60
        losses = [{"pnl": -10.0, "pnl_pct": -0.01, "strategy": "STAT_MR"}] * 40
        mock_trades = wins + losses

        with override_config(KELLY_MIN_TRADES=30, KELLY_LOOKBACK=100,
                             KELLY_FRACTION_MULT=0.5, KELLY_MIN_RISK=0.003,
                             KELLY_MAX_RISK=0.02, KELLY_ENABLED=True):
            with patch("database.get_recent_trades", return_value=mock_trades):
                with patch("database.save_kelly_params"):
                    engine = KellyEngine()
                    engine.compute_fractions()
                    frac = engine.get_fraction("STAT_MR")
                    # Kelly = 0.6 - (0.4/2.0) = 0.6 - 0.2 = 0.4, half = 0.2
                    # Clamped to max 0.02
                    assert frac == 0.02  # Clamped to KELLY_MAX_RISK

    def test_compute_fractions_negative_kelly(self, override_config):
        """With losing history, Kelly fraction falls to minimum."""
        from risk.kelly import KellyEngine

        # 20% win rate, 0.5:1 win/loss ratio (losing strategy)
        wins = [{"pnl": 5.0, "pnl_pct": 0.005, "strategy": "STAT_MR"}] * 20
        losses = [{"pnl": -10.0, "pnl_pct": -0.01, "strategy": "STAT_MR"}] * 80
        mock_trades = wins + losses

        with override_config(KELLY_MIN_TRADES=30, KELLY_LOOKBACK=100,
                             KELLY_FRACTION_MULT=0.5, KELLY_MIN_RISK=0.003,
                             KELLY_MAX_RISK=0.02, KELLY_ENABLED=True):
            with patch("database.get_recent_trades", return_value=mock_trades):
                with patch("database.save_kelly_params"):
                    engine = KellyEngine()
                    engine.compute_fractions()
                    frac = engine.get_fraction("STAT_MR")
                    assert frac == 0.003  # Floor at KELLY_MIN_RISK

    def test_compute_fractions_moderate_kelly(self, override_config):
        """With moderate stats, Kelly fraction is between min and max."""
        from risk.kelly import KellyEngine

        # 55% win rate, 1.2:1 win/loss ratio
        wins = [{"pnl": 12.0, "pnl_pct": 0.012, "strategy": "STAT_MR"}] * 55
        losses = [{"pnl": -10.0, "pnl_pct": -0.01, "strategy": "STAT_MR"}] * 45
        mock_trades = wins + losses

        with override_config(KELLY_MIN_TRADES=30, KELLY_LOOKBACK=100,
                             KELLY_FRACTION_MULT=0.5, KELLY_MIN_RISK=0.003,
                             KELLY_MAX_RISK=0.02, KELLY_ENABLED=True):
            with patch("database.get_recent_trades", return_value=mock_trades):
                with patch("database.save_kelly_params"):
                    engine = KellyEngine()
                    engine.compute_fractions()
                    frac = engine.get_fraction("STAT_MR")
                    # Kelly = 0.55 - (0.45/1.2) = 0.55 - 0.375 = 0.175
                    # half = 0.0875, clamped to max 0.02
                    assert frac == 0.02

    def test_kelly_params_stored(self, override_config):
        """Verify params dict is populated after computation."""
        from risk.kelly import KellyEngine

        wins = [{"pnl": 20.0, "pnl_pct": 0.02, "strategy": "STAT_MR"}] * 60
        losses = [{"pnl": -10.0, "pnl_pct": -0.01, "strategy": "STAT_MR"}] * 40
        mock_trades = wins + losses

        with override_config(KELLY_MIN_TRADES=30, KELLY_LOOKBACK=100,
                             KELLY_FRACTION_MULT=0.5, KELLY_MIN_RISK=0.003,
                             KELLY_MAX_RISK=0.02, KELLY_ENABLED=True):
            with patch("database.get_recent_trades", return_value=mock_trades):
                with patch("database.save_kelly_params"):
                    engine = KellyEngine()
                    engine.compute_fractions()
                    params = engine.params
                    assert "STAT_MR" in params
                    assert params["STAT_MR"]["win_rate"] == 0.6
                    assert params["STAT_MR"]["sample_size"] == 100

    def test_kelly_lookback_limit(self, override_config):
        """Only last KELLY_LOOKBACK trades are used."""
        from risk.kelly import KellyEngine

        # 200 trades but lookback is 100
        old_losses = [{"pnl": -10.0, "pnl_pct": -0.01, "strategy": "STAT_MR"}] * 100
        recent_wins = [{"pnl": 20.0, "pnl_pct": 0.02, "strategy": "STAT_MR"}] * 100
        mock_trades = old_losses + recent_wins  # DB returns chronological order

        with override_config(KELLY_MIN_TRADES=30, KELLY_LOOKBACK=100,
                             KELLY_FRACTION_MULT=0.5, KELLY_MIN_RISK=0.003,
                             KELLY_MAX_RISK=0.02, KELLY_ENABLED=True):
            with patch("database.get_recent_trades", return_value=mock_trades):
                with patch("database.save_kelly_params"):
                    engine = KellyEngine()
                    engine.compute_fractions()
                    # Should only see the last 100 (all wins)
                    params = engine.params
                    # HIGH-002: win_rate capped at 0.95
                    assert params["STAT_MR"]["win_rate"] == 0.95

    def test_last_computed_set(self, override_config):
        """last_computed is set after compute_fractions."""
        from risk.kelly import KellyEngine

        with override_config(KELLY_ENABLED=True, KELLY_MIN_TRADES=30, KELLY_LOOKBACK=100):
            with patch("database.get_recent_trades_by_strategy", return_value=[]):
                with patch("database.save_kelly_params"):
                    engine = KellyEngine()
                    assert engine.last_computed is None
                    engine.compute_fractions()
                    assert engine.last_computed is not None


class TestVolTargetingKellyIntegration:
    """Test Kelly integration with VolatilityTargetingRiskEngine."""

    def test_vol_engine_without_kelly(self):
        """Vol engine works without Kelly (backward compatible)."""
        from risk.vol_targeting import VolatilityTargetingRiskEngine
        engine = VolatilityTargetingRiskEngine()
        qty = engine.calculate_position_size(
            equity=100000, entry_price=150.0, stop_price=148.0,
            vol_scalar=1.0, strategy="STAT_MR", side="buy", pnl_lock_mult=1.0,
        )
        assert qty > 0

    def test_vol_engine_with_kelly(self, override_config):
        """Vol engine uses Kelly fraction when engine is set."""
        from risk.vol_targeting import VolatilityTargetingRiskEngine
        from risk.kelly import KellyEngine

        kelly = KellyEngine()
        kelly._fractions["STAT_MR"] = 0.015  # Manually set for test

        vol_engine = VolatilityTargetingRiskEngine()
        vol_engine.set_kelly_engine(kelly)

        # Use wide stop and high cap so position value isn't capped equally
        with override_config(KELLY_ENABLED=True, RISK_PER_TRADE_PCT=0.008,
                             MAX_POSITION_PCT=0.50):
            qty_kelly = vol_engine.calculate_position_size(
                equity=100000, entry_price=150.0, stop_price=140.0,
                vol_scalar=1.0, strategy="STAT_MR", side="buy", pnl_lock_mult=1.0,
            )

            # Without Kelly
            vol_engine2 = VolatilityTargetingRiskEngine()
            qty_flat = vol_engine2.calculate_position_size(
                equity=100000, entry_price=150.0, stop_price=140.0,
                vol_scalar=1.0, strategy="STAT_MR", side="buy", pnl_lock_mult=1.0,
            )

        # Kelly (0.015) > flat (0.008), so Kelly should give more shares
        assert qty_kelly > qty_flat

    def test_vol_engine_kelly_disabled(self, override_config):
        """When Kelly disabled, vol engine uses flat rate even with kelly engine set."""
        from risk.vol_targeting import VolatilityTargetingRiskEngine
        from risk.kelly import KellyEngine

        kelly = KellyEngine()
        kelly._fractions["STAT_MR"] = 0.015

        vol_engine = VolatilityTargetingRiskEngine()
        vol_engine.set_kelly_engine(kelly)

        with override_config(KELLY_ENABLED=False, RISK_PER_TRADE_PCT=0.008):
            qty = vol_engine.calculate_position_size(
                equity=100000, entry_price=150.0, stop_price=148.0,
                vol_scalar=1.0, strategy="STAT_MR", side="buy", pnl_lock_mult=1.0,
            )

        vol_engine2 = VolatilityTargetingRiskEngine()
        qty_flat = vol_engine2.calculate_position_size(
            equity=100000, entry_price=150.0, stop_price=148.0,
            vol_scalar=1.0, strategy="STAT_MR", side="buy", pnl_lock_mult=1.0,
        )

        assert qty == qty_flat


class TestKellyDatabase:
    """Test Kelly database operations."""

    def test_save_kelly_params(self, in_memory_db):
        """Test saving Kelly params to database."""
        import database
        database.save_kelly_params(
            strategy="STAT_MR",
            win_rate=0.6,
            avg_win_loss=2.0,
            kelly_f=0.4,
            half_kelly_f=0.2,
            sample_size=100,
        )

        rows = in_memory_db.execute("SELECT * FROM kelly_params WHERE strategy = 'STAT_MR'").fetchall()
        assert len(rows) == 1
        assert rows[0]["win_rate"] == 0.6
        assert rows[0]["sample_size"] == 100
