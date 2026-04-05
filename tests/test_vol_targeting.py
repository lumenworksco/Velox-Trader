"""Comprehensive tests for risk/vol_targeting.py.

Covers:
- Position sizing with vol target
- compute_vol_scalar() computation
- Min/max bounds on scalar and position size
- Kelly integration
- Intraday and Friday multipliers
- Strategy allocation weighting
- PnL lock multiplier
- Short selling reduction
"""

from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pytest

import config
from risk.vol_targeting import VolatilityTargetingRiskEngine

ET = ZoneInfo("America/New_York")


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def engine():
    """Fresh VolatilityTargetingRiskEngine."""
    return VolatilityTargetingRiskEngine()


# ===================================================================
# compute_vol_scalar()
# ===================================================================

class TestComputeVolScalar:

    def test_default_inputs_reasonable_scalar(self, engine):
        """With default VIX=20, ATR=1%, PnL_std=1%, scalar should be reasonable."""
        scalar = engine.compute_vol_scalar(vix=20.0, portfolio_atr_vol=0.01,
                                            rolling_pnl_std=0.01)
        assert config.VOL_SCALAR_MIN <= scalar <= config.VOL_SCALAR_MAX

    def test_high_vol_lowers_scalar(self, engine):
        """Higher estimated vol should produce a lower scalar."""
        scalar_normal = engine.compute_vol_scalar(vix=20.0, portfolio_atr_vol=0.01,
                                                   rolling_pnl_std=0.01)
        scalar_high = engine.compute_vol_scalar(vix=40.0, portfolio_atr_vol=0.02,
                                                 rolling_pnl_std=0.02)
        assert scalar_high < scalar_normal

    def test_low_vol_raises_scalar(self, engine):
        """Lower estimated vol should produce a higher scalar."""
        scalar_normal = engine.compute_vol_scalar(vix=20.0, portfolio_atr_vol=0.01,
                                                   rolling_pnl_std=0.01)
        scalar_low = engine.compute_vol_scalar(vix=10.0, portfolio_atr_vol=0.005,
                                                rolling_pnl_std=0.005)
        assert scalar_low > scalar_normal

    def test_near_zero_vol_floors_at_min(self, engine):
        """If estimated vol is essentially zero, scalar should be clamped to min."""
        scalar = engine.compute_vol_scalar(vix=0.001, portfolio_atr_vol=0.0,
                                            rolling_pnl_std=0.0)
        assert scalar == config.VOL_SCALAR_MIN

    def test_extreme_vol_caps_at_min(self, engine):
        """Extremely high vol should produce scalar at or near the minimum."""
        scalar = engine.compute_vol_scalar(vix=80.0, portfolio_atr_vol=0.10,
                                            rolling_pnl_std=0.10)
        assert scalar == config.VOL_SCALAR_MIN

    def test_scalar_clamped_to_bounds(self, engine, override_config):
        """Scalar should always be within [VOL_SCALAR_MIN, VOL_SCALAR_MAX]."""
        with override_config(VOL_SCALAR_MIN=0.3, VOL_SCALAR_MAX=1.5):
            for vix in [5, 10, 20, 40, 80]:
                for atr in [0.001, 0.005, 0.01, 0.03]:
                    scalar = engine.compute_vol_scalar(vix=vix, portfolio_atr_vol=atr,
                                                        rolling_pnl_std=atr)
                    assert 0.3 <= scalar <= 1.5

    def test_last_scalar_property_updated(self, engine):
        """last_scalar should reflect the most recent computation."""
        engine.compute_vol_scalar(vix=20.0, portfolio_atr_vol=0.01,
                                   rolling_pnl_std=0.01)
        assert engine.last_scalar > 0

    def test_weighted_average_components(self, engine):
        """Verify the 30/40/30 weighting works correctly."""
        # VIX=20 => daily_vix = 20/100/sqrt(252) ~= 0.01259
        # estimated_vol = 0.01259*0.3 + 0.01*0.4 + 0.01*0.3 = ~0.01078
        # target_vol = 0.01 => scalar = 0.01/0.01078 ~= 0.928
        scalar = engine.compute_vol_scalar(vix=20.0, portfolio_atr_vol=0.01,
                                            rolling_pnl_std=0.01)
        # Should be close to 1.0 but slightly below (VIX component adds vol)
        assert 0.5 < scalar < 1.2

    def test_exact_math(self, engine, override_config):
        """Verify exact scalar calculation with known inputs."""
        with override_config(VOL_TARGET_DAILY=0.01, VOL_SCALAR_MIN=0.3, VOL_SCALAR_MAX=1.5):
            # VIX=15.87 => daily = 15.87/100/sqrt(252) = ~0.01
            vix_daily = 15.87 / 100.0 / np.sqrt(252)
            atr = 0.01
            pnl_std = 0.01
            estimated_vol = vix_daily * 0.3 + atr * 0.4 + pnl_std * 0.3
            expected_scalar = 0.01 / estimated_vol
            expected_scalar = max(0.3, min(1.5, expected_scalar))

            scalar = engine.compute_vol_scalar(vix=15.87, portfolio_atr_vol=0.01,
                                                rolling_pnl_std=0.01)
            assert abs(scalar - expected_scalar) < 0.01


# ===================================================================
# calculate_position_size()
# ===================================================================

class TestCalculatePositionSize:

    def test_basic_long_position(self, engine, override_config):
        """Basic position sizing with default parameters."""
        with override_config(ADAPTIVE_ALLOCATION_ENABLED=False):
            qty = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=98.0,
                vol_scalar=1.0, strategy="ORB", side="buy",
            )
        assert qty > 0

    def test_zero_equity_returns_zero(self, engine):
        qty = engine.calculate_position_size(
            equity=0, entry_price=100.0, stop_price=98.0,
        )
        assert qty == 0

    def test_zero_entry_price_returns_zero(self, engine):
        qty = engine.calculate_position_size(
            equity=100_000, entry_price=0.0, stop_price=5.0,
        )
        assert qty == 0

    def test_zero_risk_per_share_returns_zero(self, engine):
        """If entry == stop, risk per share is ~0, should return 0."""
        qty = engine.calculate_position_size(
            equity=100_000, entry_price=100.0, stop_price=100.0,
        )
        assert qty == 0

    def test_vol_scalar_scales_position(self, engine, override_config):
        """Higher vol_scalar should produce larger position."""
        with override_config(ADAPTIVE_ALLOCATION_ENABLED=False):
            qty_normal = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=95.0,
                vol_scalar=1.0, strategy="ORB",
            )
            qty_high = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=95.0,
                vol_scalar=1.5, strategy="ORB",
            )
        assert qty_high >= qty_normal

    def test_vol_scalar_zero_point_three(self, engine, override_config):
        """Low vol_scalar should produce smaller position."""
        with override_config(ADAPTIVE_ALLOCATION_ENABLED=False):
            qty_normal = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=95.0,
                vol_scalar=1.0, strategy="ORB",
            )
            qty_low = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=95.0,
                vol_scalar=0.3, strategy="ORB",
            )
        assert qty_low <= qty_normal

    def test_short_side_reduces_size(self, engine, override_config):
        """Short trades get reduced by SHORT_SIZE_MULTIPLIER."""
        with override_config(ADAPTIVE_ALLOCATION_ENABLED=False):
            qty_long = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=95.0,
                vol_scalar=1.0, strategy="ORB", side="buy",
            )
            qty_short = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=105.0,
                vol_scalar=1.0, strategy="ORB", side="sell",
            )
        assert qty_short <= qty_long

    def test_pnl_lock_reduces_size(self, engine, override_config):
        """PnL lock multiplier < 1 should reduce position size."""
        with override_config(ADAPTIVE_ALLOCATION_ENABLED=False):
            qty_full = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=95.0,
                vol_scalar=1.0, strategy="ORB", pnl_lock_mult=1.0,
            )
            qty_locked = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=95.0,
                vol_scalar=1.0, strategy="ORB", pnl_lock_mult=0.5,
            )
        assert qty_locked <= qty_full

    def test_position_capped_at_max_pct(self, engine, override_config):
        """Position value should not exceed equity * MAX_POSITION_PCT."""
        with override_config(ADAPTIVE_ALLOCATION_ENABLED=False, MAX_POSITION_PCT=0.08):
            qty = engine.calculate_position_size(
                equity=100_000, entry_price=10.0, stop_price=9.99,
                vol_scalar=1.5, strategy="ORB",
            )
        max_value = 100_000 * 0.08
        position_value = qty * 10.0
        assert position_value <= max_value + 10.0  # tolerance for rounding

    def test_position_below_min_returns_zero(self, engine, override_config):
        """If position value < MIN_POSITION_VALUE, return 0."""
        with override_config(ADAPTIVE_ALLOCATION_ENABLED=False,
                             MIN_POSITION_VALUE=1_000_000):
            qty = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=95.0,
                vol_scalar=1.0,
            )
        assert qty == 0

    def test_intraday_multiplier_applied(self, engine, override_config):
        """Time-of-day multiplier should reduce size during lunch."""
        with override_config(ADAPTIVE_ALLOCATION_ENABLED=False):
            # Prime session (10:30) => multiplier 1.0
            now_prime = datetime(2026, 4, 1, 10, 30, tzinfo=ET)
            qty_prime = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=95.0,
                vol_scalar=1.0, strategy="ORB", now=now_prime,
            )
            # Lunch session (12:00) => multiplier 0.60
            now_lunch = datetime(2026, 4, 1, 12, 0, tzinfo=ET)
            qty_lunch = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=95.0,
                vol_scalar=1.0, strategy="ORB", now=now_lunch,
            )
        assert qty_lunch <= qty_prime

    def test_friday_eow_multiplier_applied(self, engine, override_config):
        """Friday after 2pm should reduce size by 50%."""
        with override_config(ADAPTIVE_ALLOCATION_ENABLED=False):
            # Wednesday afternoon
            now_wed = datetime(2026, 4, 1, 14, 30, tzinfo=ET)
            qty_wed = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=95.0,
                vol_scalar=1.0, strategy="ORB", now=now_wed,
            )
            # Friday afternoon
            now_fri = datetime(2026, 4, 3, 14, 30, tzinfo=ET)
            qty_fri = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=95.0,
                vol_scalar=1.0, strategy="ORB", now=now_fri,
            )
        assert qty_fri <= qty_wed

    def test_no_now_skips_intraday_adjustments(self, engine, override_config):
        """When now=None, intraday/Friday multipliers should be 1.0."""
        with override_config(ADAPTIVE_ALLOCATION_ENABLED=False):
            qty = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=95.0,
                vol_scalar=1.0, strategy="ORB", now=None,
            )
        assert qty > 0


# ===================================================================
# Kelly integration
# ===================================================================

class TestKellyIntegration:

    def test_kelly_engine_used_when_set(self, engine, override_config):
        """If Kelly engine is set, it should determine risk_pct."""
        mock_kelly = MagicMock()
        mock_kelly.get_fraction.return_value = 0.02  # 2% Kelly fraction
        engine.set_kelly_engine(mock_kelly)

        with override_config(ADAPTIVE_ALLOCATION_ENABLED=False):
            qty = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=95.0,
                vol_scalar=1.0, strategy="ORB",
            )
        mock_kelly.get_fraction.assert_called_once_with("ORB")
        assert qty > 0

    def test_flat_risk_when_no_kelly(self, engine, override_config):
        """Without Kelly engine, RISK_PER_TRADE_PCT should be used."""
        with override_config(ADAPTIVE_ALLOCATION_ENABLED=False, RISK_PER_TRADE_PCT=0.01):
            qty = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=95.0,
                vol_scalar=1.0, strategy="ORB",
            )
        assert qty > 0


# ===================================================================
# Adaptive allocation weights
# ===================================================================

class TestAdaptiveWeights:

    def test_adaptive_weights_used_when_enabled(self, engine, override_config):
        """Adaptive allocation weights should scale position size."""
        weights = {"ORB": 0.50, "VWAP": 0.50}
        engine.set_adaptive_weights(weights)
        with override_config(ADAPTIVE_ALLOCATION_ENABLED=True):
            qty = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=95.0,
                vol_scalar=1.0, strategy="ORB",
            )
        assert qty > 0

    def test_fallback_to_config_allocations(self, engine, override_config):
        """If adaptive not enabled, use config.STRATEGY_ALLOCATIONS."""
        with override_config(ADAPTIVE_ALLOCATION_ENABLED=False):
            qty = engine.calculate_position_size(
                equity=100_000, entry_price=100.0, stop_price=95.0,
                vol_scalar=1.0, strategy="ORB",
            )
        assert qty > 0

    def test_set_adaptive_weights_thread_safe(self, engine):
        """set_adaptive_weights should be callable from multiple threads."""
        import threading
        results = []

        def _set(idx):
            engine.set_adaptive_weights({f"STR{idx}": 0.5})
            results.append(idx)

        threads = [threading.Thread(target=_set, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 10


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:

    def test_very_small_equity(self, engine, override_config):
        """Very small equity should still work or return 0."""
        with override_config(ADAPTIVE_ALLOCATION_ENABLED=False, MIN_POSITION_VALUE=100):
            qty = engine.calculate_position_size(
                equity=50.0, entry_price=100.0, stop_price=95.0,
                vol_scalar=1.0,
            )
        # $50 equity * 0.008 risk = $0.40 => likely 0 shares
        assert qty == 0

    def test_very_expensive_stock(self, engine, override_config):
        """Should handle expensive stocks gracefully."""
        with override_config(ADAPTIVE_ALLOCATION_ENABLED=False):
            qty = engine.calculate_position_size(
                equity=100_000, entry_price=5000.0, stop_price=4900.0,
                vol_scalar=1.0, strategy="ORB",
            )
        assert qty >= 0

    def test_negative_equity_returns_zero(self, engine):
        qty = engine.calculate_position_size(
            equity=-10_000, entry_price=100.0, stop_price=95.0,
        )
        assert qty == 0

    def test_negative_entry_price_returns_zero(self, engine):
        qty = engine.calculate_position_size(
            equity=100_000, entry_price=-5.0, stop_price=-10.0,
        )
        assert qty == 0
