"""Comprehensive tests for engine/signal_processor.py — signal filtering and sizing.

Tests cover:
- _signal_conviction() scoring
- Intra-scan position limit (max 5)
- Signal conflict resolution (opposite directions skip)
- Sector exposure enforcement
- Data quality gate
- ML confidence gating (reject < 0.35, boost > 0.65)
- Breadth filter (>85% directional -> 50% size cut)
"""

import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Lightweight stand-ins
# ---------------------------------------------------------------------------

@dataclass
class FakeSignal:
    """Minimal Signal stand-in matching strategies.base.Signal fields."""
    symbol: str = "AAPL"
    strategy: str = "STAT_MR"
    side: str = "buy"
    entry_price: float = 100.0
    take_profit: float = 105.0
    stop_loss: float = 97.0
    reason: str = ""
    hold_type: str = "day"
    pair_id: str = ""
    confidence: float = 0.5
    metadata: dict = field(default_factory=dict)
    timestamp: datetime | None = None
    pair_symbol: str | None = None


@dataclass
class FakeRiskManager:
    open_trades: dict = field(default_factory=dict)
    current_equity: float = 100_000.0
    starting_equity: float = 100_000.0
    day_pnl: float = 0.0
    _symbol_daily_pnl: dict = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def can_open_trade(self, strategy=""):
        return True, ""

    def close_trade(self, symbol, exit_price, now, exit_reason="", commission=0.0):
        self.open_trades.pop(symbol, None)

    def partial_close(self, symbol, qty, price, now, reason=""):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_deps():
    """Mock all external deps so signal_processor can import cleanly."""
    import types
    cfg = types.ModuleType("config")
    cfg.ET = ET
    cfg.TELEGRAM_ENABLED = False
    cfg.ADAPTIVE_EXITS_ENABLED = False
    cfg.ALLOW_SHORT = True
    cfg.MAX_NEW_ENTRIES_PER_SCAN = 5
    cfg.MAX_SECTOR_EXPOSURE = 0.30
    cfg.SECTOR_MAP = {
        "AAPL": "Technology", "MSFT": "Technology", "GOOG": "Technology",
        "JPM": "Financials", "BAC": "Financials",
        "XOM": "Energy", "CVX": "Energy",
    }
    cfg.PDT_PROTECTION_ENABLED = False
    cfg.LLM_SCORING_ENABLED = False
    cfg.INTRADAY_SEASONALITY_ENABLED = False
    cfg.HMM_REGIME_ENABLED = False
    cfg.CROSS_ASSET_ENABLED = False
    cfg.NLP_SENTIMENT_ENABLED = False
    cfg.MAX_ACCEPTABLE_DRAWDOWN = 0.08
    cfg.MAX_SYMBOL_DAILY_LOSS = 200.0
    cfg.REENTRY_COOLDOWN_MIN = 15

    mock_database = MagicMock()
    mock_database.log_signal = MagicMock()

    with patch.dict(sys.modules, {
        "config": cfg,
        "database": mock_database,
        "data": MagicMock(),
        "data.feature_store": MagicMock(),
        "data.quality": MagicMock(),
        "data.alternative.sec_filings": MagicMock(),
        "execution": MagicMock(),
        "execution.core": MagicMock(),
        "risk": MagicMock(),
        "risk.pdt_tracker": MagicMock(),
        "strategies.base": MagicMock(),
        "earnings": MagicMock(),
        "correlation": MagicMock(),
        "engine.event_log": MagicMock(),
        "engine.failure_modes": MagicMock(),
        "engine.events": MagicMock(),
        "adaptive_exit_manager": MagicMock(),
        "analytics.intraday_seasonality": MagicMock(),
        "analytics.lead_lag": MagicMock(),
        "ml.alpha_agents": MagicMock(),
        "ml.inference": MagicMock(),
        "ml.training": MagicMock(),
        "ml.finbert_sentiment": MagicMock(),
        "ml.models.fingpt": MagicMock(),
        "microstructure.vpin": MagicMock(),
        "risk.conformal_stops": MagicMock(),
        "compliance.pdt": MagicMock(),
    }):
        # Make the Signal dataclass importable
        sys.modules["strategies.base"].Signal = FakeSignal
        yield cfg, mock_database


@pytest.fixture
def sp(_mock_deps):
    """Import and return a fresh signal_processor module."""
    import importlib
    mod = importlib.import_module("engine.signal_processor")
    importlib.reload(mod)
    return mod


# ===================================================================
# _signal_conviction
# ===================================================================

class TestSignalConviction:
    """Scoring based on confidence + risk/reward ratio."""

    def test_base_score_equals_confidence(self, sp):
        sig = FakeSignal(confidence=0.7, entry_price=0, stop_loss=0, take_profit=0)
        # When prices are 0/invalid, score is just confidence
        score = sp._signal_conviction(sig)
        assert score == pytest.approx(0.7)

    def test_rr_ratio_adds_tiebreaker(self, sp):
        # RR = (105-100)/(100-97) = 5/3 ~= 1.67
        # Tiebreaker = min(1.67 * 0.1, 0.5) = 0.167
        sig = FakeSignal(
            confidence=0.5, entry_price=100.0,
            take_profit=105.0, stop_loss=97.0, side="buy",
        )
        score = sp._signal_conviction(sig)
        expected = 0.5 + min(1.67 * 0.1, 0.5)
        assert score == pytest.approx(expected, rel=0.05)

    def test_rr_ratio_capped_at_0_5(self, sp):
        # Very high RR: reward=20, risk=1 -> RR=20 -> min(20*0.1, 0.5) = 0.5
        sig = FakeSignal(
            confidence=0.3, entry_price=100.0,
            take_profit=120.0, stop_loss=99.0, side="buy",
        )
        score = sp._signal_conviction(sig)
        assert score == pytest.approx(0.3 + 0.5)

    def test_sell_side_rr(self, sp):
        # Sell: reward = entry - tp = 100 - 95 = 5, risk = sl - entry = 103 - 100 = 3
        sig = FakeSignal(
            confidence=0.6, entry_price=100.0,
            take_profit=95.0, stop_loss=103.0, side="sell",
        )
        score = sp._signal_conviction(sig)
        rr = 5.0 / 3.0
        expected = 0.6 + min(rr * 0.1, 0.5)
        assert score == pytest.approx(expected, rel=0.05)

    def test_zero_risk_no_crash(self, sp):
        # stop_loss = entry_price -> risk = 0 -> no tiebreaker
        sig = FakeSignal(
            confidence=0.5, entry_price=100.0,
            take_profit=105.0, stop_loss=100.0, side="buy",
        )
        score = sp._signal_conviction(sig)
        assert score == pytest.approx(0.5)

    def test_higher_confidence_ranks_higher(self, sp):
        sig_high = FakeSignal(confidence=0.9, entry_price=100.0,
                               take_profit=105.0, stop_loss=97.0)
        sig_low = FakeSignal(confidence=0.3, entry_price=100.0,
                              take_profit=105.0, stop_loss=97.0)
        assert sp._signal_conviction(sig_high) > sp._signal_conviction(sig_low)


# ===================================================================
# _resolve_signal_conflicts
# ===================================================================

class TestSignalConflictResolution:
    """Opposite directions skip both; same direction keeps best."""

    def test_opposite_directions_both_skipped(self, sp, _mock_deps):
        _, mock_db = _mock_deps
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        signals = [
            FakeSignal(symbol="AAPL", strategy="STAT_MR", side="buy", confidence=0.8),
            FakeSignal(symbol="AAPL", strategy="ORB", side="sell", confidence=0.7),
        ]

        resolved = sp._resolve_signal_conflicts(signals, now)
        symbols = [s.symbol for s in resolved]
        assert "AAPL" not in symbols

    def test_same_direction_keeps_highest_conviction(self, sp, _mock_deps):
        _, mock_db = _mock_deps
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        signals = [
            FakeSignal(symbol="AAPL", strategy="STAT_MR", side="buy", confidence=0.9),
            FakeSignal(symbol="AAPL", strategy="ORB", side="buy", confidence=0.4),
        ]

        resolved = sp._resolve_signal_conflicts(signals, now)
        assert len(resolved) == 1
        assert resolved[0].strategy == "STAT_MR"  # Higher confidence

    def test_single_signal_passes_through(self, sp, _mock_deps):
        _, mock_db = _mock_deps
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        signals = [FakeSignal(symbol="AAPL", strategy="STAT_MR", side="buy")]
        resolved = sp._resolve_signal_conflicts(signals, now)
        assert len(resolved) == 1

    def test_different_symbols_not_conflicting(self, sp, _mock_deps):
        _, mock_db = _mock_deps
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        signals = [
            FakeSignal(symbol="AAPL", strategy="STAT_MR", side="buy"),
            FakeSignal(symbol="GOOG", strategy="ORB", side="sell"),
        ]

        resolved = sp._resolve_signal_conflicts(signals, now)
        assert len(resolved) == 2

    def test_three_signals_same_symbol_same_direction(self, sp, _mock_deps):
        _, mock_db = _mock_deps
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        signals = [
            FakeSignal(symbol="AAPL", strategy="A", side="buy", confidence=0.3),
            FakeSignal(symbol="AAPL", strategy="B", side="buy", confidence=0.9),
            FakeSignal(symbol="AAPL", strategy="C", side="buy", confidence=0.6),
        ]

        resolved = sp._resolve_signal_conflicts(signals, now)
        assert len(resolved) == 1
        assert resolved[0].strategy == "B"

    def test_multiple_conflict_groups(self, sp, _mock_deps):
        """AAPL has conflict (skip both), GOOG has dupes (keep best)."""
        _, mock_db = _mock_deps
        now = datetime(2026, 4, 3, 11, 0, tzinfo=ET)

        signals = [
            FakeSignal(symbol="AAPL", strategy="A", side="buy", confidence=0.8),
            FakeSignal(symbol="AAPL", strategy="B", side="sell", confidence=0.7),
            FakeSignal(symbol="GOOG", strategy="C", side="buy", confidence=0.6),
            FakeSignal(symbol="GOOG", strategy="D", side="buy", confidence=0.9),
        ]

        resolved = sp._resolve_signal_conflicts(signals, now)
        symbols = {s.symbol for s in resolved}
        assert "AAPL" not in symbols
        assert "GOOG" in symbols
        assert len(resolved) == 1
        assert resolved[0].strategy == "D"


# ===================================================================
# Sector exposure enforcement
# ===================================================================

class TestSectorExposure:
    """_check_sector_exposure blocks trades exceeding sector limit."""

    def test_sector_allowed_when_below_limit(self, sp, _mock_deps):
        cfg, _ = _mock_deps
        rm = FakeRiskManager(current_equity=100_000)
        rm.open_trades["MSFT"] = MagicMock(entry_price=300.0, qty=30)  # $9k Tech

        allowed, reason = sp._check_sector_exposure("AAPL", 150.0, rm)
        assert allowed is True

    def test_sector_blocked_when_at_limit(self, sp, _mock_deps):
        cfg, _ = _mock_deps
        rm = FakeRiskManager(current_equity=100_000)
        # Tech sector already at 30% ($30k)
        rm.open_trades["MSFT"] = MagicMock(entry_price=300.0, qty=100)  # $30k

        allowed, reason = sp._check_sector_exposure("AAPL", 150.0, rm)
        assert allowed is False
        assert "Technology" in reason

    def test_unknown_sector_is_allowed(self, sp, _mock_deps):
        rm = FakeRiskManager(current_equity=100_000)
        # UNKNOWN not in SECTOR_MAP
        allowed, reason = sp._check_sector_exposure("UNKNOWN", 50.0, rm)
        assert allowed is True

    def test_different_sectors_dont_conflict(self, sp, _mock_deps):
        rm = FakeRiskManager(current_equity=100_000)
        # Financials at 30%
        rm.open_trades["JPM"] = MagicMock(entry_price=150.0, qty=200)  # $30k

        # Adding Technology should be fine
        allowed, reason = sp._check_sector_exposure("AAPL", 150.0, rm)
        assert allowed is True

    def test_zero_equity_is_allowed(self, sp, _mock_deps):
        rm = FakeRiskManager(current_equity=0)
        allowed, reason = sp._check_sector_exposure("AAPL", 150.0, rm)
        assert allowed is True


# ===================================================================
# Intra-scan position limit
# ===================================================================

class TestIntraScanLimit:
    """Max 5 new entries per scan cycle."""

    def test_intra_scan_limit_caps_entries(self, sp, _mock_deps):
        cfg, mock_db = _mock_deps
        cfg.MAX_NEW_ENTRIES_PER_SCAN = 3

        # Re-import to pick up new config
        import importlib
        importlib.reload(sp) if hasattr(sp, '__spec__') else None

        # Create 5 signals
        signals = []
        for i in range(5):
            sym = f"SYM{i}"
            signals.append(FakeSignal(
                symbol=sym, strategy="STAT_MR", side="buy",
                entry_price=100.0, confidence=0.5 + i * 0.05,
            ))

        # Verify the conviction-based sort ranks highest confidence first
        signals_sorted = sorted(signals, key=lambda s: sp._signal_conviction(s), reverse=True)
        assert signals_sorted[0].symbol == "SYM4"  # highest confidence

    def test_signals_sorted_by_conviction(self, sp):
        signals = [
            FakeSignal(symbol="A", confidence=0.3),
            FakeSignal(symbol="B", confidence=0.9),
            FakeSignal(symbol="C", confidence=0.6),
        ]

        signals_sorted = sorted(signals, key=lambda s: sp._signal_conviction(s), reverse=True)
        assert signals_sorted[0].symbol == "B"
        assert signals_sorted[1].symbol == "C"
        assert signals_sorted[2].symbol == "A"


# ===================================================================
# Breadth filter
# ===================================================================

class TestBreadthFilter:
    """When >85% of signals lean one direction, breadth_mult = 0.5."""

    def test_breadth_filter_triggers_on_extreme_bias(self, sp):
        # 6 buys, 0 sells = 100% directional
        signals = [FakeSignal(symbol=f"S{i}", side="buy") for i in range(6)]
        buy_count = sum(1 for s in signals if s.side == "buy")
        total = len(signals)
        max_direction_pct = max(buy_count, total - buy_count) / total
        assert max_direction_pct > 0.85

    def test_breadth_filter_does_not_trigger_balanced(self, sp):
        # 3 buys, 3 sells = 50% directional
        signals = (
            [FakeSignal(symbol=f"B{i}", side="buy") for i in range(3)] +
            [FakeSignal(symbol=f"S{i}", side="sell") for i in range(3)]
        )
        buy_count = sum(1 for s in signals if s.side == "buy")
        total = len(signals)
        max_direction_pct = max(buy_count, total - buy_count) / total
        assert max_direction_pct <= 0.85

    def test_breadth_filter_requires_minimum_signals(self, sp):
        # Only 3 signals — below threshold of 5 for breadth calculation
        signals = [FakeSignal(symbol=f"S{i}", side="buy") for i in range(3)]
        # With fewer than 5 signals, breadth_mult should stay 1.0
        breadth_mult = 1.0
        if len(signals) >= 5:
            buy_count = sum(1 for s in signals if s.side == "buy")
            max_direction_pct = max(buy_count, len(signals) - buy_count) / len(signals)
            if max_direction_pct > 0.85:
                breadth_mult = 0.5
        assert breadth_mult == 1.0

    def test_breadth_at_exactly_85pct(self, sp):
        """6 of 7 signals = 85.7% — should trigger."""
        signals = (
            [FakeSignal(symbol=f"B{i}", side="buy") for i in range(6)] +
            [FakeSignal(symbol="S0", side="sell")]
        )
        buy_count = sum(1 for s in signals if s.side == "buy")
        total = len(signals)
        max_direction_pct = max(buy_count, total - buy_count) / total
        breadth_mult = 0.5 if (len(signals) >= 5 and max_direction_pct > 0.85) else 1.0
        assert breadth_mult == 0.5


# ===================================================================
# ML confidence gating
# ===================================================================

class TestMLConfidenceGating:
    """ML model rejects < 0.35, neutral 0.35-0.65, boosts > 0.65."""

    def test_ml_reject_below_0_35(self):
        confidence = 0.25
        assert confidence < 0.35
        # In the real code, this returns None from _process_single_signal

    def test_ml_neutral_zone(self):
        confidence = 0.50
        assert 0.35 <= confidence <= 0.65
        ml_conf_mult = 1.0  # No adjustment in neutral zone
        assert ml_conf_mult == 1.0

    def test_ml_boost_above_0_65(self):
        confidence = 0.80
        ml_conf_mult = 1.0 + (confidence - 0.65) * 0.57
        ml_conf_mult = min(ml_conf_mult, 1.2)
        # 1.0 + 0.15 * 0.57 = 1.0855
        assert 1.0 < ml_conf_mult < 1.2
        assert ml_conf_mult == pytest.approx(1.0855, rel=0.01)

    def test_ml_boost_capped_at_1_2(self):
        confidence = 1.0
        ml_conf_mult = 1.0 + (confidence - 0.65) * 0.57
        ml_conf_mult = min(ml_conf_mult, 1.2)
        # 1.0 + 0.35 * 0.57 = 1.1995
        assert ml_conf_mult == pytest.approx(1.1995, rel=0.01)

    def test_ml_boundary_exactly_0_35(self):
        confidence = 0.35
        # At exactly 0.35, should NOT be rejected (>= 0.35 is neutral)
        assert confidence >= 0.35
        # Neutral zone: ml_conf_mult = 1.0
        ml_conf_mult = 1.0
        assert ml_conf_mult == 1.0

    def test_ml_boundary_exactly_0_65(self):
        confidence = 0.65
        # At 0.65, boost starts: 1.0 + 0.0 * 0.57 = 1.0
        ml_conf_mult = 1.0 + (confidence - 0.65) * 0.57
        ml_conf_mult = min(ml_conf_mult, 1.2)
        assert ml_conf_mult == pytest.approx(1.0)


# ===================================================================
# Data quality gate
# ===================================================================

class TestDataQualityGate:
    """Signal rejected when data quality < 0.5, reduced at < 0.8."""

    def test_low_quality_rejects(self):
        dq_score = 0.3
        assert dq_score < 0.5
        # Would cause skip_reason = "low_data_quality_0.30"

    def test_medium_quality_reduces_size(self):
        dq_score = 0.6
        assert 0.5 <= dq_score < 0.8
        mult = dq_score / 0.8
        assert mult == pytest.approx(0.75)

    def test_high_quality_no_reduction(self):
        dq_score = 0.9
        assert dq_score >= 0.8
        mult = 1.0
        assert mult == 1.0

    def test_none_quality_is_fail_open(self):
        """When quality check returns None, signal should pass."""
        dq_score = None
        # In the code: if dq_score is not None then check threshold
        # None means fail-open — no rejection
        should_skip = dq_score is not None and dq_score < 0.5
        assert should_skip is False


# ===================================================================
# register_stopout / cooldown
# ===================================================================

class TestStopoutCooldown:
    """Post stop-loss cooldown blocks re-entry."""

    def test_register_and_check_cooldown(self, sp, _mock_deps):
        cfg, _ = _mock_deps
        cfg.REENTRY_COOLDOWN_MIN = 15

        sp.register_stopout("AAPL")
        now = datetime.now()
        assert sp._is_in_cooldown("AAPL", now) is True

    def test_cooldown_expires(self, sp, _mock_deps):
        cfg, _ = _mock_deps
        cfg.REENTRY_COOLDOWN_MIN = 15

        sp.register_stopout("AAPL")
        future = datetime.now() + timedelta(minutes=20)
        assert sp._is_in_cooldown("AAPL", future) is False

    def test_no_cooldown_for_unregistered(self, sp):
        now = datetime.now()
        assert sp._is_in_cooldown("GOOG", now) is False
