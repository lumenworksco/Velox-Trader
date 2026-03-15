"""Risk package — V6 risk engine + backward-compatible V1-V5 risk manager."""

# V6 risk modules
from risk.vol_targeting import VolatilityTargetingRiskEngine
from risk.daily_pnl_lock import DailyPnLLock, LockState
from risk.beta_neutralizer import BetaNeutralizer

# V8 risk modules
from risk.kelly import KellyEngine
from risk.portfolio_heat import PortfolioHeatTracker

# Backward compatibility: re-export RiskManager, TradeRecord, VIX functions
from risk.risk_manager import (
    RiskManager,
    TradeRecord,
    get_vix_level,
    get_vix_risk_scalar,
)

__all__ = [
    # V6
    "VolatilityTargetingRiskEngine",
    "DailyPnLLock",
    "LockState",
    "BetaNeutralizer",
    # V8
    "KellyEngine",
    "PortfolioHeatTracker",
    # V1-V5 backward compat
    "RiskManager",
    "TradeRecord",
    "get_vix_level",
    "get_vix_risk_scalar",
]
