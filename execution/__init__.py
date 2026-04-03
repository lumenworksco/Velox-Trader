"""Execution engine — smart routing, optimal execution, slippage modeling, fill analytics.

Backward-compatible: all functions previously in execution.py are re-exported here.
"""

# --- Original execution.py functions (moved to execution/core.py) ---
from execution.core import (
    ValidationResult,
    validate_order_pretrade,
    CloseResult,
    submit_bracket_order,
    submit_twap_order,
    close_position,
    close_all_positions,
    cancel_all_open_orders,
    close_orb_positions,
    check_vwap_time_stops,
    close_partial_position,
    can_short,
    _submit_order,
    _check_order_filled,
    _compute_backoff_delay,
    _validate_bracket_params,
)
from data import get_trading_client  # Re-export for backward compatibility

# --- V11 additions ---
from execution.smart_router import (
    SmartOrderRouter,
    OrderParams,
    OrderTypeChoice,
    UrgencyLevel,
    MarketConditions,
    FillMonitor,
)
from execution.optimal_execution import (
    AlmgrenChriss,
    ExecutionSlice,
    ExecutionSchedule,
    ImpactParams,
)
from execution.slippage_model import (
    SlippageModel,
    SlippageFeatures,
    SlippagePrediction,
)
from execution.fill_analytics import (
    FillAnalytics,
    FillRecord,
    ExecutionReport,
)

# T7-001: RL Execution Agent (fail-open)
try:
    from execution.rl_executor import (
        RLExecutionAgent,
        ExecutionState,
        ExecutionAction,
    )
except ImportError:
    RLExecutionAgent = None  # type: ignore
    ExecutionState = None    # type: ignore
    ExecutionAction = None   # type: ignore

__all__ = [
    # Original execution functions
    "ValidationResult",
    "validate_order_pretrade",
    "CloseResult",
    "submit_bracket_order",
    "submit_twap_order",
    "close_position",
    "close_all_positions",
    "cancel_all_open_orders",
    "close_orb_positions",
    "check_vwap_time_stops",
    "close_partial_position",
    "can_short",
    # EXEC-001: Smart Order Router
    "SmartOrderRouter",
    "OrderParams",
    "OrderTypeChoice",
    "UrgencyLevel",
    "MarketConditions",
    "FillMonitor",
    # EXEC-002: Almgren-Chriss
    "AlmgrenChriss",
    "ExecutionSlice",
    "ExecutionSchedule",
    "ImpactParams",
    # EXEC-003: Slippage Model
    "SlippageModel",
    "SlippageFeatures",
    "SlippagePrediction",
    # EXEC-004: Fill Analytics
    "FillAnalytics",
    "FillRecord",
    "ExecutionReport",
    # T7-001: RL Execution Agent
    "RLExecutionAgent",
    "ExecutionState",
    "ExecutionAction",
]
