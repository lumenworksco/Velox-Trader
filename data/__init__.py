"""data package — Alpaca data fetching, feature store, dynamic universe, data quality.

Backward-compatible: all functions previously in data.py (get_daily_bars, etc.)
are re-exported here alongside V11 additions.
"""

# --- Original data.py functions (moved to data/fetcher.py) ---
from data.fetcher import (
    get_trading_client,
    get_data_client,
    get_account,
    get_clock,
    get_positions,
    get_open_orders,
    get_bars,
    get_daily_bars,
    get_intraday_bars,
    get_filled_exit_info,
    get_filled_exit_price,
    get_snapshot,
    get_snapshots,
    get_snapshots_batch,
    verify_connectivity,
    verify_data_feed,
)

# --- T4-001: Shared bar cache ---
from data.bar_cache import BarCache, bar_cache

# --- V11 additions ---
from data.feature_store import FeatureStore, get_feature_store
from data.universe import DynamicUniverse, get_dynamic_universe
from data.quality import (
    DataQualityFramework,
    DataQualityScore,
    QualityIssue,
    check_bar_quality,
    get_quality_framework,
)

# --- V12 Item 2.3: Data feed monitor ---
from data.feed_monitor import DataFeedMonitor, get_feed_monitor

# --- V12 12.2: Overnight gap analysis ---
from data.gap_analysis import (
    GapType,
    GapInfo,
    compute_gaps,
    get_gap_flags,
    get_gap_info,
    get_mr_candidates,
    get_breakout_candidates,
)

__all__ = [
    # Original data functions
    "get_trading_client",
    "get_data_client",
    "get_account",
    "get_clock",
    "get_positions",
    "get_open_orders",
    "get_bars",
    "get_daily_bars",
    "get_intraday_bars",
    "get_filled_exit_info",
    "get_filled_exit_price",
    "get_snapshot",
    "get_snapshots",
    "get_snapshots_batch",
    "verify_connectivity",
    "verify_data_feed",
    # Bar cache (T4-001)
    "BarCache",
    "bar_cache",
    # Feature store
    "FeatureStore",
    "get_feature_store",
    # Dynamic universe
    "DynamicUniverse",
    "get_dynamic_universe",
    # Data quality
    "DataQualityFramework",
    "DataQualityScore",
    "QualityIssue",
    "check_bar_quality",
    "get_quality_framework",
    # Data feed monitor (V12 Item 2.3)
    "DataFeedMonitor",
    "get_feed_monitor",
    # Gap analysis (V12 12.2)
    "GapType",
    "GapInfo",
    "compute_gaps",
    "get_gap_flags",
    "get_gap_info",
    "get_mr_candidates",
    "get_breakout_candidates",
]
