"""SQLite database — replaces state.json for persistence and adds trade/signal logging.

PROD-005: Schema versioning with auto-migration framework.
PROD-009: Connection pooling via queue-based pattern.
"""

import json
import logging
import queue
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import config
from tz_utils import ensure_et

logger = logging.getLogger(__name__)


def _to_iso(dt) -> str | None:
    """V10 BUG-046: Standardize all datetime serialization to ISO format with timezone.

    BUG-009: Uses ensure_et() to properly handle naive datetimes (assumes ET)
    and convert aware datetimes to ET before serializing.
    """
    if dt is None:
        return None
    if hasattr(dt, 'isoformat'):
        return ensure_et(dt).isoformat()
    return str(dt)


# HIGH-013: RLock allows re-entrant locking for nested read-modify-write cycles
_db_lock = threading.RLock()


# =============================================================================
# PROD-009: Connection Pool
# =============================================================================

class ConnectionPool:
    """PROD-009: Queue-based SQLite connection pool.

    Maintains a pool of pre-created connections to reduce connection overhead.
    Connections are borrowed via `get()` and returned via `put()`, or used
    as a context manager.

    QW-011: Default pool_size raised from 3 to 10 to reduce contention
    under concurrent strategy scanning + signal processing + dashboard queries.
    SQLite WAL mode supports this level of concurrency safely.
    """

    def __init__(self, db_path: str, pool_size: int = 10):
        """Initialize the connection pool.

        Args:
            db_path: Path to the SQLite database file.
            pool_size: Number of connections to maintain in the pool.
        """
        self._db_path = db_path
        self._pool_size = pool_size
        self._pool: queue.Queue[sqlite3.Connection] = queue.Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._created = 0

        # Pre-populate the pool
        for _ in range(pool_size):
            conn = self._create_connection()
            self._pool.put(conn)
            self._created += 1

        logger.info("PROD-009: ConnectionPool initialized (size=%d, db=%s)", pool_size, db_path)

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with standard pragmas."""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def get(self, timeout: float = 5.0, retries: int = 2) -> sqlite3.Connection:
        """Borrow a connection from the pool with retry logic.

        T4-004: Added retry with backoff. On final failure, creates overflow connection.

        Args:
            timeout: Max seconds to wait for an available connection per attempt.
            retries: Number of retry attempts before creating overflow connection.

        Returns:
            A SQLite connection.
        """
        import time as _time
        for attempt in range(retries + 1):
            try:
                return self._pool.get(timeout=timeout)
            except queue.Empty:
                if attempt < retries:
                    backoff = 0.1 * (2 ** attempt)
                    logger.debug("PROD-009: Pool get attempt %d failed, retrying in %.1fs", attempt + 1, backoff)
                    _time.sleep(backoff)

        logger.warning("T4-004: Connection pool exhausted after %d attempts, creating overflow connection", retries + 1)
        return self._create_connection()

    def put(self, conn: sqlite3.Connection):
        """Return a connection to the pool."""
        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            # Overflow connection — close it
            try:
                conn.close()
            except Exception:
                pass

    def close_all(self):
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except (queue.Empty, Exception):
                break
        logger.info("PROD-009: Connection pool closed")

    def stats(self) -> dict:
        """Return pool statistics."""
        return {
            "pool_size": self._pool_size,
            "available": self._pool.qsize(),
            "created": self._created,
            "db_path": self._db_path,
        }


# Module-level pool (initialized lazily)
_connection_pool: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def get_connection_pool() -> ConnectionPool:
    """Get or create the global connection pool singleton."""
    global _connection_pool
    with _pool_lock:
        if _connection_pool is None:
            pool_size = int(getattr(config, "DB_POOL_SIZE", 10))  # QW-011: default 10
            _connection_pool = ConnectionPool(config.DB_FILE, pool_size=pool_size)
    return _connection_pool


# Legacy single-connection support (backward compatible)
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(config.DB_FILE, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA busy_timeout = 5000")
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=FULL")
    return _conn


# =============================================================================
# PROD-005: Schema Versioning & Migration Framework
# =============================================================================

# Current schema version — bump this when adding a new migration
CURRENT_SCHEMA_VERSION = 6


def _ensure_schema_version_table(conn: sqlite3.Connection):
    """Create the schema_version table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL,
            description TEXT
        )
    """)
    conn.commit()


def _get_schema_version(conn: sqlite3.Connection) -> int:
    """Get the current schema version from the database. Returns 0 if no migrations applied."""
    _ensure_schema_version_table(conn)
    row = conn.execute("SELECT MAX(version) as v FROM schema_version").fetchone()
    return row["v"] if row and row["v"] is not None else 0


def _record_migration(conn: sqlite3.Connection, version: int, description: str):
    """Record that a migration was applied."""
    conn.execute(
        "INSERT INTO schema_version (version, applied_at, description) VALUES (?, ?, ?)",
        (version, datetime.now(config.ET).isoformat(), description),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Migration functions — add new migrations here as `_migrate_N(conn)`
# ---------------------------------------------------------------------------

def _migrate_1(conn: sqlite3.Connection):
    """Migration 1: Add schema_version tracking (baseline).

    This is the initial migration that establishes the versioning system.
    All existing tables are considered part of version 0 (pre-versioning).
    """
    # No schema changes needed — the schema_version table is created by
    # _ensure_schema_version_table(). This migration just marks the baseline.
    logger.info("PROD-005: Migration 1 applied — schema versioning baseline established")


def _migrate_2(conn: sqlite3.Connection):
    """Migration 2: T1-008 — Add overnight_hold column to open_positions.

    Tracks which positions are designated as overnight holds so the EOD
    close logic can survive bot restarts without losing the hold registry.
    """
    cursor = conn.execute("PRAGMA table_info(open_positions)")
    existing_cols = {row["name"] for row in cursor.fetchall()}
    if "overnight_hold" not in existing_cols:
        conn.execute("ALTER TABLE open_positions ADD COLUMN overnight_hold INTEGER DEFAULT 0")
        logger.info("T1-008: Added overnight_hold column to open_positions")


def _migrate_3(conn: sqlite3.Connection):
    """Migration 3: QW-009 — Add composite index on (symbol, created_at) for trades table.

    Speeds up per-symbol time-range queries used by analytics and dashboards.
    Also adds (symbol, entry_time) index on open_positions.
    """
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol_exit_time ON trades(symbol, exit_time)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals(symbol, timestamp)")
    logger.info("QW-009: Added composite indexes for (symbol, time) queries")


def _migrate_4(conn: sqlite3.Connection):
    """Migration 4: T4-005 — Add indexes for hot query columns.

    Speeds up frequently-run queries during market hours:
    - trades(symbol, exit_time) — already covered by migration 3 as (symbol, exit_time)
    - signals(strategy, timestamp) — strategy + time range queries
    - execution_analytics(symbol, submitted_at) — fill analytics lookups
    - open_positions(symbol) — position lookups (already PK, but add status-aware index pattern)
    - event_log(event_type, timestamp) — audit log queries

    SQLite does not support partial indexes with WHERE clauses on all versions,
    so we use full indexes.
    """
    conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_strategy_ts ON signals(strategy, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_symbol_time ON execution_analytics(symbol, submitted_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type_ts ON event_log(event_type, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_exit_reason ON trades(exit_reason)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shadow_exit_time ON shadow_trades(exit_time)")
    logger.info("T4-005: Added hot query indexes for signals, execution_analytics, event_log, trades")


def _migrate_5(conn: sqlite3.Connection):
    """Migration 5: T2-006 — Create audit_log table for V11.2 compliance.

    Stores structured audit entries for order submissions, risk decisions,
    circuit breaker trips, and system events.
    """
    # audit_log may already exist from V11.1 compliance module with a different schema.
    # Add missing columns if they don't exist (ALTER TABLE ADD COLUMN is idempotent-safe).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            actor TEXT NOT NULL DEFAULT 'system',
            payload_json TEXT NOT NULL DEFAULT '{}',
            signature_hash TEXT NOT NULL DEFAULT ''
        )
    """)
    # Add V11.2 columns to existing table (ignore if already present)
    for col_def in [
        "source TEXT DEFAULT ''",
        "symbol TEXT",
        "strategy TEXT",
        "details TEXT",
        "severity TEXT DEFAULT 'INFO'",
        "session_id TEXT",
    ]:
        try:
            conn.execute(f"ALTER TABLE audit_log ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log(event_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_symbol ON audit_log(symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_severity ON audit_log(severity)")
    logger.info("T2-006: audit_log table ready with all indexes")


def _migrate_6(conn: sqlite3.Connection):
    """Migration 6: T2-006 — Add overnight_hold and partial tracking indexes.

    Adds performance indexes for overnight position queries and
    partial exit tracking that were identified as slow in V11.1 profiling.
    """
    # Index for overnight hold position queries
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_open_positions_overnight "
        "ON open_positions(overnight_hold)"
    )
    # Index for partial exit queries on trades
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_trades_strategy_exit_reason "
        "ON trades(strategy, exit_reason)"
    )
    logger.info("T2-006: Added overnight and partial-exit performance indexes")


# Registry of all migrations: version -> (function, description)
_MIGRATIONS: dict[int, tuple] = {
    1: (_migrate_1, "Baseline schema versioning"),
    2: (_migrate_2, "T1-008: Add overnight_hold column to open_positions"),
    3: (_migrate_3, "QW-009: Add composite indexes for (symbol, time) queries"),
    4: (_migrate_4, "T4-005: Add hot query indexes for market-hours performance"),
    5: (_migrate_5, "T2-006: Create audit_log table for V11.2 compliance"),
    6: (_migrate_6, "T2-006: Add overnight and partial-exit performance indexes"),
}


def run_migrations():
    """PROD-005: Check schema version and auto-run any pending migrations.

    Call this after init_db() during startup. Migrations run in order and
    each is recorded in the schema_version table.
    """
    conn = _get_conn()
    _ensure_schema_version_table(conn)
    current = _get_schema_version(conn)

    if current >= CURRENT_SCHEMA_VERSION:
        logger.debug("PROD-005: Schema is up to date (version=%d)", current)
        return

    pending = {v: m for v, m in _MIGRATIONS.items() if v > current}
    if not pending:
        return

    logger.info(
        "PROD-005: Running %d pending migration(s) (current=%d, target=%d)",
        len(pending), current, CURRENT_SCHEMA_VERSION,
    )

    for version in sorted(pending.keys()):
        migrate_fn, description = pending[version]
        try:
            logger.info("PROD-005: Applying migration %d: %s", version, description)
            migrate_fn(conn)
            _record_migration(conn, version, description)
            logger.info("PROD-005: Migration %d applied successfully", version)
        except Exception as e:
            logger.error("PROD-005: Migration %d failed: %s", version, e)
            raise RuntimeError(
                f"Schema migration {version} failed: {e}. "
                f"Database may be in an inconsistent state."
            ) from e

    final = _get_schema_version(conn)
    logger.info("PROD-005: All migrations complete (version=%d)", final)


def assert_schema_version():
    """T2-006: Startup assertion — verify schema version matches expected.

    Call after init_db() + run_migrations(). Raises RuntimeError if the
    database schema version does not match CURRENT_SCHEMA_VERSION, which
    would indicate a failed migration or database tampering.
    """
    conn = _get_conn()
    actual = _get_schema_version(conn)
    if actual != CURRENT_SCHEMA_VERSION:
        raise RuntimeError(
            f"T2-006: Schema version mismatch! "
            f"Expected {CURRENT_SCHEMA_VERSION}, got {actual}. "
            f"Run migrations or check database integrity."
        )
    logger.info("T2-006: Schema version assertion passed (version=%d)", actual)


def init_db():
    """Create all tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL,
            exit_price REAL,
            qty REAL,
            entry_time TEXT,
            exit_time TEXT,
            exit_reason TEXT,
            pnl REAL,
            pnl_pct REAL
        );

        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            acted_on INTEGER DEFAULT 0,
            skip_reason TEXT
        );

        CREATE TABLE IF NOT EXISTS open_positions (
            symbol TEXT PRIMARY KEY,
            strategy TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL,
            qty REAL,
            entry_time TEXT,
            take_profit REAL,
            stop_loss REAL,
            alpaca_order_id TEXT,
            hold_type TEXT DEFAULT 'day',
            time_stop TEXT,
            max_hold_date TEXT,
            pair_id TEXT DEFAULT '',
            partial_exits INTEGER DEFAULT 0,
            highest_price_seen REAL DEFAULT 0.0,
            lowest_price_seen REAL DEFAULT 0.0,
            entry_atr REAL DEFAULT 0.0,
            overnight_hold INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS daily_snapshots (
            date TEXT PRIMARY KEY,
            portfolio_value REAL,
            cash REAL,
            day_pnl REAL,
            day_pnl_pct REAL,
            total_trades INTEGER,
            win_rate REAL,
            sharpe_rolling REAL
        );

        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT,
            strategy TEXT,
            total_return REAL,
            annualized_return REAL,
            sharpe_ratio REAL,
            win_rate REAL,
            profit_factor REAL,
            max_drawdown REAL,
            total_trades INTEGER,
            avg_hold_minutes REAL
        );

        -- V3: ML model performance history
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            strategy TEXT NOT NULL,
            train_samples INTEGER,
            test_precision REAL,
            test_recall REAL,
            test_f1 REAL,
            features_used TEXT,
            model_version TEXT
        );

        -- V3: Parameter optimization history
        CREATE TABLE IF NOT EXISTS optimization_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            strategy TEXT NOT NULL,
            old_params TEXT,
            new_params TEXT,
            old_sharpe REAL,
            new_sharpe REAL,
            applied INTEGER DEFAULT 0
        );

        -- V3: Capital allocation weight history
        CREATE TABLE IF NOT EXISTS allocation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            weights TEXT NOT NULL
        );

        -- V5: Shadow trades (paper-simulated trades for strategy evaluation)
        CREATE TABLE IF NOT EXISTS shadow_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL,
            qty REAL,
            entry_time TEXT,
            take_profit REAL,
            stop_loss REAL,
            time_stop TEXT,
            exit_price REAL,
            exit_time TEXT,
            exit_reason TEXT,
            pnl REAL,
            pnl_pct REAL
        );

        CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
        CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);
        CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
        CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy);
        CREATE INDEX IF NOT EXISTS idx_signals_acted ON signals(acted_on);
        CREATE INDEX IF NOT EXISTS idx_shadow_strategy ON shadow_trades(strategy);
        CREATE INDEX IF NOT EXISTS idx_shadow_entry_time ON shadow_trades(entry_time);

        -- V6: OU process parameters for mean reversion universe
        CREATE TABLE IF NOT EXISTS ou_parameters (
            symbol TEXT,
            date TEXT,
            kappa REAL,
            mu REAL,
            sigma REAL,
            half_life REAL,
            hurst REAL,
            adf_pvalue REAL,
            PRIMARY KEY (symbol, date)
        );

        -- V6: Kalman pairs for pairs trading
        CREATE TABLE IF NOT EXISTS kalman_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol1 TEXT NOT NULL,
            symbol2 TEXT NOT NULL,
            hedge_ratio REAL,
            spread_mean REAL,
            spread_std REAL,
            correlation REAL,
            coint_pvalue REAL,
            half_life REAL,
            sector_group TEXT,
            active INTEGER DEFAULT 1,
            last_updated TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_kalman_active ON kalman_pairs(active);

        -- V6: Daily consistency metrics
        CREATE TABLE IF NOT EXISTS consistency_log (
            date TEXT PRIMARY KEY,
            consistency_score REAL,
            pct_positive_days REAL,
            sharpe REAL,
            max_drawdown REAL,
            vol_scalar_avg REAL,
            beta_avg REAL
        );

        -- V8: Kelly criterion parameters
        CREATE TABLE IF NOT EXISTS kelly_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL,
            win_rate REAL,
            avg_win_loss REAL,
            kelly_f REAL,
            half_kelly_f REAL,
            sample_size INTEGER,
            computed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_kelly_strategy ON kelly_params(strategy);

        -- V8: Monte Carlo tail risk results
        CREATE TABLE IF NOT EXISTS monte_carlo_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            var_95 REAL,
            var_99 REAL,
            cvar_95 REAL,
            cvar_99 REAL,
            horizon_days INTEGER,
            simulations INTEGER,
            computed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_mc_date ON monte_carlo_results(date);

        -- V8: Execution analytics
        CREATE TABLE IF NOT EXISTS execution_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT,
            symbol TEXT,
            strategy TEXT,
            side TEXT,
            expected_price REAL,
            filled_price REAL,
            slippage_pct REAL,
            submitted_at TEXT,
            filled_at TEXT,
            latency_ms INTEGER,
            qty_requested INTEGER,
            qty_filled INTEGER,
            fill_rate REAL
        );
        CREATE INDEX IF NOT EXISTS idx_exec_strategy ON execution_analytics(strategy);

        -- V10: Structured event log for audit trail
        CREATE TABLE IF NOT EXISTS event_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            source TEXT,
            symbol TEXT,
            strategy TEXT,
            details TEXT,
            severity TEXT DEFAULT 'INFO'
        );
        CREATE INDEX IF NOT EXISTS idx_event_ts ON event_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_event_type ON event_log(event_type);

        -- V11.2 T5-010: SEC 17a-4 inspired compliance audit log
        -- Append-only (no UPDATE/DELETE) with SHA-256 hash chain for tamper detection
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            actor TEXT NOT NULL DEFAULT 'system',
            payload_json TEXT NOT NULL,
            signature_hash TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log(event_type);
        CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log(actor);
    """)
    conn.commit()

    # V4+V10: Add new columns to open_positions if they don't exist (migration)
    # CRIT-016: Use BEGIN IMMEDIATE instead of BEGIN EXCLUSIVE to avoid deadlock
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute("PRAGMA table_info(open_positions)")
        existing_cols = {row["name"] for row in cursor.fetchall()}
        v4_cols = {
            "pair_id": "TEXT DEFAULT ''",
            "partial_exits": "INTEGER DEFAULT 0",
            "highest_price_seen": "REAL DEFAULT 0.0",
            "lowest_price_seen": "REAL DEFAULT 0.0",
            "entry_atr": "REAL DEFAULT 0.0",
        }
        for col, col_type in v4_cols.items():
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE open_positions ADD COLUMN {col} {col_type}")
                logger.info(f"Added column {col} to open_positions")
        conn.execute("COMMIT")
    except Exception as e:
        conn.execute("ROLLBACK")
        logger.warning(f"V4 schema migration note: {e}")

    # V10: Migrate event_log if old schema (from OMS SQLAlchemy) exists
    try:
        cursor = conn.execute("PRAGMA table_info(event_log)")
        existing_cols = {row["name"] for row in cursor.fetchall()}
        if "symbol" not in existing_cols:
            # Old schema had: id, timestamp, event_type, source, data_json
            # New schema adds: symbol, strategy, details, severity
            logger.info("Migrating event_log table to V10 schema...")
            conn.execute("DROP TABLE IF EXISTS event_log")
            conn.execute("""
                CREATE TABLE event_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    source TEXT,
                    symbol TEXT,
                    strategy TEXT,
                    details TEXT,
                    severity TEXT DEFAULT 'INFO'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_ts ON event_log(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON event_log(event_type)")
            conn.commit()
            logger.info("event_log table migrated to V10 schema")
    except Exception as e:
        logger.warning(f"event_log migration note: {e}")

    logger.info("Database initialized")


# --- Trade Logging ---

def log_trade(symbol: str, strategy: str, side: str, entry_price: float,
              exit_price: float, qty: float, entry_time: datetime,
              exit_time: datetime, exit_reason: str, pnl: float, pnl_pct: float):
    """Log a completed trade."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO trades (symbol, strategy, side, entry_price, exit_price,
           qty, entry_time, exit_time, exit_reason, pnl, pnl_pct)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol, strategy, side, entry_price, exit_price, qty,
         _to_iso(entry_time), _to_iso(exit_time), exit_reason, pnl, pnl_pct),
    )
    conn.commit()


# --- Signal Logging ---

def log_signal(timestamp: datetime, symbol: str, strategy: str,
               signal_type: str, acted_on: bool, skip_reason: str = ""):
    """Log a signal (whether or not it was acted on)."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO signals (timestamp, symbol, strategy, signal_type,
           acted_on, skip_reason) VALUES (?, ?, ?, ?, ?, ?)""",
        (_to_iso(timestamp), symbol, strategy, signal_type,
         1 if acted_on else 0, skip_reason),
    )
    conn.commit()


# --- Open Positions (replaces state.json) ---

def save_open_positions(open_trades: dict):
    """Replace all open_positions rows with current state.

    V10 BUG-029: Atomic transaction — if crash occurs mid-save, all positions are preserved.
    Thread-safe via _db_lock.
    """
    conn = _get_conn()
    with _db_lock:
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM open_positions")
            for symbol, trade in open_trades.items():
                conn.execute(
                    """INSERT INTO open_positions (symbol, strategy, side, entry_price,
                       qty, entry_time, take_profit, stop_loss, alpaca_order_id,
                       hold_type, time_stop, max_hold_date,
                       pair_id, partial_exits, highest_price_seen, lowest_price_seen, entry_atr,
                       overnight_hold)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (symbol, trade.strategy, trade.side, trade.entry_price,
                     trade.qty, _to_iso(trade.entry_time), trade.take_profit,
                     trade.stop_loss, trade.order_id,
                     getattr(trade, 'hold_type', 'day'),
                     _to_iso(trade.time_stop),
                     _to_iso(getattr(trade, 'max_hold_date', None)),
                     getattr(trade, 'pair_id', ''),
                     getattr(trade, 'partial_exits', 0),
                     getattr(trade, 'highest_price_seen', 0.0),
                     getattr(trade, 'lowest_price_seen', 0.0),
                     getattr(trade, 'entry_atr', 0.0),
                     1 if getattr(trade, 'overnight_hold', False) else 0),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def load_open_positions() -> list[dict]:
    """Load open positions from DB. Returns list of dicts."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM open_positions").fetchall()
    return [dict(row) for row in rows]


# --- T1-008: Overnight Hold Persistence ---

def save_overnight_holds(symbols: set[str]):
    """T1-008: Mark symbols as overnight holds in open_positions.

    Persists the overnight hold flag so it survives bot restarts.
    Only marks symbols that currently exist in open_positions.
    """
    if not symbols:
        return
    conn = _get_conn()
    with _db_lock:
        # Reset all, then set the selected ones
        conn.execute("UPDATE open_positions SET overnight_hold = 0")
        for sym in symbols:
            conn.execute(
                "UPDATE open_positions SET overnight_hold = 1 WHERE symbol = ?",
                (sym,),
            )
        conn.commit()
    logger.info("T1-008: Persisted %d overnight hold symbols to DB", len(symbols))


def load_overnight_holds() -> set[str]:
    """T1-008: Load overnight hold symbols from DB.

    Called on startup to restore the hold registry after a restart.
    """
    conn = _get_conn()
    rows = conn.execute(
        "SELECT symbol FROM open_positions WHERE overnight_hold = 1"
    ).fetchall()
    symbols = {row["symbol"] for row in rows}
    if symbols:
        logger.info("T1-008: Loaded %d overnight hold symbols from DB: %s", len(symbols), list(symbols))
    return symbols


# --- Daily Snapshots ---

def save_daily_snapshot(date: str, portfolio_value: float, cash: float,
                        day_pnl: float, day_pnl_pct: float,
                        total_trades: int, win_rate: float, sharpe_rolling: float):
    """Insert or replace daily snapshot."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO daily_snapshots
           (date, portfolio_value, cash, day_pnl, day_pnl_pct,
            total_trades, win_rate, sharpe_rolling)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (date, portfolio_value, cash, day_pnl, day_pnl_pct,
         total_trades, win_rate, sharpe_rolling),
    )
    conn.commit()


# --- Analytics Queries ---

def get_recent_trades(days: int = 7) -> list[dict]:
    """Get trades from the last N days."""
    conn = _get_conn()
    cutoff = (datetime.now(config.ET) - __import__('datetime').timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT * FROM trades WHERE exit_time >= ? ORDER BY exit_time DESC",
        (cutoff,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_all_trades() -> list[dict]:
    """Get all trades ever."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM trades ORDER BY exit_time DESC").fetchall()
    return [dict(row) for row in rows]


def get_daily_snapshots(days: int = 30) -> list[dict]:
    """Get daily snapshots for the last N days."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM daily_snapshots ORDER BY date DESC LIMIT ?",
        (days,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_daily_pnl_series(days: int = 30) -> list[float]:
    """Get list of daily P&L percentages for analytics."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT day_pnl_pct FROM daily_snapshots ORDER BY date DESC LIMIT ?",
        (days,),
    ).fetchall()
    return [row["day_pnl_pct"] for row in reversed(rows)]


def get_portfolio_values(days: int = 30) -> list[float]:
    """Get list of portfolio values for drawdown calculation."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT portfolio_value FROM daily_snapshots ORDER BY date DESC LIMIT ?",
        (days,),
    ).fetchall()
    return [row["portfolio_value"] for row in reversed(rows)]


def get_signal_stats_today() -> dict:
    """Get today's signal statistics."""
    conn = _get_conn()
    today = datetime.now(config.ET).strftime("%Y-%m-%d")
    total = conn.execute(
        "SELECT COUNT(*) as cnt FROM signals WHERE timestamp LIKE ?",
        (f"{today}%",),
    ).fetchone()["cnt"]
    acted = conn.execute(
        "SELECT COUNT(*) as cnt FROM signals WHERE timestamp LIKE ? AND acted_on = 1",
        (f"{today}%",),
    ).fetchone()["cnt"]
    return {"total": total, "acted": acted, "skipped": total - acted}


# --- Backtest Results ---

def save_backtest_result(run_date: str, strategy: str, total_return: float,
                         annualized_return: float, sharpe_ratio: float,
                         win_rate: float, profit_factor: float,
                         max_drawdown: float, total_trades: int,
                         avg_hold_minutes: float):
    """Save backtest results."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO backtest_results
           (run_date, strategy, total_return, annualized_return, sharpe_ratio,
            win_rate, profit_factor, max_drawdown, total_trades, avg_hold_minutes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_date, strategy, total_return, annualized_return, sharpe_ratio,
         win_rate, profit_factor, max_drawdown, total_trades, avg_hold_minutes),
    )
    conn.commit()


# --- Migration ---

def migrate_from_json():
    """One-time migration from state.json to SQLite."""
    json_path = Path(config.STATE_FILE)
    if not json_path.exists():
        return

    try:
        data = json.loads(json_path.read_text())
        # Migrate open trades
        for symbol, td in data.get("open_trades", {}).items():
            conn = _get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO open_positions
                   (symbol, strategy, side, entry_price, qty, entry_time,
                    take_profit, stop_loss, alpaca_order_id, hold_type, time_stop)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, td["strategy"], td["side"], td["entry_price"],
                 td["qty"], td["entry_time"], td["take_profit"],
                 td["stop_loss"], td.get("order_id", ""), "day",
                 td.get("time_stop")),
            )
        conn = _get_conn()
        conn.commit()

        # Rename json file so migration doesn't run again
        backup = json_path.with_suffix(".json.bak")
        json_path.rename(backup)
        logger.info(f"Migrated state.json to SQLite, backup at {backup}")

    except Exception as e:
        logger.error(f"Migration from state.json failed: {e}")


# =============================================================================
# V3 ADDITIONS
# =============================================================================

# --- Strategy-specific trade queries ---

def get_recent_trades_by_strategy(strategy: str, days: int = 20) -> list[dict]:
    """Get trades for a specific strategy from the last N days."""
    conn = _get_conn()
    cutoff = (datetime.now(config.ET) - __import__('datetime').timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT * FROM trades WHERE strategy = ? AND exit_time >= ? ORDER BY exit_time",
        (strategy, cutoff),
    ).fetchall()
    return [dict(row) for row in rows]


def get_signals_with_outcomes() -> list[dict]:
    """JOIN signals that were acted on with their trade outcomes for ML training.

    Returns rows with signal fields + trade pnl/pnl_pct (if trade exists).
    Only returns signals that were acted on (acted_on=1) and have a matching trade.
    """
    conn = _get_conn()
    rows = conn.execute("""
        SELECT
            s.timestamp, s.symbol, s.strategy, s.signal_type,
            t.entry_price, t.exit_price, t.qty,
            t.entry_time, t.exit_time, t.exit_reason,
            t.pnl, t.pnl_pct,
            CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END as profitable
        FROM signals s
        INNER JOIN trades t
            ON s.symbol = t.symbol
            AND s.strategy = t.strategy
            AND s.acted_on = 1
            AND ABS(julianday(s.timestamp) - julianday(t.entry_time)) < 0.01
        ORDER BY s.timestamp
    """).fetchall()
    return [dict(row) for row in rows]


def get_trade_count_by_strategy(strategy: str) -> int:
    """Get total number of completed trades for a strategy."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM trades WHERE strategy = ?",
        (strategy,),
    ).fetchone()
    return row["cnt"]


# --- ML Model Performance ---

def log_model_performance(strategy: str, train_samples: int,
                          test_precision: float, test_recall: float,
                          test_f1: float, features_used: list[str],
                          model_version: str):
    """Log ML model training results."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO model_performance
           (timestamp, strategy, train_samples, test_precision, test_recall,
            test_f1, features_used, model_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now(config.ET).isoformat(), strategy, train_samples,
         test_precision, test_recall, test_f1,
         json.dumps(features_used), model_version),
    )
    conn.commit()


# --- Optimization History ---

def log_optimization(strategy: str, old_params: dict, new_params: dict,
                     old_sharpe: float, new_sharpe: float, applied: bool):
    """Log a parameter optimization run."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO optimization_history
           (timestamp, strategy, old_params, new_params, old_sharpe, new_sharpe, applied)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now(config.ET).isoformat(), strategy,
         json.dumps(old_params), json.dumps(new_params),
         old_sharpe, new_sharpe, 1 if applied else 0),
    )
    conn.commit()


# --- Allocation History ---

def log_allocation_weights(weights: dict):
    """Log capital allocation weight change."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO allocation_history (timestamp, weights) VALUES (?, ?)",
        (datetime.now(config.ET).isoformat(), json.dumps(weights)),
    )
    conn.commit()


# --- Signal Statistics (extended for web dashboard) ---

def get_signals_by_date(date_str: str) -> list[dict]:
    """Get all signals for a specific date."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM signals WHERE timestamp LIKE ? ORDER BY timestamp",
        (f"{date_str}%",),
    ).fetchall()
    return [dict(row) for row in rows]


def get_signal_skip_reasons(days: int = 7) -> dict:
    """Get breakdown of signal skip reasons over last N days."""
    conn = _get_conn()
    cutoff = (datetime.now(config.ET) - __import__('datetime').timedelta(days=days)).isoformat()
    rows = conn.execute(
        """SELECT skip_reason, COUNT(*) as cnt
           FROM signals
           WHERE timestamp >= ? AND acted_on = 0 AND skip_reason != ''
           GROUP BY skip_reason
           ORDER BY cnt DESC""",
        (cutoff,),
    ).fetchall()
    return {row["skip_reason"]: row["cnt"] for row in rows}


def get_trades_paginated(limit: int = 100, offset: int = 0,
                         strategy: str | None = None) -> list[dict]:
    """Get trades with pagination and optional strategy filter."""
    conn = _get_conn()
    if strategy:
        rows = conn.execute(
            "SELECT * FROM trades WHERE strategy = ? ORDER BY exit_time DESC LIMIT ? OFFSET ?",
            (strategy, limit, offset),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY exit_time DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [dict(row) for row in rows]


# =============================================================================
# V5 ADDITIONS — Shadow Trades
# =============================================================================

def log_shadow_entry(symbol, strategy, side, entry_price, qty, entry_time,
                     take_profit, stop_loss, time_stop=None):
    """Record a shadow trade entry (no real order submitted)."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO shadow_trades
           (symbol, strategy, side, entry_price, qty, entry_time,
            take_profit, stop_loss, time_stop)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol, strategy, side, entry_price, qty,
         entry_time.isoformat() if hasattr(entry_time, 'isoformat') else str(entry_time),
         take_profit, stop_loss,
         time_stop.isoformat() if time_stop and hasattr(time_stop, 'isoformat') else str(time_stop) if time_stop else None),
    )
    conn.commit()


def get_open_shadow_trades():
    """Get all shadow trades that haven't been closed yet."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, symbol, strategy, side, entry_price, qty, entry_time, "
        "take_profit, stop_loss, time_stop "
        "FROM shadow_trades WHERE exit_time IS NULL"
    ).fetchall()
    cols = ["id", "symbol", "strategy", "side", "entry_price", "qty",
            "entry_time", "take_profit", "stop_loss", "time_stop"]
    return [dict(zip(cols, row)) for row in rows]


def close_shadow_trade(trade_id, exit_price, exit_time, exit_reason):
    """Close a shadow trade with simulated exit.

    HIGH-013: Lock covers full read-modify-write cycle.
    """
    conn = _get_conn()
    with _db_lock:
        row = conn.execute(
            "SELECT entry_price, qty, side FROM shadow_trades WHERE id = ?",
            (trade_id,)
        ).fetchone()
        if not row:
            return
        entry_price, qty, side = row["entry_price"], row["qty"], row["side"]
        if side == "buy":
            pnl = (exit_price - entry_price) * qty
            pnl_pct = (exit_price - entry_price) / entry_price if entry_price else 0
        else:
            pnl = (entry_price - exit_price) * qty
            pnl_pct = (entry_price - exit_price) / entry_price if entry_price else 0
        conn.execute(
            """UPDATE shadow_trades
               SET exit_price = ?, exit_time = ?, exit_reason = ?, pnl = ?, pnl_pct = ?
               WHERE id = ?""",
            (exit_price,
             exit_time.isoformat() if hasattr(exit_time, 'isoformat') else str(exit_time),
             exit_reason, round(pnl, 2), round(pnl_pct, 4), trade_id),
        )
        conn.commit()


def get_shadow_performance(strategy=None, days=14):
    """Get shadow trade performance stats."""
    conn = _get_conn()
    query = """SELECT strategy, COUNT(*) as trades,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               SUM(pnl) as total_pnl, AVG(pnl_pct) as avg_pnl_pct
               FROM shadow_trades WHERE exit_time IS NOT NULL
               AND entry_time >= datetime('now', ?)"""
    params = [f"-{days} days"]
    if strategy:
        query += " AND strategy = ?"
        params.append(strategy)
    query += " GROUP BY strategy"
    rows = conn.execute(query, params).fetchall()
    cols = ["strategy", "trades", "wins", "total_pnl", "avg_pnl_pct"]
    return [dict(zip(cols, row)) for row in rows]


# =============================================================================
# V5 ADDITIONS — Trade Analysis
# =============================================================================

def get_exit_reason_breakdown(days=7):
    """Return exit reason breakdown for recent trades."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT exit_reason, COUNT(*) as count,
           AVG(pnl) as avg_pnl, AVG(pnl_pct) as avg_pnl_pct
           FROM trades
           WHERE exit_time >= datetime('now', ?)
           AND exit_reason IS NOT NULL
           GROUP BY exit_reason
           ORDER BY count DESC""",
        (f"-{days} days",)
    ).fetchall()
    cols = ["exit_reason", "count", "avg_pnl", "avg_pnl_pct"]
    return [dict(zip(cols, row)) for row in rows]


def get_filter_block_summary(date=None):
    """Return skip reason counts for signals."""
    conn = _get_conn()
    if date is None:
        query = """SELECT skip_reason, COUNT(*) as count
                   FROM signals WHERE acted_on = 0 AND skip_reason IS NOT NULL
                   AND timestamp >= datetime('now', '-1 day')
                   GROUP BY skip_reason ORDER BY count DESC"""
        rows = conn.execute(query).fetchall()
    else:
        query = """SELECT skip_reason, COUNT(*) as count
                   FROM signals WHERE acted_on = 0 AND skip_reason IS NOT NULL
                   AND DATE(timestamp) = ?
                   GROUP BY skip_reason ORDER BY count DESC"""
        rows = conn.execute(query, (str(date),)).fetchall()
    return {row["skip_reason"]: row["count"] for row in rows}


# =============================================================================
# --- OU Parameters ---
# =============================================================================

def save_ou_parameters(symbol: str, date: str, kappa: float, mu: float,
                       sigma: float, half_life: float, hurst: float,
                       adf_pvalue: float):
    """Save OU process parameters for a symbol."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO ou_parameters
           (symbol, date, kappa, mu, sigma, half_life, hurst, adf_pvalue)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol, date, kappa, mu, sigma, half_life, hurst, adf_pvalue),
    )
    conn.commit()


def get_ou_parameters(symbol: str, date: str) -> dict | None:
    """Get OU parameters for a symbol on a specific date."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM ou_parameters WHERE symbol = ? AND date = ?",
        (symbol, date),
    ).fetchone()
    return dict(row) if row else None


def get_mr_universe(date: str) -> list[dict]:
    """Get all symbols that passed MR universe filter on a date."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM ou_parameters WHERE date = ? ORDER BY hurst ASC",
        (date,),
    ).fetchall()
    return [dict(r) for r in rows]


# =============================================================================
# --- Kalman Pairs ---
# =============================================================================

def save_kalman_pair(symbol1: str, symbol2: str, hedge_ratio: float,
                     spread_mean: float, spread_std: float, correlation: float,
                     coint_pvalue: float, half_life: float, sector_group: str):
    """Save or update a Kalman pair.

    HIGH-013: Lock covers full read-modify-write cycle to prevent races.
    """
    conn = _get_conn()
    with _db_lock:
        # Check if pair exists
        existing = conn.execute(
            "SELECT id FROM kalman_pairs WHERE symbol1 = ? AND symbol2 = ? AND active = 1",
            (symbol1, symbol2),
        ).fetchone()

        now_str = _to_iso(datetime.now(config.ET))  # V10 BUG-044: timezone-aware

        if existing:
            conn.execute(
                """UPDATE kalman_pairs SET hedge_ratio=?, spread_mean=?, spread_std=?,
                   correlation=?, coint_pvalue=?, half_life=?, sector_group=?,
                   last_updated=? WHERE id=?""",
                (hedge_ratio, spread_mean, spread_std, correlation, coint_pvalue,
                 half_life, sector_group, now_str, existing['id']),
            )
        else:
            conn.execute(
                """INSERT INTO kalman_pairs
                   (symbol1, symbol2, hedge_ratio, spread_mean, spread_std,
                    correlation, coint_pvalue, half_life, sector_group, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol1, symbol2, hedge_ratio, spread_mean, spread_std,
                 correlation, coint_pvalue, half_life, sector_group, now_str),
            )
        conn.commit()


def get_active_kalman_pairs() -> list[dict]:
    """Get all active Kalman pairs."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM kalman_pairs WHERE active = 1 ORDER BY coint_pvalue ASC"
    ).fetchall()
    return [dict(r) for r in rows]


def deactivate_kalman_pair(pair_id: int):
    """Deactivate a Kalman pair."""
    conn = _get_conn()
    conn.execute("UPDATE kalman_pairs SET active = 0 WHERE id = ?", (pair_id,))
    conn.commit()


def deactivate_all_kalman_pairs():
    """Deactivate all pairs (before weekly re-selection)."""
    conn = _get_conn()
    conn.execute("UPDATE kalman_pairs SET active = 0")
    conn.commit()


# =============================================================================
# --- Consistency Log ---
# =============================================================================

def save_consistency_log(date: str, consistency_score: float,
                         pct_positive: float, sharpe: float,
                         max_drawdown: float, vol_scalar_avg: float = 0.0,
                         beta_avg: float = 0.0):
    """Save daily consistency metrics."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO consistency_log
           (date, consistency_score, pct_positive_days, sharpe, max_drawdown,
            vol_scalar_avg, beta_avg)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (date, consistency_score, pct_positive, sharpe, max_drawdown,
         vol_scalar_avg, beta_avg),
    )
    conn.commit()


def get_consistency_log(days: int = 30) -> list[dict]:
    """Get recent consistency log entries."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM consistency_log ORDER BY date DESC LIMIT ?",
        (days,),
    ).fetchall()
    return [dict(r) for r in rows]


# --- V7 additions ---

def get_trades_by_strategy(strategy: str, days: int = 30) -> list[dict]:
    """Get closed trades for a specific strategy within the last N days."""
    from datetime import timedelta
    cutoff = _to_iso(datetime.now(config.ET) - timedelta(days=days))  # V10 BUG-043
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM trades WHERE strategy = ? AND exit_time > ? ORDER BY exit_time DESC",
        (strategy, cutoff),
    ).fetchall()
    return [dict(r) for r in rows]


def get_signals_by_strategy(strategy: str, days: int = 7) -> list[dict]:
    """Get signal records for a specific strategy within the last N days."""
    from datetime import timedelta
    cutoff = _to_iso(datetime.now(config.ET) - timedelta(days=days))  # V10 BUG-043
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM signals WHERE strategy = ? AND timestamp > ? ORDER BY timestamp DESC",
        (strategy, cutoff),
    ).fetchall()
    return [dict(r) for r in rows]


# =============================================================================
# --- Kelly Criterion Parameters ---
# =============================================================================

def save_kelly_params(strategy: str, win_rate: float, avg_win_loss: float,
                      kelly_f: float, half_kelly_f: float, sample_size: int):
    """Save Kelly criterion parameters for a strategy."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO kelly_params
           (strategy, win_rate, avg_win_loss, kelly_f, half_kelly_f, sample_size, computed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (strategy, win_rate, avg_win_loss, kelly_f, half_kelly_f, sample_size,
         datetime.now(config.ET).isoformat()),
    )
    conn.commit()


# =============================================================================
# --- Monte Carlo Tail Risk ---
# =============================================================================

def save_monte_carlo_result(date: str, var_95: float, var_99: float,
                            cvar_95: float, cvar_99: float,
                            horizon_days: int, simulations: int):
    """Save Monte Carlo risk analysis results."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO monte_carlo_results
           (date, var_95, var_99, cvar_95, cvar_99, horizon_days, simulations, computed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (date, var_95, var_99, cvar_95, cvar_99, horizon_days, simulations,
         datetime.now(config.ET).isoformat()),
    )
    conn.commit()


# =============================================================================
# --- Execution Analytics ---
# =============================================================================

def save_execution_analytics(order_id: str, symbol: str, strategy: str, side: str,
                             expected_price: float, filled_price: float,
                             slippage_pct: float, submitted_at, filled_at,
                             latency_ms: int, qty_requested: int,
                             qty_filled: int, fill_rate: float):
    """Save execution analytics record."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO execution_analytics
           (order_id, symbol, strategy, side, expected_price, filled_price,
            slippage_pct, submitted_at, filled_at, latency_ms,
            qty_requested, qty_filled, fill_rate)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (order_id, symbol, strategy, side, expected_price, filled_price,
         slippage_pct,
         submitted_at.isoformat() if hasattr(submitted_at, 'isoformat') else str(submitted_at),
         filled_at.isoformat() if hasattr(filled_at, 'isoformat') else str(filled_at),
         latency_ms, qty_requested, qty_filled, fill_rate),
    )
    conn.commit()


def get_execution_analytics(strategy: str | None = None, days: int = 30) -> list[dict]:
    """Get execution analytics records."""
    conn = _get_conn()
    if strategy:
        rows = conn.execute(
            """SELECT * FROM execution_analytics WHERE strategy = ?
               AND submitted_at >= datetime('now', ?) ORDER BY submitted_at DESC""",
            (strategy, f"-{days} days"),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT * FROM execution_analytics
               WHERE submitted_at >= datetime('now', ?) ORDER BY submitted_at DESC""",
            (f"-{days} days",),
        ).fetchall()
    return [dict(r) for r in rows]


# --- V10: Event log ---

def insert_event_log(event_type: str, source: str, symbol: str | None = None,
                     strategy: str | None = None, details: str | None = None,
                     severity: str = "INFO"):
    """Insert a structured event into the event log."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO event_log (timestamp, event_type, source, symbol, strategy, details, severity)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (_to_iso(datetime.now(config.ET)), event_type, source, symbol, strategy, details, severity),
    )
    conn.commit()


def get_event_log(event_type: str | None = None, limit: int = 100) -> list[dict]:
    """Get recent event log entries, optionally filtered by type."""
    conn = _get_conn()
    if event_type:
        rows = conn.execute(
            "SELECT * FROM event_log WHERE event_type = ? ORDER BY timestamp DESC LIMIT ?",
            (event_type, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM event_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# =============================================================================
# MED-038: Batch insert helper (uses executemany for performance)
# =============================================================================

def batch_insert(table: str, columns: list[str], rows: list[tuple]):
    """Insert multiple rows at once using executemany for better performance.

    Args:
        table: Table name.
        columns: List of column names.
        rows: List of tuples, each tuple is one row of values.

    Example:
        batch_insert("trades", ["symbol", "strategy", "pnl"],
                      [("AAPL", "STAT_MR", 42.0), ("MSFT", "VWAP", -10.0)])
    """
    if not rows:
        return
    conn = _get_conn()
    placeholders = ", ".join("?" for _ in columns)
    col_str = ", ".join(columns)
    sql = f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})"
    with _db_lock:
        conn.executemany(sql, rows)
        conn.commit()
    logger.debug(f"Batch inserted {len(rows)} rows into {table}")


def batch_log_trades(trades: list[dict]):
    """MED-038: Batch-insert multiple trade records at once.

    Each dict should have keys: symbol, strategy, side, entry_price, exit_price,
    qty, entry_time, exit_time, exit_reason, pnl, pnl_pct.
    """
    columns = ["symbol", "strategy", "side", "entry_price", "exit_price",
               "qty", "entry_time", "exit_time", "exit_reason", "pnl", "pnl_pct"]
    rows = []
    for t in trades:
        entry_time = t.get("entry_time")
        exit_time = t.get("exit_time")
        rows.append((
            t["symbol"], t["strategy"], t["side"],
            t["entry_price"], t["exit_price"], t["qty"],
            entry_time.isoformat() if hasattr(entry_time, 'isoformat') else str(entry_time),
            exit_time.isoformat() if hasattr(exit_time, 'isoformat') else str(exit_time),
            t.get("exit_reason", ""), t["pnl"], t["pnl_pct"],
        ))
    batch_insert("trades", columns, rows)
