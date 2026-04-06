# Changelog

All notable changes to this project will be documented in this file.

## [12.0.0] - 2026-04-05

### Added
- **ExitOrchestrator** -- Unified 10-priority exit system replacing 3 competing exit managers, with strategy-specific profit tiers
- **Profit Maximizer** -- Adaptive scan frequency, intraday vol regime, win streak tracking, dynamic stop tightening
- **FinBERT Sentiment** -- Local NLP sentiment scoring via ProsusAI/finbert (MIT, runs on CPU)
- **FRED Macro Data** -- Yield curve, credit spreads, unemployment via free FRED API
- **ML Model Monitor** -- Rolling 50-trade accuracy tracking with auto-retrain alerts
- **Data Feed Monitor** -- Alpaca outage detection with backup stop enforcement
- **Gap Analysis** -- Pre-open gap detection for MR/breakout signal boosting
- **Health Alerts** -- 8 system health alert types (DB failure, API rate limit, feed stale, etc.)
- **Daily P&L Reports** -- Automated EOD Telegram summary with per-strategy breakdown
- **Tax-Loss Harvesting** -- Friday EOD scan for tax-loss opportunities with wash sale enforcement
- **Disaster Recovery** -- State recovery on startup, heartbeat monitoring, position reconciliation
- **Position Recovery** -- Broker position adoption on mid-day restart
- **Corporate Actions** -- Stock split/dividend detection every 30 minutes
- **Startup Self-Test** -- Synthetic signal pipeline validation before trading
- **Kill Switch Queue** -- Atomic persistent queue survives crash mid-close
- **Onboarding Scripts** -- `setup.sh`, `status.sh`, `stop.sh` for buyer deployment
- **Tear Sheet Generator** -- Professional HTML performance reports with equity curve and SPY benchmark
- **Commercial Docs** -- DISCLAIMER.md, LICENSE_COMMERCIAL.md, SETUP_GUIDE.md
- 1,900+ tests across 85 test files (up from 911)

### Changed
- **Strategy allocations rebalanced** -- STAT_MR 25% (was 35%), KALMAN_PAIRS 27% (was 20%), PEAD 18% (was 5%), VWAP 13% (was 20%), ORB 12% (was 10%), MICRO_MOM 5% (was 10%)
- **Risk parameters** -- RISK_PER_TRADE_PCT 0.8% (was 0.5%), MAX_POSITION_PCT 8% (was 5%)
- **Pairs entry** -- PAIRS_ZSCORE_ENTRY 1.2 (was 1.5, captures more setups)
- **PEAD surprise** -- MIN_SURPRISE_PCT 2.0% (was 5.0%, academic research threshold)
- **Signal ranker** -- Removed placeholder weights (OBV/seasonality/liquidity), regime boosted to 45%
- **ML features** -- Reduced from 200+ to 50 core features (ML_CORE_FEATURES_ONLY)
- **ML confidence** -- Soft sigmoid gate (was hard reject at 0.35)
- **Breadth filter** -- Gradual 70-100% curve (was 85% cliff with 50% cut)
- **Conviction scoring** -- Floored soft multipliers prevent cascade death
- **VIX scaling** -- Now scales UP in low-vol (was down-only)
- **Position size floor** -- 15% minimum (was 30%, allows deeper drawdown cuts)
- **Intraday controls** -- Widened: 5min -0.8% (was -0.3%), 30min -1.2%, 1hr -1.8%
- **Chase logic** -- 10s/20s/30s schedule (was 15s/30s/45s)
- **Bayesian Kelly** -- Wired into production (was implemented but never called)
- **Slippage model** -- Recalibrated: w_volatility 5.0 (was 15.0), w_market_order 3.0 (was 1.0)
- **Idempotency keys** -- 2-second SHA256 buckets (was 5-second hash collision risk)
- Symbol universe reduced to ~120 (removed illiquid micro-caps)
- Docker container renamed velox-v12, added Alertmanager + backup services
- All version strings updated to V12 across all files

### Fixed
- **Config divergence** -- settings.py is canonical, config.py is passthrough, YAML files synced
- **PEAD_HOLD_DAYS** -- MIN/MAX inversion fixed (was MIN=10, MAX=5)
- **Duplicate systems** -- data_cache, smart_routing, data_quality replaced with shims
- **TWAP partial fills** -- Detects and resolves unhedged partial positions
- **Bracket gap-open** -- 5s poll detects stop-before-entry on overnight gaps
- **OU sigma zero** -- Guards in all strategies prevent division by zero
- **Kill switch exception** -- Fallback direct close if activate() throws
- **VIX spike breaker** -- 20% rise in 15min escalates to ORANGE
- **Variable scope bugs** -- day_pnl_pct, _ctr, tier_name fixed in main.py
- **Earnings filter** -- Fixed `days=2` parameter mismatch (function takes no args)
- **STAT_MR sigma consistency** -- Entry and exit now both use price_sigma
- **STAT_MR z-score stops** -- Added for both long and short trades
- **RSI gate** -- STAT_MR now validates RSI before entry (was computed but not checked)
- **Kalman hedge ratio** -- Clamped to [0.2, 5.0] (was [0.01, 100])
- **Kalman P matrix** -- Pre-update condition check prevents gain divergence
- **Pair correlation guard** -- Closes pair if live correlation drops below 0.3
- **MICRO_MOM volume scaling** -- Price move threshold scales with spike magnitude
- **VPIN hard reject** -- Blocks entries at VPIN > 0.70 (toxic order flow)
- **ORB routing** -- Mid-quote improvement skipped for HIGH urgency breakouts
- **Stale order OMS sync** -- Cancelled orders now update OrderManager state
- **Data quality gate** -- Now fetches actual bars (was checking signal._bars which was never set)
- **Redundant correlation check** -- Removed duplicate, kept V10 batch-aware version

## [10.0.0] - 2026-03-19

### Added
- **Order Management System (OMS)** -- 7-state order lifecycle (PENDING -> SUBMITTED -> FILLED/CANCELLED/REJECTED), thread-safe registry with idempotency keys, kill switch for emergency halt
- **Tiered Circuit Breaker** -- 4-tier progressive risk reduction: Yellow (-1%), Orange (-2%), Red (-3%), Black (-4% kill switch)
- **VaR Monitor** -- Real-time parametric, historical, and Monte Carlo Value-at-Risk at 95%/99%
- **Correlation Limiter** -- Eigenvalue-based effective bets calculation, sector Herfindahl concentration limits
- **Transaction Cost Model** -- Pre-trade expected value check rejects negative-EV signals (spread + slippage + commission)
- **Event Bus** -- Pub/sub decoupling for signals, orders, positions, circuit breaker, and regime change events
- **Structured Logging** -- structlog with JSON output in production, human-readable in development
- **Prometheus Metrics** -- Counters, gauges, and histograms for positions, P&L, latency, signals, circuit breaker state
- **PostgreSQL Support** -- SQLAlchemy dual-backend (SQLite for dev/tests, PostgreSQL for production) with Alembic migrations
- **JWT Authentication** -- Optional dashboard auth with token-based access control
- **Docker Production Stack** -- 4-service compose: Velox, PostgreSQL 16, Prometheus, Grafana
- **Dynamic Universe Selection** -- Regime-adaptive daily symbol filtering by volume, market cap, and volatility
- **Apple-Style Dashboard** -- Frosted glass UI with live Alpaca account data, open positions, strategy color pills
- **PEAD Strategy** -- Post-earnings announcement drift (optional)
- 911 unit tests (up from 196)

### Changed
- **main.py decomposed** -- 2329 -> 1512 lines (-35%), extracted into `engine/` package (10 modules, 1847 lines)
- **Dashboard stats** -- Now pulls live equity, day P&L, and positions directly from Alpaca API
- **Dashboard positions** -- Shows real-time broker positions with unrealized P&L instead of stale DB data
- **Signal pipeline** -- Signals now pass through OMS, transaction cost filter, VaR multiplier, and correlation limiter
- **Daily snapshots** -- Fallback to trade-derived daily returns when snapshot data is insufficient
- All 25 dependencies pinned to exact versions (==)
- Replaced `pickle` with `joblib` for model persistence (security)
- Replaced custom RSI with `pandas_ta` library

### Fixed
- **47 bugs** from V10 audit fixed across exit manager, Kalman pairs, StatMR, circuit breaker, broker sync, database, and execution modules
- **analytics module shadowing** -- `analytics.py` moved into `analytics/performance.py` to resolve package/file conflict
- **Dashboard showing zeros** -- Fixed auth blocking (empty DASHBOARD_SECRET_KEY), bot.db volume mount, analytics import chain
- **Test date sensitivity** -- Fixed 6 test failures caused by hardcoded dates falling outside query windows
- **Rate limiter breaking tests** -- Skip rate limiting when PYTEST_CURRENT_TEST is set
- Removed `.DS_Store` and `bot.db` from git tracking

### Removed
- Dead code: `news_filter.py`, `strategies/archive/`, legacy config flags, static beta tables
- Dead strategy routing in execution.py (MOMENTUM, GAP_GO, SECTOR_ROTATION)
- Junk files: `=0.49.0`, `strategies/=0.9.7`, `state.json.bak`

## [7.0.0] - 2026-03-14

### Bug Fixes
- **Broker sync 0 P&L** -- `sync_positions_with_broker()` now fetches market price via `get_snapshot()` instead of using entry_price as exit_price
- **StatMR never firing** -- Added startup initialization for `prepare_universe()` + switched to 2-min intraday bars for OU fitting (correct half-life conversion)
- **KalmanPairs never initializing** -- Loads existing pairs from DB on startup; runs `select_pairs_weekly()` if table is empty
- **MTF over-filtering** -- Per-strategy MTF control via `MTF_ENABLED_FOR` config dict; disabled for mean reversion strategies

### Added
- **VWAP v2 Hybrid strategy** (20% allocation) -- VWAP + OU z-score dual confirmation with bid-ask spread filter
- **ORB v2 strategy** (5% allocation) -- Opening range breakout 10:00-11:30 AM with gap/range quality filters
- **Alpaca News Sentiment** -- Keyword-based headline scoring, soft position-size multiplier, 30-min cache
- **LLM Signal Scoring** -- Optional Claude Haiku signal evaluation; fail-open, 3s timeout, $0.10/day cost cap
- **Adaptive VIX-Aware Exits** -- Four VIX regimes with dynamic exit parameters
- **Walk-Forward Validation** -- Weekly OOS Sharpe check per strategy with auto-demotion
- **Strategy Health Dashboard** -- `/api/strategy_health` and `/api/filter_diagnostic` endpoints
- 196 unit tests (up from ~108)

### Changed
- Strategy allocations: StatMR 50%, VWAP 20%, Pairs 20%, ORB 5%, MicroMom 5%
- Notifications switched from WhatsApp to Telegram Bot API
- Dropped Python 3.11 support; minimum is now 3.12
- Dockerfile updated to Python 3.13
- Dashboard and web UI updated to V7 branding with new strategy filters

### Dependencies
- Added `anthropic>=0.49.0` (optional, for LLM scoring)

## [6.0.0] - 2026-03-13

### Added
- Complete rebuild around statistical mean reversion
- StatMeanReversion (60%), KalmanPairsTrader (25%), IntradayMicroMomentum (15%)
- Volatility-targeted position sizing (1% daily vol target)
- Daily P&L locking (GAIN_LOCK at +1.5%, LOSS_HALT at -1.0%)
- Beta neutralization with SPY hedging
- OU parameter fitting, Hurst exponent, consistency scoring analytics
- TWAP order execution for large orders

## [4.0.0] - 2026-03-13

### Added
- Multi-timeframe (MTF) confirmation for trade entries
- VIX-based risk scaling to reduce exposure in high-volatility regimes
- News sentiment filter via API integration
- Sector rotation strategy using ETF relative strength
- Pairs trading strategy with cointegration detection
- Advanced exit types: scaled take-profits, trailing stops, RSI-based exits, volatility-based exits
- Docker and docker-compose deployment
- Comprehensive test suite with pytest
- CI/CD pipeline with GitHub Actions and Codecov

## [3.0.0] - 2025-09-01

### Added
- ML-based signal filter for trade quality scoring
- Dynamic capital allocation across strategies
- WebSocket real-time price monitoring
- Short selling support
- Gap & Go strategy (pre-market gap continuation)
- Relative strength scanning
- Telegram alerts for fills, errors, and daily P&L summaries
- Web dashboard with live positions and equity curve
- Auto-optimization of strategy parameters via walk-forward analysis

## [2.0.0] - 2025-05-01

### Added
- Momentum strategy for multi-day trend following
- SQLite database for trade logging and analytics
- Backtesting engine with historical data replay
- Earnings date filter to avoid holding through reports
- Correlation filter to limit exposure to correlated positions

## [1.0.0] - 2025-01-01

### Added
- Opening Range Breakout (ORB) strategy with 3:1 R/R
- VWAP Mean Reversion strategy with 45-minute time stop
- Bracket order execution (entry + take-profit + stop-loss)
- Market regime detection via SPY 20-day EMA
- Rich terminal dashboard with live display
- State persistence via state.json
- Circuit breaker at -2.5% daily loss
- Paper/live mode switching via ALPACA_LIVE environment variable
- 50 hardcoded liquid symbols
