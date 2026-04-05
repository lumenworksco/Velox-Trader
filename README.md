# Velox V12 — Autonomous Algorithmic Trading System

![CI](https://github.com/lumenworksco/Velox-Trader/actions/workflows/ci.yml/badge.svg)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A production-grade autonomous equity trading system built on the Alpaca API. Features 6 diversified strategies, a unified ExitOrchestrator with 10-priority-level exit cascade, an 18-stage signal pipeline with ML confidence gating, real-time VaR monitoring, FinBERT sentiment analysis, structured logging, and a Docker production stack with PostgreSQL, Prometheus, Grafana, Alertmanager, and automated backups.

**Philosophy:** Consistent returns over big wins. Target 0.3-0.8% per trade, 65-75% win rate, 15-25 trades/day across 6 active strategies.

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/lumenworksco/Velox-Trader.git
cd Velox-Trader
cp .env.example .env
# Edit .env with your Alpaca API keys
docker compose up -d
```

Services:
- **Velox Dashboard**: http://localhost:8080
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Alertmanager**: http://localhost:9093

### Option 2: Local

```bash
pip install -r requirements.txt
export ALPACA_API_KEY="your-api-key"
export ALPACA_API_SECRET="your-secret-key"
python main.py
```

---

## Strategies

| Strategy | Allocation | Type | Hold | Description |
|---|---|---|---|---|
| **StatMeanReversion** | 25% | Mean Reversion | Intraday | OU process z-score entries with Hurst/ADF filtering on 2-min bars |
| **VWAP v2 Hybrid** | 13% | Mean Reversion | Intraday | VWAP + OU z-score dual confirmation with bid-ask spread filter |
| **KalmanPairsTrader** | 27% | Market-Neutral | Multi-day | Kalman filter dynamic hedge ratios on cointegrated sector pairs |
| **ORB v2** | 12% | Breakout | Day | Opening range breakout with gap/range quality filters (10:00-11:30 AM) |
| **IntradayMicroMomentum** | 5% | Event-Driven | 8 min | SPY volume spike detection, high-beta stock scalps |
| **PEAD** | 18% | Event-Driven | Multi-day | Post-earnings announcement drift with FinBERT sentiment confirmation |

### How It Works

1. **9:00 AM** -- Dynamic universe selection filters 127+ symbols by volume, market cap, and regime
2. **9:00 AM** -- StatMR builds mean-reversion universe via Hurst exponent, ADF stationarity, and OU half-life
3. **9:00 AM** -- Pre-open gap analysis runs for mean-reversion and breakout strategy calibration
4. **9:30 AM** -- Adaptive scan cycle begins: all strategies scan for signals (interval adjusts based on market activity)
5. **Signal pipeline** -- Each signal passes through 18 stages: data quality gate, MTF confluence, gap analysis, transaction cost filter, VaR check, correlation limiter, sector exposure enforcement, news sentiment, FinBERT scoring, ML confidence gating, signal conflict resolution, and more
6. **ExitOrchestrator** -- Unified exit system with 10-priority-level cascade replaces all legacy exit logic
7. **OMS** -- Orders tracked through 7-state lifecycle (PENDING -> SUBMITTED -> FILLED/CANCELLED/REJECTED)
8. **Every 15 min** -- Beta neutralizer checks portfolio beta and hedges with SPY if |beta| > 0.3
9. **Every 30 min** -- Corporate actions monitoring and position reconciliation with broker
10. **Intraday** -- Correlation matrix refresh keeps sector exposure and effective bets current
11. **Weekly** -- KalmanPairs selects top 15 cointegrated pairs; walk-forward validator checks OOS Sharpe
12. **Friday EOD** -- Tax-loss harvesting evaluates unrealized losses for strategic realization
13. **EOD** -- End-of-day close routine with overnight hold selection

---

## Architecture

~95,700 lines of code across 175 Python files, with 63 test files.

```
trading_bot/
  main.py                    # Thin orchestrator with position recovery on restart
  config/
    settings.py              # Canonical configuration (all strategy and risk parameters)
  data.py                    # Alpaca market data (REST + WebSocket)
  execution.py               # Order routing, TWAP splitting, bracket orders, chase logic
  database.py                # SQLite/PostgreSQL persistence layer

  engine/                    # V12 engine package
    startup.py               # Module initialization and startup checks
    signal_processor.py      # 18-stage signal pipeline with ML gating + sector enforcement
    scanner.py               # Strategy scan orchestration with adaptive intervals
    daily_tasks.py           # Daily reset, weekly tasks, EOD close, tax harvesting
    broker_sync.py           # Position reconciliation with broker (open/close/30min)
    exit_orchestrator.py     # Unified exit system: 10-priority-level exit cascade
    profit_maximizer.py      # Adaptive scan, signal stacking, momentum persistence, dynamic stops
    events.py                # Pub/sub event bus
    metrics.py               # Prometheus counters/gauges/histograms
    logging_config.py        # structlog dev/prod configuration

  oms/                       # Order Management System
    order.py                 # Order dataclass with 7-state machine
    order_manager.py         # Thread-safe registry with idempotency keys
    kill_switch.py           # Emergency halt: cancel all + close all
    transaction_cost.py      # Pre-trade cost estimation (spread + slippage)

  strategies/
    base.py                  # Signal dataclass, shared types
    regime.py                # SPY HMM market regime detection
    stat_mean_reversion.py   # OU z-score mean reversion (25%)
    vwap.py                  # VWAP + OU hybrid entries (13%)
    kalman_pairs.py          # Kalman filter pairs trading (27%)
    orb_v2.py                # Opening range breakout v2 (12%)
    micro_momentum.py        # SPY vol spike micro momentum (5%)
    pead.py                  # Post-earnings announcement drift (18%)
    dynamic_universe.py      # Regime-adaptive universe selection

  risk/
    risk_manager.py          # Trade tracking, position limits, time-of-day and Friday multipliers
    circuit_breaker.py       # V12 tiered circuit breaker (4 tiers + VIX spike trigger)
    var_monitor.py           # Parametric + Historical + Monte Carlo VaR
    correlation_limiter.py   # Eigenvalue-based effective bets + sector limits + intraday refresh
    vol_targeting.py         # Volatility-targeted position sizing
    daily_pnl_lock.py        # P&L lock states (NORMAL/GAIN_LOCK/LOSS_HALT)
    beta_neutralizer.py      # Portfolio beta monitoring + SPY hedging

  ml/                        # Machine Learning
    finbert_sentiment.py     # FinBERT local sentiment scoring (no paid API)
    model_monitor.py         # ML prediction accuracy tracking and drift detection

  data/                      # Data quality and analysis
    feed_monitor.py          # Alpaca outage detection + backup stops
    gap_analysis.py          # Pre-open gap analysis for MR/breakout calibration

  db/                        # SQLAlchemy database abstraction
    __init__.py              # Dual-backend engine (SQLite/PostgreSQL)
    models.py                # Database table definitions
    migrations/              # Alembic migration scripts (includes commission column)

  auth/                      # Dashboard authentication
    jwt_auth.py              # JWT token create/verify

  analytics/                 # Performance metrics and statistical tools
    performance.py           # Sharpe, Sortino, drawdown, attribution
    ou_tools.py              # Ornstein-Uhlenbeck parameter fitting
    hurst.py                 # Hurst exponent (R/S analysis)
    consistency_score.py     # Consistency score (0-100)

  monitoring/
    prometheus.yml           # Prometheus scrape configuration
    health_alerts.py         # System health alerts (8 alert types)
    daily_report.py          # Daily P&L Telegram report

  models/                    # Trained ML models
    model_*.pkl              # 4-model averaging ensemble (50 core features, 0.947 AUC)

  scripts/
    train_ml_model.py        # ML training: Alpaca data, feature selection, Optuna HPO

  web_dashboard.py           # FastAPI dashboard with Apple-style UI

  tests/                     # 63 test files, 900+ tests
  Dockerfile
  docker-compose.yml         # Production stack: 6 services
  requirements.txt
  requirements-dev.txt       # Development dependencies
```

---

## V12 Features

### 18-Stage Signal Pipeline
Every signal passes through a full qualification pipeline including data quality gate, multi-timeframe confluence, gap analysis, transaction cost filter, VaR check, correlation limiter, sector exposure enforcement (30% max), news sentiment, FinBERT scoring, ML confidence gating, signal conflict resolution, and intra-scan entry limits (5 new entries max per scan).

### Unified ExitOrchestrator
A single exit system with 10 priority levels replaces three competing exit subsystems from prior versions. Priority cascade ensures the highest-urgency exit reason always wins (e.g., kill switch overrides trailing stop).

### ML Confidence Gating
4-model averaging ensemble trained with Optuna hyperparameter optimization. 50 core features (reduced from 200+) with 0.947 AUC. FinBERT sentiment runs locally with no paid API dependency. Bayesian Kelly criterion for position sizing.

### Order Management System (OMS)
Full order lifecycle tracking with 7-state machine (PENDING -> SUBMITTED -> PARTIALLY_FILLED -> FILLED / CANCELLED / REJECTED / EXPIRED). Thread-safe registry with idempotency keys prevents duplicate orders. API failure circuit breaker triggers kill switch after 5 failures in 5 minutes. Order rejection classification and chase logic for limit orders.

### Tiered Circuit Breaker
Progressive risk reduction based on daily P&L, plus VIX spike trigger:
- **Yellow** (-1%): Reduce new position sizes by 50%
- **Orange** (-2%): Stop all new entries
- **Red** (-3%): Close day trades, keep swing positions
- **Black** (-4%): Kill switch -- close everything
- **VIX Spike**: Automatic circuit breaker activation on volatility regime change

### VaR Monitor
Real-time portfolio Value-at-Risk using three methods:
- Parametric VaR (95% and 99%)
- Historical simulation
- Monte Carlo (10,000 paths)

### Correlation Limiter with Intraday Refresh
Prevents concentration risk via eigenvalue-based effective bets calculation and sector Herfindahl index monitoring. Correlation matrix refreshes intraday to keep exposure calculations current.

### Position Recovery on Restart
On startup, the bot reconciles with the broker to recover any open positions from a prior session. No positions are orphaned across restarts.

### Corporate Actions Monitoring
Checks for splits, dividends, and other corporate actions every 30 minutes. Adjusts position tracking and alerts on events that could affect open trades.

### Tax-Loss Harvesting
Friday end-of-day routine evaluates unrealized losses for strategic realization, capturing tax benefits while maintaining portfolio exposure.

### Risk Manager Enhancements
Time-of-day risk multiplier (tighter sizing at open/close), Friday risk reduction, and drawdown-based dynamic sizing that scales down as daily losses accumulate.

### Profit Maximizer
Adaptive scan intervals based on market activity, signal stacking for high-conviction entries, momentum persistence detection, dynamic stop adjustment on winning trades, and win-streak-aware sizing.

### Data Feed Monitoring
Alpaca outage detection with automatic fallback to backup protective stops when live data feed goes stale.

### Health Alerts
8 alert types covering system health: data feed staleness, reconciliation mismatches, high rejection rates, circuit breaker activation, kill switch events, memory/CPU thresholds, and more.

### Transaction Cost Filter
Pre-trade expected value check: rejects signals where estimated costs (spread + slippage + commission) exceed expected profit.

### Structured Logging
JSON-formatted logs in production (structlog), human-readable in development. All events include correlation IDs for tracing.

### Prometheus Metrics
Exposed at `/metrics` on port 8080:
- `velox_open_positions` -- Current position count
- `velox_daily_pnl` -- Running daily P&L
- `velox_order_latency` -- Order submission to fill latency
- `velox_signal_count` -- Signals generated per strategy
- `velox_circuit_breaker_state` -- Current circuit breaker tier

### Web Dashboard
Apple-style frosted glass UI at http://localhost:8080 with:
- Live equity and P&L from Alpaca account
- Open positions with real-time unrealized P&L
- Trade log filterable by strategy
- Signal filter analysis and exit reason breakdown
- OMS status, circuit breaker state, kill switch controls
- Auto-refresh every 30 seconds

### Dynamic Universe
Daily symbol selection at 9 AM based on volume, market cap, and current market regime. Adapts universe size and composition to volatility conditions.

---

## Risk Engine

| Component | Description |
|---|---|
| **Tiered Circuit Breaker** | 4-tier progressive risk reduction (-1% to -4%) + VIX spike trigger |
| **VaR Monitor** | Parametric + Historical + Monte Carlo VaR at 95%/99% |
| **Correlation Limiter** | Eigenvalue effective bets + sector concentration (30% max) + intraday refresh |
| **Volatility Targeting** | Scales position sizes so daily portfolio vol = 1% target |
| **Daily P&L Lock** | GAIN_LOCK at +1.5% (30% sizing), LOSS_HALT at -1.0% (stops new trades) |
| **Beta Neutralization** | Monitors portfolio beta, hedges with SPY when \|beta\| > 0.3 |
| **Transaction Cost Filter** | Rejects negative expected-value trades before submission |
| **Kill Switch** | Emergency halt: cancels all orders, closes all positions |
| **TWAP Execution** | Orders > $2,000 split into 5 time-weighted slices |
| **API Failure Breaker** | 5 failures in 5 min triggers kill switch |
| **Drawdown Sizing** | Dynamic position scaling based on cumulative daily drawdown |
| **Time-of-Day Multiplier** | Tighter sizing at market open/close, Friday risk reduction |

---

## Docker Production Stack

```bash
docker compose up -d              # Start all 6 services
docker compose logs -f velox      # Follow trading bot logs
docker compose exec velox python main.py --diagnose  # Run diagnostic
```

| Service | Port | Description |
|---|---|---|
| **velox** | 8080 | Trading bot + web dashboard |
| **postgres** | 5432 | PostgreSQL 16 database |
| **prometheus** | 9090 | Metrics collection |
| **grafana** | 3000 | Monitoring dashboards (admin/admin) |
| **alertmanager** | 9093 | Alert routing and notification |
| **backup** | -- | Automated database backups |

Environment variables are configured in `.env` (see `.env.example`).

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `ALPACA_LIVE` | `false` | Paper vs live trading |
| `MAX_POSITIONS` | `12` | Maximum concurrent positions |
| `MAX_POSITION_PCT` | `8%` | Maximum position size as % of portfolio |
| `SCAN_INTERVAL_SEC` | `120` | Seconds between strategy scans (adaptive) |
| `RISK_PER_TRADE_PCT` | `0.8%` | Max risk per trade |
| `VOL_TARGET_DAILY` | `1.0%` | Daily portfolio volatility target |
| `DATABASE_URL` | `sqlite:///bot.db` | Database connection (PostgreSQL supported) |
| `STRUCTURED_LOGGING` | `false` | Enable JSON structured logging |
| `WEB_DASHBOARD_ENABLED` | `true` | Enable web dashboard on port 8080 |
| `WATCHDOG_ENABLED` | `false` | Enable position watchdog |
| `NEWS_SENTIMENT_ENABLED` | `true` | Enable Alpaca news sentiment filter |
| `FINBERT_ENABLED` | `true` | Enable local FinBERT sentiment scoring |
| `ML_GATING_ENABLED` | `true` | Enable ML confidence gating on signals |
| `LLM_SCORING_ENABLED` | `false` | Enable Claude Haiku signal scoring |
| `ADAPTIVE_EXITS_ENABLED` | `true` | Enable VIX-aware adaptive exits |
| `WALK_FORWARD_ENABLED` | `true` | Enable weekly walk-forward validation |
| `TAX_HARVESTING_ENABLED` | `true` | Enable Friday EOD tax-loss harvesting |
| `TELEGRAM_ENABLED` | `false` | Enable Telegram trade alerts |
| `ALLOW_SHORT` | `false` | Enable short selling |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific modules
pytest tests/test_v12_exit_orchestrator.py -v
pytest tests/test_v10_oms.py -v
pytest tests/test_v10_events.py -v
pytest tests/test_v10_phase4.py -v
```

900+ tests across 63 test files covering all strategies, risk modules, OMS, exit orchestrator, event bus, circuit breaker, ML pipeline, and analytics.

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Health check (Docker/monitoring) |
| `GET /api/stats` | Live performance stats from Alpaca + analytics |
| `GET /api/positions` | Current open positions (live broker data) |
| `GET /api/trades` | Trade history (filterable by strategy) |
| `GET /api/portfolio_history` | Daily portfolio snapshots |
| `GET /api/signals` | Signal log by date |
| `GET /api/signal_stats` | Signal skip reason breakdown |
| `GET /api/trade_analysis` | Exit reason analysis |
| `GET /api/risk-state` | Risk engine state (vol scalar, beta, P&L lock) |
| `GET /api/strategy_health` | Per-strategy health metrics |
| `GET /api/v10/oms` | OMS order status and history |
| `GET /api/v10/circuit_breaker` | Circuit breaker state and history |
| `POST /api/v10/kill_switch/activate` | Activate emergency kill switch |

---

## Version History

| Version | Date | Focus |
|---|---|---|
| V1 | 2025-01 | ORB + VWAP strategies, basic risk management |
| V2 | 2025-05 | Momentum strategy, WebSocket monitoring |
| V3 | 2025-09 | ML signal filter, short selling, dynamic allocation |
| V4 | 2026-03 | Sector rotation, pairs trading, MTF, news filter |
| V5 | 2026-03 | EMA scalping, shadow mode, advanced exits |
| V6 | 2026-03 | Complete rebuild: statistical mean reversion, vol targeting |
| V7 | 2026-03 | 5-strategy diversification, news sentiment, LLM scoring, adaptive exits |
| V8 | 2026-03 | Bug fixes, thread safety, dead code removal |
| V9 | 2026-03 | Engine decomposition, PostgreSQL, OMS skeleton |
| V10 | 2026-03 | Production-grade: tiered circuit breaker, VaR monitor, correlation limiter, event bus, structured logging, Prometheus metrics, Docker stack, Apple-style dashboard |
| V11 | 2026-03 | ML ensemble (50 features, 0.947 AUC), FinBERT sentiment, Bayesian Kelly sizing |
| **V12** | **2026-04** | **Unified ExitOrchestrator, 18-stage signal pipeline, position recovery, corporate actions monitoring, tax-loss harvesting, intraday correlation refresh, profit maximizer, data feed monitoring, health alerts, 6-service Docker stack** |

---

## Risk Warning

> **This is experimental software. Use at your own risk. Past performance is not indicative of future results.** Trading equities involves substantial risk of loss. This software is provided for educational and research purposes. The authors are not responsible for any financial losses incurred through the use of this software.

---

## License

[MIT](LICENSE)
