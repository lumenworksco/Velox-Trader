# Contributing

Thanks for your interest in contributing to Velox V12.

## Adding a New Strategy

1. Create a new file in `strategies/` (e.g., `strategies/my_strategy.py`).

2. Implement the required interface:

   ```python
   from strategies.base import Signal

   class MyStrategy:
       def __init__(self):
           pass

       def scan(self, now: datetime, regime: str) -> list[Signal]:
           """Scan for entry signals. Returns list of Signal objects."""
           signals = []
           # Your logic here
           return signals

       def check_exits(self, open_trades: dict, now: datetime) -> list[dict]:
           """Check open positions for exit conditions.
           Returns list of dicts with symbol, action (full/partial), reason."""
           return []
   ```

3. Add configuration flags in `config/settings.py`:

   ```python
   MY_STRATEGY_ENABLED = os.getenv("MY_STRATEGY_ENABLED", "false") == "true"
   ```

4. Register the strategy in `engine/startup.py` inside `initialize_strategies()`.

5. Add an allocation in `STRATEGY_ALLOCATIONS` (must sum to 1.0).

6. Add earnings filter in `scan()`:
   ```python
   if _has_earnings_soon is not None and _has_earnings_soon(symbol):
       continue
   ```

7. Write tests in `tests/test_my_strategy.py`.

## Project Structure

- **`engine/`** -- Core trading engine (startup, scanning, signal processing, exit orchestrator, events)
- **`oms/`** -- Order Management System (order lifecycle, kill switch, cost model)
- **`risk/`** -- 28 risk modules (circuit breaker, VaR, correlation, vol targeting, Kelly, drawdown, etc.)
- **`strategies/`** -- 6 trading strategies (StatMR, VWAP, KalmanPairs, ORB, MicroMom, PEAD)
- **`execution/`** -- Order routing, smart router, chase logic, slippage model
- **`ml/`** -- ML pipeline (inference, features, FinBERT sentiment, model monitoring)
- **`data/`** -- Data fetching, caching, quality checks, alternative data sources
- **`analytics/`** -- HMM regime, OU tools, signal ranking, lead-lag, performance metrics
- **`microstructure/`** -- VPIN, order book, spread analysis, trade classification
- **`compliance/`** -- PDT protection, audit trail, best execution, surveillance
- **`monitoring/`** -- Health alerts, daily reports, Prometheus metrics, reconciliation
- **`ops/`** -- Disaster recovery, drawdown risk, tax-loss harvesting
- **`tests/`** -- 1,900+ unit tests across 85 test files

## Pull Request Requirements

- All existing tests must pass (`pytest tests/ -v`).
- New features must be gated behind config flags (disabled by default).
- Include tests for any new strategy or module.
- Update `CHANGELOG.md` with your changes.
- Signals must pass through the full pipeline in `engine/signal_processor.py`.
- All strategies must have earnings filters and bid-ask spread checks.

## Code Style

- Python 3.12+
- Use type hints on all function signatures.
- Follow fail-open design: wrap external calls in try/except, never crash the main loop.
- Run before submitting:

  ```bash
  pip install -r requirements-dev.txt
  ruff check .
  ruff format .
  pytest tests/ -v
  ```

## Docker Development

```bash
docker compose up -d --build velox   # Rebuild and restart bot
docker compose logs -f velox         # Follow logs
docker compose exec velox python main.py --diagnose  # Run diagnostics
bash scripts/status.sh               # Quick status check
```

## Reporting Issues

Open a GitHub issue with:
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs (redact any API keys)
