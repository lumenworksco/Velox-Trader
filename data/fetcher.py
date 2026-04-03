"""Alpaca data fetching — bars, account info, market status."""

import functools
import logging
import random
import threading
import time as _time
from datetime import datetime, timedelta

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

import config

logger = logging.getLogger(__name__)


# =============================================================================
# T2-007: Token-bucket rate limiter for Alpaca API calls
# =============================================================================

class AlpacaRateLimiter:
    """T2-007: Token-bucket rate limiter for Alpaca API endpoints.

    Alpaca limits:
    - Data API: 200 requests/minute
    - Trading API: ~50 requests/minute (varies by plan)

    Each bucket refills continuously. Callers must call ``acquire()``
    before making an API request. If the bucket is empty, acquire()
    blocks until a token is available.

    On HTTP 429 responses, callers should call ``backoff_429()``
    which applies exponential backoff with jitter.
    """

    def __init__(self, rate: float, per_seconds: float = 60.0, name: str = ""):
        """
        Args:
            rate: Max number of requests allowed per ``per_seconds``.
            per_seconds: Time window in seconds (default 60 = per minute).
            name: Human-readable label for logging.
        """
        self._rate = rate
        self._per = per_seconds
        self._name = name or f"limiter({rate}/{per_seconds}s)"

        self._tokens = rate
        self._max_tokens = rate
        self._last_refill = _time.monotonic()
        self._lock = threading.Lock()

        self._consecutive_429s = 0

    def acquire(self, timeout: float = 30.0) -> bool:
        """Block until a token is available, then consume it.

        Args:
            timeout: Max seconds to wait. Returns False if timed out.

        Returns:
            True if a token was acquired, False if timed out.
        """
        deadline = _time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

            # Wait for one token to refill
            wait = self._per / self._rate
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                logger.warning("T2-007: %s — rate limit acquire timed out", self._name)
                return False
            _time.sleep(min(wait, remaining))

    def _refill(self):
        """Refill tokens based on elapsed time since last refill (called under lock)."""
        now = _time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * (self._rate / self._per)
        self._tokens = min(self._max_tokens, self._tokens + new_tokens)
        self._last_refill = now

    def backoff_429(self):
        """T2-007: Exponential backoff with jitter after a 429 response.

        Call this when Alpaca returns HTTP 429 Too Many Requests.
        Sleeps for an increasing duration with random jitter.
        """
        self._consecutive_429s += 1
        base_delay = min(2 ** self._consecutive_429s, 60)
        jitter = random.uniform(0, base_delay * 0.5)
        delay = base_delay + jitter
        logger.warning(
            "T2-007: %s — 429 received (consecutive=%d), backing off %.1fs",
            self._name, self._consecutive_429s, delay,
        )
        _time.sleep(delay)

    def reset_429_counter(self):
        """Reset the consecutive 429 counter after a successful request."""
        self._consecutive_429s = 0

    @property
    def available_tokens(self) -> float:
        """Current available tokens (approximate, for monitoring)."""
        with self._lock:
            self._refill()
            return self._tokens


# Module-level rate limiters (singletons)
_data_limiter = AlpacaRateLimiter(rate=200, per_seconds=60.0, name="data")
_trading_limiter = AlpacaRateLimiter(rate=50, per_seconds=60.0, name="trading")


def get_data_rate_limiter() -> AlpacaRateLimiter:
    """Return the data API rate limiter (for external callers needing direct access)."""
    return _data_limiter


def get_trading_rate_limiter() -> AlpacaRateLimiter:
    """Return the trading API rate limiter."""
    return _trading_limiter


# =============================================================================
# V12-5.2: Retry decorator for transient server errors
# =============================================================================

# HTTP status codes and exception substrings that indicate a retryable server error
_RETRYABLE_STATUS_CODES = {500, 502, 503}

def _is_retryable_error(exc: Exception) -> bool:
    """Return True if the exception represents a retryable server/timeout error.

    Retries on HTTP 500/502/503 and timeouts. Does NOT retry on 4xx client errors.
    """
    exc_str = str(exc).lower()
    # Timeout errors (requests, httpx, urllib3, generic)
    timeout_keywords = ("timeout", "timed out", "connecttimeout", "readtimeout")
    if any(kw in exc_str for kw in timeout_keywords):
        return True
    # Check for retryable HTTP status codes in the exception
    for code in _RETRYABLE_STATUS_CODES:
        if str(code) in exc_str:
            return True
    # alpaca-py wraps HTTP errors — check for status_code attribute
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status and int(status) in _RETRYABLE_STATUS_CODES:
        return True
    return False


def retry_on_server_error(max_retries: int = 3, base_delay: float = 1.0):
    """V12-5.2: Decorator that retries a function on transient server errors.

    Uses exponential backoff (1s, 2s, 4s) and only retries on HTTP 500/502/503
    or timeout errors. Client errors (4xx) are raised immediately.

    Args:
        max_retries: Maximum number of retry attempts (default 3).
        base_delay: Base delay in seconds; doubles each attempt (default 1.0).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if not _is_retryable_error(exc) or attempt == max_retries:
                        raise
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "V12-5.2: %s attempt %d/%d failed (%s), retrying in %.1fs",
                        func.__name__, attempt, max_retries, exc, delay,
                    )
                    _time.sleep(delay)
            raise last_exc  # unreachable, but satisfies type checkers

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            import asyncio
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if not _is_retryable_error(exc) or attempt == max_retries:
                        raise
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "V12-5.2: %s attempt %d/%d failed (%s), retrying in %.1fs",
                        func.__name__, attempt, max_retries, exc, delay,
                    )
                    await asyncio.sleep(delay)
            raise last_exc

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


# --- Clients ---
_trading_client: TradingClient | None = None
_data_client: StockHistoricalDataClient | None = None


def get_trading_client() -> TradingClient:
    global _trading_client
    if _trading_client is None:
        _trading_client = TradingClient(
            api_key=config.API_KEY,
            secret_key=config.API_SECRET,
            paper=config.PAPER_MODE,
        )
    return _trading_client


def get_data_client() -> StockHistoricalDataClient:
    global _data_client
    if _data_client is None:
        _data_client = StockHistoricalDataClient(
            api_key=config.API_KEY or None,
            secret_key=config.API_SECRET or None,
        )
    return _data_client


@retry_on_server_error()
def get_account():
    """Get current account info."""
    _trading_limiter.acquire()  # T2-007: rate limit
    result = get_trading_client().get_account()
    _trading_limiter.reset_429_counter()
    return result


@retry_on_server_error()
def get_clock():
    """Get market clock (open/close status, next open/close times)."""
    _trading_limiter.acquire()  # T2-007: rate limit
    result = get_trading_client().get_clock()
    _trading_limiter.reset_429_counter()
    return result


@retry_on_server_error()
def get_positions():
    """Get all open positions from Alpaca."""
    _trading_limiter.acquire()  # T2-007: rate limit
    result = get_trading_client().get_all_positions()
    _trading_limiter.reset_429_counter()
    return result


@retry_on_server_error()
def get_open_orders():
    """Get all open orders."""
    _trading_limiter.acquire()  # T2-007: rate limit
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    result = get_trading_client().get_orders(request)
    _trading_limiter.reset_429_counter()
    return result


@retry_on_server_error()
def get_bars(symbol: str, timeframe: TimeFrame, start: datetime, end: datetime | None = None, limit: int | None = None) -> pd.DataFrame:
    """Fetch historical bars for a symbol, return as DataFrame."""
    _data_limiter.acquire()  # T2-007: rate limit
    params = {"symbol_or_symbols": symbol, "timeframe": timeframe, "start": start}
    if end:
        params["end"] = end
    if limit:
        params["limit"] = limit

    params["feed"] = DataFeed.IEX
    request = StockBarsRequest(**params)
    barset = get_data_client().get_stock_bars(request)
    _data_limiter.reset_429_counter()

    # alpaca-py returns a BarSet; access via .data or dict-style
    bar_list = None
    if hasattr(barset, "data") and barset.data:
        bar_list = barset.data.get(symbol)
    elif isinstance(barset, dict):
        bar_list = barset.get(symbol)
    else:
        # Try direct dict access (some alpaca-py versions)
        try:
            bar_list = barset[symbol]
        except (KeyError, TypeError):
            pass

    if not bar_list:
        return pd.DataFrame()

    records = []
    for bar in bar_list:
        records.append({
            "timestamp": bar.timestamp,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
            "vwap": float(bar.vwap) if bar.vwap else None,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
    return df


def get_bars_multi(symbols: list[str], timeframe: TimeFrame, start: datetime,
                   end: datetime | None = None, max_workers: int = 5) -> dict[str, pd.DataFrame]:
    """MED-036: Fetch bars for multiple symbols concurrently using ThreadPoolExecutor.

    Args:
        symbols: List of ticker symbols.
        timeframe: Alpaca TimeFrame.
        start: Start datetime.
        end: Optional end datetime.
        max_workers: Max concurrent threads (default 5 to stay within API rate limits).

    Returns:
        Dict mapping symbol -> DataFrame of bars.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, pd.DataFrame] = {}

    def _fetch_one(sym: str) -> tuple[str, pd.DataFrame]:
        return sym, get_bars(sym, timeframe, start=start, end=end)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                sym, df = future.result()
                results[sym] = df
            except Exception as e:
                logger.warning(f"Failed to fetch bars for {sym}: {e}")
                results[sym] = pd.DataFrame()

    return results


def get_daily_bars(symbol: str, days: int = 30) -> pd.DataFrame:
    """Get daily bars for the last N days."""
    start = datetime.now(config.ET) - timedelta(days=days + 5)  # buffer for weekends
    return get_bars(symbol, TimeFrame.Day, start=start, limit=days)


def get_intraday_bars(symbol: str, timeframe: TimeFrame, start: datetime, end: datetime | None = None) -> pd.DataFrame:
    """Get intraday bars from start time."""
    return get_bars(symbol, timeframe, start=start, end=end)


def get_filled_exit_info(symbol: str, side: str = "buy") -> tuple[float | None, str | None]:
    """Look up the actual fill price and exit reason for a recently closed position.

    When a bracket order's TP or SL leg fires at the broker, the position
    disappears. This function checks closed orders for the symbol to find
    the actual fill price and whether it was a stop or limit order.

    Args:
        symbol: The stock symbol
        side: The side of the ENTRY ("buy" for long, "sell" for short)
    Returns:
        Tuple of (fill_price, exit_reason) or (None, None) if not found.
        exit_reason is "stop_loss", "take_profit", or "market_close".
    """
    try:
        client = get_trading_client()
        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            symbols=[symbol],
            limit=10,
        )
        orders = client.get_orders(request)

        exit_side = "sell" if side == "buy" else "buy"
        for order in orders:
            if (order.symbol == symbol
                    and order.side == exit_side
                    and order.status == "filled"
                    and order.filled_avg_price is not None):
                price = float(order.filled_avg_price)
                # Determine exit reason from order type
                order_type = str(getattr(order, "order_type", "")).lower()
                if "stop" in order_type:
                    reason = "stop_loss"
                elif "limit" in order_type:
                    reason = "take_profit"
                elif "market" in order_type:
                    reason = "market_close"
                else:
                    reason = "broker_exit"
                return price, reason
    except Exception as e:
        logger.warning(f"Failed to look up fill info for {symbol}: {e}")
    return None, None


def get_filled_exit_price(symbol: str, side: str = "buy") -> float | None:
    """Convenience wrapper — returns just the fill price."""
    price, _ = get_filled_exit_info(symbol, side)
    return price


@retry_on_server_error()
def get_snapshot(symbol: str):
    """Get latest snapshot for a symbol (latest trade, quote, bar)."""
    _data_limiter.acquire()  # T2-007: rate limit
    client = get_data_client()
    snapshots = client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX))
    _data_limiter.reset_429_counter()
    if hasattr(snapshots, "data"):
        return snapshots.data.get(symbol) if snapshots.data else None
    if isinstance(snapshots, dict):
        return snapshots.get(symbol)
    return None


@retry_on_server_error()
def get_snapshots(symbols: list[str]) -> dict:
    """Get snapshots for multiple symbols at once."""
    _data_limiter.acquire()  # T2-007: rate limit
    client = get_data_client()
    result = client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=symbols, feed=DataFeed.IEX))
    _data_limiter.reset_429_counter()
    if hasattr(result, "data") and result.data:
        return dict(result.data)
    if isinstance(result, dict):
        return result
    return {}


def get_snapshots_batch(symbols: list[str], chunk_size: int = 1000) -> dict:
    """T4-002: Batch snapshot fetching — get snapshots for up to 1000 symbols per request.

    Replaces per-symbol snapshot calls in scan loops. Alpaca's multi-symbol
    endpoint supports up to 1000 symbols per request.

    Args:
        symbols: List of ticker symbols (can exceed 1000; will be chunked).
        chunk_size: Max symbols per API call (Alpaca limit is 1000).

    Returns:
        Dict mapping symbol -> Snapshot object.
    """
    if not symbols:
        return {}

    all_snapshots: dict = {}
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        try:
            batch = get_snapshots(chunk)
            all_snapshots.update(batch)
        except Exception as e:
            logger.warning("T4-002: Batch snapshot failed for chunk %d-%d: %s", i, i + len(chunk), e)
            # Fallback: fetch individually for this chunk
            for sym in chunk:
                try:
                    _data_limiter.acquire()  # V12-5.3: rate limit individual fallback fetches
                    snap = get_snapshot(sym)
                    if snap is not None:
                        all_snapshots[sym] = snap
                except Exception:
                    pass

    logger.debug("T4-002: Fetched %d/%d snapshots in batch mode", len(all_snapshots), len(symbols))
    return all_snapshots


def verify_connectivity() -> dict:
    """Verify API connectivity. Returns account info dict or raises."""
    account = get_account()
    clock = get_clock()

    info = {
        "account_id": account.id,
        "equity": float(account.equity),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "paper": config.PAPER_MODE,
        "market_open": clock.is_open,
        "next_open": clock.next_open,
        "next_close": clock.next_close,
    }
    return info


def verify_data_feed(symbol: str = "SPY") -> bool:
    """Verify that we can fetch data for at least one symbol."""
    try:
        df = get_daily_bars(symbol, days=5)
        return len(df) > 0
    except Exception as e:
        logger.error(f"Data feed verification failed for {symbol}: {e}")
        return False


# =============================================================================
# T2-001: Async I/O variants for non-blocking data fetching
# =============================================================================

# Conditional httpx import — async methods degrade gracefully if unavailable
try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    _HTTPX_AVAILABLE = False

_BASE_URL_DATA = "https://data.alpaca.markets/v2"
_BASE_URL_TRADING_PAPER = "https://paper-api.alpaca.markets/v2"
_BASE_URL_TRADING_LIVE = "https://api.alpaca.markets/v2"


def _get_async_headers() -> dict[str, str]:
    """Build auth headers for httpx async requests."""
    return {
        "APCA-API-KEY-ID": config.API_KEY,
        "APCA-API-SECRET-KEY": config.API_SECRET,
        "Accept": "application/json",
    }


def _get_trading_base_url() -> str:
    """Return the correct trading API base URL based on paper/live mode."""
    return _BASE_URL_TRADING_PAPER if config.PAPER_MODE else _BASE_URL_TRADING_LIVE


@retry_on_server_error()
async def get_daily_bars_async(symbol: str, days: int = 30) -> pd.DataFrame:
    """T2-001: Async variant of get_daily_bars using httpx.

    Falls back to sync version if httpx is not available.
    """
    if not _HTTPX_AVAILABLE:
        return get_daily_bars(symbol, days)

    from datetime import timezone
    start = (datetime.now(timezone.utc) - timedelta(days=days + 5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"{_BASE_URL_DATA}/stocks/{symbol}/bars"
    params = {
        "timeframe": "1Day",
        "start": start,
        "limit": days,
        "feed": "iex",
    }

    async with httpx.AsyncClient(headers=_get_async_headers(), timeout=10.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    bars = data.get("bars", [])
    if not bars:
        return pd.DataFrame()

    records = [
        {
            "timestamp": bar["t"],
            "open": float(bar["o"]),
            "high": float(bar["h"]),
            "low": float(bar["l"]),
            "close": float(bar["c"]),
            "volume": float(bar["v"]),
            "vwap": float(bar.get("vw", 0)) if bar.get("vw") else None,
        }
        for bar in bars
    ]
    df = pd.DataFrame(records)
    if not df.empty:
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
    return df


@retry_on_server_error()
async def get_intraday_bars_async(
    symbol: str, timeframe_str: str = "1Min",
    start: datetime | None = None, end: datetime | None = None,
) -> pd.DataFrame:
    """T2-001: Async variant of get_intraday_bars using httpx.

    Args:
        symbol: Ticker symbol.
        timeframe_str: Alpaca timeframe string (e.g. "1Min", "5Min").
        start: Start datetime.
        end: Optional end datetime.
    """
    if not _HTTPX_AVAILABLE:
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        tf_map = {"1Min": TimeFrame(1, TimeFrameUnit.Minute), "5Min": TimeFrame(5, TimeFrameUnit.Minute)}
        tf = tf_map.get(timeframe_str, TimeFrame(1, TimeFrameUnit.Minute))
        return get_intraday_bars(symbol, tf, start=start or datetime.now(), end=end)

    url = f"{_BASE_URL_DATA}/stocks/{symbol}/bars"
    params = {"timeframe": timeframe_str, "feed": "iex"}
    if start:
        params["start"] = start.isoformat()
    if end:
        params["end"] = end.isoformat()

    async with httpx.AsyncClient(headers=_get_async_headers(), timeout=10.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    bars = data.get("bars", [])
    if not bars:
        return pd.DataFrame()

    records = [
        {
            "timestamp": bar["t"],
            "open": float(bar["o"]),
            "high": float(bar["h"]),
            "low": float(bar["l"]),
            "close": float(bar["c"]),
            "volume": float(bar["v"]),
            "vwap": float(bar.get("vw", 0)) if bar.get("vw") else None,
        }
        for bar in bars
    ]
    df = pd.DataFrame(records)
    if not df.empty:
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
    return df


@retry_on_server_error()
async def get_snapshot_async(symbol: str) -> dict | None:
    """T2-001: Async variant of get_snapshot using httpx.

    Returns raw dict snapshot (latest trade, quote, bar) or None.
    """
    if not _HTTPX_AVAILABLE:
        snap = get_snapshot(symbol)
        return snap

    url = f"{_BASE_URL_DATA}/stocks/{symbol}/snapshot"
    params = {"feed": "iex"}

    async with httpx.AsyncClient(headers=_get_async_headers(), timeout=10.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


@retry_on_server_error()
async def get_account_async() -> dict:
    """T2-001: Async variant of get_account using httpx.

    Returns raw dict of account info.
    """
    if not _HTTPX_AVAILABLE:
        acct = get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
        }

    url = f"{_get_trading_base_url()}/account"

    async with httpx.AsyncClient(headers=_get_async_headers(), timeout=10.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()
