"""
Binance Price Feed - REST + WebSocket Client for BTC Spot Price

Provides:
- REST: fetch_klines() for historical OHLCV data (backtesting)
- REST: fetch_klines_range() for paginated long-range fetches
- WebSocket: BinancePriceFeed for live price streaming
- Realized volatility calculation from 1-second bars
- Automatic caching to data/binance_cache/ to avoid re-fetching

No Binance API keys required - uses public endpoints only.

Usage (REST - backtesting):
    from lib.binance_feed import fetch_klines, fetch_klines_range

    klines = fetch_klines("BTCUSDT", "1s", start_ms, end_ms)
    klines = fetch_klines_range("BTCUSDT", start_ts, end_ts)

Usage (WebSocket - live):
    from lib.binance_feed import BinancePriceFeed

    feed = BinancePriceFeed()
    await feed.connect()
    price = feed.get_price()
    vol = feed.get_realized_vol(300)
    await feed.disconnect()
"""

import os
import json
import time
import asyncio
import hashlib
from collections import deque
from pathlib import Path
from typing import List, Tuple, Optional, Callable
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import numpy as np

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / "data" / "binance_cache"

# API endpoints - try .com first, fall back to .us (for US users)
_BINANCE_REST_URLS = [
    "https://api.binance.com/api/v3",
    "https://api.binance.us/api/v3",
]

# WebSocket endpoints
_BINANCE_WS_URLS = [
    "wss://stream.binance.com:9443/ws",
    "wss://fstream.binance.com/ws",
    "wss://data-stream.binance.vision/ws",
    "wss://stream.binance.us:9443/ws",
]

# Track which endpoint works (cached after first successful call)
_working_rest_url: Optional[str] = None
_working_ws_url: Optional[str] = None

# Intervals that binance.us doesn't support
_US_UNSUPPORTED_INTERVALS = {"1s"}


def _rest_request(path: str, timeout: int = 15) -> bytes:
    """
    Make a REST API request, trying all endpoints until one works.

    Args:
        path: API path with query string (e.g., "/klines?symbol=BTCUSDT&...")
        timeout: Request timeout in seconds

    Returns:
        Response body as bytes
    """
    global _working_rest_url

    errors = []
    # If we have a known working URL, try it first
    urls_to_try = list(_BINANCE_REST_URLS)
    if _working_rest_url:
        urls_to_try = [_working_rest_url] + [u for u in urls_to_try if u != _working_rest_url]

    for base_url in urls_to_try:
        full_url = base_url + path
        req = Request(full_url, headers={"User-Agent": "polymarket-bot/1.0"})
        try:
            resp = urlopen(req, timeout=timeout)
            _working_rest_url = base_url
            return resp.read()
        except HTTPError as e:
            # 400 = bad request (e.g., unsupported interval) — endpoint is reachable
            # 451 = geo-blocked — skip this endpoint
            if e.code == 400:
                _working_rest_url = base_url
            errors.append((base_url, e))
            continue
        except URLError as e:
            errors.append((base_url, e))
            continue

    # All endpoints failed
    raise errors[-1][1] if errors else URLError("No Binance endpoints available")


# ── REST API (for backtesting) ──────────────────────────────────────

def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Fetch OHLCV klines from Binance REST API.

    Automatically tries api.binance.com first, then api.binance.us.
    If 1s interval is unavailable, automatically falls back to 1m.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Kline interval (e.g., "1s", "1m", "5m")
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        limit: Max klines per request (max 1000)

    Returns:
        List of (timestamp_s, open, high, low, close, volume) tuples
    """
    global _working_rest_url

    path = (
        f"/klines?symbol={symbol}&interval={interval}"
        f"&startTime={start_ms}&endTime={end_ms}&limit={limit}"
    )

    try:
        raw = _rest_request(path)
        data = json.loads(raw)
    except (HTTPError, URLError):
        # If sub-minute interval failed, try with 1m as fallback
        if interval in _US_UNSUPPORTED_INTERVALS:
            return fetch_klines(symbol, "1m", start_ms, end_ms, limit)
        raise

    results = []
    for k in data:
        # Binance kline format: [open_time, open, high, low, close, volume, ...]
        ts = int(k[0]) // 1000  # ms -> seconds
        o, h, l, c, v = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
        results.append((ts, o, h, l, c, v))

    return results


def _cache_key(symbol: str, interval: str, start_ms: int, end_ms: int) -> str:
    """Generate cache filename from request params."""
    raw = f"{symbol}_{interval}_{start_ms}_{end_ms}"
    h = hashlib.md5(raw.encode()).hexdigest()[:12]
    return f"{symbol}_{interval}_{h}.jsonl"


def _effective_interval(interval: str) -> str:
    """Get the interval that will actually be used (after fallback)."""
    if _working_rest_url and "binance.us" in _working_rest_url:
        if interval in _US_UNSUPPORTED_INTERVALS:
            return "1m"
    return interval


def fetch_klines_range(
    symbol: str,
    start_ts: float,
    end_ts: float,
    interval: str = "1s",
    use_cache: bool = True,
) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Fetch klines for a time range, paginating as needed (1000 per request).
    Caches results to data/binance_cache/ as JSONL.

    Automatically falls back from 1s to 1m if the endpoint doesn't support
    sub-minute intervals (e.g., Binance US).

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        start_ts: Start time in epoch seconds
        end_ts: End time in epoch seconds
        interval: Kline interval (default: "1s", may fall back to "1m")
        use_cache: Whether to use disk cache

    Returns:
        List of (timestamp_s, open, high, low, close, volume) tuples
    """
    start_ms = int(start_ts * 1000)
    end_ms = int(end_ts * 1000)

    # Check cache (for both requested interval and fallback)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Check requested interval cache
        cache_file = CACHE_DIR / _cache_key(symbol, interval, start_ms, end_ms)
        if cache_file.exists():
            results = []
            with open(cache_file) as f:
                for line in f:
                    row = json.loads(line)
                    results.append(tuple(row))
            return results
        # Also check fallback interval cache
        eff = _effective_interval(interval)
        if eff != interval:
            cache_file_fb = CACHE_DIR / _cache_key(symbol, eff, start_ms, end_ms)
            if cache_file_fb.exists():
                results = []
                with open(cache_file_fb) as f:
                    for line in f:
                        row = json.loads(line)
                        results.append(tuple(row))
                return results

    # Determine actual interval (may be different after first API call)
    actual_interval = interval

    # Paginate
    all_klines = []
    cursor_ms = start_ms

    while cursor_ms < end_ms:
        batch = fetch_klines(symbol, actual_interval, cursor_ms, end_ms, limit=1000)

        # Detect if interval was silently changed by fallback
        actual_interval = _effective_interval(interval)

        if not batch:
            break

        all_klines.extend(batch)

        # Interval to ms mapping for pagination
        interval_ms_map = {
            "1s": 1000, "1m": 60_000, "3m": 180_000, "5m": 300_000,
            "15m": 900_000, "1h": 3_600_000,
        }
        step_ms = interval_ms_map.get(actual_interval, 60_000)

        # Move cursor past the last returned kline
        last_ts_ms = batch[-1][0] * 1000
        cursor_ms = last_ts_ms + step_ms

        # Rate limit: Binance allows 1200 req/min, be conservative
        if cursor_ms < end_ms:
            time.sleep(0.1)

    # Deduplicate by timestamp
    seen = set()
    deduped = []
    for k in all_klines:
        if k[0] not in seen:
            seen.add(k[0])
            deduped.append(k)
    deduped.sort(key=lambda x: x[0])

    # Cache to disk
    if use_cache and deduped:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / _cache_key(symbol, actual_interval, start_ms, end_ms)
        with open(cache_file, "w") as f:
            for k in deduped:
                f.write(json.dumps(list(k)) + "\n")

    return deduped


def build_price_lookup(
    klines: List[Tuple[int, float, float, float, float, float]],
) -> dict:
    """
    Build timestamp -> close price lookup from klines.

    Args:
        klines: List of (ts, o, h, l, c, v) tuples

    Returns:
        Dict mapping timestamp (int seconds) -> close price
    """
    return {k[0]: k[4] for k in klines}


def get_price_at(lookup: dict, ts: float) -> Optional[float]:
    """
    Get price at timestamp, using nearest available price.

    Args:
        lookup: timestamp -> price dict
        ts: Target timestamp (epoch seconds)

    Returns:
        Price at or nearest to the timestamp, or None if lookup is empty
    """
    if not lookup:
        return None

    ts_int = int(ts)

    # Exact match
    if ts_int in lookup:
        return lookup[ts_int]

    # Find nearest (within 120 seconds — covers 1m bar resolution)
    for offset in range(1, 121):
        if ts_int - offset in lookup:
            return lookup[ts_int - offset]
        if ts_int + offset in lookup:
            return lookup[ts_int + offset]

    # Fall back to closest key
    keys = sorted(lookup.keys())
    if ts_int <= keys[0]:
        return lookup[keys[0]]
    if ts_int >= keys[-1]:
        return lookup[keys[-1]]

    # Binary search for closest
    import bisect
    idx = bisect.bisect_left(keys, ts_int)
    if idx == 0:
        return lookup[keys[0]]
    if idx >= len(keys):
        return lookup[keys[-1]]
    before = keys[idx - 1]
    after = keys[idx]
    if ts_int - before <= after - ts_int:
        return lookup[before]
    return lookup[after]


# ── Realized Volatility ─────────────────────────────────────────────

def realized_vol_from_prices(prices: List[float], seconds_per_bar: float = 1.0) -> float:
    """
    Calculate annualized realized volatility from a price series.

    Args:
        prices: List of close prices (at regular intervals)
        seconds_per_bar: Time between consecutive prices in seconds

    Returns:
        Annualized realized volatility (e.g., 0.50 = 50%)
    """
    if len(prices) < 2:
        return 0.0

    arr = np.array(prices, dtype=np.float64)
    # Filter out zeros/negatives
    arr = arr[arr > 0]
    if len(arr) < 2:
        return 0.0

    log_returns = np.diff(np.log(arr))
    if len(log_returns) == 0:
        return 0.0

    # Annualize: std of per-bar returns * sqrt(bars_per_year)
    bars_per_year = 365.25 * 24 * 3600 / seconds_per_bar
    return float(np.std(log_returns) * np.sqrt(bars_per_year))


def compute_rolling_vol(
    klines: List[Tuple[int, float, float, float, float, float]],
    window_seconds: int = 300,
) -> dict:
    """
    Pre-compute rolling realized vol at each timestamp from kline data.

    Auto-detects bar interval from the data (works with both 1s and 1m bars).

    Args:
        klines: List of (ts, o, h, l, c, v) tuples
        window_seconds: Lookback window in seconds

    Returns:
        Dict mapping timestamp -> annualized realized vol
    """
    if not klines:
        return {}

    # Detect bar interval from data
    if len(klines) >= 2:
        diffs = [klines[i+1][0] - klines[i][0] for i in range(min(10, len(klines)-1))]
        bar_interval = max(1, int(np.median(diffs)))
    else:
        bar_interval = 1

    # Window in number of bars
    window_bars = max(2, window_seconds // bar_interval)

    vol_lookup = {}
    closes = []

    for ts, o, h, l, c, v in klines:
        closes.append(c)

        # Only compute vol once we have enough data
        if len(closes) >= max(2, window_bars // 2):
            window = closes[-window_bars:] if len(closes) >= window_bars else closes
            vol_lookup[ts] = realized_vol_from_prices(window, seconds_per_bar=float(bar_interval))
        else:
            vol_lookup[ts] = 0.0

    return vol_lookup


# ── WebSocket Feed (for live trading) ───────────────────────────────

class BinancePriceFeed:
    """
    Live BTC price feed via Binance WebSocket.

    Connects to wss://stream.binance.com:9443/ws/btcusdt@trade
    Maintains current price, 1-second OHLC bars, and price history.
    """

    def __init__(self, symbol: str = "btcusdt", max_history: int = 3600):
        """
        Args:
            symbol: Binance symbol in lowercase (default: btcusdt)
            max_history: Max 1-second bars to keep (default: 3600 = 1 hour)
        """
        self.symbol = symbol.lower()
        # Will try multiple WS URLs in connect()
        self._ws_urls = [f"{base}/{self.symbol}@trade" for base in _BINANCE_WS_URLS]

        # Current state
        self._current_price: float = 0.0
        self._last_update: float = 0.0

        # 1-second OHLC bars: deque of (timestamp, open, high, low, close)
        self._bars: deque = deque(maxlen=max_history)
        self._current_bar_ts: int = 0
        self._current_bar: Optional[List[float]] = None  # [o, h, l, c]

        # Callbacks
        self._price_callbacks: List[Callable] = []

        # Connection state
        self._ws = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @property
    def is_connected(self) -> bool:
        return self._running and self._ws is not None

    def get_price(self) -> float:
        """Get current BTC spot price."""
        return self._current_price

    def get_last_update(self) -> float:
        """Get timestamp of last price update."""
        return self._last_update

    def on_price(self, callback: Callable) -> Callable:
        """Register price update callback (decorator)."""
        self._price_callbacks.append(callback)
        return callback

    def get_recent_closes(self, window_seconds: int = 300) -> List[float]:
        """Get recent 1-second close prices."""
        bars = list(self._bars)
        if not bars:
            return []
        return [b[4] for b in bars[-window_seconds:]]

    def get_realized_vol(self, window_seconds: int = 300) -> float:
        """
        Get annualized realized volatility from recent 1-second bars.

        Args:
            window_seconds: Number of seconds to look back

        Returns:
            Annualized realized vol (e.g., 0.50 = 50%)
        """
        closes = self.get_recent_closes(window_seconds)
        return realized_vol_from_prices(closes, seconds_per_bar=1.0)

    def _process_trade(self, price: float, trade_time_ms: int) -> None:
        """Process a single trade message."""
        self._current_price = price
        self._last_update = time.time()

        # Update 1-second bar
        bar_ts = trade_time_ms // 1000
        if bar_ts != self._current_bar_ts:
            # Save completed bar
            if self._current_bar is not None:
                self._bars.append((
                    self._current_bar_ts,
                    self._current_bar[0],  # open
                    self._current_bar[1],  # high
                    self._current_bar[2],  # low
                    self._current_bar[3],  # close
                ))
            # Start new bar
            self._current_bar_ts = bar_ts
            self._current_bar = [price, price, price, price]
        else:
            if self._current_bar is not None:
                self._current_bar[1] = max(self._current_bar[1], price)  # high
                self._current_bar[2] = min(self._current_bar[2], price)  # low
                self._current_bar[3] = price  # close

        # Fire callbacks
        for cb in self._price_callbacks:
            try:
                cb(price)
            except Exception:
                pass

    async def connect(self) -> bool:
        """Connect to Binance WebSocket. Tries .com first, then .us."""
        import websockets

        for ws_url in self._ws_urls:
            try:
                self._ws = await asyncio.wait_for(
                    websockets.connect(ws_url), timeout=5.0
                )
                self._running = True
                self._task = asyncio.create_task(self._read_loop())
                return True
            except Exception:
                continue
        return False

    async def _read_loop(self) -> None:
        """Read messages from WebSocket."""
        try:
            async for msg in self._ws:
                if not self._running:
                    break
                try:
                    data = json.loads(msg)
                    price = float(data["p"])
                    trade_time = int(data["T"])
                    self._process_trade(price, trade_time)
                except (KeyError, ValueError):
                    continue
        except Exception:
            pass
        finally:
            self._running = False

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def wait_for_price(self, timeout: float = 10.0) -> bool:
        """Wait until we have a price."""
        start = time.time()
        while time.time() - start < timeout:
            if self._current_price > 0:
                return True
            await asyncio.sleep(0.1)
        return False
