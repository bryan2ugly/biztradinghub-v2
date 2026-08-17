"""Public Coinbase 1-min BTC candle fetcher -- no API key needed."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import requests

COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
GRANULARITY_SECONDS = 60
MAX_CANDLES_PER_REQUEST = 300
MAX_RETRIES_PER_CHUNK = 4  # gives up on a rate-limited chunk rather than retrying forever
FETCH_TIME_BUDGET_SECONDS = 25  # hard cap so a slow/rate-limited backfill can never block startup indefinitely


def fetch_1m_candles(start: datetime, end: datetime, pause: float = 0.35) -> List[Tuple[datetime, float, float, float, float]]:
    """Returns a chronological list of (timestamp, open, high, low, close).
    Best-effort within a time budget -- if Coinbase is rate-limiting or
    slow, this returns whatever it managed to fetch rather than hanging;
    a partial history is much better than blocking the whole server."""
    candles = []
    chunk_span = timedelta(seconds=GRANULARITY_SECONDS * (MAX_CANDLES_PER_REQUEST - 1))
    cur_start = start
    session = requests.Session()
    fetch_deadline = time.monotonic() + FETCH_TIME_BUDGET_SECONDS

    while cur_start < end:
        if time.monotonic() > fetch_deadline:
            print(f"fetch_1m_candles: hit time budget, returning {len(candles)} candles so far")
            break
        cur_end = min(cur_start + chunk_span, end)
        params = {
            "start": cur_start.isoformat(),
            "end": cur_end.isoformat(),
            "granularity": GRANULARITY_SECONDS,
        }
        retries = 0
        while True:
            try:
                resp = session.get(COINBASE_CANDLES_URL, params=params, timeout=15)
            except requests.RequestException as e:
                print(f"fetch_1m_candles: request failed ({e}), skipping this chunk")
                resp = None
                break
            if resp.status_code == 429:
                retries += 1
                if retries > MAX_RETRIES_PER_CHUNK or time.monotonic() > fetch_deadline:
                    print("fetch_1m_candles: rate-limited too many times, skipping this chunk")
                    resp = None
                    break
                time.sleep(2.0)
                continue
            break
        if resp is not None:
            try:
                resp.raise_for_status()
                rows = resp.json()  # [time, low, high, open, close, volume], newest first
                for row in rows:
                    ts = datetime.fromtimestamp(row[0], tz=timezone.utc)
                    low, high, o, c = row[1], row[2], row[3], row[4]
                    candles.append((ts, o, high, low, c))
            except (requests.RequestException, ValueError, KeyError, IndexError) as e:
                print(f"fetch_1m_candles: bad response for a chunk ({e}), skipping it")
        cur_start = cur_end
        time.sleep(pause)

    candles.sort(key=lambda r: r[0])
    seen = set()
    deduped = []
    for row in candles:
        if row[0] in seen:
            continue
        seen.add(row[0])
        deduped.append(row)
    return deduped


def fetch_1m_candles_with_volume(start: datetime, end: datetime, pause: float = 0.35):
    """Same as fetch_1m_candles, but also returns each bar's trade volume
    -- needed for the anchored-VWAP overlay. Returns (ts, o, h, l, c, volume).
    Same best-effort time budget as fetch_1m_candles -- see there for why."""
    candles = []
    chunk_span = timedelta(seconds=GRANULARITY_SECONDS * (MAX_CANDLES_PER_REQUEST - 1))
    cur_start = start
    session = requests.Session()
    fetch_deadline = time.monotonic() + FETCH_TIME_BUDGET_SECONDS

    while cur_start < end:
        if time.monotonic() > fetch_deadline:
            print(f"fetch_1m_candles_with_volume: hit time budget, returning {len(candles)} candles so far")
            break
        cur_end = min(cur_start + chunk_span, end)
        params = {
            "start": cur_start.isoformat(),
            "end": cur_end.isoformat(),
            "granularity": GRANULARITY_SECONDS,
        }
        retries = 0
        while True:
            try:
                resp = session.get(COINBASE_CANDLES_URL, params=params, timeout=15)
            except requests.RequestException as e:
                print(f"fetch_1m_candles_with_volume: request failed ({e}), skipping this chunk")
                resp = None
                break
            if resp.status_code == 429:
                retries += 1
                if retries > MAX_RETRIES_PER_CHUNK or time.monotonic() > fetch_deadline:
                    print("fetch_1m_candles_with_volume: rate-limited too many times, skipping this chunk")
                    resp = None
                    break
                time.sleep(2.0)
                continue
            break
        if resp is not None:
            try:
                resp.raise_for_status()
                rows = resp.json()
                for row in rows:
                    ts = datetime.fromtimestamp(row[0], tz=timezone.utc)
                    low, high, o, c, vol = row[1], row[2], row[3], row[4], row[5]
                    candles.append((ts, o, high, low, c, vol))
            except (requests.RequestException, ValueError, KeyError, IndexError) as e:
                print(f"fetch_1m_candles_with_volume: bad response for a chunk ({e}), skipping it")
        cur_start = cur_end
        time.sleep(pause)

    candles.sort(key=lambda r: r[0])
    seen = set()
    deduped = []
    for row in candles:
        if row[0] in seen:
            continue
        seen.add(row[0])
        deduped.append(row)
    return deduped
