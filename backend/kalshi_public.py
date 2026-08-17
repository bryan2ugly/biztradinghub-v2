"""
Public, read-only Kalshi lookup -- fetches the REAL strike price for the
currently-open 15-min BTC market, instead of approximating it from our
own price feed. Deliberately contains no signing/auth code whatsoever
(not even a stub) -- Kalshi's market-listing endpoints don't require a
key, so there's nothing here that could ever touch your account even by
mistake.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

import requests

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXBTC15M"

_session = requests.Session()


def get_open_markets(series_ticker: str = SERIES_TICKER, limit: int = 50) -> list:
    resp = _session.get(
        f"{KALSHI_BASE_URL}/markets",
        params={"status": "open", "series_ticker": series_ticker, "limit": limit},
        timeout=10,
    )
    resp.raise_for_status()
    return (resp.json() or {}).get("markets", [])


def find_current_15m_btc_market(series_ticker: str = SERIES_TICKER) -> Optional[dict]:
    markets = get_open_markets(series_ticker)
    if not markets:
        return None
    now_iso = datetime.now(timezone.utc).isoformat()
    future_markets = [m for m in markets if m.get("close_time") and m["close_time"] > now_iso]
    candidates = future_markets or markets
    candidates.sort(key=lambda m: m.get("close_time", ""))
    return candidates[0]


def extract_strike_from_market(market: dict) -> Optional[float]:
    if not market:
        return None
    for field in ("floor_strike", "cap_strike", "strike_price", "strike"):
        val = market.get(field)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    ticker = market.get("ticker", "")
    match = re.search(r"-T([0-9]+(?:\.[0-9]+)?)$", ticker)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def get_current_kalshi_strike() -> dict:
    """Returns {'strike': float|None, 'ticker': str|None, 'close_time': str|None, 'error': str|None}."""
    try:
        market = find_current_15m_btc_market()
        if not market:
            return {"strike": None, "ticker": None, "close_time": None, "error": "no open market found"}
        return {
            "strike": extract_strike_from_market(market),
            "ticker": market.get("ticker"),
            "close_time": market.get("close_time"),
            "error": None,
        }
    except Exception as e:
        return {"strike": None, "ticker": None, "close_time": None, "error": str(e)}
