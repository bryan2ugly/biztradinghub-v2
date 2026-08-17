"""
Composite BTC/USD price feed -- mirrors the Pine script's 3-exchange blend
(Coinbase, Kraken, Gemini) as a proxy for Kalshi's settlement index.

No API keys needed; these are public market-data endpoints.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests

from config import FeedConfig


@dataclass
class ExchangeQuote:
    price: Optional[float] = None
    prev_price: Optional[float] = None
    ok: bool = False

    @property
    def is_fresh(self) -> bool:
        if self.price is None or self.prev_price is None:
            return True
        return self.price != self.prev_price


class CompositePriceFeed:
    def __init__(self, cfg: FeedConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.coinbase = ExchangeQuote()
        self.kraken = ExchangeQuote()
        self.gemini = ExchangeQuote()

    def _fetch_coinbase(self) -> Optional[float]:
        try:
            r = self.session.get("https://api.coinbase.com/v2/prices/BTC-USD/spot", timeout=5)
            return float(r.json()["data"]["amount"])
        except Exception:
            return None

    def _fetch_kraken(self) -> Optional[float]:
        try:
            r = self.session.get("https://api.kraken.com/0/public/Ticker?pair=XBTUSD", timeout=5)
            data = r.json()["result"]
            key = next(iter(data))
            return float(data[key]["c"][0])
        except Exception:
            return None

    def _fetch_gemini(self) -> Optional[float]:
        try:
            r = self.session.get("https://api.gemini.com/v1/pubticker/btcusd", timeout=5)
            return float(r.json()["last"])
        except Exception:
            return None

    def refresh(self) -> Optional[float]:
        """Fetch fresh quotes from every enabled exchange and recompute the
        weighted composite price. Returns None only if every feed failed."""
        if self.cfg.use_coinbase:
            price = self._fetch_coinbase()
            if price is not None:
                self.coinbase.prev_price = self.coinbase.price
                self.coinbase.price = price
                self.coinbase.ok = True
            else:
                self.coinbase.ok = False

        if self.cfg.use_kraken:
            price = self._fetch_kraken()
            if price is not None:
                self.kraken.prev_price = self.kraken.price
                self.kraken.price = price
                self.kraken.ok = True
            else:
                self.kraken.ok = False

        if self.cfg.use_gemini:
            price = self._fetch_gemini()
            if price is not None:
                self.gemini.prev_price = self.gemini.price
                self.gemini.price = price
                self.gemini.ok = True
            else:
                self.gemini.ok = False

        return self.composite_price()

    def _effective_weight(self, quote: ExchangeQuote, base_weight: float) -> float:
        if not quote.ok or quote.price is None:
            return 0.0
        return base_weight * (1.0 if quote.is_fresh else self.cfg.stale_weight_multiplier)

    def composite_price(self) -> Optional[float]:
        w_cb = self._effective_weight(self.coinbase, self.cfg.weight_coinbase)
        w_kr = self._effective_weight(self.kraken, self.cfg.weight_kraken)
        w_gm = self._effective_weight(self.gemini, self.cfg.weight_gemini)
        total = w_cb + w_kr + w_gm
        if total <= 0:
            # fall back to whatever single feed is available
            for q in (self.coinbase, self.kraken, self.gemini):
                if q.ok and q.price is not None:
                    return q.price
            return None
        weighted = 0.0
        if self.coinbase.ok:
            weighted += self.coinbase.price * w_cb
        if self.kraken.ok:
            weighted += self.kraken.price * w_kr
        if self.gemini.ok:
            weighted += self.gemini.price * w_gm
        return weighted / total

    def exchange_count(self) -> int:
        return sum(1 for q in (self.coinbase, self.kraken, self.gemini) if q.ok)

    def exchange_spread_usd(self) -> Optional[float]:
        """Max - min price across whichever exchanges are currently live --
        a real disagreement signal that was previously invisible: a wide
        spread between exchanges (thin liquidity, an outlier print, one
        feed lagging) is meaningfully different from all three agreeing
        closely, even if the composite price looks the same either way.
        Returns None if fewer than 2 exchanges are currently reporting."""
        prices = [q.price for q in (self.coinbase, self.kraken, self.gemini) if q.ok and q.price is not None]
        if len(prices) < 2:
            return None
        return max(prices) - min(prices)
