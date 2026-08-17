"""
Chart overlay computations: anchored VWAP + deviation bands, exhaustion
markers, and pivot-based support/resistance lines. This is a straight
adaptation of the same overlay math your local-only webchart.py uses
(public price+volume data only) -- not a reimplementation, so it stays
correct as your bot's math gets tuned.
"""
from __future__ import annotations

import pandas as pd

from signal_engine import pivots, rsi


def compute_overlays(candles_with_volume, pivot_left: int, pivot_right: int, rsi_len: int):
    """candles_with_volume: list of (ts, open, high, low, close, volume) tuples,
    chronological. Returns dict of chart-ready overlay series."""
    df = pd.DataFrame(candles_with_volume, columns=["time", "open", "high", "low", "close", "volume"])
    if df.empty:
        return {"vwap": [], "upper1": [], "lower1": [], "upper2": [], "lower2": [],
                "resistance": [], "support": [], "exhaustion_markers": [], "rsi": []}

    df["typical"] = (df["high"] + df["low"] + df["close"]) / 3

    vwap, upper1, lower1, upper2, lower2 = [], [], [], [], []
    cum_pv = cum_v = sum_sq_d = 0.0
    bar_cnt = 0
    for _, row in df.iterrows():
        is_new_anchor = row["time"].minute % 15 == 0
        if is_new_anchor:
            cum_pv = cum_v = sum_sq_d = 0.0
            bar_cnt = 0
        cum_pv += row["typical"] * row["volume"]
        cum_v += row["volume"]
        bar_cnt += 1
        v = cum_pv / cum_v if cum_v > 0 else row["typical"]
        dev = row["typical"] - v
        sum_sq_d += dev * dev
        stdev = (sum_sq_d / bar_cnt) ** 0.5 if bar_cnt > 1 else 0.0
        vwap.append(v)
        upper1.append(v + stdev)
        lower1.append(v - stdev)
        upper2.append(v + stdev * 2)
        lower2.append(v - stdev * 2)

    df["vwap"], df["upper1"], df["lower1"], df["upper2"], df["lower2"] = vwap, upper1, lower1, upper2, lower2

    body = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    bearish_exhaustion = (df["high"] > df["upper2"]) & (df["close"] < df["upper2"]) & (upper_wick > body)
    bullish_exhaustion = (df["low"] < df["lower2"]) & (df["close"] > df["lower2"]) & (lower_wick > body)

    exhaustion_markers = []
    for i, row in df.iterrows():
        t = int(row["time"].timestamp())
        if bearish_exhaustion.iloc[i]:
            exhaustion_markers.append({"time": t, "shape": "arrowDown", "position": "aboveBar",
                                        "color": "#ba68c8", "text": "exh"})
        elif bullish_exhaustion.iloc[i]:
            exhaustion_markers.append({"time": t, "shape": "arrowUp", "position": "belowBar",
                                        "color": "#ba68c8", "text": "exh"})

    ph = pivots(df["high"], pivot_left, pivot_right, "high")
    pl = pivots(df["low"], pivot_left, pivot_right, "low")
    resistance, support = [], []
    last_res = last_sup = None
    for i, row in df.iterrows():
        if ph.iloc[i]:
            last_res = float(row["high"])
        if pl.iloc[i]:
            last_sup = float(row["low"])
        t = int(row["time"].timestamp())
        if last_res is not None:
            resistance.append({"time": t, "value": last_res})
        if last_sup is not None:
            support.append({"time": t, "value": last_sup})

    def line(col):
        return [{"time": int(t.timestamp()), "value": v} for t, v in zip(df["time"], df[col])]

    df["rsi"] = rsi(df["close"].astype(float), rsi_len)

    return {
        "vwap": line("vwap"), "upper1": line("upper1"), "lower1": line("lower1"),
        "upper2": line("upper2"), "lower2": line("lower2"),
        "resistance": resistance, "support": support,
        "exhaustion_markers": exhaustion_markers,
        "rsi": line("rsi"),
    }
