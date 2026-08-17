"""
Multi-candle chart structure scanner -- bull/bear flags and double
top/bottom. These are different from patterns.py's candlestick patterns:
those read one candle's shape, these read the SHAPE OF SEVERAL BARS
TOGETHER (a sharp move + a contained pullback, or two matching peaks).

Deliberately scoped to two structure families (flags, double top/bottom)
rather than trying to match every pattern a commercial scanner offers
(e.g. cup & handle) -- these two are well-defined enough to detect
reliably with simple trendline fitting; more exotic ones would need a
lot more tuning to avoid false positives, which isn't worth it for a
learning-hub-style scanner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signal_engine import pivots


def _linfit(y: np.ndarray):
    """Least-squares line through y vs bar index. Returns (slope,
    intercept, r_squared) -- r_squared close to 1 means the points sit
    tightly on the line (a clean channel), close to 0 means scattered."""
    n = len(y)
    if n < 2:
        return 0.0, float(y[0]) if n else 0.0, 0.0
    x = np.arange(n)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(max(0.0, min(1.0, r2)))


def detect_flag(df: pd.DataFrame, pole_window: int = 10, flag_window: int = 10):
    """Looks for: a sharp directional 'pole' move, then a contained,
    roughly-parallel-channel pullback (the 'flag'). Returns the best
    candidate (bull or bear, whichever is cleaner right now) or None.
    Deliberately strict -- tested against pure random-walk noise to keep
    the false-positive rate low rather than flagging every wiggle."""
    if len(df) < pole_window + flag_window + 5:
        return None

    flag = df.iloc[-flag_window:]
    pole = df.iloc[-(pole_window + flag_window):-flag_window]
    if len(pole) < 4 or len(flag) < 4:
        return None

    pole_move = pole["close"].iloc[-1] - pole["close"].iloc[0]
    atr = (df["high"] - df["low"]).rolling(14, min_periods=5).mean().iloc[-1]
    if not atr or atr <= 0:
        return None

    direction = "UP" if pole_move > 0 else "DOWN"
    pole_strength = abs(pole_move) / atr  # how many ATRs the pole covered
    if pole_strength < 2.8:
        return None  # not a sharp enough move to call it a pole

    # the pole itself should be mostly one-directional, not a lucky net
    # result of a choppy back-and-forth
    pole_closes = pole["close"].values
    pole_diffs = np.diff(pole_closes)
    same_dir_frac = np.mean(pole_diffs > 0) if direction == "UP" else np.mean(pole_diffs < 0)
    if same_dir_frac < 0.7:
        return None

    # channel: fit trendlines to the flag's highs and lows
    _, _, r2_high = _linfit(flag["high"].values)
    slope_low, intercept_low, r2_low = _linfit(flag["low"].values)
    slope_high, intercept_high, _ = _linfit(flag["high"].values)
    containment = (r2_high + r2_low) / 2

    # a real flag drifts opposite (or flat) vs the pole, and stays inside
    # roughly half the pole's range -- otherwise it's just more trend
    flag_drift = flag["close"].iloc[-1] - flag["close"].iloc[0]
    counter_trend = (direction == "UP" and flag_drift <= abs(pole_move) * 0.5) or \
                     (direction == "DOWN" and flag_drift >= -abs(pole_move) * 0.5)
    if not counter_trend:
        return None
    if containment < 0.45:
        return None  # too scattered to call it a channel

    quality = round(min(100, max(0, (
        containment * 55 +                          # tightness of the channel
        min(pole_strength / 4, 1) * 30 +             # strength of the pole
        min(len(flag) / flag_window, 1) * 15         # how developed the flag is
    ))))
    if quality < 62:
        return None  # didn't clear the bar to call it a confident find

    n = len(flag)
    upper_now = slope_high * (n - 1) + intercept_high
    lower_now = slope_low * (n - 1) + intercept_low

    return {
        "direction": "Bull Flag" if direction == "UP" else "Bear Flag",
        "bias": "UP" if direction == "UP" else "DOWN",
        "quality": quality,
        "pole_start": {"time": int(pole.index[0].timestamp()), "price": float(pole["close"].iloc[0])},
        "pole_end": {"time": int(pole.index[-1].timestamp()), "price": float(pole["close"].iloc[-1])},
        "upper_line": [
            {"time": int(flag.index[0].timestamp()), "value": float(intercept_high)},
            {"time": int(flag.index[-1].timestamp()), "value": float(upper_now)},
        ],
        "lower_line": [
            {"time": int(flag.index[0].timestamp()), "value": float(intercept_low)},
            {"time": int(flag.index[-1].timestamp()), "value": float(lower_now)},
        ],
        "trigger": float(upper_now) if direction == "UP" else float(lower_now),
        "invalidate": float(lower_now) if direction == "UP" else float(upper_now),
        "label_time": int(flag.index[-1].timestamp()),
        "label_price": float(flag["high"].max() if direction == "UP" else flag["low"].min()),
    }


def detect_double_top_bottom(df: pd.DataFrame, pivot_left: int, pivot_right: int,
                              tolerance_pct: float = 0.08, min_gap_bars: int = 5,
                              min_depth_pct: float = 0.18):
    """Two comparable pivot highs (double top) or pivot lows (double
    bottom) with a meaningful pullback between them. Only checks the
    MOST RECENT pivot against earlier ones (a structure relevant right
    now, not any matching pair anywhere in history), and requires a
    real pullback depth -- both deliberately strict, tested against
    pure random-walk noise to keep false positives low."""
    ph = pivots(df["high"], pivot_left, pivot_right, "high")
    pl = pivots(df["low"], pivot_left, pivot_right, "low")

    high_points = [(i, float(df["high"].iloc[i])) for i in range(len(df)) if bool(ph.iloc[i])]
    low_points = [(i, float(df["low"].iloc[i])) for i in range(len(df)) if bool(pl.iloc[i])]

    def best_pair(points, between_extreme_fn):
        """Only pairs the LAST pivot in `points` against earlier ones --
        i.e. is there a double top/bottom active right now, not any
        matching pair that ever existed in the lookback window."""
        if len(points) < 2:
            return None
        i2, p2 = points[-1]
        best = None
        for i1, p1 in points[-6:-1]:  # only look a handful of pivots back
            if i2 - i1 < min_gap_bars:
                continue
            if abs(p2 - p1) / max(p1, p2) * 100 > tolerance_pct:
                continue
            mid_extreme = between_extreme_fn(i1, i2)
            if mid_extreme is None:
                continue
            depth_pct = abs(mid_extreme - (p1 + p2) / 2) / ((p1 + p2) / 2) * 100
            if depth_pct < min_depth_pct:
                continue
            quality = round(min(100, max(0, (
                (1 - abs(p2 - p1) / max(p1, p2) / (tolerance_pct / 100)) * 45 +
                min(depth_pct / (min_depth_pct * 2.5), 1) * 40 +
                min((i2 - i1) / (min_gap_bars * 3), 1) * 15
            ))))
            cand = {"i1": i1, "p1": p1, "i2": i2, "p2": p2, "mid": mid_extreme, "quality": quality}
            if best is None or quality > best["quality"]:
                best = cand
        return best

    def mid_low(i1, i2):
        seg = df["low"].iloc[i1:i2 + 1]
        return float(seg.min()) if len(seg) else None

    def mid_high(i1, i2):
        seg = df["high"].iloc[i1:i2 + 1]
        return float(seg.max()) if len(seg) else None

    top = best_pair(high_points, mid_low)
    bottom = best_pair(low_points, mid_high)

    # only the higher-quality one gets shown, to keep the chart clean
    candidates = []
    if top and top["quality"] >= 65:
        candidates.append(("Double Top", "DOWN", top))
    if bottom and bottom["quality"] >= 65:
        candidates.append(("Double Bottom", "UP", bottom))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[2]["quality"], reverse=True)
    name, bias, c = candidates[0]

    return {
        "name": name,
        "bias": bias,
        "quality": c["quality"],
        "peak1": {"time": int(df.index[c["i1"]].timestamp()), "price": c["p1"]},
        "peak2": {"time": int(df.index[c["i2"]].timestamp()), "price": c["p2"]},
        "neckline": c["mid"],
        "label_time": int(df.index[c["i2"]].timestamp()),
        "label_price": c["p2"],
    }
