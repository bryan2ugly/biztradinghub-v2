"""
Full candlestick pattern library -- 50 classic single/two/three-candle
patterns, vectorized across the whole bar history at once (same style as
signal_engine.py's original 10-pattern detect_candlestick_patterns, just
much larger). This is the single source of truth for pattern names/bias
used by both the engine (for pattern_text) and the site's pattern gallery
(via /api/pattern_catalog), so the two can never drift out of sync.

Bias is UP / DOWN / NEUTRAL -- NEUTRAL patterns (dojis, spinning tops,
inside bar, tri-star) signal indecision/compression rather than a
directional lean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = (df["open"].astype(float), df["high"].astype(float),
                  df["low"].astype(float), df["close"].astype(float))
    o1, h1, l1, c1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
    o2, h2, l2, c2 = o.shift(2), h.shift(2), l.shift(2), c.shift(2)

    rng = (h - l).replace(0, np.nan)
    rng1 = (h1 - l1).replace(0, np.nan)
    body = (c - o).abs()
    body1 = (c1 - o1).abs()
    body2 = (c2 - o2).abs()
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
    upper_wick1 = h1 - pd.concat([o1, c1], axis=1).max(axis=1)
    lower_wick1 = pd.concat([o1, c1], axis=1).min(axis=1) - l1

    is_bull, is_bear = c > o, c < o
    is_bull1, is_bear1 = c1 > o1, c1 < o1
    is_bull2, is_bear2 = c2 > o2, c2 < o2

    avg_body = body.rolling(10, min_periods=3).mean()
    long_body = body > avg_body  # "long-bodied" relative to recent bars
    small_body = body <= rng * 0.1
    doji_body = body <= rng * 0.05

    out = pd.DataFrame(index=df.index)

    # ---- single-candle -------------------------------------------------
    out["doji"] = (rng > 0) & (body <= rng * 0.1) & ~(lower_wick > body * 2) & ~(upper_wick > body * 2)
    out["long_legged_doji"] = (rng > 0) & doji_body & (upper_wick > rng * 0.3) & (lower_wick > rng * 0.3)
    out["dragonfly_doji"] = (rng > 0) & doji_body & (lower_wick > rng * 0.6) & (upper_wick < rng * 0.1)
    out["gravestone_doji"] = (rng > 0) & doji_body & (upper_wick > rng * 0.6) & (lower_wick < rng * 0.1)
    out["hammer"] = (rng > 0) & (lower_wick > body * 2) & (upper_wick < body) & is_bull
    out["inverted_hammer"] = (rng > 0) & (upper_wick > body * 2) & (lower_wick < body) & is_bull
    out["hanging_man"] = (rng > 0) & (lower_wick > body * 2) & (upper_wick < body) & is_bear
    out["shooting_star"] = (rng > 0) & (upper_wick > body * 2) & (lower_wick < body) & is_bear
    out["bullish_marubozu"] = is_bull & (body > 0) & ((o - l) <= body * 0.05) & ((h - c) <= body * 0.05)
    out["bearish_marubozu"] = is_bear & (body > 0) & ((h - o) <= body * 0.05) & ((c - l) <= body * 0.05)
    out["spinning_top"] = (rng > 0) & small_body & (upper_wick > body) & (lower_wick > body) & ~doji_body

    # ---- two-candle -----------------------------------------------------
    out["bullish_engulfing"] = is_bear1 & is_bull & (c > o1) & (o <= c1)
    out["bearish_engulfing"] = is_bull1 & is_bear & (c < o1) & (o >= c1)
    out["bullish_harami"] = is_bear1 & long_body.shift(1).fillna(False) & is_bull & (o > c1) & (c < o1)
    out["bearish_harami"] = is_bull1 & long_body.shift(1).fillna(False) & is_bear & (o < c1) & (c > o1)
    out["bullish_harami_cross"] = is_bear1 & (body <= rng * 0.1) & (o > c1) & (c < o1)
    out["bearish_harami_cross"] = is_bull1 & (body <= rng * 0.1) & (o < c1) & (c > o1)
    mid1 = (o1 + c1) / 2
    out["piercing_line"] = is_bear1 & is_bull & (o < l1) & (c > mid1) & (c < o1)
    out["dark_cloud_cover"] = is_bull1 & is_bear & (o > h1) & (c < mid1) & (c > o1)
    out["tweezer_bottom"] = (l.sub(l1).abs() <= rng1 * 0.1) & is_bear1 & is_bull
    out["tweezer_top"] = (h.sub(h1).abs() <= rng1 * 0.1) & is_bull1 & is_bear
    out["bullish_kicker"] = is_bear1 & is_bull & (o >= o1.combine(c1, max)) & (o > h1)
    out["bearish_kicker"] = is_bull1 & is_bear & (o <= o1.combine(c1, min)) & (o < l1)
    out["in_neck_line"] = is_bear1 & is_bull & (o < l1) & (c.sub(c1).abs() <= body1 * 0.1) & (c <= o1)
    out["thrusting_line"] = is_bear1 & is_bull & (o < l1) & (c > c1) & (c < mid1)
    out["bullish_counterattack"] = is_bear1 & is_bull & (c.sub(c1).abs() <= rng1 * 0.05) & long_body & long_body.shift(1).fillna(False)
    out["bearish_counterattack"] = is_bull1 & is_bear & (c.sub(c1).abs() <= rng1 * 0.05) & long_body & long_body.shift(1).fillna(False)
    out["inside_bar"] = (h < h1) & (l > l1)

    # ---- three-candle ----------------------------------------------------
    mid_body2 = (o2 + c2) / 2
    star_gap_up = pd.concat([o1, c1], axis=1).min(axis=1) > c2
    star_gap_down = pd.concat([o1, c1], axis=1).max(axis=1) < c2
    out["morning_star"] = is_bear2 & long_body.shift(2).fillna(False) & (body1 <= rng1 * 0.3) & star_gap_up.fillna(False) & is_bull & (c > mid_body2)
    out["evening_star"] = is_bull2 & long_body.shift(2).fillna(False) & (body1 <= rng1 * 0.3) & star_gap_down.fillna(False) & is_bear & (c < mid_body2)
    is_doji1 = (rng1 > 0) & (body1 <= rng1 * 0.1)
    out["morning_doji_star"] = is_bear2 & is_doji1 & star_gap_up.fillna(False) & is_bull & (c > mid_body2)
    out["evening_doji_star"] = is_bull2 & is_doji1 & star_gap_down.fillna(False) & is_bear & (c < mid_body2)
    out["three_white_soldiers"] = (is_bull & is_bull1 & is_bull2 & (c > c1) & (c1 > c2) &
                                    (o > o1) & (o < c1) & (o1 > o2) & (o1 < c2))
    out["three_black_crows"] = (is_bear & is_bear1 & is_bear2 & (c < c1) & (c1 < c2) &
                                 (o < o1) & (o > c1) & (o1 < o2) & (o1 > c2))
    out["three_inside_up"] = out["bullish_harami"].shift(1).fillna(False) & is_bull & (c > o1)
    out["three_inside_down"] = out["bearish_harami"].shift(1).fillna(False) & is_bear & (c < o1)
    out["three_outside_up"] = out["bullish_engulfing"].shift(1).fillna(False) & is_bull & (c > c1)
    out["three_outside_down"] = out["bearish_engulfing"].shift(1).fillna(False) & is_bear & (c < c1)
    out["bullish_abandoned_baby"] = is_bear2 & is_doji1 & (l1 > h2) & is_bull & (l > h1)
    out["bearish_abandoned_baby"] = is_bull2 & is_doji1 & (h1 < l2) & is_bear & (h < l1)
    out["three_stars_south"] = (is_bear & is_bear1 & is_bear2 & (l >= l1) & (l1 >= l2) &
                                 (body < body1) & (body1 < body2))
    out["advance_block"] = (is_bull & is_bull1 & is_bull2 & (c > c1) & (c1 > c2) &
                             (upper_wick > upper_wick1) & (body <= body1))
    out["deliberation"] = (is_bull & is_bull1 & is_bull2 & (c > c1) & (c1 > c2) &
                            (body < body1 * 0.6) & long_body.shift(1).fillna(False))
    out["stick_sandwich"] = (is_bear & is_bull1 & is_bear2 & (c.sub(c2).abs() <= rng * 0.05) & (o1 > c2))
    out["upside_gap_two_crows"] = (is_bull2 & is_bear1 & (o1 > h2) & is_bear &
                                    (o > o1) & (c < c1) & (c > c2))
    out["identical_three_crows"] = (is_bear & is_bear1 & is_bear2 & (c < c1) & (c1 < c2) &
                                     (o.sub(c1).abs() <= rng * 0.05) & (o1.sub(c2).abs() <= rng1 * 0.05))
    out["unique_three_river"] = (is_bear2 & long_body.shift(2).fillna(False) & is_bear1 & (l1 < l2) & (c1 > c2) &
                                  is_bull & (o < c1) & (c < c1))
    is_doji2 = (rng.shift(2) > 0) & (body2 <= rng.shift(2) * 0.1)
    out["tri_star"] = is_doji2 & is_doji1 & out["doji"]
    out["three_bar_push_up"] = is_bull & is_bull1 & is_bull2 & (c > c1) & (c1 > c2)
    out["three_bar_push_down"] = is_bear & is_bear1 & is_bear2 & (c < c1) & (c1 < c2)

    return out.fillna(False)


# name / bias / priority (higher = more specific, checked first when
# picking a single label for the latest bar) -- also drives the site's
# pattern gallery via /api/pattern_catalog, so this list is the one
# place pattern metadata lives.
PATTERN_CATALOG = [
    # key, display name, bias, candle count
    ("bullish_abandoned_baby", "Bullish Abandoned Baby", "UP", 3),
    ("bearish_abandoned_baby", "Bearish Abandoned Baby", "DOWN", 3),
    ("morning_star", "Morning Star", "UP", 3),
    ("evening_star", "Evening Star", "DOWN", 3),
    ("morning_doji_star", "Morning Doji Star", "UP", 3),
    ("evening_doji_star", "Evening Doji Star", "DOWN", 3),
    ("three_white_soldiers", "Three White Soldiers", "UP", 3),
    ("three_black_crows", "Three Black Crows", "DOWN", 3),
    ("identical_three_crows", "Identical Three Crows", "DOWN", 3),
    ("three_stars_south", "Three Stars In The South", "UP", 3),
    ("unique_three_river", "Unique Three River Bottom", "UP", 3),
    ("stick_sandwich", "Stick Sandwich", "UP", 3),
    ("upside_gap_two_crows", "Upside Gap Two Crows", "DOWN", 3),
    ("three_outside_up", "Three Outside Up", "UP", 3),
    ("three_outside_down", "Three Outside Down", "DOWN", 3),
    ("three_inside_up", "Three Inside Up", "UP", 3),
    ("three_inside_down", "Three Inside Down", "DOWN", 3),
    ("advance_block", "Advance Block", "DOWN", 3),
    ("deliberation", "Deliberation", "DOWN", 3),
    ("tri_star", "Tri-Star", "NEUTRAL", 3),
    ("three_bar_push_up", "3-Bar Push Up", "UP", 3),
    ("three_bar_push_down", "3-Bar Push Down", "DOWN", 3),
    ("bullish_kicker", "Bullish Kicker", "UP", 2),
    ("bearish_kicker", "Bearish Kicker", "DOWN", 2),
    ("bullish_engulfing", "Bullish Engulfing", "UP", 2),
    ("bearish_engulfing", "Bearish Engulfing", "DOWN", 2),
    ("dark_cloud_cover", "Dark Cloud Cover", "DOWN", 2),
    ("piercing_line", "Piercing Line", "UP", 2),
    ("bullish_harami_cross", "Bullish Harami Cross", "UP", 2),
    ("bearish_harami_cross", "Bearish Harami Cross", "DOWN", 2),
    ("bullish_harami", "Bullish Harami", "UP", 2),
    ("bearish_harami", "Bearish Harami", "DOWN", 2),
    ("tweezer_bottom", "Tweezer Bottom", "UP", 2),
    ("tweezer_top", "Tweezer Top", "DOWN", 2),
    ("bullish_counterattack", "Bullish Counterattack", "UP", 2),
    ("bearish_counterattack", "Bearish Counterattack", "DOWN", 2),
    ("thrusting_line", "Thrusting Line", "DOWN", 2),
    ("in_neck_line", "In Neck Line", "DOWN", 2),
    ("dragonfly_doji", "Dragonfly Doji", "UP", 1),
    ("gravestone_doji", "Gravestone Doji", "DOWN", 1),
    ("bullish_marubozu", "Bullish Marubozu", "UP", 1),
    ("bearish_marubozu", "Bearish Marubozu", "DOWN", 1),
    ("hammer", "Hammer", "UP", 1),
    ("inverted_hammer", "Inverted Hammer", "UP", 1),
    ("hanging_man", "Hanging Man", "DOWN", 1),
    ("shooting_star", "Shooting Star", "DOWN", 1),
    ("long_legged_doji", "Long-Legged Doji", "NEUTRAL", 1),
    ("spinning_top", "Spinning Top", "NEUTRAL", 1),
    ("inside_bar", "Inside Bar", "NEUTRAL", 1),
    ("doji", "Doji", "NEUTRAL", 1),
]

PATTERN_NAME_BY_KEY = {k: name for k, name, bias, n in PATTERN_CATALOG}
PATTERN_BIAS_BY_NAME = {name: bias for k, name, bias, n in PATTERN_CATALOG}
# priority order: multi-candle / more specific patterns checked before
# generic single-candle ones, matching the order PATTERN_CATALOG is defined in.
PATTERN_PRIORITY = [k for k, name, bias, n in PATTERN_CATALOG]


def latest_pattern_name(pattern_row) -> str:
    for key in PATTERN_PRIORITY:
        if pattern_row.get(key):
            return PATTERN_NAME_BY_KEY[key]
    return "None"
