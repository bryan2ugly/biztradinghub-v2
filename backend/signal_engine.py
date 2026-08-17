"""
Signal engine -- Python port of the core logic from the Pine Script
strategy (EWMA-vol probability model, regime engine, S/R pivots, and the
three signal tiers: Confirmed / Early Confirmation / Early Pick).

NOTE ON SCOPE: this ports the decision-making core, not every cosmetic
feature of the original script (candlestick pattern tags, the dozens of
HUD rows, backtest mode, etc). Volume-dependent filters (absorption,
volume z-score) are omitted because the public spot-price endpoints used
by price_feed.py don't provide real trade volume -- add a volume-bearing
feed later if you want those filters back.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from config import StrategyConfig
from patterns import detect_all_patterns, latest_pattern_name, PATTERN_CATALOG


# ---------------------------------------------------------------------------
# 1-minute bar aggregation from raw composite-price ticks
# ---------------------------------------------------------------------------
class BarBuilder:
    def __init__(self, max_bars: int = 500):
        self.max_bars = max_bars
        self.bars = pd.DataFrame(columns=["open", "high", "low", "close", "ticks"])
        self.bars.index.name = "minute"
        self._cur_minute: Optional[datetime] = None
        self._cur_open = self._cur_high = self._cur_low = self._cur_close = None
        self._cur_ticks = 0

    def add_tick(self, price: float, ts: Optional[datetime] = None) -> bool:
        """Add a price tick. Returns True if this tick closed a new bar."""
        ts = ts or datetime.now(timezone.utc)
        minute = ts.replace(second=0, microsecond=0)
        closed_a_bar = False

        if self._cur_minute is None:
            self._cur_minute = minute
            self._cur_open = self._cur_high = self._cur_low = self._cur_close = price
            self._cur_ticks = 1
            return False

        if minute != self._cur_minute:
            self._flush_bar()
            closed_a_bar = True
            self._cur_minute = minute
            self._cur_open = self._cur_high = self._cur_low = self._cur_close = price
            self._cur_ticks = 1
        else:
            self._cur_high = max(self._cur_high, price)
            self._cur_low = min(self._cur_low, price)
            self._cur_close = price
            self._cur_ticks += 1

        return closed_a_bar

    def _flush_bar(self):
        self.bars.loc[self._cur_minute] = [self._cur_open, self._cur_high, self._cur_low, self._cur_close, self._cur_ticks]
        if len(self.bars) > self.max_bars:
            self.bars = self.bars.iloc[-self.max_bars:]

    def add_bar(self, ts: datetime, o: float, h: float, l: float, c: float, ticks: int = 1):
        """Append an already-closed OHLC bar directly (used by the backtester,
        which has real historical candles and doesn't need tick aggregation)."""
        minute = ts.replace(second=0, microsecond=0)
        self.bars.loc[minute] = [o, h, l, c, ticks]
        if len(self.bars) > self.max_bars:
            self.bars = self.bars.iloc[-self.max_bars:]

    def live_bar(self) -> Optional[dict]:
        """The currently-forming (unclosed) bar, for live/intrabar reads."""
        if self._cur_minute is None:
            return None
        return {
            "minute": self._cur_minute, "open": self._cur_open, "high": self._cur_high,
            "low": self._cur_low, "close": self._cur_close, "ticks": self._cur_ticks,
        }

    def closed_and_live_df(self) -> pd.DataFrame:
        """Closed bars plus the live bar appended, for indicator calc."""
        df = self.bars.copy()
        live = self.live_bar()
        if live is not None:
            df.loc[live["minute"]] = [live["open"], live["high"], live["low"], live["close"], live["ticks"]]
        return df


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def ewma_vol_of_log_returns(close: pd.Series, lam: float) -> pd.Series:
    """Vectorized RiskMetrics-style EWMA variance: var[i] = lam*var[i-1] +
    (1-lam)*sq_ret[i]. pandas' ewm(adjust=False) implements exactly this
    recursion (with alpha = 1-lam) at C speed instead of a Python loop --
    critical for backtesting months of 1-min bars in reasonable time."""
    log_ret = np.log(close / close.shift(1))
    sq_ret = log_ret ** 2
    var = sq_ret.ewm(alpha=(1 - lam), adjust=False).mean()
    return np.sqrt(var.clip(lower=0))


def norm_cdf_logistic_approx(x: float) -> float:
    """Same fast logistic approximation to the normal CDF used in the Pine script."""
    return 1 / (1 + math.exp(-1.702 * x))


def pivots(series: pd.Series, left: int, right: int, kind: str) -> pd.Series:
    """Return a boolean series marking a confirmed pivot high/low at each
    index. Vectorized with a centered rolling max/min instead of a Python
    loop -- the loop version was fine for live ticking but far too slow
    across months of backtest bars (called once per tick)."""
    window = left + right + 1
    if kind == "high":
        roll_extreme = series.rolling(window, center=True, min_periods=window).max()
    else:
        roll_extreme = series.rolling(window, center=True, min_periods=window).min()
    return series.eq(roll_extreme) & roll_extreme.notna()


def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Full 50-pattern candlestick detector (see patterns.py) for display
    and pattern_text. pattern_bullish/pattern_bearish -- the flags that
    actually GATE Confirmed/Early Confirm/reversal-warning -- deliberately
    use only the original, narrower set of well-tested patterns instead of
    all 50. Most of the extra 40 are rarer 3-candle patterns I only
    smoke-tested on synthetic data, not real market behavior; gating live
    calls on all of them made the signal noisier, not better. The other
    40 still show up in pattern_text and the site's pattern gallery for
    learning purposes -- they just don't drive a live call."""
    out = detect_all_patterns(df)
    trusted_up = ["bullish_engulfing", "hammer", "bullish_marubozu", "three_bar_push_up"]
    trusted_down = ["bearish_engulfing", "shooting_star", "bearish_marubozu", "three_bar_push_down"]
    out["pattern_bullish"] = out[trusted_up].any(axis=1)
    out["pattern_bearish"] = out[trusted_down].any(axis=1)
    return out


REGIME_NAMES = ["Strong Trend", "Trend", "Range", "High Volatility", "Extreme Volatility", "Compression"]

CALIBRATION_BUCKETS = [
    ("50-60%", 0.50, 0.60),
    ("60-70%", 0.60, 0.70),
    ("70-85%", 0.70, 0.85),
    ("85%+", 0.85, 1.01),
]


def bucket_for_prob(prob: float) -> Optional[str]:
    for name, lo, hi in CALIBRATION_BUCKETS:
        if lo <= prob < hi:
            return name
    return None


@dataclass
class TierRecord:
    """Win/loss bookkeeping for one signal tier (Confirmed / Early Confirm /
    Early Pick), mirroring the Pine script's Track Record + Kalshi Sim rows."""
    wins: int = 0
    losses: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.losses

    @property
    def accuracy_pct(self) -> Optional[float]:
        return (self.wins / self.total * 100) if self.total > 0 else None

    def kalshi_pnl_cents(self, assumed_entry_cents: float, fee_cents: float) -> float:
        return self.wins * (100.0 - assumed_entry_cents - fee_cents) - self.losses * (assumed_entry_cents + fee_cents)


def _new_calibration_buckets() -> dict:
    return {name: TierRecord() for name, _, _ in CALIBRATION_BUCKETS}


@dataclass
class TrackRecord:
    confirmed: TierRecord = field(default_factory=TierRecord)
    early_confirm: TierRecord = field(default_factory=TierRecord)
    early_pick: TierRecord = field(default_factory=TierRecord)
    value_zone: TierRecord = field(default_factory=TierRecord)
    # Broken down by WHICH tier actually fired first in the window (Early
    # Pick / Early Confirm / Confirmed) -- a different question from the
    # per-tier stats above, which score each tier as if it always led.
    # This answers "when Confirmed happened to be the very first thing to
    # fire (nothing beat it to it), how often was that first call right?"
    first_signal_early_pick: TierRecord = field(default_factory=TierRecord)
    first_signal_early_confirm: TierRecord = field(default_factory=TierRecord)
    first_signal_confirmed: TierRecord = field(default_factory=TierRecord)
    # Calibration: for Confirmed and Early Confirm, bins each fire by the
    # model's OWN stated probability at the moment it fired, so you can
    # check "when the model said 65%, did it actually win about 65% of
    # the time?" -- the thing that's supposed to make a probability
    # meaningful, and that nothing in this bot checked until now.
    calibration_confirmed: dict = field(default_factory=_new_calibration_buckets)
    calibration_early_confirm: dict = field(default_factory=_new_calibration_buckets)
    windows_scored: int = 0

    def record(self, tier: TierRecord, correct: bool):
        if correct:
            tier.wins += 1
        else:
            tier.losses += 1

    def first_signal_tier_record(self, tier_name: str) -> Optional[TierRecord]:
        return {
            "Early Pick": self.first_signal_early_pick,
            "Early Confirm": self.first_signal_early_confirm,
            "Confirmed": self.first_signal_confirmed,
        }.get(tier_name)


@dataclass
class WindowResult:
    """One completed window's outcome, for the live activity log. Distinct
    from TrackRecord (which only keeps running win/loss counts) -- this
    keeps the actual per-window details so they can be written to a log
    file and reviewed later, e.g. compared against what the original
    TradingView script called on the same windows."""
    close_time: datetime
    strike: float
    settle_price: float
    actual_dir: str  # "UP" or "DOWN"
    confirmed_dir: str
    confirmed_correct: Optional[bool]
    early_confirm_dir: str
    early_confirm_correct: Optional[bool]
    early_pick_dir: str
    early_pick_correct: Optional[bool]
    value_zone_dir: str
    value_zone_correct: Optional[bool]
    first_signal_tier: str
    first_signal_dir: str
    first_signal_correct: Optional[bool]


@dataclass
class WindowState:
    """Per-15-minute-window bookkeeping (mirrors the Pine script's window vars)."""
    strike: Optional[float] = None
    open_time: Optional[datetime] = None
    first_signal_tier: str = "NONE"
    first_signal_dir: str = "NONE"
    first_signal_prob: Optional[float] = None
    early_pick_dir: str = "NONE"
    early_confirm_dir: str = "HOLD"
    confirmed_dir: str = "HOLD"
    value_zone_dir: str = "NONE"
    confirmed_prob_at_fire: Optional[float] = None      # model probability the moment Confirmed first fired
    early_confirm_prob_at_fire: Optional[float] = None  # same, for Early Confirm
    already_scored: bool = False  # guards against double-scoring the same
    # window when a caller (e.g. historical_backtest.py) scores it
    # externally with a known-true outcome AND the engine's own automatic
    # clock-based rollover would otherwise also try to score it moments
    # later -- whichever call happens first wins, the second is a no-op.


@dataclass
class SignalOutput:
    timestamp: datetime
    price: float
    strike: Optional[float]
    minutes_into_window: float
    minutes_remaining: float

    prob_above: Optional[float]
    prob_below: Optional[float]

    trend: str  # "Uptrend" / "Downtrend" / "Range/Choppy"
    regime: str
    regime_reliability: float

    support: Optional[float]
    resistance: Optional[float]

    confirmed_dir: str  # "UP" / "DOWN" / "HOLD"
    early_confirm_dir: str  # "UP" / "DOWN" / "HOLD"
    early_pick_dir: str  # "UP" / "DOWN" / "NONE"
    value_zone_dir: str  # "UP" / "DOWN" / "NONE" -- fires earliest, while still cheap

    too_close_to_call: bool
    tradeable_dir: Optional[str]  # best actionable direction for execution, if any

    # --- trimmed HUD fields ---
    phase: str
    hud_signal: str
    settle_up_pct: Optional[float]
    settle_down_pct: Optional[float]
    early_pick_text: str
    need_move_text: str
    value_zone_text: str
    pattern_text: str
    reversal_warning_dir: Optional[str]  # "UP"/"DOWN" = exit that held direction, or None
    reversal_chance_pct: Optional[float]  # 0-100: how many of the 4 reversal factors are currently stacked against the held direction, as a %. None if nothing is held yet.
    held_dir: Optional[str]  # the direction the reversal check is measured against, or None
    reversal_text: str
    calibration_adj_confirmed: float      # current bounded self-calibration nudge, Confirmed tier
    calibration_adj_early_confirm: float  # same, Early Confirm tier
    calibrated_pct: Optional[float]  # the REAL historical win rate for the active tier's confidence bucket, if known
    calibrated_tier: Optional[str]   # which tier's calibration was used ("confirmed"/"early_confirm"), or None
    # Raw features, exposed for decision_log.py -- all of these were already
    # computed internally to drive the tiers above, just never surfaced
    # outside the engine before now.
    volatility: float       # sigma_window: the vol estimate actually used in the probability calc
    rsi_value: float
    ema_fast_val: float
    ema_slow_val: float
    roc_pct: float          # 3-bar rate of change, %
    accel_pct: float        # acceleration (change in roc), %
    body_strength: float    # current candle's body as a fraction of its full range, 0-1


class SignalEngine:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self.bar_builder = BarBuilder(max_bars=cfg.lookback_bars)
        self.window = WindowState()
        self.track_record = TrackRecord()
        self.completed_windows: list = []  # list[WindowResult], most recent last
        self._early_up_persist = 0    # consecutive bars the Early Pick UP trigger has held
        self._early_down_persist = 0  # same, DOWN

    # -- window bookkeeping -----------------------------------------------
    def _minutes_into_window(self, ts: datetime, window_minutes: int = 15) -> float:
        return float(ts.minute % window_minutes) + ts.second / 60.0

    def _maybe_roll_window(self, ts: datetime, price: float, window_minutes: int = 15):
        minute_mark = ts.minute % window_minutes == 0 and ts.second < 5
        if minute_mark and (self.window.open_time is None or (ts - self.window.open_time).total_seconds() > 60):
            self._score_window_close(price)
            self.window = WindowState(strike=price, open_time=ts)

    def calibrate_probability(self, tier_name: str, raw_prob: float) -> float:
        """Returns the REAL historical win rate for the confidence bucket
        raw_prob falls into (for the given tier), instead of the model's
        raw stated probability -- e.g. if the model has historically said
        '70%' for this tier but those calls only actually won 61% of the
        time, this returns 0.61. Falls back to raw_prob unchanged if
        there's not yet enough real samples in that bucket to trust it.
        tier_name: "confirmed" or "early_confirm" -- the only two tiers
        with calibration bucket data (see TrackRecord)."""
        bucket = bucket_for_prob(raw_prob)
        if bucket is None:
            return raw_prob
        table = self.track_record.calibration_confirmed if tier_name == "confirmed" else self.track_record.calibration_early_confirm
        rec = table.get(bucket)
        if rec is None or rec.total < self.cfg.calibration_min_samples:
            return raw_prob
        return rec.wins / rec.total

    def _calibration_adjustment(self, tier_record: TierRecord) -> float:
        """Bounded self-calibration adjustment for one tier's threshold,
        recomputed fresh from the tier's real win rate every call -- see
        the comment where this is used in _process() for the full
        rationale. Returns 0.0 (no adjustment) until there's enough real
        data, or if calibration learning is turned off."""
        if not self.cfg.use_calibration_learning or tier_record.total < self.cfg.calibration_min_samples:
            return 0.0
        actual_win_rate = tier_record.wins / tier_record.total
        raw = (self.cfg.calibration_target_win_rate - actual_win_rate) * self.cfg.calibration_adj_scale
        return max(self.cfg.calibration_adj_min, min(self.cfg.calibration_adj_max, raw))

    def set_external_strike(self, strike: float):
        """Overrides the auto-estimated strike (price at window open) with
        Kalshi's real, exact strike for the current window. Call this
        whenever fresh market data is available -- e.g. every loop
        iteration in main.py -- so the probability model is always
        anchored to the actual number the contract will settle against,
        not an approximation."""
        if strike is not None:
            self.window.strike = strike

    def score_current_window_with_known_outcome(self, actual_up: bool, settle_price: Optional[float] = None):
        """Public wrapper so a caller driving window boundaries externally
        (e.g. historical_backtest.py, which rolls windows based on real
        Kalshi market close times, not just clock ticks) can score the
        CURRENT window against a known-true outcome -- Kalshi's own
        settled 'result' field -- without needing to derive it. settle_price
        is optional and only used for the logged WindowResult's record of
        what price was at close time; pass the real BTC price at that
        moment if you have it, otherwise the window's strike is used as a
        reasonable stand-in so the log doesn't show a misleading 0."""
        logged_price = settle_price if settle_price is not None else (self.window.strike or 0.0)
        self._score_window_close(settle_price=logged_price, actual_up_override=actual_up)

    def _score_window_close(self, settle_price: float, actual_up_override: Optional[bool] = None):
        """Called right as a window ends: grade each tier's last-held
        direction (if any fired) against where price actually settled.
        Mirrors the Pine script's Track Record + Kalshi Sim scoring, and
        also appends a WindowResult for the live activity log.

        actual_up_override lets a caller supply the REAL, authoritative
        outcome (e.g. Kalshi's own 'result' field from a settled historical
        market) instead of deriving it from settle_price vs strike -- used
        by historical_backtest.py, where the true outcome is known exactly
        rather than approximated from Coinbase's close price."""
        if self.window.strike is None or self.window.already_scored:
            return
        self.window.already_scored = True
        actual_up = actual_up_override if actual_up_override is not None else settle_price > self.window.strike
        actual_dir = "UP" if actual_up else "DOWN"

        confirmed_correct = None
        if self.window.confirmed_dir != "HOLD":
            confirmed_correct = (self.window.confirmed_dir == "UP") == actual_up
            self.track_record.record(self.track_record.confirmed, confirmed_correct)
            if self.window.confirmed_prob_at_fire is not None:
                bucket = bucket_for_prob(self.window.confirmed_prob_at_fire)
                if bucket:
                    self.track_record.record(self.track_record.calibration_confirmed[bucket], confirmed_correct)

        early_confirm_correct = None
        if self.window.early_confirm_dir != "HOLD":
            early_confirm_correct = (self.window.early_confirm_dir == "UP") == actual_up
            self.track_record.record(self.track_record.early_confirm, early_confirm_correct)
            if self.window.early_confirm_prob_at_fire is not None:
                bucket = bucket_for_prob(self.window.early_confirm_prob_at_fire)
                if bucket:
                    self.track_record.record(self.track_record.calibration_early_confirm[bucket], early_confirm_correct)

        early_pick_correct = None
        if self.window.early_pick_dir != "NONE":
            early_pick_correct = (self.window.early_pick_dir == "UP") == actual_up
            self.track_record.record(self.track_record.early_pick, early_pick_correct)

        value_zone_correct = None
        if self.window.value_zone_dir != "NONE":
            value_zone_correct = (self.window.value_zone_dir == "UP") == actual_up
            self.track_record.record(self.track_record.value_zone, value_zone_correct)

        # First signal: whichever tier fired FIRST this window (Confirmed >
        # Early Confirm > Early Pick priority only matters for ties on the
        # same bar -- see the "lock first signal" block above where these
        # get set). Scored separately, broken down by which tier happened
        # to be first, since that's a different question than "how
        # accurate is Confirmed whenever it eventually fires."
        first_signal_correct = None
        if self.window.first_signal_tier != "NONE" and self.window.first_signal_dir != "NONE":
            first_signal_correct = (self.window.first_signal_dir == "UP") == actual_up
            tier_record = self.track_record.first_signal_tier_record(self.window.first_signal_tier)
            if tier_record is not None:
                self.track_record.record(tier_record, first_signal_correct)

        self.track_record.windows_scored += 1
        self.completed_windows.append(WindowResult(
            close_time=(self.window.open_time + timedelta(minutes=15)) if self.window.open_time else datetime.now(timezone.utc),
            strike=self.window.strike, settle_price=settle_price, actual_dir=actual_dir,
            confirmed_dir=self.window.confirmed_dir, confirmed_correct=confirmed_correct,
            early_confirm_dir=self.window.early_confirm_dir, early_confirm_correct=early_confirm_correct,
            early_pick_dir=self.window.early_pick_dir, early_pick_correct=early_pick_correct,
            value_zone_dir=self.window.value_zone_dir, value_zone_correct=value_zone_correct,
            first_signal_tier=self.window.first_signal_tier, first_signal_dir=self.window.first_signal_dir,
            first_signal_correct=first_signal_correct,
        ))
        if len(self.completed_windows) > 2000:
            self.completed_windows = self.completed_windows[-2000:]

    # -- main update (live, tick-by-tick) -------------------------------------
    def update(self, price: float, ts: Optional[datetime] = None) -> Optional[SignalOutput]:
        """Live path: feed one price tick at a time. Used by main.py."""
        ts = ts or datetime.now(timezone.utc)
        self._maybe_roll_window(ts, price)
        if self.window.strike is None:
            self.window.strike = price
            self.window.open_time = ts

        self.bar_builder.add_tick(price, ts)
        df = self.bar_builder.closed_and_live_df()
        if len(df) < max(self.cfg.ema_slow_len, self.cfg.rsi_len) + 2:
            return None  # not enough history yet
        return self._process(price, ts, df)

    # -- fast path for backtesting: ingest an already-closed OHLC bar directly,
    # skipping tick-by-tick aggregation. Roughly halves backtest runtime versus
    # feeding synthetic open/close ticks through update().
    def update_bar(self, o: float, h: float, l: float, c: float, ts: datetime) -> Optional[SignalOutput]:
        self._maybe_roll_window(ts, o)
        if self.window.strike is None:
            self.window.strike = o
            self.window.open_time = ts

        self.bar_builder.add_bar(ts, o, h, l, c)
        df = self.bar_builder.bars
        if len(df) < max(self.cfg.ema_slow_len, self.cfg.rsi_len) + 2:
            return None
        return self._process(c, ts, df)

    def _process(self, price: float, ts: datetime, df: pd.DataFrame) -> SignalOutput:
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_ = df["open"].astype(float)

        ema_fast = ema(close, self.cfg.ema_fast_len)
        ema_slow = ema(close, self.cfg.ema_slow_len)
        rsi_val = rsi(close, self.cfg.rsi_len)
        sigma_per_bar = ewma_vol_of_log_returns(close, self.cfg.ewma_lambda).iloc[-1]
        sigma_per_bar = max(float(sigma_per_bar) if not pd.isna(sigma_per_bar) else 1e-6, 1e-6)

        patterns = detect_candlestick_patterns(df)
        pattern_bullish = bool(patterns["pattern_bullish"].iloc[-1])
        pattern_bearish = bool(patterns["pattern_bearish"].iloc[-1])
        pattern_text = latest_pattern_name(patterns.iloc[-1])

        minutes_into = self._minutes_into_window(ts)
        minutes_remaining = max(15.0 - minutes_into, 0.01)
        bars_remaining = minutes_remaining  # 1-min bars
        sigma_window = sigma_per_bar * math.sqrt(bars_remaining)

        strike = self.window.strike
        d = math.log(price / strike) / sigma_window if strike and sigma_window > 0 else 0.0
        prob_above = norm_cdf_logistic_approx(d)
        prob_below = 1 - prob_above

        # settlement lock-in: decisive + inside final N minutes -> floor toward certainty
        basis_buffer = self.cfg.basis_buffer_usd
        decisively_above = strike is not None and (price - strike) > basis_buffer
        decisively_below = strike is not None and (strike - price) > basis_buffer
        in_lock_window = minutes_remaining <= self.cfg.lock_window_mins
        if in_lock_window and decisively_above:
            prob_above = max(prob_above, self.cfg.lock_floor_pct / 100)
            prob_below = 1 - prob_above
        elif in_lock_window and decisively_below:
            prob_below = max(prob_below, self.cfg.lock_floor_pct / 100)
            prob_above = 1 - prob_below

        too_close = strike is not None and abs(price - strike) <= basis_buffer

        # --- trend structure from confirmed pivots ---
        ph = pivots(high, self.cfg.pivot_left, self.cfg.pivot_right, "high")
        pl = pivots(low, self.cfg.pivot_left, self.cfg.pivot_right, "low")
        ph_vals = high[ph]
        pl_vals = low[pl]
        trend_text = "Range/Choppy"
        trend_up = trend_down = False
        support = float(pl_vals.iloc[-1]) if len(pl_vals) else None
        resistance = float(ph_vals.iloc[-1]) if len(ph_vals) else None
        if len(ph_vals) >= 2 and len(pl_vals) >= 2:
            if ph_vals.iloc[-1] > ph_vals.iloc[-2] and pl_vals.iloc[-1] > pl_vals.iloc[-2]:
                trend_text, trend_up = "Uptrend", True
            elif ph_vals.iloc[-1] < ph_vals.iloc[-2] and pl_vals.iloc[-1] < pl_vals.iloc[-2]:
                trend_text, trend_down = "Downtrend", True

        # --- simplified regime classification ---
        log_ret = np.log(close / close.shift(1))
        rv = log_ret.rolling(20).std()
        rv_base = rv.rolling(200, min_periods=20).mean()
        vol_mix = float((rv.iloc[-1] / rv_base.iloc[-1]) if rv_base.iloc[-1] and not pd.isna(rv_base.iloc[-1]) and rv_base.iloc[-1] > 0 else 1.0)
        ema_spread = abs(ema_fast.iloc[-1] - ema_slow.iloc[-1])
        atr = (high - low).rolling(14).mean().iloc[-1]
        ema_spread_n = ema_spread / atr if atr and atr > 0 else 0.0
        trend_core = min(1.0, ema_spread_n / 1.5) * 0.5 + (1.0 if (trend_up or trend_down) else 0.0) * 0.5

        if vol_mix >= 2.4:
            regime_idx = 4
        elif vol_mix >= 1.4:
            regime_idx = 3
        elif trend_core >= 0.55 and vol_mix < 2.0:
            regime_idx = 0
        elif trend_core >= 0.30:
            regime_idx = 1
        elif trend_core < 0.20 and vol_mix < 0.9:
            regime_idx = 5
        else:
            regime_idx = 2
        regime_name = REGIME_NAMES[regime_idx]
        regime_reliability = min(100.0, abs(trend_core - 0.4) * 150)

        # regime-adaptive threshold nudge (bounded, no learning loop for now)
        base_thr_adj = {0: -0.03, 1: 0.0, 2: 0.04, 3: 0.05, 4: 0.08, 5: -0.01}[regime_idx]
        thr_adj = base_thr_adj if self.cfg.use_adaptive_thresholds else 0.0

        # Bounded self-calibration: recomputed FRESH each tick from the
        # tier's actual real win rate (never accumulates/drifts) -- if the
        # tier is genuinely winning more than the target rate, the model
        # is underconfident and the threshold eases (negative adjustment,
        # same convention as the regime adjustment above); if it's winning
        # less, the threshold tightens. Only kicks in once there are
        # enough real samples to trust, and always clamped to a small
        # bounded range -- this can nudge the bot, never take it over.
        calib_adj_confirmed = self._calibration_adjustment(self.track_record.confirmed)
        calib_adj_early_confirm = self._calibration_adjustment(self.track_record.early_confirm)

        eff_confirm_thr = min(0.97, self.cfg.prob_buy_thresh + thr_adj + calib_adj_confirmed)
        eff_early_confirm_thr = max(0.51, min(0.95, self.cfg.early_confirm_threshold + thr_adj + calib_adj_early_confirm))

        momentum_bullish = ema_fast.iloc[-1] > ema_slow.iloc[-1] and rsi_val.iloc[-1] > 50
        momentum_bearish = ema_fast.iloc[-1] < ema_slow.iloc[-1] and rsi_val.iloc[-1] < 50
        fast_mom_up = ema_fast.iloc[-1] > ema_slow.iloc[-1] or rsi_val.iloc[-1] > 50
        fast_mom_down = ema_fast.iloc[-1] < ema_slow.iloc[-1] or rsi_val.iloc[-1] < 50

        # Body strength: how much of the bar's full range is actual
        # directional body vs. wicks. A candle that's mostly wick is
        # indecision, not conviction, even if it technically closed up or
        # down -- this config knob existed but was never actually wired
        # into either tier below, which is a real gap given how noisy a
        # single low-conviction bar can be.
        candle_range = (high - low).iloc[-1]
        candle_body = abs(close.iloc[-1] - open_.iloc[-1])
        body_strength = (candle_body / candle_range) if candle_range > 0 else 0.0
        body_ok = (not self.cfg.require_body_strength) or (body_strength >= self.cfg.min_body_strength)

        # --- CONFIRMED tier ---
        # Pattern gate: don't confirm a direction the most recent candle's
        # own shape is actively arguing against (e.g. a fresh bearish
        # engulfing bar shouldn't let a UP confirmation through even if
        # the probability/momentum math cleared its bar) -- same "block on
        # disagreement" filter already used for trend/trap, now extended
        # to candlestick patterns instead of leaving them purely cosmetic.
        confirmed_dir = "HOLD"
        if strike is not None and not too_close and body_ok:
            if (prob_above >= eff_confirm_thr and momentum_bullish and not pattern_bearish and
                    (not self.cfg.require_trend_for_confirm or not trend_down)):
                confirmed_dir = "UP"
            elif (prob_below >= eff_confirm_thr and momentum_bearish and not pattern_bullish and
                    (not self.cfg.require_trend_for_confirm or not trend_up)):
                confirmed_dir = "DOWN"

        # --- EARLY CONFIRMATION tier ---
        ec_too_close = strike is not None and abs(price - strike) <= self.cfg.ec_basis_buffer_usd
        early_confirm_dir = "HOLD"
        if strike is not None and not ec_too_close:
            if (prob_above >= eff_early_confirm_thr and prob_above <= self.cfg.max_first_signal_prob
                    and fast_mom_up and not pattern_bearish):
                early_confirm_dir = "UP"
            elif (prob_below >= eff_early_confirm_thr and prob_below <= self.cfg.max_first_signal_prob
                    and fast_mom_down and not pattern_bullish):
                early_confirm_dir = "DOWN"

        # --- EARLY PICK tier (acceleration-based) ---
        # Three real quality filters added that were previously configured
        # but never actually applied: body strength (reject weak/indecisive
        # candles), a pattern gate (same "don't fire against an opposing
        # candle shape" rule Confirmed/Early Confirm already use), and
        # actual multi-bar persistence -- this used to fire the instant a
        # single tick crossed the acceleration threshold, which is exactly
        # the kind of one-bar noise a quality filter should be catching.
        roc = close.pct_change(3).iloc[-1] * 100 if len(close) > 3 else 0.0
        roc_prev = close.pct_change(3).iloc[-2] * 100 if len(close) > 4 else 0.0
        accel = (roc - roc_prev) if not (pd.isna(roc) or pd.isna(roc_prev)) else 0.0
        early_minute_ok = minutes_into >= self.cfg.early_pick_min_minute

        early_up_raw = (not too_close and early_minute_ok and body_ok and not pattern_bearish and
                        roc > 0 and accel > self.cfg.early_min_accel_pct and momentum_bullish and not trend_down)
        early_down_raw = (not too_close and early_minute_ok and body_ok and not pattern_bullish and
                          roc < 0 and accel < -self.cfg.early_min_accel_pct and momentum_bearish and not trend_up)

        self._early_up_persist = self._early_up_persist + 1 if early_up_raw else 0
        self._early_down_persist = self._early_down_persist + 1 if early_down_raw else 0

        early_pick_dir = "NONE"
        if self._early_up_persist >= self.cfg.early_persist_bars:
            early_pick_dir = "UP"
        elif self._early_down_persist >= self.cfg.early_persist_bars:
            early_pick_dir = "DOWN"

        # Lock each tier to whichever direction FIRST fired this window,
        # and never let it flip to the opposite side later -- previously
        # this only prevented reverting to HOLD, but a later opposite-
        # direction fire could still silently overwrite it, which is
        # exactly what was causing the displayed confirmation to keep
        # changing candle to candle. Once a side is locked in, a later
        # opposing signal is handled by the reversal engine below (an
        # explicit EMERGENCY EXIT warning) instead of quietly relabeling
        # the confirmation.
        if confirmed_dir != "HOLD" and self.window.confirmed_dir == "HOLD":
            self.window.confirmed_dir = confirmed_dir
            self.window.confirmed_prob_at_fire = prob_above if confirmed_dir == "UP" else prob_below
        if early_confirm_dir != "HOLD" and self.window.early_confirm_dir == "HOLD":
            self.window.early_confirm_dir = early_confirm_dir
            self.window.early_confirm_prob_at_fire = prob_above if early_confirm_dir == "UP" else prob_below
        if early_pick_dir != "NONE" and self.window.early_pick_dir == "NONE":
            self.window.early_pick_dir = early_pick_dir

        # Everything downstream (display, decision engine, order execution)
        # uses the LOCKED value, not this tick's raw recomputation -- the
        # raw values above are only used for detecting "did a tier just
        # fire for the first time" and for the reversal-factor math below.
        confirmed_dir = self.window.confirmed_dir
        early_confirm_dir = self.window.early_confirm_dir
        early_pick_dir = self.window.early_pick_dir

        # --- REVERSAL / EXIT WARNING ---
        # Ports the Pine script's "emergency reversal" idea: if the bot's
        # own signal this window was leaning one way and several independent
        # things flip against it at once, warn to exit that side. "Held
        # direction" = whatever this window's first tier signal pointed to
        # (Confirmed > Early Confirm > Early Pick), since the bot has no
        # actual position/portfolio state to check against -- same
        # approach the original script used (wasLeaningUp/wasLeaningDown
        # from its own prior signals, not a real position).
        held_dir = None
        if self.window.confirmed_dir != "HOLD":
            held_dir = self.window.confirmed_dir
        elif self.window.early_confirm_dir != "HOLD":
            held_dir = self.window.early_confirm_dir
        elif self.window.early_pick_dir != "NONE":
            held_dir = self.window.early_pick_dir

        ema_cross_up = len(ema_fast) > 1 and ema_fast.iloc[-2] <= ema_slow.iloc[-2] and ema_fast.iloc[-1] > ema_slow.iloc[-1]
        ema_cross_down = len(ema_fast) > 1 and ema_fast.iloc[-2] >= ema_slow.iloc[-2] and ema_fast.iloc[-1] < ema_slow.iloc[-1]
        rsi_reversal_up = len(rsi_val) > 1 and rsi_val.iloc[-2] < 35 and rsi_val.iloc[-1] >= 35
        rsi_reversal_down = len(rsi_val) > 1 and rsi_val.iloc[-2] > 65 and rsi_val.iloc[-1] <= 65
        accel_spike_up = accel > self.cfg.early_min_accel_pct * 3
        accel_spike_down = accel < -self.cfg.early_min_accel_pct * 3

        reversal_up_score = sum([ema_cross_up, rsi_reversal_up, accel_spike_up, pattern_bullish])
        reversal_down_score = sum([ema_cross_down, rsi_reversal_down, accel_spike_down, pattern_bearish])

        reversal_warning_dir = None
        reversal_text = "None"
        if held_dir == "UP" and reversal_down_score >= self.cfg.reversal_min_factors:
            reversal_warning_dir = "UP"
            reversal_text = f"🚨 EMERGENCY EXIT UP — reversal signs stacking ({reversal_down_score}/4)"
        elif held_dir == "DOWN" and reversal_up_score >= self.cfg.reversal_min_factors:
            reversal_warning_dir = "DOWN"
            reversal_text = f"🚨 EMERGENCY EXIT DOWN — reversal signs stacking ({reversal_up_score}/4)"

        # Continuous version of the same 4-factor check, expressed as a %,
        # so the HUD can show "reversal chance" even before it crosses the
        # warning threshold -- not the model's calibrated probability, just
        # how many of the 4 independent reversal factors currently agree.
        # This is separate from reversal_warning_dir/reversal_text above
        # (which stay driven by the strict this-bar-only crossing events --
        # don't touch that, it's the proven trigger logic). This continuous
        # version instead measures HOW CLOSE each factor is to firing, so
        # it moves smoothly bar to bar instead of only ever landing on
        # 0/25/50/75/100.
        reversal_chance_pct = None
        if held_dir in ("UP", "DOWN"):
            watching_down = held_dir == "UP"  # watching for signs AGAINST the held direction

            ema_gap = ema_fast.iloc[-1] - ema_slow.iloc[-1]
            ema_gap_series = (ema_fast - ema_slow).tail(20)
            ema_scale = ema_gap_series.std()
            if not ema_scale or pd.isna(ema_scale) or ema_scale <= 0:
                ema_scale = abs(price) * 0.0005 if price else 1.0
            ema_z = ema_gap / (ema_scale * 2)
            ema_factor = (0.5 - ema_z) if watching_down else (0.5 + ema_z)
            ema_factor = max(0.0, min(1.0, ema_factor))

            rsi_now = rsi_val.iloc[-1]
            if watching_down:
                rsi_factor = (55 - rsi_now) / (55 - 30)
            else:
                rsi_factor = (rsi_now - 45) / (65 - 45)
            rsi_factor = max(0.0, min(1.0, rsi_factor))

            accel_threshold = self.cfg.early_min_accel_pct * 3
            if accel_threshold > 0:
                accel_factor = (-accel / accel_threshold) if watching_down else (accel / accel_threshold)
            else:
                accel_factor = 0.0
            accel_factor = max(0.0, min(1.0, accel_factor))

            pattern_factor = 1.0 if (pattern_bearish if watching_down else pattern_bullish) else 0.0

            reversal_chance_pct = round((ema_factor + rsi_factor + accel_factor + pattern_factor) / 4 * 100, 1)

        # lock first signal of the window
        if self.window.first_signal_tier == "NONE":
            if confirmed_dir != "HOLD":
                self.window.first_signal_tier, self.window.first_signal_dir = "Confirmed", confirmed_dir
                self.window.first_signal_prob = prob_above if confirmed_dir == "UP" else prob_below
            elif early_confirm_dir != "HOLD":
                self.window.first_signal_tier, self.window.first_signal_dir = "Early Confirm", early_confirm_dir
                self.window.first_signal_prob = prob_above if early_confirm_dir == "UP" else prob_below
            elif early_pick_dir != "NONE":
                self.window.first_signal_tier, self.window.first_signal_dir = "Early Pick", early_pick_dir
                self.window.first_signal_prob = prob_above if early_pick_dir == "UP" else prob_below

        # tradeable direction for order execution: prefer Confirmed, then Early Confirm
        tradeable_dir = None
        if confirmed_dir != "HOLD":
            tradeable_dir = confirmed_dir
        elif early_confirm_dir != "HOLD":
            tradeable_dir = early_confirm_dir

        # --- trimmed HUD fields ---
        if minutes_into <= 2:
            phase = "OPEN"
        elif minutes_into <= 6:
            phase = "SETUP"
        elif minutes_into <= 10:
            phase = "COMMIT"
        elif minutes_into <= 13:
            phase = "SURVIVAL"
        else:
            phase = "LOCK-IN"

        if trend_up:
            hud_signal = "UPTREND INTACT"
        elif trend_down:
            hud_signal = "DOWNTREND INTACT"
        else:
            hud_signal = "RANGING"

        settle_up_pct = round(prob_above * 100, 1)
        settle_down_pct = round(prob_below * 100, 1)

        # Value Zone: the earliest, loosest heads-up -- fires the instant
        # a side's probability is sitting inside the target band (default
        # 40-55%) with just basic momentum agreement. No acceleration
        # threshold, no persistence-bar requirement, no body-strength
        # check -- those exist on Early Pick specifically to cut down
        # noise, but they're also what delays it until probability (and
        # price) has often already moved past this band. This is live,
        # not sticky -- it only shows a direction while price is
        # currently inside the band, not "was in the band earlier."
        value_zone_dir = "NONE"
        if strike is not None and not too_close:
            if self.cfg.value_zone_min <= prob_above <= self.cfg.value_zone_max and momentum_bullish:
                value_zone_dir = "UP"
            elif self.cfg.value_zone_min <= prob_below <= self.cfg.value_zone_max and momentum_bearish:
                value_zone_dir = "DOWN"

        if value_zone_dir != "NONE":
            side_pct = settle_up_pct if value_zone_dir == "UP" else settle_down_pct
            value_zone_text = f"{value_zone_dir} {side_pct}%"
        else:
            value_zone_text = "NONE"
        if value_zone_dir != "NONE":
            self.window.value_zone_dir = value_zone_dir

        if early_pick_dir != "NONE":
            side_pct = settle_up_pct if early_pick_dir == "UP" else settle_down_pct
            early_pick_text = f"{early_pick_dir} {side_pct}%"
        else:
            early_pick_text = "NONE YET"

        expected_move = price * sigma_window
        need_up = max(0.0, strike + basis_buffer - price) if strike else 0.0
        need_dn = max(0.0, price - (strike - basis_buffer)) if strike else 0.0

        def ease(x):
            if expected_move <= 0:
                return "?"
            if x <= expected_move * 0.5:
                return "EASY"
            if x <= expected_move * 1.25:
                return "DOABLE"
            if x <= expected_move * 2.0:
                return "HARD"
            return "V.HARD"

        need_move_text = f"UP {ease(need_up)} / DN {ease(need_dn)}"

        # Calibrated probability: the REAL historical win rate for the
        # active tier's confidence bucket, shown alongside the model's raw
        # stated probability -- "Raw: 70% -> Calibrated: 63%" is a lot more
        # useful than the raw number alone once there's real data behind it.
        # Only Confirmed/Early Confirm have calibration bucket data.
        calibrated_pct = None
        calibrated_tier = None
        if confirmed_dir != "HOLD":
            raw = prob_above if confirmed_dir == "UP" else prob_below
            calibrated_pct = round(self.calibrate_probability("confirmed", raw) * 100, 1)
            calibrated_tier = "confirmed"
        elif early_confirm_dir != "HOLD":
            raw = prob_above if early_confirm_dir == "UP" else prob_below
            calibrated_pct = round(self.calibrate_probability("early_confirm", raw) * 100, 1)
            calibrated_tier = "early_confirm"

        return SignalOutput(
            timestamp=ts, price=price, strike=strike,
            minutes_into_window=minutes_into, minutes_remaining=minutes_remaining,
            prob_above=prob_above, prob_below=prob_below,
            trend=trend_text, regime=regime_name, regime_reliability=regime_reliability,
            support=support, resistance=resistance,
            confirmed_dir=confirmed_dir, early_confirm_dir=early_confirm_dir, early_pick_dir=early_pick_dir,
            value_zone_dir=value_zone_dir,
            too_close_to_call=too_close, tradeable_dir=tradeable_dir,
            phase=phase, hud_signal=hud_signal,
            settle_up_pct=settle_up_pct, settle_down_pct=settle_down_pct,
            early_pick_text=early_pick_text, need_move_text=need_move_text,
            value_zone_text=value_zone_text, pattern_text=pattern_text,
            reversal_warning_dir=reversal_warning_dir, reversal_text=reversal_text,
            reversal_chance_pct=reversal_chance_pct, held_dir=held_dir,
            calibration_adj_confirmed=calib_adj_confirmed, calibration_adj_early_confirm=calib_adj_early_confirm,
            calibrated_pct=calibrated_pct, calibrated_tier=calibrated_tier,
            volatility=float(sigma_window), rsi_value=float(rsi_val.iloc[-1]),
            ema_fast_val=float(ema_fast.iloc[-1]), ema_slow_val=float(ema_slow.iloc[-1]),
            roc_pct=float(roc), accel_pct=float(accel), body_strength=float(body_strength),
        )
