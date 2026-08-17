"""
Public signal-dashboard backend. Runs the same signal engine your bot
uses, fed only by public exchange price data -- no Kalshi credentials,
no order placement, nothing that touches your account. Anyone with the
URL sees the live model output and can learn how it works.

Run:
    pip install -r requirements.txt
    uvicorn server:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000
"""
from __future__ import annotations

import hmac
import os
import secrets
import threading
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import pandas as pd

import auth
from candles import fetch_1m_candles, fetch_1m_candles_with_volume
from chart_patterns import detect_flag, detect_double_top_bottom
from config import FEED_CONFIG, STRATEGY_CONFIG
from kalshi_public import get_current_kalshi_strike
from overlays import compute_overlays
from patterns import PATTERN_CATALOG, PATTERN_BIAS_BY_NAME, detect_all_patterns, latest_pattern_name
from price_feed import CompositePriceFeed
import round_history
import github_store
from signal_engine import SignalEngine

# Set this in your host's environment variables (e.g. Railway -> Settings ->
# Variables), NOT hardcoded here -- this file may end up in a public repo.
# Without it set, the admin panel refuses to do anything.
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine = SignalEngine(STRATEGY_CONFIG)
feed = CompositePriceFeed(FEED_CONFIG)

_state_lock = threading.RLock()  # RLock: _sig_to_json takes this lock too, and is called from inside a `with _state_lock` block
_latest_state: dict = {"status": "warming_up"}
_recent_candles: list = []  # [{time, open, high, low, close}, ...]
_pattern_log: list = []  # [{time, pattern, direction, settle_pct}, ...] most recent last
_last_pattern_text: str = ""
_overlays: dict = {}
_chart_patterns: dict = {"flag": None, "double": None}
_closed_bar_pattern: dict = {"pattern": None, "bias": "NEUTRAL", "time": None}
_last_closed_bar_len = 0
_kalshi_strike: dict = {"strike": None, "ticker": None, "close_time": None, "error": "not fetched yet"}
_latest_price = None  # updated by _price_fetch_loop, read by _live_loop -- decoupled so a
_latest_price_lock = threading.Lock()  # network stall in one can never block the other
def _load_initial_rounds() -> list:
    """Prefers GitHub-stored history (survives Render redeploys) if
    that's configured and has data; falls back to the local file
    otherwise -- same as before for anyone not using GitHub storage."""
    if github_store.ENABLED:
        rows = github_store.load_all()
        if rows:
            return rows
    return round_history.load_all()


_persisted_rounds: list = _load_initial_rounds()  # oldest first, loaded once at import time
_scored_count = 0  # how many of engine.completed_windows we've already persisted
_prediction_markers: list = []  # [{time, dir, tier, price}, ...] one per round, oldest first
_marked_this_window = False
_last_window_open_time = None
_exit_markers: list = []  # [{time, against_dir, price, text}, ...] one per reversal-warning onset, oldest first
_reversal_warned_this_window = False

# ---------------- auth: sessions live in memory (friends re-log-in after a restart) ----------------
_sessions: dict = {}  # token -> {"username": str, "last_seen": float, "ip": str}
_sessions_lock = threading.Lock()

PUBLIC_PATHS = {"/login", "/api/login", "/admin"}  # /admin is gated by its own admin-password prompt, not a login


def _get_session(request: Request):
    token = request.cookies.get("session")
    if not token:
        return None
    with _sessions_lock:
        return _sessions.get(token)


def _get_client_ip(request: Request) -> str:
    """Behind Railway/Render's proxy, request.client.host is the proxy's
    own address, not the visitor's -- the real one shows up in
    X-Forwarded-For instead. Falls back to request.client.host for local
    runs, where there's no proxy in front of it."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if (path in PUBLIC_PATHS or path.startswith("/api/admin")
                or path.startswith("/static/") or path == "/favicon.ico"):
            return await call_next(request)
        session = _get_session(request)
        if session is None:
            if path.startswith("/api/"):
                return JSONResponse({"error": "not authenticated"}, status_code=401)
            return RedirectResponse("/login")
        ip = _get_client_ip(request)
        session["last_seen"] = time.time()
        session["ip"] = ip
        auth.touch_last_seen(session["username"], ip=ip)
        return await call_next(request)


app.add_middleware(AuthMiddleware)


def _check_admin(request: Request):
    if not ADMIN_PASSWORD:
        raise HTTPException(status_code=503, detail="ADMIN_PASSWORD not set on the server")
    supplied = request.headers.get("x-admin-password", "")
    if not hmac.compare_digest(supplied, ADMIN_PASSWORD):
        raise HTTPException(status_code=403, detail="wrong admin password")


@app.post("/api/login")
def login(payload: dict, request: Request, response: Response):
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    if not auth.verify_password(username, password):
        raise HTTPException(status_code=401, detail="wrong username or password")
    ip = _get_client_ip(request)
    token = secrets.token_urlsafe(32)
    with _sessions_lock:
        _sessions[token] = {"username": username, "last_seen": time.time(), "ip": ip}
    auth.touch_last_seen(username, ip=ip)
    response.set_cookie("session", token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30)
    return {"ok": True, "username": username}


@app.post("/api/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get("session")
    if token:
        with _sessions_lock:
            _sessions.pop(token, None)
    response.delete_cookie("session")
    return {"ok": True}


@app.get("/api/whoami")
def whoami(request: Request):
    session = _get_session(request)
    if not session:
        raise HTTPException(status_code=401)
    return {"username": session["username"]}


@app.get("/api/admin/users")
def admin_list_users(request: Request):
    _check_admin(request)
    users = auth.load_users()
    with _sessions_lock:
        online_by_username = {}
        for s in _sessions.values():
            online_by_username[s["username"]] = s.get("ip")
    return [
        {"username": u, "created_at": info.get("created_at"), "last_seen": info.get("last_seen"),
         "online": u in online_by_username,
         "ip": online_by_username.get(u) or info.get("last_ip")}
        for u, info in users.items()
    ]


@app.post("/api/admin/add_user")
def admin_add_user(payload: dict, request: Request):
    _check_admin(request)
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")
    auth.add_user(username, password)
    return {"ok": True}


@app.post("/api/admin/remove_user")
def admin_remove_user(payload: dict, request: Request):
    _check_admin(request)
    username = (payload.get("username") or "").strip()
    removed = auth.remove_user(username)
    with _sessions_lock:
        dead_tokens = [t for t, s in _sessions.items() if s["username"] == username]
        for t in dead_tokens:
            del _sessions[t]  # kicks them out immediately, even mid-session
    return {"ok": removed}


def _pattern_bias(pattern_text: str) -> str:
    return PATTERN_BIAS_BY_NAME.get((pattern_text or "").strip(), "NEUTRAL")


def _explain(sig) -> str:
    """Turns the engine's raw fields into a plain-English 'why' string
    for the learning-hub-style explanation card."""
    parts = []
    direction = None
    if sig.confirmed_dir != "HOLD":
        direction = sig.confirmed_dir
    elif sig.early_confirm_dir != "HOLD":
        direction = sig.early_confirm_dir
    elif sig.value_zone_dir != "NONE":
        direction = sig.value_zone_dir
    elif sig.early_pick_dir != "NONE":
        direction = sig.early_pick_dir

    if direction:
        pct = sig.settle_up_pct if direction == "UP" else sig.settle_down_pct
        parts.append(f"The model currently leans {direction} with about {pct:.0f}% modeled probability.")
    else:
        parts.append("No strong lean yet -- the model is still watching this window.")

    parts.append(f"Market regime: {sig.regime} ({sig.trend}), reliability {sig.regime_reliability:.0%}.")

    if sig.pattern_text and sig.pattern_text.lower() not in ("none", ""):
        parts.append(f"Recent candlestick pattern detected: {sig.pattern_text}.")

    if sig.support and sig.resistance:
        parts.append(f"Nearby support ~${sig.support:,.0f}, resistance ~${sig.resistance:,.0f}.")

    if sig.reversal_warning_dir:
        parts.append(f"Caution: reversal warning against {sig.reversal_warning_dir}. {sig.reversal_text}")
    elif sig.held_dir and sig.reversal_chance_pct:
        parts.append(f"Reversal chance against the held {sig.held_dir} call: {sig.reversal_chance_pct:.0f}%.")

    if sig.need_move_text:
        parts.append(sig.need_move_text)

    return " ".join(parts)


def _sig_to_json(sig) -> dict:
    d = asdict(sig)
    d["timestamp"] = sig.timestamp.isoformat()
    d["explanation"] = _explain(sig)
    d["window_open_strike"] = d.pop("strike")  # rename: this is OUR proxy (price at window open), not Kalshi's real one
    with _state_lock:
        d["kalshi_strike"] = _kalshi_strike.get("strike")
        d["kalshi_ticker"] = _kalshi_strike.get("ticker")
        d["kalshi_strike_error"] = _kalshi_strike.get("error")
    return d


def _refresh_candles():
    """Rebuilds the chart candle list straight from the engine's own bar
    builder, so the chart always matches exactly what the model saw."""
    df = engine.bar_builder.closed_and_live_df()
    candles = [
        {"time": int(idx.timestamp()), "open": row.open, "high": row.high,
         "low": row.low, "close": row.close}
        for idx, row in df.tail(1500).iterrows()
    ]
    with _state_lock:
        _recent_candles.clear()
        _recent_candles.extend(candles)


def _maybe_update_closed_bar_pattern():
    """Unlike sig.pattern_text (which re-reads the still-forming live bar
    on every tick, so it can flicker mid-minute), this only updates once
    a bar has actually FINISHED closing -- checked by watching
    engine.bar_builder.bars grow, then re-running the pattern scan on
    real closed bars only. This is what the 'live pattern readout' shows."""
    global _last_closed_bar_len
    bars = engine.bar_builder.bars
    n = len(bars)
    if n <= _last_closed_bar_len:
        return
    _last_closed_bar_len = n
    if n < 3:
        return
    result = detect_all_patterns(bars)
    name = latest_pattern_name(result.iloc[-1])
    with _state_lock:
        _closed_bar_pattern["pattern"] = name if name != "None" else None
        _closed_bar_pattern["bias"] = _pattern_bias(name) if name != "None" else "NEUTRAL"
        _closed_bar_pattern["time"] = bars.index[-1].isoformat()


def _maybe_log_pattern(sig):
    """Appends a pattern-feed entry only when the detected pattern is new
    (not the same one carrying over bar to bar), so the feed reads as a
    stream of fresh detections rather than one pattern repeated forever."""
    global _last_pattern_text
    text = (sig.pattern_text or "").strip()
    if not text or text.lower() == "none":
        _last_pattern_text = ""
        return
    if text == _last_pattern_text:
        return
    _last_pattern_text = text
    direction = None
    if sig.confirmed_dir != "HOLD":
        direction = sig.confirmed_dir
    elif sig.early_confirm_dir != "HOLD":
        direction = sig.early_confirm_dir
    with _state_lock:
        _pattern_log.append({
            "time": sig.timestamp.isoformat(),
            "pattern": text,
            "bias": _pattern_bias(text),
            "model_direction": direction,
            "price": sig.price,
        })
        del _pattern_log[:-30]  # keep the most recent 30


def _maybe_mark_prediction(sig):
    """Records a chart marker the moment the model makes its FIRST call
    for the current 15-min round (whichever tier fires first) -- this is
    the model's prediction for how the whole round will settle, not a
    per-bar signal. One marker per round; resets when a new round starts."""
    global _marked_this_window, _reversal_warned_this_window, _last_window_open_time
    w = engine.window
    if w.open_time != _last_window_open_time:
        _last_window_open_time = w.open_time
        _marked_this_window = False
        _reversal_warned_this_window = False
    if not _marked_this_window and w.first_signal_tier != "NONE" and w.first_signal_dir != "NONE":
        _marked_this_window = True
        with _state_lock:
            _prediction_markers.append({
                "time": sig.timestamp.isoformat(),
                "dir": w.first_signal_dir,
                "tier": w.first_signal_tier,
                "price": sig.price,
            })
            del _prediction_markers[:-100]  # keep the most recent 100


def _maybe_mark_exit_warning(sig):
    """Records a chart marker the moment the reversal-check first flags
    a call as at risk of flipping this round -- your bot's 'Confirmed'
    label is locked to whichever direction fired first and never
    literally changes (see the Signal Tiers tab), so this reversal
    warning IS the closest thing to a live flip signal: it's the engine
    telling you the original call may no longer be the right side. One
    marker per round, the first time it fires."""
    global _reversal_warned_this_window
    if not sig.reversal_warning_dir:
        return
    if _reversal_warned_this_window:
        return
    _reversal_warned_this_window = True
    with _state_lock:
        _exit_markers.append({
            "time": sig.timestamp.isoformat(),
            "against_dir": sig.reversal_warning_dir,  # the direction being warned AGAINST (i.e. exit this held call)
            "price": sig.price,
            "text": sig.reversal_text,
        })
        del _exit_markers[:-100]


def _round_result_json(w) -> dict:
    """One completed 15-min round: the first call the model made, the
    strike it needed to clear, where price actually settled, and whether
    that first call won."""
    return {
        "close_time": w.close_time.isoformat(),
        "strike": w.strike,
        "settle_price": w.settle_price,
        "actual_dir": w.actual_dir,
        "first_signal_tier": w.first_signal_tier,
        "first_signal_dir": w.first_signal_dir,
        "first_signal_correct": w.first_signal_correct,
        "confirmed_dir": w.confirmed_dir,
        "confirmed_correct": w.confirmed_correct,
    }


def _current_round_json() -> dict:
    """The in-progress round: what the model has called so far and the
    strike it's measured against, so the feed can show a 'live' row
    before the round actually settles."""
    w = engine.window
    tier = w.first_signal_tier
    direction = w.first_signal_dir if tier != "NONE" else None
    with _state_lock:
        kalshi_strike = _kalshi_strike.get("strike")
    return {
        "close_time": None,
        "strike": kalshi_strike if kalshi_strike is not None else w.strike,
        "window_open_strike": w.strike,
        "settle_price": None,
        "actual_dir": None,
        "first_signal_tier": tier,
        "first_signal_dir": direction,
        "first_signal_correct": None,
        "confirmed_dir": w.confirmed_dir,
        "confirmed_correct": None,
    }


def _kalshi_strike_loop():
    """Fetches the REAL Kalshi strike for the current open 15-min BTC
    market every 20s -- public endpoint, no key needed. This is Kalshi's
    own actual number, not our engine's window-open-price approximation."""
    while True:
        try:
            result = get_current_kalshi_strike()
            with _state_lock:
                _kalshi_strike.clear()
                _kalshi_strike.update(result)
        except Exception as e:
            print(f"kalshi strike loop error: {e}")
        time.sleep(20)


def _overlay_loop():
    """Refreshes VWAP/bands/exhaustion/S-R overlays every 30s in its own
    thread -- these need volume data pulled fresh from Coinbase, which is
    heavier than the tick loop and doesn't need to run every second.
    12 hours instead of the original 3 -- enough to actually look back at
    what the indicators were doing earlier, without re-fetching such a
    huge window every single cycle that it strains Coinbase's public API."""
    while True:
        try:
            end = datetime.now(timezone.utc) - timedelta(minutes=1)
            start = end - timedelta(hours=12)
            candles = fetch_1m_candles_with_volume(start, end, pause=0.05)
            result = compute_overlays(candles, STRATEGY_CONFIG.pivot_left,
                                       STRATEGY_CONFIG.pivot_right, STRATEGY_CONFIG.rsi_len)
            with _state_lock:
                _overlays.clear()
                _overlays.update(result)

            # chart structure scanner reuses the same fetched candles --
            # no extra network round trip needed
            df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
            df = df.set_index("time")
            flag = detect_flag(df)
            double = detect_double_top_bottom(df, STRATEGY_CONFIG.pivot_left, STRATEGY_CONFIG.pivot_right)
            with _state_lock:
                _chart_patterns["flag"] = flag
                _chart_patterns["double"] = double
        except Exception as e:
            print(f"overlay loop error: {e}")
        time.sleep(30)


def _persist_new_rounds():
    """Whenever the engine has scored a new completed window since we
    last checked, write it to disk immediately -- so history survives
    even if the server is killed a second later, not just on clean exit.
    Also pushes to GitHub if that's configured, since that's the copy
    that survives a Render redeploy; the local file alone doesn't."""
    global _scored_count
    total = len(engine.completed_windows)
    if total > _scored_count:
        for w in engine.completed_windows[_scored_count:total]:
            row = _round_result_json(w)
            round_history.append(row)
            github_store.append(row)
            with _state_lock:
                _persisted_rounds.append(row)
        _scored_count = total


def _price_fetch_loop():
    """Runs the actual network calls to the 3 exchanges in its own
    dedicated thread, forever, storing whatever price it gets into a
    shared variable that _live_loop just reads. This is the fix for the
    ~20-minute candle gap: the seed-thread hang wasn't a one-off -- a
    single feed.refresh() call can apparently also stall past its stated
    timeout on this environment (same mysterious behavior, different
    call site). Decoupling the fetch from the tick-processing loop means
    if THIS thread ever stalls on one bad call, _live_loop keeps ticking
    every second on whatever price it last received, instead of the
    entire candle/signal pipeline freezing until the stuck call resolves."""
    while True:
        try:
            p = feed.refresh()
            if p is not None:
                global _latest_price
                with _latest_price_lock:
                    _latest_price = p
        except Exception as e:
            print(f"price fetch loop error: {e}")
        time.sleep(FEED_CONFIG.poll_seconds)


def _live_loop():
    global _last_closed_bar_len
    print(f"_live_loop starting: seeding history in a watchdog thread...")
    seed_cancelled = threading.Event()

    def _seed_and_finish():
        try:
            end = datetime.now(timezone.utc) - timedelta(minutes=1)
            start = end - timedelta(minutes=180)
            candles = fetch_1m_candles(start, end, pause=0.1)
        except Exception as e:
            print(f"Preload failed ({e}); warming up live instead.")
            return
        if seed_cancelled.is_set():
            # the watchdog already gave up and the live loop has moved on --
            # writing these (now-stale, possibly out-of-order) bars into the
            # engine at this point would corrupt the chart (this is exactly
            # what caused the gap + the chart breaking last time), so this
            # fetch's results get thrown away entirely instead of merged.
            print(f"seeding finished late ({len(candles)} candles) but was already cancelled -- discarding, not merging")
            return
        for ts, o, h, l, c in candles:
            engine.update_bar(o, h, l, c, ts)
        _refresh_candles()
        global _last_closed_bar_len
        _last_closed_bar_len = len(engine.bar_builder.bars)
        _persist_new_rounds()
        print(f"seeding finished: bar_builder has {len(engine.bar_builder.bars)} closed bars")

    seed_thread = threading.Thread(target=_seed_and_finish, daemon=True)
    seed_thread.start()
    seed_thread.join(timeout=20)  # hard cap -- if seeding isn't done by now, stop waiting and go live anyway
    if seed_thread.is_alive():
        seed_cancelled.set()
        print("seeding still running after 20s -- proceeding to live ticks without it. "
              "Its results will be discarded if it ever does finish, so it can't corrupt the chart later.")

    threading.Thread(target=_price_fetch_loop, daemon=True).start()

    tick_count = 0
    none_price_streak = 0
    none_sig_streak = 0
    while True:
        try:
            tick_count += 1
            with _latest_price_lock:
                price = _latest_price
            if price is None:
                none_price_streak += 1
                if none_price_streak in (1, 10, 30) or none_price_streak % 60 == 0:
                    print(f"live tick #{tick_count}: no price available yet from the fetch loop "
                          f"({none_price_streak} in a row) -- coinbase_ok={feed.coinbase.ok} "
                          f"kraken_ok={feed.kraken.ok} gemini_ok={feed.gemini.ok}")
            if price is not None:
                none_price_streak = 0
                sig = engine.update(price)
                _refresh_candles()
                _maybe_update_closed_bar_pattern()
                if sig is None:
                    none_sig_streak += 1
                    if none_sig_streak in (1, 10, 30) or none_sig_streak % 60 == 0:
                        print(f"live tick #{tick_count}: price={price} but engine.update() returned None "
                              f"({none_sig_streak} in a row)")
                if sig is not None:
                    none_sig_streak = 0
                    with _state_lock:
                        _latest_state.clear()
                        _latest_state.update(_sig_to_json(sig))
                        _latest_state["status"] = "live"
                    _maybe_log_pattern(sig)
                    _maybe_mark_prediction(sig)
                    _maybe_mark_exit_warning(sig)
                _persist_new_rounds()
        except Exception as e:
            print(f"live loop error: {e}")
        time.sleep(FEED_CONFIG.poll_seconds)


@app.on_event("startup")
def _startup():
    threading.Thread(target=_live_loop, daemon=True).start()
    threading.Thread(target=_overlay_loop, daemon=True).start()
    threading.Thread(target=_kalshi_strike_loop, daemon=True).start()


@app.get("/api/state")
def get_state():
    with _state_lock:
        return dict(_latest_state)


@app.get("/api/candles")
def get_candles():
    with _state_lock:
        return list(_recent_candles[-1500:])


@app.get("/api/patterns")
def get_patterns():
    with _state_lock:
        return list(reversed(_pattern_log))  # most recent first


@app.get("/api/prediction_markers")
def get_prediction_markers():
    """One entry per round: the exact bar where the model made its first
    call, for plotting UP/DOWN arrows on the chart."""
    with _state_lock:
        return list(_prediction_markers)


@app.get("/api/exit_markers")
def get_exit_markers():
    """One entry per round (at most): the exact bar where the reversal
    check first flagged the held call as at risk of flipping."""
    with _state_lock:
        return list(_exit_markers)


@app.get("/api/rounds")
def get_rounds():
    """Most recent 15-min rounds, most recent first: the current
    in-progress round (if a call has fired yet) followed by completed
    ones with their real outcome -- pulled from the on-disk history, so
    this survives server restarts instead of resetting to empty."""
    rows = []
    current = _current_round_json()
    if current["first_signal_dir"] is not None or current["strike"] is not None:
        rows.append(current)
    with _state_lock:
        completed = list(reversed(_persisted_rounds[-100:]))
    rows.extend(completed)
    return rows


@app.get("/api/overlays")
def get_overlays():
    with _state_lock:
        return dict(_overlays)


@app.get("/api/chart_patterns")
def get_chart_patterns():
    """The currently-detected chart structure(s), if any -- a bull/bear
    flag and/or a double top/bottom, each with a quality score and the
    trendline points needed to draw it."""
    with _state_lock:
        return dict(_chart_patterns)


@app.get("/api/closed_bar_pattern")
def get_closed_bar_pattern():
    """The candlestick pattern on the most recently CLOSED bar only --
    updates once a minute when a bar finishes, not every tick."""
    with _state_lock:
        return dict(_closed_bar_pattern)


@app.get("/api/kalshi_strike")
def get_kalshi_strike():
    with _state_lock:
        return dict(_kalshi_strike)


@app.get("/api/pattern_catalog")
def get_pattern_catalog():
    """The full 50-pattern reference list (name, bias, candle count) --
    single source of truth shared with the engine's own pattern_text."""
    return [{"key": k, "name": name, "bias": bias, "candles": cnt} for k, name, bias, cnt in PATTERN_CATALOG]


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/login")
def login_page():
    return FileResponse(FRONTEND_DIR / "login.html")


@app.get("/admin")
def admin_page():
    return FileResponse(FRONTEND_DIR / "admin.html")
