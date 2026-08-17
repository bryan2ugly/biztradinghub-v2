"""
Persists completed 15-min rounds to a local JSONL file so the
round-by-round track record survives server restarts -- without this,
engine.completed_windows lives only in RAM and resets to empty every
time you stop and re-run uvicorn.

One JSON object per line, append-only. Simple on purpose: this is a
personal dashboard's history, not a database that needs querying.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

HISTORY_FILE = Path(__file__).resolve().parent / "rounds_history.jsonl"
MAX_KEPT = 2000  # trim the file if it grows past this many rounds

_write_lock = threading.Lock()


def load_all() -> list:
    """Returns all persisted rounds, oldest first."""
    if not HISTORY_FILE.exists():
        return []
    rows = []
    with open(HISTORY_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip a corrupted line rather than fail the whole load
    return rows


def append(round_dict: dict):
    with _write_lock:
        with open(HISTORY_FILE, "a") as f:
            f.write(json.dumps(round_dict) + "\n")
        _maybe_trim()


def _maybe_trim():
    """Keeps the file from growing forever -- rewrites it capped to the
    most recent MAX_KEPT rounds. Only runs the trim on the write path,
    and only when actually over the cap, so normal appends stay cheap."""
    rows = load_all()
    if len(rows) > MAX_KEPT:
        trimmed = rows[-MAX_KEPT:]
        with open(HISTORY_FILE, "w") as f:
            for r in trimmed:
                f.write(json.dumps(r) + "\n")
