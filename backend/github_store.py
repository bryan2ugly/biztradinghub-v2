"""
Optional persistent round-history storage using a JSON file stored in
YOUR GitHub repo, via GitHub's Contents API. This exists specifically
because Render's free tier has no persistent disk -- every redeploy
wipes round_history.py's local file clean. Storing the history in your
GitHub repo instead means it survives redeploys, restarts, everything,
as long as the repo exists -- no new account or paid service needed,
you already have this one.

Entirely optional: if GITHUB_TOKEN / GITHUB_REPO aren't set as
environment variables, every function here quietly no-ops and
round_history.py's local file keeps working exactly as before (fine
for local runs; just won't survive a Render redeploy on its own).
"""
from __future__ import annotations

import base64
import json
import os
import threading

import requests

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")  # e.g. "bryan2ugly/Testing-stuff"
GITHUB_HISTORY_PATH = os.environ.get("GITHUB_HISTORY_PATH", "round_history_store.json")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")

ENABLED = bool(GITHUB_TOKEN and GITHUB_REPO)

_lock = threading.Lock()


def _headers():
    return {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}


def _api_url():
    return f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_HISTORY_PATH}"


def load_all() -> list:
    """Fetches the full round history from GitHub. Returns [] if this
    isn't enabled, the file doesn't exist yet, or the request fails --
    callers should treat that as 'no GitHub history available' and can
    fall back to the local file."""
    if not ENABLED:
        return []
    try:
        resp = requests.get(_api_url(), headers=_headers(), params={"ref": GITHUB_BRANCH}, timeout=15)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
        content = base64.b64decode(data["content"]).decode("utf-8")
        return json.loads(content) if content.strip() else []
    except Exception as e:
        print(f"github_store load failed: {e}")
        return []


def append(round_dict: dict):
    """Appends one round to the GitHub-stored history file. Fetches the
    current file (and its required SHA) fresh each call rather than
    caching -- round completions happen at most once every 15 minutes,
    so the extra GET before each PUT is negligible, and it avoids ever
    writing with a stale SHA."""
    if not ENABLED:
        return
    with _lock:
        try:
            resp = requests.get(_api_url(), headers=_headers(), params={"ref": GITHUB_BRANCH}, timeout=15)
            if resp.status_code == 404:
                rows, sha = [], None
            else:
                resp.raise_for_status()
                data = resp.json()
                sha = data.get("sha")
                content = base64.b64decode(data["content"]).decode("utf-8")
                rows = json.loads(content) if content.strip() else []

            rows.append(round_dict)
            rows = rows[-2000:]  # cap so the file doesn't grow forever

            new_content = base64.b64encode(json.dumps(rows, indent=2).encode("utf-8")).decode("utf-8")
            payload = {"message": "round history update", "content": new_content, "branch": GITHUB_BRANCH}
            if sha:
                payload["sha"] = sha
            put_resp = requests.put(_api_url(), headers=_headers(), json=payload, timeout=15)
            put_resp.raise_for_status()
        except Exception as e:
            print(f"github_store append failed: {e}")
