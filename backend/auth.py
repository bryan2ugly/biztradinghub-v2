"""
Friend accounts for the site. No third-party auth library needed --
passwords are salted + hashed with PBKDF2 (Python's stdlib hashlib),
stored in a local JSON file. This is deliberately simple: it's gating
a handful of friends, not building a production auth system.

users.json is created the first time you add a user and lives next to
this file. It's runtime data, not code -- delete it before uploading
this project to a public GitHub repo (see README).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from pathlib import Path

USERS_FILE = Path(__file__).resolve().parent / "users.json"


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()


def load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    try:
        return json.loads(USERS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2))


def add_user(username: str, password: str):
    users = load_users()
    salt = secrets.token_hex(16)
    users[username] = {
        "salt": salt,
        "password_hash": _hash_password(password, salt),
        "created_at": time.time(),
        "last_seen": None,
    }
    save_users(users)


def remove_user(username: str) -> bool:
    users = load_users()
    if username not in users:
        return False
    del users[username]
    save_users(users)
    return True


def verify_password(username: str, password: str) -> bool:
    users = load_users()
    u = users.get(username)
    if not u:
        return False
    return hmac.compare_digest(_hash_password(password, u["salt"]), u["password_hash"])


def touch_last_seen(username: str, ip: str | None = None):
    users = load_users()
    if username in users:
        users[username]["last_seen"] = time.time()
        if ip:
            users[username]["last_ip"] = ip
        save_users(users)
