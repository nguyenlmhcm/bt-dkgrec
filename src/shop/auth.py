"""Demo accounts: salted password hashes and server-side sessions in SQLite."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

from src.shop.db import ShopDB

SESSION_SECONDS = 7 * 24 * 3600
ROLES = ("customer", "store", "provider")
COOKIE = {role: f"bt_{role}" for role in ROLES}
_ROUNDS = 200_000
_MAX_FAILS, _LOCK_SECONDS = 5, 900


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _ROUNDS).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    salt, _, _digest = stored.partition("$")
    return hmac.compare_digest(hash_password(password, salt), stored)


class Accounts:
    """Users, sessions and a per-username lock after repeated failures."""

    def __init__(self, db: ShopDB, clock=time.time) -> None:
        self.db = db
        self._clock = clock
        self._fails: dict[str, list[float]] = {}

    def upsert(self, username: str, password: str, role: str, display: str, visitor_id: int | None = None) -> None:
        if role not in ROLES:
            raise ValueError(role)
        self.db.run("INSERT INTO users(username, hash, role, display, visitor_id) VALUES (?,?,?,?,?) "
                    "ON CONFLICT(username) DO UPDATE SET hash=excluded.hash, role=excluded.role, "
                    "display=excluded.display, visitor_id=excluded.visitor_id",
                    username, hash_password(password), role, display, visitor_id)
        self.db.run("DELETE FROM sessions WHERE username=?", username)

    def login(self, username: str, password: str) -> tuple[str, dict] | None:
        """Return ``(token, user)``, or ``None`` when refused. Failures do not say why."""
        now = self._clock()
        recent = [t for t in self._fails.get(username, []) if now - t < _LOCK_SECONDS]
        if len(recent) >= _MAX_FAILS:
            return None
        user = self.db.one("SELECT * FROM users WHERE username=?", username)
        if user is None or not verify_password(password, user["hash"]):
            self._fails[username] = [*recent, now]
            return None
        self._fails.pop(username, None)
        token = secrets.token_urlsafe(32)
        self.db.run("INSERT INTO sessions(token, username, expires) VALUES (?,?,?)",
                    hashlib.sha256(token.encode()).hexdigest(), username, now + SESSION_SECONDS)
        return token, self._public(user)

    def user_for(self, token: str | None) -> dict | None:
        if not token:
            return None
        row = self.db.one("SELECT u.*, s.expires FROM sessions s JOIN users u ON u.username=s.username WHERE s.token=?",
                          hashlib.sha256(token.encode()).hexdigest())
        return self._public(row) if row and row["expires"] > self._clock() else None

    def logout(self, token: str | None) -> None:
        if token:
            self.db.run("DELETE FROM sessions WHERE token=?", hashlib.sha256(token.encode()).hexdigest())

    @staticmethod
    def _public(row: dict) -> dict:
        return {"username": row["username"], "role": row["role"], "display": row["display"], "visitor_id": row["visitor_id"]}
