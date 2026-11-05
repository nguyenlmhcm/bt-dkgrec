"""Shop records that do not belong in the graph: carts, orders, campaigns, event log."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS cart(visitor INTEGER, item INTEGER, qty INTEGER, PRIMARY KEY(visitor, item));
CREATE TABLE IF NOT EXISTS orders(id INTEGER PRIMARY KEY AUTOINCREMENT, visitor INTEGER, total INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS order_items(order_id INTEGER, item INTEGER, qty INTEGER, price INTEGER, campaign_id INTEGER);
CREATE TABLE IF NOT EXISTS campaigns(id INTEGER PRIMARY KEY AUTOINCREMENT, item INTEGER, discount INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS campaign_targets(campaign_id INTEGER, visitor INTEGER, seen REAL, clicked REAL, bought REAL,
    PRIMARY KEY(campaign_id, visitor));
CREATE TABLE IF NOT EXISTS event_log(id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, visitor INTEGER, item INTEGER,
    behavior TEXT, source TEXT, total_ms REAL);
"""


class ShopDB:
    """Thin SQLite wrapper. One connection per call, so request threads never share one."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    def rows(self, sql: str, *args: object) -> list[dict]:
        with self._connect() as con:
            return [dict(r) for r in con.execute(sql, args).fetchall()]

    def one(self, sql: str, *args: object) -> dict | None:
        rows = self.rows(sql, *args)
        return rows[0] if rows else None

    def run(self, sql: str, *args: object) -> int:
        """Execute one write; return the new row id."""
        with self._connect() as con:
            return int(con.execute(sql, args).lastrowid or 0)

    def clear(self) -> None:
        with self._connect() as con:
            for table in ("cart", "orders", "order_items", "campaigns", "campaign_targets", "event_log", "sqlite_sequence"):
                con.execute(f"DELETE FROM {table}")  # noqa: S608 - fixed names only
