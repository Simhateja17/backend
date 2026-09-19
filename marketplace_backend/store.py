from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any

from .core_schema import CORE_DDL, table_names, to_sqlite


_TABLE_NAMES = (
    "products", "carts", "orders", "approvals",
    "customers", "merchant_operators", "customer_carts", "cart_lines", "idempotency_records",
) + table_names()

# Qualify a bare table name into the `cartisan` schema, but leave alone anything
# that is not one: a column alias (`COUNT(*) AS orders`) and an already-qualified
# name (`cartisan.orders`) both share the word but neither is a table reference.
_QUALIFIERS = {
    table: re.compile(rf"(\bAS\s+)?(?<![.\w]){table}\b", re.IGNORECASE)
    for table in _TABLE_NAMES
}


class _TxHandle:
    """One connection, mid-transaction: same execute/rows shape as Store, no per-call commit."""

    def __init__(self, connection: Any, sql_fn: Any) -> None:
        self._connection, self._sql_fn = connection, sql_fn

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        return self._connection.execute(self._sql_fn(sql), params)

    def rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        return [dict(row) for row in self._connection.execute(self._sql_fn(sql), params).fetchall()]


class Store:
    """Small persistence boundary with SQLite for tests and Supabase Postgres in runtime."""

    def __init__(self, path: str | Path | None = None, database_url: str | None = None) -> None:
        self._lock = RLock()
        self._pool = None
        self._db = None

        # An explicit path always selects SQLite. This keeps tests isolated even if
        # the developer shell has SUPABASE_DATABASE_URL configured.
        if database_url and path is None:
            self.backend = "supabase"
            self.path = None
            self._connect_postgres(database_url)
        else:
            self.backend = "sqlite"
            self.path = str(path or Path(__file__).with_name("cartisan.db"))
            self._connect_sqlite()

    def _connect_sqlite(self) -> None:
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        # SQLite ignores foreign keys unless asked. Postgres never does, so leaving
        # this off would let a test pass against a schema production would reject.
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS products(id TEXT PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL,
          description TEXT NOT NULL, price INTEGER NOT NULL, stock INTEGER NOT NULL, rating TEXT,
          image_label TEXT NOT NULL, cross_sell_of TEXT, variant_of TEXT, options_json TEXT,
          option_values_json TEXT, active INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE IF NOT EXISTS carts(session_id TEXT, product_id TEXT, quantity INTEGER NOT NULL,
          PRIMARY KEY(session_id, product_id));
        CREATE TABLE IF NOT EXISTS orders(id TEXT PRIMARY KEY, session_id TEXT NOT NULL, status TEXT NOT NULL,
          amount INTEGER NOT NULL, payment_link_id TEXT, payment_url TEXT, payload TEXT NOT NULL, created_at TEXT NOT NULL,
          customer_id TEXT, origin TEXT NOT NULL DEFAULT 'live_app');
        CREATE TABLE IF NOT EXISTS approvals(id TEXT PRIMARY KEY, kind TEXT NOT NULL, target_id TEXT,
          before_json TEXT NOT NULL, after_json TEXT NOT NULL, reasoning TEXT NOT NULL, status TEXT NOT NULL,
          created_at TEXT NOT NULL, decided_at TEXT);
        CREATE TABLE IF NOT EXISTS customers(id TEXT PRIMARY KEY, email TEXT NOT NULL, display_name TEXT,
          origin TEXT NOT NULL DEFAULT 'live_app', created_at TEXT NOT NULL DEFAULT (datetime('now')));
        CREATE TABLE IF NOT EXISTS merchant_operators(id TEXT PRIMARY KEY, email TEXT NOT NULL,
          display_name TEXT, paytm_plan TEXT NOT NULL DEFAULT 'pos' CHECK (paytm_plan IN ('pos','payments')),
          created_at TEXT NOT NULL DEFAULT (datetime('now')));
        CREATE TABLE IF NOT EXISTS customer_carts(id TEXT PRIMARY KEY, customer_id TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','checked_out','abandoned')),
          state_version INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at TEXT NOT NULL DEFAULT (datetime('now')));
        CREATE UNIQUE INDEX IF NOT EXISTS customer_carts_one_active_idx
          ON customer_carts(customer_id) WHERE status='active';
        CREATE TABLE IF NOT EXISTS cart_lines(cart_id TEXT NOT NULL, product_id TEXT NOT NULL,
          quantity INTEGER NOT NULL CHECK (quantity BETWEEN 1 AND 10), PRIMARY KEY(cart_id, product_id));
        CREATE TABLE IF NOT EXISTS idempotency_records(key TEXT NOT NULL, principal_id TEXT NOT NULL,
          operation TEXT NOT NULL, request_fingerprint TEXT NOT NULL, response_json TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT (datetime('now')), PRIMARY KEY(principal_id, key));
        """)
        # The normalized commerce core comes from the one shared definition, so the
        # schema the tests exercise is the schema the migration applies.
        self._db.executescript(to_sqlite(CORE_DDL))
        self._db.commit()

    def _connect_postgres(self, database_url: str) -> None:
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise RuntimeError(
                "Supabase is configured but the Postgres driver is missing. "
                "Run: pip install -r requirements.txt"
            ) from exc

        self._pool = ConnectionPool(
            conninfo=database_url,
            min_size=1,
            max_size=5,
            timeout=10,
            kwargs={"autocommit": True, "row_factory": dict_row},
            open=True,
        )
        self._pool.wait(timeout=10)
        self.rows("SELECT 1 AS ready")

    def _postgres_sql(self, sql: str) -> str:
        # Escape literal percent signs first — a `LIKE 'sd_%'` in the query would
        # otherwise be read by psycopg as a malformed placeholder — then convert our
        # `?` markers. Order matters: escaping after the conversion would corrupt
        # the `%s` markers we just produced.
        rewritten = sql.replace("%", "%%").replace("?", "%s")
        rewritten = rewritten.replace("MIN(%s,carts.quantity+%s)", "LEAST(%s,carts.quantity+%s)")
        for table in _TABLE_NAMES:
            rewritten = _QUALIFIERS[table].sub(
                lambda match, name=table: match.group(0) if match.group(1) else f"cartisan.{name}",
                rewritten)
        return rewritten

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        if self.backend == "supabase":
            with self._pool.connection() as connection:
                return connection.execute(self._postgres_sql(sql), params)
        with self._lock:
            cursor = self._db.execute(sql, params)
            self._db.commit()
            return cursor

    def rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        if self.backend == "supabase":
            with self._pool.connection() as connection:
                return list(connection.execute(self._postgres_sql(sql), params).fetchall())
        with self._lock:
            return [dict(row) for row in self._db.execute(sql, params).fetchall()]

    @contextmanager
    def transaction(self):
        """Run several statements as one atomic write, so a multi-step invariant
        (reserve stock while capping a cart line) can't be split by a concurrent request."""
        if self.backend == "supabase":
            with self._pool.connection() as connection:
                with connection.transaction():
                    yield _TxHandle(connection, self._postgres_sql)
            return
        with self._lock:
            try:
                yield _TxHandle(self._db, lambda sql: sql)
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    def executemany(self, sql: str, params: list[tuple[Any, ...]]) -> None:
        if self.backend == "supabase":
            with self._pool.connection() as connection:
                with connection.transaction():
                    with connection.cursor() as cursor:
                        cursor.executemany(self._postgres_sql(sql), params)
            return
        with self._lock:
            self._db.executemany(sql, params)
            self._db.commit()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
        if self._db is not None:
            self._db.close()

    @staticmethod
    def dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def load(value: str) -> Any:
        return json.loads(value)
