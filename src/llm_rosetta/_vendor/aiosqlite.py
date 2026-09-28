# /// zerodep
# version = "0.1.0"
# deps = []
# tier = "subsystem"
# category = "storage"
# note = "Install/update via: https://zerodep.readthedocs.io/en/latest/guide/cli/"
# ///
"""Async wrapper for sqlite3 — dedicated worker thread, zero dependencies.

Part of zerodep: https://github.com/Oaklight/zerodep
Copyright (c) 2026 Peng Ding. MIT License.

Provides an async interface to Python's built-in ``sqlite3`` module using a
dedicated worker thread per connection.  All blocking SQLite operations run
on the worker thread and are awaited from asyncio coroutines, preventing
database I/O from blocking the event loop.

API mirrors ``sqlite3.Connection`` and ``sqlite3.Cursor``::

    async with aiosqlite.connect("app.db") as db:
        await db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        await db.execute("INSERT INTO kv VALUES (?, ?)", ("hello", "world"))
        await db.commit()

        async for row in await db.execute("SELECT * FROM kv"):
            print(row)

        rows = await db.execute_fetchall("SELECT * FROM kv WHERE k LIKE ?", ("%h%",))

Requirements:
    Python >= 3.10, no third-party packages.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import queue
import sqlite3
import threading
import warnings
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any

logger = logging.getLogger(__name__)

# Re-export sqlite3 types for convenience
Row = sqlite3.Row
Warning = sqlite3.Warning  # noqa: A001
Error = sqlite3.Error
DatabaseError = sqlite3.DatabaseError
InterfaceError = sqlite3.InterfaceError
DataError = sqlite3.DataError
OperationalError = sqlite3.OperationalError
IntegrityError = sqlite3.IntegrityError
InternalError = sqlite3.InternalError
ProgrammingError = sqlite3.ProgrammingError
NotSupportedError = sqlite3.NotSupportedError
paramstyle = sqlite3.paramstyle
sqlite_version = sqlite3.sqlite_version
sqlite_version_info = sqlite3.sqlite_version_info
register_adapter = sqlite3.register_adapter
register_converter = sqlite3.register_converter

_STOP = object()
_DEFAULT_ITER_CHUNK = 64

__all__ = [
    "connect",
    "Connection",
    "Cursor",
    "Row",
    "Warning",
    "Error",
    "DatabaseError",
    "InterfaceError",
    "DataError",
    "OperationalError",
    "IntegrityError",
    "InternalError",
    "ProgrammingError",
    "NotSupportedError",
    "paramstyle",
    "sqlite_version",
    "sqlite_version_info",
    "register_adapter",
    "register_converter",
]


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------


class Cursor:
    """Async wrapper around ``sqlite3.Cursor``.

    Returned by :meth:`Connection.execute` and friends.  Supports async
    iteration via ``async for row in cursor``.
    """

    __slots__ = ("_cursor", "_conn")

    def __init__(self, cursor: sqlite3.Cursor, conn: Connection) -> None:
        self._cursor = cursor
        self._conn = conn

    # -- Async query methods ------------------------------------------------

    async def execute(self, sql: str, parameters: Any = ()) -> Cursor:
        await self._conn._execute(self._cursor.execute, sql, parameters)
        return self

    async def executemany(self, sql: str, seq_of_parameters: Iterable[Any]) -> Cursor:
        await self._conn._execute(self._cursor.executemany, sql, seq_of_parameters)
        return self

    async def executescript(self, sql_script: str) -> Cursor:
        await self._conn._execute(self._cursor.executescript, sql_script)
        return self

    async def fetchone(self) -> Any:
        return await self._conn._execute(self._cursor.fetchone)

    async def fetchmany(self, size: int | None = None) -> list[Any]:
        if size is None:
            size = self._cursor.arraysize
        return await self._conn._execute(self._cursor.fetchmany, size)

    async def fetchall(self) -> list[Any]:
        return await self._conn._execute(self._cursor.fetchall)

    async def close(self) -> None:
        await self._conn._execute(self._cursor.close)

    # -- Sync properties (safe to read from any thread) ---------------------

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    @property
    def lastrowid(self) -> int | None:
        return self._cursor.lastrowid

    @property
    def description(self) -> Any:
        return self._cursor.description

    @property
    def arraysize(self) -> int:
        return self._cursor.arraysize

    @arraysize.setter
    def arraysize(self, value: int) -> None:
        self._cursor.arraysize = value

    @property
    def connection(self) -> Connection:
        return self._conn

    # -- Async iteration ----------------------------------------------------

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        row = await self.fetchone()
        if row is None:
            raise StopAsyncIteration
        return row


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


class Connection:
    """Async wrapper around ``sqlite3.Connection``.

    Do not instantiate directly — use :func:`connect`.
    """

    def __init__(
        self,
        connector: Callable[[], sqlite3.Connection],
        iter_chunk_size: int = _DEFAULT_ITER_CHUNK,
    ) -> None:
        self._connector = connector
        self._iter_chunk_size = iter_chunk_size
        self._conn: sqlite3.Connection | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: queue.SimpleQueue[
            tuple[
                asyncio.Future[Any] | None,
                Callable[..., Any] | object | None,
                tuple[Any, ...],
            ]
        ] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._closed = False

    # -- Worker thread ------------------------------------------------------

    def _worker(self) -> None:
        while True:
            future, fn, args = self._queue.get()
            if fn is _STOP:
                break
            assert callable(fn) and future is not None
            try:
                result = fn(*args)  # ty: ignore[call-top-callable]
            except BaseException as exc:
                if self._loop and not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(future.set_exception, exc)
            else:
                if self._loop and not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(future.set_result, result)

    # -- Internal dispatch --------------------------------------------------

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._closed:
            raise ProgrammingError("Cannot operate on a closed connection.")
        if self._conn is None:
            raise ProgrammingError("Connection not started.")
        return self._conn

    async def _execute(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self._closed:
            raise ProgrammingError("Cannot operate on a closed connection.")
        if self._loop is None:
            raise ProgrammingError("Connection not started.")
        future = self._loop.create_future()
        self._queue.put((future, fn, args))
        return await future

    # -- Lifecycle ----------------------------------------------------------

    async def _start(self) -> Connection:
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="aiosqlite-worker"
        )
        self._thread.start()
        self._conn = await self._execute(self._connector)
        return self

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._conn is not None:
            try:
                await self._execute(self._conn.close)
            except Exception:
                logger.debug("Error closing sqlite3 connection", exc_info=True)
        self._queue.put((None, _STOP, ()))
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._conn = None
        self._thread = None

    async def __aenter__(self) -> Connection:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __del__(self) -> None:
        if not self._closed and self._conn is not None:
            warnings.warn(
                "Unclosed aiosqlite connection",
                ResourceWarning,
                stacklevel=2,
            )
            self._queue.put((None, _STOP, ()))

    # -- Query methods ------------------------------------------------------

    async def execute(self, sql: str, parameters: Any = ()) -> Cursor:
        conn = self._ensure_conn()
        cursor = await self._execute(conn.execute, sql, parameters)
        return Cursor(cursor, self)

    async def executemany(self, sql: str, seq_of_parameters: Iterable[Any]) -> Cursor:
        conn = self._ensure_conn()
        cursor = await self._execute(conn.executemany, sql, seq_of_parameters)
        return Cursor(cursor, self)

    async def executescript(self, sql_script: str) -> Cursor:
        conn = self._ensure_conn()
        cursor = await self._execute(conn.executescript, sql_script)
        return Cursor(cursor, self)

    async def commit(self) -> None:
        conn = self._ensure_conn()
        await self._execute(conn.commit)

    async def rollback(self) -> None:
        conn = self._ensure_conn()
        await self._execute(conn.rollback)

    async def cursor(self) -> Cursor:
        conn = self._ensure_conn()
        raw = await self._execute(conn.cursor)
        return Cursor(raw, self)

    # -- Convenience shortcuts (single round trip) --------------------------

    async def execute_fetchall(self, sql: str, parameters: Any = ()) -> list[Any]:
        conn = self._ensure_conn()

        def _do() -> list[Any]:
            return conn.execute(sql, parameters).fetchall()

        return await self._execute(_do)

    async def execute_fetchone(self, sql: str, parameters: Any = ()) -> Any:
        conn = self._ensure_conn()

        def _do() -> Any:
            return conn.execute(sql, parameters).fetchone()

        return await self._execute(_do)

    async def execute_insert(self, sql: str, parameters: Any = ()) -> int | None:
        conn = self._ensure_conn()

        def _do() -> int | None:
            return conn.execute(sql, parameters).lastrowid

        return await self._execute(_do)

    # -- Connection properties (sync passthrough) ---------------------------

    @property
    def row_factory(self) -> Any:
        return self._conn.row_factory if self._conn else None

    @row_factory.setter
    def row_factory(self, value: Any) -> None:
        if self._conn is not None:
            self._conn.row_factory = value

    @property
    def text_factory(self) -> Any:
        return self._conn.text_factory if self._conn else None

    @text_factory.setter
    def text_factory(self, value: Any) -> None:
        if self._conn is not None:
            self._conn.text_factory = value

    @property
    def isolation_level(self) -> str | None:
        return self._conn.isolation_level if self._conn else None

    @isolation_level.setter
    def isolation_level(self, value: str | None) -> None:
        if self._conn is not None:
            self._conn.isolation_level = value  # ty: ignore[invalid-assignment]

    @property
    def in_transaction(self) -> bool:
        return self._conn.in_transaction if self._conn else False

    @property
    def total_changes(self) -> int:
        return self._conn.total_changes if self._conn else 0


# ---------------------------------------------------------------------------
# connect() factory
# ---------------------------------------------------------------------------


def connect(
    database: str | bytes,
    *,
    iter_chunk_size: int = _DEFAULT_ITER_CHUNK,
    **kwargs: Any,
) -> _ConnectionContextManager:
    """Open an async SQLite connection.

    Returns an object that is both awaitable and an async context manager::

        # As context manager (preferred)
        async with aiosqlite.connect("db.sqlite") as db:
            ...

        # As awaitable
        db = await aiosqlite.connect("db.sqlite")
        try:
            ...
        finally:
            await db.close()

    All ``kwargs`` are forwarded to ``sqlite3.connect()``.
    ``check_same_thread`` is always set to ``False`` (the worker thread
    handles thread safety).
    """
    kwargs["check_same_thread"] = False
    connector = functools.partial(sqlite3.connect, database, **kwargs)
    return _ConnectionContextManager(connector, iter_chunk_size)


class _ConnectionContextManager:
    """Awaitable + async context manager returned by :func:`connect`."""

    __slots__ = ("_conn",)

    def __init__(
        self,
        connector: Callable[[], sqlite3.Connection],
        iter_chunk_size: int,
    ) -> None:
        self._conn = Connection(connector, iter_chunk_size)

    def __await__(self):  # type: ignore[override]
        return self._conn._start().__await__()

    async def __aenter__(self) -> Connection:
        return await self._conn._start()

    async def __aexit__(self, *args: Any) -> None:
        await self._conn.close()
