# /// zerodep
# version = "0.4.1"
# deps = []
# tier = "medium"
# category = "data"
# note = "Install/update via `zerodep add persistdict`"
# ///
"""Persistent dictionary with pluggable backends.

Zero dependencies, stdlib only, Python 3.10+.

Part of zerodep: https://github.com/Oaklight/zerodep
Copyright (c) 2026 Peng Ding. MIT License.

A MutableMapping that persists key-value pairs to disk.  Supports JSON file
and SQLite backends with pluggable serialization.  Thread-safe by default.

Example::

    from persistdict import open

    # Auto-detect backend from file extension
    with open("data.json") as d:
        d["name"] = "Alice"
        d["scores"] = [95, 87, 92]

    # Reopen -- data persists
    with open("data.json") as d:
        print(d["name"])      # "Alice"
        print(len(d))         # 2

    # SQLite backend for larger datasets
    with open("data.db") as d:
        for i in range(10000):
            d[f"key_{i}"] = {"index": i}

    # Explicit backend and custom table
    with open("app.db", backend="sqlite", table="users") as users:
        users["alice"] = {"email": "alice@example.com"}

    # Direct class usage with custom serializer
    from persistdict import PersistDict, SqliteBackend

    backend = SqliteBackend("mydata.db", table="config")
    d = PersistDict(backend)
    d["debug"] = True
    d.close()

Requirements:
    Python >= 3.10, no third-party packages.
"""

from __future__ import annotations

import collections.abc
import json
import os
import re
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

__all__ = [
    # Main class
    "PersistDict",
    # Backends
    "Backend",
    "JsonFileBackend",
    "SqliteBackend",
    # Serialization
    "Serializer",
    "JsonSerializer",
    # Factory
    "open",
]


# ── Serializer ────────────────────────────────────────────────────────────


@runtime_checkable
class Serializer(Protocol):
    """Protocol for value serialization."""

    def dumps(self, obj: Any) -> str: ...
    def loads(self, s: str) -> Any: ...


class JsonSerializer:
    """JSON serializer (default).

    Args:
        ensure_ascii: Passed to ``json.dumps``.  Defaults to ``False`` so
            non-ASCII data is preserved without escaping.
        **kwargs: Extra keyword arguments forwarded to ``json.dumps``.
    """

    def __init__(self, *, ensure_ascii: bool = False, **kwargs: Any) -> None:
        self._dump_kw: dict[str, Any] = {"ensure_ascii": ensure_ascii, **kwargs}

    def dumps(self, obj: Any) -> str:
        """Serialize *obj* to a JSON string."""
        return json.dumps(obj, **self._dump_kw)

    def loads(self, s: str) -> Any:
        """Deserialize a JSON string back to a Python object."""
        return json.loads(s)


# ── Backend Protocol ──────────────────────────────────────────────────────


@runtime_checkable
class Backend(Protocol):
    """Protocol that storage backends must satisfy."""

    def get(self, key: str) -> str: ...
    def set(self, key: str, value: str) -> None: ...
    def delete(self, key: str) -> None: ...
    def contains(self, key: str) -> bool: ...
    def keys(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def clear(self) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


# ── JSON File Backend ─────────────────────────────────────────────────────

_EMPTY_MARKERS = {b"", b"{}"}


class JsonFileBackend:
    """Fully-buffered JSON file backend.

    The entire file is loaded into memory on open.  Mutations happen
    in-memory and are written atomically (temp file + ``os.replace``) on
    :meth:`flush` or :meth:`close`.

    Args:
        path: Path to the JSON file.  Created on first flush if missing.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._closed = False
        self._data: dict[str, str] = {}
        if self._path.exists():
            raw = self._path.read_bytes().strip()
            if raw and raw not in _EMPTY_MARKERS:
                try:
                    loaded = json.loads(raw)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"corrupt JSON file: {self._path}") from exc
                if not isinstance(loaded, dict):
                    raise ValueError(
                        f"expected JSON object, got {type(loaded).__name__}: "
                        f"{self._path}"
                    )
                self._data = loaded

    # -- Backend interface -------------------------------------------------

    def get(self, key: str) -> str:
        return self._data[key]

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def delete(self, key: str) -> None:
        del self._data[key]

    def contains(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> Iterator[str]:
        return iter(list(self._data))

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()

    def flush(self) -> None:
        """Write the current state to disk atomically."""
        if self._closed:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(self._path.parent),
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
            os.replace(tmp, str(self._path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def close(self) -> None:
        """Flush and mark as closed."""
        if self._closed:
            return
        self.flush()
        self._closed = True

    def __repr__(self) -> str:
        return f"JsonFileBackend({str(self._path)!r})"


# ── SQLite Backend ────────────────────────────────────────────────────────

_TABLE_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class SqliteBackend:
    """Buffered SQLite backend with deferred commits.

    Writes are accumulated in an open transaction and committed either
    periodically (every *commit_every* write operations) or explicitly
    via :meth:`flush` / :meth:`close`.  Uses WAL journal mode with
    ``synchronous=NORMAL`` for a good balance of performance and crash
    safety.

    Reads always see uncommitted writes within the same connection
    (read-your-own-writes), so the buffering is transparent to callers.

    Args:
        path: Path to the SQLite database file.
        table: Table name for storage (default ``"items"``).  Must be a
            valid SQL identifier (letters, digits, underscores).
        commit_every: Number of write operations before an automatic
            commit.  ``0`` disables periodic commits — only
            :meth:`flush` and :meth:`close` will commit.
            Defaults to ``0`` (commit only on flush/close).
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        table: str = "items",
        commit_every: int = 0,
    ) -> None:
        if not _TABLE_NAME_RE.fullmatch(table):
            raise ValueError(
                f"invalid table name {table!r}: must match [A-Za-z_][A-Za-z0-9_]*"
            )
        self._path = Path(path)
        self._table = table
        self._commit_every = commit_every
        self._pending = 0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()

    # -- Internal helpers --------------------------------------------------

    def _auto_commit(self) -> None:
        """Commit based on the *commit_every* policy.

        When *commit_every* is ``0`` (the default), every mutation is
        committed immediately (write-through).  Otherwise commits are
        deferred until *commit_every* writes have accumulated.
        """
        if not self._commit_every:
            # Write-through: commit every mutation.
            self._conn.commit()
            return
        self._pending += 1
        if self._pending >= self._commit_every:
            self._conn.commit()
            self._pending = 0

    # -- Backend interface -------------------------------------------------

    def get(self, key: str) -> str:
        row = self._conn.execute(
            f"SELECT value FROM {self._table} WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return row[0]

    def set(self, key: str, value: str) -> None:
        self._conn.execute(
            f"INSERT OR REPLACE INTO {self._table} (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._auto_commit()

    def delete(self, key: str) -> None:
        cur = self._conn.execute(f"DELETE FROM {self._table} WHERE key = ?", (key,))
        if cur.rowcount == 0:
            raise KeyError(key)
        self._auto_commit()

    def contains(self, key: str) -> bool:
        row = self._conn.execute(
            f"SELECT 1 FROM {self._table} WHERE key = ? LIMIT 1", (key,)
        ).fetchone()
        return row is not None

    def keys(self) -> Iterator[str]:
        rows = self._conn.execute(f"SELECT key FROM {self._table}").fetchall()
        return iter([r[0] for r in rows])

    def __len__(self) -> int:
        row = self._conn.execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()
        assert row is not None
        return row[0]

    def clear(self) -> None:
        self._conn.execute(f"DELETE FROM {self._table}")
        self._conn.commit()
        self._pending = 0

    def flush(self) -> None:
        """Commit any pending writes to disk."""
        self._conn.commit()
        self._pending = 0

    def close(self) -> None:
        """Flush pending writes and close the database connection."""
        try:
            self._conn.commit()
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"SqliteBackend({str(self._path)!r}, table={self._table!r})"


# ── PersistDict ───────────────────────────────────────────────────────────


class PersistDict(collections.abc.MutableMapping):
    """Persistent dictionary backed by a pluggable storage backend.

    Implements :class:`collections.abc.MutableMapping` so it can be used
    as a drop-in replacement for ``dict`` wherever persistence is needed.

    Args:
        backend: Storage backend (e.g. :class:`JsonFileBackend` or
            :class:`SqliteBackend`).
        serializer: Value serializer.  Defaults to :class:`JsonSerializer`.
        lock: Thread-safety control.  ``True`` (default) creates a new
            ``threading.Lock``; ``False`` disables locking; a
            ``threading.Lock`` instance is used as-is.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        serializer: Serializer | None = None,
        lock: threading.Lock | bool = True,
    ) -> None:
        self._backend = backend
        self._serializer: Serializer = serializer or JsonSerializer()
        if lock is True:
            self._lock: threading.Lock | None = threading.Lock()
        elif lock is False:
            self._lock = None
        else:
            self._lock = lock
        self._closed = False

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _check_key(key: Any) -> str:
        if not isinstance(key, str):
            raise TypeError(f"keys must be str, got {type(key).__name__}")
        return key

    # -- MutableMapping interface ------------------------------------------

    def __getitem__(self, key: Any) -> Any:
        k = self._check_key(key)
        if self._lock:
            with self._lock:
                raw = self._backend.get(k)
        else:
            raw = self._backend.get(k)
        return self._serializer.loads(raw)

    def __setitem__(self, key: Any, value: Any) -> None:
        k = self._check_key(key)
        raw = self._serializer.dumps(value)
        if self._lock:
            with self._lock:
                self._backend.set(k, raw)
        else:
            self._backend.set(k, raw)

    def __delitem__(self, key: Any) -> None:
        k = self._check_key(key)
        if self._lock:
            with self._lock:
                self._backend.delete(k)
        else:
            self._backend.delete(k)

    def __iter__(self) -> Iterator[str]:
        if self._lock:
            with self._lock:
                return self._backend.keys()
        return self._backend.keys()

    def __len__(self) -> int:
        if self._lock:
            with self._lock:
                return len(self._backend)
        return len(self._backend)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if self._lock:
            with self._lock:
                return self._backend.contains(key)
        return self._backend.contains(key)

    # -- Extra public API --------------------------------------------------

    def flush(self) -> None:
        """Flush pending writes to the underlying storage."""
        if self._lock:
            with self._lock:
                self._backend.flush()
        else:
            self._backend.flush()

    def close(self) -> None:
        """Flush and close the backend."""
        if self._closed:
            return
        self._closed = True
        if self._lock:
            with self._lock:
                self._backend.close()
        else:
            self._backend.close()

    # -- Context manager ---------------------------------------------------

    def __enter__(self) -> PersistDict:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # -- Representation ----------------------------------------------------

    def __repr__(self) -> str:
        return f"PersistDict({self._backend!r})"

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ── Factory ───────────────────────────────────────────────────────────────

_EXT_TO_BACKEND: dict[str, str] = {
    ".json": "json",
    ".db": "sqlite",
    ".sqlite": "sqlite",
    ".sqlite3": "sqlite",
}


def open(
    path: str | os.PathLike[str],
    *,
    backend: str = "auto",
    serializer: Serializer | None = None,
    lock: threading.Lock | bool = True,
    table: str = "items",
    commit_every: int = 0,
) -> PersistDict:
    """Open a persistent dictionary.

    Args:
        path: File path for storage.
        backend: ``"auto"`` (detect from extension), ``"json"``, or
            ``"sqlite"``.
        serializer: Value serializer.  Defaults to :class:`JsonSerializer`.
        lock: Thread-safety control (see :class:`PersistDict`).
        table: Table name for SQLite backend (ignored for JSON).
        commit_every: (SQLite only) Number of writes between automatic
            commits.  ``0`` (default) commits every write.  Set to a
            positive integer to batch writes and commit periodically —
            :meth:`~PersistDict.flush` and :meth:`~PersistDict.close`
            always commit remaining writes.  Ignored for JSON backend.

    Returns:
        A :class:`PersistDict` instance backed by the chosen storage.

    Raises:
        ValueError: If *backend* is ``"auto"`` and the file extension is
            not recognised, or if *backend* is not a known name.
    """
    p = Path(path)
    kind = backend
    if kind == "auto":
        ext = p.suffix.lower()
        kind = _EXT_TO_BACKEND.get(ext, "")
        if not kind:
            raise ValueError(
                f"cannot auto-detect backend for extension {ext!r}; "
                "use backend='json' or backend='sqlite'"
            )

    if kind == "json":
        be: Backend = JsonFileBackend(p)
    elif kind == "sqlite":
        be = SqliteBackend(p, table=table, commit_every=commit_every)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    return PersistDict(be, serializer=serializer, lock=lock)
