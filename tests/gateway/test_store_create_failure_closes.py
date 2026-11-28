"""A store whose open fails closes the connection it had opened.

``PersistenceManager.create`` and ``KeyStore.create`` connect first and then
run the schema steps; a MANDATORY migration that raises aborts the open, and
the connection (with its worker thread and its file descriptors) was left for
the garbage collector.  The open still raises the store's own error; the
half-opened store is rolled back, not committed.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from llm_rosetta_gateway import keystore as keystore_module
from llm_rosetta_gateway.keystore import KeyStore
from llm_rosetta_gateway.observability import PersistenceManager

_BROKEN = (
    "a migration that cannot run",
    ["INSERT INTO no_such_table (x) VALUES (1)"],
)


def _workers() -> int:
    return sum(t.name == "aiosqlite-worker" for t in threading.enumerate())


async def test_a_persistence_open_that_fails_leaves_no_connection(
    tmp_path, monkeypatch
):
    first = await PersistenceManager.create(str(tmp_path))
    await first.close()
    version = max(v for v, _d, _s in PersistenceManager._MIGRATIONS)
    monkeypatch.setattr(
        PersistenceManager,
        "_MIGRATIONS",
        [*PersistenceManager._MIGRATIONS, (version + 1, *_BROKEN)],
    )
    before = _workers()
    with pytest.raises(sqlite3.OperationalError, match="no_such_table"):
        await PersistenceManager.create(str(tmp_path))
    assert _workers() == before


async def test_a_keystore_open_that_fails_leaves_no_connection(tmp_path, monkeypatch):
    db = tmp_path / "keys.db"
    first = await KeyStore.create(db)
    await first.close()
    version = max(v for v, _d, _s in keystore_module.VERSIONS)
    monkeypatch.setattr(
        keystore_module,
        "VERSIONS",
        [*keystore_module.VERSIONS, (version + 1, *_BROKEN)],
    )
    before = _workers()
    with pytest.raises(sqlite3.OperationalError, match="no_such_table"):
        await KeyStore.create(db)
    assert _workers() == before
