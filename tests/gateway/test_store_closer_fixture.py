"""The gateway tests close the SQLite stores a test opened and left open.

``conftest._close_stores_left_open`` tracks what ``PersistenceManager.create``
and ``KeyStore.create`` return during an async test and closes it at teardown;
without it the suite held a ``gateway.db`` and a ``keys.db`` connection (and a
worker thread) per test and ran out of descriptors on a 1024 limit.

The first test leaves both stores open on purpose; the second runs after it, in
file order, and sees them closed.  It is skipped when run alone.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from llm_rosetta_gateway.keystore import KeyStore
from llm_rosetta_gateway.observability import PersistenceManager

_left_open: list[Any] = []


def _closed(store: Any) -> bool:
    """A store is closed once its connection is gone or marked closed."""
    conn = store._conn
    return conn is None or conn._closed


def _workers() -> int:
    return sum(t.name == "aiosqlite-worker" for t in threading.enumerate())


async def test_a_test_that_opens_two_stores_and_never_closes_them(tmp_path):
    _left_open.append(await PersistenceManager.create(str(tmp_path / "p")))
    _left_open.append(await KeyStore.create(tmp_path / "k" / "keys.db"))
    assert not any(_closed(store) for store in _left_open)


async def test_the_next_test_finds_them_closed():
    if not _left_open:
        pytest.skip("runs after the test that leaves the stores open")
    assert [_closed(store) for store in _left_open] == [True, True]
    assert _workers() == 0


def test_a_sync_test_gets_no_closer(request):
    assert "_close_stores_left_open" not in request.fixturenames


async def test_an_async_test_gets_the_closer(request):
    assert "_close_stores_left_open" in request.fixturenames
