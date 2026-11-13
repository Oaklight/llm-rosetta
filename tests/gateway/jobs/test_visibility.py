"""A cleared range is invisible to every read, and only to reads.

The visibility layer has two jobs.  One is to hide rows from the operator the
moment a clear is recorded; the other is to leave the physical layer alone, so
retention, trimming and the drift check still see what is on disk.  These tests
pin both, and the last one enumerates the read path so a new query that forgets
the predicate fails here rather than leaking rows in production.
"""

from __future__ import annotations

import io
import json
import tarfile
from datetime import datetime, timedelta, timezone
from typing import Any
from collections.abc import Callable

import pytest

from llm_rosetta_gateway.observability.persistence import PersistenceManager
from llm_rosetta_gateway.observability.visibility import (
    MAX_TS,
    MIN_TS,
    VisibilityFilter,
)


def _ago(**kwargs: Any) -> str:
    """An ISO timestamp *kwargs* before now, in the store's format."""
    return (datetime.now(timezone.utc) - timedelta(**kwargs)).isoformat()


def _after_tick(timestamp: str) -> str:
    """*timestamp* plus one second — an exclusive bound over that one row."""
    return (datetime.fromisoformat(timestamp) + timedelta(seconds=1)).isoformat()


@pytest.fixture
async def pm(tmp_path):
    manager = await PersistenceManager.create(str(tmp_path), success_max=1000)
    yield manager
    await manager.close()


async def _seed(pm: PersistenceManager) -> None:
    """Three requests, two ops entries and two dumps, all inside 24 hours."""
    await pm.insert_log_entries(
        [
            {
                "id": "r0",
                "timestamp": _ago(hours=1),
                "model": "m0",
                "source_provider": "openai",
                "target_provider": "messages",
                "is_stream": False,
                "status_code": 200,
                "duration_ms": 10.0,
                "api_key_label": "k0",
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "model_type": "llm",
            },
            {
                "id": "r1",
                "timestamp": _ago(hours=2),
                "model": "m0",
                "source_provider": "openai",
                "target_provider": "messages",
                "is_stream": False,
                "status_code": 200,
                "duration_ms": 11.0,
                "api_key_label": "k1",
                "input_tokens": 20,
                "output_tokens": 10,
                "total_tokens": 30,
                "model_type": "llm",
            },
            {
                "id": "r2",
                "timestamp": _ago(hours=30),
                "model": "m1",
                "source_provider": "openai",
                "target_provider": "messages",
                "is_stream": False,
                "status_code": 500,
                "duration_ms": 12.0,
                "api_key_label": "k2",
                "input_tokens": 30,
                "output_tokens": 15,
                "total_tokens": 45,
                "model_type": "llm",
            },
        ]
    )
    await pm.insert_ops_log_entries(
        [
            {
                "id": "o0",
                "timestamp": _ago(hours=1),
                "event_type": "data_cleared",
                "severity": "info",
                "message": "one",
                "source": "admin",
            },
            {
                "id": "o1",
                "timestamp": _ago(hours=2),
                "event_type": "data_cleared",
                "severity": "warning",
                "message": "two",
                "source": "admin",
            },
        ]
    )
    for index, hours in enumerate((1, 2)):
        await pm.insert_error_dump(
            dump_id=f"d{index}",
            request_log_id=None,
            timestamp=_ago(hours=hours),
            model="m0",
            source_provider="openai",
            target_provider="messages",
            provider_name="up",
            status_code=502,
            error_phase="upstream",
            body_hash=None,
            response_text="boom",
            upstream_url="http://example.test/v1",
        )


async def _drain(iterator) -> list[Any]:
    return [item async for item in iterator]


# ---------------------------------------------------------------------------
# The filter itself
# ---------------------------------------------------------------------------


class TestVisibilityFilter:
    def test_empty_filter_emits_no_sql(self):
        assert VisibilityFilter().clause("request_log") == ("", [])

    def test_range_becomes_a_not_predicate(self):
        flt = VisibilityFilter()
        flt.add("request_log", "2026-01-01", "2026-01-02")
        sql, params = flt.clause("request_log")
        assert sql == "NOT ((timestamp >= ? AND timestamp < ?))"
        assert params == ["2026-01-01", "2026-01-02"]

    def test_open_bounds_are_sentinels(self):
        flt = VisibilityFilter()
        flt.add("ops_log", "", "")
        sql, params = flt.clause("ops_log")
        assert params == [MIN_TS, MAX_TS]

    def test_several_ranges_are_ored(self):
        flt = VisibilityFilter()
        flt.add("request_log", "2026-01-01", "2026-01-02")
        flt.add("request_log", "2026-02-01", "2026-02-02")
        sql, _ = flt.clause("request_log")
        assert sql.count(" OR ") == 1
        assert sql.startswith("NOT (")

    def test_range_outside_the_query_window_is_skipped(self):
        flt = VisibilityFilter()
        flt.add("request_log", "2026-01-01", "2026-01-02")
        assert flt.clause("request_log", since="2026-03-01") == ("", [])
        assert flt.clause("request_log", until="2025-12-01") == ("", [])
        assert flt.clause("request_log", since="2025-12-31")[0] != ""

    def test_remove_forgets_one_range(self):
        flt = VisibilityFilter()
        flt.add("request_log", "2026-01-01", "2026-01-02")
        flt.add("request_log", "2026-02-01", "2026-02-02")
        flt.remove("request_log", "2026-01-01", "2026-01-02")
        assert flt.ranges("request_log") == [("2026-02-01", "2026-02-02")]

    def test_load_replaces_every_cached_range(self):
        flt = VisibilityFilter()
        flt.add("request_log", "2026-01-01", "2026-01-02")
        flt.load([("ops_log", "2026-03-01", "2026-03-02")])
        assert flt.ranges("request_log") == []
        assert flt.ranges("ops_log") == [("2026-03-01", "2026-03-02")]

    def test_alias_qualifies_the_column(self):
        flt = VisibilityFilter()
        flt.add("request_log", "2026-01-01", "2026-01-02")
        sql, _ = flt.clause("request_log", alias="r.")
        assert "r.timestamp >= ?" in sql


# ---------------------------------------------------------------------------
# The read path
# ---------------------------------------------------------------------------


async def test_clearing_hides_every_request_log_read(pm):
    await _seed(pm)
    await pm.add_hidden_range("request_log", MIN_TS, MAX_TS, job_id="j1")

    entries, total = await pm.query_log_entries(limit=10)
    assert entries == [] and total == 0
    assert await pm.get_log_entry("r0") is None
    assert await pm.count_log_entries() == 0
    assert await pm.count_success_entries() == 0
    assert await pm.count_error_entries() == 0
    assert await pm.log_activity_by_model(since=_ago(days=2)) == {}
    assert await pm.get_api_key_labels() == []
    assert await _drain(pm.iter_log_rows_for_rebuild()) == []
    assert (await pm.query_rolling_24h_tokens())["total_tokens_24h"] == 0
    assert (await pm.query_token_usage_by_day(days=7))["days"] == []
    histogram = await pm.query_request_histogram(minutes=180, buckets=6)
    assert histogram["total"] == 0
    groups = await pm.query_usage_groups()
    assert groups == {"windows": ["5h", "24h", "7d", "30d"], "groups": []}
    assert (await pm.query_usage_by_model())["groups"] == []

    # The physical layer still sees them, and says how many.
    assert await pm.count_log_entries(include_hidden=True) == 3
    assert await pm.count_success_entries(include_hidden=True) == 2
    assert await pm.count_hidden_entries("request_log") == 3


async def test_a_partial_range_hides_only_that_window(pm):
    await _seed(pm)
    # r1 is two hours old and is the only row in this window.
    await pm.add_hidden_range(
        "request_log", _ago(hours=3), _ago(minutes=90), job_id="j1"
    )

    entries, total = await pm.query_log_entries(limit=10)
    assert total == 2
    assert {entry["id"] for entry in entries} == {"r0", "r2"}
    assert await pm.get_log_entry("r1") is None
    assert (await pm.get_log_entry("r0"))["id"] == "r0"


async def test_ops_log_and_dumps_have_their_own_ranges(pm):
    await _seed(pm)
    await pm.add_hidden_range("ops_log", MIN_TS, MAX_TS, job_id="j-ops")

    entries, total = await pm.query_ops_log_entries(limit=10)
    assert entries == [] and total == 0
    assert await pm.count_ops_log_entries() == 0
    assert await pm.count_ops_info_entries() == 0
    assert await pm.count_ops_warn_entries() == 0
    assert await pm.count_ops_log_entries(include_hidden=True) == 2
    # Hiding the ops log must not touch the request log or the dumps.
    assert (await pm.query_log_entries(limit=10))[1] == 3
    assert (await pm.query_error_dumps(limit=10))[1] == 2

    await pm.add_hidden_range("error_dumps", MIN_TS, MAX_TS, job_id="j-dumps")
    dumps, dump_total = await pm.query_error_dumps(limit=10)
    assert dumps == [] and dump_total == 0
    assert await pm.get_error_dump("d0") is None
    assert await pm.count_error_dumps() == 0
    assert await pm.count_error_dumps(include_hidden=True) == 2
    with tarfile.open(
        fileobj=io.BytesIO(await pm.export_error_dumps()), mode="r:gz"
    ) as archive:
        metadata = archive.extractfile("metadata.json")
        assert metadata is not None
        assert json.loads(metadata.read()) == []


async def test_ranges_survive_a_reopen(tmp_path):
    first = await PersistenceManager.create(str(tmp_path), success_max=1000)
    await _seed(first)
    await first.add_hidden_range("request_log", MIN_TS, MAX_TS, job_id="j1")
    await first.close()

    second = await PersistenceManager.create(str(tmp_path), success_max=1000)
    try:
        assert second.visibility.ranges("request_log") == [(MIN_TS, MAX_TS)]
        assert (await second.query_log_entries(limit=10))[1] == 0
        assert await second.count_log_entries(include_hidden=True) == 3
    finally:
        await second.close()


async def test_removing_a_jobs_ranges_unhides_its_rows(pm):
    await _seed(pm)
    await pm.add_hidden_range("request_log", MIN_TS, MAX_TS, job_id="j1")
    await pm.add_hidden_range("ops_log", MIN_TS, MAX_TS, job_id="j2")

    assert await pm.remove_hidden_ranges_for_job("j1") == 1

    assert (await pm.query_log_entries(limit=10))[1] == 3
    assert (await pm.query_ops_log_entries(limit=10))[1] == 0
    assert await pm.count_hidden_entries("request_log") == 0


async def test_delete_batch_stops_after_the_limit(pm):
    await _seed(pm)
    assert await pm.delete_batch("request_log", "1 = 1", limit=2) == 2
    assert await pm.delete_batch("request_log", "1 = 1", limit=2) == 1
    assert await pm.delete_batch("request_log", "1 = 1", limit=2) == 0
    assert await pm.count_log_entries(include_hidden=True) == 0


async def test_transaction_rolls_back_every_statement(pm):
    with pytest.raises(RuntimeError):
        async with pm.transaction() as conn:
            await conn.execute(
                "INSERT INTO hidden_ranges "
                "(id, table_name, start_ts, end_ts, job_id, created_at) "
                "VALUES ('x', 'request_log', ?, ?, 'j', 'now')",
                (MIN_TS, MAX_TS),
            )
            raise RuntimeError("boom")

    row = await pm._conn.execute_fetchall("SELECT id FROM hidden_ranges")
    assert row == []


# ---------------------------------------------------------------------------
# The guard: no read site may forget the predicate
# ---------------------------------------------------------------------------


def _reads(pm: PersistenceManager) -> dict[str, Callable[[], Any]]:
    """Every read the UI can reach, paired with a call that runs it.

    Physical counts are deliberately absent: they are the allowlist, and the
    test below proves the recorder can tell the two apart.
    """
    return {
        "query_log_entries": lambda: pm.query_log_entries(limit=5),
        "get_log_entry": lambda: pm.get_log_entry("r0"),
        "iter_log_rows_for_rebuild": lambda: _drain(pm.iter_log_rows_for_rebuild()),
        "count_log_entries": pm.count_log_entries,
        "count_success_entries": pm.count_success_entries,
        "count_error_entries": pm.count_error_entries,
        "log_activity_by_model": lambda: pm.log_activity_by_model(since=_ago(days=2)),
        "get_api_key_labels": pm.get_api_key_labels,
        "query_ops_log_entries": lambda: pm.query_ops_log_entries(limit=5),
        "count_ops_log_entries": pm.count_ops_log_entries,
        "count_ops_info_entries": pm.count_ops_info_entries,
        "count_ops_warn_entries": pm.count_ops_warn_entries,
        "query_error_dumps": lambda: pm.query_error_dumps(limit=5),
        "get_error_dump": lambda: pm.get_error_dump("d0"),
        "count_error_dumps": pm.count_error_dumps,
        "export_error_dumps": pm.export_error_dumps,
        "query_token_usage_by_day": lambda: pm.query_token_usage_by_day(days=7),
        "query_rolling_24h_tokens": pm.query_rolling_24h_tokens,
        "query_request_histogram": lambda: pm.query_request_histogram(minutes=180),
        "query_usage_groups": pm.query_usage_groups,
        "query_usage_by_model": pm.query_usage_by_model,
    }


async def _capture(pm: PersistenceManager, call: Callable[[], Any]) -> list[str]:
    """Run *call*, returning every SQL statement it sent to the connection."""
    seen: list[str] = []
    conn = pm._conn
    names = ("execute", "execute_fetchall", "execute_fetchone")
    original = {name: getattr(conn, name) for name in names}

    def wrap(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        async def recording(sql: str, *args: Any, **kwargs: Any) -> Any:
            seen.append(sql)
            return await fn(sql, *args, **kwargs)

        return recording

    for name in names:
        setattr(conn, name, wrap(name, original[name]))
    try:
        await call()
    finally:
        for name, fn in original.items():
            setattr(conn, name, fn)
    return seen


async def test_every_read_applies_the_hidden_predicate(pm):
    await _seed(pm)
    await pm.add_hidden_range("request_log", MIN_TS, MAX_TS, job_id="j1")
    await pm.add_hidden_range("ops_log", MIN_TS, MAX_TS, job_id="j1")
    await pm.add_hidden_range("error_dumps", MIN_TS, MAX_TS, job_id="j1")

    misses: list[str] = []
    for name, call in _reads(pm).items():
        statements = await _capture(pm, call)
        assert statements, f"{name} issued no SQL, so the recorder proved nothing"
        if not all("NOT (" in sql for sql in statements):
            misses.append(name)
    assert misses == [], (
        "these reads can leak a cleared row; give each one "
        f"self._visible(...): {misses}"
    )

    # The recorder can tell a read from a physical count, or the guard above
    # would pass on anything.
    physical = await _capture(pm, lambda: pm.count_log_entries(include_hidden=True))
    assert physical and not any("NOT (" in sql for sql in physical)


async def test_the_dump_backfill_links_only_visible_rows(pm):
    """A repair step reads too: it must not link across a cleared range.

    A link to a cleared request produces a "view in request log" button that
    404s, and matching a dump that is itself queued for deletion is work on a
    row nobody will see.
    """
    t_vis, t_hid, t_dump = _ago(minutes=3), _ago(minutes=2), _ago(minutes=1)

    def row(row_id: str, timestamp: str) -> dict[str, Any]:
        return {
            "id": row_id,
            "timestamp": timestamp,
            "model": "m0",
            "source_provider": "openai",
            "target_provider": "messages",
            "is_stream": False,
            "status_code": 502,
            "duration_ms": 10.0,
            "api_key_label": "k0",
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "model_type": "llm",
        }

    await pm.insert_log_entries(
        [
            row("r-visible", t_vis),
            row("r-hidden", t_hid),
            row("r-dump-target", t_dump),
        ]
    )
    for dump_id, timestamp in (
        ("d-control", t_vis),
        ("d-hidden-row", t_hid),
        ("d-hidden-dump", t_dump),
    ):
        await pm.insert_error_dump(
            dump_id=dump_id,
            request_log_id=None,
            timestamp=timestamp,
            model="m0",
            source_provider="openai",
            target_provider="messages",
            provider_name="up",
            status_code=502,
            error_phase="upstream",
            body_hash=None,
            response_text="boom",
            upstream_url="http://example.test/v1",
        )

    # One range per table, each covering exactly one row.
    await pm.add_hidden_range("request_log", t_hid, _after_tick(t_hid), job_id="j1")
    await pm.add_hidden_range("error_dumps", t_dump, _after_tick(t_dump), job_id="j1")

    assert await pm.backfill_error_dump_log_ids() == 1
    links = dict(
        await pm.connection.execute_fetchall(
            "SELECT id, request_log_id FROM error_dumps"
        )
    )
    assert links == {
        "d-control": "r-visible",
        "d-hidden-row": None,
        "d-hidden-dump": None,
    }


async def test_a_read_outside_the_hidden_window_pays_nothing(pm):
    await _seed(pm)
    # A range that ended before the window this read asks for.
    await pm.add_hidden_range("request_log", _ago(days=40), _ago(days=30), job_id="j1")
    statements = await _capture(pm, lambda: pm.query_rolling_24h_tokens())
    assert statements and not any("NOT (" in sql for sql in statements)
