"""Tests for the persisted request histogram behind the dashboard chart."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from llm_rosetta_gateway.observability import PersistenceManager


@pytest.fixture()
async def pm(tmp_path):
    manager = await PersistenceManager.create(str(tmp_path / "gateway.db"))
    yield manager
    await manager.close()


def _row(row_id: str, ts_iso: str, status: int = 200, out: int = 10, ms: float = 100.0):
    """One request_log row; only the columns the histogram reads matter."""
    return {
        "id": row_id,
        "timestamp": ts_iso,
        "model": "m",
        "source_provider": "s",
        "target_provider": "t",
        "is_stream": False,
        "status_code": status,
        "duration_ms": ms,
        "output_tokens": out,
        "total_tokens": out,
    }


def _ago(**kwargs) -> str:
    return (datetime.now(timezone.utc) - timedelta(**kwargs)).isoformat()


class TestRequestHistogram:
    async def test_rows_count_once_and_inside_the_window(self, pm):
        await pm.insert_log_entries(
            [
                _row("a", _ago(seconds=30)),
                _row("b", _ago(seconds=45), status=502, out=5, ms=50.0),
                _row("c", _ago(minutes=61)),
            ]
        )
        data = await pm.query_request_histogram(minutes=60, buckets=60)

        assert data["minutes"] == 60
        assert data["bucket_seconds"] == 60
        assert len(data["buckets"]) == 60
        assert data["total"] == 2, "the row older than the window must not count"
        assert sum(b["count"] for b in data["buckets"]) == 2
        assert sum(b["errors"] for b in data["buckets"]) == 1
        assert sum(b["output_tokens"] for b in data["buckets"]) == 15
        assert sum(b["duration_ms"] for b in data["buckets"]) == 150.0

    async def test_timestamp_lands_in_its_own_bucket(self, pm):
        ts = _ago(minutes=30)
        await pm.insert_log_entries([_row("a", ts)])
        data = await pm.query_request_histogram(minutes=60, buckets=60)

        # Same arithmetic as the query: whole seconds, integer division.
        start_epoch = int(datetime.fromisoformat(data["start"]).timestamp())
        ts_epoch = int(datetime.fromisoformat(ts).timestamp())
        width = data["bucket_seconds"]
        expected = (ts_epoch - start_epoch) // width
        assert data["buckets"][expected]["count"] == 1
        assert [b["count"] for b in data["buckets"]].count(1) == 1

    async def test_empty_window_is_all_zeros(self, pm):
        data = await pm.query_request_histogram(minutes=300, buckets=5)

        assert data["total"] == 0
        assert [b["count"] for b in data["buckets"]] == [0, 0, 0, 0, 0]
        assert all(b["errors"] == 0 for b in data["buckets"])

    async def test_buckets_are_clipped_to_their_limits(self, pm):
        data = await pm.query_request_histogram(minutes=999999, buckets=99999)

        assert data["minutes"] == 7 * 24 * 60
        assert len(data["buckets"]) == 240
        assert data["bucket_seconds"] >= 1
