"""Tests for the time-to-first-token column on the request log."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from llm_rosetta_gateway.observability import (
    PersistenceManager,
    RequestLog,
    RequestLogEntry,
)


@pytest.fixture()
async def pm(tmp_path):
    manager = await PersistenceManager.create(str(tmp_path / "gateway.db"))
    yield manager
    await manager.close()


def _row(row_id: str, **extra):
    """One request_log row; ttft is the field under test, the rest is filler."""
    row = {
        "id": row_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "m",
        "source_provider": "s",
        "target_provider": "t",
        "is_stream": True,
        "status_code": 200,
        "duration_ms": 900.0,
        "output_tokens": 40,
    }
    row.update(extra)
    return row


class TestTtftColumn:
    async def test_a_measured_ttft_survives_the_round_trip(self, pm):
        await pm.insert_log_entries([_row("a", ttft_ms=120.5)])

        entry = await pm.get_log_entry("a")

        assert entry is not None
        assert entry["ttft_ms"] == 120.5

    async def test_a_row_written_before_the_column_has_no_ttft(self, pm):
        # No ttft key at all, the shape of a row from an older writer.
        await pm.insert_log_entries([_row("b")])

        entry = await pm.get_log_entry("b")

        assert entry is not None
        assert "ttft_ms" not in entry, "NULL must read as absent, not as zero"

    async def test_the_stream_path_fills_it_after_the_row_exists(self, pm):
        logger = RequestLog(persistence=pm)
        await logger.add(
            RequestLogEntry.create(
                model="m",
                source_provider="s",
                target_provider="t",
                is_stream=True,
                status_code=200,
                duration_ms=900.0,
                output_tokens=40,
            )
        )
        entry_id = (await pm.query_log_entries(limit=1))[0][0]["id"]
        assert "ttft_ms" not in (await pm.get_log_entry(entry_id))

        await logger.update_ttft(entry_id, 88.25)

        entry = await pm.get_log_entry(entry_id)
        assert entry is not None
        assert entry["ttft_ms"] == 88.25

    async def test_no_measurement_leaves_the_column_alone(self, pm):
        await pm.insert_log_entries([_row("c", ttft_ms=50.0)])
        logger = RequestLog(persistence=pm)

        await logger.update_ttft("c", None)

        entry = await pm.get_log_entry("c")
        assert entry is not None
        assert entry["ttft_ms"] == 50.0
