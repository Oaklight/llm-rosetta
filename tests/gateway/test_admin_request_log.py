"""Tests for the admin panel RequestLog."""

from dataclasses import replace

import pytest

from llm_rosetta_gateway.observability.request_log import RequestLog, RequestLogEntry
from llm_rosetta_gateway.admin.routes.observability import (
    _bounded_int,
    _normalise_time_bound,
)


class TestRequestLogEntry:
    @pytest.mark.asyncio
    async def test_create(self):
        e = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=123.456,
        )
        assert e.model == "gpt-4o"
        assert e.duration_ms == 123.46  # rounded
        assert e.id  # non-empty uuid
        assert e.timestamp  # non-empty ISO timestamp
        assert e.error_detail is None

    @pytest.mark.asyncio
    async def test_to_dict(self):
        e = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=True,
            status_code=500,
            duration_ms=50.0,
            error_detail="connection refused",
        )
        d = e.to_dict()
        assert d["model"] == "gpt-4o"
        assert d["is_stream"] is True
        assert d["status_code"] == 500
        assert d["error_detail"] == "connection refused"


class TestRequestLog:
    def _make_entry(
        self,
        model="gpt-4o",
        status=200,
        provider="openai_chat",
        api_key_label=None,
    ):
        return RequestLogEntry.create(
            model=model,
            source_provider="openai_chat",
            target_provider=provider,
            is_stream=False,
            status_code=status,
            duration_ms=10.0,
            api_key_label=api_key_label,
        )

    @pytest.mark.asyncio
    async def test_add_and_get(self):
        log = RequestLog(max_entries=100)
        await log.add(self._make_entry())
        entries, total = await log.get_entries()
        assert total == 1
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_max_entries_eviction(self):
        log = RequestLog(max_entries=3)
        for i in range(5):
            await log.add(self._make_entry(model=f"model-{i}"))
        assert len(log._entries) == 3
        entries, total = await log.get_entries(limit=10)
        assert total == 3
        # Should have models 2, 3, 4 (oldest evicted)
        models = [e["model"] for e in entries]
        assert "model-0" not in models
        assert "model-1" not in models
        assert "model-4" in models

    @pytest.mark.asyncio
    async def test_filter_by_model(self):
        log = RequestLog()
        await log.add(self._make_entry(model="gpt-4o"))
        await log.add(self._make_entry(model="claude"))
        await log.add(self._make_entry(model="gpt-4o"))
        entries, total = await log.get_entries(model="gpt-4o")
        assert total == 2
        assert all(e["model"] == "gpt-4o" for e in entries)

    @pytest.mark.asyncio
    async def test_filter_by_provider(self):
        log = RequestLog()
        await log.add(self._make_entry(provider="openai_chat"))
        await log.add(self._make_entry(provider="anthropic"))
        entries, total = await log.get_entries(provider="anthropic")
        assert total == 1
        assert entries[0]["target_provider"] == "anthropic"

    @pytest.mark.asyncio
    async def test_filter_by_status(self):
        log = RequestLog()
        await log.add(self._make_entry(status=200))
        await log.add(self._make_entry(status=500))
        await log.add(self._make_entry(status=404))

        ok_entries, ok_total = await log.get_entries(status="ok")
        assert ok_total == 1

        err_entries, err_total = await log.get_entries(status="error")
        assert err_total == 2

    @pytest.mark.asyncio
    async def test_pagination(self):
        log = RequestLog()
        for i in range(10):
            await log.add(self._make_entry(model=f"m-{i}"))

        entries, total = await log.get_entries(limit=3, offset=0)
        assert total == 10
        assert len(entries) == 3

        entries2, _ = await log.get_entries(limit=3, offset=3)
        assert len(entries2) == 3
        # Different entries
        assert entries[0]["id"] != entries2[0]["id"]

    @pytest.mark.asyncio
    async def test_newest_first(self):
        log = RequestLog()
        await log.add(self._make_entry(model="first"))
        await log.add(self._make_entry(model="second"))
        entries, _ = await log.get_entries()
        assert entries[0]["model"] == "second"
        assert entries[1]["model"] == "first"

    @pytest.mark.asyncio
    async def test_get_entry_by_id(self):
        log = RequestLog()
        e = self._make_entry()
        await log.add(e)
        found = await log.get_entry(e.id)
        assert found is not None
        assert found["id"] == e.id

    @pytest.mark.asyncio
    async def test_get_entry_not_found(self):
        log = RequestLog()
        assert await log.get_entry("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_api_key_labels(self):
        log = RequestLog()
        await log.add(self._make_entry(api_key_label="bob"))
        await log.add(self._make_entry(api_key_label="alice"))
        await log.add(self._make_entry(api_key_label="bob"))
        await log.add(self._make_entry())
        assert await log.get_api_key_labels() == ["alice", "bob"]

    @pytest.mark.asyncio
    async def test_filter_by_time_range(self):
        log = RequestLog()
        entries = [
            replace(
                self._make_entry(model="old"), timestamp="2026-10-01T00:00:00+00:00"
            ),
            replace(
                self._make_entry(model="mid"), timestamp="2026-10-10T00:00:00+00:00"
            ),
            replace(
                self._make_entry(model="new"), timestamp="2026-10-19T00:00:00+00:00"
            ),
        ]
        for entry in entries:
            await log.add(entry)

        _, total = await log.get_entries(since="2026-10-05T00:00:00+00:00")
        assert total == 2

        results, total = await log.get_entries(
            since="2026-10-05T00:00:00+00:00",
            until="2026-10-15T00:00:00+00:00",
        )
        assert total == 1
        assert results[0]["model"] == "mid"

        # The upper bound is exclusive, so an entry exactly on it is excluded.
        results, total = await log.get_entries(until="2026-10-10T00:00:00+00:00")
        assert total == 1
        assert results[0]["model"] == "old"

    @pytest.mark.asyncio
    async def test_clear(self):
        log = RequestLog()
        await log.add(self._make_entry())
        await log.add(self._make_entry())
        assert len(log._entries) == 2
        await log.clear()
        assert len(log._entries) == 0


class TestNormaliseTimeBound:
    def test_z_suffix(self):
        assert (
            _normalise_time_bound("2026-10-19T07:00:00Z") == "2026-10-19T07:00:00+00:00"
        )

    def test_offset_converted_to_utc(self):
        assert (
            _normalise_time_bound("2026-10-19T15:00:00+08:00")
            == "2026-10-19T07:00:00+00:00"
        )

    def test_naive_rejected(self):
        with pytest.raises(ValueError, match="has no UTC offset"):
            _normalise_time_bound("2026-10-19T07:00:00")

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid ISO 8601 timestamp"):
            _normalise_time_bound("yesterday")


class _StubRequest:
    """Minimal request stand-in for the query-param helpers."""

    def __init__(self, **params: str):
        self.query_params = {key: [value] for key, value in params.items()}


class TestBoundedInt:
    def test_default_when_absent(self):
        assert _bounded_int(_StubRequest(), "limit", 50, minimum=1, maximum=500) == 50

    def test_negative_is_clamped_up(self):
        # SQLite reads LIMIT -1 as "no limit", so this must not pass through.
        assert (
            _bounded_int(_StubRequest(limit="-1"), "limit", 50, minimum=1, maximum=500)
            == 1
        )

    def test_zero_is_clamped_up(self):
        assert (
            _bounded_int(_StubRequest(limit="0"), "limit", 50, minimum=1, maximum=500)
            == 1
        )

    def test_huge_is_clamped_down(self):
        assert (
            _bounded_int(
                _StubRequest(limit="1000000"), "limit", 50, minimum=1, maximum=500
            )
            == 500
        )

    def test_non_integer_raises(self):
        with pytest.raises(ValueError, match="limit must be an integer"):
            _bounded_int(_StubRequest(limit="abc"), "limit", 50, minimum=1, maximum=500)
