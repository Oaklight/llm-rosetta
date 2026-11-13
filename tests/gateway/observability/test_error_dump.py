"""Tests for the error dump module."""

from __future__ import annotations

import json
import zlib

import pytest

from llm_rosetta_gateway.observability.error_dump import (
    compress_body,
    compute_body_hash,
    decompress_body,
    dump_error,
    offload_images,
)
from llm_rosetta_gateway.observability.persistence import PersistenceManager


# ------------------------------------------------------------------
# offload_images
# ------------------------------------------------------------------


class TestOffloadImages:
    """Tests for base64 image offloading."""

    @pytest.mark.asyncio
    async def test_no_images(self) -> None:
        body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        result = offload_images(body)
        assert result == body

    @pytest.mark.asyncio
    async def test_replaces_inline_base64(self) -> None:
        # 200 chars of base64 data (above the 100-char threshold)
        fake_b64 = "A" * 200
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{fake_b64}"},
                        }
                    ],
                }
            ]
        }
        result = offload_images(body)
        url = result["messages"][0]["content"][0]["image_url"]["url"]
        assert url.startswith("[image image/png sha256:")
        assert "data:image" not in url

    @pytest.mark.asyncio
    async def test_preserves_short_strings(self) -> None:
        """Strings shorter than 100 chars should not be replaced."""
        body = {"data": "data:image/png;base64,abc"}
        result = offload_images(body)
        assert result == body

    @pytest.mark.asyncio
    async def test_does_not_mutate_original(self) -> None:
        fake_b64 = "B" * 200
        body = {"url": f"data:image/jpeg;base64,{fake_b64}"}
        original_url = body["url"]
        offload_images(body)
        assert body["url"] == original_url

    @pytest.mark.asyncio
    async def test_multiple_images(self) -> None:
        fake_b64 = "C" * 200
        body = {
            "messages": [
                {"url": f"data:image/png;base64,{fake_b64}"},
                {"url": f"data:image/jpeg;base64,{fake_b64}"},
            ]
        }
        result = offload_images(body)
        for msg in result["messages"]:
            assert "data:image" not in msg["url"]
            assert msg["url"].startswith("[image image/")


# ------------------------------------------------------------------
# compute_body_hash
# ------------------------------------------------------------------


class TestComputeBodyHash:
    """Tests for body hashing."""

    @pytest.mark.asyncio
    async def test_deterministic(self) -> None:
        body = {"model": "gpt-4", "messages": []}
        h1 = compute_body_hash(body)
        h2 = compute_body_hash(body)
        assert h1 == h2

    @pytest.mark.asyncio
    async def test_key_order_independent(self) -> None:
        body1 = {"a": 1, "b": 2}
        body2 = {"b": 2, "a": 1}
        assert compute_body_hash(body1) == compute_body_hash(body2)

    @pytest.mark.asyncio
    async def test_different_bodies_different_hashes(self) -> None:
        h1 = compute_body_hash({"a": 1})
        h2 = compute_body_hash({"a": 2})
        assert h1 != h2

    @pytest.mark.asyncio
    async def test_returns_hex_string(self) -> None:
        h = compute_body_hash({"test": True})
        assert len(h) == 64  # SHA256 hex digest
        assert all(c in "0123456789abcdef" for c in h)


# ------------------------------------------------------------------
# compress / decompress
# ------------------------------------------------------------------


class TestCompressDecompress:
    """Tests for body compression and decompression."""

    @pytest.mark.asyncio
    async def test_roundtrip(self) -> None:
        body = {"model": "claude-3", "messages": [{"role": "user", "content": "hello"}]}
        compressed, orig_size = compress_body(body)
        assert isinstance(compressed, bytes)
        assert orig_size > 0
        result = decompress_body(compressed)
        assert result == body

    @pytest.mark.asyncio
    async def test_compression_reduces_size(self) -> None:
        # Repetitive data should compress well
        body = {"data": "x" * 10000}
        compressed, orig_size = compress_body(body)
        assert len(compressed) < orig_size

    @pytest.mark.asyncio
    async def test_zlib_format(self) -> None:
        body = {"test": 1}
        compressed, _ = compress_body(body)
        # Verify it's valid zlib data
        raw = zlib.decompress(compressed)
        assert json.loads(raw) == body


# ------------------------------------------------------------------
# dump_error (integration with PersistenceManager)
# ------------------------------------------------------------------


class TestDumpError:
    """Integration tests for the dump_error function."""

    @pytest.fixture
    async def persistence(self, tmp_path: object) -> PersistenceManager:
        return await PersistenceManager.create(str(tmp_path))

    @pytest.mark.asyncio
    async def test_stores_error_dump(self, persistence: PersistenceManager) -> None:
        body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        dump_id = await dump_error(
            persistence,
            request_body=body,
            response_text='{"error": "rate limit"}',
            model="gpt-4",
            source_provider="openai_chat",
            target_provider="anthropic",
            provider_name="Anthropic",
            status_code=429,
            error_phase="upstream",
            upstream_url="https://api.anthropic.com/v1/messages",
        )
        assert dump_id is not None

        # Verify stored
        entry = await persistence.get_error_dump(dump_id)
        assert entry is not None
        assert entry["model"] == "gpt-4"
        assert entry["status_code"] == 429
        assert entry["error_phase"] == "upstream"
        assert entry["response_text"] == '{"error": "rate limit"}'

        # Verify body was stored and decompressed correctly
        body_hash = entry["body_hash"]
        assert isinstance(body_hash, str)
        body_data = await persistence.get_dump_body(body_hash)
        assert body_data is not None
        restored = decompress_body(body_data)
        assert restored == body

    @pytest.mark.asyncio
    async def test_deduplicates_bodies(self, persistence: PersistenceManager) -> None:
        body = {"model": "gpt-4", "messages": []}
        await dump_error(
            persistence,
            request_body=body,
            status_code=500,
            error_phase="upstream",
        )
        await dump_error(
            persistence,
            request_body=body,
            status_code=502,
            error_phase="upstream",
        )
        # Two dumps but only one body entry
        assert await persistence.count_error_dumps() == 2
        row = await persistence._conn.execute_fetchone(
            "SELECT COUNT(*) FROM dump_bodies"
        )
        assert row[0] == 1

    @pytest.mark.asyncio
    async def test_image_offload_in_dump(self, persistence: PersistenceManager) -> None:
        fake_b64 = "D" * 200
        body = {"messages": [{"content": f"data:image/png;base64,{fake_b64}"}]}
        dump_id = await dump_error(
            persistence,
            request_body=body,
            status_code=500,
            error_phase="upstream",
        )
        assert dump_id is not None

        entry = await persistence.get_error_dump(dump_id)
        assert entry is not None
        body_hash = entry["body_hash"]
        assert isinstance(body_hash, str)
        body_data = await persistence.get_dump_body(body_hash)
        assert body_data is not None
        restored = decompress_body(body_data)
        # The base64 should have been replaced
        content = restored["messages"][0]["content"]
        assert "data:image" not in content
        assert content.startswith("[image image/png sha256:")

    @pytest.mark.asyncio
    async def test_none_persistence_is_noop(self) -> None:
        result = await dump_error(
            None,
            request_body={"test": True},
            status_code=500,
            error_phase="upstream",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_none_body_ok(self, persistence: PersistenceManager) -> None:
        dump_id = await dump_error(
            persistence,
            request_body=None,
            response_text="upstream timeout",
            status_code=504,
            error_phase="upstream",
        )
        assert dump_id is not None
        entry = await persistence.get_error_dump(dump_id)
        assert entry is not None
        assert "body_hash" not in entry

    @pytest.mark.asyncio
    async def test_converted_body_stored(self, persistence: PersistenceManager) -> None:
        req_body = {"model": "gpt-4", "messages": []}
        conv_body = {"model": "claude-3", "messages": []}
        dump_id = await dump_error(
            persistence,
            request_body=req_body,
            converted_body=conv_body,
            status_code=500,
            error_phase="upstream",
        )
        assert dump_id is not None
        entry = await persistence.get_error_dump(dump_id)
        assert entry is not None
        assert "body_hash" in entry
        assert "converted_body_hash" in entry
        assert entry["body_hash"] != entry["converted_body_hash"]

    @pytest.mark.asyncio
    async def test_tool_name_collisions_stored(
        self, persistence: PersistenceManager
    ) -> None:
        dump_id = await dump_error(
            persistence,
            request_body={"model": "gpt-4", "messages": []},
            status_code=400,
            error_phase="upstream",
            tool_name_collisions={"exec": 2, "wait": 3},
        )
        assert dump_id is not None
        entry = await persistence.get_error_dump(dump_id)
        assert entry is not None
        # Stored as JSON with sorted keys.
        assert entry["tool_name_collisions"] == '{"exec": 2, "wait": 3}'

    @pytest.mark.asyncio
    async def test_tool_name_collisions_absent_is_null(
        self, persistence: PersistenceManager
    ) -> None:
        dump_id = await dump_error(
            persistence,
            request_body={"model": "gpt-4", "messages": []},
            status_code=400,
            error_phase="upstream",
        )
        assert dump_id is not None
        entry = await persistence.get_error_dump(dump_id)
        assert entry is not None
        # Empty map and None both store NULL, which the query drops.
        assert "tool_name_collisions" not in entry

    @pytest.mark.asyncio
    async def test_fire_and_forget_safety(
        self, persistence: PersistenceManager
    ) -> None:
        """dump_error should not raise even if persistence is broken."""
        await persistence.close()  # close the connection to simulate breakage
        result = await dump_error(
            persistence,
            request_body={"test": True},
            status_code=500,
            error_phase="upstream",
        )
        # Should return None (logged the error) rather than raising
        assert result is None


# ------------------------------------------------------------------
# PersistenceManager error dump methods
# ------------------------------------------------------------------


class TestPersistenceErrorDumps:
    """Direct tests for PersistenceManager error dump operations."""

    @pytest.fixture
    async def persistence(self, tmp_path: object) -> PersistenceManager:
        return await PersistenceManager.create(str(tmp_path))

    @pytest.mark.asyncio
    async def test_query_with_filters(self, persistence: PersistenceManager) -> None:
        # Insert two dumps with different phases
        for i, phase in enumerate(["upstream", "conversion"]):
            await dump_error(
                persistence,
                request_body={"index": i},
                model="test-model",
                status_code=500,
                error_phase=phase,
            )

        entries, total = await persistence.query_error_dumps(error_phase="upstream")
        assert total == 1
        assert entries[0]["error_phase"] == "upstream"

        entries, total = await persistence.query_error_dumps()
        assert total == 2

    @pytest.mark.asyncio
    async def test_clear_error_dumps(self, persistence: PersistenceManager) -> None:
        await dump_error(
            persistence,
            request_body={"test": True},
            status_code=500,
            error_phase="upstream",
        )
        assert await persistence.count_error_dumps() == 1

        await persistence.clear_error_dumps()
        assert await persistence.count_error_dumps() == 0

        # Bodies should also be cleaned up
        row = await persistence._conn.execute_fetchone(
            "SELECT COUNT(*) FROM dump_bodies"
        )
        assert row[0] == 0

    @pytest.mark.asyncio
    async def test_get_nonexistent_dump(self, persistence: PersistenceManager) -> None:
        assert await persistence.get_error_dump("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_nonexistent_body(self, persistence: PersistenceManager) -> None:
        assert await persistence.get_dump_body("nonexistent") is None

    @pytest.mark.asyncio
    async def test_query_pagination(self, persistence: PersistenceManager) -> None:
        for i in range(5):
            await dump_error(
                persistence,
                request_body={"index": i},
                status_code=500,
                error_phase="upstream",
            )

        entries, total = await persistence.query_error_dumps(limit=2, offset=0)
        assert total == 5
        assert len(entries) == 2

        entries, total = await persistence.query_error_dumps(limit=2, offset=4)
        assert total == 5
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_request_log_id_stored(self, persistence: PersistenceManager) -> None:
        from llm_rosetta_gateway.observability import RequestLogEntry

        log_entry = RequestLogEntry.create(
            model="test-model",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=500,
            duration_ms=10.0,
        )
        await persistence.insert_log_entries([log_entry.to_dict()])

        dump_id = await dump_error(
            persistence,
            request_body={"test": True},
            status_code=500,
            error_phase="upstream",
            request_log_id=log_entry.id,
        )
        assert dump_id is not None
        entry = await persistence.get_error_dump(dump_id)
        assert entry is not None
        assert entry["request_log_id"] == log_entry.id


# ------------------------------------------------------------------
# Whole-store filters
# ------------------------------------------------------------------


class TestDumpQueryFilters:
    """The reads behind the panel's server-side paging and filtering.

    Before these filters existed the panel loaded one 500-row page and
    filtered it in the browser, so a header count and every filter described
    a subset of the store.  Each case here pins a filter that must reach the
    whole store.
    """

    @pytest.fixture
    async def persistence(self, tmp_path: object) -> PersistenceManager:
        return await PersistenceManager.create(str(tmp_path))

    @staticmethod
    async def _insert(
        pm: PersistenceManager,
        *,
        dump_id: str,
        timestamp: str = "2026-11-01T00:00:00+00:00",
        model: str | None = "test-model",
        provider: str | None = None,
        status_code: int | None = 500,
        error_phase: str | None = "upstream",
        request_log_id: str | None = None,
    ) -> None:
        await pm.insert_error_dump(
            dump_id=dump_id,
            request_log_id=request_log_id,
            timestamp=timestamp,
            model=model,
            source_provider="chat_completions",
            target_provider="messages",
            provider_name=provider,
            status_code=status_code,
            error_phase=error_phase,
            body_hash=None,
            response_text=None,
            upstream_url=None,
        )

    @pytest.mark.asyncio
    async def test_status_class_filters_the_whole_store(
        self, persistence: PersistenceManager
    ) -> None:
        for i, code in enumerate([404, 429, 500, 502]):
            await self._insert(persistence, dump_id=f"d{i}", status_code=code)

        entries, total = await persistence.query_error_dumps(
            status_min=400, status_max=499
        )
        assert total == 2
        assert sorted(e["status_code"] for e in entries) == [404, 429]

        entries, total = await persistence.query_error_dumps(
            status_min=429, status_max=429
        )
        assert total == 1
        assert entries[0]["status_code"] == 429

    @pytest.mark.asyncio
    async def test_time_bounds_are_inclusive(
        self, persistence: PersistenceManager
    ) -> None:
        await self._insert(
            persistence, dump_id="early", timestamp="2026-10-31T23:59:59+00:00"
        )
        await self._insert(
            persistence, dump_id="edge", timestamp="2026-11-01T00:00:00+00:00"
        )
        await self._insert(
            persistence, dump_id="late", timestamp="2026-11-02T00:00:00+00:00"
        )

        entries, total = await persistence.query_error_dumps(
            start="2026-11-01T00:00:00+00:00"
        )
        assert total == 2
        assert {e["id"] for e in entries} == {"edge", "late"}

        entries, total = await persistence.query_error_dumps(
            start="2026-11-01T00:00:00+00:00", end="2026-11-01T00:00:00+00:00"
        )
        assert total == 1
        assert entries[0]["id"] == "edge"

    @pytest.mark.asyncio
    async def test_id_search_matches_dump_and_request_log_ids(
        self, persistence: PersistenceManager
    ) -> None:
        from llm_rosetta_gateway.observability import RequestLogEntry

        log_entry = RequestLogEntry.create(
            model="test-model",
            source_provider="chat_completions",
            target_provider="messages",
            is_stream=False,
            status_code=500,
            duration_ms=10.0,
        )
        await persistence.insert_log_entries([log_entry.to_dict()])
        await self._insert(persistence, dump_id="dump-abc")
        await self._insert(persistence, dump_id="dump-def", request_log_id=log_entry.id)

        entries, total = await persistence.query_error_dumps(q="ABC")
        assert total == 1
        assert entries[0]["id"] == "dump-abc"

        entries, total = await persistence.query_error_dumps(q=log_entry.id[-8:])
        assert total == 1
        assert entries[0]["id"] == "dump-def"

    @pytest.mark.asyncio
    async def test_model_search_is_a_substring_and_escapes_wildcards(
        self, persistence: PersistenceManager
    ) -> None:
        for i, model in enumerate(["gpt-5-nano", "gpt-5-mini", "claude-haiku"]):
            await self._insert(persistence, dump_id=f"d{i}", model=model)

        entries, total = await persistence.query_error_dumps(model_search="GPT-5")
        assert total == 2

        # `%` and `_` are LIKE wildcards; searched literally each must match
        # one row, not every row the wildcard pattern would match.
        for i, model in enumerate(["m100%", "m100x", "m100_"]):
            await self._insert(persistence, dump_id=f"w{i}", model=model)
        entries, total = await persistence.query_error_dumps(model_search="100%")
        assert total == 1
        assert entries[0]["model"] == "m100%"
        entries, total = await persistence.query_error_dumps(model_search="100_")
        assert total == 1
        assert entries[0]["model"] == "m100_"

    @pytest.mark.asyncio
    async def test_the_id_cursor_holds_while_a_row_arrives(
        self, persistence: PersistenceManager
    ) -> None:
        for i in range(6):
            await self._insert(
                persistence, dump_id=f"d{i}", timestamp=f"2026-11-01T00:00:0{i}+00:00"
            )

        first, total = await persistence.query_error_dumps(limit=3)
        assert [e["id"] for e in first] == ["d5", "d4", "d3"]
        assert total == 6

        # A dump written at the head while the walk is between pages: an
        # `OFFSET 3` would return `d3` again and lose `d0` from the file.
        await self._insert(
            persistence, dump_id="new", timestamp="2026-11-01T00:00:09+00:00"
        )
        last = first[-1]
        second, _ = await persistence.query_error_dumps(
            limit=3, before=(last["timestamp"], last["id"])
        )
        assert [e["id"] for e in second] == ["d2", "d1", "d0"]

        walked = [e["id"] for e in first + second]
        assert len(walked) == len(set(walked)) == 6

    @pytest.mark.asyncio
    async def test_a_same_timestamp_boundary_names_one_row(
        self, persistence: PersistenceManager
    ) -> None:
        # Every dump written in the same second shares a timestamp, so only
        # the id can tell two page boundaries apart.
        for i in range(4):
            await self._insert(persistence, dump_id=f"d{i}")

        first, _ = await persistence.query_error_dumps(limit=2)
        last = first[-1]
        second, _ = await persistence.query_error_dumps(
            limit=2, before=(last["timestamp"], last["id"])
        )
        walked = [e["id"] for e in first + second]
        assert walked == ["d3", "d2", "d1", "d0"]

    @pytest.mark.asyncio
    async def test_a_cursor_combines_with_a_filter(
        self, persistence: PersistenceManager
    ) -> None:
        # The cursor adds three bind parameters, so a clause/parameter mismatch
        # would show up as the wrong rows rather than as an error.
        for i, provider in enumerate(["argo", "other", "argo", "argo"]):
            await self._insert(
                persistence,
                dump_id=f"d{i}",
                timestamp=f"2026-11-01T00:00:0{i}+00:00",
                provider=provider,
            )

        first, _ = await persistence.query_error_dumps(provider="argo", limit=2)
        assert [e["id"] for e in first] == ["d3", "d2"]
        last = first[-1]
        second, _ = await persistence.query_error_dumps(
            provider="argo", limit=2, before=(last["timestamp"], last["id"])
        )
        assert [e["id"] for e in second] == ["d0"]

    @pytest.mark.asyncio
    async def test_ids_selects_named_rows_only(
        self, persistence: PersistenceManager
    ) -> None:
        for i in range(3):
            await self._insert(persistence, dump_id=f"d{i}")

        entries, total = await persistence.query_error_dumps(ids=["d0", "d2"])
        assert total == 2
        assert {e["id"] for e in entries} == {"d0", "d2"}

        # An empty selection is no rows, not a syntax error and not all rows.
        entries, total = await persistence.query_error_dumps(ids=[])
        assert total == 0
        assert entries == []

    @pytest.mark.asyncio
    async def test_id_list_spans_the_filtered_set(
        self, persistence: PersistenceManager
    ) -> None:
        for i, provider in enumerate(["argo", "argo", "other"]):
            await self._insert(persistence, dump_id=f"d{i}", provider=provider)

        ids = await persistence.query_error_dump_ids(provider="argo")
        assert set(ids) == {"d0", "d1"}

    @pytest.mark.asyncio
    async def test_facets_list_what_the_store_holds(
        self, persistence: PersistenceManager
    ) -> None:
        await self._insert(persistence, dump_id="d0", provider="argo", model="gpt-5")
        await self._insert(persistence, dump_id="d1", provider="ark", model="gpt-5")
        await self._insert(persistence, dump_id="d2", provider="argo", model="claude")

        facets = await persistence.error_dump_facets()
        assert facets["providers"] == ["argo", "ark"]
        assert facets["models"] == ["claude", "gpt-5"]
