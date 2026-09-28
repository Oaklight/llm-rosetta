"""Tests for the error dump module."""

from __future__ import annotations

import json
import zlib

import pytest

from llm_rosetta.observability.error_dump import (
    compress_body,
    compute_body_hash,
    decompress_body,
    dump_error,
    offload_images,
)
from llm_rosetta.observability.persistence import PersistenceManager


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
        dump_id = await dump_error(
            persistence,
            request_body={"test": True},
            status_code=500,
            error_phase="upstream",
            request_log_id="abc123",
        )
        assert dump_id is not None
        entry = await persistence.get_error_dump(dump_id)
        assert entry is not None
        assert entry["request_log_id"] == "abc123"
