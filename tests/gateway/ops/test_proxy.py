"""Tests for OpsProxyRequest — hot-path proxy telemetry recording."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_rosetta.gateway.ops.base import OpsContext
from llm_rosetta.gateway.ops.proxy import OpsProxyRequest


def _make_ctx(
    *, metrics: Any = None, request_log: Any = None, ops_log: Any = None
) -> OpsContext:
    return OpsContext(
        ops_log=ops_log,
        request_log=request_log,
        metrics=metrics,
    )


class TestOpsProxyRequest:
    @pytest.mark.asyncio
    async def test_records_to_request_log(self):
        rl = MagicMock()
        rl.add = AsyncMock()
        ctx = _make_ctx(request_log=rl)

        op = OpsProxyRequest(
            ctx,
            model="gpt-4",
            source_provider="openai_chat",
            target_provider="openai_chat",
            provider_name="openai",
            is_stream=False,
            status_code=200,
            duration_ms=150.5,
            error_detail=None,
            profile={
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                }
            },
        )
        await op.execute()

        rl.add.assert_awaited_once()
        entry = rl.add.call_args[0][0]
        assert entry.model == "gpt-4"
        assert entry.status_code == 200
        assert entry.input_tokens == 10
        assert entry.output_tokens == 20
        assert op.entry_id is not None

    @pytest.mark.asyncio
    async def test_records_to_metrics(self):
        m = MagicMock()
        ctx = _make_ctx(metrics=m)

        op = OpsProxyRequest(
            ctx,
            model="claude-3",
            source_provider="anthropic",
            target_provider="anthropic",
            provider_name="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=100.0,
            error_detail=None,
        )
        await op.execute()

        m.record_request.assert_called_once()
        call_kwargs = m.record_request.call_args[1]
        assert call_kwargs["model"] == "claude-3"
        assert call_kwargs["status_code"] == 200

    @pytest.mark.asyncio
    async def test_streaming_decrements_active_streams(self):
        m = MagicMock()
        m.active_streams = 5
        ctx = _make_ctx(metrics=m)

        op = OpsProxyRequest(
            ctx,
            model="gpt-4",
            source_provider="openai_chat",
            target_provider="openai_chat",
            provider_name="openai",
            is_stream=True,
            status_code=200,
            duration_ms=500.0,
            error_detail=None,
        )
        await op.execute()

        assert m.active_streams == 4

    @pytest.mark.asyncio
    async def test_entry_id_override(self):
        rl = MagicMock()
        rl.add = AsyncMock()
        ctx = _make_ctx(request_log=rl)

        op = OpsProxyRequest(
            ctx,
            model="gpt-4",
            source_provider="openai_chat",
            target_provider="openai_chat",
            provider_name="openai",
            is_stream=True,
            status_code=200,
            duration_ms=100.0,
            error_detail=None,
            entry_id_override="preset-id-123",
        )
        await op.execute()

        entry = rl.add.call_args[0][0]
        assert entry.id == "preset-id-123"
        assert op.entry_id == "preset-id-123"

    @pytest.mark.asyncio
    async def test_does_not_write_ops_log(self):
        ops_log = MagicMock()
        ops_log.add = AsyncMock()
        ctx = _make_ctx(ops_log=ops_log)

        op = OpsProxyRequest(
            ctx,
            model="gpt-4",
            source_provider="openai_chat",
            target_provider="openai_chat",
            provider_name="openai",
            is_stream=False,
            status_code=200,
            duration_ms=100.0,
            error_detail=None,
        )
        await op.execute()

        ops_log.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_graceful_no_services(self):
        ctx = _make_ctx()
        op = OpsProxyRequest(
            ctx,
            model="gpt-4",
            source_provider="openai_chat",
            target_provider="openai_chat",
            provider_name="openai",
            is_stream=False,
            status_code=200,
            duration_ms=100.0,
            error_detail=None,
        )
        await op.execute()
        assert op.entry_id is None

    @pytest.mark.asyncio
    async def test_streaming_skips_usage_extraction(self):
        rl = MagicMock()
        rl.add = AsyncMock()
        ctx = _make_ctx(request_log=rl)

        op = OpsProxyRequest(
            ctx,
            model="gpt-4",
            source_provider="openai_chat",
            target_provider="openai_chat",
            provider_name="openai",
            is_stream=True,
            status_code=200,
            duration_ms=100.0,
            error_detail=None,
            profile={
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                }
            },
        )
        await op.execute()

        entry = rl.add.call_args[0][0]
        assert entry.input_tokens is None

    def test_slots_no_dict(self):
        ctx = _make_ctx()
        op = OpsProxyRequest(
            ctx,
            model="x",
            source_provider="a",
            target_provider="b",
            provider_name="c",
            is_stream=False,
            status_code=200,
            duration_ms=0.0,
            error_detail=None,
        )
        assert not hasattr(op, "__dict__")
