"""Tests for cursor-based resumable streaming (stream replay).

Covers:
* Resume cursor protocol parsing (Last-Event-ID / explicit headers).
* Normal resume mid-stream with contiguous sequence numbers.
* Explicit errors for unknown/expired sessions and out-of-range cursors.
* Replaying after the upstream already finished (no second upstream call).
* SQLite persistence across "restarts" (a new manager on the same dir),
  orphaned-pump takeover, and cross-manager live-tail polling.
* Proxy integration: a disconnecting initial subscriber keeps the pump
  alive and resumes by cursor; the legacy path is unchanged when replay
  is disabled.
* Application-level resume interception and an end-to-end loopback
  disconnect/reconnect over a real socket.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import re
import socket
import sqlite3
import threading
import time as _time
from typing import Any

import pytest

from llm_rosetta.gateway.proxy import (
    _iter_stream_messages,
    _run_replay_pump,
    handle_streaming,
)
from llm_rosetta.gateway.stream_replay import (
    HEADER_LAST_EVENT_ID,
    HEADER_REPLAY,
    HEADER_REPLAY_CURSOR,
    HEADER_REPLAY_ERROR,
    HEADER_REPLAY_ID,
    ReplayRejected,
    STATE_COMPLETE,
    STATE_STREAMING,
    StreamReplayManager,
    StreamReplaySettings,
    encode_message,
    parse_resume_cursor,
    replay_headers,
)
from llm_rosetta.gateway.transport.sse_format import SSE_FORMATTERS
from llm_rosetta.pipeline import ConversionPipeline

from llm_rosetta._vendor.httpserver import StreamingResponse
from llm_rosetta.gateway.app import (
    GatewayExtensions,
    _try_handle_replay,
    create_app,
)
from llm_rosetta.gateway.config import GatewayConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _settings(**overrides: Any) -> StreamReplaySettings:
    base: dict[str, Any] = {"persist": False}
    base.update(overrides)
    return StreamReplaySettings(**base)


async def _drain(manager: StreamReplayManager, sid: str, cursor: int) -> list[str]:
    messages: list[str] = []
    async for message in manager.subscribe(sid, cursor):
        messages.append(message)
    return messages


def _ids(sid: str, messages: list[str]) -> list[int]:
    """Parse sequence numbers from ``id: <sid>:<seq>`` SSE field lines."""
    seqs: list[int] = []
    for message in messages:
        first_line = message.split("\n", 1)[0]
        assert first_line.startswith("id: "), message
        raw_sid, raw_seq = first_line[4:].rsplit(":", 1)
        assert raw_sid == sid
        seqs.append(int(raw_seq))
    return seqs


# ---------------------------------------------------------------------------
# Cursor protocol
# ---------------------------------------------------------------------------


class TestCursorProtocol:
    def test_last_event_id_single_header(self):
        headers = {"last-event-id": "rs_abc:12"}
        assert parse_resume_cursor(headers) == ("rs_abc", 12)

    def test_explicit_header_pair(self):
        headers = {"x-stream-replay-id": "rs_abc", "x-stream-replay-cursor": "7"}
        assert parse_resume_cursor(headers) == ("rs_abc", 7)

    def test_no_headers_is_normal_request(self):
        assert parse_resume_cursor({}) is None

    @pytest.mark.parametrize(
        "headers",
        [
            {"last-event-id": "no-colon-here"},
            {"last-event-id": "rs_abc:notanint"},
            {"last-event-id": "rs_abc:-3"},
            {"x-stream-replay-id": "rs_abc"},
            {"x-stream-replay-cursor": "5"},
        ],
    )
    def test_malformed_headers_rejected(self, headers):
        with pytest.raises(ReplayRejected) as exc:
            parse_resume_cursor(headers)
        assert exc.value.status_code == 400
        assert exc.value.code == "invalid_cursor"

    def test_encode_and_headers(self):
        encoded = encode_message("rs_x", 3, "data: {}\n\n")
        assert encoded == "id: rs_x:3\ndata: {}\n\n"

        headers = replay_headers("rs_x", 3, 300, resumed=True)
        assert headers[HEADER_REPLAY_ID] == "rs_x"
        assert headers[HEADER_REPLAY_CURSOR] == "3"
        assert headers[HEADER_REPLAY] == "resumed"


class TestStreamReplaySettings:
    def test_defaults(self):
        settings = StreamReplaySettings.from_config(None)
        assert settings.enabled is True
        assert settings.ttl_seconds == 300.0
        assert settings.persist is True

    def test_override_and_millisecond_poll(self):
        settings = StreamReplaySettings.from_config(
            {
                "enabled": False,
                "ttl_seconds": 60,
                "max_events": 100,
                "max_sessions": 5,
                "persist": False,
                "poll_interval_ms": 25,
                "orphan_timeout": 8,
            }
        )
        assert settings.enabled is False
        assert settings.ttl_seconds == 60
        assert settings.max_events == 100
        assert settings.max_sessions == 5
        assert settings.persist is False
        assert settings.poll_interval == 0.025
        assert settings.orphan_timeout == 8

    def test_invalid_values_fall_back_to_defaults(self):
        settings = StreamReplaySettings.from_config(
            {"ttl_seconds": "nope", "max_events": -3}
        )
        assert settings.ttl_seconds == 300.0
        assert settings.max_events >= 1


# ---------------------------------------------------------------------------
# In-memory manager
# ---------------------------------------------------------------------------


class TestMemoryManager:
    def test_normal_resume_contiguous_sequence(self):
        async def scenario() -> None:
            manager = StreamReplayManager(_settings())
            sid = await manager.create_session("openai_chat", "m")
            await manager.append_messages(sid, ["data: a\n\n", "data: b\n\n"])
            await manager.append_messages(sid, ["data: c\n\n", "data: d\n\n"])
            await manager.finish(sid)

            await manager.validate_resume(sid, 2)
            messages = await _drain(manager, sid, 2)
            assert _ids(sid, messages) == [3, 4]
            assert messages[0].endswith("data: c\n\n")
            await manager.aclose()

        _run(scenario())

    def test_two_subscribers_form_one_continuous_stream(self):
        """Subscriber 1 disconnects mid-stream; subscriber 2 gets the tail."""

        async def scenario() -> None:
            manager = StreamReplayManager(_settings())
            sid = await manager.create_session("openai_chat", "m")

            async def collect_head() -> list[str]:
                head: list[str] = []
                async for message in manager.subscribe(sid, 0):
                    head.append(message)
                    if len(head) == 2:
                        break
                return head  # client disconnect closes the generator

            head_task = asyncio.create_task(collect_head())
            await asyncio.sleep(0.02)
            await manager.append_messages(
                sid, ["data: a\n\n", "data: b\n\n", "data: c\n\n"]
            )
            head = await asyncio.wait_for(head_task, timeout=2)

            await manager.append_messages(sid, ["data: d\n\n", "data: [DONE]\n\n"])
            await manager.finish(sid)

            cursor = _ids(sid, head)[-1]
            tail = await _drain(manager, sid, cursor)

            combined = _ids(sid, head) + _ids(sid, tail)
            assert combined == [1, 2, 3, 4, 5]
            assert len(combined) == len(set(combined))  # no duplicates
            assert tail[-1].endswith("data: [DONE]\n\n")
            await manager.aclose()

        _run(scenario())

    def test_live_tail_waits_for_pump(self):
        """A subscriber attached before the pump finishes gets new events."""

        async def scenario() -> None:
            manager = StreamReplayManager(_settings())
            sid = await manager.create_session("openai_chat", "m")
            drain_task = asyncio.create_task(_drain(manager, sid, 0))
            await asyncio.sleep(0.02)  # subscriber is now waiting

            for i in range(3):
                await manager.append_messages(sid, [f"data: {i}\n\n"])
                await asyncio.sleep(0.01)
            await manager.finish(sid)

            messages = await asyncio.wait_for(drain_task, timeout=2)
            assert _ids(sid, messages) == [1, 2, 3]
            await manager.aclose()

        _run(scenario())

    def test_unknown_session_is_explicit_410(self):
        async def scenario() -> None:
            manager = StreamReplayManager(_settings())
            with pytest.raises(ReplayRejected) as exc:
                await manager.validate_resume("rs_missing", 0)
            assert exc.value.status_code == 410
            assert exc.value.code == "session_not_found"
            assert "new request" in exc.value.message
            await manager.aclose()

        _run(scenario())

    def test_cursor_beyond_stream_length_is_410(self):
        async def scenario() -> None:
            manager = StreamReplayManager(_settings())
            sid = await manager.create_session("openai_chat", "m")
            await manager.append_messages(sid, ["data: a\n\n"])
            with pytest.raises(ReplayRejected) as exc:
                await manager.validate_resume(sid, 99)
            assert exc.value.status_code == 410
            assert exc.value.code == "cursor_out_of_range"
            await manager.aclose()

        _run(scenario())

    def test_resume_at_tip_of_completed_session_is_410(self):
        async def scenario() -> None:
            manager = StreamReplayManager(_settings())
            sid = await manager.create_session("openai_chat", "m")
            await manager.append_messages(sid, ["data: a\n\n", "data: [DONE]\n\n"])
            await manager.finish(sid)
            with pytest.raises(ReplayRejected) as exc:
                await manager.validate_resume(sid, 2)
            assert exc.value.status_code == 410
            assert exc.value.code == "session_complete"
            await manager.aclose()

        _run(scenario())

    def test_expired_session_is_410(self):
        async def scenario() -> None:
            manager = StreamReplayManager(_settings(ttl_seconds=0.02))
            sid = await manager.create_session("openai_chat", "m")
            await manager.append_messages(sid, ["data: a\n\n"])
            await manager.finish(sid)
            await asyncio.sleep(0.05)
            with pytest.raises(ReplayRejected) as exc:
                await manager.validate_resume(sid, 0)
            assert exc.value.status_code == 410
            assert exc.value.code == "session_expired"
            await manager.aclose()

        _run(scenario())

    def test_replay_after_upstream_finished(self):
        """No live pump: a full replay comes from the cache alone."""

        async def scenario() -> None:
            manager = StreamReplayManager(_settings())
            sid = await manager.create_session("openai_chat", "m")
            await manager.append_messages(sid, [f"data: {i}\n\n" for i in range(5)])
            await manager.append_messages(sid, ["data: [DONE]\n\n"])
            await manager.finish(sid)

            # Cursor 0 replay long after completion — cache is the only source.
            messages = await _drain(manager, sid, 0)
            assert _ids(sid, messages) == [1, 2, 3, 4, 5, 6]
            assert messages[-1].endswith("data: [DONE]\n\n")
            await manager.aclose()

        _run(scenario())

    def test_ring_window_eviction_is_an_explicit_error(self):
        async def scenario() -> None:
            manager = StreamReplayManager(_settings(max_events=4))
            sid = await manager.create_session("openai_chat", "m")
            await manager.append_messages(sid, [f"data: {i}\n\n" for i in range(6)])
            await manager.finish(sid)

            # Window holds seqs 3..6; cursor 0/1 would have a gap.
            with pytest.raises(ReplayRejected) as exc:
                await manager.validate_resume(sid, 1)
            assert exc.value.code == "cursor_evicted"

            # Cursor inside the surviving window still works.
            await manager.validate_resume(sid, 2)
            messages = await _drain(manager, sid, 2)
            assert _ids(sid, messages) == [3, 4, 5, 6]
            await manager.aclose()

        _run(scenario())

    def test_session_cap_falls_back_to_none(self):
        async def scenario() -> None:
            manager = StreamReplayManager(_settings(max_sessions=2))
            a = await manager.create_session("openai_chat", "m")
            assert a is not None
            await manager.finish(a)
            b = await manager.create_session("openai_chat", "m")  # evicts a
            c = await manager.create_session("openai_chat", "m")
            assert b is not None and c is not None
            with pytest.raises(ReplayRejected):
                await manager.validate_resume(a, 0)
            await manager.aclose()

        _run(scenario())

    def test_streaming_session_does_not_expire_in_flight(self):
        async def scenario() -> None:
            manager = StreamReplayManager(_settings(ttl_seconds=0.01))
            sid = await manager.create_session("openai_chat", "m")
            await manager.append_messages(sid, ["data: a\n\n"])
            await asyncio.sleep(0.05)  # beyond ttl, but pump still active
            record = await manager.validate_resume(sid, 0)
            assert record.state == STATE_STREAMING
            await manager.finish(sid)
            await manager.aclose()

        _run(scenario())


# ---------------------------------------------------------------------------
# SQLite store: restart, orphans, cross-worker tail
# ---------------------------------------------------------------------------


class TestSqlitePersistence:
    def test_replay_after_manager_restart(self, tmp_path):
        async def write() -> str:
            manager = StreamReplayManager(
                StreamReplaySettings(), data_dir=str(tmp_path)
            )
            assert manager.persistent
            sid = await manager.create_session("openai_chat", "m")
            await manager.append_messages(sid, [f"data: {i}\n\n" for i in range(4)])
            await manager.finish(sid, state=STATE_COMPLETE)
            await manager.aclose()
            return sid

        sid = _run(write())

        async def read_back() -> None:
            manager = StreamReplayManager(
                StreamReplaySettings(), data_dir=str(tmp_path)
            )
            await manager.validate_resume(sid, 2)
            messages = await _drain(manager, sid, 2)
            assert _ids(sid, messages) == [3, 4]
            assert "data: 3" in messages[1]
            await manager.aclose()

        _run(read_back())

    def test_orphaned_stream_is_taken_over_with_interrupted_tail(self, tmp_path):
        async def die_mid_stream() -> str:
            # Manager A "crashes" (aclose without finish) while streaming.
            manager = StreamReplayManager(
                StreamReplaySettings(), data_dir=str(tmp_path)
            )
            sid = await manager.create_session("openai_chat", "m")
            await manager.append_messages(sid, ["data: a\n\n", "data: b\n\n"])
            await manager.aclose()
            return sid

        sid = _run(die_mid_stream())

        async def takeover() -> None:
            manager = StreamReplayManager(
                StreamReplaySettings(
                    persist=True, orphan_timeout=0.02, poll_interval=0.01
                ),
                data_dir=str(tmp_path),
            )
            await asyncio.sleep(0.05)  # let A's heartbeat go stale

            # Cursor past the buffered tip is rejected before streaming.
            with pytest.raises(ReplayRejected) as exc:
                await manager.validate_resume(sid, 99)
            assert exc.value.code == "cursor_out_of_range"

            record = await manager.validate_resume(sid, 1)
            assert record.state == "interrupted"
            messages = await _drain(manager, sid, 1)
            seqs = _ids(sid, messages)
            assert seqs[0] == 2
            joined = "".join(messages)
            assert "Upstream stream ended before completion" in joined
            assert "data: [DONE]" in joined
            assert seqs == list(range(2, seqs[-1] + 1))
            await manager.aclose()

        _run(takeover())

    def test_concurrent_pumps_share_store_safely(self, tmp_path):
        async def scenario() -> None:
            manager = StreamReplayManager(
                StreamReplaySettings(), data_dir=str(tmp_path)
            )
            n_sessions, n_msgs = 8, 20
            sids = [
                await manager.create_session("openai_chat", "m")
                for _ in range(n_sessions)
            ]

            async def pump(sid: str) -> None:
                for i in range(n_msgs):
                    await manager.append_messages(sid, [f"data: {sid}-{i}\n\n"])
                await manager.finish(sid)

            await asyncio.gather(*(pump(sid) for sid in sids))
            for sid in sids:
                messages = await _drain(manager, sid, 0)
                assert _ids(sid, messages) == list(range(1, n_msgs + 1))
            await manager.aclose()

        _run(scenario())

    def test_other_worker_follows_live_tail_via_polling(self, tmp_path):
        async def scenario() -> None:
            worker_a = StreamReplayManager(
                StreamReplaySettings(), data_dir=str(tmp_path)
            )
            worker_b = StreamReplayManager(
                StreamReplaySettings(persist=True, poll_interval=0.01),
                data_dir=str(tmp_path),
            )
            sid = await worker_a.create_session("openai_chat", "m")
            assert sid not in worker_b._live  # B has no local pump

            await worker_b.validate_resume(sid, 0)
            tail_task = asyncio.create_task(_drain(worker_b, sid, 0))
            await asyncio.sleep(0.03)

            for i in range(4):
                await worker_a.append_messages(sid, [f"data: {i}\n\n"])
                await asyncio.sleep(0.02)
            await worker_a.finish(sid)

            messages = await asyncio.wait_for(tail_task, timeout=3)
            assert _ids(sid, messages) == [1, 2, 3, 4]
            await worker_a.aclose()
            await worker_b.aclose()

        _run(scenario())


# ---------------------------------------------------------------------------
# Proxy integration (real pipeline + fake gated upstream)
# ---------------------------------------------------------------------------


def _openai_chat_chunks() -> list[dict[str, Any]]:
    return [
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "model": "m",
            "created": 1700000000,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        },
        *[
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "model": "m",
                "created": 1700000000,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None,
                    }
                ],
            }
            for token in ("Hel", "lo ", "world")
        ],
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "model": "m",
            "created": 1700000000,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]


class _FakeUpstreamStream:
    """Upstream stream with a small delay so subscribers can disconnect."""

    def __init__(self, chunks: list[dict[str, Any]], delay: float = 0.01):
        self._chunks = list(chunks)
        self._delay = delay
        self.status_code = 200
        self.is_error = False
        self.closed = False

    async def __aenter__(self) -> _FakeUpstreamStream:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        await self.close()
        return False

    def __aiter__(self) -> _FakeUpstreamStream:
        return self

    async def __anext__(self) -> dict[str, Any]:
        await asyncio.sleep(self._delay)
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def read_error(self) -> str:
        return ""

    async def close(self) -> None:
        self.closed = True


class _FakeTransport:
    def __init__(self, chunks: list[dict[str, Any]], *, delay: float = 0.01) -> None:
        self._chunks = chunks
        self._delay = delay
        self.streaming_calls = 0
        self.last_stream: _FakeUpstreamStream | None = None

    async def send(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("non-streaming send should not be called")

    async def send_streaming(
        self,
        provider_info: Any,
        url: str,
        body: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> _FakeUpstreamStream:
        self.streaming_calls += 1
        self.last_stream = _FakeUpstreamStream(self._chunks, delay=self._delay)
        return self.last_stream

    async def close(self) -> None:
        pass


def _route_and_body() -> tuple[Any, Any, dict[str, Any]]:
    cfg = GatewayConfig(
        {
            "providers": {
                "test-provider": {
                    "type": "openai_chat",
                    "base_url": "http://upstream.local/v1",
                    "api_key": "sk-test",
                }
            },
            "models": {"m": "test-provider"},
            "server": {"open_on_no_keys": True},
        }
    )
    route, provider_info = cfg.resolve("openai_chat", "m")
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    return route, provider_info, body


async def _wait_terminal(manager: StreamReplayManager, sid: str) -> None:
    for _ in range(500):
        record = manager._store.get_session(sid)
        if record is not None and record.state in {
            "complete",
            "failed",
            "interrupted",
        }:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("pump did not finish in time")


class TestProxyReplayIntegration:
    def test_disconnect_then_resume_with_single_upstream_call(self):
        async def scenario() -> None:
            route, provider_info, body = _route_and_body()
            transport = _FakeTransport(_openai_chat_chunks())
            manager = StreamReplayManager(_settings())

            response, _ = await handle_streaming(
                route,
                provider_info,
                body,
                transport=transport,
                replay_manager=manager,
            )
            assert HEADER_REPLAY_ID in response.headers
            sid = response.headers[HEADER_REPLAY_ID]
            assert response.headers[HEADER_REPLAY_CURSOR] == "0"

            # Initial subscriber: take two messages, then hang up.
            subscriber = response._generator
            head: list[str] = []
            async for message in subscriber:
                head.append(message)
                if len(head) == 2:
                    break
            await subscriber.aclose()

            # The upstream connection must stay open for the detached pump.
            assert transport.last_stream is not None
            await asyncio.sleep(0.02)
            assert transport.last_stream.closed is False

            await _wait_terminal(manager, sid)
            cursor = _ids(sid, head)[-1]

            # Resume by cursor: no upstream call, contiguous numbering.
            await manager.validate_resume(sid, cursor)
            tail = await _drain(manager, sid, cursor)
            full = _ids(sid, head) + _ids(sid, tail)
            assert full == list(range(1, len(full) + 1))
            assert "data: [DONE]" in tail[-1]
            assert transport.streaming_calls == 1

            # Replay again from zero after upstream ended — still cached.
            again = await _drain(manager, sid, 0)
            assert _ids(sid, again) == list(range(1, len(full) + 1))
            assert transport.streaming_calls == 1
            await manager.aclose()

        _run(scenario())

    def test_legacy_path_without_replay_manager(self):
        async def scenario() -> None:
            route, provider_info, body = _route_and_body()
            transport = _FakeTransport(_openai_chat_chunks())

            response, _ = await handle_streaming(
                route,
                provider_info,
                body,
                transport=transport,
                replay_manager=None,
            )
            assert HEADER_REPLAY_ID not in response.headers
            first = await response._generator.__anext__()
            assert not first.startswith("id: ")  # legacy byte format
            await response._generator.aclose()

        _run(scenario())

    def test_failed_pump_marks_session_failed(self):
        async def scenario() -> None:
            pipeline = ConversionPipeline("openai_chat", "openai_chat")
            pipeline.convert_request(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                }
            )
            processor = pipeline.create_stream_processor()

            class _DyingStream(_FakeUpstreamStream):
                async def __anext__(self) -> dict[str, Any]:
                    await asyncio.sleep(0)
                    raise ConnectionError("connection reset")

            manager = StreamReplayManager(_settings())
            sid = await manager.create_session("openai_chat", "m")
            messages = _iter_stream_messages(
                source_provider="openai_chat",
                stream=_DyingStream([]),
                processor=processor,
                model="m",
                format_sse=SSE_FORMATTERS["openai_chat"],
            )
            _run_replay_pump(manager, sid, messages)
            await _wait_terminal(manager, sid)
            record = manager._store.get_session(sid)
            assert record is not None and record.state == "failed"
            cached = await _drain(manager, sid, 0)
            assert "Upstream stream ended before completion" in "".join(cached)
            await manager.aclose()

        _run(scenario())


# ---------------------------------------------------------------------------
# Application-level resume interception
# ---------------------------------------------------------------------------


class _FakeApp:
    def __init__(self, manager: Any = None) -> None:
        if manager is not None:
            self.stream_replay_manager = manager


class _FakeRequest:
    def __init__(self, headers: dict[str, str], manager: Any = None) -> None:
        self.headers = {k.lower(): v for k, v in headers.items()}
        self.app = _FakeApp(manager)


class TestReplayHandlerInterception:
    def test_no_manager_passes_through(self):
        request = _FakeRequest({"last-event-id": "rs_x:1"})
        assert _run(_try_handle_replay(request, "openai_chat", "rid")) is None

    def test_no_cursor_passes_through(self):
        manager = StreamReplayManager(_settings())
        request = _FakeRequest({}, manager)
        assert _run(_try_handle_replay(request, "openai_chat", "rid")) is None

    def test_unknown_session_returns_410_json(self):
        manager = StreamReplayManager(_settings())
        request = _FakeRequest({"last-event-id": "rs_nope:0"}, manager)
        response = _run(_try_handle_replay(request, "openai_chat", "rid"))
        assert response.status_code == 410
        assert response.headers[HEADER_REPLAY_ERROR] == "session_not_found"
        payload = json.loads(response.body)
        assert "new request" in payload["error"]["message"]

    def test_malformed_cursor_returns_400_json(self):
        manager = StreamReplayManager(_settings())
        request = _FakeRequest({"last-event-id": "garbage"}, manager)
        response = _run(_try_handle_replay(request, "anthropic", "rid"))
        assert response.status_code == 400
        assert response.headers[HEADER_REPLAY_ERROR] == "invalid_cursor"
        payload = json.loads(response.body)
        assert payload["type"] == "error"  # source-format envelope

    def test_resumed_response_streams_from_cache(self):
        async def scenario() -> None:
            manager = StreamReplayManager(_settings())
            sid = await manager.create_session("openai_chat", "m")
            await manager.append_messages(sid, ["data: a\n\n", "data: b\n\n"])
            await manager.finish(sid)

            request = _FakeRequest({HEADER_LAST_EVENT_ID: f"{sid}:1"}, manager)
            response = await _try_handle_replay(request, "openai_chat", "rid")
            assert isinstance(response, StreamingResponse)
            assert response.headers[HEADER_REPLAY] == "resumed"
            assert response.headers[HEADER_REPLAY_CURSOR] == "1"
            pieces = []
            async for chunk in response._generator:
                pieces.append(chunk)
            assert _ids(sid, pieces) == [2]
            await manager.aclose()

        _run(scenario())


# ---------------------------------------------------------------------------
# End-to-end loopback: real socket disconnect / reconnect
# ---------------------------------------------------------------------------


def _e2e_config(tmp_path: Any) -> GatewayConfig:
    return GatewayConfig(
        {
            "providers": {
                "test-provider": {
                    "type": "openai_chat",
                    "base_url": "http://upstream.local/v1",
                    "api_key": "sk-test",
                }
            },
            "models": {"m": "test-provider"},
            "server": {
                "open_on_no_keys": True,
                "data_dir": str(tmp_path),
                "stream_replay": {"poll_interval_ms": 10},
            },
        }
    )


_REQUEST_BODY = json.dumps(
    {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
)


def _raw_post(
    host: str,
    port: int,
    *,
    headers: dict[str, str] | None = None,
    body: str = _REQUEST_BODY,
) -> tuple[dict[str, Any], bytes]:
    """One blocking HTTP/1.1 request; returns (headers including status, body)."""
    conn = http.client.HTTPConnection(host, port, timeout=10)
    raw_headers = {"Content-Type": "application/json"}
    if headers:
        raw_headers.update(headers)
    conn.request("POST", "/v1/chat/completions", body=body, headers=raw_headers)
    response = conn.getresponse()
    data = response.read()
    header_dump = dict(response.getheaders())
    header_dump["__status__"] = response.status
    conn.close()
    return header_dump, data


class TestEndToEndResume:
    def test_disconnect_reconnect_loopback(self, tmp_path):
        from llm_rosetta.gateway.app import run_gateway

        # Slow upstream keeps the stream open well past the first events,
        # so the abrupt close below happens deterministically mid-flight.
        transport = _FakeTransport(_openai_chat_chunks(), delay=0.15)
        extensions = GatewayExtensions(
            transport=transport,
            enable_rate_limiting=False,
            skip_admin_setup=True,
        )

        ready = threading.Event()
        state: dict[str, Any] = {}
        result: dict[str, Any] = {}

        def client_thread() -> None:
            assert ready.wait(timeout=5)
            host, port = "127.0.0.1", state["port"]
            # Independent SQLite connection from this worker thread,
            # mirroring how a different process observes the shared DB.
            poll_conn = sqlite3.connect(str(tmp_path / "stream_replay.db"), timeout=5)
            try:
                # --- Initial request over a raw socket so the client can
                # sever the connection abruptly mid-stream (http.client
                # detaches its own socket attribute during chunked reads) ---
                raw = socket.create_connection((host, port), timeout=10)
                request = (
                    "POST /v1/chat/completions HTTP/1.1\r\n"
                    f"Host: {host}:{port}\r\n"
                    "Content-Type: application/json\r\n"
                    f"Content-Length: {len(_REQUEST_BODY)}\r\n"
                    "Connection: close\r\n\r\n"
                ).encode() + _REQUEST_BODY.encode()
                raw.sendall(request)

                buf = b""
                ids_seen: set[int] = set()
                sid = None
                while len(ids_seen) < 2:
                    chunk = raw.recv(256)
                    if not chunk:
                        break
                    buf += chunk
                    if sid is None:
                        match = re.search(
                            (HEADER_REPLAY_ID + r": (rs_[0-9a-f]+)").encode(),
                            buf,
                            re.IGNORECASE,
                        )
                        if match:
                            sid = match.group(1).decode()
                    ids_seen = {
                        int(m.group(1))
                        for m in re.finditer(rb"id: rs_[0-9a-f]+:(\d+)", buf)
                    }
                assert sid is not None and len(ids_seen) >= 2
                result["sid"] = sid
                raw.close()  # abrupt client disconnect mid-stream

                # Wait for the detached pump to finish server-side,
                # observing it through an independent DB connection.
                deadline = _time.time() + 5
                session_state = None
                while _time.time() < deadline:
                    row = poll_conn.execute(
                        "SELECT state FROM replay_sessions WHERE session_id = ?",
                        (sid,),
                    ).fetchone()
                    session_state = row[0] if row else None
                    if session_state in {"complete", "failed", "interrupted"}:
                        break
                    _time.sleep(0.02)
                assert session_state == "complete"
                cursor = max(ids_seen)
                result["cursor"] = cursor

                # --- Reconnect with Last-Event-ID: no new upstream call ---
                calls_before = transport.streaming_calls
                header, data = _raw_post(
                    host,
                    port,
                    headers={HEADER_LAST_EVENT_ID: f"{sid}:{cursor}"},
                    body="",
                )
                result["resume_status"] = header["__status__"]
                result["resume_flag"] = header.get(HEADER_REPLAY)
                result["resume_body"] = data
                result["calls_after_resume"] = transport.streaming_calls - calls_before

                # --- Plain request without cursor starts fresh (legacy) ---
                calls_before = transport.streaming_calls
                _raw_post(host, port)
                result["calls_after_plain"] = transport.streaming_calls - calls_before
            except Exception as exc:  # surfaced on the main thread below
                result["thread_error"] = exc
            finally:
                poll_conn.close()
                state["loop"].call_soon_threadsafe(state["app"]._shutdown_event.set)

        async def serve() -> None:
            app = create_app(_e2e_config(tmp_path), extensions=extensions)
            serve_task = asyncio.create_task(run_gateway(app, "127.0.0.1", 0))
            while not getattr(app, "port", None):
                await asyncio.sleep(0.01)
            state["app"] = app
            state["port"] = app.port
            state["loop"] = asyncio.get_running_loop()
            thread = threading.Thread(target=client_thread)
            thread.start()
            ready.set()
            await serve_task
            thread.join(timeout=5)
            state["thread_alive"] = thread.is_alive()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(serve())
        finally:
            loop.close()

        assert "thread_error" not in result, result.get("thread_error")
        assert not state.get("thread_alive"), "client thread timed out"
        assert result["resume_status"] == 200
        assert result["resume_flag"] == "resumed"
        assert result["calls_after_resume"] == 0
        assert result["calls_after_plain"] == 1

        body = result["resume_body"].decode()
        seqs = [int(m.group(1)) for m in re.finditer(r"id: rs_[0-9a-f]+:(\d+)", body)]
        assert seqs, "resume carried no events"
        assert seqs == sorted(set(seqs))
        assert min(seqs) == result["cursor"] + 1
        assert "data: [DONE]" in body
