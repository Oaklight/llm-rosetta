"""Tests for deferred error-dump writes (#964).

A dump names the request's log entry, but that row is written when the request
finishes — after the error handling that captured the dump.  With the
``error_dumps.request_log_id`` foreign key enforced, writing the dump on the
spot fails the constraint and ``dump_error`` drops it, so dumps were silently
lost on any database created with that FK.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

import llm_rosetta.gateway.app as app_module
from llm_rosetta._vendor.httpserver import JSONResponse
from llm_rosetta.gateway.deferred_dumps import DeferredDumpStore
from llm_rosetta.gateway.ops import OpsContext
from llm_rosetta.observability.error_dump import dump_error
from llm_rosetta.observability.persistence import PersistenceManager
from llm_rosetta.observability.request_log import RequestLog
from llm_rosetta.routing import ResolvedRoute

# ---------------------------------------------------------------------------
# DeferredDumpStore
# ---------------------------------------------------------------------------


class _RecordingStore:
    """Stand-in persistence manager that records the calls it receives."""

    def __init__(self, *, fail: str | None = None) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self._fail = fail

    async def insert_dump_body(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("insert_dump_body", args, kwargs))
        if self._fail == "insert_dump_body":
            raise RuntimeError("boom")

    async def insert_error_dump(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("insert_error_dump", args, kwargs))
        if self._fail == "insert_error_dump":
            raise RuntimeError("boom")


class TestDeferredDumpStore:
    def test_queues_until_flush(self):
        store = _RecordingStore()
        deferred = DeferredDumpStore(store)

        async def run():
            await deferred.insert_dump_body("hash", b"data", 4)
            await deferred.insert_error_dump(dump_id="d1", request_log_id="e1")
            assert store.calls == []  # nothing written yet
            assert deferred.pending == 2

        asyncio.run(run())

    def test_flush_replays_in_order(self):
        store = _RecordingStore()
        deferred = DeferredDumpStore(store)

        async def run():
            await deferred.insert_dump_body("hash", b"data", 4)
            await deferred.insert_error_dump(dump_id="d1", request_log_id="e1")
            await deferred.flush()
            return store.calls

        calls = asyncio.run(run())
        assert [c[0] for c in calls] == ["insert_dump_body", "insert_error_dump"]
        assert calls[0][1] == ("hash", b"data", 4)
        assert calls[1][2] == {"dump_id": "d1", "request_log_id": "e1"}

    def test_writes_after_flush_pass_straight_through(self):
        store = _RecordingStore()
        deferred = DeferredDumpStore(store)

        async def run():
            await deferred.flush()
            await deferred.insert_error_dump(dump_id="d2")
            assert deferred.pending == 0

        asyncio.run(run())
        assert [c[0] for c in store.calls] == ["insert_error_dump"]

    def test_flush_is_idempotent(self):
        store = _RecordingStore()
        deferred = DeferredDumpStore(store)

        async def run():
            await deferred.insert_error_dump(dump_id="d1")
            await deferred.flush()
            await deferred.flush()

        asyncio.run(run())
        assert len(store.calls) == 1

    def test_failing_write_does_not_raise_and_rest_still_run(self):
        store = _RecordingStore(fail="insert_dump_body")
        deferred = DeferredDumpStore(store)

        async def run():
            await deferred.insert_dump_body("hash", b"data", 4)
            await deferred.insert_error_dump(dump_id="d1")
            await deferred.flush()  # must not raise

        asyncio.run(run())
        assert [c[0] for c in store.calls] == [
            "insert_dump_body",
            "insert_error_dump",
        ]


# ---------------------------------------------------------------------------
# _proxy_handler wiring — the #964 regression
# ---------------------------------------------------------------------------


def _config():
    from llm_rosetta.gateway.middleware.circuit_breaker import (
        CircuitBreakerConfig,
        CircuitBreakerRegistry,
    )

    class _Config:
        models = {"test-model": "test-provider"}
        error_dumps_enabled = True
        provider_key_affinity: dict[str, bool] = {}
        circuit_breaker_registry = CircuitBreakerRegistry(
            CircuitBreakerConfig(enabled=False)
        )

        def resolve(self, source_provider, model, **kwargs):
            return (
                ResolvedRoute(
                    source_provider=source_provider,
                    target_provider="chat_completions",
                    provider_name="test-provider",
                ),
                MagicMock(),
            )

    return _Config()


async def _run_proxy_handler(monkeypatch: Any, pm: PersistenceManager) -> str:
    """Drive one failing request through the real ``_proxy_handler``.

    The fake handler dumps the way the real error paths do — before returning —
    and the real ``_record_proxy_op`` writes the request-log row afterwards.
    """
    captured: dict[str, Any] = {}

    async def _fake_handle_non_streaming(
        route, provider_info, body, *, persistence=None, entry_id=None, **kwargs
    ):
        captured["entry_id"] = entry_id
        # What _upstream_error_passthrough does on an upstream 4xx.
        await dump_error(
            persistence,
            request_body=body,
            response_text='{"error": "boom"}',
            status_code=400,
            error_phase="upstream",
            request_log_id=entry_id,
        )
        return JSONResponse({"error": {"message": "boom"}}, status_code=400), {}

    monkeypatch.setattr(app_module, "_config", _config())
    monkeypatch.setattr(app_module, "handle_non_streaming", _fake_handle_non_streaming)

    request = MagicMock()
    request.headers = {}
    request.json.return_value = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    request.app.gateway_config = app_module._config
    request.app.metadata_store = MagicMock()
    request.app.metrics = None
    request.app.request_log = RequestLog(persistence=pm)
    request.app.persistence = pm
    request.app.ops_ctx = OpsContext(
        ops_log=None,
        request_log=request.app.request_log,
        metrics=None,
        persistence=pm,
    )
    request.app.transport = MagicMock()
    request.app.profiler_state = None
    request.app.capture_state = None
    await app_module._proxy_handler(request, "chat_completions")

    return captured["entry_id"]


def test_dump_is_written_once_the_request_log_row_exists(tmp_path, monkeypatch):
    """The dump survives the FK, and keeps its request_log_id (#964)."""

    async def run():
        pm = await PersistenceManager.create(str(tmp_path))
        entry_id = await _run_proxy_handler(monkeypatch, pm)
        entries, total = await pm.query_error_dumps()
        log_entries, _ = await pm.query_log_entries()
        await pm.close()
        return entry_id, entries, total, log_entries

    entry_id, entries, total, log_entries = asyncio.run(run())

    assert [e["id"] for e in log_entries] == [entry_id]
    assert total == 1
    assert entries[0]["request_log_id"] == entry_id
    assert entries[0]["error_phase"] == "upstream"


def test_flush_runs_even_if_the_log_write_fails(monkeypatch):
    """A failing ``_record_proxy_op`` must not swallow the queued dumps.

    The queued dumps are flushed in a ``finally`` around the log write, so a log
    write that raises still replays them — they fail their FK and log at
    ``WARNING``, which is the point — instead of going out of scope in silence,
    which is the defect this module exists to fix (#964).
    """
    store = _RecordingStore()

    async def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("log write failed")

    async def _fake_handle_non_streaming(
        route, provider_info, body, *, persistence=None, entry_id=None, **kwargs
    ):
        await dump_error(
            persistence,
            request_body=body,
            response_text='{"error": "boom"}',
            status_code=400,
            error_phase="upstream",
            request_log_id=entry_id,
        )
        return JSONResponse({"error": {"message": "boom"}}, status_code=400), {}

    monkeypatch.setattr(app_module, "_config", _config())
    monkeypatch.setattr(app_module, "handle_non_streaming", _fake_handle_non_streaming)
    monkeypatch.setattr(app_module, "_record_proxy_op", _boom)

    request = MagicMock()
    request.headers = {}
    request.json.return_value = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
    }
    request.app.gateway_config = app_module._config
    request.app.metadata_store = MagicMock()
    request.app.metrics = None
    request.app.request_log = None
    request.app.persistence = store
    request.app.ops_ctx = None
    request.app.transport = MagicMock()
    request.app.profiler_state = None
    request.app.capture_state = None

    with pytest.raises(RuntimeError, match="log write failed"):
        asyncio.run(app_module._proxy_handler(request, "chat_completions"))

    # The queued writes were replayed despite the log write raising.
    assert [c[0] for c in store.calls] == ["insert_dump_body", "insert_error_dump"]
