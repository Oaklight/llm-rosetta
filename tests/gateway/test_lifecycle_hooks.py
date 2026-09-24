"""Tests for httpserver lifecycle hooks (TTFB, disconnect tracking)."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from llm_rosetta._vendor.httpserver import State
from llm_rosetta.gateway.middleware.request_context import (
    RequestContext,
    request_context_var,
)
from llm_rosetta.observability import MetricsCollector
from llm_rosetta.observability.request_log import RequestLog


# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------


class FakeApp:
    """Minimal app mock with metrics and request_log attributes."""

    def __init__(
        self,
        metrics: MetricsCollector | None = None,
        request_log: RequestLog | None = None,
    ):
        self.metrics = metrics
        self.request_log = request_log


class FakeRequest:
    def __init__(
        self,
        path: str = "/v1/chat/completions",
        app: FakeApp | None = None,
    ):
        self.path = path
        self.app = app or FakeApp()
        self.state = State()


class FakeResponse:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Import the hook installer
# ---------------------------------------------------------------------------

from llm_rosetta.gateway.app import _install_lifecycle_hooks  # noqa: E402


class _HookCapture:
    """Captures hooks registered via on_response_started / etc."""

    def __init__(self) -> None:
        self.on_response_started_handlers: list = []
        self.on_response_completed_handlers: list = []
        self.on_client_disconnect_handlers: list = []

    def on_response_started(self, fn: Any) -> Any:
        self.on_response_started_handlers.append(fn)
        return fn

    def on_response_completed(self, fn: Any) -> Any:
        self.on_response_completed_handlers.append(fn)
        return fn

    def on_client_disconnect(self, fn: Any) -> Any:
        self.on_client_disconnect_handlers.append(fn)
        return fn


def _build_hooks() -> _HookCapture:
    """Install lifecycle hooks onto a capture object and return it."""
    capture = _HookCapture()
    _install_lifecycle_hooks(capture)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    return capture


# ---------------------------------------------------------------------------
# TTFB via on_response_started
# ---------------------------------------------------------------------------


class TestOnResponseStarted:
    def test_computes_ttfb(self):
        capture = _build_hooks()
        assert len(capture.on_response_started_handlers) == 1
        handler = capture.on_response_started_handlers[0]

        req = FakeRequest()
        resp = FakeResponse()

        start = time.monotonic()
        ctx = RequestContext(
            request_id="r1",
            client_ip="127.0.0.1",
            api_format="openai",
            is_admin=False,
            request_start=start - 0.05,  # 50ms ago
        )

        async def run():
            tok = request_context_var.set(ctx)
            try:
                await handler(req, resp)
            finally:
                request_context_var.reset(tok)

        _run(run())
        assert hasattr(req.state, "ttfb_ms")
        assert req.state.ttfb_ms >= 50.0  # type: ignore[operator]  # ty: ignore[unsupported-operator]

    def test_skips_when_no_context(self):
        capture = _build_hooks()
        handler = capture.on_response_started_handlers[0]

        req = FakeRequest()
        resp = FakeResponse()

        async def run():
            tok = request_context_var.set(None)
            try:
                await handler(req, resp)
            finally:
                request_context_var.reset(tok)

        _run(run())
        assert not hasattr(req.state, "ttfb_ms")


# ---------------------------------------------------------------------------
# on_response_completed — writes TTFB to request log
# ---------------------------------------------------------------------------


class TestOnResponseCompleted:
    def test_writes_ttfb_to_request_log(self):
        request_log = RequestLog()
        app = FakeApp(request_log=request_log)
        capture = _build_hooks()
        handler = capture.on_response_completed_handlers[0]

        from llm_rosetta.observability.request_log import RequestLogEntry

        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=100.0,
        )
        request_log.add(entry)

        req = FakeRequest(app=app)
        req.state.ttfb_ms = 42.5  # ty: ignore[unresolved-attribute]
        req.state.log_entry_id = entry.id  # ty: ignore[unresolved-attribute]
        resp = FakeResponse()

        _run(handler(req, resp))

        result = request_log.get_entry(entry.id)
        assert result is not None
        assert result["profile"]["ttfb_ms"] == 42.5

    def test_no_op_without_entry_id(self):
        request_log = RequestLog()
        app = FakeApp(request_log=request_log)
        capture = _build_hooks()
        handler = capture.on_response_completed_handlers[0]

        req = FakeRequest(app=app)
        req.state.ttfb_ms = 10.0  # ty: ignore[unresolved-attribute]
        resp = FakeResponse()

        _run(handler(req, resp))
        # No crash, no entries modified


# ---------------------------------------------------------------------------
# on_client_disconnect
# ---------------------------------------------------------------------------


class TestOnClientDisconnect:
    def test_increments_disconnect_counter(self):
        metrics = MetricsCollector()
        app = FakeApp(metrics=metrics)
        capture = _build_hooks()
        handler = capture.on_client_disconnect_handlers[0]

        req = FakeRequest(path="/v1/chat/completions", app=app)

        ctx = RequestContext(
            request_id="r1",
            client_ip="10.0.0.1",
            api_format="openai",
            is_admin=False,
        )

        async def run():
            tok = request_context_var.set(ctx)
            try:
                await handler(req)
            finally:
                request_context_var.reset(tok)

        _run(run())
        assert metrics.total_client_disconnects == 1

    def test_marks_entry_as_disconnected(self):
        request_log = RequestLog()
        metrics = MetricsCollector()
        app = FakeApp(metrics=metrics, request_log=request_log)
        capture = _build_hooks()
        handler = capture.on_client_disconnect_handlers[0]

        from llm_rosetta.observability.request_log import RequestLogEntry

        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=True,
            status_code=200,
            duration_ms=500.0,
        )
        request_log.add(entry)

        req = FakeRequest(app=app)
        req.state.log_entry_id = entry.id  # ty: ignore[unresolved-attribute]

        ctx = RequestContext(
            request_id="r2",
            client_ip="10.0.0.2",
            api_format="openai",
            is_admin=False,
        )

        async def run():
            tok = request_context_var.set(ctx)
            try:
                await handler(req)
            finally:
                request_context_var.reset(tok)

        _run(run())

        result = request_log.get_entry(entry.id)
        assert result is not None
        assert result["profile"]["client_disconnected"] is True

    def test_works_without_metrics(self):
        app = FakeApp(metrics=None)
        capture = _build_hooks()
        handler = capture.on_client_disconnect_handlers[0]

        req = FakeRequest(app=app)

        ctx = RequestContext(
            request_id="r3",
            client_ip="10.0.0.3",
            api_format="openai",
            is_admin=False,
        )

        async def run():
            tok = request_context_var.set(ctx)
            try:
                await handler(req)
            finally:
                request_context_var.reset(tok)

        _run(run())
        # No crash — graceful no-op
