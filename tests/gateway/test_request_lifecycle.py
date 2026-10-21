"""One instrumented lifecycle for chat and typed routes (#1110).

The lifecycle owns telemetry, profiling, error dumps, deferred dumps and the
circuit breaker.  These tests drive it through the typed-route wrapper, which
used to have its own copy of that stack (and recorded a second ledger row per
request because the handler recorded one too).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

import llm_rosetta.gateway.app as app_module
from llm_rosetta.gateway.app import _state
from llm_rosetta._vendor.httpserver import App, JSONResponse
from llm_rosetta.gateway.config import GatewayConfig
from llm_rosetta.gateway.lifecycle import begin_typed, finish_typed
from llm_rosetta.gateway.model_types import (
    ModelTypeDescriptor,
    RouteSpec,
    _reset_registry,
    register_model_type,
)
from llm_rosetta.gateway.ops import OpsContext
from llm_rosetta.observability import PersistenceManager, RequestLog
from llm_rosetta.observability.usage.normalize import MODEL_TYPE_FIELDS

FAKE_TYPE = "fake"
FAKE_PROVIDER = "fake-provider"


def _config(server: dict[str, Any] | None = None) -> GatewayConfig:
    return GatewayConfig(
        {
            "providers": {},
            "models": {},
            "server": server or {"open_on_no_keys": True},
        }
    )


async def _typed_route(request: Any, config: GatewayConfig) -> Any:
    """A typed handler: resolve, ask the breaker, then report usage."""
    calls = getattr(request.app, "handler_calls", None)
    if calls is not None:
        calls.append("resolved")
    rejection = begin_typed(
        request,
        source_format="fake-source",
        target_format="fake-target",
        provider_name=FAKE_PROVIDER,
    )
    if rejection is not None:
        return rejection
    await finish_typed(
        request,
        model="fake-model",
        source_provider="fake-source",
        target_provider="fake-target",
        provider_name=FAKE_PROVIDER,
        status_code=200,
        duration_ms=1.0,
        error_detail=None,
        model_type=FAKE_TYPE,
        usage={"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    )
    return JSONResponse({"ok": True})


async def _exploding_route(request: Any, config: GatewayConfig) -> Any:
    raise RuntimeError("boom")


class _CapturingApp:
    """Stands in for the httpserver app: records the routes registered."""

    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}

    def route(self, path: str, methods: list[str] | None = None) -> Any:
        def _register(handler: Any) -> Any:
            self.handlers[path] = handler
            return handler

        return _register


@pytest.fixture()
async def pm(tmp_path):
    manager = await PersistenceManager.create(str(tmp_path / "gateway.db"))
    yield manager
    await manager.close()


def _register(monkeypatch: Any, name: str, path: str, handler: Any) -> None:
    """Register a stand-in type, with ledger columns so its rows can be written."""
    monkeypatch.setitem(MODEL_TYPE_FIELDS, name, MODEL_TYPE_FIELDS["llm"])
    register_model_type(
        ModelTypeDescriptor(
            name=name,
            routes=[RouteSpec(path)],
            formats=["fake"],
            pipeline=lambda: handler,
        )
    )


@pytest.fixture(autouse=True)
def fake_type(monkeypatch):
    _reset_registry()
    _register(monkeypatch, FAKE_TYPE, "/v1/fake", _typed_route)
    yield
    _reset_registry()


def _request(pm: PersistenceManager, path: str = "/v1/fake") -> MagicMock:
    request = MagicMock()
    request.json.return_value = {"model": "fake-model"}
    request.headers = {}
    request.path = path
    request.state = SimpleNamespace()
    request_log = RequestLog(persistence=pm)
    request.app = SimpleNamespace(
        persistence=pm,
        request_log=request_log,
        metrics=None,
        ops_ctx=OpsContext(request_log=request_log, metrics=None, persistence=pm),
        profiler_state=None,
        capture_state=None,
        handler_calls=[],
    )
    return request


def _typed_handler(config: GatewayConfig) -> Any:
    """The wrapper `_register_non_llm_routes` would install for the fake type."""
    app = _CapturingApp()
    _state._config = config
    app_module._register_non_llm_routes(cast(App, app), config)
    return app.handlers["/v1/fake"]


class TestTypedRouteLifecycle:
    async def test_one_ledger_row_per_request(self, pm):
        """The handler reports usage; the lifecycle writes the row — once."""
        handler = _typed_handler(_config())
        request = _request(pm)

        response = await handler(request)

        assert response.status_code == 200
        rows, total = await pm.query_log_entries()
        assert total == 1
        row = rows[0]
        assert row["id"] == request.state.log_entry_id
        assert row["model"] == "fake-model"
        assert row["model_type"] == FAKE_TYPE
        assert row["source_provider"] == "fake-source"
        assert row["target_provider"] == "fake-target"
        assert row["target_provider_name"] == FAKE_PROVIDER
        assert row["input_tokens"] == 5
        assert row["output_tokens"] == 7
        assert row["total_tokens"] == 12
        assert request.app.handler_calls == ["resolved"]

    async def test_an_exception_is_dumped_with_the_log_row_id(self, pm, monkeypatch):
        _register(monkeypatch, "exploding", "/v1/exploding", _exploding_route)
        app = _CapturingApp()
        config = _config()
        _state._config = config
        app_module._register_non_llm_routes(cast(App, app), config)
        request = _request(pm, path="/v1/exploding")

        response = await app.handlers["/v1/exploding"](request)

        assert response.status_code == 500
        dumps, total = await pm.query_error_dumps()
        assert total == 1
        assert dumps[0]["request_log_id"] == request.state.log_entry_id
        assert dumps[0]["error_phase"] == "handler"
        log_rows, log_total = await pm.query_log_entries()
        assert log_total == 1
        assert log_rows[0]["id"] == request.state.log_entry_id
        assert log_rows[0]["status_code"] == 500

    async def test_profiling_covers_typed_routes(self, pm):
        state = _FakeProfilerState()
        handler = _typed_handler(_config())
        request = _request(pm)
        request.app.profiler_state = state

        await handler(request)

        assert state.started == 1
        assert len(state.stored) == 1
        stored = state.stored[0]
        assert stored["model"] == "fake-model"
        assert stored["source"] == "fake-source"
        assert stored["target"] == "fake-target"
        assert stored["is_stream"] is False
        assert isinstance(stored["request_id"], str) and stored["request_id"]
        assert state.released == 1


class TestTypedRouteCircuitBreaker:
    async def test_an_open_breaker_rejects_before_the_handler_runs(self, pm):
        config = _config(
            {
                "open_on_no_keys": True,
                "circuit_breaker": {
                    "enabled": True,
                    "error_threshold": 1,
                    "cooldown_seconds": 60,
                },
            }
        )
        breaker = config.circuit_breaker_registry.get_or_create(FAKE_PROVIDER)
        breaker.record_failure()
        assert breaker.failure_count == 1
        handler = _typed_handler(config)
        request = _request(pm)

        response = await handler(request)

        assert response.status_code == 503
        assert "circuit breaker open" in response.body.decode()
        assert request.app.handler_calls == ["resolved"]
        # A rejected request is not a new upstream failure: the cooldown
        # must not grow while the breaker is doing its job.
        assert breaker.failure_count == 1
        rows, total = await pm.query_log_entries()
        assert total == 1
        assert rows[0]["status_code"] == 503

    async def test_a_failed_upstream_trips_the_breaker(self, pm, monkeypatch):
        config = _config(
            {
                "open_on_no_keys": True,
                "circuit_breaker": {
                    "enabled": True,
                    "error_threshold": 1,
                    "cooldown_seconds": 60,
                },
            }
        )
        request = _request(pm)

        async def _failing(request: Any, config: GatewayConfig) -> Any:
            rejection = begin_typed(
                request,
                source_format="fake-source",
                target_format="fake-target",
                provider_name=FAKE_PROVIDER,
            )
            if rejection is not None:
                return rejection
            return JSONResponse({"error": "bad gateway"}, status_code=502)

        app = _CapturingApp()
        _state._config = config
        _register(monkeypatch, "failing", "/v1/failing", _failing)
        app_module._register_non_llm_routes(cast(App, app), config)
        request.path = "/v1/failing"

        response = await app.handlers["/v1/failing"](request)

        assert response.status_code == 502
        breaker = config.circuit_breaker_registry.get_or_create(FAKE_PROVIDER)
        assert breaker.failure_count == 1
        assert breaker.state.value == "OPEN"


class _FakeProfiler:
    def __init__(self) -> None:
        self.is_running = False

    def start(self) -> None:
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False


class _FakeProfilerState:
    def __init__(self) -> None:
        self.started = 0
        self.stored: list[dict[str, Any]] = []
        self.released = 0
        self.remaining = 0
        self.enabled = True

    def should_profile(self) -> bool:
        return True

    def create_profiler(self) -> _FakeProfiler:
        self.started += 1
        return _FakeProfiler()

    def store_result(self, profiler: Any, **kwargs: Any) -> None:
        self.stored.append(kwargs)

    def release(self) -> None:
        self.released += 1
