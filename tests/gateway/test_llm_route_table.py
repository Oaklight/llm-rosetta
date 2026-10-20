"""The chat route table lives in the model type registry (#1110).

``create_app`` used to restate the five chat paths next to the ``llm``
descriptor, which also carried the same five ``RouteSpec`` entries.  The
descriptor is now the only table: registration walks it and selects a handler
by ``source_format``, failing at startup when a route names no handler.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

import llm_rosetta.gateway.app as app_module
from llm_rosetta._vendor.httpserver import App
from llm_rosetta._vendor.httpserver import JSONResponse
from llm_rosetta.gateway.config import GatewayConfig
from llm_rosetta.gateway.model_types import get_model_type


class _CapturingApp:
    """Stands in for the httpserver app: records what was registered."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.handlers: dict[str, Any] = {}

    def route(self, path: str, methods: list[str] | None = None) -> Any:
        def _register(handler: Any) -> Any:
            self.paths.append(path)
            self.handlers[path] = handler
            return handler

        return _register


def _minimal_cfg() -> GatewayConfig:
    return GatewayConfig({"providers": {}, "server": {"open_on_no_keys": True}})


def test_the_descriptor_names_every_handler() -> None:
    descriptor = get_model_type("llm")
    assert descriptor is not None
    declared = {spec.source_format for spec in descriptor.routes}
    assert declared == set(app_module._LLM_ROUTE_HANDLERS)


def test_registration_walks_the_descriptor() -> None:
    descriptor = get_model_type("llm")
    assert descriptor is not None
    app = _CapturingApp()

    app_module._register_llm_routes(cast(App, app))

    assert app.paths == [spec.path for spec in descriptor.routes]


def test_an_unknown_source_format_fails_fast(monkeypatch) -> None:
    monkeypatch.delitem(app_module._LLM_ROUTE_HANDLERS, "messages")

    with pytest.raises(RuntimeError, match="messages"):
        app_module._register_llm_routes(cast(App, _CapturingApp()))


@pytest.mark.asyncio
async def test_create_app_registers_chat_routes_from_the_registry(
    monkeypatch,
) -> None:
    called: list[Any] = []
    monkeypatch.setattr(
        app_module, "_register_llm_routes", lambda app: called.append(app)
    )

    await app_module.create_app(_minimal_cfg())

    assert len(called) == 1


@pytest.mark.asyncio
async def test_path_parameters_reach_the_handler() -> None:
    seen: dict[str, str] = {}

    async def handler(request: Any, model_path: str = "") -> JSONResponse:
        seen["model_path"] = model_path
        return JSONResponse({"ok": True})

    routed = app_module._llm_route_handler(handler)

    await routed(object(), model_path="gemini-2.5-pro")

    assert seen["model_path"] == "gemini-2.5-pro"
