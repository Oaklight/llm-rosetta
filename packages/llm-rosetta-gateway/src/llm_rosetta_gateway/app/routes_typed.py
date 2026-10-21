"""Route registration for the typed (non-LLM) model types."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    pass

from llm_rosetta._vendor.httpserver import (
    App,
    Response,
    StreamingResponse,
)

from ..config import GatewayConfig
from ..lifecycle import instrumented_handler
from ..logging import get_logger


from . import _state

logger = get_logger()


def _register_non_llm_routes(app: App, config: GatewayConfig) -> None:
    """Register HTTP routes for all non-LLM model types from the registry.

    For each non-LLM descriptor whose ``pipeline`` is set, resolves the
    handler via ``desc.pipeline()`` (a lazy import wrapper) and registers
    its routes.  Descriptors with ``pipeline=None`` are skipped with a
    warning — this is expected only for the ``llm`` type, whose routes come
    from the descriptor's ``RouteSpec`` list via :func:`_register_llm_routes`.

    The handler runs inside :func:`~.lifecycle.instrumented_handler`, the same
    lifecycle the chat routes use, so telemetry, profiling, error dumps,
    deferred dumps and the circuit breaker cannot be missing from one side.
    """
    from ..model_types import all_model_types

    for desc in all_model_types():
        if desc.name == "llm":
            continue

        if desc.pipeline is None:
            logger.warning(
                "Model type %r has no pipeline — routes skipped",
                desc.name,
            )
            continue

        # Resolve the handler via the lazy import wrapper stored in
        # the descriptor.  The wrapper defers imports to avoid circular
        # dependencies between model_types and handler modules.
        raw_handler = desc.pipeline()
        handler = _make_typed_handler(raw_handler, desc.name)

        for route_spec in desc.routes:
            app.route(route_spec.path, methods=route_spec.methods)(handler)


def _make_typed_handler(raw_handler: Callable, type_name: str) -> Callable:
    """Wrap a typed pipeline handler in the shared request lifecycle."""

    async def _handler(request: Any) -> Response | StreamingResponse:
        assert _state._config is not None
        return await instrumented_handler(
            request,
            _state._config,
            raw_handler,
            source_format=type_name,
            model_type=type_name,
            provider_name=type_name,
            error_phase="handler",
            error_format="chat_completions",
        )

    return _handler
