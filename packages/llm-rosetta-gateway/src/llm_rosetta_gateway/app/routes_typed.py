"""Route registration for the typed (non-LLM) model types."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    pass

from llm_rosetta_gateway._vendor.httpserver import (
    App,
    Response,
    StreamingResponse,
)

from ..config import GatewayConfig
from ..proxy.lifecycle import instrumented_handler
from ..observability.logging import get_logger
from ..pipelines.decision import handle_decision
from ..pipelines.embeddings import handle_embeddings
from ..pipelines.rerank import handle_rerank


logger = get_logger()

#: The handler of each built-in typed model type, by descriptor name.  The
#: descriptors live in ``model_types`` (a lower layer) and name no handler; a
#: type registered from outside gives its descriptor a ``pipeline`` callable
#: instead, and that wins.
BUILTIN_HANDLERS: dict[str, Callable[..., Any]] = {
    "embeddings": handle_embeddings,
    "rerank": handle_rerank,
    "decisions": handle_decision,
}


def _handler_of(desc: Any) -> Callable[..., Any] | None:
    """The pipeline handler of a model type, or ``None`` when it has none."""
    if desc.pipeline is not None:
        return desc.pipeline()
    return BUILTIN_HANDLERS.get(desc.name)


def _register_non_llm_routes(app: App, config: GatewayConfig) -> None:
    """Register HTTP routes for all non-LLM model types from the registry.

    For each non-LLM descriptor, resolves the handler (the descriptor's own
    ``pipeline`` callable, else the entry of :data:`BUILTIN_HANDLERS`) and
    registers its routes.  A type with neither is skipped with a warning — the
    ``llm`` type is not one of them: its routes come from the descriptor's
    ``RouteSpec`` list via :func:`_register_llm_routes`.

    The handler runs inside :func:`~.lifecycle.instrumented_handler`, the same
    lifecycle the chat routes use, so telemetry, profiling, error dumps,
    deferred dumps and the circuit breaker cannot be missing from one side.
    """
    from ..model_types import all_model_types

    for desc in all_model_types():
        if desc.name == "llm":
            continue

        raw_handler = _handler_of(desc)
        if raw_handler is None:
            logger.warning(
                "Model type %r has no pipeline — routes skipped",
                desc.name,
            )
            continue

        handler = _make_typed_handler(raw_handler, desc.name)

        for route_spec in desc.routes:
            app.route(route_spec.path, methods=route_spec.methods)(handler)


def _make_typed_handler(raw_handler: Callable, type_name: str) -> Callable:
    """Wrap a typed pipeline handler in the shared request lifecycle."""

    async def _handler(request: Any) -> Response | StreamingResponse:
        config = request.app.gateway_config
        assert config is not None
        return await instrumented_handler(
            request,
            config,
            raw_handler,
            source_format=type_name,
            model_type=type_name,
            provider_name=type_name,
            error_phase="handler",
            error_format="chat_completions",
        )

    return _handler
