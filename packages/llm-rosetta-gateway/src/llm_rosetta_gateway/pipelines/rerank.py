"""Rerank proxy handler with IR-based conversion.

Proxies ``/v1/rerank`` and ``/v2/rerank`` requests to upstream rerank
providers, converting every request and response through the IR — between
Jina, Cohere, and Voyage formats, and even when the source and target format
are the same.  The request flow is the shared one in :mod:`pipelines.typed`;
this module supplies the rerank hooks.

Every request that reaches an upstream is recorded in the usage ledger with
``model_type="rerank"``.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta_gateway._vendor.httpserver import Response
from llm_rosetta.converters.rerank.pipeline import (
    RerankConversionPipeline,
    response_usage,
)

from ..config import GatewayConfig, TypedRoute
from .typed import TypedHandler, detect_source_format, handle_typed_request


def _detect_source_format(
    request: Any, body: dict[str, Any], config: GatewayConfig
) -> str:
    """Infer the source rerank format through the type descriptor.

    The detector lives in :mod:`gateway.source_formats` and is attached to the
    ``rerank`` descriptor; this wrapper keeps the handler-local name used by
    the tests.
    """
    return detect_source_format(request, body, config, "rerank")


def _resolve_route(config: GatewayConfig, model: str) -> TypedRoute | None:
    """Resolve the rerank provider, or ``None`` when the model is unknown."""
    try:
        return config.resolve_rerank(model)
    except KeyError:
        return None


_RERANK = TypedHandler(
    model_type="rerank",
    build_pipeline=RerankConversionPipeline,
    resolve=_resolve_route,
    configured_models=lambda config: config.rerank_models.keys(),
    response_usage=response_usage,
    not_found_label="rerank model",
    # Rerank marks an empty upstream body as a passthrough; embedding and
    # decision return the raw body instead.
    empty_body_is_passthrough=True,
)


async def handle_rerank(request: Any, config: GatewayConfig) -> Response:
    """Proxy a rerank request with IR-based conversion."""
    return await handle_typed_request(request, config, _RERANK)
