"""Embeddings proxy handler with IR-based conversion.

Proxies ``/v1/embeddings`` requests to upstream embedding providers.
When the model's provider declares an embedding endpoint (``embeddings_format``
/ ``embeddings_shim``, or the legacy ``embedding_providers`` section), every
request and response goes through the IR — the same rule the chat routes
follow — so a request is converted between OpenAI, Cohere, Jina, Voyage and
Google GenAI formats even when the source and target format are the same.
There is no chat-provider passthrough: an unconfigured model gets a 404 rather
than a chat-shaped request forwarded to a provider that never declared an
embedding endpoint.

The request flow itself is the shared one in :mod:`pipelines.typed`; this
module supplies the embedding hooks — the IR pipeline, the upstream model
rename, and Google's model-in-URL path with its batch variant.

Every request that reaches an upstream is recorded in the usage ledger with
``model_type="embeddings"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import Response
from llm_rosetta.converters.embedding.pipeline import (
    EmbeddingConversionPipeline,
    get_embedding_converter,
    response_usage,
)

from ..config import GatewayConfig, TypedRoute
from ..transport import ProviderInfo
from .typed import TypedHandler, detect_source_format, handle_typed_request


def _detect_embedding_source(
    request: Any, body: dict[str, Any], config: GatewayConfig
) -> str:
    """Infer the source embedding format through the type descriptor.

    The detector lives in :mod:`gateway.source_formats` and is attached to the
    ``embedding`` descriptor, so it is the same object the descriptor exposes
    and stays testable without a handler.
    """
    return detect_source_format(request, body, config, "embeddings")


@dataclass
class _ResolvedEmbedding:
    """One resolved embedding call, as the URL-construction tests read it."""

    provider_info: ProviderInfo
    upstream_url: str
    provider_name: str
    pipeline: EmbeddingConversionPipeline
    target_format: str = "openai"
    source_format: str = "openai"
    #: The configured endpoint path.  It may carry ``{model}`` (Google puts
    #: the model in the URL) and is used to build the batch variant after the
    #: request has been converted.
    path: str = ""


def _resolve_route(config: GatewayConfig, model: str) -> TypedRoute | None:
    """Resolve the embedding provider, or ``None`` when the model is unknown.

    Embedding traffic is routed only through the embedding pool.  There is no
    chat-provider fallback: an unconfigured model returns 404 instead of
    forwarding a chat-shaped request to a provider that never declared an
    embedding endpoint.
    """
    try:
        return config.resolve_embedding(model)
    except KeyError:
        return None


def _resolve_embedding_provider(
    config: GatewayConfig,
    model: str,
    body: dict[str, Any],
    request: Any = None,
) -> _ResolvedEmbedding | None:
    """Resolve one embedding call, or ``None`` when the model is unknown."""
    route = _resolve_route(config, model)
    if route is None:
        return None

    source_format = _detect_embedding_source(request, body, config)
    upstream_model = config.model_upstream_names.get(model)
    if upstream_model:
        body["model"] = upstream_model
    return _ResolvedEmbedding(
        provider_info=route.provider_info,
        upstream_url=route.provider_info.upstream_url(upstream_model or model),
        provider_name=route.provider_name,
        target_format=route.format,
        source_format=source_format,
        path=route.path,
        pipeline=EmbeddingConversionPipeline(source_format, route.format),
    )


def _prepare_body(body: dict[str, Any], model: str, config: GatewayConfig) -> None:
    """Send the provider's upstream name for the model, when it declares one."""
    upstream_model = config.model_upstream_names.get(model)
    if upstream_model:
        body["model"] = upstream_model


def _upstream_url(
    route: TypedRoute,
    body: dict[str, Any],
    upstream_model: str,
    config: GatewayConfig,
) -> str:
    """The upstream URL, after conversion.

    Google carries the model in the path and has a distinct batch suffix; both
    are decided after conversion, because the converted body is what says
    single (``content``) or batch (``requests``).
    """
    if route.format != "google_generate":
        return route.provider_info.upstream_url(upstream_model)
    path = route.path.replace("{model}", upstream_model)
    if "requests" in body and path.endswith(":embedContent"):
        path = path[: -len(":embedContent")] + ":batchEmbedContents"
    return route.provider_info.base_url + path


def _response_post(body: dict[str, Any], model: str) -> None:
    """Echo the client's model name when the upstream response omits it."""
    if not body.get("model"):
        body["model"] = model


def _upstream_error(format_name: str, body: dict[str, Any] | None) -> str | None:
    """Ask the target dialect whether *body* is an error dressed as a success.

    Only MiniMax embeds its verdict in the body today; every other embedding
    dialect inherits ``None`` from the base converter.
    """
    return get_embedding_converter(format_name).upstream_error(body)


_EMBEDDINGS = TypedHandler(
    model_type="embeddings",
    build_pipeline=EmbeddingConversionPipeline,
    resolve=_resolve_route,
    configured_models=lambda config: config.embeddings_models.keys(),
    response_usage=response_usage,
    prepare_body=_prepare_body,
    upstream_url=_upstream_url,
    response_post=_response_post,
    upstream_error=_upstream_error,
    connection_error_message="Upstream request failed",
    log_conversion=False,
)


async def handle_embeddings(request: Any, config: GatewayConfig) -> Response:
    """Proxy an embedding request with IR-based conversion."""
    return await handle_typed_request(request, config, _EMBEDDINGS)
