"""Decision proxy handler with IR-based conversion.

Proxies ``/v1/decision`` and ``/v1/systemone`` requests to upstream
decision providers (currently TypeSafe System One).  No streaming —
decision requests return a single JSON response.

Every request and response is converted through the IR via
:class:`~llm_rosetta.converters.decision.pipeline.DecisionConversionPipeline`,
including when the source format (inferred from the request) and the target
format (declared on the provider config) are the same.  The request flow is
the shared one in :mod:`pipelines.typed`; this module supplies the decision
hooks.

Registered via the model type registry in ``model_types.py``.

Every request that reaches an upstream is recorded in the usage ledger with
``model_type="decision"``.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta_gateway._vendor.httpserver import Response
from llm_rosetta.converters.decision.pipeline import (
    DecisionConversionPipeline,
    response_usage,
)

from ..config import GatewayConfig, TypedRoute
from .typed import TypedHandler, detect_source_format, handle_typed_request


def _detect_decision_source_format(
    request: Any, body: dict[str, Any], config: GatewayConfig
) -> str:
    """Infer the source decision format through the type descriptor.

    The detector lives in :mod:`gateway.source_formats` and is attached to the
    ``decision`` descriptor; this wrapper keeps the handler-local name used by
    the tests.
    """
    return detect_source_format(request, body, config, "decision")


def _resolve_route(config: GatewayConfig, model: str) -> TypedRoute | None:
    """Resolve the decision provider, or ``None`` when the model is unknown."""
    try:
        return config.resolve_decision(model)
    except KeyError:
        return None


def _response_post(body: dict[str, Any], model: str) -> None:
    """Echo the client's model name when the upstream response omits it."""
    if not body.get("model"):
        body["model"] = model


_DECISION = TypedHandler(
    model_type="decision",
    build_pipeline=DecisionConversionPipeline,
    resolve=_resolve_route,
    configured_models=lambda config: config.decision_models.keys(),
    response_usage=response_usage,
    not_found_label="decision model",
    response_post=_response_post,
)


async def handle_decision(request: Any, config: GatewayConfig) -> Response:
    """Proxy a decision request with IR-based conversion."""
    return await handle_typed_request(request, config, _DECISION)
