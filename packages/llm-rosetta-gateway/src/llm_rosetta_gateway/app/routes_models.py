"""Model listing: the OpenAI- and Google-shaped ``/models`` responses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    pass

from llm_rosetta_gateway._vendor.httpserver import (
    JSONResponse,
    Response,
)

from ..logging import get_logger


from . import _state

logger = get_logger()


async def handle_list_models(request: Any) -> Response:
    """List configured models in a format compatible with OpenAI and Anthropic SDKs.

    Supports an optional ``?type=`` query parameter to filter by model type:

    * ``type=llm`` — only LLM models (default pool)
    * ``type=embeddings`` — only embeddings models (the legacy
      ``type=embedding`` is accepted)
    * ``type=rerank`` — only rerank models
    * ``type=all`` or omitted — all model types
    """
    assert _state._config is not None

    # Parse optional type filter from query params
    type_filter = None
    qp = getattr(request, "query_params", {})
    type_vals = qp.get("type")
    if type_vals:
        from llm_rosetta.formats import canonical_model_type

        type_filter = type_vals[0] if isinstance(type_vals, list) else type_vals
        if type_filter == "all":
            type_filter = None
        else:
            type_filter = canonical_model_type(type_filter)

    data = []

    # LLM models (main pool)
    if type_filter is None or type_filter == "llm":
        for name in sorted(_state._config.models.keys()):
            model_route = _state._config.models[name]
            provider_name = model_route.providers[0].name
            api_standard = _state._config.provider_types.get(provider_name, "unknown")
            capabilities = _state._config.model_capabilities.get(name, ["text"])
            entry: dict[str, Any] = {
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": provider_name,
                "api_standard": api_standard,
                "capabilities": capabilities,
                "type": "llm",
                "display_name": name,
                "created_at": "1970-01-01T00:00:00Z",
            }
            if model_route.is_multi:
                entry["providers"] = [
                    {"name": p.name, "weight": p.weight} for p in model_route.providers
                ]
                entry["routing_strategy"] = "weighted_round_robin"
            data.append(entry)

    # Embedding models
    if type_filter is None or type_filter == "embeddings":
        for name in sorted(_state._config.embeddings_models.keys()):
            provider_name = _state._config.embeddings_models[name]
            api_standard = _state._config.provider_types.get(provider_name, "unknown")
            data.append(
                {
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": provider_name,
                    "api_standard": api_standard,
                    "capabilities": ["embeddings"],
                    "type": "embeddings",
                    "display_name": name,
                    "created_at": "1970-01-01T00:00:00Z",
                }
            )

    # Rerank models
    if type_filter is None or type_filter == "rerank":
        for name in sorted(_state._config.rerank_models.keys()):
            provider_name = _state._config.rerank_models[name]
            api_standard = _state._config.provider_types.get(provider_name, "unknown")
            data.append(
                {
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": provider_name,
                    "api_standard": api_standard,
                    "capabilities": ["rerank"],
                    "type": "rerank",
                    "display_name": name,
                    "created_at": "1970-01-01T00:00:00Z",
                }
            )

    all_ids = [d["id"] for d in data]
    return JSONResponse(
        {
            "object": "list",
            "data": data,
            "has_more": False,
            "first_id": all_ids[0] if all_ids else None,
            "last_id": all_ids[-1] if all_ids else None,
        }
    )


async def handle_list_models_google(request: Any) -> Response:
    """List configured models in Google GenAI SDK format.

    Supports the same ``?type=`` filter as :func:`handle_list_models`.
    """
    assert _state._config is not None

    # Parse optional type filter from query params
    type_filter = None
    qp = getattr(request, "query_params", {})
    type_vals = qp.get("type")
    if type_vals:
        from llm_rosetta.formats import canonical_model_type

        type_filter = type_vals[0] if isinstance(type_vals, list) else type_vals
        if type_filter == "all":
            type_filter = None
        else:
            type_filter = canonical_model_type(type_filter)

    models_list: list[dict[str, Any]] = []

    if type_filter is None or type_filter == "llm":
        for name in sorted(_state._config.models.keys()):
            models_list.append(
                {
                    "name": f"models/{name}",
                    "displayName": name,
                    "supportedGenerationMethods": [
                        "generateContent",
                        "streamGenerateContent",
                    ],
                }
            )

    if type_filter is None or type_filter == "embeddings":
        for name in sorted(_state._config.embeddings_models.keys()):
            models_list.append(
                {
                    "name": f"models/{name}",
                    "displayName": name,
                    "supportedGenerationMethods": ["embedContent"],
                }
            )

    if type_filter is None or type_filter == "rerank":
        for name in sorted(_state._config.rerank_models.keys()):
            models_list.append(
                {
                    "name": f"models/{name}",
                    "displayName": name,
                    "supportedGenerationMethods": ["rerank"],
                }
            )

    return JSONResponse({"models": models_list})
