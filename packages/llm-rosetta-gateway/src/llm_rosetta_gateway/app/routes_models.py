"""Model listing: the OpenAI- and Google-shaped ``/models`` responses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    pass

from llm_rosetta_gateway._vendor.httpserver import (
    JSONResponse,
    Response,
)

from ..observability.logging import get_logger
from ..model_types import retired_type_message


from . import _state

logger = get_logger()


def _type_filter(request: Any) -> tuple[str | None, Response | None]:
    """The ``?type=`` filter of a listing, or a 400 for a retired spelling.

    Returns:
        ``(filter, None)`` where *filter* is ``None`` for no filter or ``all``;
        or ``(None, response)`` when the value is a retired model type
        (``embedding``, ``decision``), answered with its replacement.
    """
    type_vals = getattr(request, "query_params", {}).get("type")
    if not type_vals:
        return None, None
    value = type_vals[0] if isinstance(type_vals, list) else type_vals
    if value == "all":
        return None, None
    retired = retired_type_message(value)
    if retired is not None:
        return None, JSONResponse(dict(error=retired), status_code=400)
    return value, None


def _visible(names: list[str]) -> list[str]:
    """The names the requesting key may see.

    A key scoped to a model reads the models it can call, not the deployment's
    catalogue; a key without a scope reads all of them.  The same rule the
    chat route enforces decides what a listing answers.
    """
    from ..middleware.auth import api_key_context_var
    from .routes_llm import visible_models

    return visible_models(api_key_context_var.get(), names)


async def handle_list_models(request: Any) -> Response:
    """List configured models in a format compatible with OpenAI and Anthropic SDKs.

    Supports an optional ``?type=`` query parameter to filter by model type:

    * ``type=llm`` — only LLM models (default pool)
    * ``type=embeddings`` — only embeddings models (the retired
      ``type=embedding`` is answered with 400 and its replacement)
    * ``type=rerank`` — only rerank models
    * ``type=all`` or omitted — all model types
    """
    assert _state._config is not None

    # Parse optional type filter from query params
    type_filter, rejected = _type_filter(request)
    if rejected is not None:
        return rejected

    data = []

    # LLM models (main pool)
    if type_filter is None or type_filter == "llm":
        for name in _visible(sorted(_state._config.models.keys())):
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
        for name in _visible(sorted(_state._config.embeddings_models.keys())):
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
        for name in _visible(sorted(_state._config.rerank_models.keys())):
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
    type_filter, rejected = _type_filter(request)
    if rejected is not None:
        return rejected

    models_list: list[dict[str, Any]] = []

    if type_filter is None or type_filter == "llm":
        for name in _visible(sorted(_state._config.models.keys())):
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
        for name in _visible(sorted(_state._config.embeddings_models.keys())):
            models_list.append(
                {
                    "name": f"models/{name}",
                    "displayName": name,
                    "supportedGenerationMethods": ["embedContent"],
                }
            )

    if type_filter is None or type_filter == "rerank":
        for name in _visible(sorted(_state._config.rerank_models.keys())):
            models_list.append(
                {
                    "name": f"models/{name}",
                    "displayName": name,
                    "supportedGenerationMethods": ["rerank"],
                }
            )

    return JSONResponse({"models": models_list})
