"""Model-list transform registry and the built-in Jina/Cohere transforms."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


# Convention-based model list transforms registered by shim transforms modules.
# Each entry maps a shim name to a callable that converts upstream model
# entries into (display_ids, upstream_map).
ModelListTransform = Callable[[list[dict[str, Any]]], tuple[list[str], dict[str, str]]]

_model_list_transforms: dict[str, ModelListTransform] = {}


def get_model_list_transform(
    shim_name: str,
) -> ModelListTransform | None:
    """Return the model_list_transform for *shim_name*, or ``None``.

    Transforms are registered under the canonical shim name.
    """
    return _model_list_transforms.get(shim_name)


def register_model_list_transform(name: str, transform: ModelListTransform) -> None:
    """Register a model_list_transform for providers without a shim directory."""
    _model_list_transforms[name] = transform


# --------------- Built-in model list transforms ---------------


def _jina_model_list_transform(
    entries: list[dict[str, Any]],
) -> tuple[list[str], dict[str, str]]:
    """Strip the ``jina-ai/`` org prefix from Jina model IDs."""
    ids: list[str] = []
    upstream_map: dict[str, str] = {}
    for m in entries:
        raw_id = m.get("id", "")
        if not raw_id:
            continue
        display = raw_id.removeprefix("jina-ai/")
        ids.append(display)
        if display != raw_id:
            upstream_map[display] = raw_id
    return ids, upstream_map


def _cohere_model_list_transform(
    entries: list[dict[str, Any]],
) -> tuple[list[str], dict[str, str]]:
    """Parse Cohere's ``models[].name`` format."""
    ids: list[str] = []
    for m in entries:
        name = m.get("name", "")
        if name:
            ids.append(name)
    return ids, {}
