"""Shared model-list utilities for ALCF provider shims.

ALCF's ``/resource_server/{cluster}/models`` endpoint returns a bare
JSON array of model objects (not wrapped in ``{data: [...]}``) and may
include models from multiple frameworks.  Each cluster shim filters
by its target framework.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ModelListTransform = Callable[[list[dict[str, Any]]], tuple[list[str], dict[str, str]]]


def make_alcf_model_list_transform(
    framework: str,
) -> ModelListTransform:
    """Return a model_list_transform that filters by *framework*.

    Args:
        framework: Only include models whose ``framework`` field matches
            this value (e.g. ``"vllm"``, ``"api"``).
    """

    def model_list_transform(
        raw_entries: list[dict[str, Any]],
    ) -> tuple[list[str], dict[str, str]]:
        ids: list[str] = []
        for m in raw_entries:
            fw = m.get("framework", "")
            if fw != framework:
                continue
            model_id = m.get("id", "")
            if model_id:
                ids.append(model_id)
        return ids, {}

    return model_list_transform
