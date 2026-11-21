"""Model-list transform for the SiliconFlow embedding shim.

The playground catalogue mixes every endpoint.  Only the active
``subType: embedding`` entries answer ``/embeddings``.
"""

from __future__ import annotations

from typing import Any


def model_list_transform(
    entries: list[dict[str, Any]],
) -> tuple[list[str], dict[str, str]]:
    """Keep the active embedding models.

    Args:
        entries: Upstream catalogue entries from
            ``/api/v1/playground/comprehensive/all``.

    Returns:
        ``(model_ids, upstream_map)``.  The id is ``modelName``, so no mapping
        is needed.
    """
    ids = [
        m["modelName"]
        for m in entries
        if m.get("status") != "disable"
        and m.get("subType") == "embedding"
        and m.get("modelName")
    ]
    return ids, {}
