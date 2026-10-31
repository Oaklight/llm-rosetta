"""Model-list transform for the Kilo shim.

Kilo's gateway advertises its whole catalogue and marks the free slice with
``isFree``. Only that slice is exposed.
"""

from __future__ import annotations

from typing import Any


def model_list_transform(
    entries: list[dict[str, Any]],
) -> tuple[list[str], dict[str, str]]:
    """Keep only entries the upstream marks free.

    Args:
        entries: Upstream ``/models`` entries (OpenAI ``{data: [...]}`` slice).

    Returns:
        ``(model_ids, upstream_map)``. Ids pass through unchanged, so the map
        is empty.
    """
    ids = [m["id"] for m in entries if m.get("isFree") is True and m.get("id")]
    return ids, {}
