"""Model-list transform for the free-source shim.

The upstream advertises its whole catalogue and marks the free slice with
``isFree``. Only that slice is exposed, and ids whose first path segment is
the source's own brand (its auto-router rather than a concrete model) are
dropped so nothing branded reaches the roster.
"""

from __future__ import annotations

from typing import Any

# First path segments that identify the source's own brand / router entries
# rather than a concrete model from some other upstream. Kept out of the
# roster so the provider stays vendor-neutral in-product.
_BRAND_SEGMENTS = ("kilo",)


def model_list_transform(
    entries: list[dict[str, Any]],
) -> tuple[list[str], dict[str, str]]:
    """Keep only free models, dropping the source's branded router ids.

    Args:
        entries: Upstream ``/models`` entries (OpenAI ``{data: [...]}`` slice).

    Returns:
        ``(model_ids, upstream_map)``. Ids are passed through unchanged, so
        the map is empty unless a rename is introduced later.
    """
    ids: list[str] = []
    for m in entries:
        if m.get("isFree") is not True:
            continue
        raw_id = m.get("id", "")
        if not raw_id:
            continue
        head = raw_id.split("/", 1)[0].lower()
        if any(head.startswith(seg) for seg in _BRAND_SEGMENTS):
            continue
        ids.append(raw_id)
    return ids, {}
