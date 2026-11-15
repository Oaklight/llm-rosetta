"""Model-list transform for the SiliconFlow chat shim.

The upstream playground catalogue tags every entry with ``type``/``subType``
and a ``status``.  Only the active chat slice belongs in a chat picker; the
image, video, speech and embedding entries answer a different endpoint.
"""

from __future__ import annotations

from typing import Any


def model_list_transform(
    entries: list[dict[str, Any]],
) -> tuple[list[str], dict[str, str]]:
    """Keep the active chat models.

    Args:
        entries: Upstream catalogue entries from
            ``/api/v1/playground/comprehensive/all``.

    Returns:
        ``(model_ids, upstream_map)``.  The id is ``modelName``, which is the
        string the chat endpoint accepts, so no mapping is needed.
    """
    ids = [
        m["modelName"]
        for m in entries
        if m.get("status") != "disable"
        and m.get("subType") == "chat"
        and m.get("modelName")
    ]
    return ids, {}
