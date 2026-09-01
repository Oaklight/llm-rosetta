"""Shared model-list utilities for Argo provider shims."""

from __future__ import annotations

import re


def model_list_transform(
    raw_entries: list[dict],
) -> tuple[list[str], dict[str, str]]:
    """Transform raw Argo model entries into (display_names, upstream_map).

    Argo's ``/models`` endpoint returns entries where ``id`` is a
    human-readable name ("Claude Opus 5") and ``internal_id`` is the
    compact upstream ID ("claudeopus5").  This converts to slug-style
    display names suitable for gateway routing.

    When ``internal_id`` is absent (e.g. when the upstream is an
    argo-proxy instance that already returns ``argo:...`` style IDs),
    the entry is passed through as-is — no slug transformation or
    prefix is applied.

    Args:
        raw_entries: List of model dicts from the upstream ``/models``
            response, each containing at least an ``id`` key and
            optionally ``internal_id``.

    Returns:
        A ``(model_ids, upstream_map)`` tuple where *model_ids* is a
        list of display names and *upstream_map* maps each display
        name to its upstream internal ID (only for entries where the
        two differ).
    """
    model_ids: list[str] = []
    upstream_map: dict[str, str] = {}
    for m in raw_entries:
        raw_id = m.get("id", "")
        if not raw_id:
            continue
        internal_id = m.get("internal_id")
        if internal_id:
            slug = re.sub(r"[^a-z0-9]+", "-", raw_id.lower()).strip("-")
            if not slug:
                continue
            display = f"argo:{slug}"
            upstream_map[display] = internal_id
        else:
            display = raw_id
        model_ids.append(display)
    return model_ids, upstream_map
