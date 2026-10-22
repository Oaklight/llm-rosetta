"""Model-list transform for the ChatGPT plan route.

The plan route does not answer ``{data: [...]}``: it answers
``{models: [{slug, display_name, visibility}]}``, and it advertises models the
account cannot use.  Only entries it marks ``visibility: "list"`` are real
choices, so those are the ones offered — the same shape the Codex client sees.

The shim loader picks this up by convention (``model_list_transform``) and the
admin's ``/models`` fetch calls it with the ``models`` array.
"""

from __future__ import annotations

from typing import Any

#: The one visibility value that means "this model may be selected".
_VISIBLE = "list"


def model_list_transform(
    entries: list[dict[str, Any]],
) -> tuple[list[str], dict[str, str]]:
    """Return the plan route's selectable model slugs, in the server's order.

    Args:
        entries: The upstream ``models`` array.

    Returns:
        ``(model_ids, upstream_map)``.  The id *is* the slug the route expects,
        so nothing is renamed and the map is empty — a slug passed back
        upstream is already correct.  ``display_name`` is not carried: the
        hook's contract has no channel for a label, and inventing one here
        would be invisible to the caller.
    """
    ids = [
        entry["slug"]
        for entry in entries
        if entry.get("visibility") == _VISIBLE and entry.get("slug")
    ]
    return ids, {}
