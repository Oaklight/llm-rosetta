"""ChatGPT plan route: the request surface it narrows, and its model list.

The plan route speaks the same Open Responses wire format as the OpenAI API
and accepts a good deal less of it.  Two facts belong to the vendor, so they
live here rather than in the converter:

* **A list of fields the route refuses.**  A client is entitled to send them —
  they are valid Responses fields — and the route rejects the request outright
  when it sees one, so they are stripped on the way out.  Dropping a field a
  client asked for is not free: the transform's name appears in the applied
  transform list, which is what the request log shows.
* **``store`` is forced to ``false``.**  ``store: true`` is a valid Responses
  value and the route rejects it; an omitted value is rejected too.  This is
  the request side — ``store_default`` on the shim only shapes the response
  echo.

``previous_response_id`` is in the strip list for a different reason: the route
keeps no server-side conversation state, so the client must send its history in
``input``.  A request that relied on the pointer would silently lose context,
which is worse than being refused.

Model-list transform for the ChatGPT plan route.

The plan route does not answer ``{data: [...]}``: it answers
``{models: [{slug, display_name, visibility}]}``, and it advertises models the
account cannot use.  Only entries it marks ``visibility: "list"`` are real
choices, so those are the ones offered — the same shape the Codex client sees.

The shim loader picks this up by convention (``model_list_transform``) and the
admin's ``/models`` fetch calls it with the ``models`` array.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_rosetta.transforms import Transform, _NamedTransform, force_fields

logger = logging.getLogger(__name__)

#: Fields the plan route rejects.  Every one is a legitimate Responses field,
#: so the client may well have set it; the route refuses the whole request.
UNSUPPORTED_FIELDS = (
    "background",
    "conversation",
    "max_output_tokens",
    "max_tool_calls",
    "metadata",
    "moderation",
    "multi_agent",
    "prompt",
    "prompt_cache_retention",
    "safety_identifier",
    "temperature",
    "top_logprobs",
    "top_p",
    "truncation",
    "user",
    # Not a field the route refuses but one it cannot honour: it keeps no
    # server-side conversation state, so history has to arrive in ``input``.
    "previous_response_id",
)


def drop_refused_fields(*keys: str) -> Transform:
    """Strip *keys*, naming what was actually dropped.

    ``strip_fields`` is silent, which is fine for a field an upstream ignores
    and wrong for one it *refuses*: a caller that sent
    ``max_output_tokens: 100`` would get a full-length answer and no sign that
    the cap was discarded — it looks like it worked.  Refusing the request
    instead (400 naming the field) would be worse: most clients send
    ``temperature`` unconditionally, so the lane would be unusable.  Logging
    the drop is the honest middle: the request still works, and the operator
    can see what was ignored.
    """

    def _drop(body: dict[str, Any]) -> dict[str, Any]:
        dropped = [k for k in keys if k in body]
        if dropped:
            logger.info(
                "ChatGPT plan route: dropped field(s) %s, which the route refuses",
                ", ".join(dropped),
            )
        for k in dropped:
            body.pop(k, None)
        return body

    return _NamedTransform(_drop, f"drop_refused_fields({', '.join(keys)})")


#: Applied after IR conversion, so the body is already in Responses shape.
post_ir_transforms = (
    drop_refused_fields(*UNSUPPORTED_FIELDS),
    force_fields(store=False),
)

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
