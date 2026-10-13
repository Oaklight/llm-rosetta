"""Deterministic identifiers for tool containers that arrive without a name.

A client may group tools by declaring them inside a named container, addressing
them by ``(name, group)``.  When the container has no name of its own there is
nothing to address the group by, and the empty string is not usable either:
providers match tool names against ``^[a-zA-Z0-9_-]+$``, so a group of ``""``
asks the naming ladder to build a qualified name out of nothing while still
counting as "came from a container".  Such a group is given an identifier
derived from the container's own content instead, so the same request always
produces the same one - the prompt-cache invariant the naming ladder already
depends on (:func:`~llm_rosetta.capabilities.qualify_tool_names`).
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

#: Digest bytes kept.  Four is 32 bits, and encodes to seven base32 symbols.
_DIGEST_BYTES = 4

#: Leading character, marking the identifier as one we generated rather than
#: one the client wrote.  It is the only readable part of the name.
_STEM = "g"

#: Length of every identifier this module produces.
GENERATED_GROUP_ID_LENGTH = len(_STEM) + 7

#: ``metadata`` key marking a group name as generated rather than client-supplied.
#: Underscored like ``_original_name``: both record something we decided, not
#: something the client sent.  Passes that must treat a nameless container
#: differently from a named one read it, and so does the per-request warning
#: about it.
GENERATED_GROUP_MARKER = "_generated_group"


def generated_group_id(description: str, child_names: Iterable[str]) -> str:
    """Derive a short, stable identifier for an unnamed tool container.

    The seed is the container's own content: its description and the names of
    its children, in declaration order, NUL-joined so no field can impersonate
    a boundary.  Child *names* alone, not their full schemas - a schema is far
    larger, and it changes under edits that do not change which group this is.

    Being content-derived is the point.  A per-request counter would be
    shorter, but it counts position: inserting one container would renumber
    every container after it and miss the prompt cache on a request that
    barely changed.

    Args:
        description: The container's description, or ``""`` when it had none.
        child_names: Names of the container's children, in declaration order.

    Returns:
        A name-safe identifier of
        :data:`GENERATED_GROUP_ID_LENGTH` characters: a ``"g"`` stem followed
        by lowercase base32.  It contains neither ``_`` nor ``-``, so it can
        never be mistaken for the separator the naming ladder joins it to a
        tool name with.
    """
    seed = "\0".join([description, *child_names])
    digest = hashlib.sha256(seed.encode("utf-8")).digest()[:_DIGEST_BYTES]
    return _STEM + base64.b32encode(digest).decode("ascii").rstrip("=").lower()


@dataclass(frozen=True)
class ToolGroup:
    """A named set of tools, as one entry of the list a target receives.

    Not IR: the IR keeps every tool flat and records where it came from in
    ``tool_group``.  This is what a writer walks when its target can carry a
    group on the wire, and what :func:`group_tools` hands it.
    """

    name: str
    description: str
    tools: tuple[Any, ...]


#: Distinguishes a group's slot from a tool while the list is being assembled.
#: A tool dict is never this object, and the list may hold arbitrary values.
_GROUP_SLOT = object()


def group_tools(ir_tools: Sequence[Any], *, carries_groups: bool) -> list[Any]:
    """Order a tool list for a target, collapsing declared groups into entries.

    A client may address a tool by ``(name, group)``, and only some targets can
    carry that grouping on the wire.  For those, each group comes back as one
    :class:`ToolGroup` holding its tools, placed where the group's first member
    stood.  That position is not incidental: the tool list is the client's, and
    its order is what an upstream prompt cache is keyed on, so a group takes the
    place the client's own declaration began at.  The one reordering that follows
    is a member declared apart from its group: the group takes the first
    member's place and its members follow in declaration order, so anything that
    stood between them moves after it.  No client produces that — a container's
    tools are declared together — but the rule is this one, not "nothing moves".

    Everything else is returned untouched and in place, so a writer can walk
    the result and render whichever kind of entry it finds.  When
    *carries_groups* is false the tools come back as they are — what every
    target with no grouping concept of its own gets, which today is every
    target but the Responses family.

    Membership is read from ``tool_group``, a group's description from
    ``tool_group_description``.  An **empty** ``tool_group`` is not a group
    here, because there is no name to put on the wire as the container's —
    deliberately, and not the reading ``build_tool_name_map`` gives the same
    field, which uses ``is None`` so that ``""`` still counts as container
    membership.  The two ask different questions: this one is whether the
    members may be merged into one entry, that one is whether a bare call could
    have meant the tool.  Nothing the library produces reaches the difference,
    since the harvest gives a nameless container a generated id.  A name this library generated, because the
    client's container had none, is carried like any other: it is a valid name
    and the target needs one, which is why it was generated.  Keeping such a
    name away from the client is the response leg's job, not this one's.

    Args:
        ir_tools: The IR tool definitions, in the order the client declared.
        carries_groups: Whether the target understands a named group of tools.

    Returns:
        A new list holding the same tool dicts, with each group replaced by a
        single :class:`ToolGroup`.
    """
    if not carries_groups:
        return list(ir_tools)

    placed: list[Any] = []
    slot: dict[str, int] = {}
    members: dict[str, list[Any]] = {}
    descriptions: dict[str, str] = {}

    for tool in ir_tools:
        group = tool.get("tool_group") if isinstance(tool, dict) else None
        if not group:
            placed.append(tool)
            continue
        if group not in slot:
            slot[group] = len(placed)
            members[group] = []
            placed.append(_GROUP_SLOT)
        members[group].append(tool)
        description = tool.get("tool_group_description")
        if isinstance(description, str) and description:
            descriptions.setdefault(group, description)

    if not slot:
        return list(ir_tools)

    for group, position in slot.items():
        placed[position] = ToolGroup(
            name=group,
            description=descriptions.get(group, ""),
            tools=tuple(members[group]),
        )
    return placed
