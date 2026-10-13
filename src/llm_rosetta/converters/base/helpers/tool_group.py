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
from collections.abc import Iterable

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
