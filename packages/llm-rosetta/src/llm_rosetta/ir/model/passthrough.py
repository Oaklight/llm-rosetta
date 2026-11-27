"""Provider-specific opaque IR payload types."""

from __future__ import annotations

import sys
from typing import Any, Literal, TypedDict

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required
else:
    from typing_extensions import NotRequired, Required


class ProviderPassthroughEvent(TypedDict):
    """Opaque provider-native streaming event."""

    type: Required[Literal["provider_passthrough"]]
    provider: Required[str]
    payload: Required[dict[str, Any]]
    #: The output item this event is a frame of, when the dialect streams items
    #: (Open Responses ``output_item.added`` / ``done``).  Folding a stream keeps
    #: the item, not the frame, so the folded body holds the record the body leg
    #: would have given.  The later frame of the same ``position`` supersedes the
    #: earlier one.  ``payload`` stays the whole frame for the writer to replay.
    item: NotRequired[dict[str, Any]]
    #: The item's index in the response's output list.
    position: NotRequired[int]


class ProviderPassthroughItem(TypedDict):
    """Opaque provider-native non-streaming item."""

    type: Required[Literal["provider_passthrough_item"]]
    provider: Required[str]
    payload: Required[dict[str, Any]]
    position: NotRequired[int]


__all__ = ["ProviderPassthroughEvent", "ProviderPassthroughItem"]
