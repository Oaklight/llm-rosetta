"""Provider-specific opaque IR payload types."""

from __future__ import annotations

import sys
from typing import Any, Literal

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required, TypedDict
else:
    from typing_extensions import NotRequired, Required, TypedDict


class ProviderPassthroughEvent(TypedDict):
    """Opaque provider-native streaming event."""

    type: Required[Literal["provider_passthrough"]]
    provider: Required[str]
    payload: Required[dict[str, Any]]


class ProviderPassthroughItem(TypedDict):
    """Opaque provider-native non-streaming item."""

    type: Required[Literal["provider_passthrough_item"]]
    provider: Required[str]
    payload: Required[dict[str, Any]]
    position: NotRequired[int]


__all__ = ["ProviderPassthroughEvent", "ProviderPassthroughItem"]
