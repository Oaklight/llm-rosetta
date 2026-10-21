"""Config value types: what a configured provider is, and where it routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from ..transport import ProviderInfo


class TypedRoute(NamedTuple):
    """Resolved provider routing info for a typed (non-LLM) model type.

    Embedding, rerank and decision resolve to the same four facts — which
    provider, in which format, at which upstream path, with which connection
    — so they share one value.  ``path`` is the configured endpoint path; the
    upstream URL itself lives in ``provider_info``.
    """

    provider_name: str
    format: str
    path: str
    provider_info: ProviderInfo


@dataclass(frozen=True)
class ProviderIdentity:
    """What a configured provider is: its capability category and wire format.

    The pair is the gateway-side counterpart of a shim's
    ``(model_type, format)``.  ``model_type`` is one of the registry's
    categories (``llm``, ``embedding``, ``rerank``, ``decision``); ``format``
    is the dialect within that category.  A format name alone is ambiguous —
    ``openai`` is a vendor, a chat shim and an embedding format — so consumers
    that route or convert read the pair together.
    """

    model_type: str
    format: str
