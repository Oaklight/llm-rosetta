"""What each provider's usage payload means, declared once.

Two facts about token counters are asked for in several places, and the answers
have to agree with each other:

**Where a counter sits on the wire.** :class:`~llm_rosetta.pipeline.ConversionPipeline`
reads a raw upstream chunk while a response streams; the converters read a whole
provider response.

**Whether the cache counters add to the input or sit inside it.** Deriving a
total depends on it.  The Messages API reports ``cache_read_input_tokens`` and
``cache_creation_input_tokens`` *on top of* ``input_tokens``, so a total that
leaves them out under-counts; the OpenAI-shaped and Google-shaped APIs report
``cached_tokens`` / ``cachedContentTokenCount`` *inside* the prompt count, so a
total that adds them counts the same tokens twice.

Both are declared here and nowhere else::

    shape             counters                                     cache
    openai-details    prompt_tokens_details.cached_tokens           subset
                      completion_tokens_details.reasoning_tokens
    messages-fields   cache_read_input_tokens                      additive
                      cache_creation_input_tokens
                      output_tokens_details.thinking_tokens
    google-metadata   cachedContentTokenCount, cached_content_...   subset
                      thoughtsTokenCount, thoughts_token_count

:meth:`UsageShape.extended` answers the first question for any payload,
:func:`cache_is_additive` the second for any API standard, and
:data:`ADDITIVE_CACHE_STANDARDS` is the derived set a repair can key on.

Reasoning counters are declared next to the cache ones because they share the
one property that matters when a total is computed: they are always a *subset*
of the output tokens, and are never added to it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

__all__ = [
    "ADDITIVE_CACHE_STANDARDS",
    "GOOGLE_METADATA",
    "MESSAGES_FIELDS",
    "OPENAI_DETAILS",
    "STANDARD_SHAPES",
    "USAGE_SHAPES",
    "UsageShape",
    "cache_is_additive",
    "shape_for",
]


def _read(payload: Mapping[str, Any], path: tuple[str, ...]) -> int | None:
    """The non-negative int at *path* inside *payload*, or ``None``."""
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current if isinstance(current, int) and current >= 0 else None


@dataclass(frozen=True)
class UsageShape:
    """Where one dialect keeps its extended counters, and what they mean.

    Each counter lists candidate paths rather than one path: a dialect may spell
    the same counter two ways (Google's ``camelCase`` and ``snake_case``), and
    the first one present wins.
    """

    name: str
    cache_read: tuple[tuple[str, ...], ...] = ()
    cache_creation: tuple[tuple[str, ...], ...] = ()
    reasoning: tuple[tuple[str, ...], ...] = ()
    #: Whether the cache counters are *additional* to the input token count.
    cache_additive: bool = False

    def extended(self, payload: Mapping[str, Any]) -> dict[str, int]:
        """The cache and reasoning counters this shape finds in *payload*.

        Returns:
            A ledger-column dict, holding only the counters that are present.
        """
        found: dict[str, int] = {}
        for column, paths in (
            ("cache_read_tokens", self.cache_read),
            ("cache_creation_tokens", self.cache_creation),
            ("reasoning_tokens", self.reasoning),
        ):
            for path in paths:
                value = _read(payload, path)
                if value is not None:
                    found[column] = value
                    break
        return found


#: OpenAI-shaped usage: the cache counter is nested inside the prompt details,
#: so it is already part of ``prompt_tokens``.
OPENAI_DETAILS = UsageShape(
    name="openai-details",
    cache_read=(("prompt_tokens_details", "cached_tokens"),),
    reasoning=(("completion_tokens_details", "reasoning_tokens"),),
)

#: Messages-shaped usage: the API sends no total at all, and both cache counters
#: are additional to ``input_tokens``.
MESSAGES_FIELDS = UsageShape(
    name="messages-fields",
    cache_read=(("cache_read_input_tokens",),),
    cache_creation=(("cache_creation_input_tokens",),),
    reasoning=(("output_tokens_details", "thinking_tokens"),),
    cache_additive=True,
)

#: Google ``usageMetadata``: the cache counter is part of ``promptTokenCount``.
GOOGLE_METADATA = UsageShape(
    name="google-metadata",
    cache_read=(("cachedContentTokenCount",), ("cached_content_token_count",)),
    reasoning=(("thoughtsTokenCount",), ("thoughts_token_count",)),
)

#: Every shape this table knows, for a reader that wants to try them all.
USAGE_SHAPES: tuple[UsageShape, ...] = (
    OPENAI_DETAILS,
    MESSAGES_FIELDS,
    GOOGLE_METADATA,
)

#: API standard -> the shape its usage payload takes.  Every spelling a log row
#: may carry is listed, legacy ones included, because the repair that reads this
#: table works on rows written years apart.
#:
#: ``google_interactions`` is deliberately absent: its payload is none of the
#: three shapes (it uses ``total_input_tokens`` / ``total_thought_tokens``) and
#: is read by its own converter.  Nothing here claims to know its cache
#: semantics, so nothing derives a total for it.
STANDARD_SHAPES: Mapping[str, UsageShape] = MappingProxyType(
    {
        "chat_completions": OPENAI_DETAILS,
        "openai_chat": OPENAI_DETAILS,  # legacy spelling
        "open_responses": OPENAI_DETAILS,
        "openai_responses": OPENAI_DETAILS,  # legacy spelling
        "messages": MESSAGES_FIELDS,
        "anthropic": MESSAGES_FIELDS,  # legacy spelling
        "google_generate": GOOGLE_METADATA,
        "google": GOOGLE_METADATA,  # legacy spelling
    }
)

#: The standards whose cache counters are additive, derived from the table above
#: — never listed by hand.
ADDITIVE_CACHE_STANDARDS: frozenset[str] = frozenset(
    standard for standard, shape in STANDARD_SHAPES.items() if shape.cache_additive
)


def shape_for(standard: str) -> UsageShape | None:
    """The usage shape *standard*'s payload takes, or ``None`` if unknown."""
    return STANDARD_SHAPES.get(standard)


def cache_is_additive(standard: str) -> bool:
    """Whether *standard*'s cache counters are additional to its input tokens.

    An unknown standard answers ``False``: a total is only recomputed for a
    standard whose cache semantics are actually known.
    """
    shape = STANDARD_SHAPES.get(standard)
    return bool(shape and shape.cache_additive)
