"""Anthropic Messages API response types (TypedDict replicas).

This module contains TypedDict replicas of Anthropic SDK response types.
These are used for type hints and validation in the LLM-Rosetta conversion layer.
"""

from __future__ import annotations

import sys
from typing import Literal, TypedDict

if sys.version_info >= (3, 11):
    from typing import Required
else:
    from typing_extensions import Required

__all__ = [
    "StopReason",
    "CharLocationCitation",
    "PageLocationCitation",
    "ContentBlockLocationCitation",
    "SearchResultLocationCitation",
    "WebSearchResultLocationCitation",
    "TextCitation",
    "TextBlock",
    "ThinkingBlock",
    "ToolUseBlock",
    "ContentBlock",
    "Usage",
    "Message",
]


# Stop reason type alias
StopReason = Literal[
    "end_turn",  # The model reached a natural stopping point
    "max_tokens",  # We exceeded the requested max_tokens or the model's maximum
    "stop_sequence",  # One of your provided custom stop_sequences was generated
    "tool_use",  # The model invoked one or more tools
    "pause_turn",  # We paused a long-running turn
    "refusal",  # When streaming classifiers intervene to handle potential policy violations
]


# Content block types
class CharLocationCitation(TypedDict, total=False):
    """A citation of a character range in a document the caller supplied."""

    type: Required[Literal["char_location"]]
    cited_text: str
    document_index: int
    document_title: str
    start_char_index: int
    end_char_index: int


class PageLocationCitation(TypedDict, total=False):
    """A citation of a page range in a PDF the caller supplied."""

    type: Required[Literal["page_location"]]
    cited_text: str
    document_index: int
    document_title: str
    start_page_number: int
    end_page_number: int


class ContentBlockLocationCitation(TypedDict, total=False):
    """A citation of content blocks in a document the caller supplied."""

    type: Required[Literal["content_block_location"]]
    cited_text: str
    document_index: int
    document_title: str
    start_block_index: int
    end_block_index: int


class SearchResultLocationCitation(TypedDict, total=False):
    """A citation of a ``search_result`` block the caller supplied."""

    type: Required[Literal["search_result_location"]]
    cited_text: str
    source: str
    title: str
    search_result_index: int
    start_block_index: int
    end_block_index: int


class WebSearchResultLocationCitation(TypedDict, total=False):
    """A citation of a result this provider's own web search returned.

    ``encrypted_index`` is opaque provider-issued material: a citation of this
    kind cannot be constructed by anyone but the provider that issued it.
    """

    type: Required[Literal["web_search_result_location"]]
    cited_text: str
    url: str
    title: str
    encrypted_index: str


#: Every citation this API accepts, measured against the live endpoint: these
#: five ``type`` tags and no others.  (It rejects ``text_citation`` and
#: ``url_citation`` outright.)  A citation also has to point at something the
#: message itself carries — a document block, a ``search_result`` block — or at
#: an ``encrypted_index`` this provider issued, which is why a citation minted
#: by another dialect has no form here.


TextCitation = (
    CharLocationCitation
    | PageLocationCitation
    | ContentBlockLocationCitation
    | SearchResultLocationCitation
    | WebSearchResultLocationCitation
)


class TextBlock(TypedDict, total=False):
    """Text content block in response."""

    type: Literal["text"]
    text: str
    citations: list[TextCitation] | None


class ThinkingBlock(TypedDict, total=False):
    """Thinking/reasoning content block in response."""

    type: Literal["thinking"]
    signature: str
    thinking: str


class ToolUseBlock(TypedDict, total=False):
    """Tool use content block in response."""

    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, object]


# Union type for all content blocks
ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock


# Usage statistics
class CacheCreation(TypedDict, total=False):
    """Breakdown of cached tokens by TTL."""

    pass  # Structure not fully defined in SDK


class ServerToolUsage(TypedDict, total=False):
    """The number of server tool requests."""

    pass  # Structure not fully defined in SDK


class Usage(TypedDict, total=False):
    """Billing and rate-limit usage information."""

    input_tokens: int
    """The number of input tokens which were used."""

    output_tokens: int
    """The number of output tokens which were used."""

    cache_creation_input_tokens: int | None
    """The number of input tokens used to create the cache entry."""

    cache_read_input_tokens: int | None
    """The number of input tokens read from the cache."""

    cache_creation: CacheCreation | None
    """Breakdown of cached tokens by TTL."""

    server_tool_use: ServerToolUsage | None
    """The number of server tool requests."""

    service_tier: Literal["standard", "priority", "batch"] | None
    """If the request used the priority, standard, or batch tier."""


# Main response message
class Message(TypedDict, total=False):
    """Anthropic Messages API response message."""

    id: str
    """Unique object identifier."""

    type: Literal["message"]
    """Object type, always 'message'."""

    role: Literal["assistant"]
    """Conversational role of the generated message, always 'assistant'."""

    content: list[ContentBlock]
    """Content generated by the model."""

    model: str
    """The model that completed your prompt."""

    stop_reason: StopReason | None
    """The reason that we stopped."""

    stop_sequence: str | None
    """Which custom stop sequence was generated, if any."""

    usage: Usage
    """Billing and rate-limit usage."""
