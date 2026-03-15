"""Anthropic Messages API request types (TypedDict replicas).

This module contains TypedDict replicas of Anthropic SDK request types.
These are used for type hints and validation in the LLM-Rosetta conversion layer.
"""

from __future__ import annotations

import sys
from typing import Dict, Iterable, List, Literal, Optional, TypedDict, Union

if sys.version_info >= (3, 11):
    from typing import Required
else:
    from typing_extensions import Required

__all__ = [
    "CacheControlEphemeralParam",
    "TextBlockParam",
    "MessageParam",
    "InputSchema",
    "ToolParam",
    "ToolChoiceAutoParam",
    "ToolChoiceAnyParam",
    "ToolChoiceToolParam",
    "ToolChoiceNoneParam",
    "ToolChoiceParam",
    "ThinkingConfigEnabledParam",
    "ThinkingConfigDisabledParam",
    "ThinkingConfigParam",
    "MetadataParam",
    "MessageCreateParamsBase",
    "MessageCreateParamsNonStreaming",
    "MessageCreateParamsStreaming",
    "MessageCreateParams",
]


# Cache control
class CacheControlEphemeralParam(TypedDict, total=False):
    """Cache control parameter for ephemeral caching."""

    type: Required[Literal["ephemeral"]]


# Text citation parameter
class TextCitationParam(TypedDict, total=False):
    """Text citation parameter."""

    cited_text: str
    type: Literal["text_citation"]


# Content block parameters
class TextBlockParam(TypedDict, total=False):
    """Text block parameter for request."""

    text: Required[str]
    type: Required[Literal["text"]]
    cache_control: Optional[CacheControlEphemeralParam]
    citations: Optional[Iterable[TextCitationParam]]


# Message parameter
class MessageParam(TypedDict, total=False):
    """Message parameter for request."""

    content: Required[Union[str, Iterable[TextBlockParam]]]
    """Message content - can be a string or list of content blocks."""

    role: Required[Literal["user", "assistant"]]
    """Message role - either 'user' or 'assistant'."""


# Tool definition
class InputSchemaTyped(TypedDict, total=False):
    """Typed input schema for tool parameters."""

    type: Required[Literal["object"]]
    properties: Optional[Dict[str, object]]
    required: Optional[List[str]]


InputSchema = Union[InputSchemaTyped, Dict[str, object]]
"""Tool input schema - JSON Schema format."""


class ToolParam(TypedDict, total=False):
    """Tool definition parameter."""

    input_schema: Required[InputSchema]
    """JSON schema for this tool's input."""

    name: Required[str]
    """Name of the tool."""

    description: str
    """Description of what this tool does."""

    cache_control: Optional[CacheControlEphemeralParam]
    """Create a cache control breakpoint at this content block."""

    type: Optional[Literal["custom"]]
    """Tool type, typically 'custom'."""


# Tool choice parameters
class ToolChoiceAutoParam(TypedDict, total=False):
    """Auto tool choice - model decides whether to use tools."""

    type: Required[Literal["auto"]]
    disable_parallel_tool_use: bool
    """Whether to disable parallel tool use."""


class ToolChoiceAnyParam(TypedDict, total=False):
    """Any tool choice - model must use at least one tool."""

    type: Required[Literal["any"]]
    disable_parallel_tool_use: bool
    """Whether to disable parallel tool use."""


class ToolChoiceToolParam(TypedDict, total=False):
    """Specific tool choice - model must use the specified tool."""

    type: Required[Literal["tool"]]
    name: Required[str]
    """The name of the tool to use."""
    disable_parallel_tool_use: bool
    """Whether to disable parallel tool use."""


class ToolChoiceNoneParam(TypedDict, total=False):
    """None tool choice - model should not use tools."""

    type: Required[Literal["none"]]
    disable_parallel_tool_use: bool
    """Whether to disable parallel tool use."""


ToolChoiceParam = Union[
    ToolChoiceAutoParam, ToolChoiceAnyParam, ToolChoiceToolParam, ToolChoiceNoneParam
]
"""Tool choice parameter - how the model should use tools."""


# Thinking configuration
class ThinkingConfigEnabledParam(TypedDict, total=False):
    """Enabled thinking configuration."""

    type: Required[Literal["enabled"]]
    budget_tokens: int
    """Budget for thinking tokens (minimum 1024)."""


class ThinkingConfigDisabledParam(TypedDict, total=False):
    """Disabled thinking configuration."""

    type: Required[Literal["disabled"]]


ThinkingConfigParam = Union[ThinkingConfigEnabledParam, ThinkingConfigDisabledParam]
"""Thinking configuration parameter."""


# Metadata
class MetadataParam(TypedDict, total=False):
    """Metadata parameter for request."""

    user_id: str
    """User ID for tracking."""


# Main request parameters
class MessageCreateParamsBase(TypedDict, total=False):
    """Base parameters for creating a message."""

    max_tokens: Required[int]
    """The maximum number of tokens to generate before stopping."""

    messages: Required[Iterable[MessageParam]]
    """Input messages."""

    model: Required[str]
    """The model that will complete your prompt."""

    metadata: MetadataParam
    """An object describing metadata about the request."""

    service_tier: Literal["auto", "standard_only"]
    """Determines whether to use priority capacity or standard capacity."""

    stop_sequences: List[str]
    """Custom text sequences that will cause the model to stop generating."""

    system: Union[str, Iterable[TextBlockParam]]
    """System prompt."""

    temperature: float
    """Amount of randomness injected into the response (0.0-1.0)."""

    thinking: ThinkingConfigParam
    """Configuration for enabling Claude's extended thinking."""

    tool_choice: ToolChoiceParam
    """How the model should use the provided tools."""

    tools: Iterable[ToolParam]
    """Definitions of tools that the model may use."""

    top_k: int
    """Only sample from the top K options for each subsequent token."""

    top_p: float
    """Use nucleus sampling."""


class MessageCreateParamsNonStreaming(MessageCreateParamsBase, total=False):
    """Parameters for non-streaming message creation."""

    stream: Literal[False]
    """Whether to incrementally stream the response."""


class MessageCreateParamsStreaming(MessageCreateParamsBase):
    """Parameters for streaming message creation."""

    stream: Required[Literal[True]]
    """Whether to incrementally stream the response."""


MessageCreateParams = Union[
    MessageCreateParamsNonStreaming, MessageCreateParamsStreaming
]
"""Message creation parameters - streaming or non-streaming."""
