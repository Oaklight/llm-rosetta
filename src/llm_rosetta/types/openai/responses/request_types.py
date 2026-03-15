"""OpenAI Responses API request types (TypedDict replicas).

This module contains TypedDict replicas of OpenAI SDK's Responses API request types.
These types are used for type hints and validation in the LLM-Rosetta conversion layer.

Supported OpenAI SDK Versions: 1.x.x through 2.14.0

Reference: openai.types.responses.response_create_params
SDK Source: <python_env>/lib/python3.10/site-packages/openai/types/responses/
"""

from __future__ import annotations

import sys
from typing import Any, Literal, TypedDict, Union
from collections.abc import Iterable

if sys.version_info >= (3, 11):
    from typing import Required
else:
    from typing_extensions import Required

__all__ = [
    # Input types
    "TextInputParam",
    "ImageInputParam",
    "AudioInputParam",
    "ResponseInputParam",
    # Config types
    "ResponsePromptParam",
    "ResponseTextConfigParam",
    "StreamOptions",
    "Reasoning",
    # Tool types
    "FunctionToolParam",
    "ToolChoice",
    # Metadata types
    "Metadata",
    "ResponseIncludable",
    "Conversation",
    # Main request
    "ResponseCreateParams",
]


# ============================================================================
# Input Content Types
# ============================================================================


class TextInputParam(TypedDict, total=False):
    """Text input parameter.

    Reference: openai.types.responses.EasyInputMessageParam (text content)
    """

    type: Required[Literal["text"]]
    """Content type, always 'text'."""

    text: Required[str]
    """The text content."""


class ImageInputParam(TypedDict, total=False):
    """Image input parameter.

    Reference: openai.types.responses.ResponseInputImageParam
    """

    type: Required[Literal["image"]]
    """Content type, always 'image'."""

    image: Required[str]
    """Image URL or base64-encoded image data."""


class AudioInputParam(TypedDict, total=False):
    """Audio input parameter.

    Reference: openai.types.responses.ResponseInputAudioParam
    """

    type: Required[Literal["audio"]]
    """Content type, always 'audio'."""

    audio: Required[str]
    """Base64-encoded audio data."""


ResponseInputParam = Union[
    str,
    TextInputParam,
    ImageInputParam,
    AudioInputParam,
    list[TextInputParam | ImageInputParam | AudioInputParam],
]
"""Input content for the Responses API.

Can be a simple string, a typed input parameter, or a list of typed inputs.
The actual SDK type is extremely complex with 30+ union types; this is a
simplified replica covering the most common cases.

Reference: openai.types.responses.ResponseInputParam
"""


# ============================================================================
# Configuration Types
# ============================================================================


class ResponsePromptParam(TypedDict, total=False):
    """Prompt template reference parameter.

    Reference: openai.types.responses.ResponsePromptParam
    """

    id: str
    """The prompt template ID."""

    name: str
    """The prompt template name."""

    version: int
    """The prompt template version."""


class ResponseTextConfigParam(TypedDict, total=False):
    """Text output configuration parameter.

    Reference: openai.types.responses.ResponseTextConfigParam
    """

    type: Literal["text"]
    """Configuration type."""

    text: str
    """Text configuration value."""


class StreamOptions(TypedDict, total=False):
    """Stream options for the Responses API.

    Reference: openai.types.responses.response_create_params.StreamOptions
    """

    include_usage: bool
    """Whether to include usage statistics in the stream."""


class Reasoning(TypedDict, total=False):
    """Reasoning configuration for the Responses API.

    Reference: openai.types.responses.response_create_params.Reasoning
    """

    enabled: bool
    """Whether reasoning is enabled."""

    max_tokens: int
    """Maximum number of tokens for reasoning."""


# ============================================================================
# Tool Types
# ============================================================================


class FunctionToolParam(TypedDict, total=False):
    """Function tool definition parameter for the Responses API.

    Named FunctionToolParam (not ToolParam) to avoid conflicts with
    OpenAI Chat's ToolParam and Anthropic's ToolParam.

    Reference: openai.types.responses.ToolParam (function variant)
    """

    type: Required[Literal["function"]]
    """Tool type, always 'function'."""

    function: Required[dict[str, Any]]
    """Function definition containing name, description, and parameters."""


ToolChoice = Union[
    Literal["auto", "none", "required"],
    dict[str, Any],
]
"""Tool choice configuration.

Can be a string literal ('auto', 'none', 'required') or a dict
specifying a particular tool.

Reference: openai.types.responses.response_create_params.ToolChoice
"""


# ============================================================================
# Metadata Types
# ============================================================================

Metadata = dict[str, str]
"""Metadata key-value pairs. Up to 16 pairs allowed.

Reference: openai.types.shared.Metadata
"""

ResponseIncludable = Literal[
    "file_search_call.results",
    "message.input_image.image_url",
    "computer_call_output.output.image_url",
    "reasoning.encrypted_content",
    "code_interpreter_call.outputs",
]
"""Includable response fields.

Reference: openai.types.responses.response_create_params.ResponseIncludable
"""


class Conversation(TypedDict, total=False):
    """Conversation context for multi-turn interactions.

    Reference: openai.types.responses.response_create_params.Conversation
    """

    id: Required[str]
    """The conversation ID."""

    messages: list[dict[str, Any]]
    """Previous messages in the conversation."""


# ============================================================================
# Main Request Parameter Type
# ============================================================================


class ResponseCreateParams(TypedDict, total=False):
    """OpenAI Responses API request body structure.

    This is the main request body type for the OpenAI Responses API.
    Contains 28 parameters organized by category.

    Reference: openai.types.responses.response_create_params.ResponseCreateParams
    """

    # Required parameters
    input: Required[str | ResponseInputParam]
    """Input content. Can be a simple string or structured input."""

    model: Required[str]
    """The model ID to use for generation."""

    # Content parameters
    instructions: str | None
    """System instructions, similar to a system message."""

    conversation: Conversation | None
    """Conversation context for multi-turn interactions."""

    # Tool-related parameters
    tools: Iterable[FunctionToolParam]
    """Available tools for the model to use."""

    tool_choice: ToolChoice
    """How the model should choose which tool to use."""

    parallel_tool_calls: bool | None
    """Whether to allow parallel tool calls."""

    max_tool_calls: int | None
    """Maximum number of tool calls allowed."""

    # Generation control parameters
    temperature: float | None
    """Sampling temperature (0.0-2.0). Lower values are more deterministic."""

    top_p: float | None
    """Nucleus sampling parameter (0.0-1.0)."""

    max_output_tokens: int | None
    """Maximum number of output tokens to generate."""

    top_logprobs: int | None
    """Number of top log probabilities to return."""

    frequency_penalty: float | None
    """Frequency penalty (-2.0 to 2.0)."""

    presence_penalty: float | None
    """Presence penalty (-2.0 to 2.0)."""

    logit_bias: dict[str, int] | None
    """Token logit bias. Keys are token IDs, values are bias values."""

    # Control parameters
    stream: Literal[False] | None
    """Whether to stream the response."""

    stream_options: StreamOptions | None
    """Options for streaming responses."""

    response_format: dict[str, Any]
    """Response format configuration."""

    truncation: Literal["auto", "disabled"] | None
    """Truncation strategy for input."""

    user: str
    """User identifier for tracking."""

    metadata: Metadata | None
    """Metadata key-value pairs (up to 16)."""

    # Other parameters
    background: bool | None
    """Whether to run the response in the background."""

    include: list[ResponseIncludable] | None
    """Additional data to include in the response."""

    previous_response_id: str | None
    """ID of the previous response for continuation."""

    prompt: ResponsePromptParam | None
    """Prompt template reference."""

    prompt_cache_key: str
    """Key for prompt caching."""

    prompt_cache_retention: Literal["in-memory", "24h"] | None
    """Prompt cache retention policy."""

    reasoning: Reasoning | None
    """Reasoning configuration."""

    safety_identifier: str
    """Safety identifier."""

    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None
    """Service tier for request routing."""

    store: bool | None
    """Whether to store the response."""

    text: ResponseTextConfigParam
    """Text output configuration."""

    # Additional fields
    audio: dict[str, Any]
    """Audio configuration."""

    modalities: list[str]
    """Supported modalities."""

    prediction: dict[str, Any]
    """Prediction configuration."""

    reasoning_effort: Literal["low", "medium", "high"]
    """Reasoning effort level."""
