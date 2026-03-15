"""OpenAI Chat Completion API Request Types.

This module contains TypedDict replicas of OpenAI SDK's request types.
These types are used for type hints and validation in the LLM-Rosetta library.

Supported OpenAI SDK Versions: 1.x.x through 2.14.0
Last Updated: 2025-01-10

Reference: openai.types.chat.completion_create_params
SDK Source: <python_env>/lib/python3.10/site-packages/openai/types/chat/
"""

import sys
from typing import (
    TYPE_CHECKING,
    Literal,
    TypedDict,
    Union,
)
from collections.abc import Iterable

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required
else:
    from typing_extensions import NotRequired, Required


# ============================================================================
# Model Types
# ============================================================================

ChatModel = Literal[
    "gpt-4o",
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "gpt-4o-audio-preview",
    "gpt-4o-audio-preview-2024-10-01",
    "gpt-4o-audio-preview-2024-12-17",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-4-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "o1",
    "o1-2024-12-17",
    "o1-preview",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o3-mini",
    "o3-mini-2025-01-31",
]


# ============================================================================
# Tool Related Types
# ============================================================================


class FunctionParameters(TypedDict, total=False):
    """JSON Schema for function parameters.

    This represents a JSON Schema object that defines the parameters
    a function accepts. It follows the JSON Schema specification.

    Reference: openai.types.shared_params.FunctionParameters

    Example:
        {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    """

    pass  # This is a flexible dict representing JSON Schema


class FunctionDefinition(TypedDict, total=False):
    """Function definition for tools.

    Reference: openai.types.shared_params.FunctionDefinition
    """

    name: Required[str]
    description: NotRequired[str]
    parameters: NotRequired[FunctionParameters]
    strict: NotRequired[bool | None]


class ChatCompletionFunctionToolParam(TypedDict, total=False):
    """Function tool parameter.

    Reference: openai.types.chat.ChatCompletionToolParam
    """

    type: Required[Literal["function"]]
    function: Required[FunctionDefinition]


# Tool choice types
class ChatCompletionNamedToolChoiceFunction(TypedDict):
    """Named tool choice function."""

    name: str


class ChatCompletionNamedToolChoiceParam(TypedDict):
    """Named tool choice parameter."""

    type: Literal["function"]
    function: ChatCompletionNamedToolChoiceFunction


ChatCompletionToolChoiceOptionParam = Union[
    Literal["none", "auto", "required"], ChatCompletionNamedToolChoiceParam
]


# ============================================================================
# Generation Control Types
# ============================================================================

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


# ============================================================================
# Response Format Types
# ============================================================================


class ResponseFormatText(TypedDict):
    """Text response format."""

    type: Literal["text"]


class ResponseFormatJSONObject(TypedDict):
    """JSON object response format."""

    type: Literal["json_object"]


class ResponseFormatJSONSchemaSchema(TypedDict, total=False):
    """JSON schema definition."""

    name: Required[str]
    description: NotRequired[str]
    schema: NotRequired[dict[str, object]]
    strict: NotRequired[bool | None]


class ResponseFormatJSONSchema(TypedDict):
    """JSON schema response format."""

    type: Literal["json_schema"]
    json_schema: ResponseFormatJSONSchemaSchema


ResponseFormat = Union[
    ResponseFormatText, ResponseFormatJSONObject, ResponseFormatJSONSchema
]


# ============================================================================
# Stream Options
# ============================================================================


class ChatCompletionStreamOptionsParam(TypedDict, total=False):
    """Stream options parameter.

    Reference: openai.types.chat.ChatCompletionStreamOptionsParam
    """

    include_usage: NotRequired[bool]


# ============================================================================
# Metadata Types
# ============================================================================

Metadata = dict[str, str]  # Up to 16 key-value pairs


# ============================================================================
# Main Request Parameter Type
# ============================================================================


class CompletionCreateParams(TypedDict, total=False):
    """Chat completion create parameters.

    This is the main request body type for OpenAI Chat Completion API.

    Reference: openai.types.chat.completion_create_params.CompletionCreateParams
    """

    # Required parameters
    messages: Required[Iterable["ChatCompletionMessageParam"]]
    model: Required[str | ChatModel]

    # Tool related parameters
    tools: NotRequired[Iterable[ChatCompletionFunctionToolParam]]
    tool_choice: NotRequired[ChatCompletionToolChoiceOptionParam]
    parallel_tool_calls: NotRequired[bool]

    # Generation control parameters
    temperature: NotRequired[float | None]
    top_p: NotRequired[float | None]
    max_completion_tokens: NotRequired[int | None]
    max_tokens: NotRequired[int | None]  # Deprecated, use max_completion_tokens
    n: NotRequired[int | None]
    frequency_penalty: NotRequired[float | None]
    presence_penalty: NotRequired[float | None]
    logit_bias: NotRequired[dict[str, int] | None]
    seed: NotRequired[int | None]
    top_logprobs: NotRequired[int | None]
    logprobs: NotRequired[bool | None]

    # Control parameters
    stop: NotRequired[str | None | list[str] | None]
    stream: NotRequired[Literal[False] | None]
    stream_options: NotRequired[ChatCompletionStreamOptionsParam | None]
    response_format: NotRequired[ResponseFormat]
    user: NotRequired[str]
    metadata: NotRequired[Metadata | None]

    # Reasoning parameters
    reasoning_effort: NotRequired[ReasoningEffort | None]

    # Service tier (optional, for request routing)
    service_tier: NotRequired[
        Literal["auto", "default", "flex", "scale", "priority"] | None
    ]


if TYPE_CHECKING:
    from .message_types import ChatCompletionMessageParam
