"""OpenAI Chat Completion API Response Types.

This module contains TypedDict replicas of OpenAI SDK's response types.
These types are used for type hints and validation in the LLM-Rosetta library.

Supported OpenAI SDK Versions: 1.x.x through 2.14.0
Last Updated: 2025-01-10

Reference: openai.types.chat.ChatCompletion
SDK Source: <python_env>/lib/python3.10/site-packages/openai/types/chat/
"""

import sys
from typing import Literal, TypedDict, Union

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required
else:
    from typing_extensions import NotRequired, Required


# ============================================================================
# Usage Statistics Types
# ============================================================================


class PromptTokensDetails(TypedDict, total=False):
    """Prompt tokens details.

    Reference: openai.types.CompletionUsage.PromptTokensDetails
    """

    cached_tokens: NotRequired[int | None]
    audio_tokens: NotRequired[int | None]


class CompletionTokensDetails(TypedDict, total=False):
    """Completion tokens details.

    Reference: openai.types.CompletionUsage.CompletionTokensDetails
    """

    reasoning_tokens: NotRequired[int | None]
    audio_tokens: NotRequired[int | None]
    accepted_prediction_tokens: NotRequired[int | None]
    rejected_prediction_tokens: NotRequired[int | None]


class CompletionUsage(TypedDict, total=False):
    """Token usage statistics.

    Reference: openai.types.CompletionUsage
    """

    prompt_tokens: Required[int]
    completion_tokens: Required[int]
    total_tokens: Required[int]
    prompt_tokens_details: NotRequired[PromptTokensDetails | None]
    completion_tokens_details: NotRequired[CompletionTokensDetails | None]


# ============================================================================
# Message Content Types (Response)
# ============================================================================


class Function(TypedDict):
    """Function in tool call.

    Reference: openai.types.chat.ChatCompletionMessageToolCall.Function
    """

    name: str
    arguments: str  # JSON string


class ChatCompletionMessageFunctionToolCall(TypedDict):
    """Function tool call in response.

    Reference: openai.types.chat.ChatCompletionMessageToolCall
    """

    id: str
    type: Literal["function"]
    function: Function


class ChatCompletionMessageCustomToolCall(TypedDict):
    """Custom tool call in response.

    Reference: openai.types.chat.ChatCompletionMessageToolCall (custom type)
    """

    id: str
    type: str  # Custom type string
    # Additional fields depend on custom tool type


ChatCompletionMessageToolCallUnion = Union[
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageCustomToolCall,
]


# ============================================================================
# Annotation Types
# ============================================================================


class AnnotationURLCitation(TypedDict):
    """URL citation annotation.

    Reference: openai.types.chat.ChatCompletionMessage.Annotation.URLCitation
    """

    start_index: int
    end_index: int
    title: str
    url: str


class Annotation(TypedDict):
    """Annotation in message.

    Reference: openai.types.chat.ChatCompletionMessage.Annotation
    """

    type: Literal["url_citation"]
    url_citation: AnnotationURLCitation


# ============================================================================
# Audio Types (Response)
# ============================================================================


class ChatCompletionAudio(TypedDict):
    """Audio in response message.

    Reference: openai.types.chat.ChatCompletionMessage.Audio
    """

    id: str
    expires_at: int
    data: str  # Base64 encoded audio
    transcript: str


# ============================================================================
# Message Types (Response)
# ============================================================================


class ChatCompletionMessage(TypedDict, total=False):
    """Chat completion message in response.

    Reference: openai.types.chat.ChatCompletionMessage
    """

    role: Required[Literal["assistant"]]
    content: NotRequired[str | None]
    refusal: NotRequired[str | None]
    tool_calls: NotRequired[list[ChatCompletionMessageToolCallUnion] | None]
    function_call: NotRequired["FunctionCallResponse | None"]  # Deprecated
    annotations: NotRequired[list[Annotation] | None]
    audio: NotRequired[ChatCompletionAudio | None]


class FunctionCallResponse(TypedDict):
    """Function call in response (deprecated).

    Reference: openai.types.chat.ChatCompletionMessage.FunctionCall
    """

    name: str
    arguments: str  # JSON string


# ============================================================================
# Logprobs Types
# ============================================================================


class TopLogprob(TypedDict):
    """Top logprob entry.

    Reference: openai.types.chat.ChatCompletionTokenLogprob.TopLogprob
    """

    token: str
    logprob: float
    bytes: list[int] | None


class ChatCompletionTokenLogprob(TypedDict):
    """Token logprob information.

    Reference: openai.types.chat.ChatCompletionTokenLogprob
    """

    token: str
    logprob: float
    bytes: list[int] | None
    top_logprobs: list[TopLogprob]


class ChoiceLogprobs(TypedDict):
    """Logprobs for a choice.

    Reference: openai.types.chat.ChatCompletion.Choice.Logprobs
    """

    content: list[ChatCompletionTokenLogprob] | None
    refusal: list[ChatCompletionTokenLogprob] | None


# ============================================================================
# Choice Types
# ============================================================================


class Choice(TypedDict, total=False):
    """A choice in chat completion response.

    Reference: openai.types.chat.ChatCompletion.Choice
    """

    index: Required[int]
    message: Required[ChatCompletionMessage]
    finish_reason: Required[
        Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    ]
    logprobs: NotRequired[ChoiceLogprobs | None]


# ============================================================================
# Main Response Type
# ============================================================================


class ChatCompletion(TypedDict, total=False):
    """Chat completion response.

    This is the main response type for OpenAI Chat Completion API.

    Reference: openai.types.chat.ChatCompletion
    """

    id: Required[str]
    object: Required[Literal["chat.completion"]]
    created: Required[int]
    model: Required[str]
    choices: Required[list[Choice]]
    usage: NotRequired[CompletionUsage | None]
    service_tier: NotRequired[
        Literal["auto", "default", "flex", "scale", "priority"] | None
    ]
    system_fingerprint: NotRequired[str | None]
