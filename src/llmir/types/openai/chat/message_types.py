"""OpenAI Chat Completion API Message Types.

This module contains TypedDict replicas of OpenAI SDK's message types.
These types are used for type hints and validation in the LLMIR library.

Supported OpenAI SDK Versions: 1.x.x through 2.14.0
Last Updated: 2025-01-10

Reference: openai.types.chat.*MessageParam
SDK Source: <python_env>/lib/python3.10/site-packages/openai/types/chat/
"""

from typing import Iterable, Literal, Optional, Union

from typing_extensions import NotRequired, Required, TypedDict

# ============================================================================
# Content Part Types
# ============================================================================


class ChatCompletionContentPartTextParam(TypedDict):
    """Text content part parameter.

    Reference: openai.types.chat.ChatCompletionContentPartTextParam
    """

    type: Literal["text"]
    text: str


class ChatCompletionContentPartImageParam(TypedDict, total=False):
    """Image content part parameter.

    Reference: openai.types.chat.ChatCompletionContentPartImageParam
    """

    type: Required[Literal["image_url"]]
    image_url: Required["ImageURL"]


class ImageURL(TypedDict, total=False):
    """Image URL configuration for image content parts.

    Reference: openai.types.chat.ChatCompletionContentPartImageParam.ImageURL

    Args:
        url: The URL of the image, or a base64-encoded image data URL
        detail: Specifies the detail level of the image analysis:
            - "auto": Let the model choose the detail level
            - "low": Low detail, faster processing
            - "high": High detail, more thorough analysis
    """

    url: Required[str]
    detail: NotRequired[Literal["auto", "low", "high"]]


class ChatCompletionContentPartInputAudioParam(TypedDict, total=False):
    """Input audio content part parameter.

    Reference: openai.types.chat.ChatCompletionContentPartInputAudioParam
    """

    type: Required[Literal["input_audio"]]
    input_audio: Required["InputAudio"]


class InputAudio(TypedDict, total=False):
    """Input audio configuration."""

    data: Required[str]  # Base64 encoded audio
    format: Required[Literal["wav", "mp3"]]


# Union of all content part types
ChatCompletionContentPartParam = Union[
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
]


# ============================================================================
# Tool Call Types (for assistant messages)
# ============================================================================


class FunctionCall(TypedDict):
    """Function call (deprecated).

    Reference: openai.types.chat.ChatCompletionMessageParam.FunctionCall
    """

    name: str
    arguments: str  # JSON string


class ChatCompletionMessageToolCallFunction(TypedDict):
    """Tool call function."""

    name: str
    arguments: str  # JSON string


class ChatCompletionMessageToolCallParam(TypedDict):
    """Tool call parameter.

    Reference: openai.types.chat.ChatCompletionMessageToolCallParam
    """

    id: str
    type: Literal["function"]
    function: ChatCompletionMessageToolCallFunction


# ============================================================================
# Message Parameter Types
# ============================================================================


class ChatCompletionSystemMessageParam(TypedDict, total=False):
    """System message parameter.

    Reference: openai.types.chat.ChatCompletionSystemMessageParam
    """

    role: Required[Literal["system"]]
    content: Required[Union[str, Iterable[ChatCompletionContentPartTextParam]]]
    name: NotRequired[str]


class ChatCompletionDeveloperMessageParam(TypedDict, total=False):
    """Developer message parameter.

    Reference: openai.types.chat.ChatCompletionDeveloperMessageParam
    """

    role: Required[Literal["developer"]]
    content: Required[Union[str, Iterable[ChatCompletionContentPartTextParam]]]
    name: NotRequired[str]


class ChatCompletionUserMessageParam(TypedDict, total=False):
    """User message parameter.

    Reference: openai.types.chat.ChatCompletionUserMessageParam
    """

    role: Required[Literal["user"]]
    content: Required[Union[str, Iterable[ChatCompletionContentPartParam]]]
    name: NotRequired[str]


class ChatCompletionAssistantMessageParam(TypedDict, total=False):
    """Assistant message parameter.

    Reference: openai.types.chat.ChatCompletionAssistantMessageParam
    """

    role: Required[Literal["assistant"]]
    content: NotRequired[Union[str, None]]
    name: NotRequired[str]
    tool_calls: NotRequired[Iterable[ChatCompletionMessageToolCallParam]]
    function_call: NotRequired[FunctionCall]  # Deprecated
    refusal: NotRequired[Optional[str]]


class ChatCompletionToolMessageParam(TypedDict, total=False):
    """Tool message parameter.

    Reference: openai.types.chat.ChatCompletionToolMessageParam
    """

    role: Required[Literal["tool"]]
    content: Required[Union[str, Iterable[ChatCompletionContentPartTextParam]]]
    tool_call_id: Required[str]


class ChatCompletionFunctionMessageParam(TypedDict, total=False):
    """Function message parameter (deprecated).

    Reference: openai.types.chat.ChatCompletionFunctionMessageParam
    """

    role: Required[Literal["function"]]
    content: Required[Optional[str]]
    name: Required[str]


# Union of all message parameter types
ChatCompletionMessageParam = Union[
    ChatCompletionDeveloperMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionFunctionMessageParam,
]
