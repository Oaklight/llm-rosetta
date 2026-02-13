"""Anthropic Messages API types (TypedDict replicas).

This package contains TypedDict replicas of Anthropic SDK types for use in LLMIR.
These types are used for type hints and validation in the conversion layer.
"""

from .request_types import (
    CacheControlEphemeralParam,
    InputSchema,
    InputSchemaTyped,
    MessageCreateParams,
    MessageCreateParamsBase,
    MessageCreateParamsNonStreaming,
    MessageCreateParamsStreaming,
    MessageParam,
    MetadataParam,
    TextBlockParam,
    TextCitationParam,
    ThinkingConfigDisabledParam,
    ThinkingConfigEnabledParam,
    ThinkingConfigParam,
    ToolChoiceAnyParam,
    ToolChoiceAutoParam,
    ToolChoiceNoneParam,
    ToolChoiceParam,
    ToolChoiceToolParam,
    ToolParam,
)
from .response_types import (
    CacheCreation,
    ContentBlock,
    Message,
    ServerToolUsage,
    StopReason,
    TextBlock,
    TextCitation,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)

__all__ = [
    # Request types
    "CacheControlEphemeralParam",
    "InputSchema",
    "InputSchemaTyped",
    "MessageCreateParams",
    "MessageCreateParamsBase",
    "MessageCreateParamsNonStreaming",
    "MessageCreateParamsStreaming",
    "MessageParam",
    "MetadataParam",
    "TextBlockParam",
    "TextCitationParam",
    "ThinkingConfigDisabledParam",
    "ThinkingConfigEnabledParam",
    "ThinkingConfigParam",
    "ToolChoiceAnyParam",
    "ToolChoiceAutoParam",
    "ToolChoiceNoneParam",
    "ToolChoiceParam",
    "ToolChoiceToolParam",
    "ToolParam",
    # Response types
    "CacheCreation",
    "ContentBlock",
    "Message",
    "ServerToolUsage",
    "StopReason",
    "TextBlock",
    "TextCitation",
    "ThinkingBlock",
    "ToolUseBlock",
    "Usage",
]
