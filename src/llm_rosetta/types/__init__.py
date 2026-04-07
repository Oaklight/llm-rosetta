"""
LLM-Rosetta Types Package

This package contains type definitions for:
- IR (Intermediate Representation) types
- Provider-specific types (Anthropic, Google, OpenAI)
"""

from .ir import (
    # Content part types
    ContentPart,
    TextPart,
    ImagePart,
    ImageData,
    FilePart,
    FileData,
    ToolCallPart,
    ToolResultPart,
    ReasoningPart,
    RefusalPart,
    CitationPart,
    AudioPart,
    AudioData,
    # Role-specific content types
    SystemContentPart,
    UserContentPart,
    AssistantContentPart,
    ToolContentPart,
    # Message types
    Message,
    BaseMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    LegacyMessage,
    MessageMetadata,
    StreamingMetadata,
    # Tool types
    ToolDefinition,
    ToolChoice,
    ToolCallConfig,
    # Generation configuration types
    GenerationConfig,
    ResponseFormatConfig,
    StreamConfig,
    ReasoningConfig,
    CacheConfig,
    # Request types
    IRRequest,
    # Response types
    IRResponse,
    ExtensionItem,
    SystemEvent,
    BatchMarker,
    SessionControl,
    ToolChainNode,
    UsageInfo,
    FinishReason,
    ChoiceInfo,
    # Stream event types
    IRStreamEvent,
    StreamStartEvent,
    StreamEndEvent,
    ContentBlockStartEvent,
    ContentBlockEndEvent,
    TextDeltaEvent,
    ReasoningDeltaEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    FinishEvent,
    UsageEvent,
    # Backward compatibility types
    IRInput,
    IRInputSimple,
)

__all__ = [
    # Content part types
    "ContentPart",
    "TextPart",
    "ImagePart",
    "ImageData",
    "FilePart",
    "FileData",
    "ToolCallPart",
    "ToolResultPart",
    "ReasoningPart",
    "RefusalPart",
    "CitationPart",
    "AudioPart",
    "AudioData",
    # Role-specific content types
    "SystemContentPart",
    "UserContentPart",
    "AssistantContentPart",
    "ToolContentPart",
    # Message types
    "Message",
    "BaseMessage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "LegacyMessage",
    "MessageMetadata",
    "StreamingMetadata",
    # Tool types
    "ToolDefinition",
    "ToolChoice",
    "ToolCallConfig",
    # Generation configuration types
    "GenerationConfig",
    "ResponseFormatConfig",
    "StreamConfig",
    "ReasoningConfig",
    "CacheConfig",
    # Request types
    "IRRequest",
    # Response types
    "IRResponse",
    "ExtensionItem",
    "SystemEvent",
    "BatchMarker",
    "SessionControl",
    "ToolChainNode",
    "UsageInfo",
    "FinishReason",
    "ChoiceInfo",
    # Stream event types
    "IRStreamEvent",
    "StreamStartEvent",
    "StreamEndEvent",
    "ContentBlockStartEvent",
    "ContentBlockEndEvent",
    "TextDeltaEvent",
    "ReasoningDeltaEvent",
    "ToolCallStartEvent",
    "ToolCallDeltaEvent",
    "FinishEvent",
    "UsageEvent",
    # Backward compatibility types
    "IRInput",
    "IRInputSimple",
]
