"""Deprecated: moved to :mod:`llm_rosetta.ir.model.messages`.

``llm_rosetta.types.ir.messages`` shipped in 0.13.0.  It forwards the names 0.13.0
had, as the same objects; importing it imports the deprecated package
:mod:`llm_rosetta.types.ir`, which warns that it is removed in 0.15.0.
"""

from __future__ import annotations

from llm_rosetta.ir.model.messages import (
    AssistantMessage,
    BaseMessage,
    LegacyMessage,
    Message,
    MessageMetadata,
    StreamingMetadata,
    SystemMessage,
    ToolMessage,
    UserMessage,
    create_assistant_message,
    create_system_message,
    create_tool_message,
    create_user_message,
    is_assistant_message,
    is_message,
    is_system_message,
    is_tool_message,
    is_user_message,
)

__all__ = [
    "AssistantMessage",
    "BaseMessage",
    "LegacyMessage",
    "Message",
    "MessageMetadata",
    "StreamingMetadata",
    "SystemMessage",
    "ToolMessage",
    "UserMessage",
    "create_assistant_message",
    "create_system_message",
    "create_tool_message",
    "create_user_message",
    "is_assistant_message",
    "is_message",
    "is_system_message",
    "is_tool_message",
    "is_user_message",
]
