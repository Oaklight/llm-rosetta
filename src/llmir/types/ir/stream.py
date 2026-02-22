"""
LLMIR - IR Stream Event Types

IR流式事件类型定义，用于支持SSE chunk级别的实时转换
IR stream event type definitions for supporting SSE chunk-level real-time conversion

包含以下事件类型：
- StreamStartEvent: 流开始事件（会话级元数据）
- StreamEndEvent: 流结束事件
- ContentBlockStartEvent: 内容块开始事件
- ContentBlockEndEvent: 内容块结束事件
- TextDeltaEvent: 文本增量事件
- ReasoningDeltaEvent: 推理/思考内容增量事件
- ToolCallStartEvent: 工具调用开始事件
- ToolCallDeltaEvent: 工具调用增量事件
- FinishEvent: 完成事件
- UsageEvent: 使用统计事件

Contains the following event types:
- StreamStartEvent: Stream start event (session-level metadata)
- StreamEndEvent: Stream end event
- ContentBlockStartEvent: Content block start event
- ContentBlockEndEvent: Content block end event
- TextDeltaEvent: Text delta event
- ReasoningDeltaEvent: Reasoning/thinking content delta event
- ToolCallStartEvent: Tool call start event
- ToolCallDeltaEvent: Tool call delta event
- FinishEvent: Finish event
- UsageEvent: Usage statistics event
"""

from typing import Literal, Union

from typing_extensions import NotRequired, Required, TypedDict

from .response import FinishReason, UsageInfo

# ============================================================================
# Stream lifecycle event types
# ============================================================================


class StreamStartEvent(TypedDict):
    """Emitted at the beginning of a stream. Carries session-level metadata.

    This event is synthesized by the converter when the first provider chunk
    arrives, providing a unified place for response-level information such as
    the response ID and model name.
    """

    type: Required[Literal["stream_start"]]
    response_id: Required[str]  # Provider response ID (e.g., chatcmpl-xxx, msg_xxx)
    model: Required[str]  # Model name
    created: NotRequired[int]  # Unix timestamp (optional)


class StreamEndEvent(TypedDict):
    """Emitted at the end of a stream.

    Signals that no more events will follow. Converters emit this after
    processing the final provider chunk.
    """

    type: Required[Literal["stream_end"]]


class ContentBlockStartEvent(TypedDict):
    """Emitted when a new content block begins.

    Content blocks group related deltas (e.g., a text block, a thinking
    block, or a tool_use block). The block_type indicates what kind of
    deltas to expect.
    """

    type: Required[Literal["content_block_start"]]
    block_index: Required[int]  # 0-based block index
    block_type: Required[str]  # "text", "thinking", "tool_use", etc.


class ContentBlockEndEvent(TypedDict):
    """Emitted when a content block ends.

    Signals that no more deltas will arrive for the given block_index.
    """

    type: Required[Literal["content_block_end"]]
    block_index: Required[int]  # 0-based block index


# ============================================================================
# Stream delta event types
# ============================================================================


class TextDeltaEvent(TypedDict):
    """Text content delta event.

    Emitted when a new text fragment is received from the model.
    """

    type: Required[Literal["text_delta"]]
    text: Required[str]
    choice_index: NotRequired[int]


class ReasoningDeltaEvent(TypedDict):
    """Reasoning/thinking content delta event.

    Emitted when a new reasoning/thinking text fragment is received from the model.
    """

    type: Required[Literal["reasoning_delta"]]
    reasoning: Required[str]  # The reasoning text delta
    signature: NotRequired[str]  # Anthropic thinking signature delta
    choice_index: NotRequired[int]


class ToolCallStartEvent(TypedDict):
    """Tool call start event.

    Emitted when a new tool call begins, providing the tool call ID and name.
    """

    type: Required[Literal["tool_call_start"]]
    tool_call_id: Required[str]
    tool_name: Required[str]
    tool_call_index: NotRequired[int]  # Index for multiple parallel tool calls
    choice_index: NotRequired[int]


class ToolCallDeltaEvent(TypedDict):
    """Tool call arguments delta event.

    Emitted when a new fragment of tool call arguments JSON string is received.
    """

    type: Required[Literal["tool_call_delta"]]
    tool_call_id: Required[str]
    arguments_delta: Required[str]  # JSON string fragment
    tool_call_index: NotRequired[int]  # Index for multiple parallel tool calls
    choice_index: NotRequired[int]


class FinishEvent(TypedDict):
    """Finish event.

    Emitted when the model finishes generating for a choice.
    """

    type: Required[Literal["finish"]]
    finish_reason: Required[FinishReason]
    choice_index: NotRequired[int]


class UsageEvent(TypedDict):
    """Usage statistics event.

    Emitted when token usage statistics are available (typically at the end of stream).
    """

    type: Required[Literal["usage"]]
    usage: Required[UsageInfo]


# ============================================================================
# Union type
# ============================================================================

IRStreamEvent = Union[
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
]

# ============================================================================
# Exports
# ============================================================================

__all__ = [
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
    "IRStreamEvent",
]
