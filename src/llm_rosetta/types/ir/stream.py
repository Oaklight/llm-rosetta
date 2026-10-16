"""
LLM-Rosetta - IR Stream Event Types

IR流式事件类型定义，用于支持SSE chunk级别的实时转换
IR stream event type definitions for supporting SSE chunk-level real-time conversion

包含以下事件类型：
- StreamStartEvent: 流开始事件（会话级元数据）
- StreamEndEvent: 流结束事件
- ContentBlockStartEvent: 内容块开始事件
- ContentBlockEndEvent: 内容块结束事件
- TextDeltaEvent: 文本增量事件
- ReasoningDeltaEvent: 推理/思考内容增量事件
- RefusalDeltaEvent: 拒绝内容增量事件
- ToolCallStartEvent: 工具调用开始事件
- ToolCallDeltaEvent: 工具调用增量事件
- ToolResultEvent: 工具结果事件（服务端工具结果，整块到达）
- FinishEvent: 完成事件
- UsageEvent: 使用统计事件

Contains the following event types:
- StreamStartEvent: Stream start event (session-level metadata)
- StreamEndEvent: Stream end event
- ContentBlockStartEvent: Content block start event
- ContentBlockEndEvent: Content block end event
- TextDeltaEvent: Text delta event
- ReasoningDeltaEvent: Reasoning/thinking content delta event
- RefusalDeltaEvent: Refusal content delta event
- ToolCallStartEvent: Tool call start event
- ToolCallDeltaEvent: Tool call delta event
- ToolResultEvent: Tool result event (server-tool result, delivered whole)
- FinishEvent: Finish event
- UsageEvent: Usage statistics event
"""

import sys
from typing import Any, Literal, Union

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required, TypedDict
else:
    from typing_extensions import NotRequired, Required, TypedDict

from .parts import CitationPart
from .passthrough import ProviderPassthroughEvent
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
    block_index: NotRequired[int]  # Provider content block index (e.g. Anthropic)
    choice_index: NotRequired[int]
    provider_metadata: NotRequired[dict[str, Any]]


class ReasoningDeltaEvent(TypedDict):
    """Reasoning/thinking content delta event.

    Emitted when a new reasoning/thinking text fragment is received from the model.
    """

    type: Required[Literal["reasoning_delta"]]
    reasoning: Required[str]  # The reasoning text delta
    signature: NotRequired[str]  # Anthropic thinking signature delta
    encrypted_content: NotRequired[str]  # Responses reasoning encrypted state
    block_index: NotRequired[int]  # Provider content block index (e.g. Anthropic)
    choice_index: NotRequired[int]
    provider_metadata: NotRequired[dict[str, Any]]


class RefusalDeltaEvent(TypedDict):
    """Refusal content delta event.

    Emitted when a new refusal text fragment is received from the model.
    """

    type: Required[Literal["refusal_delta"]]
    refusal: Required[str]
    block_index: NotRequired[int]
    choice_index: NotRequired[int]


class CitationDeltaEvent(TypedDict):
    """Citation delta event.

    Emitted when the provider streams a citation/annotation for the text it has
    been sending — Anthropic's ``citations_delta``, the Responses API's
    annotation events.  One event per citation, in the order the provider gave
    them, mirroring the non-streaming ``citation`` content part so a streamed
    answer and a whole one carry the same value.

    ``block_index`` names the content block the citation annotates where the
    dialect numbers its blocks.
    """

    type: Required[Literal["citation_delta"]]
    citation: Required[CitationPart]
    block_index: NotRequired[int]
    choice_index: NotRequired[int]
    provider_metadata: NotRequired[dict[str, Any]]


class ToolCallStartEvent(TypedDict):
    """Tool call start event.

    Emitted when a new tool call begins, providing the tool call ID and name.

    ``tool_group`` names the group the called tool belongs to, the streaming
    counterpart of :attr:`~llm_rosetta.types.ir.parts.ToolCallPart.tool_group`.
    Present when the client addressed the tool by ``(name, group)``; absent on
    a flat target.  The source converter reads it here and carries it onto the
    later done/completed items.

    ``intrinsic_kind`` is the streaming counterpart of
    :attr:`~llm_rosetta.types.ir.parts.ToolCallPart.intrinsic_kind`: present
    when ``tool_type == "intrinsic"``, holding the canonical kind (e.g.
    ``"web_search"``).  A reader falls back to the legacy
    ``provider_metadata["intrinsic_kind"]`` only when this field is absent; the
    tool name is never used as a fallback.
    """

    type: Required[Literal["tool_call_start"]]
    tool_call_id: Required[str]
    tool_name: Required[str]
    tool_type: NotRequired[str]  # "function" (default), "custom", etc.
    tool_group: NotRequired[str]  # Group the called tool belongs to
    intrinsic_kind: NotRequired[str]  # Canonical kind for an intrinsic call
    cache_hint: NotRequired[dict[str, Any]]  # Block-level cache breakpoint
    tool_call_index: NotRequired[int]  # Index for multiple parallel tool calls
    choice_index: NotRequired[int]
    provider_metadata: NotRequired[dict[str, Any]]


class ToolCallDeltaEvent(TypedDict):
    """Tool call arguments delta event.

    Emitted when a new fragment of tool call arguments JSON string is received.

    ``intrinsic_kind`` is the streaming counterpart of
    :attr:`~llm_rosetta.types.ir.parts.ToolCallPart.intrinsic_kind`, carried on
    *every* event of a provider-executed call and not only on the
    :class:`ToolCallStartEvent` that opened it: a reader that sees a fragment
    without the start (a start this dialect dropped, a delta replayed on its
    own) still knows which server tool it belongs to.  It is read from the
    field first, with the legacy ``provider_metadata["intrinsic_kind"]``
    spelling as a fallback, exactly as on a start event.
    """

    type: Required[Literal["tool_call_delta"]]
    tool_call_id: Required[str]
    arguments_delta: Required[str]  # JSON string fragment
    intrinsic_kind: NotRequired[str]  # Canonical kind for an intrinsic call
    block_index: NotRequired[int]  # Provider content block index (e.g. Anthropic)
    tool_call_index: NotRequired[int]  # Index for multiple parallel tool calls
    choice_index: NotRequired[int]
    signature: NotRequired[str]  # Provider signature on a tool-call delta
    provider_metadata: NotRequired[dict[str, Any]]


class ToolResultEvent(TypedDict):
    """Tool result event.

    A tool result the provider executed and returned inside the turn.  Anthropic
    delivers server-tool results (``web_search_tool_result``,
    ``code_execution_tool_result``, …) whole in a single ``content_block_start``,
    with no deltas to follow, so like :class:`ToolCallStartEvent` this event
    carries everything at once and has no delta counterpart.

    Fields mirror :class:`~llm_rosetta.types.ir.parts.ToolResultPart`, so the
    streaming and non-streaming halves describe the same thing.  In particular
    ``result`` is ``Any``, not a list: a web-search result is a list of result
    blocks, but a code-execution result can be a string (or an object).

    ``intrinsic_kind`` is the streaming counterpart of
    :attr:`~llm_rosetta.types.ir.parts.ToolResultPart.intrinsic_kind`: present
    when ``tool_type == "intrinsic"``, holding the canonical kind.  A reader
    falls back to the legacy ``provider_metadata["intrinsic_kind"]`` only when
    this field is absent; the tool name is never used as a fallback.
    """

    type: Required[Literal["tool_result"]]
    tool_call_id: Required[str]  # Provider's tool_use_id
    result: Required[Any]  # Can be a string, list, object, etc.
    cache_hint: NotRequired[dict[str, Any]]  # Block-level cache breakpoint
    intrinsic_kind: NotRequired[str]  # Canonical kind for an intrinsic result
    tool_type: NotRequired[
        Literal["function", "mcp", "custom", "intrinsic"]
    ]  # absent means "infer", as on ToolResultPart
    is_error: NotRequired[bool]
    block_index: NotRequired[int]  # Provider content block index (e.g. Anthropic)
    provider_metadata: NotRequired[dict[str, Any]]


class FinishEvent(TypedDict):
    """Finish event.

    Emitted when the model finishes generating for a choice.
    """

    type: Required[Literal["finish"]]
    finish_reason: Required[FinishReason]
    choice_index: NotRequired[int]
    provider_metadata: NotRequired[dict[str, Any]]


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
    RefusalDeltaEvent,
    CitationDeltaEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolResultEvent,
    FinishEvent,
    UsageEvent,
    ProviderPassthroughEvent,
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
    "RefusalDeltaEvent",
    "CitationDeltaEvent",
    "ToolCallStartEvent",
    "ToolCallDeltaEvent",
    "ToolResultEvent",
    "FinishEvent",
    "UsageEvent",
    "ProviderPassthroughEvent",
    "IRStreamEvent",
]
