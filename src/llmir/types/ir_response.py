"""
LLMIR - IR Response Types

IR响应消息类型，基于 ir_design_final.md 的最终设计
IR response message types based on the final design of ir_design_final.md
"""

from typing import Any, Dict, List, Literal, TypedDict, Union

from typing_extensions import NotRequired, Required

# ============================================================================
# 核心消息类型 Core message types
# ============================================================================


class MessageMetadata(TypedDict, total=False):
    """消息的元数据，用于存储额外信息
    Metadata of the message, used to store extra information
    """

    message_id: str
    timestamp: str
    streaming: "StreamingMetadata"
    custom: Dict[str, Any]


class StreamingMetadata(TypedDict, total=False):
    """流式传输的元数据
    Metadata for streaming transmission
    """

    is_streaming: bool
    is_final: bool
    chunk_index: int


class Message(TypedDict):
    """
    核心消息类型，代表对话中的一条消息。
    Core message type, represents a message in the conversation.

    这是IR的主要组成部分，90%的场景只需要使用这个类型。
    This is the main component of IR, 90% of scenarios only need this type.
    """

    role: Required[Literal["system", "user", "assistant"]]
    content: Required[List["ContentPart"]]
    metadata: NotRequired[MessageMetadata]


# ============================================================================
# 内容部分类型 Content part types
# ============================================================================


class TextPart(TypedDict):
    """纯文本内容
    Plain text content
    """

    type: Required[Literal["text"]]
    text: Required[str]


class ImageData(TypedDict):
    """Base64编码的图像数据
    Base64 encoded image data
    """

    data: Required[str]  # base64编码 base64 encoded
    media_type: Required[str]  # 如 "image/png" e.g. "image/png"


class ImagePart(TypedDict):
    """图像内容，支持URL或base64
    Image content, supports URL or base64
    """

    type: Required[Literal["image"]]
    image_url: NotRequired[str]  # URL形式 URL form
    image_data: NotRequired[ImageData]  # base64形式 base64 form
    detail: NotRequired[Literal["auto", "low", "high"]]  # OpenAI特性 OpenAI feature


class FileData(TypedDict):
    """Base64编码的文件数据
    Base64 encoded file data
    """

    data: Required[str]  # base64编码 base64 encoded
    media_type: Required[str]  # 如 "application/pdf" e.g. "application/pdf"


class FilePart(TypedDict):
    """
    文件内容，支持多种文件类型。
    File content, supports multiple file types.

    Examples:
        - PDF文档 PDF document
        - 音频文件 Audio file
        - 视频文件 Video file
    """

    type: Required[Literal["file"]]
    file_url: NotRequired[str]  # URL形式 URL form
    file_data: NotRequired[FileData]  # base64形式 base64 form
    file_name: NotRequired[str]
    file_type: NotRequired[str]  # MIME type


class ToolCallPart(TypedDict):
    """
    工具调用内容。
    Tool call content.

    使用两层类型系统：
    - type: 固定为 "tool_call"
    - tool_type: 区分不同的工具类型（function, mcp, web_search等）
    Uses a two-layer type system:
    - type: fixed as "tool_call"
    - tool_type: distinguishes different tool types (function, mcp, web_search, etc.)

    这样设计避免了类型爆炸，同时保持扩展性。
    This design avoids type explosion while maintaining extensibility.

    provider_metadata字段用于存储provider特定的元数据，例如：
    - Google的thought_signature（Gemini 3必需，Gemini 2.5推荐）
    - 其他provider的特殊字段
    The provider_metadata field is used to store provider-specific metadata, e.g.:
    - Google's thought_signature (required for Gemini 3, recommended for Gemini 2.5)
    - Other provider's special fields
    """

    type: Required[Literal["tool_call"]]
    tool_call_id: Required[str]
    tool_name: Required[str]
    tool_input: Required[Dict[str, Any]]
    tool_type: NotRequired[
        Literal[
            "function",
            "mcp",
            "web_search",
            "code_interpreter",
            "file_search",
        ]
    ]  # 默认为 "function" Default is "function"
    provider_metadata: NotRequired[
        Dict[str, Any]
    ]  # Provider特定的元数据 Provider-specific metadata


class ToolResultPart(TypedDict):
    """
    工具调用的结果。
    Tool call result.

    对应一个ToolCallPart，通过tool_call_id关联。
    Corresponds to a ToolCallPart, linked by tool_call_id.
    """

    type: Required[Literal["tool_result"]]
    tool_call_id: Required[str]
    result: Required[Any]  # 可以是字符串、对象等 Can be string, object, etc.
    is_error: NotRequired[bool]  # 是否是错误结果 Whether it is an error result


class RefusalPart(TypedDict):
    """
    拒绝响应内容（如OpenAI的refusal）。
    Refusal response content (e.g. OpenAI's refusal).

    当模型拒绝回答用户请求时使用，常见于安全过滤。
    Used when the model refuses to answer the user's request, common in safety filtering.
    """

    type: Required[Literal["refusal"]]
    refusal: Required[str]  # 拒绝原因文本 The refusal reason text


class ReasoningPart(TypedDict):
    """
    推理过程内容（如OpenAI的reasoning或Anthropic的thinking）。
    Reasoning process content (e.g. OpenAI's reasoning or Anthropic's thinking).

    用于存储模型的思考过程，通常不显示给用户。
    Used to store the model's thought process, usually not shown to the user.
    """

    type: Required[Literal["reasoning"]]
    reasoning: Required[str]
    status: NotRequired[
        Literal["in_progress", "completed", "incomplete"]
    ]  # 推理状态 Reasoning status


class CitationPart(TypedDict):
    """
    引用/注释内容（如OpenAI的annotations、Anthropic的citations）。
    Citation/annotation content (e.g. OpenAI's annotations, Anthropic's citations).

    用于标注信息来源，如网络搜索结果、文档引用等。
    Used to标注信息来源，如网络搜索结果、文档引用等。
    Used to mark information sources, such as web search results, document citations, etc.
    """

    type: Required[Literal["citation"]]
    # OpenAI-style URL citation
    url_citation: NotRequired[
        Dict[
            Literal["start_index", "end_index", "title", "url"],
            Any,
        ]
    ]
    # Anthropic-style text citation
    text_citation: NotRequired[Dict[Literal["cited_text"], Any]]


class AudioPart(TypedDict):
    """
    音频内容（如OpenAI的audio响应）。
    Audio content (e.g. OpenAI's audio response).

    用于音频输出模态数据。
    Used for audio output modality data.
    """

    type: Required[Literal["audio"]]
    audio_id: Required[str]  # 音频ID Audio ID
    detail: NotRequired[
        Literal["auto", "low", "high"]
    ]  # 音频细节级别 Audio detail level


# 内容部分联合类型 Content part union type
ContentPart = Union[
    TextPart,
    ImagePart,
    FilePart,
    ToolCallPart,
    ToolResultPart,
    ReasoningPart,
    RefusalPart,
    CitationPart,
    AudioPart,
]

# ============================================================================
# 扩展项类型（用于特殊场景） Extension item types (for special scenarios)
# ============================================================================


class SystemEvent(TypedDict):
    """
    系统级事件，用于记录会话状态变化。
    System-level events, used to record session state changes.

    Examples:
        - 会话开始/结束 Session start/end
        - 会话暂停/恢复 Session pause/resume
        - 超时警告 Timeout warning
        - 错误事件 Error event
    """

    type: Required[Literal["system_event"]]
    event_type: Required[
        Literal[
            "session_start",
            "session_pause",
            "session_resume",
            "session_timeout",
            "session_end",
            "error",
            "warning",
        ]
    ]
    timestamp: Required[str]  # ISO 8601格式 ISO 8601 format
    event_data: NotRequired[Dict[str, Any]]
    message: NotRequired[str]


class BatchMarker(TypedDict):
    """
    批次标记，用于标记一组相关的操作。
    Batch marker, used to mark a group of related operations.

    Examples:
        - 并行工具调用的开始/结束 Start/end of parallel tool calls
        - 部分结果的进度跟踪 Progress tracking of partial results
    """

    type: Required[Literal["batch_marker"]]
    batch_id: Required[str]
    batch_type: Required[Literal["start", "end", "partial"]]
    total_items: NotRequired[int]
    completed_items: NotRequired[int]
    metadata: NotRequired[Dict[str, Any]]


class SessionControl(TypedDict):
    """
    会话控制指令，用于控制工具调用的执行。
    Session control instructions, used to control the execution of tool calls.

    Examples:
        - 取消工具调用 Cancel tool call
        - 修改工具调用参数 Modify tool call parameters
        - 暂停/恢复工具执行 Pause/resume tool execution
    """

    type: Required[Literal["session_control"]]
    control_type: Required[
        Literal[
            "cancel_tool",
            "modify_tool",
            "pause_tool",
            "resume_tool",
        ]
    ]
    target_id: Required[str]  # 目标tool_call_id Target tool_call_id
    reason: NotRequired[str]
    new_input: NotRequired[Dict[str, Any]]  # 用于modify_tool Used for modify_tool


class ToolChainNode(TypedDict):
    """
    工具链节点，用于表示工具调用的依赖关系。
    Tool chain node, used to represent dependencies between tool calls.

    支持DAG结构，一个工具的输出可以作为另一个工具的输入。
    Supports DAG structure, the output of one tool can be used as the input of another.

    Examples:
        - 搜索 → 总结 Search → Summarize
        - 数据获取 → 分析 → 可视化 Data acquisition → Analysis → Visualization
    """

    type: Required[Literal["tool_chain_node"]]
    node_id: Required[str]
    tool_call: Required[ToolCallPart]
    depends_on: NotRequired[List[str]]  # 依赖的节点ID列表 List of dependent node IDs
    auto_execute: NotRequired[bool]  # 是否自动执行 Whether to auto execute


# 扩展项联合类型 Extension item union type
ExtensionItem = Union[
    SystemEvent,
    BatchMarker,
    SessionControl,
    ToolChainNode,
]

# ============================================================================
# 响应统计信息 Response statistics
# ============================================================================


class UsageInfo(TypedDict):
    """
    Token使用统计信息。
    Token usage statistics.

    来自各SDK的usage/usage_metadata字段，用于计费和监控。
    From each SDK's usage/usage_metadata field, used for billing and monitoring.
    """

    prompt_tokens: Required[int]  # 输入Token数 Input token count
    completion_tokens: Required[int]  # 输出Token数 Output token count
    reasoning_tokens: NotRequired[
        int
    ]  # 推理内容消耗的Token数 Token count for reasoning/thinking content (Google: thoughts_token_count)
    total_tokens: Required[int]  # 总Token数 Total token count

    # 详细统计（可选） Detailed statistics (optional)
    prompt_tokens_details: NotRequired[
        Dict[str, int]
    ]  # 输入详细统计 Input details (如缓存Token数 e.g. cached token count)
    completion_tokens_details: NotRequired[
        Dict[str, int]
    ]  # 输出详细统计 Output details (如推理Token数 e.g. reasoning token count)
    cache_read_tokens: NotRequired[int]  # 缓存读取Token数 Cache read token count


class FinishReason(TypedDict):
    """
    停止原因信息。
    Stop reason information.

    来自各SDK的finish_reason/stop_reason字段，说明模型停止生成的原因。
    From each SDK's finish_reason/stop_reason field, explaining why the model stopped generating.
    """

    reason: Required[
        Literal[
            "stop",  # 正常停止 Normal stop
            "length",  # 达到最大长度 Reached max length
            "tool_calls",  # 工具调用 Tool calls
            "content_filter",  # 内容过滤 Content filter
            "refusal",  # 拒绝回答 Refusal
            "error",  # 错误 Error
            "cancelled",  # 取消 Cancelled
        ]
    ]
    # 原始停止序列（部分SDK支持） Original stop sequence (supported by some SDKs)
    stop_sequence: NotRequired[str]


class ChoiceInfo(TypedDict):
    """
    选择结果信息（对应OpenAI的Choice）。
    Choice result information (corresponds to OpenAI's Choice).

    用于存储单个选择的结果，包含消息、停止原因、logprobs等。
    Used to store the result of a single choice, including message, stop reason, logprobs, etc.
    """

    index: Required[int]  # 选择索引 Choice index
    message: Required[Message]  # 生成的消息 Generated message
    finish_reason: Required[FinishReason]  # 停止原因 Stop reason
    logprobs: NotRequired[Dict[str, Any]]  # Log概率信息 Log probability information


# ============================================================================
# 顶层响应类型 Top-level response types
# ============================================================================


class IRResponse(TypedDict):
    """
    统一的IR响应类型。
    Unified IR response type.

    包含响应的所有信息：ID、时间戳、模型、选择列表、使用统计等。
    Contains all information of the response: ID, timestamp, model, choices list, usage statistics, etc.
    """

    # 必需字段 Required fields
    id: Required[str]  # 响应唯一ID Response unique ID
    object: Required[Literal["response"]]  # 对象类型 Object type
    created: Required[
        int
    ]  # 创建时间戳（Unix时间戳） Creation timestamp (Unix timestamp)
    model: Required[str]  # 使用的模型 Used model
    choices: Required[List[ChoiceInfo]]  # 选择结果列表 Choice result list

    # 可选字段 Optional fields
    usage: NotRequired[UsageInfo]  # Token使用统计 Token usage statistics
    service_tier: NotRequired[str]  # 服务等级 Service tier
    system_fingerprint: NotRequired[str]  # 系统指纹 System fingerprint


# 完整的IR输入（支持扩展项） Complete IR input (supports extension items)
IRInput = List[Union[Message, ExtensionItem]]

# 简化的IR输入（只有消息） Simplified IR input (messages only)
IRInputSimple = List[Message]

# ============================================================================
# 类型守卫函数 Type guard functions
# ============================================================================


def is_message(item: Union[Message, ExtensionItem]) -> bool:
    """判断是否是Message
    Determine if it is a Message
    """
    return "role" in item


def is_extension_item(item: Union[Message, ExtensionItem]) -> bool:
    """判断是否是ExtensionItem
    Determine if it is an ExtensionItem
    """
    return "type" in item and item.get("type") in [
        "system_event",
        "batch_marker",
        "session_control",
        "tool_chain_node",
    ]


def is_text_part(part: ContentPart) -> bool:
    """判断是否是文本内容
    Determine if it is text content
    """
    return part.get("type") == "text"


def is_tool_call_part(part: ContentPart) -> bool:
    """判断是否是工具调用
    Determine if it is a tool call
    """
    return part.get("type") == "tool_call"


def is_tool_result_part(part: ContentPart) -> bool:
    """判断是否是工具结果
    Determine if it is a tool result
    """
    return part.get("type") == "tool_result"


# ============================================================================
# 辅助函数 - 用于处理IR消息 Helper functions - for processing IR messages
# ============================================================================


def extract_text_content(message: Message) -> str:
    """从消息中提取所有文本内容
    Extract all text content from message

    Args:
        message: IR格式的消息 IR format message

    Returns:
        拼接后的文本内容 Concatenated text content
    """
    texts = []
    for part in message.get("content", []):
        if is_text_part(part):
            text = part.get("text", "")
            if text is not None:  # 确保text不是None Ensure text is not None
                texts.append(str(text))
    return "".join(texts)


def extract_tool_calls(
    message: Message, limit: Union[int, None] = None
) -> List[ToolCallPart]:
    """从消息中提取工具调用
    Extract tool calls from message

    Args:
        message: IR格式的消息 IR format message
        limit: 限制返回的工具调用数量。None表示返回所有，1表示只返回第一个
               Limit the number of tool calls returned. None means return all, 1 means return only the first

    Returns:
        工具调用部分的列表 List of tool call parts

    Examples:
        >>> # 获取所有工具调用 Get all tool calls
        >>> all_calls = extract_tool_calls(message)
        >>>
        >>> # 只获取第一个工具调用 Get only the first tool call
        >>> first_call = extract_tool_calls(message, limit=1)
        >>> if first_call:
        >>>     tool_call = first_call[0]
        >>>
        >>> # 获取前3个工具调用 Get the first 3 tool calls
        >>> first_three = extract_tool_calls(message, limit=3)
    """
    tool_calls = [
        part  # type: ignore
        for part in message.get("content", [])
        if is_tool_call_part(part)
    ]

    if limit is not None:
        return tool_calls[:limit]
    return tool_calls


def create_tool_result_message(
    tool_call_id: str, result: Any, is_error: bool = False
) -> Message:
    """创建工具结果消息
    Create tool result message

    Args:
        tool_call_id: 工具调用ID Tool call ID
        result: 工具执行结果 Tool execution result
        is_error: 是否为错误结果 Whether it is an error result

    Returns:
        IR格式的工具结果消息 IR format tool result message
    """
    return Message(
        role="user",
        content=[
            ToolResultPart(
                type="tool_result",
                tool_call_id=tool_call_id,
                result=result,
                is_error=is_error,
            )
        ],
    )


# ============================================================================
# 导出的主要类型 Main Exported Types
# ============================================================================

__all__ = [
    # 核心类型 Core types
    "Message",
    "MessageMetadata",
    "StreamingMetadata",
    "ContentPart",
    "ExtensionItem",
    "IRInput",
    "IRInputSimple",
    # 内容部分类型 Content part types
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
    # 扩展项类型 Extension item types
    "SystemEvent",
    "BatchMarker",
    "SessionControl",
    "ToolChainNode",
    # 响应统计类型 Response statistics types
    "UsageInfo",
    "FinishReason",
    "ChoiceInfo",
    # 顶层响应类型 Top-level response types
    "IRResponse",
    # 类型守卫函数 Type guard functions
    "is_message",
    "is_extension_item",
    "is_text_part",
    "is_tool_call_part",
    "is_tool_result_part",
    # 辅助函数 Helper functions
    "extract_text_content",
    "extract_tool_calls",
    "create_tool_result_message",
]
