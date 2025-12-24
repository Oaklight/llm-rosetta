"""
LLM Provider Converter - IR (Intermediate Representation) Types

基于 ir_design_final.md 的最终设计实现
"""

from typing import Any, Dict, List, Literal, TypedDict, Union

from typing_extensions import NotRequired, Required

# ============================================================================
# 核心消息类型
# ============================================================================


class MessageMetadata(TypedDict, total=False):
    """消息的元数据，用于存储额外信息"""

    message_id: str
    timestamp: str
    streaming: "StreamingMetadata"
    custom: Dict[str, Any]


class StreamingMetadata(TypedDict, total=False):
    """流式传输的元数据"""

    is_streaming: bool
    is_final: bool
    chunk_index: int


class Message(TypedDict):
    """
    核心消息类型，代表对话中的一条消息。

    这是IR的主要组成部分，90%的场景只需要使用这个类型。
    """

    role: Required[Literal["system", "user", "assistant"]]
    content: Required[List["ContentPart"]]
    metadata: NotRequired[MessageMetadata]


# ============================================================================
# 内容部分类型
# ============================================================================


class TextPart(TypedDict):
    """纯文本内容"""

    type: Required[Literal["text"]]
    text: Required[str]


class ImageData(TypedDict):
    """Base64编码的图像数据"""

    data: Required[str]  # base64编码
    media_type: Required[str]  # 如 "image/png"


class ImagePart(TypedDict):
    """图像内容，支持URL或base64"""

    type: Required[Literal["image"]]
    image_url: NotRequired[str]  # URL形式
    image_data: NotRequired[ImageData]  # base64形式
    detail: NotRequired[Literal["auto", "low", "high"]]  # OpenAI特性


class FileData(TypedDict):
    """Base64编码的文件数据"""

    data: Required[str]  # base64编码
    media_type: Required[str]  # 如 "application/pdf"


class FilePart(TypedDict):
    """
    文件内容，支持多种文件类型。

    Examples:
        - PDF文档
        - 音频文件
        - 视频文件
    """

    type: Required[Literal["file"]]
    file_url: NotRequired[str]  # URL形式
    file_data: NotRequired[FileData]  # base64形式
    file_name: NotRequired[str]
    file_type: NotRequired[str]  # MIME type


class ToolCallPart(TypedDict):
    """
    工具调用内容。

    使用两层类型系统：
    - type: 固定为 "tool_call"
    - tool_type: 区分不同的工具类型（function, mcp, web_search等）

    这样设计避免了类型爆炸，同时保持扩展性。
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
    ]  # 默认为 "function"


class ToolResultPart(TypedDict):
    """
    工具调用的结果。

    对应一个ToolCallPart，通过tool_call_id关联。
    """

    type: Required[Literal["tool_result"]]
    tool_call_id: Required[str]
    result: Required[Any]  # 可以是字符串、对象等
    is_error: NotRequired[bool]  # 是否是错误结果


class ReasoningPart(TypedDict):
    """
    推理过程内容（如OpenAI的reasoning）。

    用于存储模型的思考过程，通常不显示给用户。
    """

    type: Required[Literal["reasoning"]]
    reasoning: Required[str]


# 内容部分联合类型
ContentPart = Union[
    TextPart,
    ImagePart,
    FilePart,
    ToolCallPart,
    ToolResultPart,
    ReasoningPart,
]

# ============================================================================
# 扩展项类型（用于特殊场景）
# ============================================================================


class SystemEvent(TypedDict):
    """
    系统级事件，用于记录会话状态变化。

    Examples:
        - 会话开始/结束
        - 会话暂停/恢复
        - 超时警告
        - 错误事件
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
    timestamp: Required[str]  # ISO 8601格式
    event_data: NotRequired[Dict[str, Any]]
    message: NotRequired[str]


class BatchMarker(TypedDict):
    """
    批次标记，用于标记一组相关的操作。

    Examples:
        - 并行工具调用的开始/结束
        - 部分结果的进度跟踪
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

    Examples:
        - 取消工具调用
        - 修改工具调用参数
        - 暂停/恢复工具执行
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
    target_id: Required[str]  # 目标tool_call_id
    reason: NotRequired[str]
    new_input: NotRequired[Dict[str, Any]]  # 用于modify_tool


class ToolChainNode(TypedDict):
    """
    工具链节点，用于表示工具调用的依赖关系。

    支持DAG结构，一个工具的输出可以作为另一个工具的输入。

    Examples:
        - 搜索 → 总结
        - 数据获取 → 分析 → 可视化
    """

    type: Required[Literal["tool_chain_node"]]
    node_id: Required[str]
    tool_call: Required[ToolCallPart]
    depends_on: NotRequired[List[str]]  # 依赖的节点ID列表
    auto_execute: NotRequired[bool]  # 是否自动执行


# 扩展项联合类型
ExtensionItem = Union[
    SystemEvent,
    BatchMarker,
    SessionControl,
    ToolChainNode,
]

# ============================================================================
# 顶层类型
# ============================================================================

# 完整的IR输入（支持扩展项）
IRInput = List[Union[Message, ExtensionItem]]

# 简化的IR输入（只有消息）
IRInputSimple = List[Message]

# ============================================================================
# 类型守卫函数
# ============================================================================


def is_message(item: Union[Message, ExtensionItem]) -> bool:
    """判断是否是Message"""
    return "role" in item


def is_extension_item(item: Union[Message, ExtensionItem]) -> bool:
    """判断是否是ExtensionItem"""
    return "type" in item and item.get("type") in [
        "system_event",
        "batch_marker",
        "session_control",
        "tool_chain_node",
    ]


def is_text_part(part: ContentPart) -> bool:
    """判断是否是文本内容"""
    return part.get("type") == "text"


def is_tool_call_part(part: ContentPart) -> bool:
    """判断是否是工具调用"""
    return part.get("type") == "tool_call"


def is_tool_result_part(part: ContentPart) -> bool:
    """判断是否是工具结果"""
    return part.get("type") == "tool_result"


# ============================================================================
# 工具定义和选择类型
# ============================================================================


class ToolDefinition(TypedDict):
    """工具定义"""

    type: Required[
        Literal["function", "mcp", "web_search", "code_interpreter", "file_search"]
    ]
    name: Required[str]
    description: NotRequired[str]
    parameters: NotRequired[Dict[str, Any]]  # JSON Schema
    required_parameters: NotRequired[List[str]]
    metadata: NotRequired[Dict[str, Any]]


class ToolChoice(TypedDict):
    """工具选择配置"""

    mode: Required[Literal["none", "auto", "any", "tool"]]
    tool_name: NotRequired[str]  # 当mode为"tool"时必需
    disable_parallel: NotRequired[bool]  # 控制是否禁用并行工具使用


# ============================================================================
# 导出的主要类型
# ============================================================================

__all__ = [
    # 核心类型
    "Message",
    "ContentPart",
    "ExtensionItem",
    "IRInput",
    "IRInputSimple",
    # 内容部分类型
    "TextPart",
    "ImagePart",
    "FilePart",
    "ToolCallPart",
    "ToolResultPart",
    "ReasoningPart",
    # 扩展项类型
    "SystemEvent",
    "BatchMarker",
    "SessionControl",
    "ToolChainNode",
    # 工具相关类型
    "ToolDefinition",
    "ToolChoice",
    # 类型守卫函数
    "is_message",
    "is_extension_item",
    "is_text_part",
    "is_tool_call_part",
    "is_tool_result_part",
]
