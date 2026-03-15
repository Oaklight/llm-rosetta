"""
LLM-Rosetta - IR Content Parts Types

IR内容部分类型定义
IR content parts type definitions
"""

import sys
from typing import Any, Dict, Literal, TypedDict, Union

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required
else:
    from typing_extensions import NotRequired, Required

# ============================================================================
# 基础内容部分类型 Basic content part types
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


# ============================================================================
# 工具相关内容部分 Tool-related content parts
# ============================================================================


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


# ============================================================================
# 特殊内容部分 Special content parts
# ============================================================================


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

    有些provider只返回signature而不返回完整的reasoning内容。
    Some providers only return signature without full reasoning content.
    """

    type: Required[Literal["reasoning"]]
    reasoning: NotRequired[
        str
    ]  # 推理内容，某些provider可能不提供 Reasoning content, may not be provided by some providers
    signature: NotRequired[
        str
    ]  # 推理签名，某些provider只提供这个 Reasoning signature, some providers only provide this
    status: NotRequired[
        Literal["in_progress", "completed", "incomplete"]
    ]  # 推理状态 Reasoning status


class CitationPart(TypedDict):
    """
    引用/注释内容（如OpenAI的annotations、Anthropic的citations）。
    Citation/annotation content (e.g. OpenAI's annotations, Anthropic's citations).

    用于标注信息来源，如网络搜索结果、文档引用等。
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


# ============================================================================
# 角色特定内容类型 Role-specific content types
# ============================================================================

# 系统消息内容类型 - 目前只允许文本
SystemContentPart = TextPart

# 用户消息内容类型 - 文本、图像，未来支持文件、音频
UserContentPart = Union[
    TextPart,
    ImagePart,
    # 未来支持 Future support:
    # FilePart,
    # AudioPart,
]

# 助手消息内容类型 - 文本、工具调用、推理、引用，未来支持音频、图像
AssistantContentPart = Union[
    TextPart,
    ToolCallPart,
    ReasoningPart,
    CitationPart,
    # 未来支持 Future support:
    # AudioPart,
    # ImagePart,
]

# 工具消息内容类型 - 只允许工具结果
ToolContentPart = ToolResultPart

# ============================================================================
# 内容部分联合类型 Content part union type
# ============================================================================

# 通用内容部分联合类型 General content part union type
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
# 类型守卫函数 Type guard functions
# ============================================================================


# 类型守卫函数已移动到 type_guards.py 中的通用 is_part_type 函数
# Type guard functions have been moved to the generic is_part_type function in type_guards.py


# ============================================================================
# 导出的主要类型 Main Exported Types
# ============================================================================

__all__ = [
    # 基础内容部分类型 Basic content part types
    "TextPart",
    "ImagePart",
    "ImageData",
    "FilePart",
    "FileData",
    # 工具相关内容部分 Tool-related content parts
    "ToolCallPart",
    "ToolResultPart",
    # 特殊内容部分 Special content parts
    "ReasoningPart",
    "RefusalPart",
    "CitationPart",
    "AudioPart",
    # 角色特定内容类型 Role-specific content types
    "SystemContentPart",
    "UserContentPart",
    "AssistantContentPart",
    "ToolContentPart",
    # 联合类型 Union types
    "ContentPart",
    # 类型守卫函数已移动到 type_guards.py
    # Type guard functions moved to type_guards.py
]
