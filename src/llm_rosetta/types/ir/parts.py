"""
LLM-Rosetta - IR Content Parts Types

IR内容部分类型定义
IR content parts type definitions
"""

import sys
from typing import Any, Literal, Union

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required, TypedDict
else:
    from typing_extensions import NotRequired, Required, TypedDict

# ============================================================================
# 基础内容部分类型 Basic content part types
# ============================================================================


class TextPart(TypedDict):
    """纯文本内容
    Plain text content
    """

    type: Required[Literal["text"]]
    text: Required[str]
    provider_metadata: NotRequired[dict[str, Any]]
    cache_hint: NotRequired[dict[str, Any]]


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
    provider_ref: NotRequired[
        dict[str, Any]
    ]  # Provider-specific reference (e.g. file_id)
    cache_hint: NotRequired[dict[str, Any]]


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
    provider_ref: NotRequired[
        dict[str, Any]
    ]  # Provider-specific reference (e.g. file_id)
    cache_hint: NotRequired[dict[str, Any]]


# ============================================================================
# 工具相关内容部分 Tool-related content parts
# ============================================================================


class ToolCallPart(TypedDict):
    """
    工具调用内容。
    Tool call content.

    使用两层类型系统：
    - type: 固定为 "tool_call"
    - tool_type: 区分不同的工具类型（function, mcp, custom, intrinsic）
    Uses a two-layer type system:
    - type: fixed as "tool_call"
    - tool_type: distinguishes different tool types (function, mcp, custom, intrinsic)

    这样设计避免了类型爆炸，同时保持扩展性。
    This design avoids type explosion while maintaining extensibility.

    ``intrinsic`` 表示 provider 托管的内置工具（web search、code execution 等），
    具体种类通过 ``provider_metadata["intrinsic_kind"]`` 区分。
    ``intrinsic`` represents provider-hosted built-in tools (web search, code
    execution, etc.); the specific kind is stored in
    ``provider_metadata["intrinsic_kind"]``.

    provider_metadata字段用于存储provider特定的元数据，例如：
    - Google的thought_signature（Gemini 3必需，Gemini 2.5推荐）
    - intrinsic工具的intrinsic_kind（如 "web_search", "code_execution"）
    - 其他provider的特殊字段
    The provider_metadata field is used to store provider-specific metadata, e.g.:
    - Google's thought_signature (required for Gemini 3, recommended for Gemini 2.5)
    - intrinsic_kind for intrinsic tools (e.g. "web_search", "code_execution")
    - Other provider's special fields

    ``tool_group`` 是本次调用所属工具组的名字，与工具定义上的
    ``ToolDefinition.tool_group`` 是同一身份的两端：客户端按 ``(name, group)``
    发起调用，就把 group 一并带回来；扁平目标上的调用不带该字段。
    ``tool_group`` names the group the called tool belongs to, the call-side
    counterpart of ``ToolDefinition.tool_group``.  A client that addresses a
    tool by ``(name, group)`` gets the group back beside the name; a call on a
    flat target carries none.  Read it directly; converters still accept the
    legacy ``provider_metadata["namespace"]`` spelling on input for
    backward compatibility.
    """

    type: Required[Literal["tool_call"]]
    tool_call_id: Required[str]
    tool_name: Required[str]
    tool_input: Required[dict[str, Any]]
    tool_type: NotRequired[
        Literal[
            "function",
            "mcp",
            "custom",
            "intrinsic",
        ]
    ]  # 默认为 "function" Default is "function"
    tool_group: NotRequired[str]
    """本次调用所属的工具组 The group this call's tool belongs to.

    Present when the client addressed the tool by ``(name, group)``; absent on
    a target that carries no grouping and on any tool declared on its own.
    A converter that writes this field removes a legacy
    ``provider_metadata["namespace"]`` from the same call, so a group is never
    stored twice with the two copies able to disagree.
    """
    provider_metadata: NotRequired[
        dict[str, Any]
    ]  # Provider特定的元数据 Provider-specific metadata
    cache_hint: NotRequired[dict[str, Any]]


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
    # 与对应ToolCallPart的tool_type一致，缺省时由调用方推断。
    # Mirrors the matching ToolCallPart's tool_type; absent means "infer".
    tool_type: NotRequired[
        Literal[
            "function",
            "mcp",
            "custom",
            "intrinsic",
        ]
    ]
    is_error: NotRequired[bool]  # 是否是错误结果 Whether it is an error result
    provider_metadata: NotRequired[
        dict[str, Any]
    ]  # Provider特定的元数据 Provider-specific metadata
    cache_hint: NotRequired[dict[str, Any]]


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

    ``signature`` 与 ``redacted_data`` 是 **provider 专有的不透明值**，仅对签发的
    provider 有效。为避免把它发给别的 provider（对方会以 400 拒绝），转换器在读取
    时会把出处记录在 ``provider_metadata["_opaque_origin"]`` 下：

        {"signature": "<converter_tag>", "redacted_data": "<converter_tag>"}

    未记录出处视为“未知”，按同源处理以兼容手工构造的 IR。跨 provider 转换时：
    Anthropic/OpenAI 丢弃异源值；Google 目标在每轮首个 ``functionCall`` 上注入
    文档规定的 dummy 签名。

    ``signature`` and ``redacted_data`` are **provider-specific opaque values**
    valid only for the provider that issued them.  Converters record the origin
    under ``provider_metadata["_opaque_origin"]`` so a foreign target can drop
    the value (Anthropic/OpenAI) or replace it with Google's documented dummy
    signature.  See :mod:`llm_rosetta.converters.base.helpers.opaque`.
    """

    type: Required[Literal["reasoning"]]
    reasoning: NotRequired[
        str
    ]  # 推理内容，某些provider可能不提供 Reasoning content, may not be provided by some providers
    signature: NotRequired[
        str
    ]  # Provider-specific verification signature (Anthropic thinking, Google thoughtSignature); see _opaque_origin provenance above
    status: NotRequired[
        Literal["in_progress", "completed", "incomplete"]
    ]  # 推理状态 Reasoning status
    provider_metadata: NotRequired[dict[str, Any]]
    cache_hint: NotRequired[dict[str, Any]]
    redacted_data: NotRequired[
        str
    ]  # Provider-specific opaque blob (Anthropic redacted_thinking, OpenAI encrypted_content); see _opaque_origin provenance above


class UrlCitation(TypedDict, total=False):
    """URL引用详情
    URL citation details
    """

    start_index: int
    end_index: int
    title: str
    url: str


class TextCitation(TypedDict, total=False):
    """文本引用详情
    Text citation details
    """

    cited_text: str


class CitationPart(TypedDict):
    """
    引用/注释内容（如OpenAI的annotations、Anthropic的citations）。
    Citation/annotation content (e.g. OpenAI's annotations, Anthropic's citations).

    用于标注信息来源，如网络搜索结果、文档引用等。
    Used to mark information sources, such as web search results, document citations, etc.
    """

    type: Required[Literal["citation"]]
    # OpenAI-style URL citation
    url_citation: NotRequired[UrlCitation]
    # Anthropic-style text citation
    text_citation: NotRequired[TextCitation]


class AudioData(TypedDict):
    """Base64编码的音频数据
    Base64 encoded audio data
    """

    data: Required[str]  # base64编码 base64 encoded
    media_type: Required[str]  # 如 "audio/wav" e.g. "audio/wav"


class AudioPart(TypedDict):
    """
    音频内容（如OpenAI的audio响应）。
    Audio content (e.g. OpenAI's audio response).

    用于音频输出模态数据。
    Used for audio output modality data.
    """

    type: Required[Literal["audio"]]
    audio_data: NotRequired[AudioData]  # base64形式 base64 form
    url: NotRequired[str]  # URL形式 URL form
    detail: NotRequired[
        Literal["auto", "low", "high"]
    ]  # 音频细节级别 Audio detail level
    provider_ref: NotRequired[
        dict[str, Any]
    ]  # Provider-specific reference (e.g. audio_id)


# ============================================================================
# 角色特定内容类型 Role-specific content types
# ============================================================================

# 系统消息内容类型 - 目前只允许文本
SystemContentPart = TextPart

# 用户消息内容类型 - 文本、图像、文件，未来支持音频
# User message content types - text, images, files, future support for audio
UserContentPart = Union[
    TextPart,
    ImagePart,
    FilePart,
    # 未来支持 Future support:
    # AudioPart,
]

# 助手消息内容类型 - 文本、工具调用、推理、拒绝、引用、服务端工具结果，
# 未来支持音频、图像
# Assistant message content types - text, tool calls, reasoning, refusal,
# citations, and server-executed tool results, future support for audio and
# images.  A ``*_tool_result`` that a server tool (web search, code execution)
# ran inside the assistant turn belongs to that turn; the client-result shape
# stays in the separate ``role: "tool"`` message.
AssistantContentPart = Union[
    TextPart,
    ToolCallPart,
    ReasoningPart,
    RefusalPart,
    CitationPart,
    ToolResultPart,
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
    "UrlCitation",
    "TextCitation",
    "AudioPart",
    "AudioData",
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
