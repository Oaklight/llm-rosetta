"""
LLM-Rosetta - IR Response Types

IR响应类型定义，包含响应统计信息
IR response type definitions including response statistics
"""

import sys
from typing import Any, Literal

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required, TypedDict
else:
    from typing_extensions import NotRequired, Required, TypedDict

from .extensions import ExtensionItem, is_extension_item
from .messages import Message

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
        dict[str, int]
    ]  # 输入详细统计 Input details (如缓存Token数 e.g. cached token count)
    completion_tokens_details: NotRequired[
        dict[str, int]
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
    logprobs: NotRequired[dict[str, Any]]  # Log概率信息 Log probability information


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
    choices: Required[list[ChoiceInfo]]  # 选择结果列表 Choice result list

    # 可选字段 Optional fields
    usage: NotRequired[UsageInfo]  # Token使用统计 Token usage statistics
    service_tier: NotRequired[str]  # 服务等级 Service tier
    system_fingerprint: NotRequired[str]  # 系统指纹 System fingerprint


# ============================================================================
# 导出的主要类型 Main Exported Types
# ============================================================================

__all__ = [
    # 响应统计类型 Response statistics types
    "UsageInfo",
    "FinishReason",
    "ChoiceInfo",
    # 顶层响应类型 Top-level response types
    "IRResponse",
    # 从 extensions 模块重新导出 Re-exported from extensions module
    "ExtensionItem",
    "is_extension_item",
]
