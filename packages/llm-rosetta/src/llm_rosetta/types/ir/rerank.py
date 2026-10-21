"""
LLM-Rosetta - IR Rerank Types

Rerank 中间表示类型定义
Rerank intermediate representation type definitions

Covers 4 major rerank API format families:
- Jina: results + top-level usage
- Cohere: results + meta.billed_units + meta.tokens
- Siliconflow: Cohere variant with richer meta
- Voyage: data + top-level usage (OpenAI Embeddings-style)
"""

import sys
from typing import Any, Literal

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required, TypedDict
else:
    from typing_extensions import NotRequired, Required, TypedDict


# ============================================================================
# 文档类型 Document type
# ============================================================================


class RerankDocument(TypedDict):
    """
    归一化的文档表示。
    Normalized document representation.

    所有 provider 接受 list[str]，converter 在 IR 边界上
    将其归一化为 list[RerankDocument]。
    All providers accept list[str]; converters normalize
    to list[RerankDocument] at the IR boundary.
    """

    text: Required[str]


# ============================================================================
# 用量统计 Usage statistics
# ============================================================================


class RerankUsageInfo(TypedDict, total=False):
    """
    Rerank 专用的 token 用量统计。
    Rerank-specific token usage statistics.

    与 chat 的 UsageInfo 分离，因为 rerank 没有 completion_tokens。
    Cohere 特有的计费字段（search_units、image_tokens）不进入 IR，
    converter 可将其存入 response 级别的 provider_extensions。

    Separate from chat UsageInfo because rerank has no completion_tokens.
    Cohere-specific billing fields (search_units, image_tokens) are not
    included in the IR — converters can stash them in response-level
    provider_extensions if needed.
    """

    total_tokens: int
    prompt_tokens: int
    cached_tokens: int


# ============================================================================
# 排序结果 Ranked result
# ============================================================================


class RerankResultItem(TypedDict):
    """
    单个排序结果。
    Single ranked result.

    index 和 relevance_score 在所有 4 种格式族中通用。
    index and relevance_score are universal across all 4 format families.
    """

    index: Required[int]
    relevance_score: Required[float]
    document: NotRequired[RerankDocument]


# ============================================================================
# 请求类型 Request type
# ============================================================================


class IRRerankRequest(TypedDict):
    """
    统一的 IR Rerank 请求类型。
    Unified IR rerank request type.

    必需字段 Required fields:
    - model: 模型 ID
    - query: 搜索查询
    - documents: 归一化文档列表

    可选字段 Optional fields:
    - top_n: 返回前 N 个结果（Voyage 使用 top_k）
    - return_documents: 是否在结果中包含文档文本
    - max_tokens_per_doc: 每个文档的截断长度（Cohere）
    - truncation: 是否截断长输入（Voyage）
    - provider_extensions: provider 特有参数的兜底字段
    """

    # ========== 必需字段 Required Fields ==========
    model: Required[str]
    query: Required[str]
    documents: Required[list[RerankDocument]]

    # ========== 可选字段 Optional Fields ==========
    top_n: NotRequired[int]
    return_documents: NotRequired[bool]
    max_tokens_per_doc: NotRequired[int]
    truncation: NotRequired[bool]

    # ========== Provider 特定扩展 Provider-specific Extensions ==========
    provider_extensions: NotRequired[dict[str, Any]]


# ============================================================================
# 响应类型 Response type
# ============================================================================


class IRRerankResponse(TypedDict):
    """
    统一的 IR Rerank 响应类型。
    Unified IR rerank response type.

    results 为统一字段名——Voyage 的 data 在 converter 中重命名。
    results is the canonical field name — Voyage's data is renamed
    by the converter.
    """

    # ========== 必需字段 Required Fields ==========
    object: Required[Literal["rerank"]]
    model: Required[str]
    results: Required[list[RerankResultItem]]

    # ========== 可选字段 Optional Fields ==========
    id: NotRequired[str]
    usage: NotRequired[RerankUsageInfo]


# ============================================================================
# 导出的主要类型 Main Exported Types
# ============================================================================

__all__ = [
    "RerankDocument",
    "RerankUsageInfo",
    "RerankResultItem",
    "IRRerankRequest",
    "IRRerankResponse",
]
