"""
LLM-Rosetta - IR Embedding Types

Embedding 中间表示类型定义
Embedding intermediate representation type definitions

Covers 5 major embedding API format families:
- OpenAI: input + encoding_format + dimensions → data[].embedding
- Google GenAI: embedContent / batchEmbedContents with Content/Parts + taskType
- Cohere: texts + input_type + embedding_types (multi-format) → embeddings.{type}[][]
- Voyage: input + input_type + output_dtype + encoding_format → data[].embedding
- Jina: input + task + embedding_type + dimensions → data[].embedding
"""

import sys
from typing import Any, Literal

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required, TypedDict
else:
    from typing_extensions import NotRequired, Required, TypedDict


# ============================================================================
# 任务类型 Task type
# ============================================================================

EmbeddingTaskType = Literal[
    "retrieval_query",
    "retrieval_document",
    "semantic_similarity",
    "classification",
    "clustering",
    "question_answering",
    "fact_verification",
    "code_retrieval_query",
    "code_retrieval_document",
]
"""
归一化的 embedding 任务类型（所有 provider 枚举值的超集）。
Canonical embedding task type (superset of all provider enum values).

Provider 映射 Provider mapping:
- retrieval_query:        Google RETRIEVAL_QUERY, Cohere search_query, Voyage query, Jina retrieval.query
- retrieval_document:     Google RETRIEVAL_DOCUMENT, Cohere search_document, Voyage document, Jina retrieval.passage
- semantic_similarity:    Google SEMANTIC_SIMILARITY, Jina text-matching
- classification:         Google CLASSIFICATION, Cohere classification, Jina classification
- clustering:             Google CLUSTERING, Cohere clustering, Jina clustering (v5)
- question_answering:     Google QUESTION_ANSWERING
- fact_verification:      Google FACT_VERIFICATION
- code_retrieval_query:   Google CODE_RETRIEVAL_QUERY, Jina code.query (v4)
- code_retrieval_document: Jina code.passage (v4)
"""


# ============================================================================
# 编码格式 Encoding format
# ============================================================================

EmbeddingEncodingFormat = Literal[
    "float",
    "base64",
    "int8",
    "uint8",
    "binary",
    "ubinary",
]
"""
输出数据类型与序列化格式的统一表示。
Unified representation of output data type and serialization format.

合并了 Voyage 的 output_dtype（数据类型）和 encoding_format（序列化）两个维度。
Cohere 支持单次请求返回多种格式，这属于 provider 特有行为，不进入 IR。
Merges Voyage's output_dtype (data type) and encoding_format (serialization) axes.
Cohere's simultaneous multi-type request is provider-specific, not modeled in IR.

- float:   float32 数组 float32 array (default, all providers)
- base64:  base64 编码的浮点字节 base64-encoded float bytes (OpenAI, Cohere, Voyage, Jina)
- int8:    有符号 8 位整数数组 signed 8-bit int array (Cohere, Voyage)
- uint8:   无符号 8 位整数数组 unsigned 8-bit int array (Cohere, Voyage)
- binary:  打包有符号二进制 packed signed binary (Cohere, Voyage, Jina)
- ubinary: 打包无符号二进制 packed unsigned binary (Cohere, Voyage, Jina)
"""


# ============================================================================
# 用量统计 Usage statistics
# ============================================================================


class EmbeddingUsageInfo(TypedDict, total=False):
    """
    Embedding 专用的 token 用量统计。
    Embedding-specific token usage statistics.

    与 chat 的 UsageInfo 分离，因为 embedding 没有 completion_tokens。
    Cohere 的 billed_units、Jina 的多模态 token 细分不进入 IR，
    converter 可将其存入 response 级别的 provider_extensions。

    Separate from chat UsageInfo because embedding has no completion_tokens.
    Cohere's billed_units and Jina's per-modality token breakdown are not
    included in the IR — converters can stash them in provider_extensions.
    """

    total_tokens: int
    prompt_tokens: int


# ============================================================================
# 嵌入结果 Embedding result
# ============================================================================


class EmbeddingItem(TypedDict):
    """
    单个嵌入向量结果。
    Single embedding vector result.

    embedding 字段的实际类型取决于 encoding_format:
    - float/int8/uint8: list[float] 或 list[int]
    - binary/ubinary:   list[int]（打包后长度为原始维度的 1/8）
    - base64:           str（base64 编码的字节）

    The actual type of the embedding field depends on encoding_format:
    - float/int8/uint8: list[float] or list[int]
    - binary/ubinary:   list[int] (packed, 1/8 of original dimensions)
    - base64:           str (base64-encoded bytes)
    """

    index: Required[int]
    embedding: Required[list[float] | list[int] | str]


# ============================================================================
# 请求类型 Request type
# ============================================================================


class IREmbeddingRequest(TypedDict):
    """
    统一的 IR Embedding 请求类型。
    Unified IR embedding request type.

    必需字段 Required fields:
    - model: 模型 ID
    - input: 归一化的文本列表（所有 provider 格式在 IR 边界上统一为 list[str]）

    可选字段 Optional fields:
    - task_type: 归一化的任务类型（Google taskType, Cohere input_type, Jina task 等）
    - dimensions: 输出向量维度（Google outputDimensionality, Voyage/Cohere output_dimension）
    - encoding_format: 输出编码格式
    - truncation: 是否截断过长输入（简化自 Cohere 的 NONE/START/END 枚举）
    - user: 最终用户标识符（OpenAI 透传）
    - provider_extensions: provider 特有参数的兜底字段
    """

    # ========== 必需字段 Required Fields ==========
    model: Required[str]
    input: Required[list[str]]

    # ========== 可选字段 Optional Fields ==========
    task_type: NotRequired[EmbeddingTaskType]
    dimensions: NotRequired[int]
    encoding_format: NotRequired[EmbeddingEncodingFormat]
    truncation: NotRequired[bool]
    user: NotRequired[str]

    # ========== Provider 特定扩展 Provider-specific Extensions ==========
    provider_extensions: NotRequired[dict[str, Any]]


# ============================================================================
# 响应类型 Response type
# ============================================================================


class IREmbeddingResponse(TypedDict):
    """
    统一的 IR Embedding 响应类型。
    Unified IR embedding response type.

    遵循 OpenAI 约定使用 "list" 作为 object 字段值。
    Google 的 embedding.values 和 Cohere 的 embeddings.{type}[][] 由
    converter 归一化为 data 列表。
    encoding_format 回显请求中指定的格式，帮助消费者解读 embedding 字段。

    Follows the OpenAI convention of using "list" as the object field value.
    Google's embedding.values and Cohere's embeddings.{type}[][] are
    normalized to the data list by converters.
    encoding_format echoes the requested format to help consumers
    interpret the embedding field.
    """

    # ========== 必需字段 Required Fields ==========
    object: Required[Literal["list"]]
    model: Required[str]
    data: Required[list[EmbeddingItem]]

    # ========== 可选字段 Optional Fields ==========
    id: NotRequired[str]
    usage: NotRequired[EmbeddingUsageInfo]
    encoding_format: NotRequired[EmbeddingEncodingFormat]

    # ========== Provider 特定扩展 Provider-specific Extensions ==========
    provider_extensions: NotRequired[dict[str, Any]]


# ============================================================================
# 导出的主要类型 Main Exported Types
# ============================================================================

__all__ = [
    "EmbeddingTaskType",
    "EmbeddingEncodingFormat",
    "EmbeddingUsageInfo",
    "EmbeddingItem",
    "IREmbeddingRequest",
    "IREmbeddingResponse",
]
