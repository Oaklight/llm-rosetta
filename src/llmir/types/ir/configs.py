"""
LLMIR - IR Generation Configuration Types

IR生成控制配置类型定义
IR generation control configuration type definitions

包含模型生成行为控制的各种配置参数：
- 温度控制、采样参数
- 推理配置
- 流式输出配置
- 响应格式配置
- 缓存配置

Contains various configuration parameters for controlling model generation behavior:
- Temperature control, sampling parameters
- Reasoning configuration
- Streaming configuration
- Response format configuration
- Cache configuration
"""

from typing import Any, Dict, Iterable, Literal

from typing_extensions import TypedDict

# ============================================================================
# 生成控制配置 Generation control configuration
# ============================================================================


class GenerationConfig(TypedDict, total=False):
    """生成控制参数
    Generation control parameters

    统一了各provider的生成控制参数，映射关系：
    Unified generation control parameters across providers, mapping:

    - temperature: 所有provider都支持 All providers support
    - top_p: 所有provider都支持 All providers support
    - top_k: Anthropic, Google支持 Anthropic, Google support
    - max_tokens:
        - OpenAI Chat: max_completion_tokens
        - OpenAI Responses: max_output_tokens
        - Anthropic: max_tokens (必需 required)
        - Google: config.max_output_tokens
    - frequency_penalty: OpenAI, Google支持 OpenAI, Google support
    - presence_penalty: OpenAI, Google支持 OpenAI, Google support
    - seed: OpenAI, Google支持 OpenAI, Google support
    - logprobs: 各provider实现不同 Different implementations across providers
    """

    # 温度控制 Temperature control (0.0-2.0 for OpenAI, 0.0-1.0 for Anthropic/Google)
    temperature: float

    # Nucleus采样 Nucleus sampling (0.0-1.0)
    top_p: float

    # Top-k采样 Top-k sampling (Anthropic, Google)
    top_k: int

    # 最大生成token数 Maximum tokens to generate
    # OpenAI Chat: max_completion_tokens
    # OpenAI Responses: max_output_tokens
    # Anthropic: max_tokens (必需)
    # Google: config.max_output_tokens
    max_tokens: int

    # 停止序列 Stop sequences
    # OpenAI: stop (str | List[str])
    # Anthropic: stop_sequences (List[str])
    # Google: config.stop_sequences (List[str])
    stop_sequences: Iterable[str]

    # 截断策略 Truncation strategy (OpenAI Responses, 少见)
    truncation: Literal["auto", "disabled"]

    # 频率惩罚 Frequency penalty (-2.0 to 2.0, OpenAI, Google)
    frequency_penalty: float

    # 存在惩罚 Presence penalty (-2.0 to 2.0, OpenAI, Google)
    presence_penalty: float

    # Logit偏置 Logit bias (OpenAI)
    logit_bias: Dict[str, int]

    # 随机种子 Random seed (OpenAI, Google)
    seed: int

    # Log概率 Log probabilities
    logprobs: bool
    top_logprobs: int

    # 生成选择数量 Number of choices to generate (OpenAI)
    n: int

    # 候选数量 Number of candidates (Google, 少见)
    candidate_count: int


# ============================================================================
# 推理配置 Reasoning configuration
# ============================================================================


class ReasoningConfig(TypedDict, total=False):
    """推理配置
    Reasoning configuration

    不同provider有不同的实现方式：
    - OpenAI: reasoning_effort (low/medium/high)
    - Anthropic: thinking (ThinkingConfig) + max_tokens + budget_tokens
    - Google: thinking_config.thoughts_token_limit
    """

    effort: Literal["low", "medium", "high"]  # OpenAI: reasoning_effort
    type: Literal["enabled", "disabled"]  # Anthropic: thinking.type
    budget_tokens: (
        int  # Anthropic: thinking.budget_tokens / Google: thoughts_token_limit
    )


# ============================================================================
# 流式输出配置 Streaming configuration
# ============================================================================


class StreamConfig(TypedDict, total=False):
    """流式输出配置
    Streaming configuration
    """

    enabled: bool  # 是否启用流式输出
    include_usage: bool  # OpenAI: stream_options.include_usage


# ============================================================================
# 响应格式配置 Response format configuration
# ============================================================================


class ResponseFormatConfig(TypedDict, total=False):
    """响应格式配置
    Response format configuration

    用于控制响应内容的格式：
    - OpenAI: response_format
    - Google: response_mime_type + response_schema
    """

    type: Literal["text", "json_object", "json_schema"]
    json_schema: Dict[str, Any]  # 当type为json_schema时使用
    mime_type: str  # Google的response_mime_type


# ============================================================================
# 缓存配置 Cache configuration
# ============================================================================


class CacheConfig(TypedDict, total=False):
    """缓存配置
    Cache configuration (OpenAI)

    用于提示缓存（Prompt Caching）功能。
    """

    key: str  # prompt_cache_key
    retention: Literal["in-memory", "24h"]  # prompt_cache_retention


# ============================================================================
# 导出的主要类型 Main Exported Types
# ============================================================================

__all__ = [
    # 生成控制配置 Generation control configuration
    "GenerationConfig",
    # 推理配置 Reasoning configuration
    "ReasoningConfig",
    # 流式输出配置 Streaming configuration
    "StreamConfig",
    # 响应格式配置 Response format configuration
    "ResponseFormatConfig",
    # 缓存配置 Cache configuration
    "CacheConfig",
]
