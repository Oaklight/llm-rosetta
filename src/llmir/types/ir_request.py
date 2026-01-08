"""
LLMIR - IR Request Types

基于 sdk_body_structures.md 设计的统一请求参数类型
Unified request parameter types based on sdk_body_structures.md

设计原则：
- 核心参数（90%场景）：必需且普遍支持
- 可选参数（少见）：通过 provider_extensions 支持
- 渐进式复杂度：简单场景不需要了解所有参数
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Literal,
    TypedDict,
    Union,
)

from typing_extensions import NotRequired, Required

if TYPE_CHECKING:
    from .ir_response import IRInput


# ============================================================================
# 工具相关类型 Tool Related Types
# ============================================================================


class ToolDefinition(TypedDict):
    """工具定义
    Tool definition

    统一了各provider的工具定义格式：
    - OpenAI Chat: {"type": "function", "function": {...}}
    - OpenAI Responses: {"type": "function", "name": "...", ...}
    - Anthropic: {"name": "...", "description": "...", "input_schema": {...}}
    - Google: {"function_declarations": [{"name": "...", ...}]}
    """

    type: Required[
        Literal["function", "mcp", "web_search", "code_interpreter", "file_search"]
    ]
    name: Required[str]
    description: NotRequired[str]
    parameters: NotRequired[Dict[str, Any]]  # JSON Schema
    required_parameters: NotRequired[Iterable[str]]
    metadata: NotRequired[Dict[str, Any]]


class ToolChoice(TypedDict):
    """工具选择配置
    Tool choice configuration

    统一了各provider的工具选择策略：
    - none: 不使用工具
    - auto: 自动决定是否使用工具
    - any: 必须使用某个工具（Anthropic的"any"）
    - tool: 使用指定的工具（需要tool_name）
    """

    mode: Required[Literal["none", "auto", "any", "tool"]]
    tool_name: NotRequired[str]  # 当mode为"tool"时必需


class ToolCallConfig(TypedDict, total=False):
    """工具调用配置（少见参数）
    Tool call configuration (less common parameters)

    这些参数不是所有provider都支持，放在这里但主要通过provider_extensions使用：
    - disable_parallel: 禁用并行工具调用
    - max_calls: 最大工具调用数
    """

    disable_parallel: bool  # Anthropic: disable_parallel_tool_use
    max_calls: int  # OpenAI Responses: max_tool_calls


# ============================================================================
# 生成控制参数 Generation Control Parameters
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
# 推理配置 Reasoning Configuration
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
# 流式输出配置 Streaming Configuration
# ============================================================================


class StreamConfig(TypedDict, total=False):
    """流式输出配置
    Streaming configuration
    """

    enabled: bool  # 是否启用流式输出
    include_usage: bool  # OpenAI: stream_options.include_usage


# ============================================================================
# 响应格式配置 Response Format Configuration
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
# 缓存配置 Cache Configuration
# ============================================================================


class CacheConfig(TypedDict, total=False):
    """缓存配置
    Cache configuration (OpenAI)

    用于提示缓存（Prompt Caching）功能。
    """

    key: str  # prompt_cache_key
    retention: Literal["in-memory", "24h"]  # prompt_cache_retention


# ============================================================================
# 主请求类型 Main Request Type
# ============================================================================


class IRRequest(TypedDict):
    """
    统一的IR请求类型
    Unified IR request type

    这个类型整合了所有provider的核心请求参数，提供统一的接口。

    必需字段 Required fields:
    - model: 模型ID
    - messages: 消息列表

    可选字段按功能分组:
    - 系统指令: system_instruction
    - 工具相关: tools, tool_choice, tool_config
    - 生成控制: generation, response_format
    - 流式输出: stream
    - 推理配置: reasoning
    - 缓存配置: cache
    """

    # ========== 必需字段 Required Fields ==========
    model: Required[str]
    messages: Required["IRInput"]

    # ========== 系统指令 System Instruction ==========
    # 映射关系:
    # - OpenAI Chat: messages[0] with role="system"
    # - OpenAI Responses: instructions
    # - Anthropic: system
    # - Google: config.system_instruction
    system_instruction: NotRequired[Union[str, Iterable[Dict[str, Any]]]]

    # ========== 工具相关 Tool Related ==========
    tools: NotRequired[Iterable[ToolDefinition]]
    tool_choice: NotRequired[ToolChoice]
    tool_config: NotRequired[ToolCallConfig]

    # ========== 生成控制 Generation Control ==========
    generation: NotRequired[GenerationConfig]
    response_format: NotRequired[ResponseFormatConfig]

    # ========== 流式输出 Streaming ==========
    stream: NotRequired[StreamConfig]

    # ========== 推理配置 Reasoning ==========
    reasoning: NotRequired[ReasoningConfig]

    # ========== 缓存配置 Cache ==========
    cache: NotRequired[CacheConfig]

    # ========== Provider特定扩展 Provider-specific Extensions ==========
    # 用于存储少见或provider特定的参数:
    # - 元数据相关: metadata, user (少见)
    # - 音频配置: audio (少见)
    # - 服务配置: service_tier, store, background (少见)
    # - 安全配置: safety_identifier, safety_settings (少见)
    # - 会话配置: conversation, previous_response_id (少见，OpenAI Responses)
    # - 其他: truncation已移到generation，verbosity等
    provider_extensions: NotRequired[Dict[str, Any]]


# ============================================================================
# 导出的主要类型 Main Exported Types
# ============================================================================

__all__ = [
    # 主请求类型
    "IRRequest",
    # 工具相关
    "ToolDefinition",
    "ToolChoice",
    "ToolCallConfig",
    # 配置类型
    "GenerationConfig",
    "ResponseFormatConfig",
    "StreamConfig",
    "ReasoningConfig",
    "CacheConfig",
]
