"""
LLM-Rosetta - Base Configuration Operations
配置转换操作的抽象基类
Abstract base class for configuration conversion operations

处理所有配置相关的转换：
- 生成配置：温度、top_p、max_tokens等生成控制参数
- 响应格式：JSON schema、MIME类型等输出格式控制
- 流式配置：流式输出、使用统计等流式控制
- 推理配置：推理努力、预算token等推理控制
- 缓存配置：缓存键、保留策略等缓存控制
Handles all configuration-related conversions:
- Generation config: temperature, top_p, max_tokens and other generation control parameters
- Response format: JSON schema, MIME types and other output format controls
- Stream config: streaming output, usage statistics and other streaming controls
- Reasoning config: reasoning effort, budget tokens and other reasoning controls
- Cache config: cache keys, retention policies and other cache controls
"""

from abc import ABC, abstractmethod
from typing import Any

from ...types.ir.configs import (
    CacheConfig,
    GenerationConfig,
    ReasoningConfig,
    ResponseFormatConfig,
    StreamConfig,
)


class BaseConfigOps(ABC):
    """配置转换操作的抽象基类
    Abstract base class for configuration conversion operations

    统一处理各种控制参数的转换，这些参数影响模型的行为和输出格式。
    Uniformly handles conversion of various control parameters that affect model behavior and output format.
    """

    # ==================== 生成配置转换 Generation configuration conversion ====================

    @staticmethod
    @abstractmethod
    def ir_generation_config_to_p(ir_config: GenerationConfig, **kwargs: Any) -> Any:
        """IR GenerationConfig → Provider Generation Config
        将IR生成配置转换为Provider生成配置

        处理模型生成行为的控制参数：
        - 采样参数：temperature, top_p, top_k
        - 长度控制：max_tokens, stop_sequences
        - 惩罚参数：frequency_penalty, presence_penalty
        - 其他参数：seed, logprobs等

        Handles control parameters for model generation behavior:
        - Sampling parameters: temperature, top_p, top_k
        - Length control: max_tokens, stop_sequences
        - Penalty parameters: frequency_penalty, presence_penalty
        - Other parameters: seed, logprobs, etc.

        Args:
            ir_config: IR格式的生成配置
            **kwargs: 额外参数

        Returns:
            Provider格式的生成配置
        """
        pass

    @staticmethod
    @abstractmethod
    def p_generation_config_to_ir(
        provider_config: Any, **kwargs: Any
    ) -> GenerationConfig:
        """Provider Generation Config → IR GenerationConfig
        将Provider生成配置转换为IR生成配置

        Args:
            provider_config: Provider格式的生成配置
            **kwargs: 额外参数

        Returns:
            IR格式的生成配置
        """
        pass

    # ==================== 响应格式配置转换 Response format configuration conversion ====================

    @staticmethod
    @abstractmethod
    def ir_response_format_to_p(ir_format: ResponseFormatConfig, **kwargs: Any) -> Any:
        """IR ResponseFormatConfig → Provider Response Format
        将IR响应格式配置转换为Provider响应格式

        处理输出格式的控制：
        - 格式类型：text, json_object, json_schema
        - Schema定义：JSON schema规范
        - MIME类型：response_mime_type（Google）

        Handles output format control:
        - Format types: text, json_object, json_schema
        - Schema definition: JSON schema specification
        - MIME types: response_mime_type (Google)

        Args:
            ir_format: IR格式的响应格式配置
            **kwargs: 额外参数

        Returns:
            Provider格式的响应格式
        """
        pass

    @staticmethod
    @abstractmethod
    def p_response_format_to_ir(
        provider_format: Any, **kwargs: Any
    ) -> ResponseFormatConfig:
        """Provider Response Format → IR ResponseFormatConfig
        将Provider响应格式转换为IR响应格式配置

        Args:
            provider_format: Provider格式的响应格式
            **kwargs: 额外参数

        Returns:
            IR格式的响应格式配置
        """
        pass

    # ==================== 流式配置转换 Stream configuration conversion ====================

    @staticmethod
    @abstractmethod
    def ir_stream_config_to_p(ir_stream: StreamConfig, **kwargs: Any) -> Any:
        """IR StreamConfig → Provider Stream Config
        将IR流式配置转换为Provider流式配置

        处理流式输出的控制：
        - 启用状态：enabled
        - 使用统计：include_usage（OpenAI）

        Handles streaming output control:
        - Enable status: enabled
        - Usage statistics: include_usage (OpenAI)

        Args:
            ir_stream: IR格式的流式配置
            **kwargs: 额外参数

        Returns:
            Provider格式的流式配置
        """
        pass

    @staticmethod
    @abstractmethod
    def p_stream_config_to_ir(provider_stream: Any, **kwargs: Any) -> StreamConfig:
        """Provider Stream Config → IR StreamConfig
        将Provider流式配置转换为IR流式配置

        Args:
            provider_stream: Provider格式的流式配置
            **kwargs: 额外参数

        Returns:
            IR格式的流式配置
        """
        pass

    # ==================== 推理配置转换 Reasoning configuration conversion ====================

    @staticmethod
    @abstractmethod
    def ir_reasoning_config_to_p(ir_reasoning: ReasoningConfig, **kwargs: Any) -> Any:
        """IR ReasoningConfig → Provider Reasoning Config
        将IR推理配置转换为Provider推理配置

        处理推理过程的控制：
        - 推理模式：mode (auto/enabled/disabled) - Anthropic, OpenAI Responses, Google
        - 推理努力：effort (minimal/low/medium/high/max)
        - 预算token：budget_tokens - Anthropic/Google

        Handles reasoning process control:
        - Reasoning mode: mode (auto/enabled/disabled) - Anthropic, OpenAI Responses, Google
        - Reasoning effort: effort (minimal/low/medium/high/max)
        - Budget tokens: budget_tokens - Anthropic/Google

        Args:
            ir_reasoning: IR格式的推理配置
            **kwargs: 额外参数

        Returns:
            Provider格式的推理配置
        """
        pass

    @staticmethod
    @abstractmethod
    def p_reasoning_config_to_ir(
        provider_reasoning: Any, **kwargs: Any
    ) -> ReasoningConfig:
        """Provider Reasoning Config → IR ReasoningConfig
        将Provider推理配置转换为IR推理配置

        Args:
            provider_reasoning: Provider请求字典（完整的provider_request）
            **kwargs: 额外参数

        Returns:
            IR格式的推理配置
        """
        pass

    # ==================== 缓存配置转换 Cache configuration conversion ====================

    @staticmethod
    @abstractmethod
    def ir_cache_config_to_p(ir_cache: CacheConfig, **kwargs: Any) -> Any:
        """IR CacheConfig → Provider Cache Config
        将IR缓存配置转换为Provider缓存配置

        处理提示缓存的控制（主要是OpenAI）：
        - 缓存键：key (prompt_cache_key)
        - 保留策略：retention (in-memory/24h)

        Handles prompt cache control (mainly OpenAI):
        - Cache key: key (prompt_cache_key)
        - Retention policy: retention (in-memory/24h)

        Args:
            ir_cache: IR格式的缓存配置
            **kwargs: 额外参数

        Returns:
            Provider格式的缓存配置
        """
        pass

    @staticmethod
    @abstractmethod
    def p_cache_config_to_ir(provider_cache: Any, **kwargs: Any) -> CacheConfig:
        """Provider Cache Config → IR CacheConfig
        将Provider缓存配置转换为IR缓存配置

        Args:
            provider_cache: Provider格式的缓存配置
            **kwargs: 额外参数

        Returns:
            IR格式的缓存配置
        """
        pass
