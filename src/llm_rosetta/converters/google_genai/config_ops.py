"""
LLM-Rosetta - Google GenAI Configuration Operations

Google GenAI API configuration conversion operations.
Handles bidirectional conversion of generation, stream, reasoning,
cache, and response format configurations.

Google-specific:
- Generation params go into GenerateContentConfig (temperature, top_p, top_k, etc.)
- max_tokens → max_output_tokens
- Response format uses response_mime_type and response_schema
- Reasoning is usually automatic (model-specific, e.g. Gemini 2.0 Thinking)
- Stream is handled by client method choice (generate_content vs generate_content_stream)
"""

import warnings
from typing import Any, cast

from ...types.ir.configs import (
    CacheConfig,
    GenerationConfig,
    ReasoningConfig,
    ResponseFormatConfig,
    StreamConfig,
)
from ..base import BaseConfigOps


class GoogleGenAIConfigOps(BaseConfigOps):
    """Google GenAI configuration conversion operations.

    All methods are static and stateless.
    """

    # ==================== Generation Config ====================

    @staticmethod
    def ir_generation_config_to_p(ir_config: GenerationConfig, **kwargs: Any) -> dict:
        """IR GenerationConfig → Google GenAI generation parameters.

        Field mapping:
        - ``temperature`` → ``temperature`` (direct)
        - ``top_p`` → ``top_p`` (direct)
        - ``top_k`` → ``top_k`` (direct, Google supports this)
        - ``max_tokens`` → ``max_output_tokens``
        - ``stop_sequences`` → ``stop_sequences`` (direct)
        - ``frequency_penalty`` → ``frequency_penalty`` (direct)
        - ``presence_penalty`` → ``presence_penalty`` (direct)
        - ``seed`` → ``seed`` (direct)
        - ``candidate_count`` → ``candidate_count`` (Google-specific)
        - ``n`` → ``candidate_count`` (alias)

        Args:
            ir_config: IR generation config.

        Returns:
            Dict of Google config fields to merge.
        """
        result: dict[str, Any] = {}

        # Direct mapping fields
        _DIRECT_FIELDS = [
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "candidate_count",
        ]
        for field in _DIRECT_FIELDS:
            if field in ir_config:
                result[field] = cast(dict, ir_config)[field]

        # Renamed fields
        if "max_tokens" in ir_config:
            result["max_output_tokens"] = ir_config["max_tokens"]

        # n → candidate_count (if candidate_count not already set)
        if "n" in ir_config and "candidate_count" not in result:
            result["candidate_count"] = ir_config["n"]

        # Unsupported fields
        if "logit_bias" in ir_config:
            warnings.warn(
                "Google GenAI does not support logit_bias, ignored",
                stacklevel=2,
            )
        if "logprobs" in ir_config:
            warnings.warn(
                "Google GenAI does not support logprobs, ignored",
                stacklevel=2,
            )

        return result

    @staticmethod
    def p_generation_config_to_ir(
        provider_config: Any, **kwargs: Any
    ) -> GenerationConfig:
        """Google GenAI generation parameters → IR GenerationConfig.

        Extracts generation-related fields from the provider config dict.

        Args:
            provider_config: Dict with Google generation fields.

        Returns:
            IR GenerationConfig.
        """
        result: dict[str, Any] = {}

        if not isinstance(provider_config, dict):
            return cast(GenerationConfig, result)

        # Direct mapping fields (same name in both snake_case and camelCase)
        _DIRECT_FIELDS = [
            "temperature",
            "top_p",
            "top_k",
            "seed",
        ]
        for field in _DIRECT_FIELDS:
            if field in provider_config:
                result[field] = provider_config[field]

        # Fields with camelCase variants
        _CAMEL_FIELDS = [
            ("stop_sequences", "stopSequences", "stop_sequences"),
            ("frequency_penalty", "frequencyPenalty", "frequency_penalty"),
            ("presence_penalty", "presencePenalty", "presence_penalty"),
        ]
        for snake, camel, ir_name in _CAMEL_FIELDS:
            val = provider_config.get(snake, provider_config.get(camel))
            if val is not None:
                result[ir_name] = val

        # Renamed fields
        max_tokens = provider_config.get(
            "max_output_tokens", provider_config.get("maxOutputTokens")
        )
        if max_tokens is not None:
            result["max_tokens"] = max_tokens

        # candidate_count → n (if present)
        candidate_count = provider_config.get(
            "candidate_count", provider_config.get("candidateCount")
        )
        if candidate_count is not None:
            result["candidate_count"] = candidate_count

        return cast(GenerationConfig, result)

    # ==================== Response Format ====================

    @staticmethod
    def ir_response_format_to_p(ir_format: ResponseFormatConfig, **kwargs: Any) -> dict:
        """IR ResponseFormatConfig → Google GenAI response format parameters.

        Google uses ``response_mime_type`` and ``response_schema`` instead of
        a nested response_format object.

        Args:
            ir_format: IR response format config.

        Returns:
            Dict of Google config fields to merge.
        """
        result: dict[str, Any] = {}
        fmt_type = ir_format.get("type", "text")

        if fmt_type == "json_object":
            result["response_mime_type"] = "application/json"
        elif fmt_type == "json_schema":
            result["response_mime_type"] = "application/json"
            json_schema = ir_format.get("json_schema")
            if json_schema:
                result["response_schema"] = json_schema

        return result

    @staticmethod
    def p_response_format_to_ir(
        provider_format: Any, **kwargs: Any
    ) -> ResponseFormatConfig:
        """Google GenAI response format → IR ResponseFormatConfig.

        Args:
            provider_format: Dict with ``response_mime_type`` and
                optionally ``response_schema``.

        Returns:
            IR ResponseFormatConfig.
        """
        if not isinstance(provider_format, dict):
            return cast(ResponseFormatConfig, {})

        result: dict[str, Any] = {}
        mime_type = provider_format.get("response_mime_type") or provider_format.get(
            "responseMimeType"
        )

        if mime_type == "application/json":
            schema = provider_format.get("response_schema") or provider_format.get(
                "responseSchema"
            )
            if schema:
                result["type"] = "json_schema"
                result["json_schema"] = schema
            else:
                result["type"] = "json_object"

        return cast(ResponseFormatConfig, result)

    # ==================== Stream Config ====================

    @staticmethod
    def ir_stream_config_to_p(ir_stream: StreamConfig, **kwargs: Any) -> dict:
        """IR StreamConfig → Google GenAI stream parameters.

        Google streaming is controlled by the client method choice
        (generate_content vs generate_content_stream), not by a config field.
        We pass through the enabled flag for the caller to use.

        Args:
            ir_stream: IR stream config.

        Returns:
            Dict with stream flag (for caller to interpret).
        """
        result: dict[str, Any] = {}

        if "enabled" in ir_stream:
            result["stream"] = ir_stream["enabled"]

        return result

    @staticmethod
    def p_stream_config_to_ir(provider_stream: Any, **kwargs: Any) -> StreamConfig:
        """Google GenAI stream parameters → IR StreamConfig.

        Args:
            provider_stream: Dict with stream flag.

        Returns:
            IR StreamConfig.
        """
        result: dict[str, Any] = {}

        if not isinstance(provider_stream, dict):
            return cast(StreamConfig, result)

        stream = provider_stream.get("stream")
        if stream is not None:
            result["enabled"] = stream

        return cast(StreamConfig, result)

    # ==================== Reasoning Config ====================

    @staticmethod
    def ir_reasoning_config_to_p(ir_reasoning: ReasoningConfig, **kwargs: Any) -> dict:
        """IR ReasoningConfig → Google GenAI reasoning parameters.

        Google reasoning is usually automatic or model-specific
        (e.g. Gemini 2.0 Thinking). There's no direct config field
        for reasoning effort in the standard API.

        Args:
            ir_reasoning: IR reasoning config.

        Returns:
            Dict of Google config fields to merge (may be empty).
        """
        result: dict[str, Any] = {}

        # Google doesn't have a direct reasoning_effort equivalent
        if "effort" in ir_reasoning:
            warnings.warn(
                "Google GenAI does not support reasoning effort config, ignored",
                stacklevel=2,
            )

        if "budget_tokens" in ir_reasoning:
            # Some Google models may support thinking budget
            result["thinking_config"] = {
                "thinking_budget": ir_reasoning["budget_tokens"]
            }

        return result

    @staticmethod
    def p_reasoning_config_to_ir(
        provider_reasoning: Any, **kwargs: Any
    ) -> ReasoningConfig:
        """Google GenAI reasoning parameters → IR ReasoningConfig.

        Args:
            provider_reasoning: Dict with Google reasoning fields.

        Returns:
            IR ReasoningConfig.
        """
        result: dict[str, Any] = {}

        if not isinstance(provider_reasoning, dict):
            return cast(ReasoningConfig, result)

        thinking_config = provider_reasoning.get(
            "thinking_config"
        ) or provider_reasoning.get("thinkingConfig")
        if thinking_config:
            budget = thinking_config.get("thinking_budget") or thinking_config.get(
                "thinkingBudget"
            )
            if budget is not None:
                result["budget_tokens"] = budget

        return cast(ReasoningConfig, result)

    # ==================== Cache Config ====================

    @staticmethod
    def ir_cache_config_to_p(ir_cache: CacheConfig, **kwargs: Any) -> dict:
        """IR CacheConfig → Google GenAI cache parameters.

        Google uses ``cached_content`` for caching, which is a separate
        resource that must be created beforehand.

        Args:
            ir_cache: IR cache config.

        Returns:
            Dict of Google config fields to merge.
        """
        result: dict[str, Any] = {}

        if "key" in ir_cache:
            # Google uses cached_content resource name
            result["cached_content"] = ir_cache["key"]

        if "retention" in ir_cache:
            warnings.warn(
                "Google GenAI cache retention is managed at the CachedContent "
                "resource level, not per-request. Ignored.",
                stacklevel=2,
            )

        return result

    @staticmethod
    def p_cache_config_to_ir(provider_cache: Any, **kwargs: Any) -> CacheConfig:
        """Google GenAI cache parameters → IR CacheConfig.

        Args:
            provider_cache: Dict with ``cached_content`` field.

        Returns:
            IR CacheConfig.
        """
        result: dict[str, Any] = {}

        if not isinstance(provider_cache, dict):
            return cast(CacheConfig, result)

        if "cached_content" in provider_cache:
            result["key"] = provider_cache["cached_content"]

        return cast(CacheConfig, result)
