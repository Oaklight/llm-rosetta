"""
LLM-Rosetta - OpenAI Responses Configuration Operations

OpenAI Responses API configuration conversion operations.
Handles bidirectional conversion of generation, stream, reasoning,
cache, and response format configurations.

Note: Responses API uses different field names than Chat API:
- max_tokens → max_output_tokens
- response_format → text (format config)
- reasoning_effort → reasoning.effort
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


class OpenAIResponsesConfigOps(BaseConfigOps):
    """OpenAI Responses API configuration conversion operations.

    All methods are static and stateless.
    """

    # ==================== Generation Config ====================

    @staticmethod
    def ir_generation_config_to_p(ir_config: GenerationConfig, **kwargs: Any) -> dict:
        """IR GenerationConfig → OpenAI Responses generation parameters.

        Field mapping:
        - ``temperature`` → ``temperature`` (direct)
        - ``top_p`` → ``top_p`` (direct)
        - ``max_tokens`` → ``max_output_tokens``
        - ``top_logprobs`` → ``top_logprobs`` (direct)
        - ``truncation`` → ``truncation`` (direct)
        - ``top_k`` → not supported (warning)
        - ``frequency_penalty`` → not supported (warning)
        - ``presence_penalty`` → not supported (warning)
        - ``logit_bias`` → not supported (warning)
        - ``seed`` → not supported (warning)
        - ``n`` → not supported (warning)
        - ``stop_sequences`` → not supported (warning)

        Args:
            ir_config: IR generation config.

        Returns:
            Dict of OpenAI Responses request fields to merge.
        """
        result: dict[str, Any] = {}

        # Direct mapping fields
        _DIRECT_FIELDS = ["temperature", "top_p", "top_logprobs"]
        for field in _DIRECT_FIELDS:
            if field in ir_config:
                result[field] = cast(dict, ir_config)[field]

        # Renamed fields
        if "max_tokens" in ir_config:
            result["max_output_tokens"] = ir_config["max_tokens"]

        # Truncation (Responses API specific)
        if "truncation" in ir_config:
            result["truncation"] = ir_config["truncation"]

        # Unsupported fields
        _UNSUPPORTED = [
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
            "seed",
            "n",
            "stop_sequences",
        ]
        for field in _UNSUPPORTED:
            if field in ir_config:
                warnings.warn(
                    f"OpenAI Responses API does not support {field}, ignored",
                    stacklevel=2,
                )

        return result

    @staticmethod
    def p_generation_config_to_ir(
        provider_config: Any, **kwargs: Any
    ) -> GenerationConfig:
        """OpenAI Responses generation parameters → IR GenerationConfig.

        Extracts generation-related fields from the provider request dict.

        Args:
            provider_config: Dict with OpenAI Responses generation fields.

        Returns:
            IR GenerationConfig.
        """
        result: dict[str, Any] = {}

        if not isinstance(provider_config, dict):
            return cast(GenerationConfig, result)

        # Direct mapping fields
        _DIRECT_FIELDS = ["temperature", "top_p", "top_logprobs"]
        for field in _DIRECT_FIELDS:
            if field in provider_config:
                result[field] = provider_config[field]

        # Renamed fields
        if "max_output_tokens" in provider_config:
            result["max_tokens"] = provider_config["max_output_tokens"]

        # Truncation
        if "truncation" in provider_config:
            result["truncation"] = provider_config["truncation"]

        return cast(GenerationConfig, result)

    # ==================== Response Format ====================

    @staticmethod
    def ir_response_format_to_p(ir_format: ResponseFormatConfig, **kwargs: Any) -> dict:
        """IR ResponseFormatConfig → OpenAI Responses text format parameter.

        Responses API uses ``text`` field instead of ``response_format``.

        Args:
            ir_format: IR response format config.

        Returns:
            Dict with ``text`` field for OpenAI Responses request.
        """
        fmt_type = ir_format.get("type", "text")

        if fmt_type == "text":
            return {"text": {"type": "text"}}
        elif fmt_type == "json_object":
            return {"text": {"type": "json_object"}}
        elif fmt_type == "json_schema":
            text_config: dict[str, Any] = {"type": "json_schema"}
            json_schema = ir_format.get("json_schema")
            if json_schema:
                text_config["json_schema"] = json_schema
            return {"text": text_config}

        return {}

    @staticmethod
    def p_response_format_to_ir(
        provider_format: Any, **kwargs: Any
    ) -> ResponseFormatConfig:
        """OpenAI Responses text format → IR ResponseFormatConfig.

        Args:
            provider_format: OpenAI Responses ``text`` field dict.

        Returns:
            IR ResponseFormatConfig.
        """
        if not isinstance(provider_format, dict):
            return cast(ResponseFormatConfig, {})

        result: dict[str, Any] = {}
        fmt_type = provider_format.get("type")
        if fmt_type:
            result["type"] = fmt_type

        if fmt_type == "json_schema" and "json_schema" in provider_format:
            result["json_schema"] = provider_format["json_schema"]

        return cast(ResponseFormatConfig, result)

    # ==================== Stream Config ====================

    @staticmethod
    def ir_stream_config_to_p(ir_stream: StreamConfig, **kwargs: Any) -> dict:
        """IR StreamConfig → OpenAI Responses stream parameters.

        Mapping:
        - ``enabled`` → ``stream``
        - ``include_usage`` → ``stream_options.include_usage``

        Args:
            ir_stream: IR stream config.

        Returns:
            Dict of OpenAI Responses request fields to merge.
        """
        result: dict[str, Any] = {}

        if "enabled" in ir_stream:
            result["stream"] = ir_stream["enabled"]

        if ir_stream.get("include_usage") and ir_stream.get("enabled", False):
            result["stream_options"] = {"include_usage": True}

        return result

    @staticmethod
    def p_stream_config_to_ir(provider_stream: Any, **kwargs: Any) -> StreamConfig:
        """OpenAI Responses stream parameters → IR StreamConfig.

        Args:
            provider_stream: Dict with ``stream`` and ``stream_options`` fields.

        Returns:
            IR StreamConfig.
        """
        result: dict[str, Any] = {}

        if not isinstance(provider_stream, dict):
            return cast(StreamConfig, result)

        stream = provider_stream.get("stream")
        if stream is not None:
            result["enabled"] = stream

        stream_options = provider_stream.get("stream_options")
        if stream_options and stream_options.get("include_usage"):
            result["include_usage"] = True

        return cast(StreamConfig, result)

    # ==================== Reasoning Config ====================

    @staticmethod
    def ir_reasoning_config_to_p(ir_reasoning: ReasoningConfig, **kwargs: Any) -> dict:
        """IR ReasoningConfig → OpenAI Responses reasoning parameters.

        Responses API uses a ``reasoning`` object with ``type`` and ``effort``.

        Mapping:
        - ``type`` → ``reasoning.type``
        - ``effort`` → ``reasoning.effort``
        - ``budget_tokens`` → not supported (warning)

        Args:
            ir_reasoning: IR reasoning config.

        Returns:
            Dict of OpenAI Responses request fields to merge.
        """
        result: dict[str, Any] = {}
        reasoning_p: dict[str, Any] = {}

        if "type" in ir_reasoning:
            reasoning_p["type"] = ir_reasoning["type"]

        if "effort" in ir_reasoning:
            reasoning_p["effort"] = ir_reasoning["effort"]

        if reasoning_p:
            result["reasoning"] = reasoning_p

        if "budget_tokens" in ir_reasoning:
            warnings.warn(
                "OpenAI Responses API does not support reasoning budget_tokens, ignored",
                stacklevel=2,
            )

        return result

    @staticmethod
    def p_reasoning_config_to_ir(
        provider_reasoning: Any, **kwargs: Any
    ) -> ReasoningConfig:
        """OpenAI Responses reasoning parameters → IR ReasoningConfig.

        Args:
            provider_reasoning: Dict with ``reasoning`` field or reasoning object.

        Returns:
            IR ReasoningConfig.
        """
        result: dict[str, Any] = {}

        if not isinstance(provider_reasoning, dict):
            return cast(ReasoningConfig, result)

        # Handle both top-level reasoning object and nested
        reasoning = provider_reasoning.get("reasoning", provider_reasoning)
        if not isinstance(reasoning, dict):
            return cast(ReasoningConfig, result)

        if "type" in reasoning:
            result["type"] = reasoning["type"]

        if "effort" in reasoning:
            result["effort"] = reasoning["effort"]

        return cast(ReasoningConfig, result)

    # ==================== Cache Config ====================

    @staticmethod
    def ir_cache_config_to_p(ir_cache: CacheConfig, **kwargs: Any) -> dict:
        """IR CacheConfig → OpenAI Responses cache parameters.

        Mapping:
        - ``key`` → ``prompt_cache_key``
        - ``retention`` → ``prompt_cache_retention``

        Args:
            ir_cache: IR cache config.

        Returns:
            Dict of OpenAI Responses request fields to merge.
        """
        result: dict[str, Any] = {}

        if "key" in ir_cache:
            result["prompt_cache_key"] = ir_cache["key"]
        if "retention" in ir_cache:
            result["prompt_cache_retention"] = ir_cache["retention"]

        return result

    @staticmethod
    def p_cache_config_to_ir(provider_cache: Any, **kwargs: Any) -> CacheConfig:
        """OpenAI Responses cache parameters → IR CacheConfig.

        Args:
            provider_cache: Dict with ``prompt_cache_key`` and
                ``prompt_cache_retention`` fields.

        Returns:
            IR CacheConfig.
        """
        result: dict[str, Any] = {}

        if not isinstance(provider_cache, dict):
            return cast(CacheConfig, result)

        if "prompt_cache_key" in provider_cache:
            result["key"] = provider_cache["prompt_cache_key"]
        if "prompt_cache_retention" in provider_cache:
            result["retention"] = provider_cache["prompt_cache_retention"]

        return cast(CacheConfig, result)
