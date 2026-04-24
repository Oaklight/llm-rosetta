"""
LLM-Rosetta - OpenAI Chat Configuration Operations

OpenAI Chat Completions API configuration conversion operations.
Handles bidirectional conversion of generation, stream, reasoning,
cache, and response format configurations.
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


class OpenAIChatConfigOps(BaseConfigOps):
    """OpenAI Chat Completions configuration conversion operations.

    All methods are static and stateless.
    """

    # ==================== Generation Config ====================

    @staticmethod
    def ir_generation_config_to_p(ir_config: GenerationConfig, **kwargs: Any) -> dict:
        """IR GenerationConfig → OpenAI Chat generation parameters.

        Field mapping:
        - ``temperature`` → ``temperature`` (direct)
        - ``top_p`` → ``top_p`` (direct)
        - ``top_k`` → not supported (warning)
        - ``max_tokens`` → ``max_completion_tokens``
        - ``stop_sequences`` → ``stop``
        - ``frequency_penalty`` → ``frequency_penalty`` (direct)
        - ``presence_penalty`` → ``presence_penalty`` (direct)
        - ``logit_bias`` → ``logit_bias`` (direct)
        - ``seed`` → ``seed`` (direct)
        - ``logprobs`` → ``logprobs`` (direct)
        - ``top_logprobs`` → ``top_logprobs`` (direct)
        - ``n`` → ``n`` (direct)

        Args:
            ir_config: IR generation config.

        Returns:
            Dict of OpenAI request fields to merge.
        """
        result: dict[str, Any] = {}

        # Direct mapping fields
        _DIRECT_FIELDS = [
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
            "seed",
            "logprobs",
            "top_logprobs",
            "n",
        ]
        for field in _DIRECT_FIELDS:
            if field in ir_config:
                result[field] = cast(dict, ir_config)[field]

        # Renamed fields
        if "max_tokens" in ir_config:
            result["max_completion_tokens"] = ir_config["max_tokens"]

        # stop_sequences → stop
        if "stop_sequences" in ir_config:
            stop = list(ir_config["stop_sequences"])
            if len(stop) == 1:
                result["stop"] = stop[0]
            elif len(stop) > 1:
                result["stop"] = stop

        # Unsupported fields
        if "top_k" in ir_config:
            warnings.warn(
                "OpenAI Chat does not support top_k, ignored",
                stacklevel=2,
            )

        return result

    @staticmethod
    def p_generation_config_to_ir(
        provider_config: Any, **kwargs: Any
    ) -> GenerationConfig:
        """OpenAI Chat generation parameters → IR GenerationConfig.

        Extracts generation-related fields from the provider request dict.

        Args:
            provider_config: Dict with OpenAI generation fields.

        Returns:
            IR GenerationConfig.
        """
        result: dict[str, Any] = {}

        if not isinstance(provider_config, dict):
            return cast(GenerationConfig, result)

        # Direct mapping fields
        _DIRECT_FIELDS = [
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
            "seed",
            "logprobs",
            "top_logprobs",
            "n",
        ]
        for field in _DIRECT_FIELDS:
            if field in provider_config:
                result[field] = provider_config[field]

        # Renamed fields
        if "max_completion_tokens" in provider_config:
            result["max_tokens"] = provider_config["max_completion_tokens"]
        elif "max_tokens" in provider_config:
            result["max_tokens"] = provider_config["max_tokens"]

        # stop → stop_sequences
        if "stop" in provider_config:
            stop = provider_config["stop"]
            if isinstance(stop, str):
                result["stop_sequences"] = [stop]
            elif isinstance(stop, list):
                result["stop_sequences"] = stop

        return cast(GenerationConfig, result)

    # ==================== Response Format ====================

    @staticmethod
    def ir_response_format_to_p(ir_format: ResponseFormatConfig, **kwargs: Any) -> dict:
        """IR ResponseFormatConfig → OpenAI Chat response_format parameter.

        Args:
            ir_format: IR response format config.

        Returns:
            OpenAI response_format dict.
        """
        fmt_type = ir_format.get("type", "text")

        if fmt_type == "text":
            return {"response_format": {"type": "text"}}
        elif fmt_type == "json_object":
            return {"response_format": {"type": "json_object"}}
        elif fmt_type == "json_schema":
            rf: dict[str, Any] = {"type": "json_schema"}
            json_schema = ir_format.get("json_schema")
            if json_schema:
                rf["json_schema"] = json_schema
            return {"response_format": rf}

        return {}

    @staticmethod
    def p_response_format_to_ir(
        provider_format: Any, **kwargs: Any
    ) -> ResponseFormatConfig:
        """OpenAI Chat response_format → IR ResponseFormatConfig.

        Args:
            provider_format: OpenAI response_format dict.

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
        """IR StreamConfig → OpenAI Chat stream parameters.

        Mapping:
        - ``enabled`` → ``stream``
        - ``include_usage`` → ``stream_options.include_usage``

        Args:
            ir_stream: IR stream config.

        Returns:
            Dict of OpenAI request fields to merge.
        """
        result: dict[str, Any] = {}

        if "enabled" in ir_stream:
            result["stream"] = ir_stream["enabled"]

        if ir_stream.get("include_usage") and ir_stream.get("enabled", False):
            result["stream_options"] = {"include_usage": True}

        return result

    @staticmethod
    def p_stream_config_to_ir(provider_stream: Any, **kwargs: Any) -> StreamConfig:
        """OpenAI Chat stream parameters → IR StreamConfig.

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
        """IR ReasoningConfig → OpenAI Chat reasoning parameters.

        Mapping:
        - ``effort`` → ``reasoning_effort``
          (``"minimal"`` → ``"low"``, ``"max"`` → ``"high"`` with warning)
        - ``mode`` → ``thinking.type`` (DeepSeek/Volcengine extension)
        - ``budget_tokens`` → ``thinking.budget_tokens``
        - ``mode: "disabled"`` → skip ``reasoning_effort``, emit
          ``thinking.type = "disabled"``

        Args:
            ir_reasoning: IR reasoning config.

        Returns:
            Dict of OpenAI request fields to merge.
        """
        result: dict[str, Any] = {}
        mode = ir_reasoning.get("mode")

        # Build thinking object for mode / budget_tokens
        thinking: dict[str, Any] = {}
        if mode:
            thinking["type"] = mode
        if "budget_tokens" in ir_reasoning:
            thinking["budget_tokens"] = ir_reasoning["budget_tokens"]
        if thinking:
            result["thinking"] = thinking

        # mode: "disabled" → skip reasoning_effort
        if mode == "disabled":
            return result

        if "effort" in ir_reasoning:
            effort = ir_reasoning["effort"]
            if effort == "minimal":
                warnings.warn(
                    "OpenAI Chat does not support 'minimal' effort, "
                    "downgrading to 'low'",
                    stacklevel=2,
                )
                effort = "low"
            elif effort == "max":
                warnings.warn(
                    "OpenAI Chat does not support 'max' effort, downgrading to 'high'",
                    stacklevel=2,
                )
                effort = "high"
            result["reasoning_effort"] = effort

        return result

    @staticmethod
    def p_reasoning_config_to_ir(
        provider_reasoning: Any, **kwargs: Any
    ) -> ReasoningConfig:
        """OpenAI Chat reasoning parameters → IR ReasoningConfig.

        Extracts ``reasoning_effort`` and ``thinking`` config from the
        provider request.

        Args:
            provider_reasoning: Provider request dict.

        Returns:
            IR ReasoningConfig.
        """
        result: dict[str, Any] = {}

        if not isinstance(provider_reasoning, dict):
            return cast(ReasoningConfig, result)

        effort = provider_reasoning.get("reasoning_effort")
        if effort:
            result["effort"] = effort

        # DeepSeek/Volcengine thinking extension
        thinking = provider_reasoning.get("thinking")
        if isinstance(thinking, dict):
            thinking_type = thinking.get("type")
            if thinking_type:
                result["mode"] = thinking_type
            budget = thinking.get("budget_tokens")
            if budget is not None:
                result["budget_tokens"] = budget

        return cast(ReasoningConfig, result)

    # ==================== Cache Config ====================

    @staticmethod
    def ir_cache_config_to_p(ir_cache: CacheConfig, **kwargs: Any) -> dict:
        """IR CacheConfig → OpenAI Chat cache parameters.

        Mapping:
        - ``key`` → ``prompt_cache_key``
        - ``retention`` → ``prompt_cache_retention``

        Args:
            ir_cache: IR cache config.

        Returns:
            Dict of OpenAI request fields to merge.
        """
        result: dict[str, Any] = {}

        if "key" in ir_cache:
            result["prompt_cache_key"] = ir_cache["key"]
        if "retention" in ir_cache:
            result["prompt_cache_retention"] = ir_cache["retention"]

        return result

    @staticmethod
    def p_cache_config_to_ir(provider_cache: Any, **kwargs: Any) -> CacheConfig:
        """OpenAI Chat cache parameters → IR CacheConfig.

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
