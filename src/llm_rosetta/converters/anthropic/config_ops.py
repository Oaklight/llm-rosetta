"""
LLM-Rosetta - Anthropic Configuration Operations

Anthropic Messages API configuration conversion operations.
Handles bidirectional conversion of generation, stream, reasoning,
cache, and response format configurations.

Key Anthropic differences:
- ``max_tokens`` is required (default 4096)
- ``temperature`` range is 0.0-1.0 (needs clamping)
- ``top_k`` is supported
- Response format is not supported (use tools or system instructions)
- Cache control is block-level, not request-level
- Reasoning uses ``thinking`` with ``type`` and ``budget_tokens``
"""

import warnings
from typing import Any, Dict

from ...types.ir.configs import (
    CacheConfig,
    GenerationConfig,
    ReasoningConfig,
    ResponseFormatConfig,
    StreamConfig,
)
from ..base import BaseConfigOps


class AnthropicConfigOps(BaseConfigOps):
    """Anthropic Messages API configuration conversion operations.

    All methods are static and stateless.
    """

    # ==================== Generation Config ====================

    @staticmethod
    def ir_generation_config_to_p(ir_config: GenerationConfig, **kwargs: Any) -> dict:
        """IR GenerationConfig → Anthropic generation parameters.

        Field mapping:
        - ``temperature`` → ``temperature`` (clamped to 0.0-1.0)
        - ``top_p`` → ``top_p`` (direct)
        - ``top_k`` → ``top_k`` (direct, supported by Anthropic)
        - ``max_tokens`` → ``max_tokens`` (required, default 4096)
        - ``stop_sequences`` → ``stop_sequences`` (direct)
        - ``frequency_penalty`` → not supported (warning)
        - ``presence_penalty`` → not supported (warning)
        - ``logit_bias`` → not supported (warning)
        - ``seed`` → not supported (warning)
        - ``logprobs`` → not supported (warning)
        - ``n`` → not supported (warning)

        Args:
            ir_config: IR generation config.

        Returns:
            Dict of Anthropic request fields to merge.
        """
        result: Dict[str, Any] = {}

        # max_tokens is required for Anthropic
        result["max_tokens"] = ir_config.get("max_tokens", 4096)

        # temperature (clamped to 0.0-1.0)
        if "temperature" in ir_config:
            result["temperature"] = min(float(ir_config["temperature"]), 1.0)

        # Direct mapping fields
        if "top_p" in ir_config:
            result["top_p"] = ir_config["top_p"]

        if "top_k" in ir_config:
            result["top_k"] = ir_config["top_k"]

        # stop_sequences (direct mapping, same name)
        if "stop_sequences" in ir_config:
            result["stop_sequences"] = list(ir_config["stop_sequences"])

        # Unsupported fields
        _UNSUPPORTED = [
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
            "seed",
            "logprobs",
            "top_logprobs",
            "n",
        ]
        unsupported_found = [f for f in _UNSUPPORTED if f in ir_config]
        if unsupported_found:
            for field in unsupported_found:
                warnings.warn(
                    f"Anthropic does not support {field}, ignored",
                    stacklevel=2,
                )

        return result

    @staticmethod
    def p_generation_config_to_ir(
        provider_config: Any, **kwargs: Any
    ) -> GenerationConfig:
        """Anthropic generation parameters → IR GenerationConfig.

        Extracts generation-related fields from the provider request dict.

        Args:
            provider_config: Dict with Anthropic generation fields.

        Returns:
            IR GenerationConfig.
        """
        result: Dict[str, Any] = {}

        if not isinstance(provider_config, dict):
            return result

        if "max_tokens" in provider_config:
            result["max_tokens"] = provider_config["max_tokens"]

        if "temperature" in provider_config:
            result["temperature"] = provider_config["temperature"]

        if "top_p" in provider_config:
            result["top_p"] = provider_config["top_p"]

        if "top_k" in provider_config:
            result["top_k"] = provider_config["top_k"]

        if "stop_sequences" in provider_config:
            result["stop_sequences"] = provider_config["stop_sequences"]

        return result

    # ==================== Response Format ====================

    @staticmethod
    def ir_response_format_to_p(ir_format: ResponseFormatConfig, **kwargs: Any) -> dict:
        """IR ResponseFormatConfig → Anthropic response format.

        Anthropic does not support response_format parameter.
        Returns empty dict and emits a warning.

        Args:
            ir_format: IR response format config.

        Returns:
            Empty dict (not supported).
        """
        warnings.warn(
            "Anthropic does not support response_format, "
            "use system instructions or tools instead",
            stacklevel=2,
        )
        return {}

    @staticmethod
    def p_response_format_to_ir(
        provider_format: Any, **kwargs: Any
    ) -> ResponseFormatConfig:
        """Anthropic response format → IR ResponseFormatConfig.

        Anthropic does not have response_format, returns empty config.

        Args:
            provider_format: Not applicable for Anthropic.

        Returns:
            Empty IR ResponseFormatConfig.
        """
        return {}

    # ==================== Stream Config ====================

    @staticmethod
    def ir_stream_config_to_p(ir_stream: StreamConfig, **kwargs: Any) -> dict:
        """IR StreamConfig → Anthropic stream parameter.

        Mapping:
        - ``enabled`` → ``stream``

        Anthropic does not have ``stream_options`` like OpenAI.

        Args:
            ir_stream: IR stream config.

        Returns:
            Dict of Anthropic request fields to merge.
        """
        result: Dict[str, Any] = {}

        if "enabled" in ir_stream:
            result["stream"] = ir_stream["enabled"]

        if ir_stream.get("include_usage"):
            warnings.warn(
                "Anthropic always includes usage in responses, "
                "include_usage option ignored",
                stacklevel=2,
            )

        return result

    @staticmethod
    def p_stream_config_to_ir(provider_stream: Any, **kwargs: Any) -> StreamConfig:
        """Anthropic stream parameter → IR StreamConfig.

        Args:
            provider_stream: Dict with ``stream`` field.

        Returns:
            IR StreamConfig.
        """
        result: Dict[str, Any] = {}

        if not isinstance(provider_stream, dict):
            return result

        stream = provider_stream.get("stream")
        if stream is not None:
            result["enabled"] = stream

        return result

    # ==================== Reasoning Config ====================

    @staticmethod
    def ir_reasoning_config_to_p(ir_reasoning: ReasoningConfig, **kwargs: Any) -> dict:
        """IR ReasoningConfig → Anthropic thinking parameter.

        Mapping:
        - ``type:"enabled"`` → ``thinking.type = "enabled"``
        - ``type:"disabled"`` → ``thinking.type = "disabled"``
        - ``budget_tokens`` → ``thinking.budget_tokens``
        - ``effort`` → not directly supported (warning)

        Args:
            ir_reasoning: IR reasoning config.

        Returns:
            Dict of Anthropic request fields to merge.
        """
        result: Dict[str, Any] = {}

        reasoning_type = ir_reasoning.get("type")
        if reasoning_type == "enabled":
            thinking: Dict[str, Any] = {"type": "enabled"}
            if "budget_tokens" in ir_reasoning:
                thinking["budget_tokens"] = ir_reasoning["budget_tokens"]
            result["thinking"] = thinking
        elif reasoning_type == "disabled":
            result["thinking"] = {"type": "disabled"}

        if "effort" in ir_reasoning:
            warnings.warn(
                "Anthropic does not support reasoning effort level, "
                "use type and budget_tokens instead",
                stacklevel=2,
            )

        return result

    @staticmethod
    def p_reasoning_config_to_ir(
        provider_reasoning: Any, **kwargs: Any
    ) -> ReasoningConfig:
        """Anthropic thinking parameter → IR ReasoningConfig.

        Args:
            provider_reasoning: Dict with ``thinking`` field.

        Returns:
            IR ReasoningConfig.
        """
        result: Dict[str, Any] = {}

        if not isinstance(provider_reasoning, dict):
            return result

        thinking = provider_reasoning.get("thinking")
        if not isinstance(thinking, dict):
            return result

        thinking_type = thinking.get("type")
        if thinking_type:
            result["type"] = thinking_type

        budget_tokens = thinking.get("budget_tokens")
        if budget_tokens is not None:
            result["budget_tokens"] = budget_tokens

        return result

    # ==================== Cache Config ====================

    @staticmethod
    def ir_cache_config_to_p(ir_cache: CacheConfig, **kwargs: Any) -> dict:
        """IR CacheConfig → Anthropic cache parameters.

        Anthropic cache control is block-level (``cache_control`` on content blocks),
        not request-level. Returns empty dict and emits a warning.

        Args:
            ir_cache: IR cache config.

        Returns:
            Empty dict (block-level cache not handled here).
        """
        warnings.warn(
            "Anthropic cache control is block-level, "
            "request-level cache config ignored",
            stacklevel=2,
        )
        return {}

    @staticmethod
    def p_cache_config_to_ir(provider_cache: Any, **kwargs: Any) -> CacheConfig:
        """Anthropic cache parameters → IR CacheConfig.

        Anthropic does not have request-level cache config.

        Args:
            provider_cache: Not applicable for Anthropic.

        Returns:
            Empty IR CacheConfig.
        """
        return {}
