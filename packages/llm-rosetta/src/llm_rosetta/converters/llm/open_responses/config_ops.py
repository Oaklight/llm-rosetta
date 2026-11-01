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

from typing import Any, cast

from ....types.ir.configs import (
    CacheConfig,
    GenerationConfig,
    ReasoningConfig,
    ResponseFormatConfig,
    StreamConfig,
)
from llm_rosetta.conversion_warnings import report_loss
from ...base import BaseConfigOps
from ...base.helpers.reasoning import DEFAULT_REASONING_CAPS, apply_reasoning_config
from ...base.helpers.retention import retention_to_ir, retention_to_wire

#: The response formats the Responses ``text`` parameter defines.  The spec
#: names ``text`` and ``json_schema``; ``json_object`` is accepted by the live
#: API and is a value the IR carries, so it is written too.
_TEXT_FORMAT_TYPES = frozenset({"text", "json_object", "json_schema"})

#: The keys a ``json_schema`` format carries beside ``type``, flat on the wire
#: and nested under ``json_schema`` in the IR.
_SCHEMA_KEYS = ("name", "description", "schema", "strict")


class OpenResponsesConfigOps(BaseConfigOps):
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
        - ``frequency_penalty`` → ``frequency_penalty`` (direct)
        - ``presence_penalty`` → ``presence_penalty`` (direct)
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
        _DIRECT_FIELDS = [
            "temperature",
            "top_p",
            "top_logprobs",
            "frequency_penalty",
            "presence_penalty",
        ]
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
            "logit_bias",
            "seed",
            "n",
            "stop_sequences",
            "logprobs",
        ]
        for field in _UNSUPPORTED:
            if field in ir_config:
                report_loss(
                    f"OpenAI Responses API does not support {field}; ignored",
                    subject=field,
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
        _DIRECT_FIELDS = [
            "temperature",
            "top_p",
            "top_logprobs",
            "frequency_penalty",
            "presence_penalty",
        ]
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
        """IR ResponseFormatConfig → the Responses ``text`` parameter.

        The spec keeps the format one level down — ``text`` is
        ``{format: {...}, verbosity}`` — and spells a JSON schema flat, with
        ``name`` / ``description`` / ``schema`` / ``strict`` beside ``type``
        instead of nested under ``json_schema`` as the Chat dialect does.
        Writing the Chat shape here produced a body the spec does not define
        and the reader could not read back (#1294).

        Args:
            ir_format: IR response format config.

        Returns:
            Dict with the ``text`` field, empty for a format the API does not
            define.
        """
        fmt_type = ir_format.get("type", "text")
        if fmt_type not in _TEXT_FORMAT_TYPES:
            return {}

        fmt: dict[str, Any] = {"type": fmt_type}
        if fmt_type == "json_schema":
            schema = ir_format.get("json_schema") or {}
            fmt.update({key: schema[key] for key in _SCHEMA_KEYS if key in schema})
        return {"text": {"format": fmt}}

    @staticmethod
    def p_response_format_to_ir(
        provider_format: Any, **kwargs: Any
    ) -> ResponseFormatConfig:
        """The Responses ``text`` parameter → IR ResponseFormatConfig.

        Reads the spec's nested ``text: {format: {...}}`` and, as a fallback,
        the Chat-shaped ``text: {type: ...}`` this converter wrote before
        #1294 — a body built for a Chat-compatible lane may still carry it.
        A JSON schema is unflattened into the IR's ``json_schema`` object.

        Args:
            provider_format: OpenAI Responses ``text`` field dict.

        Returns:
            IR ResponseFormatConfig.
        """
        if not isinstance(provider_format, dict):
            return cast(ResponseFormatConfig, {})

        nested_format = provider_format.get("format")
        fmt = nested_format if isinstance(nested_format, dict) else provider_format

        result: dict[str, Any] = {}
        fmt_type = fmt.get("type")
        if fmt_type:
            result["type"] = fmt_type

        json_schema = fmt.get("json_schema")
        if isinstance(json_schema, dict):
            result["json_schema"] = json_schema
        elif fmt_type == "json_schema":
            schema = {key: fmt[key] for key in _SCHEMA_KEYS if key in fmt}
            if schema:
                result["json_schema"] = schema

        return cast(ResponseFormatConfig, result)

    # ==================== Stream Config ====================

    @staticmethod
    def ir_stream_config_to_p(ir_stream: StreamConfig, **kwargs: Any) -> dict:
        """IR StreamConfig → OpenAI Responses stream parameters.

        Mapping:
        - ``enabled`` → ``stream``

        Note: The Responses API does NOT support ``stream_options``
        (that is a Chat Completions-only field).  Usage is always
        included in Responses streaming events automatically.

        Args:
            ir_stream: IR stream config.

        Returns:
            Dict of OpenAI Responses request fields to merge.
        """
        result: dict[str, Any] = {}

        if "enabled" in ir_stream:
            result["stream"] = ir_stream["enabled"]

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

        Delegates to the shared shim-driven helper.  A ``reasoning_cap``
        kwarg overrides the built-in default.

        Args:
            ir_reasoning: IR reasoning config.

        Returns:
            Dict of OpenAI Responses request fields to merge.
        """
        cap = kwargs.get("reasoning_cap", DEFAULT_REASONING_CAPS["open_responses"])
        return apply_reasoning_config(
            ir_reasoning,
            cap,
            converter_type="open_responses",
        )

    @staticmethod
    def p_reasoning_config_to_ir(
        provider_reasoning: Any, **kwargs: Any
    ) -> ReasoningConfig:
        """OpenAI Responses reasoning parameters → IR ReasoningConfig.

        Args:
            provider_reasoning: Provider request dict (or subset with
                ``reasoning`` field).

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

        # A vendor dialect: some OpenAI-compatible lanes spell the on/off
        # switch ``reasoning.type``.  The spec declares no such key, and the
        # SDK's ``reasoning.mode`` is a different axis ("standard" | "pro"),
        # so the two are never interchangeable.
        reasoning_type = reasoning.get("type")
        if reasoning_type == "enabled":
            result["mode"] = "enabled"
        elif reasoning_type == "disabled":
            result["mode"] = "disabled"

        if "effort" in reasoning:
            effort = reasoning["effort"]
            if effort == "none":
                result["mode"] = "disabled"
            else:
                result["effort"] = effort

        # ``summary`` is the declared spelling; ``generate_summary`` is the
        # deprecated one the SDK still types, over the same value space, so a
        # body carrying only that field is not a body without a summary.
        if isinstance(reasoning.get("summary"), str):
            result["summary"] = reasoning["summary"]
        elif isinstance(reasoning.get("generate_summary"), str):
            result["summary"] = reasoning["generate_summary"]

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
            result["prompt_cache_retention"] = retention_to_wire(ir_cache["retention"])

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
            result["retention"] = retention_to_ir(
                provider_cache["prompt_cache_retention"]
            )

        return cast(CacheConfig, result)


# Backward-compatible alias (deprecated): the OpenAI Responses profile reuses
# the same config operations as the vendor-neutral Open Responses base.
OpenAIResponsesConfigOps = OpenResponsesConfigOps
