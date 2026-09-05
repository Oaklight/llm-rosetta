"""
LLM-Rosetta - Google Interactions Config Operations

Bidirectional conversion between Interactions API generation_config
and IR GenerationConfig / ReasoningConfig.
"""

from typing import Any, cast

from ...types.ir.configs import (
    CacheConfig,
    GenerationConfig,
    ReasoningConfig,
    ResponseFormatConfig,
    StreamConfig,
)
from ...types.ir.reasoning import IREffort, IRVisibility
from ..base import BaseConfigOps

# Interactions thinking_level ↔ IR effort (nearly 1:1)
_THINKING_LEVEL_TO_IR: dict[str, IREffort] = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
}

_IR_EFFORT_TO_THINKING_LEVEL: dict[str, str] = {
    v: k for k, v in _THINKING_LEVEL_TO_IR.items()
}
# IR has xhigh/max which Interactions doesn't support; map to high
_IR_EFFORT_TO_THINKING_LEVEL["xhigh"] = "high"
_IR_EFFORT_TO_THINKING_LEVEL["max"] = "high"


class GoogleInteractionsConfigOps(BaseConfigOps):
    """Google Interactions API config conversion operations."""

    # ── Generation config ──────────────────────────────────────────

    @staticmethod
    def ir_generation_config_to_p(ir_config: GenerationConfig, **kwargs: Any) -> dict:
        result: dict[str, Any] = {}
        if "max_tokens" in ir_config:
            result["max_output_tokens"] = ir_config["max_tokens"]
        if "stop_sequences" in ir_config:
            result["stop_sequences"] = ir_config["stop_sequences"]
        if "seed" in ir_config:
            result["seed"] = ir_config["seed"]
        return result

    @staticmethod
    def p_generation_config_to_ir(
        provider_config: Any, **kwargs: Any
    ) -> GenerationConfig:
        result: GenerationConfig = {}
        if not isinstance(provider_config, dict):
            return result
        if "max_output_tokens" in provider_config:
            result["max_tokens"] = provider_config["max_output_tokens"]
        if "stop_sequences" in provider_config:
            result["stop_sequences"] = provider_config["stop_sequences"]
        if "seed" in provider_config:
            result["seed"] = provider_config["seed"]
        return result

    # ── Reasoning config ───────────────────────────────────────────

    @staticmethod
    def ir_reasoning_to_p(ir_config: ReasoningConfig) -> dict:
        result: dict[str, Any] = {}
        if "effort" in ir_config:
            level = _IR_EFFORT_TO_THINKING_LEVEL.get(ir_config["effort"])
            if level:
                result["thinking_level"] = level
        if "summary" in ir_config:
            val = ir_config["summary"]
            if val in ("auto", "none"):
                result["thinking_summaries"] = val
        return result

    @staticmethod
    def p_reasoning_to_ir(provider_config: dict) -> ReasoningConfig:
        result: ReasoningConfig = {}
        thinking_level = provider_config.get("thinking_level")
        if thinking_level:
            effort = _THINKING_LEVEL_TO_IR.get(thinking_level)
            if effort:
                result["effort"] = effort
                result["mode"] = "enabled"
        thinking_summaries = provider_config.get("thinking_summaries")
        if thinking_summaries in ("auto", "none"):
            result["summary"] = cast(IRVisibility, thinking_summaries)
        return result

    # ── Response format ────────────────────────────────────────────

    @staticmethod
    def ir_response_format_to_p(ir_format: ResponseFormatConfig, **kwargs: Any) -> Any:
        fmt_type = ir_format.get("type", "text")
        if fmt_type == "json_object":
            return {"type": "text", "mime_type": "application/json"}
        if fmt_type == "json_schema":
            result: dict[str, Any] = {
                "type": "text",
                "mime_type": "application/json",
            }
            schema = ir_format.get("json_schema")
            if schema:
                result["response_schema"] = schema
            return result
        return None

    @staticmethod
    def p_response_format_to_ir(
        provider_format: Any, **kwargs: Any
    ) -> ResponseFormatConfig:
        result: ResponseFormatConfig = {"type": "text"}
        if isinstance(provider_format, dict):
            mime = provider_format.get("mime_type", "")
            if "json" in mime:
                schema = provider_format.get("response_schema")
                if schema:
                    result["type"] = "json_schema"
                    result["json_schema"] = schema
                else:
                    result["type"] = "json_object"
        return result

    # ── Stream/Cache stubs ─────────────────────────────────────────

    @staticmethod
    def ir_stream_config_to_p(ir_stream: StreamConfig, **kwargs: Any) -> Any:
        return ir_stream.get("enabled", False)

    @staticmethod
    def p_stream_config_to_ir(provider_stream: Any, **kwargs: Any) -> StreamConfig:
        return {"enabled": bool(provider_stream)}

    @staticmethod
    def ir_cache_config_to_p(ir_cache: CacheConfig, **kwargs: Any) -> Any:
        return {}

    @staticmethod
    def p_cache_config_to_ir(provider_cache: Any, **kwargs: Any) -> CacheConfig:
        return {}

    # ── Base class abstract method aliases ─────────────────────────

    @staticmethod
    def ir_reasoning_config_to_p(ir_reasoning: ReasoningConfig, **kwargs: Any) -> Any:
        return GoogleInteractionsConfigOps.ir_reasoning_to_p(ir_reasoning)

    @staticmethod
    def p_reasoning_config_to_ir(
        provider_reasoning: Any, **kwargs: Any
    ) -> ReasoningConfig:
        if isinstance(provider_reasoning, dict):
            return GoogleInteractionsConfigOps.p_reasoning_to_ir(provider_reasoning)
        return {}
