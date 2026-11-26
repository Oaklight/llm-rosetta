"""Configuration operations for the ACP lane.

ACP sets a model and a thinking effort through session configuration
options, not through the request body, so nothing here is rendered into the
turn.  The writers report what the lane cannot carry (per-request sampling,
response formats, cache control) instead of dropping it silently, and the
readers reproduce the IR shapes the lane is able to express.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta.ir.model.configs import (
    CacheConfig,
    GenerationConfig,
    ReasoningConfig,
    ResponseFormatConfig,
    StreamConfig,
)
from ...base.configs import BaseConfigOps
from llm_rosetta.conversion_warnings import DROPPED, report_loss


def _drop(what: str, **kwargs: Any) -> None:
    report_loss(
        f"ACP lanes cannot set {what} per request; it was dropped.",
        code=DROPPED,
        subject=what,
        warnings=kwargs.get("warnings"),
    )


class AcpConfigOps(BaseConfigOps):
    """Configuration vocabulary of the ACP lane."""

    @staticmethod
    def ir_generation_config_to_p(
        ir_config: GenerationConfig, **kwargs: Any
    ) -> dict[str, Any] | None:
        _drop("generation parameters", **kwargs)
        return None

    @staticmethod
    def p_generation_config_to_ir(
        provider_config: Any, **kwargs: Any
    ) -> GenerationConfig:
        return GenerationConfig()

    @staticmethod
    def ir_response_format_to_p(
        ir_format: ResponseFormatConfig, **kwargs: Any
    ) -> dict[str, Any] | None:
        _drop("an output format", **kwargs)
        return None

    @staticmethod
    def p_response_format_to_ir(
        provider_format: Any, **kwargs: Any
    ) -> ResponseFormatConfig:
        return ResponseFormatConfig()

    @staticmethod
    def ir_stream_config_to_p(
        ir_stream: StreamConfig, **kwargs: Any
    ) -> dict[str, Any] | None:
        # Streaming is a property of how the transport reads the agent, not a
        # field of the turn: nothing to write, and nothing lost.
        return None

    @staticmethod
    def p_stream_config_to_ir(provider_stream: Any, **kwargs: Any) -> StreamConfig:
        return StreamConfig()

    @staticmethod
    def ir_reasoning_config_to_p(
        ir_reasoning: ReasoningConfig, **kwargs: Any
    ) -> dict[str, Any] | None:
        _drop("a reasoning configuration", **kwargs)
        return None

    @staticmethod
    def p_reasoning_config_to_ir(
        provider_reasoning: Any, **kwargs: Any
    ) -> ReasoningConfig:
        return ReasoningConfig()

    @staticmethod
    def ir_cache_config_to_p(
        ir_cache: CacheConfig, **kwargs: Any
    ) -> dict[str, Any] | None:
        _drop("cache control", **kwargs)
        return None

    @staticmethod
    def p_cache_config_to_ir(provider_cache: Any, **kwargs: Any) -> CacheConfig:
        return CacheConfig()


__all__ = ["AcpConfigOps"]
