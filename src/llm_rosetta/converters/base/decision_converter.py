"""
LLM-Rosetta - Base Decision Converter

Decision 转换器抽象基类
Abstract base class for decision converters

Parallel hierarchy to BaseConverter, BaseEmbeddingConverter, and
BaseRerankConverter — decision APIs have a fundamentally different shape
(state + typed questions → probabilistic answers, no messages, no
streaming, no text generation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from llm_rosetta.types.ir.decision import (
    DecisionUsageInfo,
    IRDecisionRequest,
    IRDecisionResponse,
)

from .context import ConversionContext


class BaseDecisionConverter(ABC):
    """Abstract base class for decision format converters.

    Subclasses MUST:
    - Set ``_CONVERTER_TAG`` to a unique string identifier
    - Implement all ``_do_*`` hooks and usage conversion methods
    """

    _CONVERTER_TAG: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None) and not hasattr(
            cls, "_CONVERTER_TAG"
        ):
            raise TypeError(
                f"{cls.__name__} must define a _CONVERTER_TAG class attribute"
            )

    # ==================== Public template methods ====================

    def request_to_provider(
        self,
        ir_request: IRDecisionRequest,
        *,
        context: ConversionContext | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        """Convert IR decision request to provider request format."""
        ctx = context if context is not None else ConversionContext()
        result = self._do_request_to_provider(ir_request, context=ctx)
        return result, ctx.warnings

    def request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext | None = None,
    ) -> IRDecisionRequest:
        """Convert provider decision request to IR format."""
        provider_request = self._normalize(provider_request)
        ctx = context if context is not None else ConversionContext()
        return self._do_request_from_provider(provider_request, context=ctx)

    def response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext | None = None,
    ) -> IRDecisionResponse:
        """Convert provider decision response to IR format."""
        provider_response = self._normalize(provider_response)
        ctx = context if context is not None else ConversionContext()
        return self._do_response_from_provider(provider_response, context=ctx)

    def response_to_provider(
        self,
        ir_response: IRDecisionResponse,
        *,
        context: ConversionContext | None = None,
    ) -> dict[str, Any]:
        """Convert IR decision response to provider format."""
        ctx = context if context is not None else ConversionContext()
        return self._do_response_to_provider(ir_response, context=ctx)

    # ==================== Abstract hooks ====================

    @abstractmethod
    def _do_request_to_provider(
        self,
        ir_request: IRDecisionRequest,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def _do_request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRDecisionRequest: ...

    @abstractmethod
    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRDecisionResponse: ...

    @abstractmethod
    def _do_response_to_provider(
        self,
        ir_response: IRDecisionResponse,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]: ...

    @staticmethod
    @abstractmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> DecisionUsageInfo: ...

    @staticmethod
    @abstractmethod
    def _build_ir_usage_to_p(ir_usage: DecisionUsageInfo) -> dict[str, Any]: ...

    # ==================== Utilities ====================

    @staticmethod
    def _normalize(data: Any) -> dict[str, Any]:
        """Normalize SDK objects to plain dicts."""
        if isinstance(data, dict):
            return data
        if hasattr(data, "model_dump"):
            return data.model_dump()
        if hasattr(data, "to_dict"):
            return data.to_dict()
        if hasattr(data, "__dict__"):
            return dict(data.__dict__)
        raise TypeError(f"Cannot normalize {type(data).__name__} to dict")

    @classmethod
    def create_conversion_context(cls, **options: Any) -> ConversionContext:
        """Create a conversion context for decision conversions."""
        return ConversionContext(options=dict(options) if options else {})
