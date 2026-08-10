"""
LLM-Rosetta - Base Rerank Converter

Rerank 转换器抽象基类
Abstract base class for rerank converters

Parallel hierarchy to BaseConverter — rerank and chat completions are
different API categories with fundamentally different shapes (no messages,
tools, or streaming).  Shares ConversionContext for warnings/options but
otherwise stands alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from llm_rosetta.types.ir.rerank import (
    IRRerankRequest,
    IRRerankResponse,
    RerankUsageInfo,
)

from .context import ConversionContext


class BaseRerankConverter(ABC):
    """Abstract base class for rerank format converters.

    Each concrete converter implements bidirectional conversion between
    a provider's rerank API format and the IR rerank types.

    Subclasses MUST:
    - Set ``_CONVERTER_TAG`` to a unique string identifier
    - Implement all ``_do_*`` hooks and usage conversion methods

    Public methods follow the template-method pattern: they create a
    fallback ``ConversionContext``, delegate to the abstract ``_do_*``
    hook, and return the result along with any accumulated warnings.
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
        ir_request: IRRerankRequest,
        *,
        context: ConversionContext | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        """Convert IR rerank request to provider request format.

        Returns:
            Tuple of (provider_request_dict, warnings).
        """
        ctx = context if context is not None else ConversionContext()
        result = self._do_request_to_provider(ir_request, context=ctx)
        return result, ctx.warnings

    def request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext | None = None,
    ) -> IRRerankRequest:
        """Convert provider rerank request to IR format."""
        provider_request = self._normalize(provider_request)
        ctx = context if context is not None else ConversionContext()
        return self._do_request_from_provider(provider_request, context=ctx)

    def response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext | None = None,
    ) -> IRRerankResponse:
        """Convert provider rerank response to IR format."""
        provider_response = self._normalize(provider_response)
        ctx = context if context is not None else ConversionContext()
        return self._do_response_from_provider(provider_response, context=ctx)

    def response_to_provider(
        self,
        ir_response: IRRerankResponse,
        *,
        context: ConversionContext | None = None,
    ) -> dict[str, Any]:
        """Convert IR rerank response to provider format."""
        ctx = context if context is not None else ConversionContext()
        return self._do_response_to_provider(ir_response, context=ctx)

    # ==================== Abstract hooks ====================

    @abstractmethod
    def _do_request_to_provider(
        self,
        ir_request: IRRerankRequest,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def _do_request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRRerankRequest: ...

    @abstractmethod
    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRRerankResponse: ...

    @abstractmethod
    def _do_response_to_provider(
        self,
        ir_response: IRRerankResponse,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]: ...

    @staticmethod
    @abstractmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> RerankUsageInfo: ...

    @staticmethod
    @abstractmethod
    def _build_ir_usage_to_p(ir_usage: RerankUsageInfo) -> dict[str, Any]: ...

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
        """Create a conversion context for rerank conversions."""
        return ConversionContext(options=dict(options) if options else {})
