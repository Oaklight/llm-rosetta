"""
LLM-Rosetta - Base Simple Converter

Generic abstract base class for non-chat converters (embedding, rerank,
decision, etc.).  Captures the shared template-method pattern — public
methods that create a fallback ConversionContext, delegate to abstract
``_do_*`` hooks, and return the result along with any accumulated
warnings.

Chat converters use the separate ``BaseConverter`` hierarchy which has
a fundamentally richer shape (messages, streaming, tools, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from .context import ConversionContext

# Type variables for IR request, response, and usage types.
# Each concrete paradigm (embedding, rerank, decision) binds these to
# its own IR dataclasses.
IRReq = TypeVar("IRReq")
IRResp = TypeVar("IRResp")
IRUsage = TypeVar("IRUsage")


class BaseSimpleConverter(ABC, Generic[IRReq, IRResp, IRUsage]):
    """Generic abstract base class for non-chat format converters.

    Provides the shared template-method scaffold that embedding, rerank,
    and decision converters all use:

    - 4 public methods (``request_to_provider``, ``request_from_provider``,
      ``response_from_provider``, ``response_to_provider``)
    - 4 abstract ``_do_*`` hooks for subclass implementation
    - 2 abstract static usage conversion methods
    - ``_normalize`` utility and ``create_conversion_context`` factory

    Subclasses MUST:
    - Set ``_CONVERTER_TAG`` to a unique string identifier
    - Implement all ``_do_*`` hooks and usage conversion methods

    Type Parameters:
        IRReq: The IR request dataclass for this paradigm.
        IRResp: The IR response dataclass for this paradigm.
        IRUsage: The IR usage-info dataclass for this paradigm.
    """

    _CONVERTER_TAG: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Detect whether this subclass is still abstract.  At
        # __init_subclass__ time, ABCMeta has not yet populated
        # __abstractmethods__ on *cls* — so we compute it ourselves by
        # checking which abstract method names from the MRO remain
        # unoverridden in the new class's own __dict__.
        abstract_names: set[str] = set()
        for base in cls.__mro__:
            for name, val in vars(base).items():
                if getattr(val, "__isabstractmethod__", False):
                    # Still abstract only if the subclass hasn't overridden it
                    if name not in cls.__dict__:
                        abstract_names.add(name)
        is_abstract = bool(abstract_names)

        if not is_abstract and not hasattr(cls, "_CONVERTER_TAG"):
            raise TypeError(
                f"{cls.__name__} must define a _CONVERTER_TAG class attribute"
            )

    # ==================== Public template methods ====================

    def request_to_provider(
        self,
        ir_request: IRReq,
        *,
        context: ConversionContext | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        """Convert an IR request to provider request format.

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
    ) -> IRReq:
        """Convert a provider request to IR format."""
        provider_request = self._normalize(provider_request)
        ctx = context if context is not None else ConversionContext()
        return self._do_request_from_provider(provider_request, context=ctx)

    def response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext | None = None,
    ) -> IRResp:
        """Convert a provider response to IR format."""
        provider_response = self._normalize(provider_response)
        ctx = context if context is not None else ConversionContext()
        return self._do_response_from_provider(provider_response, context=ctx)

    def response_to_provider(
        self,
        ir_response: IRResp,
        *,
        context: ConversionContext | None = None,
    ) -> dict[str, Any]:
        """Convert an IR response to provider format."""
        ctx = context if context is not None else ConversionContext()
        return self._do_response_to_provider(ir_response, context=ctx)

    # ==================== Abstract hooks ====================

    @abstractmethod
    def _do_request_to_provider(
        self,
        ir_request: IRReq,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def _do_request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRReq: ...

    @abstractmethod
    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRResp: ...

    @abstractmethod
    def _do_response_to_provider(
        self,
        ir_response: IRResp,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]: ...

    @staticmethod
    @abstractmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> IRUsage: ...  # type: ignore[type-var]

    @staticmethod
    @abstractmethod
    def _build_ir_usage_to_p(ir_usage: IRUsage) -> dict[str, Any]: ...  # type: ignore[type-var]

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
        """Create a conversion context for this paradigm."""
        return ConversionContext(options=dict(options) if options else {})
