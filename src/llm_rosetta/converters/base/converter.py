"""
LLM-Rosetta - Base Converter

定义转换器的基础接口（抽象基类，功能域组织）
Defines the basic interface for converters (abstract base class, functional domain organization)
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, cast

from ...types.ir.passthrough import ProviderPassthroughEvent
from ...types.ir.request import IRInputItem, IRRequest
from ...types.ir.response import IRResponse, UsageInfo
from ...types.ir.stream import IRStreamEvent
from ...types.ir.validation import (
    validate_ir_request,
    validate_ir_response,
    validate_messages,
    validate_tools,
)
from .context import ConversionContext, StreamContext
from .passthrough import merge_provider_output_items


class BaseConverter(ABC):
    """转换器基类，定义统一的转换接口（功能域组织）
    Base class for converters, defines a unified conversion interface (functional domain organization)

    新的设计原则：
    - 按功能域组织：content, tools, messages, configs
    - 明确的转换层次：content → messages → requests/responses
    - 组合模式：子类通过类属性指定使用的ops类
    - 保持高层接口简洁：只暴露必要的转换方法

    New design principles:
    - Organized by functional domains: content, tools, messages, configs
    - Clear conversion hierarchy: content → messages → requests/responses
    - Composition pattern: subclasses specify ops classes via class attributes
    - Keep high-level interface simple: only expose necessary conversion methods
    """

    # 子类需要指定使用的ops类（按功能域组织）
    # Subclasses should specify the ops classes to use (organized by functional domains)
    content_ops_class: type | None = None
    tool_ops_class: type | None = None
    message_ops_class: type | None = None
    config_ops_class: type | None = None

    # Instance-level ops (set by subclass __init__).
    # Declared here so the type checker sees them on BaseConverter.
    tool_ops: Any
    message_ops: Any

    # Converter identity tag for cache key namespacing.
    # Subclasses MUST set this to a unique string (e.g. "anthropic").
    _CONVERTER_TAG: str = ""

    # Provider response list field for passthrough item restoration.
    # Concrete subclasses MUST set this (enforced by __init_subclass__).
    _PASSTHROUGH_RESTORE_KEY: str = ""

    # Enable/disable IR validation on from_provider output
    validate_output: bool = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if (
            not getattr(cls, "__abstractmethods__", set())
            and not cls._PASSTHROUGH_RESTORE_KEY
        ):
            raise TypeError(f"{cls.__name__} must set _PASSTHROUGH_RESTORE_KEY")

    # Default dispatch table for stream_response_to_provider.
    # Maps IR stream event types to handler method names.
    # Subclasses may override to extend or customise the mapping.
    _IR_TO_P_DISPATCH: dict[str, str] = {
        "stream_start": "_handle_ir_stream_start_to_p",
        "stream_end": "_handle_ir_stream_end_to_p",
        "content_block_start": "_handle_ir_content_block_start_to_p",
        "content_block_end": "_handle_ir_content_block_end_to_p",
        "text_delta": "_handle_ir_text_delta_to_p",
        "reasoning_delta": "_handle_ir_reasoning_delta_to_p",
        "tool_call_start": "_handle_ir_tool_call_start_to_p",
        "tool_call_delta": "_handle_ir_tool_call_delta_to_p",
        "finish": "_handle_ir_finish_to_p",
        "usage": "_handle_ir_usage_to_p",
        "provider_passthrough": "_handle_ir_passthrough_to_p",
    }

    # ==================== ID prefix utilities ====================

    @staticmethod
    def strip_response_id_prefix(raw_id: str, prefix: str) -> str:
        """Strip a provider-specific prefix from a response ID.

        Returns the stem (the ID without the prefix).  If the prefix is
        empty or the ID does not start with it, the ID is returned
        unchanged.
        """
        if prefix:
            return raw_id.removeprefix(prefix)
        return raw_id

    @staticmethod
    def add_response_id_prefix(stem: str, prefix: str) -> str:
        """Add a provider-specific prefix to a response ID stem.

        If the stem already starts with the prefix, it is returned
        unchanged to avoid double-prefixing.
        """
        if prefix and not stem.startswith(prefix):
            return f"{prefix}{stem}"
        return stem

    # ==================== Template methods (public API) ====================

    def request_to_provider(
        self,
        ir_request: IRRequest,
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], list[str]]:
        """Convert IRRequest to provider request parameters.

        Template method: creates a fallback ConversionContext, delegates to
        ``_do_request_to_provider``, and returns ``(result, ctx.warnings)``.
        """
        ctx = context if context is not None else ConversionContext()
        result = self._do_request_to_provider(ir_request, context=ctx, **kwargs)
        return result, ctx.warnings

    def request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> IRRequest:
        """Convert provider request to IRRequest.

        Template method: normalizes input, delegates to
        ``_do_request_from_provider``, and validates the IR output.
        """
        provider_request = self._normalize(provider_request)
        ctx = context if context is not None else ConversionContext()
        ir_request = self._do_request_from_provider(
            provider_request, context=ctx, **kwargs
        )
        return self._validate_ir_request(ir_request)

    def response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> IRResponse:
        """Convert provider response to IRResponse.

        Template method: normalizes input, delegates to
        ``_do_response_from_provider``, runs preserve-mode capture,
        and validates the IR output.
        """
        provider_response = self._normalize(provider_response)
        ctx = context if context is not None else ConversionContext()
        ir_response = self._do_response_from_provider(
            provider_response, context=ctx, **kwargs
        )
        if getattr(ctx, "metadata_mode", None) == "preserve":
            self._capture_preserve_metadata(provider_response, ir_response, ctx)
        return self._validate_ir_response(ir_response)

    def response_to_provider(
        self,
        ir_response: IRResponse,
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert IRResponse to provider response.

        Template method: creates a fallback ConversionContext, delegates to
        ``_do_response_to_provider``, runs preserve-mode apply,
        and restores passthrough items.
        """
        ctx = context if context is not None else ConversionContext()
        provider_response = self._do_response_to_provider(
            ir_response, context=ctx, **kwargs
        )
        if getattr(ctx, "metadata_mode", None) == "preserve":
            self._apply_preserve_metadata(
                provider_response, cast(dict[str, Any], ir_response), ctx
            )
        self._restore_response_passthrough_items(
            provider_response,
            ir_response,
            output_key=self._PASSTHROUGH_RESTORE_KEY,
            context=ctx,
        )
        return provider_response

    def messages_to_provider(
        self,
        messages: Sequence[IRInputItem],
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], list[str]]:
        """Convert IR messages to provider format.

        Default implementation delegates to ``message_ops.ir_messages_to_p``.
        Override if the converter needs custom pre/post processing.
        """
        kwargs["target_provider"] = self._CONVERTER_TAG
        return self.message_ops.ir_messages_to_p(messages, **kwargs)

    def messages_from_provider(
        self,
        provider_messages: list[Any],
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> list[IRInputItem]:
        """Convert provider messages to IR format.

        Default implementation delegates to ``message_ops.p_messages_to_ir``.
        Override if the converter needs custom pre/post processing.
        """
        return self.message_ops.p_messages_to_ir(provider_messages, **kwargs)

    # ==================== Abstract hooks (subclass implements) ====================

    @abstractmethod
    def _do_request_to_provider(
        self,
        ir_request: IRRequest,
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Format-specific IR → provider request conversion.

        Context is guaranteed non-None. Warnings should be added to
        ``context.warnings``. Return the provider request dict only.
        """
        ...

    @abstractmethod
    def _do_request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Format-specific provider → IR request conversion.

        Input is already normalized. Context is guaranteed non-None.
        Do not call ``_normalize()`` or ``_validate_ir_request()``.
        """
        ...

    @abstractmethod
    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Format-specific provider → IR response conversion.

        Input is already normalized. Context is guaranteed non-None.
        Do not call ``_normalize()`` or ``_validate_ir_response()``.
        """
        ...

    @abstractmethod
    def _do_response_to_provider(
        self,
        ir_response: IRResponse,
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Format-specific IR → provider response conversion.

        Context is guaranteed non-None. Do not call
        ``_restore_response_passthrough_items()`` or preserve-mode hooks.
        """
        ...

    # ==================== Preserve-mode hooks (override in subclass) ====================

    def _capture_preserve_metadata(
        self,
        provider_response: dict[str, Any],
        ir_response: dict[str, Any],
        ctx: ConversionContext,
    ) -> None:
        """Capture provider-specific fields for lossless round-trip.

        Called by ``response_from_provider`` after the ``_do_*`` hook.
        Override in converters that support preserve-mode metadata
        (currently Anthropic and OpenAI Responses).
        """

    def _apply_preserve_metadata(
        self,
        provider_response: dict[str, Any],
        ir_response: dict[str, Any],
        ctx: ConversionContext,
    ) -> None:
        """Re-inject captured metadata for lossless round-trip.

        Called by ``response_to_provider`` after the ``_do_*`` hook.
        Override in converters that support preserve-mode metadata
        (currently Anthropic and OpenAI Responses).
        """

    # ==================== Stream转换接口 Stream conversion interface ====================

    @abstractmethod
    def stream_response_from_provider(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None = None,
    ) -> list[IRStreamEvent]:
        """Convert a provider-native stream chunk to a list of IR stream events.

        A single provider chunk may produce zero or more IR events depending on
        the provider's SSE protocol.  For example, a chunk that carries both a
        text delta and a finish reason would yield two events.

        Args:
            chunk: Provider-native stream chunk (dict or SDK object that will
                be normalized internally by each concrete converter).
            context: Optional stream context for stateful conversions.
                When provided, converters may emit lifecycle events
                (StreamStart/End, ContentBlockStart/End) and track
                cross-chunk state.

        Returns:
            List of IR stream events extracted from the chunk.
        """
        pass

    def stream_response_to_provider(
        self,
        event: IRStreamEvent,
        context: StreamContext | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert an IR stream event to provider-native stream chunk(s).

        Uses ``_IR_TO_P_DISPATCH`` to route each event type to its handler,
        then applies ``_post_process_ir_to_p`` for any provider-specific
        decoration of the result.

        Subclasses that need pre-dispatch logic (e.g., context upgrades)
        may override this method, perform their pre-processing, and call
        ``super().stream_response_to_provider(event, context)``.

        Args:
            event: IR stream event to convert.
            context: Optional stream context for stateful conversions.

        Returns:
            A single provider-native stream chunk dict, or a list of chunk
            dicts when the event maps to multiple provider-level messages.
        """
        handler_name = self._IR_TO_P_DISPATCH.get(event.get("type", ""))
        if handler_name is None:
            return {}
        result = getattr(self, handler_name)(event, context)
        return self._post_process_ir_to_p(result, event, context)

    def _handle_p_passthrough_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Wrap a provider-native chunk in a generic passthrough event."""
        events.append(
            ProviderPassthroughEvent(
                type="provider_passthrough",
                provider=self._CONVERTER_TAG,
                payload=dict(chunk),
            )
        )

    def _handle_ir_passthrough_to_p(
        self,
        event: ProviderPassthroughEvent,
        context: StreamContext | None,
    ) -> dict[str, Any]:
        """Restore opaque events only for the originating converter dialect.

        Cross-format drops return an empty dict by convention; StreamProcessor
        filters falsy provider chunks from its output list.
        """
        if event["provider"] != self._CONVERTER_TAG:
            return {}
        return dict(event["payload"])

    def _post_process_ir_to_p(
        self,
        result: dict[str, Any] | list[dict[str, Any]],
        event: IRStreamEvent,
        context: StreamContext | None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Hook for provider-specific post-processing of stream handler results.

        Called by ``stream_response_to_provider`` after the dispatch handler
        produces its result.  The default implementation is a no-op;
        subclasses override to inject provider-specific envelope fields.

        Args:
            result: The handler's raw result (dict or list of dicts).
            event: The original IR stream event (for reference).
            context: The stream context.

        Returns:
            The (possibly modified) result.
        """
        return result

    def _restore_response_passthrough_items(
        self,
        provider_response: dict[str, Any],
        ir_response: IRResponse,
        *,
        output_key: str,
        context: ConversionContext | None,
    ) -> None:
        """Restore matching opaque output items into a provider list field."""
        passthrough_items = ir_response.get("provider_passthrough_items")
        if not passthrough_items:
            return
        portable_items = provider_response.get(output_key, [])
        if not isinstance(portable_items, list):
            portable_items = []
        merged, warnings = merge_provider_output_items(
            portable_items,
            passthrough_items,
            target_provider=self._CONVERTER_TAG,
        )
        provider_response[output_key] = merged
        if context is not None:
            context.warnings.extend(warnings)

    # ==================== Provider-specific helpers (abstract) ====================

    @staticmethod
    @abstractmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> UsageInfo:
        """Convert provider usage dict to IR usage format.

        Called by ``response_from_provider`` to normalize provider-specific
        token usage fields (e.g. ``input_tokens``, ``prompt_token_count``)
        into the IR schema (``prompt_tokens``, ``completion_tokens``, ...).
        """
        ...

    @staticmethod
    @abstractmethod
    def _build_ir_usage_to_p(ir_usage: Mapping[str, Any]) -> dict[str, Any]:
        """Convert IR usage dict to provider-specific usage format.

        Called by ``response_to_provider`` to map IR token usage fields
        back to the provider's native naming (e.g. ``promptTokenCount``
        for Google, ``input_tokens`` for Anthropic).
        """
        ...

    def _convert_p_tools_to_ir(self, tools: list[Any]) -> list[Any]:
        """Convert provider tool definitions to IR ToolDefinition list.

        Default implementation iterates *tools* and calls
        ``self.tool_ops.p_tool_definition_to_ir()`` for each entry.
        Handles Google's list/None return transparently.

        .. note::
            This is the **uncached** fallback.  In normal operation,
            ``_get_cached_p_tools_to_ir`` calls ``tool_ops`` directly
            with per-entry caching.  This method is retained for
            direct use in tests or subclass customisation.
        """
        ir_tools: list[Any] = []
        for t in tools:
            try:
                result = self.tool_ops.p_tool_definition_to_ir(t)
            except Exception as e:
                tool_type = (
                    t.get("type", "unknown")
                    if isinstance(t, dict)
                    else type(t).__name__
                )
                tool_name = (
                    (t.get("function", {}).get("name") or t.get("name", "unnamed"))
                    if isinstance(t, dict)
                    else str(t)
                )
                raise ValueError(
                    f"Unsupported tool type={tool_type!r} name={tool_name!r}: {e}"
                ) from e
            if isinstance(result, list):
                ir_tools.extend(result)
            elif result is not None:
                ir_tools.append(result)
        return ir_tools

    @abstractmethod
    def _apply_tool_config(
        self,
        ir_request: IRRequest,
        result: dict[str, Any],
        ctx: "ConversionContext",
    ) -> None:
        """Apply tools, tool_choice, and tool_config from IR to provider request.

        Called by ``request_to_provider`` to populate tool-related fields in
        the provider request dict.  Implementations should handle all three
        IR fields (``tools``, ``tool_choice``, ``tool_config``) and emit
        warnings to ``ctx`` for unsupported options.
        """
        ...

    # ==================== Normalization ====================

    @staticmethod
    def _normalize(data: Any) -> dict:
        """Normalize SDK objects to plain dicts.

        Handles Pydantic models (``model_dump()``), dataclasses, and other
        objects with dict-like conversion methods.  Subclasses may override
        this to handle provider-specific quirks (e.g. tuple unwrapping).

        Args:
            data: Input data, possibly an SDK object.

        Returns:
            Plain dict representation.

        Raises:
            TypeError: If data cannot be normalized.
        """
        if isinstance(data, dict):
            return data
        if hasattr(data, "model_dump"):
            return data.model_dump()
        if hasattr(data, "to_dict"):
            return data.to_dict()
        if hasattr(data, "__dict__"):
            return dict(data.__dict__)
        raise TypeError(f"Cannot normalize {type(data).__name__} to dict")

    # ==================== Factory methods ====================

    @classmethod
    def create_conversion_context(cls, **options: Any) -> ConversionContext:
        """Create a conversion context for non-streaming conversions.

        Args:
            **options: Initial options to populate in the context
                (e.g., ``output_format="rest"``).

        Returns:
            A new ConversionContext instance.
        """
        return ConversionContext(options=dict(options) if options else {})

    @classmethod
    def create_stream_context(cls) -> StreamContext:
        """Create a stream context appropriate for this converter.

        Subclasses may override to return a provider-specific context
        subclass with additional state fields.

        Returns:
            A new StreamContext instance.
        """
        return StreamContext()

    # ==================== IR Validation helpers ====================

    # List fields in IRRequest that support incremental validation.
    # (field_name, ir_type_tag, standalone_validator, placeholder_for_skip)
    # placeholder: value to substitute when all entries are cached.
    #   [] for Required fields (messages), None to pop for NotRequired (tools).
    _IR_VALIDATED_FIELDS: tuple[tuple[str, str, Any, Any], ...] = (
        ("tools", "ir.tool", staticmethod(validate_tools), None),
        ("messages", "ir.message", staticmethod(validate_messages), []),
    )

    @staticmethod
    def _check_field_incremental(
        data: dict[str, Any],
        field: str,
        tag: str,
        validator: Any,
        placeholder: Any,
        saved: dict[str, list[Any]],
        newly_validated: list[tuple[str, Any]],
    ) -> None:
        """Check one list field against the IR validation cache.

        Partitions entries into cached (skip) and new (validate).
        On partial/full hit, swaps the field with a placeholder so the
        main ``validate_ir_request`` pass skips it.
        """
        from .helpers.cache import is_ir_validated

        original = data.get(field)
        if not original:
            return

        new = [e for e in original if not is_ir_validated(tag, e)]
        if not new:
            # All cached — swap in placeholder
            saved[field] = original
            if placeholder is not None:
                data[field] = placeholder
            else:
                data.pop(field, None)
        elif len(new) < len(original):
            # Partial hit — validate only new entries separately
            validator(new)
            newly_validated.extend((tag, e) for e in new)
            saved[field] = original
            if placeholder is not None:
                data[field] = placeholder
            else:
                data.pop(field, None)
        # else: all new — leave in place for the main pass

    def _validate_ir_request(self, data: dict[str, Any]) -> IRRequest:
        """Validate and return an IRRequest if validate_output is enabled.

        Uses the unified ``ir_validation_cache`` (the hub) to skip
        re-validation of IR entries that have already been validated,
        regardless of which converter produced them.  Both tools and
        messages are checked against the same cache.

        Args:
            data: Dict built by a concrete converter's request_from_provider.

        Returns:
            The validated IRRequest (same object, typed).

        Raises:
            ValidationError: If validation is enabled and data is malformed.
        """
        if not self.validate_output:
            return cast(IRRequest, data)

        from .helpers.cache import mark_ir_validated

        saved: dict[str, list[Any]] = {}
        newly_validated: list[tuple[str, Any]] = []

        # Single try/finally guards both the per-field incremental
        # checks (which may swap fields) and the main validation pass.
        # If any validator raises, all swaps are restored.
        try:
            for field, tag, validator, placeholder in self._IR_VALIDATED_FIELDS:
                self._check_field_incremental(
                    data, field, tag, validator, placeholder, saved, newly_validated
                )
            result = validate_ir_request(data)
        finally:
            for field, original in saved.items():
                data[field] = original

        # Restore saved fields into result (use dict view to avoid ty
        # "invalid-key" error — TypedDict doesn't allow variable keys).
        result_dict = cast(dict[str, Any], result)
        for field, original in saved.items():
            result_dict[field] = original

        # Mark newly validated entries
        for tag, entry in newly_validated:
            mark_ir_validated(tag, entry)
        # Mark entries validated by the main pass (all-new case)
        for field, tag, _validator, _ph in self._IR_VALIDATED_FIELDS:
            if field in saved:
                continue
            entries = result_dict.get(field)
            if entries:
                for e in entries:
                    mark_ir_validated(tag, e)

        return result

    def _validate_ir_response(self, data: dict[str, Any]) -> IRResponse:
        """Validate and return an IRResponse if validate_output is enabled.

        Args:
            data: Dict built by a concrete converter's response_from_provider.

        Returns:
            The validated IRResponse (same object, typed).

        Raises:
            ValidationError: If validation is enabled and data is malformed.
        """
        if self.validate_output:
            return validate_ir_response(data)
        return cast(IRResponse, data)

    # ==================== Per-entry conversion caching ====================

    def _get_cached_p_tools_to_ir(self, tools: list[Any]) -> list[Any]:
        """Per-entry provider→IR tool conversion with caching.

        Looks up each tool individually in the conversion cache (spoke).
        On hit, also marks the IR tool in the validation hub cache so
        ``_validate_ir_request`` can skip re-hashing it.  On miss,
        converts and caches (the hub mark happens later in
        ``_validate_ir_request`` after successful validation).

        Handles Google's list/None return from ``p_tool_definition_to_ir``
        transparently.

        .. warning::
            Returned dicts are **shared references** into the cache.
            Callers **must not** mutate them.

        Args:
            tools: Provider-format tool definition list.

        Returns:
            IR tool definition list.
        """
        from .helpers.cache import _SENTINEL, get_cached_tool, put_cached_tool

        tag = self._CONVERTER_TAG + ":from_p"
        ir_tools: list[Any] = []

        for t in tools:
            cached = get_cached_tool(tag, t)
            if cached is not _SENTINEL:
                if isinstance(cached, list):
                    ir_tools.extend(cached)
                elif cached is not None:
                    ir_tools.append(cached)
                continue

            try:
                result = self.tool_ops.p_tool_definition_to_ir(t)
            except Exception as e:
                tool_type = (
                    t.get("type", "unknown")
                    if isinstance(t, dict)
                    else type(t).__name__
                )
                tool_name = (
                    (t.get("function", {}).get("name") or t.get("name", "unnamed"))
                    if isinstance(t, dict)
                    else str(t)
                )
                raise ValueError(
                    f"Unsupported tool type={tool_type!r} name={tool_name!r}: {e}"
                ) from e

            put_cached_tool(tag, t, result)
            if isinstance(result, list):
                ir_tools.extend(result)
            elif result is not None:
                ir_tools.append(result)

        return ir_tools

    def _get_cached_ir_tools_to_p(self, ir_tools: list[Any]) -> list[Any]:
        """Per-entry IR→provider tool conversion with caching.

        Looks up each IR tool individually.  On miss, converts and
        caches.

        .. warning::
            Returned dicts are **shared references** into the cache.
            Callers **must not** mutate them.

        Args:
            ir_tools: IR tool definition list.

        Returns:
            Provider-format tool definition list.
        """
        from .helpers.cache import _SENTINEL, get_cached_tool, put_cached_tool

        tag = self._CONVERTER_TAG + ":to_p"
        results: list[Any] = []

        for t in ir_tools:
            cached = get_cached_tool(tag, t)
            if cached is not _SENTINEL:
                results.append(cached)
            else:
                converted = self.tool_ops.ir_tool_definition_to_p(t)
                put_cached_tool(tag, t, converted)
                results.append(converted)

        return results

    # ==================== 便利方法 Convenience methods ====================

    def message_to_provider(
        self,
        message: IRInputItem,
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> tuple[Any, list[str]]:
        """将单个消息转换为provider格式（便利方法）
        Convert single message to provider format (convenience method)

        Args:
            message: IR格式的单个消息
            context: Optional conversion context.
            **kwargs: 额外参数

        Returns:
            Tuple[转换后的消息, 警告信息列表]
        """
        result, warnings = self.messages_to_provider(
            [message], context=context, **kwargs
        )
        return result[0] if result else None, warnings

    def message_from_provider(
        self,
        provider_message: Any,
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> IRInputItem:
        """将provider消息转换为IR格式（便利方法）
        Convert provider message to IR format (convenience method)

        Args:
            provider_message: Provider格式的消息
            context: Optional conversion context.
            **kwargs: 额外参数

        Returns:
            IR格式的消息
        """
        result = self.messages_from_provider(
            [provider_message], context=context, **kwargs
        )
        return result[0] if result else cast(IRInputItem, {})
