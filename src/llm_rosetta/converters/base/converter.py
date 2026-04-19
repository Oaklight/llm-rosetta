"""
LLM-Rosetta - Base Converter

定义转换器的基础接口（抽象基类，功能域组织）
Defines the basic interface for converters (abstract base class, functional domain organization)
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, cast

from ...types.ir.extensions import ExtensionItem
from ...types.ir.messages import Message
from ...types.ir.request import IRRequest
from ...types.ir.response import IRResponse
from ...types.ir.stream import IRStreamEvent
from ...types.ir.validation import validate_ir_request, validate_ir_response
from .context import ConversionContext, StreamContext


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

    # Enable/disable IR validation on from_provider output
    validate_output: bool = True

    # Default dispatch table for stream_response_to_provider.
    # Maps IR stream event types to handler method names.
    # Subclasses may override to extend or customise the mapping.
    _TO_P_DISPATCH: dict[str, str] = {
        "stream_start": "_handle_stream_start_to_p",
        "stream_end": "_handle_stream_end_to_p",
        "content_block_start": "_handle_content_block_start_to_p",
        "content_block_end": "_handle_content_block_end_to_p",
        "text_delta": "_handle_text_delta_to_p",
        "reasoning_delta": "_handle_reasoning_delta_to_p",
        "tool_call_start": "_handle_tool_call_start_to_p",
        "tool_call_delta": "_handle_tool_call_delta_to_p",
        "finish": "_handle_finish_to_p",
        "usage": "_handle_usage_to_p",
    }

    # ==================== 顶层转换接口 Top-level conversion interface ====================

    @abstractmethod
    def request_to_provider(
        self,
        ir_request: IRRequest,
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], list[str]]:
        """将IRRequest转换为provider请求参数
        Convert IRRequest to provider request parameters

        这是最高层的转换方法，会调用各个功能域的ops类来完成转换：
        - 使用message_ops处理messages字段
        - 使用config_ops处理generation、stream等配置字段
        - 使用tool_ops处理tools、tool_choice等工具字段

        This is the highest-level conversion method that calls ops classes from various functional domains:
        - Uses message_ops to handle messages field
        - Uses config_ops to handle generation, stream and other config fields
        - Uses tool_ops to handle tools, tool_choice and other tool fields

        Subclass helper: call ``self._apply_tool_config(ir_request, result, ctx)``
        to handle the tools / tool_choice / tool_config fields.

        Args:
            ir_request: IR格式的完整请求
            context: Optional conversion context for carrying warnings,
                options, and metadata through the pipeline.
            **kwargs: 额外参数

        Returns:
            Tuple[转换后的请求参数, 警告信息列表]
        """
        pass

    @abstractmethod
    def request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> IRRequest:
        """将provider请求转换为IRRequest
        Convert provider request to IRRequest

        Subclass helper: call ``self._convert_tools_from_p(tools)`` to convert
        provider tool definitions to IR format.

        Args:
            provider_request: Provider格式的请求
            context: Optional conversion context.
            **kwargs: 额外参数

        Returns:
            IR格式的请求
        """
        pass

    @abstractmethod
    def response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> IRResponse:
        """将provider响应转换为IRResponse
        Convert provider response to IRResponse

        Subclass helper: call ``self._build_ir_usage(p_usage)`` to convert
        provider usage to IR format.

        Args:
            provider_response: Provider格式的响应
            context: Optional conversion context.
            **kwargs: 额外参数

        Returns:
            IR格式的响应
        """
        pass

    @abstractmethod
    def response_to_provider(
        self,
        ir_response: IRResponse,
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """将IRResponse转换为provider响应
        Convert IRResponse to provider response

        Subclass helper: call ``self._build_provider_usage(ir_usage)`` to convert
        IR usage to provider format.

        Args:
            ir_response: IR格式的响应
            context: Optional conversion context.
            **kwargs: 额外参数

        Returns:
            Provider格式的响应
        """
        pass

    @abstractmethod
    def messages_to_provider(
        self,
        messages: Sequence[Message | ExtensionItem],
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], list[str]]:
        """将消息列表转换为provider消息格式
        Convert message list to provider message format

        这个方法通常会委托给message_ops_class来处理。
        This method typically delegates to message_ops_class for processing.

        Args:
            messages: IR格式的消息列表（可包含扩展项）
            context: Optional conversion context.
            **kwargs: 额外参数

        Returns:
            Tuple[转换后的消息列表, 警告信息列表]
        """
        pass

    @abstractmethod
    def messages_from_provider(
        self,
        provider_messages: list[Any],
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> list[Message | ExtensionItem]:
        """将provider消息转换为IR消息列表
        Convert provider messages to IR message list

        Args:
            provider_messages: Provider格式的消息列表
            context: Optional conversion context.
            **kwargs: 额外参数

        Returns:
            IR格式的消息列表
        """
        pass

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

        Uses ``_TO_P_DISPATCH`` to route each event type to its handler,
        then applies ``_post_process_to_provider`` for any provider-specific
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
        handler_name = self._TO_P_DISPATCH.get(event.get("type", ""))
        if handler_name is None:
            return {}
        result = getattr(self, handler_name)(event, context)
        return self._post_process_to_provider(result, event, context)

    def _post_process_to_provider(
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

    # ==================== Provider-specific helpers (abstract) ====================

    @staticmethod
    @abstractmethod
    def _build_ir_usage(p_usage: dict[str, Any]) -> dict[str, Any]:
        """Convert provider usage dict to IR usage format.

        Called by ``response_from_provider`` to normalize provider-specific
        token usage fields (e.g. ``input_tokens``, ``prompt_token_count``)
        into the IR schema (``prompt_tokens``, ``completion_tokens``, ...).
        """
        ...

    @staticmethod
    @abstractmethod
    def _build_provider_usage(ir_usage: dict[str, Any]) -> dict[str, Any]:
        """Convert IR usage dict to provider-specific usage format.

        Called by ``response_to_provider`` to map IR token usage fields
        back to the provider's native naming (e.g. ``promptTokenCount``
        for Google, ``input_tokens`` for Anthropic).
        """
        ...

    @abstractmethod
    def _convert_tools_from_p(self, tools: list[Any]) -> list[Any]:
        """Convert provider tool definitions to IR ToolDefinition list.

        Called by ``request_from_provider`` to normalize the provider's
        tool schema into IR format.  Implementations should iterate
        ``tools``, call ``self.tool_ops.p_tool_definition_to_ir()``,
        and raise ``ValueError`` for unsupported tool types.
        """
        ...

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

    # Optional preserve-mode hooks (implement if provider supports lossless
    # round-trip, currently anthropic and openai_responses):
    #   _capture_preserve_metadata(provider_response: dict, ctx) -> None
    #       Called in response_from_provider to capture non-core fields.
    #   _apply_preserve_metadata(provider_response: dict, ctx) -> None
    #       Called in response_to_provider to re-inject captured metadata.

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

    def _validate_ir_request(self, data: dict[str, Any]) -> IRRequest:
        """Validate and return an IRRequest if validate_output is enabled.

        Args:
            data: Dict built by a concrete converter's request_from_provider.

        Returns:
            The validated IRRequest (same object, typed).

        Raises:
            ValidationError: If validation is enabled and data is malformed.
        """
        if self.validate_output:
            return validate_ir_request(data)
        return cast(IRRequest, data)

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

    # ==================== 便利方法 Convenience methods ====================

    def message_to_provider(
        self,
        message: Message | ExtensionItem,
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
    ) -> Message | ExtensionItem:
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
        return result[0] if result else cast(Message, {})
