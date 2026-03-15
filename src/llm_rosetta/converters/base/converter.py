"""
LLM-Rosetta - Base Converter

定义转换器的基础接口（抽象基类，功能域组织）
Defines the basic interface for converters (abstract base class, functional domain organization)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

from ...types.ir.extensions import ExtensionItem
from ...types.ir.messages import Message
from ...types.ir.request import IRRequest
from ...types.ir.response import IRResponse
from ...types.ir.stream import IRStreamEvent
from .stream_context import StreamContext


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
    content_ops_class: Optional[Type] = None
    tool_ops_class: Optional[Type] = None
    message_ops_class: Optional[Type] = None
    config_ops_class: Optional[Type] = None

    # ==================== 顶层转换接口 Top-level conversion interface ====================

    @abstractmethod
    def request_to_provider(
        self,
        ir_request: IRRequest,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], List[str]]:
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

        Args:
            ir_request: IR格式的完整请求
            **kwargs: 额外参数

        Returns:
            Tuple[转换后的请求参数, 警告信息列表]
        """
        pass

    @abstractmethod
    def request_from_provider(
        self,
        provider_request: Dict[str, Any],
        **kwargs: Any,
    ) -> IRRequest:
        """将provider请求转换为IRRequest
        Convert provider request to IRRequest

        Args:
            provider_request: Provider格式的请求
            **kwargs: 额外参数

        Returns:
            IR格式的请求
        """
        pass

    @abstractmethod
    def response_from_provider(
        self,
        provider_response: Dict[str, Any],
        **kwargs: Any,
    ) -> IRResponse:
        """将provider响应转换为IRResponse
        Convert provider response to IRResponse

        Args:
            provider_response: Provider格式的响应
            **kwargs: 额外参数

        Returns:
            IR格式的响应
        """
        pass

    @abstractmethod
    def response_to_provider(
        self,
        ir_response: IRResponse,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """将IRResponse转换为provider响应
        Convert IRResponse to provider response

        Args:
            ir_response: IR格式的响应
            **kwargs: 额外参数

        Returns:
            Provider格式的响应
        """
        pass

    @abstractmethod
    def messages_to_provider(
        self,
        messages: Iterable[Union[Message, ExtensionItem]],
        **kwargs: Any,
    ) -> Tuple[List[Any], List[str]]:
        """将消息列表转换为provider消息格式
        Convert message list to provider message format

        这个方法通常会委托给message_ops_class来处理。
        This method typically delegates to message_ops_class for processing.

        Args:
            messages: IR格式的消息列表（可包含扩展项）
            **kwargs: 额外参数

        Returns:
            Tuple[转换后的消息列表, 警告信息列表]
        """
        pass

    @abstractmethod
    def messages_from_provider(
        self,
        provider_messages: List[Any],
        **kwargs: Any,
    ) -> List[Union[Message, ExtensionItem]]:
        """将provider消息转换为IR消息列表
        Convert provider messages to IR message list

        Args:
            provider_messages: Provider格式的消息列表
            **kwargs: 额外参数

        Returns:
            IR格式的消息列表
        """
        pass

    # ==================== Stream转换接口 Stream conversion interface ====================

    @abstractmethod
    def stream_response_from_provider(
        self,
        chunk: Dict[str, Any],
        context: Optional[StreamContext] = None,
    ) -> List[IRStreamEvent]:
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

    @abstractmethod
    def stream_response_to_provider(
        self,
        event: IRStreamEvent,
        context: Optional[StreamContext] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Convert an IR stream event to provider-native stream chunk(s).

        Some IR events (e.g., lifecycle events) may need to produce multiple
        provider chunks, or may be silently consumed (returning an empty list).

        Args:
            event: IR stream event to convert.
            context: Optional stream context for stateful conversions.

        Returns:
            A single provider-native stream chunk dict, or a list of chunk
            dicts when the event maps to multiple provider-level messages.
        """
        pass

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

    # ==================== 便利方法 Convenience methods ====================

    def message_to_provider(
        self,
        message: Union[Message, ExtensionItem],
        **kwargs: Any,
    ) -> Tuple[Any, List[str]]:
        """将单个消息转换为provider格式（便利方法）
        Convert single message to provider format (convenience method)

        Args:
            message: IR格式的单个消息
            **kwargs: 额外参数

        Returns:
            Tuple[转换后的消息, 警告信息列表]
        """
        result, warnings = self.messages_to_provider([message], **kwargs)
        return result[0] if result else None, warnings

    def message_from_provider(
        self,
        provider_message: Any,
        **kwargs: Any,
    ) -> Union[Message, ExtensionItem]:
        """将provider消息转换为IR格式（便利方法）
        Convert provider message to IR format (convenience method)

        Args:
            provider_message: Provider格式的消息
            **kwargs: 额外参数

        Returns:
            IR格式的消息
        """
        result = self.messages_from_provider([provider_message], **kwargs)
        return result[0] if result else None
