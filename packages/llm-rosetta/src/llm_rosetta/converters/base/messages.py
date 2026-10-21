"""
LLM-Rosetta - Base Message Operations
消息转换操作的抽象基类
Abstract base class for message conversion operations

处理消息级别的转换：
- 批量消息转换：处理完整的消息列表
- 消息组合：role + content的组合处理
- 扩展项处理：系统事件、批次标记等（如果需要）
Handles message-level conversions:
- Batch message conversion: processing complete message lists
- Message composition: combined processing of role + content
- Extension item handling: system events, batch markers, etc. (if needed)

注意：这一层会调用content.py和tools.py中的方法来处理消息内容。
Note: This layer will call methods from content.py and tools.py to handle message content.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from ..._vendor.validate import ValidationError, validate
from ...types.ir.request import IRInputItem
from ...types.ir.type_guards import is_provider_passthrough_item
from .passthrough import restore_provider_passthrough_item


class BaseMessageOps(ABC):
    """消息转换操作的抽象基类
    Abstract base class for message conversion operations

    处理完整消息（role + content）的转换，是content层和request/response层之间的桥梁。
    Handles conversion of complete messages (role + content), serving as a bridge between content layer and request/response layer.
    """

    def _restore_provider_passthrough_item(
        self,
        item: IRInputItem,
        output: list[Any],
        *,
        target_provider: str,
    ) -> list[str] | None:
        """Restore a matching provider item, or return None if not passthrough."""
        if not is_provider_passthrough_item(item):
            return None
        payload, warning = restore_provider_passthrough_item(
            item, target_provider=target_provider
        )
        if payload is not None:
            output.append(payload)
        return [warning] if warning is not None else []

    # ==================== 批量消息转换 Batch message conversion ====================

    @staticmethod
    @abstractmethod
    def ir_messages_to_p(
        ir_messages: Sequence[IRInputItem], **kwargs: Any
    ) -> tuple[list[Any], list[str]]:
        """IR Messages → Provider Messages
        将IR消息列表转换为Provider消息列表

        这是消息转换的核心方法，处理：
        - 不同角色的消息：system, user, assistant, tool
        - 消息内容的转换：调用content和tools层的方法
        - 扩展项的处理：根据provider能力决定如何处理
        - 警告信息的收集：不支持的功能、转换损失等

        This is the core method for message conversion, handling:
        - Messages of different roles: system, user, assistant, tool
        - Message content conversion: calling methods from content and tools layers
        - Extension item processing: deciding how to handle based on provider capabilities
        - Warning collection: unsupported features, conversion losses, etc.

        Args:
            ir_messages: IR格式的消息列表（可包含扩展项）
            **kwargs: 额外参数，可能包含上下文信息

        Returns:
            Tuple[转换后的消息列表, 警告信息列表]
        """
        pass

    @staticmethod
    @abstractmethod
    def p_messages_to_ir(
        provider_messages: list[Any], **kwargs: Any
    ) -> list[IRInputItem]:
        """Provider Messages → IR Messages
        将Provider消息列表转换为IR消息列表

        处理从provider格式到IR格式的转换：
        - 识别消息角色和内容类型
        - 调用相应的content和tools转换方法
        - 处理provider特有的消息格式
        - 生成适当的扩展项（如果需要）

        Handles conversion from provider format to IR format:
        - Identifying message roles and content types
        - Calling appropriate content and tools conversion methods
        - Handling provider-specific message formats
        - Generating appropriate extension items (if needed)

        Args:
            provider_messages: Provider格式的消息列表
            **kwargs: 额外参数

        Returns:
            IR格式的消息列表
        """
        pass

    # ==================== 单个消息转换（可选的便利方法） Single message conversion (optional convenience methods) ====================

    def ir_message_to_p(
        self, ir_message: IRInputItem, **kwargs: Any
    ) -> tuple[Any, list[str]]:
        """IR Message → Provider Message（便利方法）
        将单个IR消息转换为Provider消息（便利方法）

        这是一个便利方法，内部调用ir_messages_to_p处理单个消息。
        子类通常不需要重写此方法。

        This is a convenience method that internally calls ir_messages_to_p for a single message.
        Subclasses typically don't need to override this method.

        Args:
            ir_message: IR格式的单个消息
            **kwargs: 额外参数

        Returns:
            Tuple[转换后的消息, 警告信息列表]
        """
        result, warnings = self.ir_messages_to_p([ir_message], **kwargs)
        return result[0] if result else None, warnings

    def p_message_to_ir(
        self, provider_message: Any, **kwargs: Any
    ) -> IRInputItem | None:
        """Provider Message → IR Message（便利方法）
        将Provider消息转换为IR消息（便利方法）

        这是一个便利方法，内部调用p_messages_to_ir处理单个消息。
        子类通常不需要重写此方法。

        This is a convenience method that internally calls p_messages_to_ir for a single message.
        Subclasses typically don't need to override this method.

        Args:
            provider_message: Provider格式的消息
            **kwargs: 额外参数

        Returns:
            IR格式的消息
        """
        result = self.p_messages_to_ir([provider_message], **kwargs)
        return result[0] if result else None

    # ==================== 辅助方法（子类可选实现） Helper methods (optional for subclasses) ====================

    def validate_messages(self, messages: Sequence[IRInputItem]) -> list[str]:
        """Validate message list against IR Message/ExtensionItem types.

        Uses zerodep validate for structural validation against TypedDict
        definitions. Subclasses can override for provider-specific logic.

        Args:
            messages: Message list to validate.

        Returns:
            List of validation error strings; empty list means valid.
        """
        try:
            validate(list(messages), list[IRInputItem])
            return []
        except ValidationError as e:
            return [err.message for err in e.errors]
