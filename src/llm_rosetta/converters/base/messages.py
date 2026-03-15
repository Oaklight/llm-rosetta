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
from typing import Any, Iterable, List, Optional, Tuple, Union, cast

from ...types.ir.extensions import ExtensionItem
from ...types.ir.messages import Message


class BaseMessageOps(ABC):
    """消息转换操作的抽象基类
    Abstract base class for message conversion operations

    处理完整消息（role + content）的转换，是content层和request/response层之间的桥梁。
    Handles conversion of complete messages (role + content), serving as a bridge between content layer and request/response layer.
    """

    # ==================== 批量消息转换 Batch message conversion ====================

    @staticmethod
    @abstractmethod
    def ir_messages_to_p(
        ir_messages: Iterable[Union[Message, ExtensionItem]], **kwargs: Any
    ) -> Tuple[List[Any], List[str]]:
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
        provider_messages: List[Any], **kwargs: Any
    ) -> List[Union[Message, ExtensionItem]]:
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
        self, ir_message: Union[Message, ExtensionItem], **kwargs: Any
    ) -> Tuple[Any, List[str]]:
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
    ) -> Optional[Union[Message, ExtensionItem]]:
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

    def validate_messages(
        self, messages: Iterable[Union[Message, ExtensionItem]]
    ) -> List[str]:
        """验证消息列表的有效性（可选实现）
        Validate message list validity (optional implementation)

        子类可以重写此方法来提供特定的验证逻辑。
        Subclasses can override this method to provide specific validation logic.

        Args:
            messages: 要验证的消息列表

        Returns:
            验证错误信息列表，空列表表示验证通过
        """
        errors = []

        if not isinstance(messages, Iterable) or isinstance(messages, (str, dict)):
            errors.append("Messages must be an iterable (but not a string or dict)")
            return errors

        for i, item in enumerate(messages):
            if not isinstance(item, dict):
                errors.append(f"Item {i} must be a dictionary")
                continue

            # 检查是否是Message或ExtensionItem
            if "role" in item:
                # 这是一个Message
                if item.get("role") not in ["system", "user", "assistant", "developer"]:
                    errors.append(f"Message {i} has invalid role: {item.get('role')}")

                if "content" not in item:
                    errors.append(f"Message {i} missing required 'content' field")
                elif not isinstance(cast(dict, item)["content"], list):
                    errors.append(f"Message {i} 'content' must be a list")

            elif "type" in item:
                # 这是一个ExtensionItem
                item_type = item.get("type")
                if item_type not in [
                    "system_event",
                    "batch_marker",
                    "session_control",
                    "tool_chain_node",
                ]:
                    errors.append(f"ExtensionItem {i} has invalid type: {item_type}")
            else:
                errors.append(
                    f"Item {i} must have either 'role' (Message) or 'type' (ExtensionItem)"
                )

        return errors
