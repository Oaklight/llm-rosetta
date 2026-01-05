"""
LLM Provider Converter - Base Converter

定义转换器的基础接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ..types.ir import IRInput, ToolChoice, ToolDefinition


class BaseConverter(ABC):
    """转换器基类，定义统一的转换接口"""

    @abstractmethod
    def to_provider(
        self,
        ir_input: IRInput,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        将IR格式转换为provider特定格式

        Args:
            ir_input: IR格式的输入
            tools: 工具定义列表
            tool_choice: 工具选择配置

        Returns:
            Tuple[转换后的数据, 警告信息列表]
        """
        pass

    @abstractmethod
    def from_provider(self, provider_data: Any) -> IRInput:
        """
        将provider特定格式转换为IR格式

        Args:
            provider_data: provider特定格式的数据

        Returns:
            IR格式的数据
        """
        pass

    def validate_ir_input(self, ir_input: IRInput) -> List[str]:
        """
        验证IR输入的有效性

        Args:
            ir_input: 要验证的IR输入

        Returns:
            验证错误信息列表，空列表表示验证通过
        """
        errors = []

        if not isinstance(ir_input, list):
            errors.append("IRInput must be a list")
            return errors

        for i, item in enumerate(ir_input):
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
                elif not isinstance(item["content"], list):
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
