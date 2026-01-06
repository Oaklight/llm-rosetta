"""
LLMIR - Base Converter

定义转换器的基础接口（抽象基类，分层模板）
Defines the basic interface for converters (abstract base class, layered template)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ..types.ir import (
    ImagePart,
    IRInput,
    TextPart,
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
)


class BaseConverter(ABC):
    """转换器基类，定义统一的分层转换接口
    Base class for converters, defines a unified layered conversion interface
    """

    @abstractmethod
    def to_provider(
        self,
        ir_input: IRInput,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """将IR格式转换为provider特定格式
        Convert IR format to provider-specific format

        Args:
            ir_input: IR格式的输入 / IR format input
            tools: 工具定义列表 / Tool definition list
            tool_choice: 工具选择配置 / Tool choice configuration

        Returns:
            Tuple[转换后的数据, 警告信息列表] / Tuple[converted data, warning list]
        """
        pass

    @abstractmethod
    def from_provider(self, provider_data: Any) -> IRInput:
        """将provider特定格式转换为IR格式
        Convert provider-specific format to IR format

        Args:
            provider_data: provider特定格式的数据 / Provider-specific format data

        Returns:
            IR格式的数据 / IR format data
        """
        pass

    # ==================== 分层抽象方法 Layered abstract methods ====================

    @abstractmethod
    def _ir_message_to_p(self, message: Dict[str, Any], ir_input: IRInput) -> Any:
        """IR Message → Provider Message / IR消息转换为Provider消息"""
        pass

    @abstractmethod
    def _ir_content_part_to_p(
        self, content_part: Dict[str, Any], ir_input: IRInput
    ) -> Any:
        """IR ContentPart → Provider Content/Part / IR内容部分转换为Provider内容/Part"""
        pass

    @abstractmethod
    def _p_message_to_ir(self, provider_message: Any) -> Dict[str, Any]:
        """Provider Message → IR Message / Provider消息转换为IR消息"""
        pass

    @abstractmethod
    def _p_content_part_to_ir(self, provider_part: Any) -> List[Dict[str, Any]]:
        """Provider Content/Part → IR ContentPart(s) / Provider内容/Part转换为IR内容部分"""
        pass

    # ==================== 共性内容类型转换接口 Common content type conversion interfaces ====================

    @abstractmethod
    def _ir_text_to_p(self, text_part: TextPart) -> Any:
        """IR TextPart → Provider Text Content / IR文本部分转换为Provider文本内容"""
        pass

    @abstractmethod
    def _p_text_to_ir(self, provider_text: Any) -> TextPart:
        """Provider Text Content → IR TextPart / Provider文本内容转换为IR文本部分"""
        pass

    @abstractmethod
    def _ir_image_to_p(self, image_part: ImagePart) -> Any:
        """IR ImagePart → Provider Image Content / IR图像部分转换为Provider图像内容"""
        pass

    @abstractmethod
    def _p_image_to_ir(self, provider_image: Any) -> ImagePart:
        """Provider Image Content → IR ImagePart / Provider图像内容转换为IR图像部分"""
        pass

    @abstractmethod
    def _ir_tool_call_to_p(self, tool_call_part: ToolCallPart) -> Any:
        """IR ToolCallPart → Provider Tool Call / IR工具调用部分转换为Provider工具调用"""
        pass

    @abstractmethod
    def _p_tool_call_to_ir(self, provider_tool_call: Any) -> ToolCallPart:
        """Provider Tool Call → IR ToolCallPart / Provider工具调用转换为IR工具调用部分"""
        pass

    @abstractmethod
    def _ir_tool_result_to_p(self, tool_result_part: ToolResultPart) -> Any:
        """IR ToolResultPart → Provider Tool Result / IR工具结果部分转换为Provider工具结果"""
        pass

    @abstractmethod
    def _p_tool_result_to_ir(self, provider_tool_result: Any) -> ToolResultPart:
        """Provider Tool Result → IR ToolResultPart / Provider工具结果转换为IR工具结果部分"""
        pass

    @abstractmethod
    def _ir_tool_to_p(self, tool: ToolDefinition) -> Any:
        """IR ToolDefinition → Provider Tool Definition / IR工具定义转换为Provider工具定义"""
        pass

    @abstractmethod
    def _p_tool_to_ir(self, provider_tool: Any) -> ToolDefinition:
        """Provider Tool Definition → IR ToolDefinition / Provider工具定义转换为IR工具定义"""
        pass

    @abstractmethod
    def _ir_tool_choice_to_p(self, tool_choice: ToolChoice) -> Any:
        """IR ToolChoice → Provider Tool Choice Config / IR工具选择转换为Provider工具选择配置"""
        pass

    @abstractmethod
    def _p_tool_choice_to_ir(self, provider_tool_choice: Any) -> ToolChoice:
        """Provider Tool Choice Config → IR ToolChoice / Provider工具选择配置转换为IR工具选择"""
        pass

    def validate_ir_input(self, ir_input: IRInput) -> List[str]:
        """验证IR输入的有效性
        Validate IR input

        Args:
            ir_input: 要验证的IR输入 / IR input to validate

        Returns:
            验证错误信息列表，空列表表示验证通过 / Validation error list, empty means valid
        """
        errors = []

        if not isinstance(ir_input, list):
            errors.append("IRInput must be a list")
            return errors

        for i, item in enumerate(ir_input):
            if not isinstance(item, dict):
                errors.append(f"Item {i} must be a dictionary")
                continue

            # 检查是否是Message或ExtensionItem / Check if it is Message or ExtensionItem
            if "role" in item:
                # 这是一个Message / This is a Message
                if item.get("role") not in ["system", "user", "assistant", "developer"]:
                    errors.append(f"Message {i} has invalid role: {item.get('role')}")

                if "content" not in item:
                    errors.append(f"Message {i} missing required 'content' field")
                elif not isinstance(item["content"], list):
                    errors.append(f"Message {i} 'content' must be a list")

            elif "type" in item:
                # 这是一个ExtensionItem / This is an ExtensionItem
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
