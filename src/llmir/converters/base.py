"""
LLMIR - Base Converter

定义转换器的基础接口
Defines the basic interface for converters
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from ..exceptions import (
    ConversionError,
    ErrorCategory,
    ErrorHandler,
    ErrorSeverity,
    ValidationError,
    WarningInfo,
    create_validation_error,
    create_warning,
)
from ..types.ir import IRInput, ToolChoice, ToolDefinition


class BaseConverter(ABC):
    """转换器基类，定义统一的转换接口
    Base class for converters, defines a unified conversion interface
    """

    @abstractmethod
    def to_provider(
        self,
        ir_input: IRInput,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> Tuple[Dict[str, Any], List[Union[str, WarningInfo]]]:
        """
        将IR格式转换为provider特定格式

        Args:
            ir_input: IR格式的输入
            tools: 工具定义列表
            tool_choice: 工具选择配置

        Returns:
            Tuple[转换后的数据, 警告信息列表]

        Raises:
            ValidationError: 输入验证失败
            ConversionError: 转换过程中出现错误
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

    def validate_ir_input(self, ir_input: IRInput) -> List[ValidationError]:
        """
        验证IR输入的有效性

        Args:
            ir_input: 要验证的IR输入

        Returns:
            验证错误列表，空列表表示验证通过
        """
        errors = []

        if not isinstance(ir_input, list):
            errors.append(
                create_validation_error(
                    "IRInput must be a list",
                    field_path="ir_input",
                    invalid_value=type(ir_input).__name__,
                    expected_type="list",
                    suggestions=[
                        "Ensure the input is a list of Message or ExtensionItem objects"
                    ],
                )
            )
            return errors

        for i, item in enumerate(ir_input):
            if not isinstance(item, dict):
                errors.append(
                    create_validation_error(
                        f"Item {i} must be a dictionary",
                        field_path=f"ir_input[{i}]",
                        invalid_value=type(item).__name__,
                        expected_type="dict",
                        suggestions=[
                            "Each item should be a Message or ExtensionItem dictionary"
                        ],
                    )
                )
                continue

            # 检查是否是Message或ExtensionItem Check if it is Message or ExtensionItem
            if "role" in item:
                # 这是一个Message This is a Message
                role = item.get("role")
                if role not in ["system", "user", "assistant", "developer"]:
                    errors.append(
                        create_validation_error(
                            f"Message {i} has invalid role: {role}",
                            field_path=f"ir_input[{i}].role",
                            invalid_value=role,
                            expected_type="'system', 'user', 'assistant', or 'developer'",
                            suggestions=["Use one of the supported role values"],
                        )
                    )

                if "content" not in item:
                    errors.append(
                        create_validation_error(
                            f"Message {i} missing required 'content' field",
                            field_path=f"ir_input[{i}].content",
                            expected_type="list",
                            suggestions=[
                                "Add a 'content' field with a list of ContentPart objects"
                            ],
                        )
                    )
                elif not isinstance(item["content"], list):
                    errors.append(
                        create_validation_error(
                            f"Message {i} 'content' must be a list",
                            field_path=f"ir_input[{i}].content",
                            invalid_value=type(item["content"]).__name__,
                            expected_type="list",
                            suggestions=[
                                "Content should be a list of ContentPart objects"
                            ],
                        )
                    )

            elif "type" in item:
                # 这是一个ExtensionItem This is an ExtensionItem
                item_type = item.get("type")
                valid_types = [
                    "system_event",
                    "batch_marker",
                    "session_control",
                    "tool_chain_node",
                ]
                if item_type not in valid_types:
                    errors.append(
                        create_validation_error(
                            f"ExtensionItem {i} has invalid type: {item_type}",
                            field_path=f"ir_input[{i}].type",
                            invalid_value=item_type,
                            expected_type=f"one of {valid_types}",
                            suggestions=["Use a supported ExtensionItem type"],
                        )
                    )
            else:
                errors.append(
                    create_validation_error(
                        f"Item {i} must have either 'role' (Message) or 'type' (ExtensionItem)",
                        field_path=f"ir_input[{i}]",
                        suggestions=[
                            "Add a 'role' field for Message objects",
                            "Add a 'type' field for ExtensionItem objects",
                        ],
                    )
                )

        return errors

    def validate_and_raise(self, ir_input: IRInput) -> None:
        """
        验证IR输入，如果有错误则抛出异常
        Validate IR input and raise exception if there are errors

        Args:
            ir_input: 要验证的IR输入

        Raises:
            ValidationError: 如果验证失败
        """
        errors = self.validate_ir_input(ir_input)
        if errors:
            # 合并多个错误信息
            error_messages = [str(error) for error in errors]
            combined_message = (
                f"IR input validation failed with {len(errors)} error(s): "
                + "; ".join(error_messages)
            )

            raise ValidationError(
                combined_message,
                context={
                    "total_errors": len(errors),
                    "error_details": [error.to_dict() for error in errors],
                },
                suggestions=[
                    "Check the IR input format according to the documentation",
                    "Ensure all required fields are present",
                    "Verify data types match the expected schema",
                ],
            )

    def create_warning(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.CONVERSION,
        severity: ErrorSeverity = ErrorSeverity.LOW,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
    ) -> WarningInfo:
        """
        创建警告信息的便捷方法
        Convenience method to create warning information
        """
        return create_warning(message, category, severity, context, suggestions)

    def handle_conversion_error(
        self,
        error: Exception,
        source_format: str,
        target_format: str,
        item_index: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ConversionError:
        """
        处理转换错误的便捷方法
        Convenience method to handle conversion errors
        """
        return ErrorHandler.wrap_exception(
            error,
            message=f"Failed to convert from {source_format} to {target_format}",
            category=ErrorCategory.CONVERSION,
            context={
                "source_format": source_format,
                "target_format": target_format,
                "item_index": item_index,
                **(context or {}),
            },
        )
