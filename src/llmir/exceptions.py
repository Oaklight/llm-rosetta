"""
LLMIR - 统一错误处理和异常定义
Unified error handling and exception definitions
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ErrorSeverity(Enum):
    """错误严重程度
    Error severity levels
    """

    LOW = "low"  # 轻微错误，可以继续处理
    MEDIUM = "medium"  # 中等错误，可能影响结果质量
    HIGH = "high"  # 严重错误，需要用户注意
    CRITICAL = "critical"  # 致命错误，必须停止处理


class ErrorCategory(Enum):
    """错误分类
    Error categories
    """

    VALIDATION = "validation"  # 输入验证错误
    CONVERSION = "conversion"  # 转换过程错误
    TOOL_EXECUTION = "tool_execution"  # 工具执行错误
    PROVIDER_API = "provider_api"  # Provider API错误
    CONFIGURATION = "configuration"  # 配置错误
    NETWORK = "network"  # 网络错误
    AUTHENTICATION = "authentication"  # 认证错误
    RATE_LIMIT = "rate_limit"  # 速率限制错误
    UNKNOWN = "unknown"  # 未知错误


class LLMIRError(Exception):
    """LLMIR基础异常类
    Base exception class for LLMIR
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
        suggestions: Optional[List[str]] = None,
    ):
        """
        初始化LLMIR错误

        Args:
            message: 错误消息
            category: 错误分类
            severity: 错误严重程度
            context: 错误上下文信息
            original_error: 原始异常（如果有）
            suggestions: 解决建议
        """
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.original_error = original_error
        self.suggestions = suggestions or []

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        Convert to dictionary format
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context,
            "original_error": str(self.original_error) if self.original_error else None,
            "suggestions": self.suggestions,
        }

    def __str__(self) -> str:
        """字符串表示"""
        parts = [f"[{self.category.value.upper()}] {self.message}"]

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")

        if self.suggestions:
            parts.append(f"Suggestions: {'; '.join(self.suggestions)}")

        return " | ".join(parts)


class ValidationError(LLMIRError):
    """输入验证错误
    Input validation error
    """

    def __init__(
        self,
        message: str,
        field_path: Optional[str] = None,
        invalid_value: Any = None,
        expected_type: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {}) or {}
        if field_path:
            context["field_path"] = field_path
        if invalid_value is not None:
            context["invalid_value"] = str(invalid_value)
        if expected_type:
            context["expected_type"] = expected_type

        kwargs["context"] = context
        kwargs["category"] = ErrorCategory.VALIDATION
        kwargs["severity"] = ErrorSeverity.HIGH

        super().__init__(message, **kwargs)


class ConversionError(LLMIRError):
    """转换错误
    Conversion error
    """

    def __init__(
        self,
        message: str,
        source_format: Optional[str] = None,
        target_format: Optional[str] = None,
        item_index: Optional[int] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if source_format:
            context["source_format"] = source_format
        if target_format:
            context["target_format"] = target_format
        if item_index is not None:
            context["item_index"] = item_index

        kwargs["context"] = context
        kwargs["category"] = ErrorCategory.CONVERSION
        kwargs["severity"] = ErrorSeverity.MEDIUM

        super().__init__(message, **kwargs)


class ToolExecutionError(LLMIRError):
    """工具执行错误
    Tool execution error
    """

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_input: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if tool_name:
            context["tool_name"] = tool_name
        if tool_call_id:
            context["tool_call_id"] = tool_call_id
        if tool_input:
            context["tool_input"] = tool_input

        kwargs["context"] = context
        kwargs["category"] = ErrorCategory.TOOL_EXECUTION
        kwargs["severity"] = ErrorSeverity.MEDIUM

        super().__init__(message, **kwargs)


class ProviderAPIError(LLMIRError):
    """Provider API错误
    Provider API error
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if provider:
            context["provider"] = provider
        if status_code:
            context["status_code"] = status_code
        if error_code:
            context["error_code"] = error_code

        kwargs["context"] = context
        kwargs["category"] = ErrorCategory.PROVIDER_API
        kwargs["severity"] = ErrorSeverity.HIGH

        super().__init__(message, **kwargs)


class ConfigurationError(LLMIRError):
    """配置错误
    Configuration error
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Any = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if config_key:
            context["config_key"] = config_key
        if config_value is not None:
            context["config_value"] = str(config_value)

        kwargs["context"] = context
        kwargs["category"] = ErrorCategory.CONFIGURATION
        kwargs["severity"] = ErrorSeverity.HIGH

        super().__init__(message, **kwargs)


class WarningInfo:
    """警告信息类
    Warning information class
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.LOW,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
    ):
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.suggestions = suggestions or []

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context,
            "suggestions": self.suggestions,
        }

    def __str__(self) -> str:
        """字符串表示"""
        parts = [f"[{self.category.value.upper()}] {self.message}"]

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")

        return " | ".join(parts)


class ErrorHandler:
    """统一错误处理器
    Unified error handler
    """

    @staticmethod
    def wrap_exception(
        original_error: Exception,
        message: Optional[str] = None,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
    ) -> LLMIRError:
        """
        包装原始异常为LLMIR错误
        Wrap original exception as LLMIR error
        """
        if isinstance(original_error, LLMIRError):
            return original_error

        error_message = message or str(original_error)

        # 根据原始异常类型选择合适的LLMIR异常类型
        if isinstance(original_error, (ValueError, TypeError)):
            return ValidationError(
                error_message,
                original_error=original_error,
                context=context,
                suggestions=suggestions,
            )
        elif isinstance(original_error, KeyError):
            return ValidationError(
                error_message,
                field_path=str(original_error).strip("'\""),
                original_error=original_error,
                context=context,
                suggestions=suggestions,
            )
        else:
            return LLMIRError(
                error_message,
                category=category,
                severity=severity,
                context=context,
                original_error=original_error,
                suggestions=suggestions,
            )

    @staticmethod
    def create_validation_error(
        message: str,
        field_path: Optional[str] = None,
        invalid_value: Any = None,
        expected_type: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
    ) -> ValidationError:
        """创建验证错误"""
        return ValidationError(
            message=message,
            field_path=field_path,
            invalid_value=invalid_value,
            expected_type=expected_type,
            suggestions=suggestions,
        )

    @staticmethod
    def create_conversion_error(
        message: str,
        source_format: Optional[str] = None,
        target_format: Optional[str] = None,
        item_index: Optional[int] = None,
        suggestions: Optional[List[str]] = None,
    ) -> ConversionError:
        """创建转换错误"""
        return ConversionError(
            message=message,
            source_format=source_format,
            target_format=target_format,
            item_index=item_index,
            suggestions=suggestions,
        )

    @staticmethod
    def create_tool_error(
        message: str,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_input: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
    ) -> ToolExecutionError:
        """创建工具执行错误"""
        return ToolExecutionError(
            message=message,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_input=tool_input,
            suggestions=suggestions,
        )

    @staticmethod
    def create_warning(
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.LOW,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
    ) -> WarningInfo:
        """创建警告信息"""
        return WarningInfo(
            message=message,
            category=category,
            severity=severity,
            context=context,
            suggestions=suggestions,
        )


# 便捷函数
def create_validation_error(
    message: str,
    field_path: Optional[str] = None,
    invalid_value: Any = None,
    expected_type: Optional[str] = None,
    suggestions: Optional[List[str]] = None,
) -> ValidationError:
    """创建验证错误的便捷函数"""
    return ErrorHandler.create_validation_error(
        message, field_path, invalid_value, expected_type, suggestions
    )


def create_conversion_error(
    message: str,
    source_format: Optional[str] = None,
    target_format: Optional[str] = None,
    item_index: Optional[int] = None,
    suggestions: Optional[List[str]] = None,
) -> ConversionError:
    """创建转换错误的便捷函数"""
    return ErrorHandler.create_conversion_error(
        message, source_format, target_format, item_index, suggestions
    )


def create_tool_error(
    message: str,
    tool_name: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    tool_input: Optional[Dict[str, Any]] = None,
    suggestions: Optional[List[str]] = None,
) -> ToolExecutionError:
    """创建工具执行错误的便捷函数"""
    return ErrorHandler.create_tool_error(
        message, tool_name, tool_call_id, tool_input, suggestions
    )


def create_warning(
    message: str,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.LOW,
    context: Optional[Dict[str, Any]] = None,
    suggestions: Optional[List[str]] = None,
) -> WarningInfo:
    """创建警告信息的便捷函数"""
    return ErrorHandler.create_warning(
        message, category, severity, context, suggestions
    )
