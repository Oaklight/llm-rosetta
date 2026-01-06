"""
测试统一错误处理机制
Test unified error handling mechanism
"""

import pytest

from llmir.converters.anthropic_converter import AnthropicConverter
from llmir.exceptions import (
    ConfigurationError,
    ConversionError,
    ErrorCategory,
    ErrorHandler,
    ErrorSeverity,
    LLMIRError,
    ProviderAPIError,
    ToolExecutionError,
    ValidationError,
    WarningInfo,
    create_conversion_error,
    create_tool_error,
    create_validation_error,
    create_warning,
)
from llmir.types.ir import IRInput, ToolResultPart
from llmir.utils.error_converter import ErrorConverter


class TestErrorExceptions:
    """测试错误异常类"""

    def test_llmir_error_basic(self):
        """测试基础LLMIR错误"""
        error = LLMIRError(
            "Test error",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            context={"field": "test"},
            suggestions=["Fix the field"],
        )

        assert (
            str(error)
            == "[VALIDATION] Test error | Context: field=test | Suggestions: Fix the field"
        )
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.HIGH

        error_dict = error.to_dict()
        assert error_dict["error_type"] == "LLMIRError"
        assert error_dict["message"] == "Test error"
        assert error_dict["category"] == "validation"
        assert error_dict["severity"] == "high"

    def test_validation_error(self):
        """测试验证错误"""
        error = create_validation_error(
            "Invalid field value",
            field_path="ir_input[0].role",
            invalid_value="invalid_role",
            expected_type="'system', 'user', 'assistant', or 'developer'",
            suggestions=["Use a valid role"],
        )

        assert isinstance(error, ValidationError)
        assert error.category == ErrorCategory.VALIDATION
        assert error.context["field_path"] == "ir_input[0].role"
        assert error.context["invalid_value"] == "invalid_role"

    def test_conversion_error(self):
        """测试转换错误"""
        error = create_conversion_error(
            "Failed to convert content",
            source_format="IR",
            target_format="Anthropic",
            item_index=0,
            suggestions=["Check content format"],
        )

        assert isinstance(error, ConversionError)
        assert error.category == ErrorCategory.CONVERSION
        assert error.context["source_format"] == "IR"
        assert error.context["target_format"] == "Anthropic"
        assert error.context["item_index"] == 0

    def test_tool_execution_error(self):
        """测试工具执行错误"""
        error = create_tool_error(
            "Tool execution failed",
            tool_name="get_weather",
            tool_call_id="call_123",
            tool_input={"city": "Beijing"},
            suggestions=["Check API key"],
        )

        assert isinstance(error, ToolExecutionError)
        assert error.category == ErrorCategory.TOOL_EXECUTION
        assert error.context["tool_name"] == "get_weather"
        assert error.context["tool_call_id"] == "call_123"

    def test_provider_api_error(self):
        """测试Provider API错误"""
        error = ProviderAPIError(
            "API request failed",
            provider="anthropic",
            status_code=429,
            error_code="RATE_LIMIT",
            suggestions=["Reduce request rate", "Check API quota"],
        )

        assert isinstance(error, ProviderAPIError)
        assert error.category == ErrorCategory.PROVIDER_API
        assert error.context["provider"] == "anthropic"
        assert error.context["status_code"] == 429
        assert error.context["error_code"] == "RATE_LIMIT"

    def test_configuration_error(self):
        """测试配置错误"""
        error = ConfigurationError(
            "Invalid API key format",
            config_key="api_key",
            config_value="invalid_key",
            suggestions=["Check API key format", "Verify configuration file"],
        )

        assert isinstance(error, ConfigurationError)
        assert error.category == ErrorCategory.CONFIGURATION
        assert error.context["config_key"] == "api_key"
        assert error.context["config_value"] == "invalid_key"

    def test_warning_info(self):
        """测试警告信息"""
        warning = create_warning(
            "Feature not supported",
            category=ErrorCategory.CONVERSION,
            severity=ErrorSeverity.MEDIUM,
            context={"feature": "system_event"},
            suggestions=["Use alternative approach"],
        )

        assert isinstance(warning, WarningInfo)
        assert warning.category == ErrorCategory.CONVERSION
        assert warning.severity == ErrorSeverity.MEDIUM
        assert (
            str(warning)
            == "[CONVERSION] Feature not supported | Context: feature=system_event"
        )

    def test_error_handler_wrap_exception(self):
        """测试错误处理器包装异常"""
        original_error = ValueError("Invalid value")

        wrapped_error = ErrorHandler.wrap_exception(
            original_error,
            message="Custom message",
            category=ErrorCategory.VALIDATION,
            context={"field": "test"},
        )

        assert isinstance(wrapped_error, ValidationError)
        assert wrapped_error.message == "Custom message"
        assert wrapped_error.original_error == original_error

    def test_error_handler_wrap_key_error(self):
        """测试错误处理器包装KeyError"""
        original_error = KeyError("missing_field")

        wrapped_error = ErrorHandler.wrap_exception(original_error)

        assert isinstance(wrapped_error, ValidationError)
        assert wrapped_error.context["field_path"] == "missing_field"


class TestErrorConverter:
    """测试错误转换器"""

    def test_normalize_tool_error_dict_with_error(self):
        """测试标准化包含错误的字典结果"""
        result = {
            "error": "API connection failed",
            "error_type": "ConnectionError",
            "error_code": "CONN_001",
        }

        error_msg, is_error, metadata = ErrorConverter.normalize_tool_error(
            result, "api_tool", "call_123"
        )

        assert is_error is True
        assert error_msg == "API connection failed"
        assert metadata["error_type"] == "ConnectionError"
        assert metadata["error_code"] == "CONN_001"

    def test_normalize_tool_error_exception(self):
        """测试标准化异常结果"""
        result = ValueError("Invalid parameter")

        error_msg, is_error, metadata = ErrorConverter.normalize_tool_error(result)

        assert is_error is True
        assert error_msg == "Invalid parameter"
        assert metadata["error_type"] == "ValueError"

    def test_normalize_tool_error_success_result(self):
        """测试标准化成功结果"""
        result = {"status": "success", "data": "result data"}

        error_msg, is_error, metadata = ErrorConverter.normalize_tool_error(result)

        assert is_error is False
        assert error_msg == "{'status': 'success', 'data': 'result data'}"
        assert metadata is None

    def test_create_tool_result_part(self):
        """测试创建工具结果部分"""
        result = {"error": "Tool failed", "error_code": "TOOL_001"}

        tool_result = ErrorConverter.create_tool_result_part(
            "call_123", result, "test_tool", auto_detect_error=True
        )

        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_call_id"] == "call_123"
        assert tool_result["is_error"] is True
        assert "provider_metadata" in tool_result
        assert "error_info" in tool_result["provider_metadata"]

    def test_convert_anthropic_error(self):
        """测试转换Anthropic错误格式"""
        error_data = {
            "tool_use_id": "call_123",
            "content": "Tool execution failed",
            "is_error": True,
        }

        tool_result = ErrorConverter.convert_provider_error_to_ir(
            error_data, "anthropic"
        )

        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_call_id"] == "call_123"
        assert tool_result["result"] == "Tool execution failed"
        assert tool_result["is_error"] is True

    def test_convert_ir_to_anthropic_error(self):
        """测试将IR错误转换为Anthropic格式"""
        tool_result = ToolResultPart(
            type="tool_result",
            tool_call_id="call_123",
            result="Tool failed",
            is_error=True,
        )

        anthropic_error = ErrorConverter.convert_ir_error_to_provider(
            tool_result, "anthropic"
        )

        assert anthropic_error["type"] == "tool_result"
        assert anthropic_error["tool_use_id"] == "call_123"
        assert anthropic_error["content"] == "Tool failed"
        assert anthropic_error["is_error"] is True

    def test_create_error_summary(self):
        """测试创建错误汇总"""
        errors = [
            ValidationError("Validation failed"),
            ToolResultPart(
                type="tool_result",
                tool_call_id="call_123",
                result="Tool failed",
                is_error=True,
            ),
        ]

        warnings = [create_warning("Feature ignored"), "Simple warning string"]

        summary = ErrorConverter.create_error_summary(errors, warnings)

        assert summary["total_errors"] == 2
        assert summary["total_warnings"] == 2
        assert summary["severity_counts"]["high"] == 1  # ValidationError
        assert summary["severity_counts"]["medium"] == 1  # ToolResult
        assert summary["severity_counts"]["low"] == 2  # 2 warnings
        assert "validation" in summary["category_counts"]
        assert "tool_execution" in summary["category_counts"]

    def test_format_error_for_display(self):
        """测试格式化错误显示"""
        error = ValidationError(
            "Field is required",
            field_path="ir_input[0].content",
            suggestions=["Add content field"],
        )

        formatted = ErrorConverter.format_error_for_display(error)

        assert "[ValidationError]" in formatted
        assert "Field is required" in formatted
        assert "field_path=ir_input[0].content" in formatted
        assert "Add content field" in formatted


class TestAnthropicConverterErrorHandling:
    """测试Anthropic转换器的错误处理"""

    def setup_method(self):
        """设置测试环境"""
        self.converter = AnthropicConverter()

    def test_validation_error_handling(self):
        """测试验证错误处理"""
        invalid_ir = [
            {"invalid": "data"}  # 既没有role也没有type
        ]

        with pytest.raises(ValidationError) as exc_info:
            self.converter.to_provider(invalid_ir)

        error = exc_info.value
        assert error.category == ErrorCategory.VALIDATION
        assert "validation failed" in str(error).lower()

    def test_content_conversion_error_handling(self):
        """测试内容转换错误处理"""
        ir_input: IRInput = [
            {
                "role": "user",
                "content": [
                    {"type": "text"}  # 缺少text字段
                ],
            }
        ]

        with pytest.raises(ConversionError) as exc_info:
            self.converter.to_provider(ir_input)

        error = exc_info.value
        assert error.category == ErrorCategory.CONVERSION
        assert "missing 'text' field" in str(error)

    def test_unsupported_content_type_error(self):
        """测试不支持的内容类型错误"""
        ir_input: IRInput = [
            {"role": "user", "content": [{"type": "unsupported_type", "data": "test"}]}
        ]

        with pytest.raises(ConversionError) as exc_info:
            self.converter.to_provider(ir_input)

        error = exc_info.value
        assert "Unsupported content part type" in str(error)
        assert "unsupported_type" in str(error)

    def test_warning_generation(self):
        """测试警告生成"""
        ir_input: IRInput = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {
                "type": "system_event",
                "event_type": "session_start",
                "timestamp": "2024-01-01T00:00:00Z",
            },
        ]

        result, warnings = self.converter.to_provider(ir_input)

        assert len(warnings) == 1
        warning = warnings[0]
        assert isinstance(warning, WarningInfo)
        assert warning.category == ErrorCategory.CONVERSION
        assert "System event ignored" in warning.message

    def test_tool_result_error_conversion(self):
        """测试工具结果错误转换"""
        ir_input: IRInput = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_123",
                        "result": "Tool execution failed",
                        "is_error": True,
                    }
                ],
            }
        ]

        result, warnings = self.converter.to_provider(ir_input)

        # 验证工具结果被正确转换
        msg = result["messages"][0]
        tool_result = msg["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "call_123"
        assert tool_result["content"] == "Tool execution failed"
        assert tool_result["is_error"] is True


if __name__ == "__main__":
    pytest.main([__file__])
