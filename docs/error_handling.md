# LLMIR 统一错误处理机制

本文档描述了 LLMIR 库的统一错误处理和转换机制，旨在提供一致、结构化的错误处理体验。

## 概述

LLMIR 的错误处理机制包含以下核心组件：

1. **统一异常类型** - 结构化的错误分类和严重程度
2. **错误转换工具** - 处理工具执行结果中的错误信息
3. **警告机制** - 非致命问题的通知系统
4. **上下文信息** - 丰富的错误上下文和解决建议

## 错误分类

### 错误类别 (ErrorCategory)

- `VALIDATION` - 输入验证错误
- `CONVERSION` - 转换过程错误
- `TOOL_EXECUTION` - 工具执行错误
- `PROVIDER_API` - Provider API 错误
- `CONFIGURATION` - 配置错误
- `NETWORK` - 网络错误
- `AUTHENTICATION` - 认证错误
- `RATE_LIMIT` - 速率限制错误
- `UNKNOWN` - 未知错误

### 严重程度 (ErrorSeverity)

- `LOW` - 轻微错误，可以继续处理
- `MEDIUM` - 中等错误，可能影响结果质量
- `HIGH` - 严重错误，需要用户注意
- `CRITICAL` - 致命错误，必须停止处理

## 核心异常类型

### LLMIRError (基础异常)

所有 LLMIR 异常的基类，提供统一的错误信息结构。

```python
from llmir.exceptions import LLMIRError, ErrorCategory, ErrorSeverity

error = LLMIRError(
    message="Something went wrong",
    category=ErrorCategory.VALIDATION,
    severity=ErrorSeverity.HIGH,
    context={"field": "user_input"},
    suggestions=["Check the input format"]
)

print(error)  # [VALIDATION] Something went wrong | Context: field=user_input | Suggestions: Check the input format
```

### ValidationError

用于输入验证失败的错误。

```python
from llmir.exceptions import create_validation_error

error = create_validation_error(
    "Invalid role value",
    field_path="ir_input[0].role",
    invalid_value="invalid_role",
    expected_type="'system', 'user', 'assistant', or 'developer'",
    suggestions=["Use one of the supported role values"]
)
```

### ConversionError

用于格式转换过程中的错误。

```python
from llmir.exceptions import create_conversion_error

error = create_conversion_error(
    "Failed to convert content",
    source_format="IR",
    target_format="Anthropic",
    item_index=0,
    suggestions=["Check content format"]
)
```

### ToolExecutionError

用于工具执行失败的错误。

```python
from llmir.exceptions import create_tool_error

error = create_tool_error(
    "Tool execution failed",
    tool_name="get_weather",
    tool_call_id="call_123",
    tool_input={"city": "Beijing"},
    suggestions=["Check API key", "Verify network connection"]
)
```

## 警告机制

### WarningInfo

用于非致命问题的通知。

```python
from llmir.exceptions import create_warning, ErrorCategory, ErrorSeverity

warning = create_warning(
    "Feature not supported",
    category=ErrorCategory.CONVERSION,
    severity=ErrorSeverity.MEDIUM,
    context={"feature": "system_event"},
    suggestions=["Use alternative approach"]
)
```

## 错误转换工具

### ErrorConverter

提供工具执行结果的错误处理和转换功能。

#### 标准化工具错误

```python
from llmir.utils.error_converter import ErrorConverter

# 自动检测和标准化错误
result = {"error": "API connection failed", "error_code": "CONN_001"}
error_msg, is_error, metadata = ErrorConverter.normalize_tool_error(
    result, "api_tool", "call_123"
)

print(f"Error: {is_error}, Message: {error_msg}")
# Error: True, Message: API connection failed
```

#### 创建工具结果部分

```python
# 创建带错误检测的工具结果
tool_result = ErrorConverter.create_tool_result_part(
    tool_call_id="call_123",
    result={"error": "Tool failed"},
    tool_name="test_tool",
    auto_detect_error=True
)

print(tool_result["is_error"])  # True
```

#### Provider 间错误转换

```python
# 将Anthropic错误转换为IR格式
anthropic_error = {
    "tool_use_id": "call_123",
    "content": "Tool execution failed",
    "is_error": True
}

ir_tool_result = ErrorConverter.convert_provider_error_to_ir(
    anthropic_error, "anthropic"
)

# 将IR错误转换为OpenAI格式
openai_error = ErrorConverter.convert_ir_error_to_provider(
    ir_tool_result, "openai"
)
```

## 转换器中的错误处理

### 基础转换器 (BaseConverter)

提供统一的验证和错误处理方法。

```python
from llmir.converters.base import BaseConverter

class MyConverter(BaseConverter):
    def to_provider(self, ir_input, tools=None, tool_choice=None):
        # 验证输入并在失败时抛出异常
        self.validate_and_raise(ir_input)

        try:
            # 转换逻辑
            result = self._convert(ir_input)
            return result, []
        except Exception as e:
            # 包装异常为ConversionError
            conversion_error = self.handle_conversion_error(
                e, "IR", "MyProvider", context={"input_length": len(ir_input)}
            )
            raise conversion_error
```

### Anthropic 转换器示例

```python
from llmir.converters.anthropic_converter import AnthropicConverter
from llmir.exceptions import ValidationError, ConversionError

converter = AnthropicConverter()

try:
    result, warnings = converter.to_provider(ir_input)

    # 处理警告
    for warning in warnings:
        if isinstance(warning, WarningInfo):
            print(f"Warning: {warning.message}")
        else:
            print(f"Warning: {warning}")

except ValidationError as e:
    print(f"Validation failed: {e}")
    print(f"Suggestions: {'; '.join(e.suggestions)}")

except ConversionError as e:
    print(f"Conversion failed: {e}")
    print(f"Context: {e.context}")
```

## 错误汇总和报告

### 创建错误汇总

```python
from llmir.utils.error_converter import ErrorConverter

errors = [ValidationError("Invalid input"), ToolExecutionError("Tool failed")]
warnings = [create_warning("Feature ignored"), "Simple warning"]

summary = ErrorConverter.create_error_summary(errors, warnings)

print(f"Total errors: {summary['total_errors']}")
print(f"Total warnings: {summary['total_warnings']}")
print(f"Severity distribution: {summary['severity_counts']}")
print(f"Category distribution: {summary['category_counts']}")
```

### 格式化错误显示

```python
# 格式化单个错误
formatted = ErrorConverter.format_error_for_display(
    error,
    include_context=True,
    include_suggestions=True
)
print(formatted)

# 输出示例:
# [ValidationError] Invalid field value | Context: field_path=ir_input[0].role, invalid_value=invalid_role | Suggestions: Use a valid role
```

## 最佳实践

### 1. 错误处理策略

```python
def process_ir_input(ir_input):
    try:
        # 验证输入
        converter.validate_and_raise(ir_input)

        # 执行转换
        result, warnings = converter.to_provider(ir_input)

        # 处理警告
        for warning in warnings:
            if isinstance(warning, WarningInfo) and warning.severity == ErrorSeverity.HIGH:
                logger.warning(f"High severity warning: {warning.message}")

        return result

    except ValidationError as e:
        # 记录验证错误并返回用户友好的消息
        logger.error(f"Input validation failed: {e}")
        raise ValueError(f"Invalid input: {e.message}")

    except ConversionError as e:
        # 记录转换错误并提供恢复建议
        logger.error(f"Conversion failed: {e}")
        if e.severity == ErrorSeverity.CRITICAL:
            raise
        else:
            # 尝试降级处理
            return fallback_conversion(ir_input)
```

### 2. 工具错误处理

```python
def execute_tool_with_error_handling(tool_call):
    try:
        result = execute_tool(tool_call)

        # 使用ErrorConverter标准化结果
        tool_result = ErrorConverter.create_tool_result_part(
            tool_call["tool_call_id"],
            result,
            tool_call["tool_name"],
            auto_detect_error=True
        )

        if tool_result["is_error"]:
            logger.warning(f"Tool execution failed: {tool_result['result']}")

        return tool_result

    except Exception as e:
        # 创建错误结果
        error_result = ErrorConverter.create_tool_result_part(
            tool_call["tool_call_id"],
            str(e),
            tool_call["tool_name"],
            auto_detect_error=False  # 明确标记为错误
        )
        error_result["is_error"] = True
        return error_result
```

### 3. 自定义错误处理

```python
from llmir.exceptions import LLMIRError, ErrorCategory, ErrorSeverity

class CustomAPIError(LLMIRError):
    """自定义API错误"""

    def __init__(self, message, api_endpoint=None, status_code=None, **kwargs):
        context = kwargs.get("context", {}) or {}
        if api_endpoint:
            context["api_endpoint"] = api_endpoint
        if status_code:
            context["status_code"] = status_code

        kwargs["context"] = context
        kwargs["category"] = ErrorCategory.PROVIDER_API
        kwargs["severity"] = ErrorSeverity.HIGH

        super().__init__(message, **kwargs)

# 使用自定义错误
try:
    response = api_client.call_endpoint("/api/v1/chat")
except requests.RequestException as e:
    raise CustomAPIError(
        "API call failed",
        api_endpoint="/api/v1/chat",
        status_code=getattr(e.response, 'status_code', None),
        original_error=e,
        suggestions=["Check API endpoint", "Verify authentication"]
    )
```

## 配置和扩展

### 错误处理配置

```python
# 可以通过环境变量或配置文件控制错误处理行为
import os

class ErrorConfig:
    # 是否在警告时记录详细信息
    LOG_WARNINGS = os.getenv("LLMIR_LOG_WARNINGS", "true").lower() == "true"

    # 是否在转换错误时尝试降级处理
    ENABLE_FALLBACK = os.getenv("LLMIR_ENABLE_FALLBACK", "false").lower() == "true"

    # 错误上下文的最大长度
    MAX_CONTEXT_LENGTH = int(os.getenv("LLMIR_MAX_CONTEXT_LENGTH", "1000"))
```

### 扩展错误类型

```python
# 添加新的错误类别
class ErrorCategory(Enum):
    # ... 现有类别
    CUSTOM_CATEGORY = "custom_category"

# 添加新的严重程度
class ErrorSeverity(Enum):
    # ... 现有严重程度
    CUSTOM_SEVERITY = "custom_severity"
```

## 总结

LLMIR 的统一错误处理机制提供了：

1. **结构化错误信息** - 包含类别、严重程度、上下文和建议
2. **一致的错误接口** - 所有组件使用相同的错误处理模式
3. **丰富的上下文信息** - 帮助快速定位和解决问题
4. **灵活的扩展性** - 支持自定义错误类型和处理逻辑
5. **工具错误转换** - 统一处理不同 provider 的工具执行错误
6. **警告机制** - 区分致命错误和非致命警告

通过使用这套错误处理机制，开发者可以：

- 快速识别和解决问题
- 获得详细的错误上下文信息
- 实现优雅的错误恢复
- 提供更好的用户体验
- 维护一致的错误处理标准

更多详细信息和示例，请参考源代码中的测试文件 `tests/test_error_handling.py`。
