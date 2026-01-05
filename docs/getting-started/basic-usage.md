# 基本用法

本页面介绍 LLMIR 的基本使用方法。

## 导入库

首先导入 LLMIR 的主要模块：

```python
from llmir import auto_detect, convert_to_openai, convert_to_anthropic
from llmir.converters import OpenAIChatConverter, AnthropicConverter
```

## 自动检测和转换

LLMIR 提供了自动检测消息格式的功能：

```python
# 示例消息
messages = [
    {"role": "user", "content": "你好，世界！"}
]

# 自动检测消息格式
provider = auto_detect(messages)
print(f"检测到的提供商: {provider}")

# 转换为 OpenAI 格式
openai_messages = convert_to_openai(messages, provider)
print(openai_messages)
```

## 手动转换

您也可以手动指定转换器：

```python
from llmir.converters import AnthropicConverter

# 创建转换器实例
converter = AnthropicConverter()

# Anthropic 格式的消息
anthropic_messages = [
    {"role": "user", "content": "Hello, Claude!"}
]

# 转换为中间表示
ir_messages = converter.to_ir(anthropic_messages)

# 转换为 OpenAI 格式
openai_converter = OpenAIChatConverter()
openai_messages = openai_converter.from_ir(ir_messages)
```

## 处理工具调用

LLMIR 支持工具调用的转换：

```python
# 包含工具调用的消息
messages_with_tools = [
    {"role": "user", "content": "今天天气怎么样？"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "北京"}'
                }
            }
        ]
    }
]

# 转换工具调用消息
converted = convert_to_anthropic(messages_with_tools, "openai")
```

## 错误处理

LLMIR 提供了详细的错误信息：

```python
try:
    result = convert_to_openai(invalid_messages, "unknown_provider")
except ValueError as e:
    print(f"转换错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

## 配置选项

您可以通过配置选项自定义转换行为：

```python
from llmir.converters import OpenAIChatConverter

# 创建带配置的转换器
converter = OpenAIChatConverter(
    strict_mode=True,  # 严格模式
    preserve_metadata=True  # 保留元数据
)
```

## 下一步

- [了解转换器详情](../guide/converters/)
- [查看更多示例](../examples/)
- [阅读 API 文档](../api/)