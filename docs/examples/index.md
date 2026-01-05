# 示例

本节提供了 LLMIR 的实际使用示例，帮助您快速理解和应用各种功能。

## 基础示例

### 简单消息转换

```python
from llmir import convert_to_openai

# Anthropic 格式的消息
anthropic_messages = [
    {"role": "user", "content": "你好，Claude！"}
]

# 转换为 OpenAI 格式
openai_messages = convert_to_openai(anthropic_messages, "anthropic")
print(openai_messages)
```

### 自动检测和转换

```python
from llmir import auto_detect, convert_to_openai

# 未知格式的消息
messages = [
    {"role": "user", "content": "Hello, world!"}
]

# 自动检测并转换
provider = auto_detect(messages)
if provider:
    result = convert_to_openai(messages, provider)
    print(f"检测到 {provider} 格式，转换结果：{result}")
```

## 高级示例

### [基本转换](basic-conversion.md)

学习如何进行基本的消息格式转换，包括：
- 单条消息转换
- 批量消息转换
- 错误处理

### [多轮对话](multi-turn-chat.md)

了解如何处理多轮对话场景：
- 对话历史管理
- 上下文保持
- 角色转换

### [工具调用](tool-calling.md)

掌握工具调用的转换技巧：
- 函数调用格式转换
- 工具结果处理
- 复杂工具链

## 实际应用场景

### 聊天机器人集成

```python
from llmir import auto_detect, convert_to_openai

class UniversalChatBot:
    def process_message(self, messages, target_provider="openai"):
        # 自动检测输入格式
        source_provider = auto_detect(messages)
        
        # 转换为目标格式
        if target_provider == "openai":
            return convert_to_openai(messages, source_provider)
        # 添加其他提供商支持...
```

### 数据迁移工具

```python
from llmir.converters import OpenAIChatConverter, AnthropicConverter

def migrate_conversations(openai_conversations):
    """将 OpenAI 对话迁移到 Anthropic 格式"""
    openai_converter = OpenAIChatConverter()
    anthropic_converter = AnthropicConverter()
    
    migrated = []
    for conversation in openai_conversations:
        # 转换为中间表示
        ir_messages = openai_converter.to_ir(conversation)
        # 转换为 Anthropic 格式
        anthropic_messages = anthropic_converter.from_ir(ir_messages)
        migrated.append(anthropic_messages)
    
    return migrated
```

### 测试工具

```python
from llmir import convert_to_openai, convert_to_anthropic

def test_message_compatibility(messages):
    """测试消息在不同提供商间的兼容性"""
    results = {}
    
    try:
        # 测试转换为 OpenAI 格式
        openai_result = convert_to_openai(messages, "anthropic")
        results["openai"] = {"success": True, "data": openai_result}
    except Exception as e:
        results["openai"] = {"success": False, "error": str(e)}
    
    try:
        # 测试转换为 Anthropic 格式
        anthropic_result = convert_to_anthropic(messages, "openai")
        results["anthropic"] = {"success": True, "data": anthropic_result}
    except Exception as e:
        results["anthropic"] = {"success": False, "error": str(e)}
    
    return results
```

## 性能优化示例

### 批量处理

```python
from llmir.converters import OpenAIChatConverter

def batch_convert(message_batches):
    """批量转换消息以提高性能"""
    converter = OpenAIChatConverter()  # 重用转换器实例
    
    results = []
    for batch in message_batches:
        ir_messages = converter.to_ir(batch)
        results.append(ir_messages)
    
    return results
```

## 下一步

- [查看具体示例代码](basic-conversion.md)
- [了解 API 文档](../api/)
- [阅读用户指南](../guide/)