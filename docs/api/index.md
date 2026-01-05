# API 参考

本节提供 LLMIR 的完整 API 文档。

## 主要模块

### 核心函数

```python
from llmir import auto_detect, convert_to_openai, convert_to_anthropic, convert_to_google
```

#### `auto_detect(messages)`

自动检测消息格式的提供商类型。

**参数:**
- `messages` (List[Dict]): 要检测的消息列表

**返回:**
- `str`: 检测到的提供商名称 ('openai', 'anthropic', 'google')

**示例:**
```python
provider = auto_detect([{"role": "user", "content": "Hello"}])
```

#### `convert_to_openai(messages, source_provider)`

将消息转换为 OpenAI 格式。

**参数:**
- `messages` (List[Dict]): 源消息列表
- `source_provider` (str): 源提供商名称

**返回:**
- `List[Dict]`: OpenAI 格式的消息列表

#### `convert_to_anthropic(messages, source_provider)`

将消息转换为 Anthropic 格式。

#### `convert_to_google(messages, source_provider)`

将消息转换为 Google 格式。

### 转换器类

所有转换器都继承自 `BaseConverter` 基类。

#### 基类方法

- `to_ir(messages)`: 转换为中间表示
- `from_ir(ir_messages)`: 从中间表示转换
- `validate(messages)`: 验证消息格式

### 类型定义

LLMIR 提供了完整的类型定义，位于 `llmir.types` 模块中。

## 详细文档

- [转换器 API](converters.md) - 转换器类的详细文档
- [类型定义](types.md) - 数据类型和接口定义
- [工具函数](utils.md) - 辅助工具和实用函数

## 使用示例

```python
from llmir import auto_detect, convert_to_openai
from llmir.converters import OpenAIChatConverter

# 基本用法
messages = [{"role": "user", "content": "Hello"}]
provider = auto_detect(messages)
openai_format = convert_to_openai(messages, provider)

# 高级用法
converter = OpenAIChatConverter()
ir_messages = converter.to_ir(messages)
result = converter.from_ir(ir_messages)