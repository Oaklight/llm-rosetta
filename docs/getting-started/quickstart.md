---
title: 快速开始
---

# 快速开始

## 基本转换

核心工作流：**提供商 A → IR → 提供商 B**。

```python
from llm_rosetta import OpenAIChatConverter, AnthropicConverter

openai_conv = OpenAIChatConverter()
anthropic_conv = AnthropicConverter()

# OpenAI Chat Completions 请求
openai_request = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    "temperature": 0.7,
    "max_tokens": 100,
}

# 转换：OpenAI → IR → Anthropic
ir_request = openai_conv.request_from_provider(openai_request)
anthropic_request, warnings = anthropic_conv.request_to_provider(ir_request)
```

## 转换响应

```python
# 调用 Anthropic API 后
response = client.messages.create(**anthropic_request)

# 转换响应为 IR
ir_response = anthropic_conv.response_from_provider(response.model_dump())

# 提取文本
from llm_rosetta.types.ir import extract_text_content
text = extract_text_content(ir_response["choices"][0]["message"])
```

## 自动检测

```python
from llm_rosetta import detect_provider, convert

# 从请求结构自动检测提供商
provider = detect_provider(some_request)

# 一步完成转换
converted = convert(some_request, target_provider="anthropic")
```

## 下一步

- [核心概念](../guide/concepts.md) — 了解架构设计
- [使用转换器](../guide/converters.md) — 转换器详细用法
- [IR 类型](../guide/ir-types.md) — 中间表示类型系统
