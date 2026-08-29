---
title: 库快速开始
---

# 库快速开始

## Hello World

最简用法：将一个 dict 转换为另一个 dict。无需 API key，无需 SDK，无需网络调用。

```python
from llm_rosetta import OpenAIChatConverter, AnthropicConverter

# 将 OpenAI 请求 dict 转换为 Anthropic 请求 dict
openai_request = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}],
}

ir = OpenAIChatConverter().request_from_provider(openai_request)
anthropic_request, warnings = AnthropicConverter().request_to_provider(ir)

print(anthropic_request)
# {'model': 'gpt-4o', 'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Hello!'}]}], ...}
```

就这么简单——LLM-Rosetta 只做数据结构转换，API 调用由你自己完成。

## 完整转换示例

包含系统消息、生成参数和多条消息的完整工作流：

```python
from llm_rosetta import OpenAIChatConverter, AnthropicConverter

openai_conv = OpenAIChatConverter()
anthropic_conv = AnthropicConverter()

openai_request = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    "temperature": 0.7,
    "max_tokens": 100,
}

# OpenAI → IR → Anthropic
ir_request = openai_conv.request_from_provider(openai_request)
anthropic_request, warnings = anthropic_conv.request_to_provider(ir_request)
```

## 转换响应

```python
# 用你自己的客户端调用 Anthropic API 后
response = client.messages.create(**anthropic_request)

# 将响应转换回 IR
ir_response = anthropic_conv.response_from_provider(response.model_dump())

# 从 IR 响应中提取文本
from llm_rosetta.types.ir import extract_text_content
text = extract_text_content(ir_response["choices"][0]["message"])
```

## 自动检测

不知道源格式？让 LLM-Rosetta 自动识别：

```python
from llm_rosetta import detect_provider, convert

# 从请求结构自动检测提供商
provider = detect_provider(some_request)

# 一步完成转换
converted = convert(some_request, target_provider="anthropic")
```

## 下一步

- [核心概念](../guide/concepts.md) — 了解中枢辐射架构
- [使用转换器](../guide/converters.md) — 转换器详细用法和元数据保留
- [流式处理](../guide/streaming.md) — 跨提供商转换流式数据块
- [IR 类型](../guide/ir-types.md) — 中间表示类型系统
