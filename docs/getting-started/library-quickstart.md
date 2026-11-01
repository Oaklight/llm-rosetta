---
title: 库快速开始
---

# 库快速开始

LLM-Rosetta 在四种控制粒度上提供同一套翻译。本页从原始转换器开始；管线（`convert()` / `ConversionPipeline`）增加 shim 解析和转换告警，网关再往上增加 HTTP 代理。完整分层见[四种控制粒度](../index.md#four-levels)。

## Hello World

最简用法：一个 dict 进去，另一个 dict 出来。不需要 API key，不需要 SDK，不走网络。

```python
from llm_rosetta import ChatCompletionsConverter, MessagesConverter

# 将 OpenAI 请求 dict 转换为 Anthropic 请求 dict
openai_request = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}],
}

ir = ChatCompletionsConverter().request_from_provider(openai_request)
anthropic_request, warnings = MessagesConverter().request_to_provider(ir)

print(anthropic_request)
# {'model': 'gpt-4o', 'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Hello!'}]}], ...}
```

就这么简单——LLM-Rosetta 只做数据结构转换，API 调用由你自己完成。

## 完整转换示例

带系统消息、生成参数和多轮对话的完整例子：

```python
from llm_rosetta import ChatCompletionsConverter, MessagesConverter

openai_conv = ChatCompletionsConverter()
anthropic_conv = MessagesConverter()

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

## 用 IR 作为 payload

IR 是类型化的、提供方中立的。你可以把对话历史、路由状态或评测数据保存在 IR 里，只在调用边界把它渲染成目标提供方的方言。换提供方时改变的是渲染器，不是应用的数据模型。

## 自动检测

不知道源格式？让 LLM-Rosetta 自动识别：

```python
from llm_rosetta import detect_provider, convert

# 从请求结构自动检测提供方
provider = detect_provider(some_request)

# 一步完成转换
converted = convert(some_request, target_provider="messages")
```

## 下一步

- [核心概念](../guide/concepts.md) — 了解中枢辐射架构
- [使用转换器](../guide/converters.md) — 转换器详细用法和元数据保留
- [流式处理](../guide/streaming.md) — 跨提供方转换流式数据块
- [IR 类型](../guide/ir-types.md) — 中间表示类型系统
- [四种控制粒度](../index.md#four-levels) — 函数、原始转换器、管线、网关
- [用量](../api/observability.md#usage) — 无需网关即可计量 token 并读取报告
