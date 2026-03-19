---
title: 使用转换器
---

# 使用转换器

## 创建转换器

```python
from llm_rosetta import OpenAIChatConverter, AnthropicConverter

converter = OpenAIChatConverter()
```

## 转换请求

### 提供商 → IR

```python
ir_request = converter.request_from_provider(provider_request)
```

### IR → 提供商

```python
provider_request, warnings = converter.request_to_provider(ir_request)
```

`warnings` 列表包含所有转换注意事项（如不支持的特性被丢弃）。

## 转换响应

```python
# 提供商响应 → IR
ir_response = converter.response_from_provider(provider_response_dict)

# IR → 提供商响应
provider_response = converter.response_to_provider(ir_response)
```

## 仅转换消息

当只需要消息转换而非完整的请求/响应时：

```python
ir_messages = converter.messages_from_provider(provider_messages)
provider_messages, warnings = converter.messages_to_provider(ir_messages)
```

## 跨提供商工作流

```python
from llm_rosetta import OpenAIChatConverter, GoogleGenAIConverter

openai_conv = OpenAIChatConverter()
google_conv = GoogleGenAIConverter()

# OpenAI → IR
ir_request = openai_conv.request_from_provider(openai_request)

# IR → Google
google_request, warnings = google_conv.request_to_provider(ir_request)

# 调用 Google API，获取响应
google_response = google_client.generate_content(**google_request)

# Google 响应 → IR
ir_response = google_conv.response_from_provider(google_response)
```

## Google：SDK 与 REST API 输出格式

默认情况下，Google 转换器生成的字典包含嵌套的 `config` 键，适用于 Google GenAI Python SDK（`google.genai`）。如果你直接调用 Google REST API（例如通过 `httpx` 或 `requests`），传入 `output_format="rest"` 可获得适合 HTTP 请求的扁平化请求体：

```python
from llm_rosetta import GoogleGenAIConverter

google_conv = GoogleGenAIConverter()

# SDK 格式（默认）— 适用于 google.genai SDK
sdk_request, warnings = google_conv.request_to_provider(ir_request)
# sdk_request: {"contents": [...], "config": {"tools": [...], "temperature": 0.7, ...}}

# REST 格式 — 适用于直接 HTTP 调用
rest_body, warnings = google_conv.request_to_provider(ir_request, output_format="rest")
# rest_body: {"contents": [...], "tools": [...], "generationConfig": {"temperature": 0.7, ...}}
```

`"rest"` 格式会将 `tools`、`tool_config`、`response_mime_type` 和 `response_schema` 提升到顶层，并将生成参数（temperature、top_p 等）包装到 `generationConfig` 对象中——与 [Google Gemini REST API](https://ai.google.dev/api/generate-content) 的请求格式完全一致。
