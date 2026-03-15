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
