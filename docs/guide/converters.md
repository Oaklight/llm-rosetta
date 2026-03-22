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

## 提供商方言差异

不同 LLM 提供商对同一概念性操作有着细微不同的要求。LLM-Rosetta 在转换过程中自动处理这些**方言差异**，无需手动干预。

### 工具 Schema 清理

上游端点（尤其是 Vertex AI 的 OpenAI 兼容层）会拒绝某些在 JSON Schema 规范中有效但不被支持的关键字：

- `propertyNames`、`$schema`、`const`、`deprecated`、`readOnly` 等关键字
- `$ref` / `$defs` — 通过内联引用定义来解析
- `anyOf` / `oneOf` / `allOf` — 展平为简单类型 schema（例如可空联合类型）

LLM-Rosetta 的 `sanitize_schema()`（位于 `converters.base.tools`）会在所有 4 个转换器中递归地清除这些关键字。

### 孤立工具调用（OpenAI Chat 严格配对要求）

OpenAI Chat Completions API **严格要求**每个 assistant 消息中的 `tool_call_id` 都必须有对应的 `role: "tool"` 响应。如果工具调用被中断（例如用户在 Agent 编码工具中取消了执行），`tool_calls` 条目会保留在对话历史中但缺少匹配的结果，OpenAI 会返回 **400 错误**。

其他提供商（Anthropic、Google）对此比较宽松——它们会直接忽略孤立的工具调用。

LLM-Rosetta 自动处理此问题：

- **跨格式转换**：`OpenAIChatConverter.request_to_provider()` 会在输出消息上调用 `fix_orphaned_tool_calls()`，为未匹配的 `tool_call_id` 注入合成的 `role: "tool"` 占位消息。
- **直通 / 直接使用**：可以显式导入并调用该函数：

```python
from llm_rosetta.converters.openai_chat.tool_ops import fix_orphaned_tool_calls

# 在发送到 OpenAI 之前修补消息
messages = fix_orphaned_tool_calls(messages)
# 可以自定义占位文本：
messages = fix_orphaned_tool_calls(messages, placeholder="[已取消]")
```

### Google camelCase 与 snake_case

Google 的 REST API 使用 camelCase（`functionDeclarations`、`toolConfig`、`functionCallingConfig`），而 Python SDK 使用 snake_case（`function_declarations`、`tool_config`）。LLM-Rosetta 的 Google 转换器透明地接受两种命名约定。
