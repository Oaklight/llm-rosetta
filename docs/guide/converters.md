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

### 工具调用/结果配对（严格校验）

大多数 LLM 提供商**严格要求**工具调用和工具结果必须双向配对。任何方向的不匹配都会导致 **400 错误**：

| 方向 | OpenAI | Anthropic | Google |
|---|---|---|---|
| 有工具调用无结果 | **400** | **400** | OK |
| 有工具结果无调用 | **400** | **400** | OK |

代表性错误信息：

- **OpenAI**：`"An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'."` / `"Invalid parameter: messages with role 'tool' must be a response to a preceding message with 'tool_calls'."`
- **Anthropic**：`"tool_use ids were found without tool_result blocks immediately after: <id>"` / `"unexpected tool_use_id found in tool_result blocks: <id>"`
- **OpenAI Responses API**：`"No tool output found for function call <call_id>."`

只有 Google Gemini 对这两种情况都比较宽松。这些不匹配通常在以下场景中出现：

- 工具调用被中断（用户取消执行）——留下了没有结果的调用。
- 上下文压缩或内容过滤删除了包含 `tool_calls` 的 assistant 消息——留下了没有前置调用的工具结果。

LLM-Rosetta 自动修复两个方向的不匹配，并输出 `WARNING` 级别日志以便追踪每次修复：

- **孤立工具调用** → 注入包含占位内容 `"[No output available yet]"` 的合成工具结果。
- **孤立工具结果** → 移除悬空的结果消息。

**跨格式转换**：所有转换器都会在 IR 层级修复不匹配。

**直通 / 直接使用**：可以导入对应格式的函数：

```python
# OpenAI Chat Completions 格式
from llm_rosetta.converters.openai_chat.tool_ops import fix_orphaned_tool_calls

messages = fix_orphaned_tool_calls(messages)
messages = fix_orphaned_tool_calls(messages, placeholder="[已跳过]")

# OpenAI Responses 格式
from llm_rosetta.converters.openai_responses.tool_ops import fix_orphaned_tool_calls

items = fix_orphaned_tool_calls(items)
```

### Google camelCase 与 snake_case

Google 的 REST API 和 CLI 工具（如 Gemini CLI）使用 camelCase（`inlineData`、`mimeType`、`functionCall`、`functionResponse`、`functionDeclarations`、`responseMimeType`、`thinkingConfig` 等），而 Python SDK 使用 snake_case。LLM-Rosetta 的 Google 转换器在所有层级（内容、工具、配置和响应字段）透明地接受两种命名约定。所有 IR→Provider 输出使用 camelCase 以兼容 REST API。

有关所有 camelCase/snake_case 字段对及其他在实际测试中发现的真实兼容性问题的完整列表，请参阅[提供商与 CLI 兼容性矩阵](compatibility.md)。
