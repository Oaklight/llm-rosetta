---
title: API 标准
---

# API 标准

LLM-Rosetta 支持 4 家 LLM 提供商的 5 种 API 标准。每种标准定义了各自的请求/响应结构、认证机制和流式格式。本页介绍各标准并重点说明关键差异。

## 概览

| ProviderType | API 标准 | 端点 | 转换器类 |
|---|---|---|---|
| `openai_chat` | OpenAI Chat Completions | `POST /v1/chat/completions` | `OpenAIChatConverter` |
| `openai_responses` | OpenAI Responses | `POST /v1/responses` | `OpenAIResponsesConverter` |
| `open_responses` | Open Responses | `POST /v1/responses` | `OpenAIResponsesConverter` |
| `anthropic` | Anthropic Messages | `POST /v1/messages` | `AnthropicConverter` |
| `google` | Google GenAI | `POST /v1beta/models/{model}:generateContent` | `GoogleConverter` |

## OpenAI Chat Completions (`openai_chat`)

最广泛采用的 LLM API 标准。使用基于角色的消息数组，响应以 `choices[]` 数组返回。

**请求结构：**

```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_completion_tokens": 1000,
  "temperature": 0.7
}
```

**响应结构：**

```json
{
  "id": "chatcmpl-123",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Hi there!"},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
}
```

**关键特征：**

- 系统指令作为 `"role": "system"` 消息
- 工具调用参数为 **JSON 字符串**（非字典）
- 工具结果以独立的 `"role": "tool"` 消息传递
- **不支持** `top_k`
- 流式传输通过 `"stream": true` 启用，SSE `data: {...}` 分块

## OpenAI Responses (`openai_responses`)

OpenAI 较新的 API 格式（2025 年）。使用扁平的类型化项目列表代替嵌套消息，支持有状态的服务端对话。

**请求结构：**

```json
{
  "model": "gpt-4o",
  "instructions": "You are helpful.",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": [{"type": "input_text", "text": "Hello!"}]
    }
  ],
  "max_output_tokens": 1000
}
```

**响应结构：**

```json
{
  "id": "resp_123",
  "status": "completed",
  "output": [
    {
      "type": "message",
      "role": "assistant",
      "content": [{"type": "output_text", "text": "Hi there!"}]
    }
  ],
  "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}
}
```

**关键特征：**

- 系统指令通过顶层 `"instructions"` 字段
- 扁平项目列表：消息、`function_call` 和 `function_call_output` 同级排列
- 工具调用参数为 **JSON 字符串**
- 内容部分具有显式类型（`input_text`、`output_text`、`input_image` 等）
- 推理配置为嵌套对象：`"reasoning": {"type": "enabled", "effort": "high"}`
- 流式传输通过 SSE 类型化事件（`response.output_item.added`、`response.output_text.delta` 等）

## Open Responses (`open_responses`)

[Open Responses](https://www.openresponses.org/) 是一个开源的、厂商中立的规范（Apache 2.0），扩展了 OpenAI Responses API。由 OpenAI 于 2026 年 1 月发起，在保持完全向后兼容的同时增加了正式的可扩展性规则。

在 LLM-Rosetta 中，`open_responses` 是 `openai_responses` 的别名 — 同一个 `OpenAIResponsesConverter` 处理两种格式。

**与 OpenAI Responses 的区别：**

| 功能 | 说明 |
|-----|------|
| `OpenResponses-Version` 头部 | 规范版本控制机制 — 网关将此头部转发到上游 |
| Slug 前缀扩展 | `implementor:type_name` 格式的项目、工具和事件（如 `openai:web_search_call`） |
| Reasoning `content` 字段 | 开源模型的原始推理链 |
| `allowed_tools` 字段 | 缓存友好的工具限制 |
| 默认无状态 | 不假设服务端状态 |

**采用者：** OpenRouter、Hugging Face、Vercel、LM Studio、Ollama、vLLM。

## Anthropic Messages (`anthropic`)

Anthropic 为 Claude 模型设计的原生 API。以必需的 `max_tokens`、支持 `top_k` 和扩展思考为特色。

**请求结构：**

```json
{
  "model": "claude-sonnet-4-20250514",
  "system": "You are helpful.",
  "messages": [
    {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
  ],
  "max_tokens": 4096
}
```

**响应结构：**

```json
{
  "id": "msg_123",
  "type": "message",
  "role": "assistant",
  "content": [{"type": "text", "text": "Hi there!"}],
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 20, "output_tokens": 5}
}
```

**关键特征：**

- 系统指令通过顶层 `"system"` 字段（不在消息数组中）
- **`max_tokens` 必需** — LLM-Rosetta 未提供时默认使用 4096
- 温度限制在 0.0–1.0（OpenAI 允许到 2.0）
- **支持** `top_k`
- 工具调用为 `"tool_use"` 块；参数为**字典**（非 JSON 字符串）
- 工具结果为用户消息中的 `"tool_result"` 块
- 单一响应（无 `choices[]` / `candidates[]` 数组）
- 扩展思考：`"thinking": {"type": "enabled", "budget_tokens": 10000}`
- 认证通过 `x-api-key` 头部（非 `Authorization: Bearer`）

## Google GenAI (`google`)

Google 为 Gemini 模型设计的生成式 AI API。使用 `contents[]` 代替 `messages[]`，使用 `parts[]` 代替 `content`。

**请求结构：**

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [{"text": "Hello!"}]
    }
  ],
  "system_instruction": {"parts": [{"text": "You are helpful."}]},
  "generationConfig": {
    "maxOutputTokens": 1000,
    "temperature": 0.7
  }
}
```

**响应结构：**

```json
{
  "candidates": [
    {
      "content": {
        "role": "model",
        "parts": [{"text": "Hi there!"}]
      },
      "finishReason": "STOP"
    }
  ],
  "usageMetadata": {
    "promptTokenCount": 20,
    "candidatesTokenCount": 5
  }
}
```

**关键特征：**

- 系统指令通过顶层 `"system_instruction"` 字段
- 消息为 `"contents"`，内容块为 `"parts"`
- 助手角色为 `"model"`（非 `"assistant"`）
- 工具调用为 `functionCall` 部分；参数为**字典**
- 工具结果为用户回合中的 `functionResponse` 部分
- 响应格式通过独立字段：`responseMimeType` + `responseSchema`
- **支持** `top_k`
- REST API 使用 **camelCase**；Python SDK 使用 **snake_case** — 转换器透明处理两者
- 认证通过 `x-goog-api-key` 头部或查询参数
- 流式传输通过独立端点：`streamGenerateContent`

## 对比

| 特性 | OpenAI Chat | OpenAI Responses | Anthropic | Google GenAI |
|-----|:-----------:|:----------------:|:---------:|:------------:|
| 系统指令 | 消息角色 | `instructions` | `system` | `system_instruction` |
| 消息容器 | `messages` | `input` 项目 | `messages` | `contents` |
| 工具调用参数 | JSON 字符串 | JSON 字符串 | 字典 | 字典 |
| 工具结果传递 | `tool` 消息 | `function_call_output` 项目 | 用户消息中 `tool_result` | `functionResponse` 部分 |
| 响应包装 | `choices[]` | `output[]` | 单一消息 | `candidates[]` |
| `max_tokens` 字段 | `max_completion_tokens` | `max_output_tokens` | `max_tokens`（必需） | `maxOutputTokens` |
| `top_k` | 否 | 否 | 是 | 是 |
| 温度范围 | 0–2 | 0–2 | 0–1 | 0–2 |
| 流式传输 | `stream: true` | `stream: true` | `stream: true` | 独立端点 |
| 认证头部 | `Authorization: Bearer` | `Authorization: Bearer` | `x-api-key` | `x-goog-api-key` |

## 在 LLM-Rosetta 中

每种 API 标准都有对应的[转换器](converters.md)，负责在提供商格式和 [IR（中间表示）](concepts.md)之间转换。[网关](../gateway/configuration.md)通过提供商配置中的 `type` 字段选择正确的转换器：

```jsonc
"providers": {
  "my-openai":    { "type": "openai_chat",      "api_key": "...", "base_url": "..." },
  "my-anthropic": { "type": "anthropic",         "api_key": "...", "base_url": "..." },
  "my-google":    { "type": "google",            "api_key": "...", "base_url": "..." }
}
```

编程使用详见 [API 分层](api-layers.md)导入指南。
