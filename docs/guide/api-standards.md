---
title: API Standards
---

# API Standards

LLM-Rosetta supports 5 API standards across 4 LLM providers. Each standard defines its own request/response structure, authentication mechanism, and streaming format. This page describes each standard and highlights the key differences.

## Overview

| ProviderType | API Standard | Endpoint | Converter Class |
|---|---|---|---|
| `openai_chat` | OpenAI Chat Completions | `POST /v1/chat/completions` | `OpenAIChatConverter` |
| `openai_responses` | OpenAI Responses | `POST /v1/responses` | `OpenAIResponsesConverter` |
| `open_responses` | Open Responses | `POST /v1/responses` | `OpenAIResponsesConverter` |
| `anthropic` | Anthropic Messages | `POST /v1/messages` | `AnthropicConverter` |
| `google` | Google GenAI | `POST /v1beta/models/{model}:generateContent` | `GoogleConverter` |

## OpenAI Chat Completions (`openai_chat`)

The most widely adopted LLM API standard. Uses a role-based message array and returns responses in a `choices[]` array.

**Request shape:**

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

**Response shape:**

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

**Key characteristics:**

- System instructions as a message with `"role": "system"`
- Tool call arguments are **JSON strings** (not dicts)
- Tool results are separate messages with `"role": "tool"`
- `top_k` is **not supported**
- Streaming via `"stream": true` with SSE `data: {...}` chunks

## OpenAI Responses (`openai_responses`)

OpenAI's newer API format (2025). Uses a flat list of typed items instead of nested messages, and supports stateful server-side conversations.

**Request shape:**

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

**Response shape:**

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

**Key characteristics:**

- System instructions via top-level `"instructions"` field
- Flat item list: messages, `function_call`, and `function_call_output` are siblings
- Tool call arguments are **JSON strings**
- Content parts have explicit types (`input_text`, `output_text`, `input_image`, etc.)
- Reasoning config as nested object: `"reasoning": {"type": "enabled", "effort": "high"}`
- Streaming via SSE with typed events (`response.output_item.added`, `response.output_text.delta`, etc.)

## Open Responses (`open_responses`)

[Open Responses](https://www.openresponses.org/) is an open-source, vendor-neutral specification (Apache 2.0) that extends the OpenAI Responses API. Initiated by OpenAI in January 2026, it adds formal extensibility rules while maintaining full backward compatibility.

In LLM-Rosetta, `open_responses` is an alias for `openai_responses` — the same `OpenAIResponsesConverter` handles both formats.

**Differences from OpenAI Responses:**

| Feature | Description |
|---------|-------------|
| `OpenResponses-Version` header | Spec versioning mechanism — the gateway forwards this header to upstream |
| Slug-prefixed extensions | `implementor:type_name` items, tools, and events (e.g., `openai:web_search_call`) |
| Reasoning `content` field | Raw reasoning traces from open-weight models |
| `allowed_tools` field | Cache-preserving tool restriction |
| Stateless default | No server-side state assumption |

**Adopters:** OpenRouter, Hugging Face, Vercel, LM Studio, Ollama, vLLM.

## Anthropic Messages (`anthropic`)

Anthropic's native API for Claude models. Notable for requiring `max_tokens` and supporting `top_k` and extended thinking.

**Request shape:**

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

**Response shape:**

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

**Key characteristics:**

- System instructions via top-level `"system"` field (not in messages array)
- **`max_tokens` is required** — LLM-Rosetta defaults to 4096 if not provided
- Temperature is clamped to 0.0–1.0 (OpenAI allows up to 2.0)
- `top_k` is **supported**
- Tool calls are `"tool_use"` blocks; arguments are **dicts** (not JSON strings)
- Tool results are `"tool_result"` blocks inside a user message
- Single response (no `choices[]` / `candidates[]` array)
- Extended thinking: `"thinking": {"type": "enabled", "budget_tokens": 10000}`
- Auth via `x-api-key` header (not `Authorization: Bearer`)

## Google GenAI (`google`)

Google's Generative AI API for Gemini models. Uses `contents[]` instead of `messages[]` and `parts[]` instead of `content`.

**Request shape:**

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

**Response shape:**

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

**Key characteristics:**

- System instructions via top-level `"system_instruction"` field
- Messages are `"contents"`, content blocks are `"parts"`
- Assistant role is `"model"` (not `"assistant"`)
- Tool calls are `functionCall` parts; arguments are **dicts**
- Tool results are `functionResponse` parts in a user turn
- Response format via separate fields: `responseMimeType` + `responseSchema`
- `top_k` is **supported**
- REST API uses **camelCase**; Python SDK uses **snake_case** — the converter handles both transparently
- Auth via `x-goog-api-key` header or query parameter
- Streaming via separate endpoint: `streamGenerateContent`

## Comparison

| Feature | OpenAI Chat | OpenAI Responses | Anthropic | Google GenAI |
|---------|:-----------:|:----------------:|:---------:|:------------:|
| System instructions | Message role | `instructions` | `system` | `system_instruction` |
| Message container | `messages` | `input` items | `messages` | `contents` |
| Tool call args | JSON string | JSON string | Dict | Dict |
| Tool result delivery | `tool` message | `function_call_output` item | `tool_result` in user msg | `functionResponse` part |
| Response wrapper | `choices[]` | `output[]` | Single message | `candidates[]` |
| `max_tokens` field | `max_completion_tokens` | `max_output_tokens` | `max_tokens` (required) | `maxOutputTokens` |
| `top_k` | No | No | Yes | Yes |
| Temperature range | 0–2 | 0–2 | 0–1 | 0–2 |
| Streaming | `stream: true` | `stream: true` | `stream: true` | Separate endpoint |
| Auth header | `Authorization: Bearer` | `Authorization: Bearer` | `x-api-key` | `x-goog-api-key` |

## In LLM-Rosetta

Each API standard has a corresponding [converter](converters.md) that translates between the provider format and the [IR (Intermediate Representation)](concepts.md). The [gateway](../gateway/configuration.md) uses the `type` field in provider config to select the right converter:

```jsonc
"providers": {
  "my-openai":    { "type": "openai_chat",      "api_key": "...", "base_url": "..." },
  "my-anthropic": { "type": "anthropic",         "api_key": "...", "base_url": "..." },
  "my-google":    { "type": "google",            "api_key": "...", "base_url": "..." }
}
```

For programmatic use, see [API Layers](api-layers.md) for the import guide.
