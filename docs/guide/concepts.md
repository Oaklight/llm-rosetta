---
title: Core Concepts
---

# Core Concepts

## The N² Problem

With N LLM providers, direct conversion between every pair requires N×(N-1) converters. For 4 providers, that's 12 converters to maintain.

## Hub-and-Spoke Solution

LLM-Rosetta introduces a central **Intermediate Representation (IR)** as the hub. Each provider only needs one converter to/from the IR, reducing the total to 2×N converters (8 for 4 providers).

```text
Provider A ←→ IR ←→ Provider B
Provider C ←→ IR ←→ Provider D
```

## Converter Architecture

Each converter (e.g., `OpenAIChatConverter`) is composed of four specialized operations classes:

| Component | Responsibility |
|-----------|---------------|
| `ContentOps` | Convert content parts (text, images, tool calls, etc.) |
| `MessageOps` | Convert complete messages (role + content) |
| `ToolOps` | Convert tool definitions and tool choice settings |
| `ConfigOps` | Convert generation parameters (temperature, max_tokens, etc.) |

These compose into the 6 main converter interfaces:

- `request_to_provider()` / `request_from_provider()`
- `response_to_provider()` / `response_from_provider()`
- `messages_to_provider()` / `messages_from_provider()`

Plus 2 streaming interfaces:

- `stream_response_from_provider()` / `stream_response_to_provider()`

## Conversion Context

All converter methods accept an optional `ConversionContext` (non-streaming) or `StreamContext` (streaming) that threads shared state through the pipeline:

- **`warnings`** — accumulated conversion notes (e.g., unsupported features dropped)
- **`options`** — structured conversion options (e.g., `output_format`, `metadata_mode`)
- **`metadata`** — opaque store for provider-specific state

The `metadata_mode` option (`"strip"` or `"preserve"`) controls whether provider-specific fields survive the round-trip. See [Using Converters — Metadata Preservation](converters.md#metadata-preservation-lossless-round-trip) for details.

`StreamContext` extends `ConversionContext` with session-level metadata, tool call tracking, and lifecycle flags for stateful stream transformations. See the [Streaming](streaming.md) guide for full details.

The base `stream_response_to_provider()` implementation uses a class-level dispatch table (`_TO_P_DISPATCH`) to route IR stream events to handler methods. Provider converters customize behavior through the `_post_process_to_provider()` hook rather than reimplementing the dispatch logic.

## Provider-Specific Data Preservation

LLM-Rosetta separates portable IR data, durable provider passthrough data, and per-conversion context state:

| Layer | Lifetime | Purpose |
|-------|----------|---------|
| Portable IR (`Message`, content parts, tools, reasoning) | Serializable and reusable | Semantics that can be translated between provider formats |
| `ProviderPassthroughEvent` / `ProviderPassthroughItem` | Serializable and reusable | Opaque provider-native chunks or items that have no portable representation |
| `ConversionContext` / `StreamContext` | One conversion pipeline | Warnings, options, echo fields, original IDs/status/annotations, and state used while reconstructing a response |

`ProviderPassthroughEvent` carries provider-native streaming chunks. `ProviderPassthroughItem` carries non-stream request/history items, while `IRResponse.provider_passthrough_items` stores independent non-stream output items together with their original positions.

Passthrough data is tagged with a converter dialect such as `openai_responses` or `anthropic`. A matching target dialect restores a copied native payload. A different target format drops the item; semantic items produce a conversion warning, while lifecycle/heartbeat stream events are dropped silently to avoid warning floods.

`ConversionContext` remains necessary because it serves a different role. It is an ephemeral side channel for a single request/response pipeline and is not serialized into conversation history. Provider passthrough carriers are durable IR data that can survive caching, persistence, and a later HTTP request.

## IR Message Types

The IR defines four message roles:

- **SystemMessage** — system instructions
- **UserMessage** — user input (text, images, files)
- **AssistantMessage** — model responses (text, tool calls, reasoning)
- **ToolMessage** — tool execution results

Each message contains a list of typed **content parts** (TextPart, ImagePart, ToolCallPart, etc.).
