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

## IR Message Types

The IR defines four message roles:

- **SystemMessage** — system instructions
- **UserMessage** — user input (text, images, files)
- **AssistantMessage** — model responses (text, tool calls, reasoning)
- **ToolMessage** — tool execution results

Each message contains a list of typed **content parts** (TextPart, ImagePart, ToolCallPart, etc.).
