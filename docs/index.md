---
title: Home
author: Oaklight
hide:
  - navigation
---

# LLMIR

[![PyPI version](https://badge.fury.io/py/llmir.svg?icon=si%3Apython)](https://badge.fury.io/py/llmir)
[![GitHub version](https://badge.fury.io/gh/oaklight%2Fllmir.svg?icon=si%3Agithub)](https://badge.fury.io/gh/oaklight%2Fllmir)

**Large Language Model Intermediate Representation** — a unified message format conversion library for LLM provider APIs.

## Overview

Different LLM providers (OpenAI, Anthropic, Google) use incompatible API formats. LLMIR solves this with a hub-and-spoke architecture: each provider converts to/from a central Intermediate Representation (IR), requiring only N converters instead of N².

## Quick Start

```bash
pip install llmir
```

```python
from llmir import OpenAIChatConverter, AnthropicConverter

openai_conv = OpenAIChatConverter()
anthropic_conv = AnthropicConverter()

# OpenAI format → IR → Anthropic format
ir_request = openai_conv.request_from_provider(openai_request)
anthropic_request, warnings = anthropic_conv.request_to_provider(ir_request)
```

## Supported Providers

| Provider | API | Converter |
|----------|-----|-----------|
| OpenAI | Chat Completions | `OpenAIChatConverter` |
| OpenAI | Responses | `OpenAIResponsesConverter` |
| Anthropic | Messages | `AnthropicConverter` |
| Google | GenAI | `GoogleGenAIConverter` |

## Key Features

- **Hub-and-Spoke Architecture** — central IR eliminates N² conversion problem
- **Bidirectional Conversion** — requests, responses, and messages in both directions
- **Streaming Support** — convert streaming chunks with stateful context management
- **Tool Calling** — unified tool definition and tool call handling across providers
- **Auto Detection** — automatically detect provider format from request structure
- **Type Safe** — full TypedDict annotations for all types
- **Zero Runtime Overhead** — pure dict transformations, no validation cost

## Architecture

```text
OpenAI Chat ──────┐
                   │
OpenAI Responses ──┤
                   ├──── IR (Intermediate Representation)
Anthropic ─────────┤
                   │
Google GenAI ──────┘
```

## Documentation

- **[Getting Started](getting-started/installation.md)** — Installation and first steps
- **[Guide](guide/concepts.md)** — Core concepts, converters, IR types, streaming
- **[Examples](examples/)** — Cross-provider conversations, tool calling
- **[API Reference](api/)** — Complete API documentation
- **[Changelog](changelog.md)** — Version history

## License

MIT License
