# LLMIR

[![PyPI version](https://badge.fury.io/py/llmir.svg)](https://badge.fury.io/py/llmir)
[![GitHub version](https://badge.fury.io/gh/oaklight%2Fllmir.svg)](https://badge.fury.io/gh/oaklight%2Fllmir)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Oaklight/llmir)

[English Version](README_en.md) | [中文版](README_zh.md)

**Large Language Model Intermediate Representation** — A Python library for converting between different LLM provider API formats using a hub-and-spoke architecture with a central IR (Intermediate Representation).

## Full Documentation

Full documentation is available at:

- **English**: [https://llmir.readthedocs.io/en/latest/](https://llmir.readthedocs.io/en/latest/)
- **中文**: [https://llmir.readthedocs.io/zh-cn/latest/](https://llmir.readthedocs.io/zh-cn/latest/)

## The Problem

When building applications that work with multiple LLM providers, you face an N² conversion problem — every provider pair requires its own conversion logic. LLMIR solves this with a hub-and-spoke approach: each provider only needs a single converter to/from the shared IR format.

```
Provider A ──→ IR ──→ Provider B
Provider C ──→ IR ──→ Provider D
         ... and so on
```

## Supported Providers

| Provider | API Standard | Request | Response | Streaming |
|----------|-------------|:-------:|:--------:|:---------:|
| OpenAI | Chat Completions | ✅ | ✅ | ✅ |
| OpenAI | Responses API | ✅ | ✅ | ✅ |
| Anthropic | Messages API | ✅ | ✅ | ✅ |
| Google | GenAI API | ✅ | ✅ | ✅ |

## Features

- Unified IR format for messages, tool calls, and content parts
- Bidirectional conversion: requests to provider format, responses from provider format
- Streaming support with typed stream events
- Auto-detection of provider from request/response objects
- Support for text, images, tool calls, and tool results
- Zero required dependencies (only `typing_extensions`); provider SDKs are optional

## Installation

### Basic Installation

Install the core package (requires **Python >= 3.8**):

```bash
pip install llmir
```

### Installing with Provider SDKs

```bash
# Individual providers
pip install llmir[openai]
pip install llmir[anthropic]
pip install llmir[google]

# All providers
pip install llmir[openai,anthropic,google]
```

### Optional Dependencies

| Extra | Packages | Description |
|-------|----------|-------------|
| `openai` | `openai` | OpenAI Chat Completions & Responses API |
| `anthropic` | `anthropic` | Anthropic Messages API |
| `google` | `google-genai` | Google GenAI API |

## Quick Start

```python
from llmir import OpenAIChatConverter, AnthropicConverter

# Create converters
openai_conv = OpenAIChatConverter()
anthropic_conv = AnthropicConverter()

# Convert an OpenAI response to IR, then to Anthropic format
ir_messages = openai_conv.response_from_provider(openai_response)
anthropic_request = anthropic_conv.request_to_provider(ir_messages)
```

### Auto-Detection

```python
from llmir import convert, detect_provider

# Automatically detect provider and convert
provider = detect_provider(some_response)
ir_messages = convert(some_response, direction="from_provider")
```

### Cross-Provider Conversation

```python
from llmir import OpenAIChatConverter, GoogleGenAIConverter
from llmir.types.ir import Message, ContentPart

# Shared IR message history
ir_messages = []

# Turn 1: Ask OpenAI
ir_messages.append(Message(role="user", content=[ContentPart(type="text", text="Hello!")]))
openai_request = openai_conv.request_to_provider({"messages": ir_messages})
openai_response = openai_client.chat.completions.create(**openai_request)
ir_messages.extend(openai_conv.response_from_provider(openai_response))

# Turn 2: Continue with Google — full context preserved
google_request = google_conv.request_to_provider({"messages": ir_messages})
```

## Related Projects

LLMIR is part of the ToolRegistry ecosystem:

- **[ToolRegistry](https://github.com/Oaklight/ToolRegistry)** — Tool registration and management for LLM function calling
- **[ToolRegistry Hub](https://github.com/Oaklight/toolregistry-hub)** — Ready-to-use tool collection
- **[ToolRegistry Server](https://github.com/Oaklight/toolregistry-server)** — OpenAPI and MCP server adapters

## Contributing

Contributions are welcome! Please visit the [GitHub repository](https://github.com/Oaklight/llmir) to get started.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
