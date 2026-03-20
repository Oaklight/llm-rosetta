# LLM-Rosetta

[![PyPI version](https://img.shields.io/pypi/v/llm-rosetta?color=green)](https://pypi.org/project/llm-rosetta/)
[![GitHub release](https://img.shields.io/github/v/release/Oaklight/llm-rosetta)](https://github.com/Oaklight/llm-rosetta/releases/latest)
[![CI](https://github.com/Oaklight/llm-rosetta/actions/workflows/ci.yml/badge.svg)](https://github.com/Oaklight/llm-rosetta/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Oaklight/llm-rosetta)

[English Version](README_en.md) | [中文版](README_zh.md)

**LLM-Rosetta** — A Python library for converting between different LLM provider API formats using a hub-and-spoke architecture with a central IR (Intermediate Representation).

## Full Documentation

Full documentation is available at:

- **English**: [https://llm-rosetta.readthedocs.io/en/latest/](https://llm-rosetta.readthedocs.io/en/latest/)
- **中文**: [https://llm-rosetta.readthedocs.io/zh-cn/latest/](https://llm-rosetta.readthedocs.io/zh-cn/latest/)

## The Problem

When building applications that work with multiple LLM providers, you face an N² conversion problem — every provider pair requires its own conversion logic. LLM-Rosetta solves this with a hub-and-spoke approach: each provider only needs a single converter to/from the shared IR format.

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

### Ollama & Other OpenAI-Compatible Servers

LLM-Rosetta works out of the box with any server that exposes OpenAI-compatible endpoints. [Ollama](https://ollama.com/) (v0.13+) is a great example — it supports three of the four API formats that LLM-Rosetta converts between:

| Ollama Endpoint | LLM-Rosetta Converter | Since |
|---|---|---|
| `/v1/chat/completions` | `openai_chat` | Early versions |
| `/v1/responses` | `openai_responses` | v0.13.3 |
| `/v1/messages` | `anthropic` | v0.14.0 |

Other compatible servers include [HuggingFace TGI](https://github.com/huggingface/text-generation-inference), [vLLM](https://github.com/vllm-project/vllm), and [LM Studio](https://lmstudio.ai/).

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
pip install llm-rosetta
```

### Installing with Provider SDKs

```bash
# Individual providers
pip install llm-rosetta[openai]
pip install llm-rosetta[anthropic]
pip install llm-rosetta[google]

# All providers
pip install llm-rosetta[openai,anthropic,google]
```

### Optional Dependencies

| Extra | Packages | Description |
|-------|----------|-------------|
| `openai` | `openai` | OpenAI Chat Completions & Responses API |
| `anthropic` | `anthropic` | Anthropic Messages API |
| `google` | `google-genai` | Google GenAI API |

## Quick Start

```python
from llm_rosetta import OpenAIChatConverter, AnthropicConverter

# Create converters
openai_conv = OpenAIChatConverter()
anthropic_conv = AnthropicConverter()

# Convert an OpenAI response to IR, then to Anthropic format
ir_messages = openai_conv.response_from_provider(openai_response)
anthropic_request = anthropic_conv.request_to_provider(ir_messages)
```

### Auto-Detection

```python
from llm_rosetta import convert, detect_provider

# Automatically detect provider and convert
provider = detect_provider(some_response)
ir_messages = convert(some_response, direction="from_provider")
```

### Cross-Provider Conversation

```python
from llm_rosetta import OpenAIChatConverter, GoogleGenAIConverter
from llm_rosetta.types.ir import Message, ContentPart

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

LLM-Rosetta is part of the ToolRegistry ecosystem:

- **[ToolRegistry](https://github.com/Oaklight/ToolRegistry)** — Tool registration and management for LLM function calling
- **[ToolRegistry Hub](https://github.com/Oaklight/toolregistry-hub)** — Ready-to-use tool collection
- **[ToolRegistry Server](https://github.com/Oaklight/toolregistry-server)** — OpenAPI and MCP server adapters

## Contributing

Contributions are welcome! Please visit the [GitHub repository](https://github.com/Oaklight/llm-rosetta) to get started.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
