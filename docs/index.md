---
title: Home
summary: LLMIR - Large Language Model Intermediate Representation
description: LLMIR is a Python library that provides unified intermediate representation for handling message formats from different LLM providers, supporting OpenAI, Anthropic, Google, and other major providers.
keywords: python, llm, large language model, intermediate representation, openai, anthropic, google
author: LLMIR Team
hide:
  - navigation
---

# LLMIR

[![PyPI version](https://badge.fury.io/py/llmir.svg?icon=si%3Apython)](https://badge.fury.io/py/llmir)
[![GitHub version](https://badge.fury.io/gh/Oaklight%2Fllmir.svg?icon=si%3Agithub)](https://badge.fury.io/gh/Oaklight%2Fllmir)

**Large Language Model Intermediate Representation** - Unified message format conversion for different LLM providers.

## 🚀 Quick Start

```bash
pip install llmir
```

```python
from llmir import auto_detect, convert_to_openai

# Auto-detect and convert message format
messages = [{"role": "user", "content": "Hello, world!"}]
provider = auto_detect(messages)
openai_format = convert_to_openai(messages, provider)
```

## 🎯 Core Features

- **🔄 Unified Conversion**: Support for OpenAI, Anthropic, Google, and other major LLM providers
- **🤖 Auto Detection**: Intelligent identification of message format sources
- **⚡ High Performance**: Optimized conversion algorithms with minimal performance overhead
- **🛡️ Type Safety**: Complete TypeScript-style type annotations
- **📚 Easy to Use**: Simple API design for quick adoption
- **🔧 Extensible**: Support for custom converters and formats

## 🛠️ Supported Providers

LLMIR currently supports message format conversion for the following LLM providers:

- **OpenAI** - Chat Completions and Responses API
- **Anthropic** - Claude message format
- **Google** - Gemini/PaLM message format
- **Universal Format** - Standardized intermediate representation

## 📖 Documentation Navigation

- **[Getting Started](getting-started/)** - Installation and basic usage
- **[User Guide](guide/)** - Detailed usage guide and best practices
- **[API Reference](api/)** - Complete API documentation
- **[Examples](examples/)** - Real-world use cases and code examples

## 🌟 Why Choose LLMIR?

- **🎯 Focused**: Specifically designed for LLM message format conversion
- **🔧 Flexible**: Support for multiple conversion modes and custom configurations
- **📈 Reliable**: Thoroughly tested and suitable for production environments
- **🌐 Open Source**: MIT license with community-driven development
- **📚 Well Documented**: Comprehensive documentation and examples

## 🤝 Get Involved

- **[GitHub Repository](https://github.com/Oaklight/llmir)** - Source code and issue tracking
- **[中文文档](../zh/)** - Chinese documentation
- **[Development Guide](development/)** - Contributing code and development guide

---

_LLMIR: Making LLM message format conversion simple and reliable._