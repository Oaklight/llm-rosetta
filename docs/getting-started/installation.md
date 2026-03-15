---
title: Installation
---

# Installation

## Basic Installation

```bash
pip install llm-rosetta
```

The core library has minimal dependencies (`typing_extensions>=4.0.0`).

## Provider SDKs (Optional)

Install provider SDKs if you need to make direct API calls:

```bash
# OpenAI
pip install openai

# Anthropic
pip install anthropic

# Google GenAI
pip install google-genai

# All providers
pip install llm-rosetta[openai,anthropic,google]
```

!!! note

    Provider SDKs are only needed for making API calls. LLM-Rosetta's conversion functions work with plain dictionaries and don't require the SDKs.

## Development Installation

```bash
git clone https://github.com/Oaklight/llmir.git
cd llmir
pip install -e ".[dev]"
```
