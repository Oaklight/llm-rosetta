---
title: Quick Start
---

# Quick Start

## Basic Conversion

The core workflow: **Provider A → IR → Provider B**.

```python
from llmir import OpenAIChatConverter, AnthropicConverter

openai_conv = OpenAIChatConverter()
anthropic_conv = AnthropicConverter()

# An OpenAI Chat Completions request
openai_request = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    "temperature": 0.7,
    "max_tokens": 100,
}

# Convert: OpenAI → IR → Anthropic
ir_request = openai_conv.request_from_provider(openai_request)
anthropic_request, warnings = anthropic_conv.request_to_provider(ir_request)
```

## Converting Responses

```python
# After calling the Anthropic API
response = client.messages.create(**anthropic_request)

# Convert response to IR
ir_response = anthropic_conv.response_from_provider(response.model_dump())

# Extract text
from llmir import extract_text_content
text = extract_text_content(ir_response["choices"][0]["message"])
```

## Auto Detection

```python
from llmir import detect_provider, convert

# Detect provider from request structure
provider = detect_provider(some_request)

# One-step conversion
converted = convert(some_request, target_provider="anthropic")
```

## Next Steps

- [Core Concepts](../guide/concepts.md) — understand the architecture
- [Using Converters](../guide/converters.md) — detailed converter usage
- [IR Types](../guide/ir-types.md) — the Intermediate Representation
