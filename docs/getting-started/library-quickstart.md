---
title: Library Quick Start
---

# Library Quick Start

## Hello World

The simplest possible use: convert one dict to another. No API keys, no SDK, no network calls.

```python
from llm_rosetta import OpenAIChatConverter, AnthropicConverter

# Convert an OpenAI request dict → Anthropic request dict
openai_request = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}],
}

ir = OpenAIChatConverter().request_from_provider(openai_request)
anthropic_request, warnings = AnthropicConverter().request_to_provider(ir)

print(anthropic_request)
# {'model': 'gpt-4o', 'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Hello!'}]}], ...}
```

That's it — LLM-Rosetta only transforms data structures. You call the APIs yourself.

## Full Conversion Example

A more complete workflow with system messages, generation config, and multiple messages:

```python
from llm_rosetta import OpenAIChatConverter, AnthropicConverter

openai_conv = OpenAIChatConverter()
anthropic_conv = AnthropicConverter()

openai_request = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    "temperature": 0.7,
    "max_tokens": 100,
}

# OpenAI → IR → Anthropic
ir_request = openai_conv.request_from_provider(openai_request)
anthropic_request, warnings = anthropic_conv.request_to_provider(ir_request)
```

## Converting Responses

```python
# After calling the Anthropic API with your own client
response = client.messages.create(**anthropic_request)

# Convert the response back to IR
ir_response = anthropic_conv.response_from_provider(response.model_dump())

# Extract text from the IR response
from llm_rosetta.types.ir import extract_text_content
text = extract_text_content(ir_response["choices"][0]["message"])
```

## Auto Detection

Don't know the source format? Let LLM-Rosetta figure it out:

```python
from llm_rosetta import detect_provider, convert

# Detect provider from request structure
provider = detect_provider(some_request)

# One-step conversion
converted = convert(some_request, target_provider="anthropic")
```

## Next Steps

- [Core Concepts](../guide/concepts.md) — understand the hub-and-spoke architecture
- [Using Converters](../guide/converters.md) — detailed converter usage and metadata preservation
- [Streaming](../guide/streaming.md) — convert streaming chunks between providers
- [IR Types](../guide/ir-types.md) — the Intermediate Representation type system
