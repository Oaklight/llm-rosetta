---
title: Tool Ops
---

# Tool Ops API

The `tool_ops` module provides a lightweight convenience API for converting
tool definitions between IR (Intermediate Representation) and provider-native
formats — without instantiating full converter pipelines.

## Quick Example

```python
from llm_rosetta import tool_ops

# Define an IR tool
ir_tool = {
    "type": "function",
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}

# Convert to any provider format
openai_tool = tool_ops.to_openai_chat(ir_tool)
anthropic_tool = tool_ops.to_anthropic(ir_tool)
google_tool = tool_ops.to_google_genai(ir_tool)

# Or use unified dispatch
tool = tool_ops.to_provider(ir_tool, provider="anthropic")

# Convert back to IR
recovered = tool_ops.from_anthropic(anthropic_tool)
recovered = tool_ops.from_provider(anthropic_tool, provider="anthropic")
```

## Supported Providers

| Canonical Name      | Aliases                              |
|---------------------|--------------------------------------|
| `openai_chat`       | `openai-chat`                        |
| `openai_responses`  | `openai-responses`, `open_responses`, `open-responses` |
| `anthropic`         | —                                    |
| `google`            | `google-genai`                       |

## Unified Dispatch

::: llm_rosetta.tool_ops.to_provider

::: llm_rosetta.tool_ops.from_provider

## Per-Provider Shortcuts

### IR to Provider

::: llm_rosetta.tool_ops.to_openai_chat

::: llm_rosetta.tool_ops.to_openai_responses

::: llm_rosetta.tool_ops.to_anthropic

::: llm_rosetta.tool_ops.to_google_genai

### Provider to IR

::: llm_rosetta.tool_ops.from_openai_chat

::: llm_rosetta.tool_ops.from_openai_responses

::: llm_rosetta.tool_ops.from_anthropic

::: llm_rosetta.tool_ops.from_google_genai
