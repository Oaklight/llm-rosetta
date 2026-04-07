---
title: IR Types
---

# IR Types

The Intermediate Representation uses TypedDict-based types for zero-overhead type safety.

## Messages

```python
from llm_rosetta import SystemMessage, UserMessage, AssistantMessage, ToolMessage
```

| Type | Role | Typical Content |
|------|------|----------------|
| `SystemMessage` | `"system"` | TextPart |
| `UserMessage` | `"user"` | TextPart, ImagePart, FilePart |
| `AssistantMessage` | `"assistant"` | TextPart, ToolCallPart, ReasoningPart |
| `ToolMessage` | `"tool"` | ToolResultPart |

## Content Parts

| Part | Description |
|------|-------------|
| `TextPart` | Plain text content |
| `ImagePart` | Image (URL or base64) |
| `FilePart` | File attachment |
| `ToolCallPart` | Function call from the model |
| `ToolResultPart` | Result of a tool execution (string or multimodal content) |
| `ReasoningPart` | Model's chain-of-thought |
| `RefusalPart` | Model's refusal to respond |
| `CitationPart` | Source citations |
| `AudioPart` | Audio content |

## IRRequest

```python
ir_request: IRRequest = {
    "model": "gpt-4o",
    "messages": [...],          # list of Message
    "tools": [...],             # optional list of ToolDefinition
    "tool_choice": "auto",      # optional ToolChoice
    "generation": {             # optional GenerationConfig
        "temperature": 0.7,
        "max_tokens": 1000,
    },
}
```

## IRResponse

```python
ir_response: IRResponse = {
    "id": "...",
    "model": "gpt-4o",
    "choices": [
        {
            "message": {...},       # AssistantMessage
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    },
}
```

## Tool Types

```python
from llm_rosetta import ToolDefinition, ToolChoice

tool: ToolDefinition = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
            "required": ["location"],
        },
    },
}
```

## Multimodal Tool Results

`ToolResultPart.result` can be a string (text-only) or a list of content parts (multimodal):

```python
# Text-only tool result
tool_msg = create_tool_result_message(call_id, '{"temperature": "72°F"}')

# Multimodal tool result (text + image)
tool_msg = create_tool_result_message(call_id, [
    {"type": "text", "text": "Generated chart:"},
    {"type": "image", "image_data": {"data": "<base64>", "media_type": "image/png"}},
])
```

Three providers (Anthropic, OpenAI Responses, Google GenAI) support multimodal tool results natively. OpenAI Chat uses a dual encoding strategy — `json.dumps()` in the tool message plus a synthetic user message with visual content — enabling lossless round-trip conversion.

## Helper Functions

```python
from llm_rosetta import (
    extract_text_content,
    extract_all_text,
    extract_tool_calls,
    create_tool_result_message,
)

# Extract text from TextPart content only
text = extract_text_content(message)

# Extract text from both TextPart and ReasoningPart
# (useful for thinking models like gemini-2.5-flash)
text = extract_all_text(message)

# Extract tool calls from a message
tool_calls = extract_tool_calls(message)

# Create a tool result message (string or multimodal list)
tool_msg = create_tool_result_message(tool_call_id, result)
```
