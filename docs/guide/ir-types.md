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
| `ToolResultPart` | Result of a tool execution |
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

## Helper Functions

```python
from llm_rosetta import extract_text_content, extract_tool_calls, create_tool_result_message

# Extract text from a message
text = extract_text_content(message)

# Extract tool calls from a message
tool_calls = extract_tool_calls(message)

# Create a tool result message
tool_msg = create_tool_result_message(tool_call_id, result)
```
