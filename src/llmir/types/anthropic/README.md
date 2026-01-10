# Anthropic Messages API Types

This directory contains TypedDict replicas of Anthropic SDK types for use in LLMIR's conversion layer.

## Overview

These types are **replicas** of the original Anthropic SDK types, converted from Pydantic `BaseModel` to `TypedDict` for better compatibility with LLMIR's type system and to avoid runtime dependencies on the Anthropic SDK.

## Structure

### Request Types (`request_types.py`)

Request-related types for the Anthropic Messages API:

- **Message Parameters**
  - `MessageParam` - Individual message in conversation
  - `TextBlockParam` - Text content block
  - `MessageCreateParams` - Main request parameters (union of streaming/non-streaming)
  - `MessageCreateParamsBase` - Base parameters shared by both modes
  - `MessageCreateParamsNonStreaming` - Non-streaming request
  - `MessageCreateParamsStreaming` - Streaming request

- **Tool Parameters**
  - `ToolParam` - Tool definition
  - `InputSchema` - Tool input schema (JSON Schema)
  - `ToolChoiceParam` - Tool choice strategy (union type)
  - `ToolChoiceAutoParam` - Auto tool selection
  - `ToolChoiceAnyParam` - Must use any tool
  - `ToolChoiceToolParam` - Must use specific tool
  - `ToolChoiceNoneParam` - Don't use tools

- **Configuration Parameters**
  - `ThinkingConfigParam` - Thinking/reasoning configuration
  - `ThinkingConfigEnabledParam` - Enabled thinking
  - `ThinkingConfigDisabledParam` - Disabled thinking
  - `CacheControlEphemeralParam` - Cache control
  - `MetadataParam` - Request metadata

### Response Types (`response_types.py`)

Response-related types from the Anthropic Messages API:

- **Main Response**
  - `Message` - Complete response message
  - `Usage` - Token usage statistics

- **Content Blocks**
  - `ContentBlock` - Union of all content block types
  - `TextBlock` - Text content
  - `ThinkingBlock` - Reasoning/thinking content
  - `ToolUseBlock` - Tool invocation

- **Supporting Types**
  - `StopReason` - Why the model stopped generating
  - `TextCitation` - Citation information
  - `CacheCreation` - Cache creation details
  - `ServerToolUsage` - Server tool usage stats

## Key Differences from Original SDK

1. **TypedDict vs BaseModel**: All types use `TypedDict` instead of Pydantic `BaseModel`
2. **Union Types**: Use Python's `|` operator or `Union[]` for type unions
3. **Optional Fields**: Use `total=False` in TypedDict for optional fields
4. **Required Fields**: Use `Required[]` annotation for required fields in `total=False` TypedDict

## Usage Example

```python
from llmir.types.anthropic import (
    MessageCreateParams,
    MessageParam,
    ToolParam,
    Message,
)

# Create request parameters
request: MessageCreateParams = {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
        {
            "role": "user",
            "content": "Hello, Claude!"
        }
    ]
}

# Type hint for response
response: Message = {
    "id": "msg_123",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "Hello! How can I help you?"
        }
    ],
    "model": "claude-3-5-sonnet-20241022",
    "usage": {
        "input_tokens": 10,
        "output_tokens": 20
    }
}
```

## Reference

These types are based on the Anthropic SDK types found in:
- `/data/pding/miniforge3/envs/llmir/lib/python3.10/site-packages/anthropic/types/`

For the original SDK documentation, see:
- [Anthropic API Documentation](https://docs.anthropic.com/en/api/messages)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)