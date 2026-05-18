---
title: Tool Calling
---

# Tool Calling

LLM-Rosetta provides a unified tool definition format that works across all providers.

## Defining Tools in IR Format

```python
from llm_rosetta import ToolDefinition

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name",
                    },
                },
                "required": ["location"],
            },
        },
    }
]
```

## Cross-Provider Tool Calling

```python
from llm_rosetta import OpenAIChatConverter, AnthropicConverter
from llm_rosetta.types.ir import extract_tool_calls, create_tool_result_message

openai_conv = OpenAIChatConverter()
anthropic_conv = AnthropicConverter()

# IR request with tools
ir_request = {
    "model": "gpt-4o",
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "What's the weather in Paris?"}]}
    ],
    "tools": tools,
    "tool_choice": "auto",
}

# Convert to OpenAI and call
openai_req, _ = openai_conv.request_to_provider(ir_request)
response = openai_client.chat.completions.create(**openai_req)
ir_response = openai_conv.response_from_provider(response.model_dump())

# Extract tool calls from IR response
tool_calls = extract_tool_calls(ir_response["choices"][0]["message"])

# Execute tools and create result messages
for tc in tool_calls:
    result = execute_tool(tc["function"]["name"], tc["function"]["arguments"])
    ir_messages.append(create_tool_result_message(tc["id"], result))

# Continue with Anthropic using the same tool results
ir_request["messages"] = ir_messages
ir_request["model"] = "claude-sonnet-4-20250514"
anthropic_req, _ = anthropic_conv.request_to_provider(ir_request)
```

The tool definitions and tool call results are automatically converted to each provider's native format.

## Multimodal Tool Results

Tools can return rich content (text + images + files) instead of plain strings. This is useful for tools that generate charts, diagrams, or other visual outputs.

```python
from llm_rosetta.types.ir import create_tool_result_message

# Tool function returning multimodal content
def generate_chart(chart_type="bar"):
    return [
        {"type": "text", "text": f"Generated {chart_type} chart:"},
        {"type": "image", "image_data": {"data": "<base64>", "media_type": "image/png"}},
    ]

# Execute tool and create multimodal result message
result = generate_chart(**tool_call["function"]["arguments"])
tool_msg = create_tool_result_message(tool_call["id"], result)
```

### Provider Support

| Provider | Multimodal Tool Results | Handling |
|----------|:----------------------:|----------|
| Anthropic | Native | Content blocks (text, image, document) |
| OpenAI Responses | Native | Content blocks (input_text, input_image, input_file) |
| Google Gemini | Native | inline_data blobs |
| OpenAI Chat | Emulated | Dual encoding: `json.dumps()` + synthetic user message with visual content |

For OpenAI Chat, the converter automatically handles the dual encoding — no special code needed from the caller.

## Custom Tool Calls (OpenAI Responses API)

The OpenAI Responses API supports a `"type": "custom"` tool variant, used by some extensions and integrations. llm-rosetta handles these end-to-end: ingestion, streaming, and cross-provider forwarding.

### Request

Define a custom tool by setting `"type": "custom"` in the tool object:

```json
{
    "type": "custom",
    "name": "my_custom_tool",
    "description": "A custom extension tool",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
}
```

In Python with the Responses API converter:

```python
from llm_rosetta import OpenAIResponsesConverter

conv = OpenAIResponsesConverter()

ir_request = {
    "model": "gpt-4o",
    "input": [
        {"role": "user", "content": [{"type": "input_text", "text": "Run my custom tool."}]}
    ],
    "tools": [
        {
            "type": "custom",
            "name": "my_custom_tool",
            "description": "A custom extension tool",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ],
}

responses_req, _ = conv.request_to_provider(ir_request)
```

### Response

When the model invokes a custom tool, the output contains a `custom_tool_call` item. Unlike `function_call`, the `input` field is **plain text**, not JSON:

```json
{
    "type": "custom_tool_call",
    "id": "ctc_abc123",
    "name": "my_custom_tool",
    "input": "Run query: find all active users"
}
```

llm-rosetta normalises this into the IR as a `type: "function"` tool call with a `_passthrough` marker, preserving enough information to reconstruct the original `custom_tool_call` format on the way back out.

### Streaming

Streaming custom tool call inputs uses two dedicated events that work identically to their `function_call` counterparts:

| Event | Description |
|-------|-------------|
| `response.custom_tool_call_input.delta` | Incremental chunk of the plain-text `input` field |
| `response.custom_tool_call_input.done` | Final assembled `input` value |

No extra handling is required — the streaming converter accumulates deltas and emits them through the same IR streaming interface as regular function calls.

### Cross-provider behavior

Anthropic and Google do not have a native `"custom"` tool type. When llm-rosetta forwards a request containing custom tools to either of these providers, it synthesizes a standard function tool with a single string parameter (`input`) so the tool remains callable:

```json
{
    "type": "function",
    "function": {
        "name": "my_custom_tool",
        "description": "A custom extension tool",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            },
            "required": ["input"]
        }
    }
}
```

On the return path, the synthesized function call result is converted back into a `custom_tool_call` output item before it reaches the original client — the round-trip is transparent to both the upstream provider and the downstream consumer.
