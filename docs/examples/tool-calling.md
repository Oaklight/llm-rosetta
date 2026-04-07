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
