# Basic Usage

This page introduces the basic usage of LLMIR.

## Import Library

First, import the main modules from LLMIR:

```python
from llmir import auto_detect, convert_to_openai, convert_to_anthropic
from llmir.converters import OpenAIChatConverter, AnthropicConverter
```

## Auto Detection and Conversion

LLMIR provides automatic message format detection:

```python
# Example messages
messages = [
    {"role": "user", "content": "Hello, world!"}
]

# Auto-detect message format
provider = auto_detect(messages)
print(f"Detected provider: {provider}")

# Convert to OpenAI format
openai_messages = convert_to_openai(messages, provider)
print(openai_messages)
```

## Manual Conversion

You can also manually specify converters:

```python
from llmir.converters import AnthropicConverter

# Create converter instance
converter = AnthropicConverter()

# Anthropic format messages
anthropic_messages = [
    {"role": "user", "content": "Hello, Claude!"}
]

# Convert to intermediate representation
ir_messages = converter.to_ir(anthropic_messages)

# Convert to OpenAI format
openai_converter = OpenAIChatConverter()
openai_messages = openai_converter.from_ir(ir_messages)
```

## Handling Tool Calls

LLMIR supports tool call conversion:

```python
# Messages with tool calls
messages_with_tools = [
    {"role": "user", "content": "What's the weather like today?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "New York"}'
                }
            }
        ]
    }
]

# Convert tool call messages
converted = convert_to_anthropic(messages_with_tools, "openai")
```

## Error Handling

LLMIR provides detailed error information:

```python
try:
    result = convert_to_openai(invalid_messages, "unknown_provider")
except ValueError as e:
    print(f"Conversion error: {e}")
except Exception as e:
    print(f"Unknown error: {e}")
```

## Configuration Options

You can customize conversion behavior through configuration options:

```python
from llmir.converters import OpenAIChatConverter

# Create converter with configuration
converter = OpenAIChatConverter(
    strict_mode=True,  # Strict mode
    preserve_metadata=True  # Preserve metadata
)
```

## Next Steps

- [Learn about converter details](../guide/converters/)
- [View more examples](../examples/)
- [Read API documentation](../api/)