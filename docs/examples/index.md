# Examples

This section provides real-world usage examples of LLMIR to help you quickly understand and apply various features.

## Basic Examples

### Simple Message Conversion

```python
from llmir import convert_to_openai

# Anthropic format messages
anthropic_messages = [
    {"role": "user", "content": "Hello, Claude!"}
]

# Convert to OpenAI format
openai_messages = convert_to_openai(anthropic_messages, "anthropic")
print(openai_messages)
```

### Auto Detection and Conversion

```python
from llmir import auto_detect, convert_to_openai

# Unknown format messages
messages = [
    {"role": "user", "content": "Hello, world!"}
]

# Auto-detect and convert
provider = auto_detect(messages)
if provider:
    result = convert_to_openai(messages, provider)
    print(f"Detected {provider} format, conversion result: {result}")
```

## Advanced Examples

### [Basic Conversion](basic-conversion.md)

Learn how to perform basic message format conversion, including:
- Single message conversion
- Batch message conversion
- Error handling

### [Multi-turn Chat](multi-turn-chat.md)

Understand how to handle multi-turn conversation scenarios:
- Conversation history management
- Context preservation
- Role conversion

### [Tool Calling](tool-calling.md)

Master tool calling conversion techniques:
- Function call format conversion
- Tool result handling
- Complex tool chains

## Real-world Application Scenarios

### Chatbot Integration

```python
from llmir import auto_detect, convert_to_openai

class UniversalChatBot:
    def process_message(self, messages, target_provider="openai"):
        # Auto-detect input format
        source_provider = auto_detect(messages)
        
        # Convert to target format
        if target_provider == "openai":
            return convert_to_openai(messages, source_provider)
        # Add support for other providers...
```

### Data Migration Tool

```python
from llmir.converters import OpenAIChatConverter, AnthropicConverter

def migrate_conversations(openai_conversations):
    """Migrate OpenAI conversations to Anthropic format"""
    openai_converter = OpenAIChatConverter()
    anthropic_converter = AnthropicConverter()
    
    migrated = []
    for conversation in openai_conversations:
        # Convert to intermediate representation
        ir_messages = openai_converter.to_ir(conversation)
        # Convert to Anthropic format
        anthropic_messages = anthropic_converter.from_ir(ir_messages)
        migrated.append(anthropic_messages)
    
    return migrated
```

### Testing Tool

```python
from llmir import convert_to_openai, convert_to_anthropic

def test_message_compatibility(messages):
    """Test message compatibility across different providers"""
    results = {}
    
    try:
        # Test conversion to OpenAI format
        openai_result = convert_to_openai(messages, "anthropic")
        results["openai"] = {"success": True, "data": openai_result}
    except Exception as e:
        results["openai"] = {"success": False, "error": str(e)}
    
    try:
        # Test conversion to Anthropic format
        anthropic_result = convert_to_anthropic(messages, "openai")
        results["anthropic"] = {"success": True, "data": anthropic_result}
    except Exception as e:
        results["anthropic"] = {"success": False, "error": str(e)}
    
    return results
```

## Performance Optimization Examples

### Batch Processing

```python
from llmir.converters import OpenAIChatConverter

def batch_convert(message_batches):
    """Batch convert messages for improved performance"""
    converter = OpenAIChatConverter()  # Reuse converter instance
    
    results = []
    for batch in message_batches:
        ir_messages = converter.to_ir(batch)
        results.append(ir_messages)
    
    return results
```

## Next Steps

- [View specific example code](basic-conversion.md)
- [Learn about API documentation](../api/)
- [Read user guide](../guide/)