# API Reference

This section provides complete API documentation for LLMIR.

## Main Modules

### Core Functions

```python
from llmir import auto_detect, convert_to_openai, convert_to_anthropic, convert_to_google
```

#### `auto_detect(messages)`

Automatically detect the provider type of message format.

**Parameters:**
- `messages` (List[Dict]): List of messages to detect

**Returns:**
- `str`: Detected provider name ('openai', 'anthropic', 'google')

**Example:**
```python
provider = auto_detect([{"role": "user", "content": "Hello"}])
```

#### `convert_to_openai(messages, source_provider)`

Convert messages to OpenAI format.

**Parameters:**
- `messages` (List[Dict]): Source message list
- `source_provider` (str): Source provider name

**Returns:**
- `List[Dict]`: OpenAI format message list

#### `convert_to_anthropic(messages, source_provider)`

Convert messages to Anthropic format.

#### `convert_to_google(messages, source_provider)`

Convert messages to Google format.

### Converter Classes

All converters inherit from the `BaseConverter` base class.

#### Base Class Methods

- `to_ir(messages)`: Convert to intermediate representation
- `from_ir(ir_messages)`: Convert from intermediate representation
- `validate(messages)`: Validate message format

### Type Definitions

LLMIR provides complete type definitions located in the `llmir.types` module.

## Detailed Documentation

- [Converter API](converters.md) - Detailed converter class documentation
- [Type Definitions](types.md) - Data types and interface definitions
- [Utilities](utils.md) - Helper tools and utility functions

## Usage Examples

```python
from llmir import auto_detect, convert_to_openai
from llmir.converters import OpenAIChatConverter

# Basic usage
messages = [{"role": "user", "content": "Hello"}]
provider = auto_detect(messages)
openai_format = convert_to_openai(messages, provider)

# Advanced usage
converter = OpenAIChatConverter()
ir_messages = converter.to_ir(messages)
result = converter.from_ir(ir_messages)