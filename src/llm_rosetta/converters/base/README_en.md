# Base Converter Module — Bottom-Up Ops Pattern

[中文版](./README_zh.md) | [English Version](./README_en.md)

## Overview

The Base Converter Module provides abstract base classes for all provider converters, organized by **functional domains** using the **Bottom-Up Ops Pattern**. Each domain handles a specific aspect of the conversion pipeline, from low-level content parts up to full request/response objects.

All 4 provider converters (OpenAI Chat, OpenAI Responses, Anthropic, Google GenAI) have been refactored to follow this pattern.

## File Structure

```
src/llm_rosetta/converters/base/
├── __init__.py      # Module exports
├── content.py       # BaseContentOps — content part conversion (text, image, file, audio, reasoning, refusal, citation)
├── tools.py         # BaseToolOps — tool conversion (definition, choice, call, result, config)
├── messages.py      # BaseMessageOps — message conversion (batch/single messages, validation)
├── configs.py       # BaseConfigOps — configuration conversion (generation, response_format, stream, reasoning, cache)
└── converter.py     # BaseConverter — top-level converter (composes ops classes via class attributes)
```

## Architecture: Bottom-Up Ops Pattern

The conversion pipeline is organized in layers, where higher layers call into lower layers:

```
L3  Converter       — request_to_provider / request_from_provider / response_*
 │
 ├── L2  ConfigOps   — generation, response_format, stream, reasoning, cache
 ├── L1  MessageOps  — batch message conversion (calls ContentOps + ToolOps)
 ├── L0.5 ToolOps    — tool definition, choice, call, result, config
 └── L0  ContentOps  — text, image, file, audio, reasoning, refusal, citation
```

**Key principles:**

- **Functional domain organization**: Each ops class handles one domain (content, tools, messages, configs)
- **Clear conversion hierarchy**: content → messages → requests/responses
- **Composition pattern**: Subclasses specify ops classes via class attributes, not inheritance
- **Bidirectional conversion**: Every concept has `ir_*_to_p()` and `p_*_to_ir()` methods

## Module Description

### `content.py` — BaseContentOps

Handles conversion of individual content parts. All methods are `@staticmethod @abstractmethod`.

| Content Type | IR → Provider | Provider → IR |
|-------------|---------------|---------------|
| Text | `ir_text_to_p(ir_text: TextPart) → Any` | `p_text_to_ir(provider_text) → TextPart` |
| Image | `ir_image_to_p(ir_image: ImagePart) → Any` | `p_image_to_ir(provider_image) → ImagePart` |
| File | `ir_file_to_p(ir_file: FilePart) → Any` | `p_file_to_ir(provider_file) → FilePart` |
| Audio | `ir_audio_to_p(ir_audio: AudioPart) → Any` | `p_audio_to_ir(provider_audio) → AudioPart` |
| Reasoning | `ir_reasoning_to_p(ir_reasoning: ReasoningPart) → Any` | `p_reasoning_to_ir(provider_reasoning) → ReasoningPart` |
| Refusal | `ir_refusal_to_p(ir_refusal: RefusalPart) → Any` | `p_refusal_to_ir(provider_refusal) → RefusalPart` |
| Citation | `ir_citation_to_p(ir_citation: CitationPart) → Any` | `p_citation_to_ir(provider_citation) → CitationPart` |

### `tools.py` — BaseToolOps

Handles the full tool lifecycle: definition → choice → call → result → config. All methods are `@staticmethod @abstractmethod`.

| Tool Concept | IR → Provider | Provider → IR |
|-------------|---------------|---------------|
| Definition | `ir_tool_definition_to_p(ir_tool: ToolDefinition) → Any` | `p_tool_definition_to_ir(provider_tool) → ToolDefinition` |
| Choice | `ir_tool_choice_to_p(ir_tool_choice: ToolChoice) → Any` | `p_tool_choice_to_ir(provider_tool_choice) → ToolChoice` |
| Call | `ir_tool_call_to_p(ir_tool_call: ToolCallPart) → Any` | `p_tool_call_to_ir(provider_tool_call) → ToolCallPart` |
| Result | `ir_tool_result_to_p(ir_tool_result: ToolResultPart) → Any` | `p_tool_result_to_ir(provider_tool_result) → ToolResultPart` |
| Config | `ir_tool_config_to_p(ir_tool_config: ToolCallConfig) → Any` | `p_tool_config_to_ir(provider_tool_config) → ToolCallConfig` |

### `messages.py` — BaseMessageOps

Handles message-level conversion. Serves as a bridge between content/tool layers and the request/response layer.

| Method | Signature | Description |
|--------|-----------|-------------|
| `ir_messages_to_p` | `(ir_messages: Iterable[Message \| ExtensionItem]) → Tuple[List[Any], List[str]]` | Batch convert IR messages to provider format (abstract) |
| `p_messages_to_ir` | `(provider_messages: List[Any]) → List[Message \| ExtensionItem]` | Batch convert provider messages to IR format (abstract) |
| `ir_message_to_p` | `(ir_message) → Tuple[Any, List[str]]` | Single message convenience method (concrete) |
| `p_message_to_ir` | `(provider_message) → Message \| ExtensionItem` | Single message convenience method (concrete) |
| `validate_messages` | `(messages) → List[str]` | Validate message list (concrete, overridable) |

### `configs.py` — BaseConfigOps

Handles conversion of all configuration parameters. All methods are `@staticmethod @abstractmethod`.

| Config Type | IR → Provider | Provider → IR |
|------------|---------------|---------------|
| Generation | `ir_generation_config_to_p(ir_config: GenerationConfig) → Any` | `p_generation_config_to_ir(provider_config) → GenerationConfig` |
| Response Format | `ir_response_format_to_p(ir_format: ResponseFormatConfig) → Any` | `p_response_format_to_ir(provider_format) → ResponseFormatConfig` |
| Stream | `ir_stream_config_to_p(ir_stream: StreamConfig) → Any` | `p_stream_config_to_ir(provider_stream) → StreamConfig` |
| Reasoning | `ir_reasoning_config_to_p(ir_reasoning: ReasoningConfig) → Any` | `p_reasoning_config_to_ir(provider_reasoning) → ReasoningConfig` |
| Cache | `ir_cache_config_to_p(ir_cache: CacheConfig) → Any` | `p_cache_config_to_ir(provider_cache) → CacheConfig` |

### `converter.py` — BaseConverter

Top-level converter that composes all ops classes. Defines 6 abstract methods + 2 convenience methods.

**Class attributes** (set by subclasses):

```python
content_ops_class: Optional[Type] = None   # → BaseContentOps subclass
tool_ops_class: Optional[Type] = None      # → BaseToolOps subclass
message_ops_class: Optional[Type] = None   # → BaseMessageOps subclass
config_ops_class: Optional[Type] = None    # → BaseConfigOps subclass
```

**Abstract methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `request_to_provider` | `(ir_request: IRRequest) → Tuple[Dict, List[str]]` | IR request → provider request |
| `request_from_provider` | `(provider_request: Dict) → IRRequest` | Provider request → IR request |
| `response_from_provider` | `(provider_response: Dict) → IRResponse` | Provider response → IR response |
| `response_to_provider` | `(ir_response: IRResponse) → Dict` | IR response → provider response |
| `messages_to_provider` | `(messages: Iterable[Message \| ExtensionItem]) → Tuple[List, List[str]]` | IR messages → provider messages |
| `messages_from_provider` | `(provider_messages: List) → List[Message \| ExtensionItem]` | Provider messages → IR messages |

**Convenience methods** (concrete):

| Method | Description |
|--------|-------------|
| `message_to_provider` | Single message wrapper around `messages_to_provider` |
| `message_from_provider` | Single message wrapper around `messages_from_provider` |

## Implementation Guide

### Creating a New Provider Converter

1. **Create provider directory**:

    ```
    src/llm_rosetta/converters/your_provider/
    ├── __init__.py
    ├── content_ops.py
    ├── tool_ops.py
    ├── message_ops.py
    ├── config_ops.py
    └── converter.py
    ```

2. **Implement content operations**:

    ```python
    from ..base import BaseContentOps

    class YourProviderContentOps(BaseContentOps):
        @staticmethod
        def ir_text_to_p(ir_text, **kwargs):
            return {"type": "text", "value": ir_text["text"]}

        @staticmethod
        def p_text_to_ir(provider_text, **kwargs):
            return {"type": "text", "text": provider_text["value"]}

        # Implement all other abstract methods...
    ```

3. **Implement tool operations**:

    ```python
    from ..base import BaseToolOps

    class YourProviderToolOps(BaseToolOps):
        @staticmethod
        def ir_tool_definition_to_p(ir_tool, **kwargs):
            return {"name": ir_tool["name"], "params": ir_tool["parameters"]}

        # Implement all other abstract methods...
    ```

4. **Implement message operations**:

    ```python
    from ..base import BaseMessageOps

    class YourProviderMessageOps(BaseMessageOps):
        def __init__(self, content_ops, tool_ops):
            self.content_ops = content_ops
            self.tool_ops = tool_ops

        @staticmethod
        def ir_messages_to_p(ir_messages, **kwargs):
            # Use content_ops and tool_ops to convert message content
            ...

        # Implement all other abstract methods...
    ```

5. **Implement config operations**:

    ```python
    from ..base import BaseConfigOps

    class YourProviderConfigOps(BaseConfigOps):
        @staticmethod
        def ir_generation_config_to_p(ir_config, **kwargs):
            return {"temp": ir_config.get("temperature", 1.0)}

        # Implement all other abstract methods...
    ```

6. **Compose the converter**:

    ```python
    from ..base import BaseConverter
    from .content_ops import YourProviderContentOps
    from .tool_ops import YourProviderToolOps
    from .message_ops import YourProviderMessageOps
    from .config_ops import YourProviderConfigOps

    class YourProviderConverter(BaseConverter):
        content_ops_class = YourProviderContentOps
        tool_ops_class = YourProviderToolOps
        message_ops_class = YourProviderMessageOps
        config_ops_class = YourProviderConfigOps

        def __init__(self):
            self.content_ops = self.content_ops_class()
            self.tool_ops = self.tool_ops_class()
            self.message_ops = self.message_ops_class(self.content_ops, self.tool_ops)
            self.config_ops = self.config_ops_class()

        def request_to_provider(self, ir_request, **kwargs):
            warnings = []
            result = {"model": ir_request["model"]}
            # Use self.message_ops, self.config_ops, self.tool_ops
            # to convert each field of the request
            ...
            return result, warnings

        # Implement all other abstract methods...
    ```

## Existing Implementations

All 4 provider converters follow the Bottom-Up Ops Pattern:

| Provider | Converter | ContentOps | ToolOps | MessageOps | ConfigOps |
|----------|-----------|------------|---------|------------|-----------|
| OpenAI Chat | `OpenAIChatConverter` | `OpenAIChatContentOps` | `OpenAIChatToolOps` | `OpenAIChatMessageOps` | `OpenAIChatConfigOps` |
| OpenAI Responses | `OpenAIResponsesConverter` | `OpenAIResponsesContentOps` | `OpenAIResponsesToolOps` | `OpenAIResponsesMessageOps` | `OpenAIResponsesConfigOps` |
| Anthropic | `AnthropicConverter` | `AnthropicContentOps` | `AnthropicToolOps` | `AnthropicMessageOps` | `AnthropicConfigOps` |
| Google GenAI | `GoogleGenAIConverter` | `GoogleGenAIContentOps` | `GoogleGenAIToolOps` | `GoogleGenAIMessageOps` | `GoogleGenAIConfigOps` |

## Method Naming Convention

- **`ir_*_to_p()`**: Convert from IR format to Provider format
- **`p_*_to_ir()`**: Convert from Provider format to IR format
- **`*_text_*`**, **`*_image_*`**, etc.: Content type operations
- **`*_tool_definition_*`**, **`*_tool_call_*`**, etc.: Tool lifecycle operations
- **`*_messages_*`**: Batch message operations
- **`*_generation_config_*`**, **`*_stream_config_*`**, etc.: Configuration operations
- **`*_request_*`**, **`*_response_*`**: Top-level request/response operations

## Testing Strategy

Each ops class can be tested independently:

- **ContentOps tests**: Test individual content part conversions
- **ToolOps tests**: Test tool definition, choice, call, result conversions
- **MessageOps tests**: Test batch message conversion with mixed content types
- **ConfigOps tests**: Test configuration parameter conversions
- **Converter tests**: Test full request/response conversion pipelines

See `tests/test_converters_base.py` for base module tests and `tests/converters/<provider>/` for provider-specific tests.