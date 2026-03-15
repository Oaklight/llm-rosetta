# OpenAI Chat Completion API Type Replicas

This directory contains TypedDict replicas of OpenAI Chat Completion API types, providing type-safe interfaces for the LLM-Rosetta library's OpenAI Chat converter.

## Why Type Replicas?

### 1. **Dependency Independence**

- Avoids direct dependency on the OpenAI SDK for type definitions
- Prevents version conflicts and reduces package size
- Allows LLM-Rosetta to work without requiring the full OpenAI SDK installation

### 2. **Type Safety & Validation**

- Provides strict TypedDict definitions for runtime type checking
- Enables better IDE support and static analysis
- Ensures compatibility across different OpenAI SDK versions

### 3. **Converter Implementation**

- Essential for implementing the IR ↔ OpenAI Chat type conversion
- Provides clear mapping targets for IR types
- Enables efficient serialization/deserialization

## File Organization

The types are organized into three focused modules:

### [`request_types.py`](./request_types.py)

Request parameters and configuration types:

- **`CompletionCreateParams`** - Main request body type
- **Tool Types** - `ChatCompletionFunctionToolParam`, `ChatCompletionToolChoiceOptionParam`
- **Generation Control** - `ReasoningEffort`, temperature, top_p, etc.
- **Response Format** - `ResponseFormat`, JSON schema configurations
- **Stream Options** - `ChatCompletionStreamOptionsParam`

### [`message_types.py`](./message_types.py)

Message and content part types:

- **Message Parameters** - `ChatCompletionMessageParam` (union of all message types)
  - `ChatCompletionSystemMessageParam`
  - `ChatCompletionUserMessageParam`
  - `ChatCompletionAssistantMessageParam`
  - `ChatCompletionToolMessageParam`
- **Content Parts** - `ChatCompletionContentPartParam` (union of content types)
  - `ChatCompletionContentPartTextParam`
  - `ChatCompletionContentPartImageParam`
  - `ChatCompletionContentPartInputAudioParam`
- **Tool Calls** - `ChatCompletionMessageToolCallParam`

### [`response_types.py`](./response_types.py)

Response structure and metadata types:

- **`ChatCompletion`** - Main response type
- **Choices** - `Choice`, `ChoiceLogprobs`
- **Messages** - `ChatCompletionMessage`
- **Tool Calls** - `ChatCompletionMessageToolCallUnion`
- **Usage Statistics** - `CompletionUsage`, token details
- **Annotations** - `Annotation`, `AnnotationURLCitation`
- **Audio** - `ChatCompletionAudio`

## IR Type Mapping

This section shows how LLM-Rosetta's IR types map to OpenAI Chat types for conversion:

### Request Mapping

| IR Type                                    | OpenAI Chat Type                           | Notes                                |
| ------------------------------------------ | ------------------------------------------ | ------------------------------------ |
| [`IRRequest`](../ir/request.py)            | `CompletionCreateParams`                   | Main request structure               |
| [`SystemMessage`](../ir/messages.py)       | `ChatCompletionSystemMessageParam`         | System instructions                  |
| [`UserMessage`](../ir/messages.py)         | `ChatCompletionUserMessageParam`           | User input with multimodal support   |
| [`AssistantMessage`](../ir/messages.py)    | `ChatCompletionAssistantMessageParam`      | AI responses with tool calls         |
| [`ToolMessage`](../ir/messages.py)         | `ChatCompletionToolMessageParam`           | Tool execution results               |
| [`ToolDefinition`](../ir/tools.py)         | `ChatCompletionFunctionToolParam`          | Function tool definitions            |
| [`ToolChoice`](../ir/tools.py)             | `ChatCompletionToolChoiceOptionParam`      | Tool selection strategy              |
| [`GenerationConfig`](../ir/configs.py)     | Various fields in `CompletionCreateParams` | Temperature, top_p, max_tokens, etc. |
| [`ResponseFormatConfig`](../ir/configs.py) | `ResponseFormat`                           | JSON schema, structured output       |
| [`StreamConfig`](../ir/configs.py)         | `ChatCompletionStreamOptionsParam`         | Streaming configuration              |
| [`ReasoningConfig`](../ir/configs.py)      | `ReasoningEffort`                          | Reasoning effort levels              |

### Content Part Mapping

| IR Content Part                    | OpenAI Chat Type                              | Conversion Notes       |
| ---------------------------------- | --------------------------------------------- | ---------------------- |
| [`TextPart`](../ir/parts.py)       | `ChatCompletionContentPartTextParam`          | Direct mapping         |
| [`ImagePart`](../ir/parts.py)      | `ChatCompletionContentPartImageParam`         | URL or base64 data     |
| [`AudioPart`](../ir/parts.py)      | `ChatCompletionContentPartInputAudioParam`    | Input audio only       |
| [`ToolCallPart`](../ir/parts.py)   | `ChatCompletionMessageToolCallParam`          | Function calls         |
| [`ToolResultPart`](../ir/parts.py) | `ChatCompletionToolMessageParam`              | Tool execution results |
| [`RefusalPart`](../ir/parts.py)    | `ChatCompletionAssistantMessageParam.refusal` | Safety refusals        |
| [`CitationPart`](../ir/parts.py)   | `Annotation`                                  | Web search citations   |

### Response Mapping

| IR Type                             | OpenAI Chat Type       | Notes                       |
| ----------------------------------- | ---------------------- | --------------------------- |
| [`IRResponse`](../ir/response.py)   | `ChatCompletion`       | Main response structure     |
| [`ChoiceInfo`](../ir/response.py)   | `Choice`               | Individual response choices |
| [`UsageInfo`](../ir/response.py)    | `CompletionUsage`      | Token usage statistics      |
| [`FinishReason`](../ir/response.py) | `Choice.finish_reason` | Stop reason enumeration     |

### Key Conversion Patterns

1. **Message Structure**: IR uses `List[ContentPart]` while OpenAI uses separate fields (`content`, `tool_calls`, `refusal`)
2. **Tool Definitions**: IR uses flat structure, OpenAI uses nested `{"type": "function", "function": {...}}`
3. **Tool Arguments**: IR uses `Dict[str, Any]`, OpenAI uses JSON string
4. **System Instructions**: IR has dedicated field, OpenAI embeds in messages array

## Maintenance Guidelines

### 1. **Sync with OpenAI SDK**

When OpenAI releases SDK updates:

```bash
# Check current SDK version
pip show openai

# Compare with our replicas
diff -r /path/to/openai/types/chat/ ./src/llm_rosetta/types/openai/chat/
```

**Update Process:**

1. Review OpenAI SDK changelog for type changes
2. Update affected TypedDict definitions
3. Maintain backward compatibility where possible
4. Update this README if new mappings are added
5. Run tests to ensure converter compatibility

### 2. **Adding New Types**

When adding new OpenAI Chat types:

```python
# Follow this pattern in the appropriate file
class NewOpenAIType(TypedDict, total=False):
    """Brief description.

    Reference: openai.types.chat.NewOpenAIType
    """

    required_field: Required[str]
    optional_field: NotRequired[Optional[int]]
```

**Guidelines:**

- Always include SDK reference in docstring
- Use `total=False` for optional fields
- Use `Required[]` and `NotRequired[]` appropriately
- Add to `__all__` list in module
- Export in `__init__.py`

### 3. **Version Compatibility**

**Supported OpenAI SDK Versions:** `1.x.x` through `2.14.0`

**Version Compatibility:**

- **OpenAI SDK 1.x → 2.x**: **Chat Completion API is backward compatible**
  - v2.0.0 (released 2025-09-30) introduced breaking changes only in **Responses API**
  - Last 1.x version: v1.109.1 (released 2025-09-24)
  - **Chat Completion API types remain unchanged** between 1.x and 2.x
  - Breaking changes were limited to `ResponseFunctionToolCallOutputItem.output` (Responses API only)
- **Within 2.x series**: Fully backward compatible for minor/patch versions

**Breaking Changes Policy:**

- Major version bumps (1.x → 2.x) introduce breaking changes
- Minor version bumps within same major version maintain backward compatibility
- Patch versions only fix bugs or add optional fields

### 4. **Testing Strategy**

```python
# Test type compatibility
def test_openai_type_compatibility():
    from openai.types.chat import ChatCompletion as SDKChatCompletion
    from llm_rosetta.types.openai.chat import ChatCompletion as LLMRosettaChatCompletion

    # Ensure our replica accepts SDK data
    sdk_response: SDKChatCompletion = get_openai_response()
    llm_rosetta_response: LLMRosettaChatCompletion = sdk_response.model_dump()
```

### 5. **Documentation Updates**

When modifying types:

1. Update type mapping tables in this README
2. Update converter documentation if mappings change
3. Add migration notes for breaking changes
4. Update examples if API patterns change

## Usage Examples

### Basic Request Construction

```python
from llm_rosetta.types.openai.chat import (
    CompletionCreateParams,
    ChatCompletionUserMessageParam,
)

request: CompletionCreateParams = {
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ],
    "temperature": 0.7,
    "max_completion_tokens": 1000,
}
```

### Tool Usage

```python
from llm_rosetta.types.openai.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionToolChoiceOptionParam,
)

tools: list[ChatCompletionFunctionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

tool_choice: ChatCompletionToolChoiceOptionParam = {
    "type": "function",
    "function": {"name": "get_weather"}
}
```

### Response Processing

```python
from llm_rosetta.types.openai.chat import ChatCompletion

def extract_content(response: ChatCompletion) -> str:
    """Extract text content from response."""
    choice = response["choices"][0]
    message = choice["message"]
    return message.get("content") or ""

def extract_tool_calls(response: ChatCompletion) -> list:
    """Extract tool calls from response."""
    choice = response["choices"][0]
    message = choice["message"]
    return message.get("tool_calls") or []
```

## References

- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [OpenAI Chat Completion API Documentation](https://platform.openai.com/docs/api-reference/chat)
- [LLM-Rosetta IR Types](../ir/)
- [LLM-Rosetta OpenAI Chat Converter](../../converters/openai/chat/)

## SDK Source Locations

For reference when updating types:

```
<python_env>/lib/python3.10/site-packages/openai/types/chat/
├── chat_completion.py                    → response_types.py
├── completion_create_params.py           → request_types.py
├── chat_completion_message_param.py      → message_types.py
├── chat_completion_message.py            → response_types.py
└── ... (other specific type files)
```
