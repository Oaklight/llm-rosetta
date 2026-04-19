---
title: Streaming
---

# Streaming

LLM-Rosetta supports converting streaming chunks between providers. A stateful `StreamContext` tracks session metadata, tool calls, and deferred events across the chunk sequence.

## Stream Events

Streaming produces a sequence of `IRStreamEvent` types:

| Event | Description |
|-------|-------------|
| `StreamStartEvent` | Stream has started |
| `ContentBlockStartEvent` | A new content block begins |
| `TextDeltaEvent` | Incremental text content |
| `ReasoningDeltaEvent` | Incremental reasoning/thinking content |
| `ToolCallStartEvent` | Tool call begins (name + ID) |
| `ToolCallDeltaEvent` | Incremental tool call arguments |
| `ContentBlockEndEvent` | Current content block ends |
| `FinishEvent` | Model finished generating (stop reason) |
| `UsageEvent` | Token usage statistics |
| `StreamEndEvent` | Stream has finished |

## Converting Stream Chunks

Use `stream_response_from_provider()` to convert provider-native chunks into IR events:

```python
from llm_rosetta import OpenAIChatConverter
from llm_rosetta.converters.base import StreamContext

converter = OpenAIChatConverter()
ctx = StreamContext()

for chunk in provider_stream:
    ir_events = converter.stream_response_from_provider(
        chunk.model_dump(), context=ctx
    )
    for event in ir_events:
        if event["type"] == "text_delta":
            print(event["text"], end="")
```

Use `stream_response_to_provider()` to convert IR events back into a target provider's format:

```python
from llm_rosetta import AnthropicConverter
from llm_rosetta.converters.base import StreamContext

target = AnthropicConverter()
target_ctx = StreamContext()

for ir_event in ir_events:
    provider_chunk = target.stream_response_to_provider(ir_event, context=target_ctx)
    # provider_chunk is a dict (or list of dicts) in the target format
```

## StreamContext

`StreamContext` is a dataclass that extends `ConversionContext`, adding session-level state for stateful stream transformations.

```python
from llm_rosetta.converters.base import StreamContext

# Create directly
ctx = StreamContext()

# Or via factory (equivalent)
from llm_rosetta import BaseConverter
ctx = BaseConverter.create_stream_context()
```

### Inheritance

```text
ConversionContext          # warnings, options, metadata
  └── StreamContext        # + session metadata, tool tracking, lifecycle
        └── OpenAIResponsesStreamContext   # + sequence_number, item tracking
```

Since `StreamContext` IS-A `ConversionContext`, it carries the same `warnings`, `options`, and `metadata` fields. You can pass `metadata_mode="preserve"` for lossless round-trip:

```python
ctx = StreamContext(options={"metadata_mode": "preserve"})
```

### Session Metadata

The converter populates these fields from the first provider chunk:

| Field | Type | Description |
|-------|------|-------------|
| `response_id` | `str` | Provider response ID (e.g., `chatcmpl-xxx`) |
| `model` | `str` | Model name from the response |
| `created` | `int` | Unix timestamp |
| `current_block_index` | `int` | Current 0-based content block index |

### Lifecycle

```python
ctx.mark_started()     # Called by StreamStartEvent handler
ctx.mark_ended()       # Called by StreamEndEvent handler

ctx.is_started  # bool — has the stream begun?
ctx.is_ended    # bool — has the stream finished?
```

Lifecycle guards prevent duplicate events — for example, `content_block_end` is only emitted if a block is actually open.

## Tool Call Tracking

During streaming, tool call arguments arrive incrementally. `StreamContext` accumulates them:

```python
# Typically called by the converter automatically:
ctx.register_tool_call("call_abc", "get_weather")
ctx.append_tool_call_args("call_abc", '{"city":')
ctx.append_tool_call_args("call_abc", '"NYC"}')

# Query accumulated state:
ctx.get_tool_name("call_abc")        # "get_weather"
ctx.get_tool_call_args("call_abc")   # '{"city":"NYC"}'

# Get all registered tool calls in order:
for call_id, name, args in ctx.get_pending_tool_calls():
    print(f"{name}({args})")
```

For OpenAI Responses, tool call item IDs are also tracked:

```python
ctx.register_tool_call_item("call_abc", "item_xyz")
ctx.get_tool_call_item_id("call_abc")  # "item_xyz"
```

## Deferred Event Buffering

Some providers send usage and finish information in separate chunks, or combine text and finish in a single frame. To prevent duplicate terminal events and event inflation, `StreamContext` provides buffer methods:

```python
# Buffer usage for later merging into a finish event
ctx.buffer_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
usage = ctx.pop_pending_usage()  # returns dict and clears buffer

# Buffer a finish event for later emission
ctx.buffer_finish({"stop_reason": "end_turn"})
finish = ctx.pop_pending_finish()  # returns dict and clears buffer
```

This pattern is used internally by converters to merge usage into finish events, avoiding separate `UsageEvent` + `FinishEvent` pairs that would inflate the output stream during cross-provider conversion.

## Cross-Provider Streaming

A complete example converting OpenAI Chat SSE → IR → Anthropic SSE:

```python
from llm_rosetta import OpenAIChatConverter, AnthropicConverter
from llm_rosetta.converters.base import StreamContext

source = OpenAIChatConverter()
target = AnthropicConverter()
from_ctx = StreamContext()
to_ctx = StreamContext()

for chunk in openai_stream:
    # Provider A → IR
    ir_events = source.stream_response_from_provider(
        chunk.model_dump(), context=from_ctx
    )
    # IR → Provider B
    for event in ir_events:
        result = target.stream_response_to_provider(event, context=to_ctx)
        if result:
            yield result  # SSE chunk in Anthropic format
```

The base `stream_response_to_provider()` uses a class-level dispatch table (`_TO_P_DISPATCH`) to route each IR event type to its handler method. Provider converters customize output through a `_post_process_to_provider()` hook — for example, OpenAI Chat injects `id`, `object`, `model`, and `created` envelope fields into every chunk.

## Provider-Specific StreamContext

The OpenAI Responses API requires additional per-event state (sequence numbers, output item tracking). `OpenAIResponsesStreamContext` extends `StreamContext` with these fields.

When a base `StreamContext` is passed to `OpenAIResponsesConverter.stream_response_to_provider()`, it is automatically upgraded via `OpenAIResponsesStreamContext.from_base()`:

```python
from llm_rosetta import OpenAIResponsesConverter
from llm_rosetta.converters.base import StreamContext

converter = OpenAIResponsesConverter()
ctx = StreamContext()  # base context is fine

# Internally upgraded to OpenAIResponsesStreamContext on first call
result = converter.stream_response_to_provider(event, context=ctx)
```
