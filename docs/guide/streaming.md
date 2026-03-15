---
title: Streaming
---

# Streaming

LLMIR supports converting streaming chunks between providers.

## Stream Events

Streaming produces a sequence of `IRStreamEvent` types:

| Event | Description |
|-------|-------------|
| `StreamStartEvent` | Stream has started |
| `TextDeltaEvent` | Incremental text content |
| `ToolCallStartEvent` | Tool call begins |
| `ToolCallDeltaEvent` | Incremental tool call arguments |
| `ReasoningDeltaEvent` | Incremental reasoning content |
| `StreamEndEvent` | Stream has finished |

## Converting Stream Chunks

```python
from llmir.converters.base import StreamContext

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

## Accumulating to Messages

After streaming completes, reconstruct the full message:

```python
from llmir import accumulate_stream_to_assistant_message

full_message = accumulate_stream_to_assistant_message(all_events)
```
