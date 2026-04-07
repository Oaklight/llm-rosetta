---
title: 流式处理
---

# 流式处理

LLM-Rosetta 支持在提供商之间转换流式数据块。

## 流式事件

流式处理产生一系列 `IRStreamEvent` 类型：

| 事件 | 描述 |
|------|------|
| `StreamStartEvent` | 流已开始 |
| `TextDeltaEvent` | 增量文本内容 |
| `ToolCallStartEvent` | 工具调用开始 |
| `ToolCallDeltaEvent` | 增量工具调用参数 |
| `ReasoningDeltaEvent` | 增量推理内容 |
| `StreamEndEvent` | 流已结束 |

## 转换流式数据块

```python
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

