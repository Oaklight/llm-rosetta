---
title: 流式处理
---

# 流式处理

LLM-Rosetta 支持在提供商之间转换流式数据块。有状态的 `StreamContext` 在数据块序列中追踪会话元数据、工具调用和延迟事件。

## 流式事件

流式处理产生一系列 `IRStreamEvent` 类型：

| 事件 | 描述 |
|------|------|
| `StreamStartEvent` | 流已开始 |
| `ContentBlockStartEvent` | 新内容块开始 |
| `TextDeltaEvent` | 增量文本内容 |
| `ReasoningDeltaEvent` | 增量推理/思考内容 |
| `ToolCallStartEvent` | 工具调用开始（名称 + ID） |
| `ToolCallDeltaEvent` | 增量工具调用参数 |
| `ContentBlockEndEvent` | 当前内容块结束 |
| `FinishEvent` | 模型完成生成（停止原因） |
| `UsageEvent` | Token 使用统计 |
| `StreamEndEvent` | 流已结束 |

## 转换流式数据块

使用 `stream_response_from_provider()` 将提供商原生数据块转换为 IR 事件：

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

使用 `stream_response_to_provider()` 将 IR 事件转换回目标提供商格式：

```python
from llm_rosetta import AnthropicConverter
from llm_rosetta.converters.base import StreamContext

target = AnthropicConverter()
target_ctx = StreamContext()

for ir_event in ir_events:
    provider_chunk = target.stream_response_to_provider(ir_event, context=target_ctx)
    # provider_chunk 是目标格式的 dict（或 dict 列表）
```

## StreamContext

`StreamContext` 是一个 dataclass，继承自 `ConversionContext`，增加了用于有状态流式转换的会话级状态。

```python
from llm_rosetta.converters.base import StreamContext

# 直接创建
ctx = StreamContext()

# 或通过工厂方法（等价）
from llm_rosetta import BaseConverter
ctx = BaseConverter.create_stream_context()
```

### 继承关系

```text
ConversionContext          # warnings, options, metadata
  └── StreamContext        # + 会话元数据、工具追踪、生命周期
        └── OpenAIResponsesStreamContext   # + sequence_number、item 追踪
```

由于 `StreamContext` 是 `ConversionContext` 的子类（IS-A 关系），它继承了相同的 `warnings`、`options` 和 `metadata` 字段。可以传入 `metadata_mode="preserve"` 实现无损往返：

```python
ctx = StreamContext(options={"metadata_mode": "preserve"})
```

### 会话元数据

转换器从第一个提供商数据块中填充以下字段：

| 字段 | 类型 | 描述 |
|------|------|------|
| `response_id` | `str` | 提供商响应 ID（如 `chatcmpl-xxx`） |
| `model` | `str` | 响应中的模型名称 |
| `created` | `int` | Unix 时间戳 |
| `current_block_index` | `int` | 当前从 0 开始的内容块索引 |

### 生命周期

```python
ctx.mark_started()     # 由 StreamStartEvent 处理器调用
ctx.mark_ended()       # 由 StreamEndEvent 处理器调用

ctx.is_started  # bool — 流是否已开始？
ctx.is_ended    # bool — 流是否已结束？
```

生命周期守卫防止重复事件——例如，`content_block_end` 仅在内容块确实打开时才会发出。

## 工具调用追踪

流式传输中，工具调用参数以增量方式到达。`StreamContext` 负责累积：

```python
# 通常由转换器自动调用：
ctx.register_tool_call("call_abc", "get_weather")
ctx.append_tool_call_args("call_abc", '{"city":')
ctx.append_tool_call_args("call_abc", '"NYC"}')

# 查询累积状态：
ctx.get_tool_name("call_abc")        # "get_weather"
ctx.get_tool_call_args("call_abc")   # '{"city":"NYC"}'

# 按注册顺序获取所有工具调用：
for call_id, name, args in ctx.get_pending_tool_calls():
    print(f"{name}({args})")
```

对于 OpenAI Responses，还会追踪工具调用 item ID：

```python
ctx.register_tool_call_item("call_abc", "item_xyz")
ctx.get_tool_call_item_id("call_abc")  # "item_xyz"
```

## 延迟事件缓冲

某些提供商在不同的 chunk 中发送 usage 和 finish 信息，或在单个帧中组合 text 和 finish。为防止重复终端事件和事件膨胀，`StreamContext` 提供缓冲方法：

```python
# 缓冲 usage 以便后续合并到 finish 事件
ctx.buffer_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
usage = ctx.pop_pending_usage()  # 返回 dict 并清空缓冲

# 缓冲 finish 事件以便后续发出
ctx.buffer_finish({"stop_reason": "end_turn"})
finish = ctx.pop_pending_finish()  # 返回 dict 并清空缓冲
```

此模式在转换器内部使用，将 usage 合并到 finish 事件中，避免跨提供商转换时产生独立的 `UsageEvent` + `FinishEvent` 对导致输出流膨胀。

## 跨提供商流式转换

完整示例：将 OpenAI Chat SSE → IR → Anthropic SSE：

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
            yield result  # Anthropic 格式的 SSE chunk
```

基类 `stream_response_to_provider()` 使用类级分派表（`_TO_P_DISPATCH`）将每个 IR 事件类型路由到对应的处理器方法。各 provider converter 通过 `_post_process_to_provider()` 钩子定制输出——例如，OpenAI Chat 在每个 chunk 中注入 `id`、`object`、`model` 和 `created` envelope 字段。

## 提供商特定 StreamContext

OpenAI Responses API 需要额外的逐事件状态（序列号、输出 item 追踪）。`OpenAIResponsesStreamContext` 扩展了 `StreamContext` 以包含这些字段。

当基础 `StreamContext` 被传递给 `OpenAIResponsesConverter.stream_response_to_provider()` 时，会通过 `OpenAIResponsesStreamContext.from_base()` 自动升级：

```python
from llm_rosetta import OpenAIResponsesConverter
from llm_rosetta.converters.base import StreamContext

converter = OpenAIResponsesConverter()
ctx = StreamContext()  # 基础 context 即可

# 首次调用时内部自动升级为 OpenAIResponsesStreamContext
result = converter.stream_response_to_provider(event, context=ctx)
```
