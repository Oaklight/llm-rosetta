---
title: 核心概念
---

# 核心概念

## N² 问题

有 N 个 LLM 提供商时，每对之间的直接转换需要 N×(N-1) 个转换器。4 个提供商就需要维护 12 个转换器。

## 中枢辐射解决方案

LLM-Rosetta 引入中央**中间表示（IR）**作为枢纽。每个提供商只需一个与 IR 之间的转换器，总数降为 2×N（4 个提供商仅需 8 个）。

```text
提供商 A ←→ IR ←→ 提供商 B
提供商 C ←→ IR ←→ 提供商 D
```

## 转换器架构

每个转换器（如 `OpenAIChatConverter`）由四个专用操作类组合而成：

| 组件 | 职责 |
|------|------|
| `ContentOps` | 转换内容部分（文本、图片、工具调用等） |
| `MessageOps` | 转换完整消息（角色 + 内容） |
| `ToolOps` | 转换工具定义和工具选择设置 |
| `ConfigOps` | 转换生成参数（temperature、max_tokens 等） |

这些组合成 6 个主要转换器接口：

- `request_to_provider()` / `request_from_provider()`
- `response_to_provider()` / `response_from_provider()`
- `messages_to_provider()` / `messages_from_provider()`

以及 2 个流式接口：

- `stream_response_from_provider()` / `stream_response_to_provider()`

## 转换上下文

所有转换器方法接受可选的 `ConversionContext`（非流式）或 `StreamContext`（流式），用于在管线中传递共享状态：

- **`warnings`** — 累积的转换注意事项（如不支持的特性被丢弃）
- **`options`** — 结构化转换选项（如 `output_format`、`metadata_mode`）
- **`metadata`** — 不透明存储，用于提供商特有状态

`metadata_mode` 选项（`"strip"` 或 `"preserve"`）控制提供商特有字段是否在往返中保留。详见[使用转换器 — 元数据保留](converters.md#元数据保留无损往返)。

`StreamContext` 继承自 `ConversionContext`，增加了会话级元数据、工具调用追踪和生命周期标志，用于有状态的流式转换。详见[流式处理](streaming.md)指南。

基类 `stream_response_to_provider()` 实现使用类级分派表（`_TO_P_DISPATCH`）将 IR 流式事件路由到处理器方法。各 provider converter 通过 `_post_process_to_provider()` 钩子定制行为，无需重新实现分派逻辑。

## IR 消息类型

IR 定义了四种消息角色：

- **SystemMessage** — 系统指令
- **UserMessage** — 用户输入（文本、图片、文件）
- **AssistantMessage** — 模型响应（文本、工具调用、推理）
- **ToolMessage** — 工具执行结果

每条消息包含一个类型化的**内容部分**列表（TextPart、ImagePart、ToolCallPart 等）。
