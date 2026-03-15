---
title: 核心概念
---

# 核心概念

## N² 问题

有 N 个 LLM 提供商时，每对之间的直接转换需要 N×(N-1) 个转换器。4 个提供商就需要维护 12 个转换器。

## 中枢辐射解决方案

LLMIR 引入中央**中间表示（IR）**作为枢纽。每个提供商只需一个与 IR 之间的转换器，总数降为 2×N（4 个提供商仅需 8 个）。

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

## IR 消息类型

IR 定义了四种消息角色：

- **SystemMessage** — 系统指令
- **UserMessage** — 用户输入（文本、图片、文件）
- **AssistantMessage** — 模型响应（文本、工具调用、推理）
- **ToolMessage** — 工具执行结果

每条消息包含一个类型化的**内容部分**列表（TextPart、ImagePart、ToolCallPart 等）。
