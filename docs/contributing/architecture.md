---
title: 架构指南
---

# 架构指南

## 转换器结构

LLM-Rosetta 使用轴辐式（hub-and-spoke）架构，每个转换器负责特定 API 格式与共享 IR（Intermediate Representation）之间的双向转换。

每个转换器位于 `src/llm_rosetta/converters/<name>/` 下，继承 `BaseConverter`。基类使用**组合模式**——子类通过类属性声明各关注点的 ops 类：

```
converters/<name>/
├── converter.py      # 主转换器类（继承 BaseConverter）
├── content_ops.py    # 内容块转换（文本、图片、refusal 等）
├── message_ops.py    # 消息级转换（角色、多轮对话）
├── tool_ops.py       # 工具定义、工具调用、工具结果
├── config_ops.py     # 请求配置（temperature、top_p、流式选项）
└── _constants.py     # 格式特定的常量
```

### 新代码放哪里

为已有格式添加新功能（如新的内容类型、新的字段）时，应在该转换器**对应的 ops 模块**中实现，而非写在独立的临时代码中。

尽可能复用 `converters/base/` 中的共享逻辑——基础模块提供了通用构建块：

| 基础模块 | 用途 |
|---------|------|
| `content.py` | 内容块辅助函数（文本、图片、refusal） |
| `messages.py` | 消息级工具函数 |
| `tools.py` | 工具定义和调用辅助函数 |
| `reasoning.py` | Reasoning/thinking 字段处理 |
| `schema.py` | JSON Schema 规范化 |
| `passthrough.py` | Provider 特定透传项 |
| `context.py` | `ConversionContext`（承载 shim、选项、状态） |

如果发现跨转换器重复逻辑，应考虑提取到基础模块中。

## Shim 层

`ProviderShim` 是一个轻量级身份卡，声明 provider 使用的基础转换器以及连接默认值和字段级变换。Shim 位于 `src/llm_rosetta/shims/providers/<name>/provider.yaml`。

转换器保持格式通用；shim 声明 provider 特定的差异（如 response ID 前缀、默认请求头、字段重命名）。

## 添加新的转换器

要添加对新 API 标准的支持：

1. 在 `src/llm_rosetta/converters/<name>/` 下创建转换器目录
2. 继承 `BaseConverter` 并实现所有抽象方法
3. 按照上述模式创建 ops 类
4. 在 `src/llm_rosetta/shims/providers/<name>/` 下添加 shim
5. 在 `tests/converters/` 下添加测试
6. 提交 PR

可参考现有转换器。

## Provider 特定数据保留

LLM-Rosetta 将 portable IR 数据、可持久化的 provider passthrough 数据和单次转换上下文状态分为三层：

| 层 | 生命周期 | 用途 |
|----|----------|------|
| Portable IR（`Message`、内容部分、工具、推理） | 可序列化、可复用 | 可在不同 provider 格式之间转换的语义 |
| `ProviderPassthroughEvent` / `ProviderPassthroughItem` | 可序列化、可复用 | 没有 portable 表达的 provider 原生 chunk 或 item |
| `ConversionContext` / `StreamContext` | 单次转换管线 | warning、option、echo 字段、原始 ID/status/annotation，以及响应重建状态 |

`ProviderPassthroughEvent` 携带 provider 原生流式 chunk。`ProviderPassthroughItem` 携带非流式请求/历史 item；`IRResponse.provider_passthrough_items` 保存独立的非流式输出 item 及其原始位置。

Passthrough 数据带有 converter dialect 标签，例如 `openai_responses` 或 `anthropic`。目标 dialect 相同时恢复原生 payload 的副本；目标格式不同时丢弃。语义 item 会产生 conversion warning；生命周期/心跳 stream event 为避免 warning 泛滥而静默丢弃。

`ConversionContext` 仍然必要，因为它承担不同职责：它是单次 request/response 管线内的临时 side channel，不会序列化进 conversation history。Provider passthrough carrier 是可持久化的 IR 数据，可以跨缓存、持久化和后续 HTTP 请求继续存在。

### 流式分派

基类 `stream_response_to_provider()` 实现使用类级分派表（`_TO_P_DISPATCH`）将 IR 流式事件路由到处理器方法。各 provider converter 通过 `_post_process_to_provider()` 钩子定制行为，无需重新实现分派逻辑。

## Round-Trip 兼容性

所有转换路径必须保持 **round-trip 兼容性**。每个变更都必须针对以下场景进行测试：

- **A → IR → A**（同格式 round-trip）：转换为 IR 再转回相同格式，必须产生有效且语义等价的结果。不允许静默丢弃字段。
- **A → IR → B**（跨格式）：从一种格式转换为另一种格式，必须为目标格式产生有效输出，即使源格式存在没有直接对应的字段。
- **A → IR → B → IR → A**（完整 round-trip）：经过两次转换再转回的消息必须保持可用。这是网关的实际执行路径——请求在入站时转换，响应在出站时转换。

添加或修改转换器逻辑时，至少要编写覆盖前两个场景的测试。网关的跨格式路由依赖于所有转换器在 IR 语义上的一致性——一个转换器的 IR 输出出错，会导致其他所有转换器的级联失败。

## 测试

测试位于 `tests/converters/<name>/` 下，与转换器结构对应。关键模式：

- **单元测试** — 按 ops 模块测试，隔离验证单个转换函数
- **Round-trip 测试** — A → IR → A 转换并断言等价性
- **跨格式测试** — A → IR → B 转换并验证输出符合格式 B 的 spec
- **流式测试** — 验证流事件顺序和生命周期

运行 `make test` 执行完整测试套件（不包括需要 API key 的集成测试）。
