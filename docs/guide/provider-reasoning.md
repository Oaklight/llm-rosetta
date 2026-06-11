---
title: 提供商推理行为
---

# 提供商推理行为

[推理 / 思考参数](reasoning.md)指南介绍了 LLM-Rosetta 如何通过 IR 层统一推理参数。本页面记录**各提供商的实际行为**——接受哪些 effort 值、如何处理禁用推理、`thinking_type` 的具体语义，以及推理元数据在跨提供商往返中的保留情况。

以下所有数据均来自对线上提供商 API 的实际探测，而非仅依赖文档。详见[探测方法](#探测方法)。

!!! tip "Shim 配置参考"
    每个提供商的 shim YAML 编码了本页记录的行为。shim 架构和 `provider.yaml` 格式详见[提供商 Shim](shims.md)。

## Effort 值接受情况

各提供商对 `reasoning_effort` / `output_config.effort` 值的接受情况。未列出的值会返回 400 错误，除非另有说明。

### 官方上游 API

2026-06-10 使用真实 API key 对官方端点探测（[#185](https://github.com/Oaklight/llm-rosetta/issues/185#issuecomment-4667473346)）。

| 提供商 | 端点 | `none` | `minimal` | `low` | `medium` | `high` | `xhigh` | `max` |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **OpenAI** | Chat (`reasoning_effort`) | — | — | ✅ | ✅ | ✅ | — | — |
| **OpenAI** | Responses (`reasoning.effort`) | — | — | ✅ | ✅ | ✅ | — | — |
| **Anthropic** | Messages (`output_config.effort`) | — | — | ✅ | ✅ | ✅ | ✅ | ✅ |
| **MiniMax** | Anthropic (`output_config.effort`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **MiniMax** | Chat (`reasoning_effort`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **OpenRouter** | Chat (`reasoning_effort`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Volcengine** | Chat (`reasoning_effort`) | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **DeepSeek** | Chat (`reasoning_effort`) | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |

!!! note
    Google GenAI 不使用 effort 字符串，而是通过 `thinking_config.thinking_budget`（整数 token 数）和 `thinking_config.thinking_level`（`"minimal"` / `"low"` / `"medium"` / `"high"`）控制推理。

### Argo 网关

2026-05-23 及 2026-06-09 探测（[#220](https://github.com/Oaklight/llm-rosetta/issues/220#issuecomment-4526638344)）。

Argo 代理请求到上游提供商，但会进行自己的参数校验。Anthropic 端点使用 `output_config.effort`：

| 模型 | 单独 `effort` | `adaptive` + `effort` | `effort=low` |
|---|:---:|:---:|:---:|
| haiku45 / sonnet45 / opus41 / opus45 | ✅ | ❌ 400 | ✅ |
| sonnet46 / opus46 | ✅ | ✅ | ✅ |
| opus47 | ✅ | ✅ | ✅ |

仅 sonnet46、opus46、opus47 支持 `thinking` 与 `output_config` 共存。

## 禁用行为

各提供商禁用推理的方式。shim 的 `disabled` 字段控制 LLM-Rosetta 使用哪种策略。

| 提供商 | 策略 | Shim `disabled` | 行为 |
|---|---|---|---|
| **OpenAI**（Chat / Responses） | 省略 thinking 字段 | `omit` | 不发送 `thinking` / `reasoning` 对象 |
| **Anthropic** | 显式禁用 | `thinking_disabled` | `thinking: { type: "disabled" }` |
| **Google GenAI** | 零预算 | `thinking_budget_zero` | `thinking_config: { thinking_budget: 0 }` |
| **DeepSeek** | 显式禁用 | `thinking_disabled` | `thinking: { type: "disabled" }` |
| **Volcengine**（Chat） | 显式禁用 | `thinking_disabled` | `thinking: { type: "disabled" }` |
| **Volcengine**（Responses） | 省略 thinking 字段 | `omit` | 不发送 `thinking` / `reasoning` 对象 |
| **MiniMax**（Anthropic） | 显式禁用 | `thinking_disabled` | `thinking: { type: "disabled" }` |
| **MiniMax**（Chat） | 显式禁用 | `thinking_disabled` | `thinking: { type: "disabled" }` |
| **OpenRouter** | 省略 | `omit` | 不发送 `thinking` 对象 |
| **Argo**（Anthropic） | 显式禁用 | `thinking_disabled` | `thinking: { type: "disabled" }` |
| **Argo**（OpenAI Chat） | 省略 | `omit` | 不发送 `thinking` 对象 |

!!! warning "Volcengine 和 DeepSeek 拒绝 `none`"
    向 Volcengine 或 DeepSeek 发送 `reasoning_effort: "none"` 会返回 400 错误。要在这些提供商上禁用推理，需使用 `thinking.type: "disabled"`——shim 的 `disabled: thinking_disabled` 策略会在 IR 模式为 `disabled` 时自动处理。

## Thinking Type

提供商使用 `thinking.type = "enabled"`（需要 `budget_tokens`）还是 `"adaptive"`（模型自行决定）。部分提供商和模型只接受其中一种。

### 提供商级默认值

| 提供商 | `thinking_type` | 说明 |
|---|---|---|
| **OpenAI** | — | 无 `thinking.type` 字段；推理在推理模型上隐式启用 |
| **Anthropic** | *(两者均可)* | 接受 `enabled` 和 `adaptive`；`enabled` 需要 `budget_tokens` |
| **Google GenAI** | — | 使用整数 `thinking_budget`，无 type 字符串 |
| **DeepSeek** | — | 使用 `thinking.type`，converter 处理 `auto → adaptive` 映射 |
| **Volcengine** | `enabled` | 拒绝 `adaptive`；shim 强制 `thinking_type: enabled` |
| **MiniMax**（Chat） | `adaptive` | 拒绝 `enabled`；shim 强制 `thinking_type: adaptive` |
| **Argo**（Anthropic） | `enabled` | 默认值；但有模型级覆盖，见下方 |

### Argo Anthropic 模型级覆盖

2026-05-23 探测（[#220](https://github.com/Oaklight/llm-rosetta/issues/220#issuecomment-4526638344)）。Argo Anthropic 端点的 thinking 行为因模型而异：

| 模型 | `enabled`（无 budget） | `enabled` + `budget_tokens` | `adaptive` | 说明 |
|---|:---:|:---:|:---:|---|
| haiku45 / sonnet45 | ❌ 需要 budget | ✅ | ❌ | 仅 `enabled` + budget |
| opus41 / opus45 | ❌ 需要 budget | ✅ | ❌ | 仅 `enabled` + budget |
| sonnet46 / opus46 | ❌ 需要 budget | ✅ | ✅ | 两种模式均可 |
| **opus47** | ❌ type 不被接受 | ❌ type 不被接受 | ✅ | **仅接受 `adaptive`** |

!!! info "opus47 shim 覆盖"
    `argo--anthropic` shim 在提供商级声明 `thinking_type: enabled`，同时为 `claudeopus47` 设置 `model_overrides` 覆盖为 `thinking_type: adaptive`。参见 [PR #256](https://github.com/Oaklight/llm-rosetta/pull/256)。

### 自动回退

当 shim 声明 `thinking_type: enabled` 但请求中没有 `budget_tokens`（Anthropic 的 `type: "enabled"` 必须提供此字段）时，converter 会自动回退为 `type: "adaptive"`，避免生成无效请求体，无需客户端了解提供商约束。

## 提供商元数据往返

推理块携带的提供商特定元数据需要在跨提供商转换中保留。LLM-Rosetta 通过内容块上的 IR `provider_metadata` 字段保留这些数据。

| 元数据字段 | 来源提供商 | 附着位置 | 用途 |
|---|---|---|---|
| `signature` | Anthropic | `ReasoningPart` | thinking 块重放的加密签名 |
| `thoughtSignature` | Google GenAI | `provider_metadata.google` | Gemini 2.5+ 的签名，等同于 Anthropic 的 signature |
| `encrypted_content` | Anthropic | `provider_metadata.anthropic` | 加密的推理内容（对 converter 不透明） |
| `reasoning_details` | OpenAI | `provider_metadata.openai` | 扩展推理元数据 |

### 跨提供商签名处理

在提供商之间路由时（如 Anthropic 客户端 → Google 后端），来源提供商的签名必须保留，以确保：

1. 来源客户端可以使用有效签名重放对话历史
2. 目标提供商收到其自身格式的签名（如适用）

LLM-Rosetta 在 IR 转换时将签名存储在各内容块的 `provider_metadata` 中。Anthropic converter 将 `provider_metadata` 序列化为 `thinking`、`tool_use`、`tool_result` 和 `text` 块上的 `_provider_metadata` 字段（[PR #257](https://github.com/Oaklight/llm-rosetta/pull/257)、[PR #263](https://github.com/Oaklight/llm-rosetta/pull/263)）。

## 未签名推理块

部分客户端发送的对话历史中包含没有有效签名的 `thinking` 块。这种情况出现在：

- 客户端剥离或从未收到签名（如 Claude CLI 发送 `signature: ""`）
- 对话跨越了提供商，原始签名不再适用

### 问题

校验签名的提供商（如 Argo 的 Anthropic 端点）会拒绝这些块：

```text
messages.61.content.0.thinking.signature: Field required
messages.9.content.0: Invalid `signature` in `thinking` block
```

### Shim 策略：`unsigned_reasoning_blocks`

`ReasoningCapability` 支持 `unsigned_reasoning_blocks` 字段，有两个取值：

| 值 | 行为 |
|---|---|
| `as_is`（默认） | 原样转发未签名推理块，由目标提供商决定接受或拒绝。 |
| `preserve` | 从出站消息中移除未签名推理块。推理内容保存在 IR 部件的 `provider_metadata.anthropic.unsigned_reasoning_blocks` 中，供下游消费者使用。 |

目前仅 `argo--anthropic` 使用 `preserve`。直接调用 Anthropic API 使用 `as_is`，因为官方 API 对未签名块的处理方式与 Argo 代理层不同。

当 `preserve` 过滤掉 assistant 消息中的所有推理部件时，该消息会被整体跳过（附带警告），而非发送空的 `content: []` 数组。

参见 [#268](https://github.com/Oaklight/llm-rosetta/issues/268) 和 [PR #269](https://github.com/Oaklight/llm-rosetta/pull/269)。

## Shim 配置参考

`provider.yaml` 中 `ReasoningCapability` 各字段与上文行为的对应关系。

```yaml
reasoning:
  disabled: thinking_disabled  # "omit" | "thinking_disabled" | "thinking_budget_zero"
  effort_field: output_config.effort  # effort 值放置的请求字段
  thinking_type: enabled       # 强制 thinking.type: "enabled" | "adaptive"
  max_effort: high             # 将 IR effort 封顶到此级别
  unsigned_reasoning_blocks: preserve  # "as_is" | "preserve"
  effort_map:                  # IR effort → 提供商 effort 字符串
    minimal: low
    low: low
    medium: medium
    high: high
    xhigh: xhigh
    max: max
  model_overrides:             # 逐模型覆盖（按上游模型 ID 索引）
    claudeopus47:
      thinking_type: adaptive
```

### 字段参考

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `disabled` | string | `"omit"` | IR 模式为 `disabled` 时的策略。`omit`：不发送 thinking 字段。`thinking_disabled`：发送 `type: "disabled"`。`thinking_budget_zero`：发送 `thinking_budget: 0`。 |
| `effort_field` | string | `"reasoning_effort"` | effort 值在请求中的字段路径。`"none"` 表示不输出 effort。 |
| `thinking_type` | string | *(无)* | 强制出站 `thinking.type`。在 converter 自身映射之后应用。 |
| `max_effort` | string | *(无)* | 允许输出的最高 IR effort 级别。更高级别会被截断到此值。 |
| `unsigned_reasoning_blocks` | string | `"as_is"` | 出站无有效签名的推理块的处理策略。 |
| `effort_map` | map | *(恒等映射)* | IR effort 级别到提供商特定字符串的映射。未映射的级别会被丢弃并发出警告。 |
| `model_overrides` | map | *(无)* | 逐模型覆盖上述字段。键为 shim `model_id_field` 指定的上游模型 ID。 |

### 提供商 Shim 汇总

| 提供商 | `disabled` | `effort_field` | `thinking_type` | `max_effort` | `unsigned_reasoning_blocks` |
|---|---|---|---|---|---|
| **openai** | `omit` | `reasoning_effort` | — | `high` | `as_is` |
| **openai_responses** | `omit` | `reasoning.effort` | — | `high` | `as_is` |
| **anthropic** | `thinking_disabled` | `output_config.effort` | — | — | `as_is` |
| **google** | `thinking_budget_zero` | `none` | — | — | `as_is` |
| **deepseek** | `thinking_disabled` | `reasoning_effort` | — | — | `as_is` |
| **volcengine**（Chat） | `thinking_disabled` | `reasoning_effort` | `enabled` | `high` | `as_is` |
| **volcengine**（Responses） | `omit` | `reasoning.effort` | — | — | `as_is` |
| **minimax**（Anthropic） | `thinking_disabled` | `output_config.effort` | — | — | `as_is` |
| **minimax**（Chat） | `thinking_disabled` | `reasoning_effort` | `adaptive` | — | `as_is` |
| **openrouter** | `omit` | `reasoning_effort` | — | `xhigh` | `as_is` |
| **argo**（Anthropic） | `thinking_disabled` | `output_config.effort` | `enabled` | — | `preserve` |
| **argo**（OpenAI Chat） | `omit` | `reasoning_effort` | — | — | `as_is` |

## 探测方法

本页面所有行为数据均通过向线上提供商端点发送真实 API 请求并观察接受/拒绝响应获得。这种方法能发现提供商文档可能未覆盖的未记录约束。

### 数据来源

- **Argo 参数探测** — [#220](https://github.com/Oaklight/llm-rosetta/issues/220)：对 24 个非 embedding 模型在 OpenAI Chat 和 Anthropic 两个端点上进行探测，覆盖采样参数、thinking 模式、effort 值、工具 schema 和字段约束。
- **官方 API effort 探测** — [#185 评论](https://github.com/Oaklight/llm-rosetta/issues/185#issuecomment-4667473346)：对 OpenAI、Anthropic（通过 OpenRouter）、DeepSeek、Volcengine 和 MiniMax 的 effort 值接受情况进行测试。
- **未签名推理块发现** — [#268](https://github.com/Oaklight/llm-rosetta/issues/268)：dev-test 重放 Claude CLI 会话时发现 Argo Anthropic 拒绝带空签名的 `thinking` 块。

### 发现的文档偏差

提供商文档与实际 API 行为并不总是一致。Argo 探测中的典型案例：

| 文档描述 | 实际行为 |
|---|---|
| o 系列模型拒绝 `temperature` / `top_p` | 接受并静默忽略 |
| gpt5 / gpt5mini / gpt5nano 接受 `temperature` / `top_p` | 仅接受 `temperature=1.0`；其他值返回 400 |
| opus47 支持 `thinking.type: "enabled"` | 仅接受 `adaptive` |

这些偏差说明了经验探测的必要性——也解释了 shim 配置存在的意义：编码的是实际行为，而非文档描述的行为。
