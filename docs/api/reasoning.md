---
title: 推理能力配置
---

# 推理能力配置

`ReasoningCapability` 数据类声明了 provider 如何处理推理和思考配置。它是 shim 层面控制推理行为三个维度的接口。

概念概述请参见[推理 / 思考指南](../guide/reasoning.md)。

## 三个维度

### 思考开关 (`thinking_modes`)

将 IR 推理模式映射到 provider 特定的思考类型值。

- `None` — provider 不支持思考块（安全默认值；标准 OpenAI、Google）
- `dict` — IR 模式 → provider 值映射

```yaml
# Anthropic
thinking_modes:
  auto: adaptive
  enabled: enabled
  disabled: disabled

# DeepSeek（不支持 auto）
thinking_modes:
  enabled: enabled
  disabled: disabled
```

不在映射中的 IR 模式会被静默丢弃，防止不支持的思考块发送到上游 API。

`thinking_default` 设置请求没有显式模式时的默认 IR 模式，必须是 `thinking_modes` 中的一个 key。

### 程度 (`effort_field`, `effort_range`)

`effort_field` 是 provider 侧放置 effort 值的字段路径：

| 值 | 输出结构 |
|----|----------|
| `reasoning_effort` | `{"reasoning_effort": v}` |
| `reasoning.effort` | `{"reasoning": {"effort": v}}` |
| `output_config.effort` | `{"output_config": {"effort": v}}` |
| `thinking_level` | `{"thinking_config": {"thinking_level": v}}` |
| `none` | 不输出 effort |

`effort_range` 是 IR effort 值的 `(下限, 上限)` 元组。超出范围的值会被夹紧到最近的边界。`None` 表示支持完整的 IR 阶梯（minimal–max）。

```yaml
# OpenAI：支持 minimal–high，xhigh/max 夹紧到 high
effort_range: [minimal, high]

# Anthropic：low–max，minimal 夹紧到 low
effort_range: [low, max]

# Argo OpenAI：完整范围
# 不设置 effort_range = 不夹紧
```

### 可见性 (`visibility_modes`)

将 IR summary 值映射到 provider 特定的可见性控制。

- `None` — 使用转换器默认行为
- `dict` — IR 值 → provider 值；不在映射中的值会被省略

```yaml
# Anthropic (thinking.display)
visibility_modes:
  auto: summarized
  concise: summarized
  detailed: summarized
  none: omitted

# OpenAI (reasoning.summary)
visibility_modes:
  auto: auto
  concise: concise
  detailed: detailed
```

## 其他字段

### 预算 (`budget_ratio`)

设置后，将 `budget_tokens` 推导为 `max(1024, int(max_tokens × ratio))`，夹紧到 `max_tokens − 1`。用于 IR 请求 `mode: enabled` 但没有显式 `budget_tokens` 的情况。

### 响应处理 (`unsigned_blocks`)

控制响应中未签名（未编辑）推理块的处理方式。值：`"as_is"`（默认）、`"preserve"`。

### 回放 (`replay_missing`)

部分思考模式上游要求：一旦请求携带 `tools`，就必须回传模型的推理内容。若历史不以 `user` 结尾，且任一 assistant 消息缺少该字段，上游会返回 `400 The reasoning_content in the thinking mode must be passed back to the API`。跨格式回放历史的客户端（例如 Claude Code → `messages` → `chat_completions`）会丢掉部分轮次的推理，因此即便转换本身忠实，也会触发该错误。

- `"as_is"`（默认）—— 原样发送回放的历史。
- `"fill_empty"` —— 声明该要求。源请求转换为 IR 之后，能力层 pass 会给每条缺少推理的 assistant 消息补一个空的 reasoning part，由目标转换器按自己的方言渲染（Chat Completions 的 `reasoning_content: ""`）。空值可以满足上游 —— 已在 `deepseek-flash` 上实测 —— 请求得以正常完成。

该要求属于**模型**，而不是某个 vendor。把它声明为 shim 级默认值只能保护那一个端点；同一个模型由另一家厂商转发时仍需单独声明。对于代理多个模型家族的厂商，请放进 `model_overrides`：

```yaml
# opencode--chat_completions：只有 DeepSeek 思考模型需要回放
reasoning:
  effort_field: reasoning_effort
  effort_range: [minimal, high]
  model_overrides:
    deepseek-v4.1-flash:
      replay_missing: fill_empty
```

`backfill_reasoning_content()` 仍作为 body transform 保留，供自行构造 wire body 的调用方使用；但 shim 应改为声明 `replay_missing`：挂在单个 vendor shim 上的 transform 覆盖不到同一模型背后的其他端点。

## 按模型覆盖

Provider YAML 支持 `model_overrides` 实现按模型的推理行为：

```yaml
reasoning:
  thinking_modes:
    auto: adaptive
    enabled: enabled
    disabled: disabled
  effort_range: [low, max]
  model_overrides:
    claude-haiku-4-5-20251001:
      # Haiku 仅支持 enabled+budget，不支持 auto
      thinking_modes: {enabled: enabled, disabled: disabled}
      budget_ratio: 0.8
      effort_field: none
    claude-opus-4-7:
      # Opus 4.7+ 仅支持 adaptive
      thinking_modes: {auto: adaptive, enabled: adaptive, disabled: disabled}
```

每个模型覆盖会从 provider 级别配置继承未设置的字段。

## API 参考

::: llm_rosetta.shims.provider_shim.ReasoningCapability
    options:
      show_source: false
      heading_level: 3
