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
