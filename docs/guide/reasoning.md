---
title: 推理 / 思考参数
---

# 推理 / 思考参数

现代 LLM 可以在生成最终回答之前进行显式的思维链推理。各提供商通过不同的参数名称、结构和语义来暴露这一能力。LLM-Rosetta 的 `ReasoningConfig` 提供了统一的 IR 层，映射到所有支持的提供商。

## 提供商对比

### 模式控制

各提供商如何控制推理的开启、关闭或自动模式：

| 提供商 | 参数 | 取值 |
|--------|------|------|
| **Anthropic** | `thinking.type` | `"adaptive"`（模型自行决定）、`"enabled"`（始终开启，需要 `budget_tokens`）、`"disabled"`（关闭） |
| **OpenAI Chat** | *（隐式）* | 推理模型（o1、o3 等）上始终自动启用，无显式开关 |
| **OpenAI Responses** | `reasoning.type` | `"enabled"`、`"disabled"` |
| **Google GenAI** | `thinking_config.thinking_budget` | `0` = 关闭、`-1` = 动态、正整数 = 指定预算 |

### 努力级别

模型应投入多少推理"努力"。各提供商支持的粒度不同：

| 提供商 | 参数 | 支持的值 |
|--------|------|----------|
| **Anthropic** | `thinking.effort` | `"low"`、`"medium"`、`"high"`、`"max"`（需要 `type: "adaptive"`） |
| **OpenAI Chat** | `reasoning_effort` | `"low"`、`"medium"`、`"high"` |
| **OpenAI Responses** | `reasoning.effort` | `"low"`、`"medium"`、`"high"` |
| **Google GenAI** | `thinking_config.thinking_level` | `"minimal"`、`"low"`、`"medium"`、`"high"` |

### 预算 Token 数

模型可用于推理的最大 token 数：

| 提供商 | 参数 | 是否支持 |
|--------|------|:---:|
| **Anthropic** | `thinking.budget_tokens` | 支持（`type: "enabled"` 时必需） |
| **OpenAI Chat** | *（无）* | 不支持 |
| **OpenAI Responses** | *（无）* | 不支持 |
| **Google GenAI** | `thinking_config.thinking_budget` | 支持 |

### 总览矩阵

| 特性 | Anthropic | OpenAI Chat | OpenAI Responses | Google GenAI |
|------|-----------|-------------|------------------|-------------|
| 模式控制 | `thinking.type` | 隐式 | `reasoning.type` | `thinking_budget` 值 |
| 努力级别 | `thinking.effort` | `reasoning_effort` | `reasoning.effort` | `thinking_config.thinking_level` |
| 预算 token | `thinking.budget_tokens` | N/A | N/A | `thinking_config.thinking_budget` |
| 自动/自适应 | `type: "adaptive"` | 默认行为 | N/A | `thinking_budget: -1` |

## IR ReasoningConfig

LLM-Rosetta 定义了统一的 `ReasoningConfig` TypedDict，包含三个字段：

```python
class ReasoningConfig(TypedDict, total=False):
    mode: Literal["auto", "enabled", "disabled"]
    effort: Literal["minimal", "low", "medium", "high", "max"]
    budget_tokens: int   # 推理的最大 token 数
```

三个字段均为可选。可根据需要设置任意组合。

### 字段语义

- **`mode`** -- 控制推理行为：`"enabled"`（始终开启）、`"disabled"`（关闭）或 `"auto"`（由模型决定）。省略则由提供商使用默认行为。
- **`effort`** -- 模型在推理上应投入多少"努力"。这是一个独立于 mode 的横切关注点。
- **`budget_tokens`** -- 推理 token 数的硬上限。仅对支持该功能的提供商有效（Anthropic、Google）。

## IR 到提供商的映射

### `mode: "enabled"`（无 effort、无 budget）

最简单的开启推理方式：

```python
ir_request: IRRequest = {
    "model": "claude-sonnet-4-20250514",
    "messages": [...],
    "reasoning": {"mode": "enabled"},
}
```

=== "Anthropic"

    ```json
    {
      "thinking": {"type": "adaptive"}
    }
    ```

    !!! note "说明"
        回退为 `"adaptive"` 而非 `"enabled"`，因为 Anthropic 的 `"enabled"` 类型要求提供 `budget_tokens`。

=== "OpenAI Chat"

    不产生额外参数。推理模型默认自动推理。

=== "OpenAI Responses"

    ```json
    {
      "reasoning": {"type": "enabled"}
    }
    ```

=== "Google GenAI"

    不产生额外参数。Google 对支持思考的模型自动启用推理。

---

### `mode: "enabled"` + `budget_tokens`

显式控制推理预算：

```python
ir_request: IRRequest = {
    "model": "claude-sonnet-4-20250514",
    "messages": [...],
    "reasoning": {"mode": "enabled", "budget_tokens": 10000},
}
```

=== "Anthropic"

    ```json
    {
      "thinking": {
        "type": "enabled",
        "budget_tokens": 10000
      }
    }
    ```

=== "OpenAI Chat"

    ```json
    {}
    ```

    !!! warning "警告"
        OpenAI Chat 不支持 `budget_tokens`。将发出警告并忽略该字段。

=== "OpenAI Responses"

    ```json
    {
      "reasoning": {"type": "enabled"}
    }
    ```

    !!! warning "警告"
        OpenAI Responses 不支持 `budget_tokens`。将发出警告并忽略该字段。

=== "Google GenAI"

    ```json
    {
      "thinking_config": {
        "thinking_budget": 10000
      }
    }
    ```

---

### `mode: "disabled"`

显式关闭推理：

```python
ir_request: IRRequest = {
    "model": "claude-sonnet-4-20250514",
    "messages": [...],
    "reasoning": {"mode": "disabled"},
}
```

=== "Anthropic"

    ```json
    {
      "thinking": {"type": "disabled"}
    }
    ```

=== "OpenAI Chat"

    不产生参数（推理是隐式的，无法通过配置显式禁用）。

=== "OpenAI Responses"

    ```json
    {
      "reasoning": {"type": "disabled"}
    }
    ```

=== "Google GenAI"

    不产生参数。

---

### `effort`（可搭配或不搭配 `mode`）

设置推理努力级别：

```python
ir_request: IRRequest = {
    "model": "claude-sonnet-4-20250514",
    "messages": [...],
    "reasoning": {"effort": "high"},
}
```

=== "Anthropic"

    ```json
    {
      "thinking": {
        "type": "adaptive",
        "effort": "high"
      }
    }
    ```

    !!! info "说明"
        当设置了 `effort` 时，Anthropic 始终使用 `type: "adaptive"`，无论 `mode` 字段的值如何。`effort` 参数具有优先权。

=== "OpenAI Chat"

    ```json
    {
      "reasoning_effort": "high"
    }
    ```

=== "OpenAI Responses"

    ```json
    {
      "reasoning": {
        "effort": "high"
      }
    }
    ```

=== "Google GenAI"

    ```json
    {
      "thinking_config": {
        "thinking_level": "high"
      }
    }
    ```

## 努力级别映射

IR 支持五个努力级别。并非所有提供商都接受每个值：

| IR effort | Anthropic | OpenAI Chat | OpenAI Responses | Google GenAI |
|-----------|-----------|-------------|------------------|-------------|
| `"minimal"` | `"low"` :material-alert: | `"low"` :material-alert: | `"low"` :material-alert: | `"minimal"` |
| `"low"` | `"low"` | `"low"` | `"low"` | `"low"` |
| `"medium"` | `"medium"` | `"medium"` | `"medium"` | `"medium"` |
| `"high"` | `"high"` | `"high"` | `"high"` | `"high"` |
| `"max"` | `"max"` | `"high"` :material-alert: | `"high"` :material-alert: | `"high"` :material-alert: |

:material-alert: = 降级并发出警告

!!! warning "有损转换"
    - `"minimal"` 在 Anthropic、OpenAI Chat 和 OpenAI Responses 上降级为 `"low"`（这些提供商不支持 "minimal" 级别）。
    - `"max"` 在 OpenAI Chat、OpenAI Responses 和 Google GenAI 上降级为 `"high"`（这些提供商不支持 "max" 级别）。

    两种情况下都会发出 `UserWarning`，以便调用方检测到降级。

## 提供商到 IR 的映射（反向）

将提供商原生请求转换为 IR 时的映射：

| 提供商字段 | IR 字段 |
|------------|---------|
| `thinking.type = "enabled"` | `mode: "enabled"` |
| `thinking.type = "adaptive"` | `mode: "auto"` |
| `thinking.type = "disabled"` | `mode: "disabled"` |
| `thinking.effort` | `effort` |
| `thinking.budget_tokens` | `budget_tokens` |
| `reasoning_effort`（OpenAI Chat） | `effort` |
| `reasoning.type = "enabled"`（Responses） | `mode: "enabled"` |
| `reasoning.type = "disabled"`（Responses） | `mode: "disabled"` |
| `reasoning.effort`（Responses） | `effort` |
| `thinking_config.thinking_level` / `thinkingLevel`（Google） | `effort` |
| `thinking_config.thinking_budget` / `thinkingBudget`（Google） | `budget_tokens` |

!!! note "Google camelCase 支持"
    Google 转换器同时接受 snake_case（`thinking_config`、`thinking_budget`、`thinking_level`）和 camelCase（`thinkingConfig`、`thinkingBudget`、`thinkingLevel`）两种格式，分别对应 REST API 和 SDK 的命名约定。

## 设计决策

### 为什么 `mode` 是三态字段

IR 使用显式的 `mode: "auto" | "enabled" | "disabled"` 而非布尔值：

1. **与提供商直接对齐。** Anthropic 的 `thinking.type` 有三个值（`"adaptive"`、`"enabled"`、`"disabled"`），OpenAI Responses 的 `reasoning.type` 同样如此。三态 `mode` 实现了 1:1 映射，支持无损往返转换。
2. **省略仍然有效。** 当 `mode` 未设置时，提供商使用其默认行为 -- 即思考能力模型的自动推理。这与 `mode: "auto"` 不同，后者是显式请求自适应行为。
3. **effort 作为横切关注点。** 单独设置 `effort`（不设 `mode`）可以让模型自行决定是否推理，同时控制推理时投入的努力程度。

### 为什么 effort 有 5 个级别

IR 支持 `minimal`、`low`、`medium`、`high` 和 `max`，使其成为所有提供商级别的**超集**：

- Google 支持 `minimal`，但其他提供商不支持（降级为 `low`）
- Anthropic 支持 `max`，但其他提供商不支持（降级为 `high`）
- 中间三个级别（`low`、`medium`、`high`）被所有提供商普遍支持

这确保了同一提供商内的无损往返转换，同时在跨提供商时提供尽力而为的映射。

### 预算 token：仅 Anthropic 和 Google 支持

OpenAI（Chat 和 Responses）不支持显式的推理预算控制。当 IR 中设置了 `budget_tokens` 且目标为 OpenAI 时，将发出警告并静默丢弃该字段。这是设计意图 -- IR 是超集，有损转换会被显式标记。

## 完整示例：跨提供商推理

```python
from llm_rosetta import convert

# 带自适应推理的 Anthropic 请求
anthropic_request = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 8096,
    "thinking": {
        "type": "adaptive",
        "effort": "high",
    },
    "messages": [
        {"role": "user", "content": "请逐步解释量子纠缠。"}
    ],
}

# 转换为 OpenAI Chat 格式
openai_request = convert(anthropic_request, target="openai_chat")
# 结果包含: {"reasoning_effort": "high", ...}

# 转换为 Google GenAI 格式
google_request = convert(anthropic_request, target="google_genai")
# 结果包含: {"thinking_config": {"thinking_level": "high"}, ...}
```
