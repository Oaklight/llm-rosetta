---
title: 推理 / 思考参数
---

# 推理 / 思考参数

现代 LLM 可以在出最终答案之前先做一轮显式的思维链推理（chain-of-thought）。各家提供方对这个功能的参数名、结构和语义各不相同。LLM-Rosetta 的 `ReasoningConfig` 在 IR 层做了统一。

## 提供方对比

### 模式控制

各家怎么控制推理的开/关/自动：

| 提供方 | 参数 | 取值 |
|--------|------|------|
| **Anthropic** | `thinking.type` | `"adaptive"`（模型自行决定）、`"enabled"`（始终开启，需要 `budget_tokens`）、`"disabled"`（关闭） |
| **OpenAI Chat** | *（隐式）* | 推理模型（o1、o3 等）上始终自动启用，无显式开关 |
| **OpenAI Responses** | `reasoning.type` | `"enabled"`、`"disabled"` |
| **Google GenAI** | `thinking_config.thinking_budget` | `0` = 关闭、`-1` = 动态、正整数 = 指定预算 |

### 努力级别

模型应投入多少推理"努力"。各提供方支持的粒度不同：

| 提供方 | 参数 | 支持的值 |
|--------|------|----------|
| **Anthropic** | `thinking.effort` | `"low"`、`"medium"`、`"high"`、`"max"`（需要 `type: "adaptive"`） |
| **OpenAI Chat** | `reasoning_effort` | `"low"`、`"medium"`、`"high"` |
| **OpenAI Responses** | `reasoning.effort` | `"low"`、`"medium"`、`"high"` |
| **Google GenAI** | `thinking_config.thinking_level` | `"minimal"`、`"low"`、`"medium"`、`"high"` |

### 预算 Token 数

模型可用于推理的最大 token 数：

| 提供方 | 参数 | 是否支持 |
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
    effort: Literal["minimal", "low", "medium", "high", "xhigh", "max"]
    budget_tokens: int   # 推理的最大 token 数
```

三个字段均为可选。可根据需要设置任意组合。

### 输入归一化

转换器在处理之前会对外部输入的 effort 值进行归一化：

| 输入值 | 归一化结果 | 说明 |
|--------|------------|------|
| `"none"` | `mode: "disabled"`，移除 effort | `none` 表示禁用推理，不是努力级别 |
| `"xhigh"` | `"xhigh"` | 规范 IR 努力级别，直接透传 |
| `"max"` | `"max"` | 规范 IR 努力级别，直接透传 |

归一化由 `normalize_reasoning_input()` 在进入转换器之前执行，不会修改原始输入。

### 字段语义

- **`mode`** -- 控制推理行为：`"enabled"`（始终开启）、`"disabled"`（关闭）或 `"auto"`（由模型决定）。省略则由提供方使用默认行为。
- **`effort`** -- 模型在推理上应投入多少"努力"。这是一个独立于 mode 的横切关注点。
- **`budget_tokens`** -- 推理 token 数的硬上限。仅对支持该功能的提供方有效（Anthropic、Google）。

## IR 到提供方的映射

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

IR 支持六个努力级别。实际映射由各提供方 shim 的 `effort_map` 声明，以下是内置 shim 的默认映射：

| IR effort | Anthropic | OpenAI Chat | OpenAI Responses | Google GenAI |
|-----------|-----------|-------------|------------------|-------------|
| `"minimal"` | `"low"` :material-alert: | `"minimal"` | `"minimal"` | — |
| `"low"` | `"low"` | `"low"` | `"low"` | — |
| `"medium"` | `"medium"` | `"medium"` | `"medium"` | — |
| `"high"` | `"high"` | `"high"` | `"high"` | — |
| `"xhigh"` | `"xhigh"` | `"high"` :material-alert: | `"high"` :material-alert: | — |
| `"max"` | `"max"` | `"high"` :material-alert: | `"high"` :material-alert: | — |

:material-alert: = 被 `max_effort` 上限或 `effort_map` 降级

— = Google shim 的 `effort_field` 为 `"none"`，effort 不发送给上游

!!! info "Shim 驱动的 effort 映射"
    effort 映射现在由各提供方的 `provider.yaml` 中的 `reasoning.effort_map` 声明，不再硬编码在转换器中。同时 `max_effort` 可以声明最高允许的 effort 级别（例如 OpenAI 的 `max_effort: high` 会将 `xhigh`/`max` 截断为 `high`）。

    如果 IR effort 不在目标 shim 的 `effort_map` 中，会发出警告并跳过。

## 提供方到 IR 的映射（反向）

将提供方原生请求转换为 IR 时的映射：

| 提供方字段 | IR 字段 |
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

1. **与提供方直接对齐。** Anthropic 的 `thinking.type` 有三个值（`"adaptive"`、`"enabled"`、`"disabled"`），OpenAI Responses 的 `reasoning.type` 同样如此。三态 `mode` 实现了 1:1 映射，支持无损往返转换。
2. **省略仍然有效。** 当 `mode` 未设置时，提供方使用其默认行为 -- 即思考能力模型的自动推理。这与 `mode: "auto"` 不同，后者是显式请求自适应行为。
3. **effort 作为横切关注点。** 单独设置 `effort`（不设 `mode`）可以让模型自行决定是否推理，同时控制推理时投入的努力程度。

### 为什么 effort 有 6 个级别

IR 支持 `minimal`、`low`、`medium`、`high`、`xhigh` 和 `max`，使其成为所有提供方级别的**超集**：

- Google 支持 `minimal`，但其他提供方不支持（降级为 `low`）
- Anthropic 支持 `xhigh` 和 `max`，但 OpenAI 不支持（被 `max_effort: high` 截断）
- 中间三个级别（`low`、`medium`、`high`）被所有提供方普遍支持

这确保了同一提供方内的无损往返转换，同时在跨提供方时提供尽力而为的映射。

### Shim 驱动的 effort 映射

从 v0.6.8 起，effort 映射不再硬编码在各转换器中，而是由提供方 shim 的 `ReasoningCapability` 配置声明。运行时流程：

1. 网关加载提供方 shim，将 `provider.yaml` 中的 `reasoning` 段解析为 `ReasoningCapability`
2. 请求到达时，`_inject_shim_reasoning()` 将 `ReasoningCapability` 注入转换上下文
3. 各转换器的 `ir_reasoning_config_to_p` 委托给 `apply_reasoning_config()`，传入 shim 配置
4. `apply_reasoning_config()` 先调用 `normalize_reasoning_input()` 归一化输入，然后按 `effort_map` 和 `max_effort` 映射

详细的 `ReasoningCapability` 字段和 YAML 配置请参见 [提供方 Shim · 推理配置](shims.md#推理配置)。

### 预算 token：仅 Anthropic 和 Google 支持

OpenAI（Chat 和 Responses）不支持显式的推理预算控制。当 IR 中设置了 `budget_tokens` 且目标为 OpenAI 时，将发出警告并静默丢弃该字段。这是设计意图 -- IR 是超集，有损转换会被显式标记。

## 完整示例：跨提供方推理

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
