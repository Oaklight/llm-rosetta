---
title: Reasoning / Thinking Parameters
---

# Reasoning / Thinking Parameters

Modern LLMs can perform explicit chain-of-thought reasoning before generating their final answer. Each provider exposes this capability through different parameter names, structures, and semantics. LLM-Rosetta's `ReasoningConfig` provides a unified IR layer that maps to all supported providers.

## Provider Comparison

### Mode Control

How each provider controls whether reasoning is on, off, or automatic:

| Provider | Parameter | Values |
|----------|-----------|--------|
| **Anthropic** | `thinking.type` | `"adaptive"` (model decides), `"enabled"` (always on, requires `budget_tokens`), `"disabled"` (off) |
| **OpenAI Chat** | *(implicit)* | Reasoning is always active on reasoning models (o1, o3, etc.); no explicit toggle |
| **OpenAI Responses** | `reasoning.type` | `"enabled"`, `"disabled"` |
| **Google GenAI** | `thinking_config.thinking_budget` | `0` = off, `-1` = dynamic, positive integer = explicit budget |

### Effort Level

How hard the model should think. Not all providers support the same granularity:

| Provider | Parameter | Accepted Values |
|----------|-----------|-----------------|
| **Anthropic** | `thinking.effort` | `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"` (requires `type: "adaptive"`) |
| **OpenAI Chat** | `reasoning_effort` | `"low"`, `"medium"`, `"high"` |
| **OpenAI Responses** | `reasoning.effort` | `"none"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"` (input; normalized by IR) |
| **Google GenAI** | `thinking_config.thinking_level` | `"minimal"`, `"low"`, `"medium"`, `"high"` |

### Budget Tokens

Maximum tokens the model can spend on reasoning:

| Provider | Parameter | Supported? |
|----------|-----------|:---:|
| **Anthropic** | `thinking.budget_tokens` | Yes (required when `type: "enabled"`) |
| **OpenAI Chat** | *(none)* | No |
| **OpenAI Responses** | *(none)* | No |
| **Google GenAI** | `thinking_config.thinking_budget` | Yes |

### Summary Matrix

| Feature | Anthropic | OpenAI Chat | OpenAI Responses | Google GenAI |
|---------|-----------|-------------|------------------|-------------|
| Mode control | `thinking.type` | implicit | `reasoning.type` | `thinking_budget` value |
| Effort level | `thinking.effort` | `reasoning_effort` | `reasoning.effort` | `thinking_config.thinking_level` |
| Budget tokens | `thinking.budget_tokens` | N/A | N/A | `thinking_config.thinking_budget` |
| Auto/adaptive | `type: "adaptive"` | default behavior | N/A | `thinking_budget: -1` |

## IR ReasoningConfig

LLM-Rosetta defines a unified `ReasoningConfig` TypedDict with three fields:

```python
class ReasoningConfig(TypedDict, total=False):
    mode: Literal["auto", "enabled", "disabled"]
    effort: ReasoningEffortLevel  # See effort ladder below
    budget_tokens: int   # Max tokens for reasoning

ReasoningEffortLevel = Literal["minimal", "low", "medium", "high", "xhigh", "max"]
```

All three fields are optional. You can set any combination depending on what you want to control.

### Field Semantics

- **`mode`** -- Controls reasoning behavior: `"enabled"` (always on), `"disabled"` (off), or `"auto"` (let the model decide). Omit to let the provider use its default behavior.
- **`effort`** -- How much effort the model should put into reasoning. Six levels from `"minimal"` to `"max"`. This is a cross-cutting concern independent of the mode.
- **`budget_tokens`** -- Hard cap on reasoning token count. Only meaningful for providers that support it (Anthropic, Google).

### Input Normalization

The Responses API accepts non-standard effort values. These are normalized on input:

| Input value | Normalized to |
|-------------|---------------|
| `"none"` | `mode: "disabled"` (not an effort level) |
| `"xhigh"` | `effort: "xhigh"` (first-class IR level) |
| `"max"` | `effort: "max"` (first-class IR level) |

All other standard values (`"minimal"`, `"low"`, `"medium"`, `"high"`) pass through unchanged.

## IR-to-Provider Mapping

### `mode: "enabled"` (no effort, no budget)

The simplest way to turn on reasoning:

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

    !!! note
        Falls back to `"adaptive"` instead of `"enabled"` because Anthropic's `"enabled"` type requires `budget_tokens`.

=== "OpenAI Chat"

    No additional parameters emitted. Reasoning models reason by default.

=== "OpenAI Responses"

    ```json
    {
      "reasoning": {"type": "enabled"}
    }
    ```

=== "Google GenAI"

    No additional parameters emitted. Google reasoning is automatic for thinking-capable models.

---

### `mode: "enabled"` + `budget_tokens`

Explicit control over reasoning budget:

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

    !!! warning
        OpenAI Chat does not support `budget_tokens`. A warning is emitted and the field is ignored.

=== "OpenAI Responses"

    ```json
    {
      "reasoning": {"type": "enabled"}
    }
    ```

    !!! warning
        OpenAI Responses does not support `budget_tokens`. A warning is emitted and the field is ignored.

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

Explicitly disable reasoning:

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

    No parameters emitted (reasoning is implicit; cannot be explicitly disabled via config).

=== "OpenAI Responses"

    ```json
    {
      "reasoning": {"type": "disabled"}
    }
    ```

=== "Google GenAI"

    No parameters emitted.

---

### `effort` (with or without `mode`)

Set reasoning effort level:

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

    !!! info
        When `effort` is set, Anthropic always uses `type: "adaptive"` regardless of the `mode` field. The `effort` parameter takes priority.

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

## Effort Level Mapping

The IR supports six effort levels. Not all providers accept every value — the shim's `effort_map` controls the mapping:

| IR effort | Anthropic | OpenAI Chat | OpenAI Responses | Google GenAI |
|-----------|-----------|-------------|------------------|-------------|
| `"minimal"` | `"low"` :material-alert: | `"low"` :material-alert: | `"low"` :material-alert: | `"minimal"` |
| `"low"` | `"low"` | `"low"` | `"low"` | `"low"` |
| `"medium"` | `"medium"` | `"medium"` | `"medium"` | `"medium"` |
| `"high"` | `"high"` | `"high"` | `"high"` | `"high"` |
| `"xhigh"` | `"xhigh"` | `"high"` :material-alert: | `"high"` :material-alert: | `"high"` :material-alert: |
| `"max"` | `"max"` | `"high"` :material-alert: | `"high"` :material-alert: | `"high"` :material-alert: |

:material-alert: = downgraded via shim `effort_map`

!!! warning "Lossy conversions"
    - `"minimal"` is downgraded to `"low"` for Anthropic, OpenAI Chat, and OpenAI Responses (they do not support a "minimal" level).
    - `"xhigh"` and `"max"` are downgraded to `"high"` for OpenAI Chat, OpenAI Responses, and Google GenAI.
    - Anthropic supports `"xhigh"` and `"max"` natively.

    Downgrades are defined declaratively in each shim's `effort_map`. A `max_effort` cap can further limit the highest level emitted.

## Provider-to-IR Mapping (Reverse)

When converting provider-native requests to IR:

| Provider field | IR field |
|----------------|----------|
| `thinking.type = "enabled"` | `mode: "enabled"` |
| `thinking.type = "adaptive"` | `mode: "auto"` |
| `thinking.type = "disabled"` | `mode: "disabled"` |
| `thinking.effort` | `effort` |
| `thinking.budget_tokens` | `budget_tokens` |
| `reasoning_effort` (OpenAI Chat) | `effort` |
| `reasoning.type = "enabled"` (Responses) | `mode: "enabled"` |
| `reasoning.type = "disabled"` (Responses) | `mode: "disabled"` |
| `reasoning.effort` (Responses) | `effort` |
| `thinking_config.thinking_level` / `thinkingLevel` (Google) | `effort` |
| `thinking_config.thinking_budget` / `thinkingBudget` (Google) | `budget_tokens` |

!!! note "Google camelCase support"
    The Google converter accepts both snake_case (`thinking_config`, `thinking_budget`, `thinking_level`) and camelCase (`thinkingConfig`, `thinkingBudget`, `thinkingLevel`) variants, matching the REST API and SDK conventions respectively.

## Design Decisions

### Why `mode` is a three-way field

The IR uses an explicit `mode: "auto" | "enabled" | "disabled"` instead of a boolean:

1. **Direct provider alignment.** Anthropic's `thinking.type` has three values (`"adaptive"`, `"enabled"`, `"disabled"`), as does OpenAI Responses' `reasoning.type`. A three-way `mode` maps 1:1, enabling lossless round-trips.
2. **Omission is still valid.** When `mode` is not set, the provider uses its default behavior -- which is automatic reasoning for thinking-capable models. This is distinct from `mode: "auto"`, which explicitly requests adaptive behavior.
3. **Effort as a cross-cutting concern.** Setting `effort` alone (without `mode`) lets the model decide whether to think while controlling how much effort to use when it does.

### Why effort has 6 levels

The IR supports `minimal`, `low`, `medium`, `high`, `xhigh`, and `max` to be a **superset** of all provider levels:

- Google supports `minimal` but others do not (downgrade to `low`)
- Anthropic supports `xhigh` and `max` natively; others downgrade to `high`
- The three middle levels (`low`, `medium`, `high`) are universally supported

This ensures lossless round-trips within a single provider while providing best-effort mapping across providers.

### Shim-driven effort mapping

Since v0.6.8, effort mapping is **declarative** via provider shims rather than hardcoded in converters. Each shim's `ReasoningCapability` declares:

- **`effort_map`** — maps each IR effort level to the provider-specific string (or omits unsupported levels)
- **`max_effort`** — optional cap; efforts above this are clamped down
- **`disabled`** — how `mode: "disabled"` is serialized (`"omit"` or `"thinking_disabled"`)
- **`effort_field`** — where the provider expects the effort value (`"reasoning_effort"`, `"output_config.effort"`, etc.)
- **`thinking_type`** — force `thinking.type` to `"enabled"` or `"adaptive"` (per-model overrides via `model_overrides`)

See the [Provider Shims](shims.md#reasoning-configuration) guide for details on declaring custom reasoning configs.

### Budget tokens: Anthropic and Google only

OpenAI (both Chat and Responses) does not support explicit reasoning budget control. When `budget_tokens` is set in the IR and the target is OpenAI, a warning is emitted and the field is silently dropped. This is by design -- the IR is a superset, and lossy conversions are explicitly flagged.

## Full Example: Cross-Provider Reasoning

```python
from llm_rosetta import convert

# Anthropic request with adaptive thinking
anthropic_request = {
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 8096,
    "thinking": {
        "type": "adaptive",
        "effort": "high",
    },
    "messages": [
        {"role": "user", "content": "Explain quantum entanglement step by step."}
    ],
}

# Convert to OpenAI Chat format
openai_request = convert(anthropic_request, target="openai_chat")
# Result includes: {"reasoning_effort": "high", ...}

# Convert to Google GenAI format
google_request = convert(anthropic_request, target="google_genai")
# Result includes: {"thinking_config": {"thinking_level": "high"}, ...}
```
